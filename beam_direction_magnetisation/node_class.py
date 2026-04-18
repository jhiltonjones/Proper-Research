import numpy as np
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.node_optimisation import (
    solve_nodes_twist_min,
    rod_section_stiffness,
    make_Kbt_inv_profile,
    make_initial_nodes_straight,
    set_axes_equal_3d,
)
class DEREnergyMinForwardWithLumen:
    def __init__(
        self,
        *,
        p0_ur,
        q0_ur,
        Kinv_fun,
        EA_wire,
        EA_tip,
        M_ref_local,
        m_body,
        lumen_C=None,
        lumen_R=None,
        N_nodes=15,
        maxiter=300,
        amp_init=2e-4,
        enforce_inextensibility=False,
        use_lumen=True,
        L0_init=0.01,
        dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01,
        ref_twist=None,
        use_continuation=True,
        N_coarse=9,
        N_schedule=None,
    ):
        self.use_continuation = bool(use_continuation)
        self.N_coarse = int(N_coarse)


        self.p0_ur = np.asarray(p0_ur, float).reshape(3,)
        self.q0_ur = np.asarray(q0_ur, float).reshape(4,)

        self.Kinv_fun = Kinv_fun
        self.EA_wire = float(EA_wire)
        self.EA_tip = float(EA_tip)

        self.M_ref_local = np.asarray(M_ref_local, float).reshape(3,)
        self.m_body = np.asarray(m_body, float).reshape(3,)

        self.lumen_C = None if lumen_C is None else np.asarray(lumen_C, float)
        self.lumen_R = None if lumen_R is None else np.asarray(lumen_R, float)

        self.N_nodes = int(N_nodes)
        self.maxiter = int(maxiter)
        self.amp_init = float(amp_init)
        self.enforce_inextensibility = bool(enforce_inextensibility)
        self.use_lumen = bool(use_lumen)

        self.L0_init = float(L0_init)
        self.dL_internal = float(dL_internal)
        self.L_tip_full = float(L_tip_full)
        self.L_tip_min = float(L_tip_min)
        self.ref_twist = ref_twist

        # internal cache
        self._last = dict(
            p=None,             # last 7D command
            L=None,             # last modeled length
            q_init=None,        # warm-start state vector
            tip=None,
            p_centerline=None,
            theta=None,
            info=None,
        )

        # public mirrors
        self.last_tip = None
        self.last_p_centerline = None
        self.last_theta = None
        self.last_info = None

    def _quat_from_rotvec_ur(self, rvec):
        Rm = Rot.from_rotvec(np.asarray(rvec, float).reshape(3,)).as_matrix()
        q_xyzw = Rot.from_matrix(Rm).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
        q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
        return q_wxyz
    def _make_L_schedule(self, L_model):
        L_start = max(self.L_tip_min, min(self.L0_init, L_model))

        if L_model <= L_start + 1e-12:
            return np.array([L_model], float)

        n_steps = max(2, int(np.ceil((L_model - L_start) / self.dL_internal)) + 1)
        return np.linspace(L_start, L_model, n_steps)
    def _dipole_from_pose(self, q_src_wxyz):
        q = np.asarray(q_src_wxyz, float).reshape(4,)
        Rm = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return Rm @ self.m_body

    def _effective_lengths(self, L_ins):
        L_ins = float(L_ins)
        tip_len = min(L_ins, self.L_tip_full)
        wire_len = max(L_ins - self.L_tip_full, 0.0)
        L_model = max(L_ins, self.L_tip_min)
        tip_len = min(tip_len, L_model)
        return L_model, wire_len, tip_len
    def _solve_single(self, *, L_model, wire_len, tip_len, r_src, m_src, use_lumen_now, q_init):
        return solve_nodes_twist_min(
            p0=self.p0_ur,
            q0=self.q0_ur,
            L=L_model,
            N=self.N_nodes,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=self.Kinv_fun,
            r_src=r_src,
            m_src=m_src,
            mu_tip=np.linalg.norm(self.M_ref_local),
            EA_wire=self.EA_wire,
            EA_tip=self.EA_tip,
            lumen_C=self.lumen_C,
            lumen_R=self.lumen_R,
            use_lumen=use_lumen_now,
            maxiter=self.maxiter,
            M_ref_local=self.M_ref_local,
            ref_twist=self.ref_twist,
            enforce_inextensibility=self.enforce_inextensibility,
            amp_init=self.amp_init,
            q_init=q_init,
        )
    def __call__(self, p):
        """
        p = [x, y, z, rx, ry, rz, L]
        returns tip position (3,)
        """
        p = np.asarray(p, float).reshape(-1)
        if p.size != 7:
            raise ValueError("Expected p = [x,y,z, rx,ry,rz, L]")

        r_src = p[0:3]
        rvec = p[3:6]
        L_ins = float(p[6])

        # magnet pose
        q_src = self._quat_from_rotvec_ur(rvec)
        m_src = self._dipole_from_pose(q_src)

        # effective geometry split
        L_model, wire_len, tip_len = self._effective_lengths(L_ins)
        print("\n[FWD NODE DEBUG]")
        print("L_ins =", L_ins)
        print("L_model =", L_model)
        print("wire_len =", wire_len)
        print("tip_len =", tip_len)
        # lumen use
        use_lumen_now = (
            self.use_lumen
            and (self.lumen_C is not None)
            and (self.lumen_R is not None)
        )

        # warm start
        q_init = None
        if self._last["q_init"] is not None and self._last["L"] is not None:
            if abs(L_model - self._last["L"]) < 0.01:
                q_init = self._last["q_init"].copy()

        # solve
        p_opt, theta_opt, info = solve_nodes_twist_min(
            p0=self.p0_ur,
            q0=self.q0_ur,
            L=L_model,
            N=self.N_nodes,
            wire_len=wire_len,
            tip_len = tip_len,
            Kinv_fun=self.Kinv_fun,
            r_src=r_src,
            m_src=m_src,
            mu_tip=np.linalg.norm(self.M_ref_local),
            EA_wire=self.EA_wire,
            EA_tip=self.EA_tip,
            lumen_C=self.lumen_C,
            lumen_R=self.lumen_R,
            use_lumen=use_lumen_now,
            maxiter=self.maxiter,
            M_ref_local=self.M_ref_local,
            ref_twist=self.ref_twist,
            enforce_inextensibility=self.enforce_inextensibility,
            amp_init=self.amp_init,
            q_init=q_init,   # <-- add this
        )

        # reconstruct packed state for warm-start
        q_opt = info["q_opt"].copy()
        tip = p_opt[:, -1].copy()

        # enrich info with wrapper-level metadata
        info = dict(info)
        info["wrapper_input"] = p.copy()
        info["L_model"] = L_model
        info["wire_len"] = wire_len
        info["tip_len"] = tip_len
        info["r_src"] = r_src.copy()
        info["q_src"] = q_src.copy()
        info["m_src"] = m_src.copy()
        dbg = info.get("dbg", {})

        mag = info["dbg"]["mag"]

        # print("\n[DER MAG DEBUG]")
        # print("Wm                :", info["dbg"]["Wm"])
        # print("wire_len          :", mag.get("wire_len", wire_len))
        # print("tip_len           :", mag.get("tip_len", tip_len))
        # print("s_seg             :", mag.get("s_seg"))
        # print("ell_i             :", mag.get("ell_i"))
        # print("mdotB             :", mag.get("mdotB"))
        # print("wm_density        :", mag.get("wm_density"))
        # print("wm_seg            :", mag.get("wm_seg"))
        # print("Bmag              :", mag.get("Bmag"))
        # print("Mmag              :", mag.get("Mmag"))
        # print_der_magnetic_debug(info)
        # cache
        if info["success"]:
            self._last.update(
                p=p.copy(),
                L=L_model,
                q_init=q_opt.copy(),
                tip=tip.copy(),
                p_centerline=p_opt.copy(),
                theta=theta_opt.copy(),
                info=info,
            )
        else:
            # keep previous warm start, do not poison cache
            pass

        self.last_tip = tip.copy()
        self.last_p_centerline = p_opt.copy()
        self.last_theta = theta_opt.copy()
        self.last_info = info

        return tip
def print_der_magnetic_debug(info, max_rows=None):
    dbg = info.get("dbg", {})
    mag = dbg.get("mag", {})

    if not mag:
        print("\n[DER MAG DEBUG] no magnetic debug info found")
        return

    s_seg = np.asarray(mag.get("s_seg", []), float)
    ell_i = np.asarray(mag.get("ell_i", []), float)
    Bmag = np.asarray(mag.get("Bmag", []), float)
    Mmag = np.asarray(mag.get("Mmag", []), float)
    mdotB = np.asarray(mag.get("mdotB", []), float)
    wm_density = np.asarray(mag.get("wm_density", []), float)
    wm_seg = np.asarray(mag.get("wm_seg", []), float)

    if len(s_seg) == 0:
        print("\n[DER MAG DEBUG] magnetic arrays are empty")
        return

    active = Mmag > 1e-9

    # print("\n[DER MAGNETIC BREAKDOWN]")
    # print(f"Wm total          : {dbg.get('Wm', np.nan):.6e}")
    # print(f"wire_len          : {mag.get('wire_len', np.nan):.6e}")
    # print(f"tip_len           : {mag.get('tip_len', np.nan):.6e}")
    # print(f"active segs       : {int(np.sum(active))}")
    # print(f"max |B|           : {np.max(Bmag):.6e}")
    # print(f"max |M|           : {np.max(Mmag):.6e}")
    # print(f"min M·B           : {np.min(mdotB):.6e}")
    # print(f"max M·B           : {np.max(mdotB):.6e}")
    # print(f"sum wm_seg        : {np.sum(wm_seg):.6e}")

    # print("\n[DER MAG SEGMENTS]")
    # print(" i |    s_seg |    ell_i | active |      |B| |      |M| |       M·B |  wm_density |     wm_seg")
    # print("-" * 100)

    n = len(s_seg)
    rows = range(n) if max_rows is None else range(min(n, max_rows))

    # for i in rows:
    #     print(
    #         f"{i:2d} | "
    #         f"{s_seg[i]:8.5f} | "
    #         f"{ell_i[i]:8.5f} | "
    #         f"{int(active[i]):6d} | "
    #         f"{Bmag[i]:9.3e} | "
    #         f"{Mmag[i]:9.3e} | "
    #         f"{mdotB[i]:11.3e} | "
    #         f"{wm_density[i]:11.3e} | "
    #         f"{wm_seg[i]:11.3e}"
    #     )
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot

    from proper_research.parameters import default_magnet_params, default_beam_params
    from proper_research.robot.transformations import get_point
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
    from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning, LumenQuery
    from proper_research.control.mpc_boundary import resample_polyline

    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------
    # base / magnet setup
    # ------------------------------------------------------------
    beam_params = default_beam_params()
    mag_params = default_magnet_params()
    L_cmd = 0.075
    N_nodes = 9
    pivot_point = np.array([
        0.7681328220229531, -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
    ], float)
    base_point = np.array([
        (pivot_point[0]-(L_cmd+0.08)), -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
    ], float)
    start_point = get_point(0, 60, base_point,  pivot_point)
    start_point[2] = -0.1

    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    # external source dipole defined in magnet body frame
    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)

    # ------------------------------------------------------------
    # DER beam setup
    # ------------------------------------------------------------


    mag_len = beam_params.length_of_mag
    mu_line = beam_params.mag * beam_params.A_cs
    M_ref_local = np.array([0.0, 0.0, mu_line], float)

    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )

    EA_wire = wire["EA"]
    EA_tip = tip["EA"]
    EI_wire = wire["EI"]
    EI_tip = tip["EI"]
    GJ_wire = wire["GJ"]
    GJ_tip = tip["GJ"]

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )

    # ------------------------------------------------------------
    # optional lumen
    # ------------------------------------------------------------
    USE_LUMEN = False

    lumen_C = None
    lumen_R = None

    if USE_LUMEN:
        Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
        t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

        lumen_C = make_lumen_centerline_turning(
            p_start=p0_ur,
            t0=t0,
            length=0.08,
            n_pts=130,
            bend_axis=np.array([0.0, 0.0, 1.0]),
            bend_angle=np.deg2rad(40.0),
            bend_start=0.02,
            bend_end=0.05,
        )
        lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
        lumen_R = np.full(len(lumen_C), 0.004)

    # ------------------------------------------------------------
    # build forward model
    # ------------------------------------------------------------
    fwd = DEREnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        EA_wire=EA_wire,
        EA_tip=EA_tip,
        M_ref_local=M_ref_local,
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        N_nodes=N_nodes,
        maxiter=80,
        amp_init=2e-3,
        enforce_inextensibility=False,
        use_lumen=USE_LUMEN,
        L_tip_full=mag_len,
        L_tip_min=0.01,
        ref_twist=None,
    )

    # ------------------------------------------------------------
    # single-call test
    # ------------------------------------------------------------
    r_src = start_point[:3]
    rvec_src = start_point[3:6]
    p7 = np.hstack([r_src, rvec_src, L_cmd])

    tip = fwd(p7)

    print("\n" + "=" * 80)
    print("SINGLE CALL TEST")
    print("=" * 80)
    print("input p7           :", p7)
    print("returned tip       :", tip)
    print("tip shape          :", tip.shape)
    print("last_tip is None   :", fwd.last_tip is None)
    print("last_centerline    :", None if fwd.last_p_centerline is None else fwd.last_p_centerline.shape)
    print("last_theta         :", None if fwd.last_theta is None else fwd.last_theta.shape)
    print("solver success     :", fwd.last_info["success"])
    print("solver message     :", fwd.last_info["message"])
    print("total energy       :", fwd.last_info["W"])
    print("wire_len           :", fwd.last_info["wire_len"])
    print("tip_len            :", fwd.last_info["tip_len"])
    print("r_src used         :", fwd.last_info["r_src"])
    print("m_src used         :", fwd.last_info["m_src"])

    assert tip.shape == (3,)
    assert fwd.last_tip is not None
    assert fwd.last_p_centerline is not None
    assert fwd.last_theta is not None
    assert fwd.last_info is not None

    # ------------------------------------------------------------
    # compare to straight rod
    # ------------------------------------------------------------
    p_straight = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N_nodes)
    tip_straight = p_straight[:, -1]
    tip_deflection = tip - tip_straight

    print("\n--- geometry check ---")
    print("straight tip        :", tip_straight)
    print("tip deflection      :", tip_deflection)
    print("tip deflection norm :", np.linalg.norm(tip_deflection))

    # ------------------------------------------------------------
    # warm-start test: repeat same call
    # ------------------------------------------------------------
    tip2 = fwd(p7)

    print("\n--- warm-start repeat ---")
    print("tip2                :", tip2)
    print("repeat diff norm    :", np.linalg.norm(tip2 - tip))

    # ------------------------------------------------------------
    # small sweep in insertion length
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LENGTH SWEEP TEST")
    print("=" * 80)

    L_vals = np.linspace(0.045, L_cmd, 6)
    tips = []

    for L_ins in L_vals:

        pivot_point = np.array([
            0.7681328220229531, -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
        ], float)
        base_point = np.array([
            (pivot_point[0]-(L_ins+0.08)), -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
        ], float)
        start_point = get_point(0, 20, base_point,  pivot_point)
        start_point[2] = -0.1

        T_ur_pivot = ur_pose6_to_T(pivot_point)
        p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

        T_ur_mag = ur_pose6_to_T(start_point)
        r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

        # external source dipole defined in magnet body frame
        m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)

        r_src = start_point[:3]
        rvec_src = start_point[3:6]


        p7_i = np.hstack([r_src, rvec_src, L_ins])
        tip_i = fwd(p7_i)
        tips.append(tip_i.copy())
        print(
            f"L={L_ins:.6f}, "
            f"tip={tip_i}, "
            f"success={fwd.last_info['success']}, "
            f"W={fwd.last_info['W']:.6e}"
        )

    tips = np.asarray(tips)

    # ------------------------------------------------------------
    # plots
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(L_vals, tips[:, 0], "-o", label="tip x")
    plt.plot(L_vals, tips[:, 1], "-o", label="tip y")
    plt.plot(L_vals, tips[:, 2], "-o", label="tip z")
    plt.xlabel("Insertion length [m]")
    plt.ylabel("Tip position [m]")
    plt.title("Forward wrapper: tip vs insertion")
    plt.grid(True)
    plt.legend()

    if fwd.last_p_centerline is not None:
        p = fwd.last_p_centerline
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(p[0], p[1], p[2], "-o", label="DER centerline")
        ax.plot(p_straight[0], p_straight[1], p_straight[2], "--", label="straight")
        if lumen_C is not None:
            ax.plot(lumen_C[:, 0], lumen_C[:, 1], lumen_C[:, 2], "k:", label="lumen")
        ax.scatter([r_src[0]], [r_src[1]], [r_src[2]], marker="s", s=80, label="magnet")
        ax.legend()
        ax.set_title("Final centerline")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_axes_equal_3d(ax)
    plt.show()