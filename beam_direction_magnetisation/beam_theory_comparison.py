if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot

    from proper_research.parameters import default_magnet_params, default_beam_params
    from proper_research.robot.transformations import get_point
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
    from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
    from proper_research.control.mpc_boundary import resample_polyline
    from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz

    # import BOTH wrappers
    from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen
    from beam_direction_magnetisation.node_class import DEREnergyMinForwardWithLumen, set_axes_equal_3d,make_initial_nodes_straight,rod_section_stiffness,make_Kbt_inv_profile

    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------
    # helper
    # ------------------------------------------------------------
    def compare_models(fwd_cos, fwd_der, p7, p0_ur, q0_ur, L_cmd, lumen_C=None):
        tip_cos = fwd_cos(p7)
        tip_der = fwd_der(p7)

        print("\n" + "=" * 80)
        print("FORWARD MODEL COMPARISON")
        print("=" * 80)
        print("input p7              :", p7)

        print("\n--- tip comparison ---")
        print("Cosserat tip          :", tip_cos)
        print("DER tip               :", tip_der)
        print("tip difference        :", tip_der - tip_cos)
        print("tip difference norm   :", np.linalg.norm(tip_der - tip_cos))

        p_straight_cos = None
        p_straight_der = None

        if fwd_cos.last_p_centerline is not None:
            N_cos = fwd_cos.last_p_centerline.shape[1]
            p_straight_cos = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N_cos)
            tip_str_cos = p_straight_cos[:, -1]
            print("\n--- Cosserat summary ---")
            print("centerline shape      :", fwd_cos.last_p_centerline.shape)
            print("straight tip          :", tip_str_cos)
            print("tip deflection        :", tip_cos - tip_str_cos)
            print("tip defl norm         :", np.linalg.norm(tip_cos - tip_str_cos))

        if fwd_der.last_p_centerline is not None:
            N_der = fwd_der.last_p_centerline.shape[1]
            p_straight_der = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N_der)
            tip_str_der = p_straight_der[:, -1]
            print("\n--- DER summary ---")
            print("success               :", fwd_der.last_info["success"])
            print("message               :", fwd_der.last_info["message"])
            print("W                     :", fwd_der.last_info["W"])
            print("wire_len              :", fwd_der.last_info["wire_len"])
            print("tip_len               :", fwd_der.last_info["tip_len"])
            print("centerline shape      :", fwd_der.last_p_centerline.shape)
            print("straight tip          :", tip_str_der)
            print("tip deflection        :", tip_der - tip_str_der)
            print("tip defl norm         :", np.linalg.norm(tip_der - tip_str_der))

        if (fwd_cos.last_p_centerline is not None) and (fwd_der.last_p_centerline is not None):
            p_cos = fwd_cos.last_p_centerline
            p_der = fwd_der.last_p_centerline

            print("\n--- nodewise comparison ---")
            print("Cosserat shape        :", p_cos.shape)
            print("DER shape             :", p_der.shape)

            if p_cos.shape == p_der.shape:
                err = np.linalg.norm(p_der - p_cos, axis=0)
                print("mean node error       :", np.mean(err))
                print("max  node error       :", np.max(err))
            else:
                print("shapes differ, skipping nodewise error")

        # 3D overlay
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        if p_straight_der is not None:
            ax.plot(
                p_straight_der[0], p_straight_der[1], p_straight_der[2],
                "--", linewidth=1.5, label="straight rod"
            )

        if fwd_cos.last_p_centerline is not None:
            p_cos = fwd_cos.last_p_centerline
            ax.plot(
                p_cos[0], p_cos[1], p_cos[2],
                "-o", linewidth=2, markersize=4, label="Cosserat"
            )

        if fwd_der.last_p_centerline is not None:
            p_der = fwd_der.last_p_centerline
            ax.plot(
                p_der[0], p_der[1], p_der[2],
                "-o", linewidth=2, markersize=4, label="DER"
            )

        r_src = p7[:3]
        ax.scatter([p0_ur[0]], [p0_ur[1]], [p0_ur[2]], s=80, marker="o", label="base")
        ax.scatter([r_src[0]], [r_src[1]], [r_src[2]], s=90, marker="s", label="magnet")

        if lumen_C is not None:
            ax.plot(lumen_C[:, 0], lumen_C[:, 1], lumen_C[:, 2], "k:", linewidth=1.5, label="lumen")

        ax.set_title("Cosserat vs DER forward models")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        set_axes_equal_3d(ax)
        plt.tight_layout()
        plt.show()

        return tip_cos, tip_der

    # ------------------------------------------------------------
    # global setup
    # ------------------------------------------------------------
    beam_params = default_beam_params()
    mag_params = default_magnet_params()

    L_cmd = 0.05
    N_nodes = 10
    USE_LUMEN = False

    pivot_point = np.array([
        0.7681328220229531, -0.7112731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.08),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    start_point[2] = -0.1

    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)

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
    EA_tip  = tip["EA"]
    EI_wire = wire["EI"]
    EI_tip  = tip["EI"]
    GJ_wire = wire["GJ"]
    GJ_tip  = tip["GJ"]

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )

    print("\n" + "=" * 80)
    print("COMMON CASE SETUP")
    print("=" * 80)
    print(f"L_cmd            : {L_cmd:.6e}")
    print(f"N_nodes          : {N_nodes}")
    print(f"base position    : {p0_ur}")
    print(f"magnet position  : {r_src_ur}")
    print(f"M_ref_local      : {M_ref_local}")
    print(f"EA_wire          : {EA_wire:.6e}")
    print(f"EA_tip           : {EA_tip:.6e}")
    print(f"EI_wire          : {EI_wire:.6e}")
    print(f"EI_tip           : {EI_tip:.6e}")
    print(f"GJ_wire          : {GJ_wire:.6e}")
    print(f"GJ_tip           : {GJ_tip:.6e}")

    # ------------------------------------------------------------
    # optional lumen
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # optional lumen
    # ------------------------------------------------------------
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
    # build Cosserat wrapper only
    # ------------------------------------------------------------
    if lumen_C is None:
        lumen_C_cos = np.zeros((2, 3), float)
        lumen_R_cos = np.full(2, 1.0, float)
    else:
        lumen_C_cos = lumen_C
        lumen_R_cos = lumen_R

    fwd_cos = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C_cos,
        lumen_R=lumen_R_cos,
        N_nodes=N_nodes,
        maxiter=30,
        L0_init=0.01,
        dL_internal=0.002,
        use_lumen_jac=USE_LUMEN,
        L_tip_full=mag_len,
        L_tip_min=0.01,
    )

    # ------------------------------------------------------------
    # helper printing
    # ------------------------------------------------------------
    def print_jacobian_table(J, row_labels=None, col_labels=None, title="Jacobian", fmt="{: .6e}"):
        J = np.asarray(J, float)
        n_row, n_col = J.shape

        if row_labels is None:
            row_labels = [f"y{i}" for i in range(n_row)]
        if col_labels is None:
            col_labels = [f"u{i}" for i in range(n_col)]

        row_name_w = max(len(r) for r in row_labels + ["out"])
        col_w = max(max(len(c) for c in col_labels), 13)

        print(f"\n[{title}]")
        header = "out".ljust(row_name_w) + " | " + " | ".join(c.rjust(col_w) for c in col_labels)
        print(header)
        print("-" * len(header))

        for i in range(n_row):
            vals = " | ".join(fmt.format(J[i, j]).rjust(col_w) for j in range(n_col))
            print(row_labels[i].ljust(row_name_w) + " | " + vals)

    def print_jacobian_comparison(Ja, Jb, row_labels=None, col_labels=None, name_a="A", name_b="B"):
        Ja = np.asarray(Ja, float)
        Jb = np.asarray(Jb, float)

        if Ja.shape != Jb.shape:
            print(f"\n[ERROR] shape mismatch: {name_a} {Ja.shape} vs {name_b} {Jb.shape}")
            return

        dJ = Jb - Ja

        print_jacobian_table(Ja, row_labels, col_labels, title=name_a)
        print_jacobian_table(Jb, row_labels, col_labels, title=name_b)
        print_jacobian_table(dJ, row_labels, col_labels, title=f"{name_b} - {name_a}")

        print("\n[DIFF NORMS]")
        print("columnwise 2-norms :", np.linalg.norm(dJ, axis=0))
        print("frobenius norm     :", np.linalg.norm(dJ))
        print("max abs diff       :", np.max(np.abs(dJ)))

    # ------------------------------------------------------------
    # pose propagation
    # ------------------------------------------------------------
    def integrate_pose8_body(p8, u7, dt):
        """
        p8 = [x,y,z, qw,qx,qy,qz, L]
        u7 = [vx,vy,vz, wx,wy,wz, dL]  (angular velocity in BODY frame)
        """
        p8 = np.asarray(p8, float).copy()
        u7 = np.asarray(u7, float).ravel()

        p8[0:3] += dt * u7[0:3]
        p8[7]   += dt * u7[6]

        q = quat_wxyz_normalize(p8[3:7])
        dphi = dt * u7[3:6]
        dq = small_rot_quat_wxyz(dphi)
        q_new = quat_wxyz_mul(q, dq)
        p8[3:7] = quat_wxyz_normalize(q_new)

        return p8

    def pose8_quat_to_pose7_rotvec(p8):
        p8 = np.asarray(p8, float).ravel()
        t = p8[0:3]
        q = quat_wxyz_normalize(p8[3:7])
        L = p8[7]
        rvec = quat_wxyz_to_rotvec(q)
        return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

    # ------------------------------------------------------------
    # deterministic forward wrapper
    # ------------------------------------------------------------
    class DeterministicForward6D:
        """
        Small wrapper so every evaluation is as deterministic as possible.
        """
        def __init__(self, fwd_model):
            self.fwd = fwd_model
            self.last_info = None

        def __call__(self, p7):
            # clear any warm-start / cache state if present
            if hasattr(self.fwd, "_last"):
                try:
                    self.fwd._last["q_init"] = None
                    self.fwd._last["L"] = None
                except Exception:
                    pass

            y = self.fwd(p7)
            self.last_info = getattr(self.fwd, "last_info", None)
            return y

    def make_forward_tip_fn_no_cache(fwd_model, name="Cosserat"):
        def forward_y_fn(p8):
            p7 = pose8_quat_to_pose7_rotvec(p8)
            y = np.asarray(fwd_model(p7), float).reshape(3,)
            info = getattr(fwd_model, "last_info", None)
            if info is not None and not info.get("success", True):
                print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
            return y
        return forward_y_fn
    class ColdForward6D:
        """
        Always clears internal warm-start/cache state before evaluation.
        """
        def __init__(self, fwd_model):
            self.fwd = fwd_model
            self.last_info = None

        def __call__(self, p7):
            if hasattr(self.fwd, "_last"):
                try:
                    self.fwd._last["q_init"] = None
                    self.fwd._last["L"] = None
                except Exception:
                    pass

            y = self.fwd(p7)
            self.last_info = getattr(self.fwd, "last_info", None)
            return y
    def benchmark_jacobian_mode(
        label,
        forward_tip_fn,
        p8_nom,
        dt,
        eps_u,
        dL,
        n_out=3,
        n_repeat=3,
    ):
        import time

        forward_times = []
        full_times = []
        red_times = []

        B_full_last = None
        B_red_last = None

        for k in range(n_repeat):
            t0 = time.perf_counter()
            y = forward_tip_fn(p8_nom)
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1e3)

            t0 = time.perf_counter()
            B_full_last = numerical_B_y_wrt_u_central(
                p8_nom, forward_tip_fn, dt=dt, eps_u=eps_u, n_out=n_out, delta_L=dL
            )
            t1 = time.perf_counter()
            full_times.append((t1 - t0) * 1e3)

            t0 = time.perf_counter()
            B_red_last = numerical_B_y_wrt_u_xy_wz_L_central(
                p8_nom, forward_tip_fn, dt=dt, eps_u=eps_u, n_out=n_out, delta_L=dL
            )
            t1 = time.perf_counter()
            red_times.append((t1 - t0) * 1e3)

        print("\n" + "=" * 80)
        print(f"JACOBIAN BENCHMARK: {label}")
        print("=" * 80)
        print(f"forward mean [ms]       : {np.mean(forward_times):.2f}")
        print(f"forward min/max [ms]    : {np.min(forward_times):.2f} / {np.max(forward_times):.2f}")
        print(f"full jac mean [ms]      : {np.mean(full_times):.2f}")
        print(f"full jac min/max [ms]   : {np.min(full_times):.2f} / {np.max(full_times):.2f}")
        print(f"reduced jac mean [ms]   : {np.mean(red_times):.2f}")
        print(f"reduced jac min/max [ms]: {np.min(red_times):.2f} / {np.max(red_times):.2f}")

        return {
            "label": label,
            "forward_times_ms": np.array(forward_times),
            "full_times_ms": np.array(full_times),
            "reduced_times_ms": np.array(red_times),
            "B_full": B_full_last,
            "B_red": B_red_last,
        }
    def compare_jacobian_modes(cold, warm):
        print("\n" + "=" * 80)
        print("COLD vs WARM JACOBIAN COMPARISON")
        print("=" * 80)

        speedup_fwd = np.mean(cold["forward_times_ms"]) / np.mean(warm["forward_times_ms"])
        speedup_full = np.mean(cold["full_times_ms"]) / np.mean(warm["full_times_ms"])
        speedup_red = np.mean(cold["reduced_times_ms"]) / np.mean(warm["reduced_times_ms"])

        print(f"forward speedup (warm)       : {speedup_fwd:.2f}x")
        print(f"full Jacobian speedup (warm) : {speedup_full:.2f}x")
        print(f"red Jacobian speedup (warm)  : {speedup_red:.2f}x")

        for name in ["B_full", "B_red"]:
            Jc = cold[name]
            Jw = warm[name]
            dJ = Jw - Jc

            abs_fro = np.linalg.norm(dJ)
            rel_fro = abs_fro / (np.linalg.norm(Jc) + 1e-12)
            col_err = np.linalg.norm(dJ, axis=0)
            col_ref = np.linalg.norm(Jc, axis=0) + 1e-12
            col_rel = col_err / col_ref

            print(f"\n{name}:")
            print(f"  abs Fro diff        : {abs_fro:.6e}")
            print(f"  rel Fro diff        : {rel_fro:.6e}")
            print(f"  col abs diff norms  : {col_err}")
            print(f"  col rel diff norms  : {col_rel}")
            print(f"  max abs entry diff  : {np.max(np.abs(dJ)):.6e}")
    class WarmForward6D:
        """
        Leaves internal cache/warm-start state untouched.
        """
        def __init__(self, fwd_model):
            self.fwd = fwd_model
            self.last_info = None

        def __call__(self, p7):
            y = self.fwd(p7)
            self.last_info = getattr(self.fwd, "last_info", None)
            return y
    # ------------------------------------------------------------
    # central-difference Jacobians only
    # ------------------------------------------------------------
    def numerical_B_y_wrt_u_central(p8, forward_y_fn, dt, eps_u, n_out, delta_L=None):
        import time
        t0 = time.perf_counter()

        p8 = np.asarray(p8, float).ravel()
        B = np.zeros((n_out, 7), float)

        if delta_L is None:
            delta_L = dt * eps_u[6]

        tint = 0.0
        tfwd = 0.0

        for i in range(6):
            du = np.zeros(7, float)
            du[i] = eps_u[i]

            a = time.perf_counter()
            p_plus  = integrate_pose8_body(p8, +du, dt)
            p_minus = integrate_pose8_body(p8, -du, dt)
            b = time.perf_counter()
            tint += (b - a)

            a = time.perf_counter()
            y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
            y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
            b = time.perf_counter()
            tfwd += (b - a)

            B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

        p_plus = p8.copy()
        p_minus = p8.copy()
        p_plus[7]  += delta_L
        p_minus[7] -= delta_L

        a = time.perf_counter()
        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
        B[:, 6] = dt * dy_dL

        t1 = time.perf_counter()
        print(f"[TIME] central full Jacobian total: {(t1-t0)*1e3:.2f} ms")
        print(f"[TIME] integrate total          : {(tint)*1e3:.2f} ms")
        print(f"[TIME] forward total            : {(tfwd)*1e3:.2f} ms")

        return B

    def numerical_B_y_wrt_u_xy_wz_L_central(p8, forward_y_fn, dt, eps_u, n_out, delta_L=None):
        import time
        t0 = time.perf_counter()

        p8 = np.asarray(p8, float).ravel()
        B_red = np.zeros((n_out, 4), float)

        if delta_L is None:
            delta_L = dt * eps_u[6]

        tint = 0.0
        tfwd = 0.0

        selected = [0, 1, 5]  # vx, vy, wz

        for j, i in enumerate(selected):
            du = np.zeros(7, float)
            du[i] = eps_u[i]

            a = time.perf_counter()
            p_plus  = integrate_pose8_body(p8, +du, dt)
            p_minus = integrate_pose8_body(p8, -du, dt)
            b = time.perf_counter()
            tint += (b - a)

            a = time.perf_counter()
            y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
            y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
            b = time.perf_counter()
            tfwd += (b - a)

            B_red[:, j] = (y_plus - y_minus) / (2.0 * eps_u[i])

        p_plus = p8.copy()
        p_minus = p8.copy()
        p_plus[7]  += delta_L
        p_minus[7] -= delta_L

        a = time.perf_counter()
        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
        B_red[:, 3] = dt * dy_dL

        t1 = time.perf_counter()
        print(f"[TIME] central reduced Jacobian total: {(t1-t0)*1e3:.2f} ms")
        print(f"[TIME] integrate total             : {(tint)*1e3:.2f} ms")
        print(f"[TIME] forward total               : {(tfwd)*1e3:.2f} ms")

        return B_red

    # ------------------------------------------------------------
    # perturbation sweep
    # ------------------------------------------------------------
    def fd_column_sweep_central(p8, forward_y_fn, dt, channel, deltas, n_out=3):
        """
        channel:
            0=vx, 1=vy, 2=vz, 3=wx, 4=wy, 5=wz, 6=L

        deltas:
            actual perturbation magnitudes in state space:
              x,y,z -> meters
              wx,wy,wz -> radians
              L -> meters
        """
        p8 = np.asarray(p8, float).ravel()
        cols = []

        for d in deltas:
            if channel < 6:
                du = np.zeros(7, float)
                du[channel] = d / dt

                p_plus  = integrate_pose8_body(p8, +du, dt)
                p_minus = integrate_pose8_body(p8, -du, dt)

                y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
                y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

                col = (y_plus - y_minus) / (2.0 * du[channel])

            else:
                p_plus = p8.copy()
                p_minus = p8.copy()
                p_plus[7]  += d
                p_minus[7] -= d

                y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
                y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

                col = dt * (y_plus - y_minus) / (2.0 * d)

            cols.append(col)

        return np.asarray(cols, float)   # (n_delta, n_out)

    def print_fd_sweep_results(name, deltas, cols):
        print(f"\n[FD SWEEP] {name}")
        print("delta         | dtip_x          dtip_y          dtip_z          ||col||")
        print("-" * 78)
        for d, c in zip(deltas, cols):
            print(f"{d:12.3e} | {c[0]: .6e}  {c[1]: .6e}  {c[2]: .6e}  {np.linalg.norm(c):.6e}")

    def analyze_fd_plateau(name, deltas, cols, rel_tol=0.15):
        """
        Very simple plateau detector:
        compares neighboring columns and marks where relative change is small.
        """
        norms = np.linalg.norm(cols, axis=1)
        print(f"\n[FD PLATEAU ANALYSIS] {name}")
        print("delta         | ||col||        | rel change to next")
        print("-" * 56)

        rel_changes = []
        for i in range(len(deltas)):
            if i < len(deltas) - 1:
                denom = np.linalg.norm(cols[i]) + 1e-12
                rel = np.linalg.norm(cols[i+1] - cols[i]) / denom
                rel_changes.append(rel)
                print(f"{deltas[i]:12.3e} | {norms[i]:12.6e} | {rel:12.6e}")
            else:
                print(f"{deltas[i]:12.3e} | {norms[i]:12.6e} | {'-':>12}")

        good = [i for i, r in enumerate(rel_changes) if r < rel_tol]
        if len(good) == 0:
            print("No obvious plateau found.")
        else:
            print("Candidate plateau intervals:")
            for i in good:
                print(f"  {deltas[i]:.3e} -> {deltas[i+1]:.3e}")

    def benchmark_forward(forward_y_fn, p8_nom, n_repeat=3):
        import time
        times = []
        y_last = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            y_last = forward_y_fn(p8_nom)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
        print("\n[FORWARD BENCHMARK]")
        print(f"mean forward time : {np.mean(times):.2f} ms")
        print(f"min  forward time : {np.min(times):.2f} ms")
        print(f"max  forward time : {np.max(times):.2f} ms")
        print(f"last y            : {y_last}")
        return np.array(times)

    # ------------------------------------------------------------
    # nominal state
    # ------------------------------------------------------------
    r_src = start_point[:3]
    rvec_src = start_point[3:6]
    p7_nom = np.hstack([r_src, rvec_src, L_cmd])
    p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

    # # deterministic forward handle
    forward6d = DeterministicForward6D(fwd_cos)
    forward_tip_cos = make_forward_tip_fn_no_cache(forward6d, "Cosserat")

    # # ------------------------------------------------------------
    # # choose nominal perturbations
    # # ------------------------------------------------------------
    dt = 1
    # dr = 5e-3
    # dtheta = 3e-1
    # dL = 1e-3

    # eps_u = np.array([
    #     dr / dt, dr / dt, dr / dt,
    #     dtheta / dt, dtheta / dt, dtheta / dt,
    #     dL / dt
    # ], dtype=float)

    # # ------------------------------------------------------------
    # # run nominal central Jacobians
    # # ------------------------------------------------------------
    # benchmark_forward(forward_tip_cos, p8_nom, n_repeat=3)

    # B_full = numerical_B_y_wrt_u_central(
    #     p8_nom, forward_tip_cos, dt=dt, eps_u=eps_u, n_out=3, delta_L=dL
    # )
    # B_red = numerical_B_y_wrt_u_xy_wz_L_central(
    #     p8_nom, forward_tip_cos, dt=dt, eps_u=eps_u, n_out=3, delta_L=dL
    # )

    # row_labels = ["tip_x", "tip_y", "tip_z"]
    # col_labels_full = ["vx", "vy", "vz", "wx", "wy", "wz", "L"]
    # col_labels_red = ["vx", "vy", "wz", "L"]

    # print_jacobian_table(B_full, row_labels, col_labels_full, title="Full Central Jacobian")
    # print_jacobian_table(B_red, row_labels, col_labels_red, title="Reduced Central Jacobian")

    # B_full_subset = B_full[:, [0, 1, 5, 6]]
    # print_jacobian_comparison(
    #     B_full_subset,
    #     B_red,
    #     row_labels=row_labels,
    #     col_labels=col_labels_red,
    #     name_a="B_full_subset",
    #     name_b="B_red",
    # )

    # # ------------------------------------------------------------
    # # perturbation sweeps
    # # ------------------------------------------------------------
    deltas_vx = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2], float)
    deltas_vy = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2], float)
    deltas_wz = np.array([5e-2, 1e-1, 2e-1, 3e-1, 5e-1], float)
    deltas_L  = np.array([1e-3, 2e-3, 5e-3, 1e-2, 3e-2], float)

    cols_vx = fd_column_sweep_central(p8_nom, forward_tip_cos, dt, channel=0, deltas=deltas_vx)
    cols_vy = fd_column_sweep_central(p8_nom, forward_tip_cos, dt, channel=1, deltas=deltas_vy)
    cols_wz = fd_column_sweep_central(p8_nom, forward_tip_cos, dt, channel=5, deltas=deltas_wz)
    cols_L  = fd_column_sweep_central(p8_nom, forward_tip_cos, dt, channel=6, deltas=deltas_L)

    print_fd_sweep_results("vx", deltas_vx, cols_vx)
    print_fd_sweep_results("vy", deltas_vy, cols_vy)
    print_fd_sweep_results("wz", deltas_wz, cols_wz)
    print_fd_sweep_results("L",  deltas_L, cols_L)

    analyze_fd_plateau("vx", deltas_vx, cols_vx, rel_tol=0.15)
    analyze_fd_plateau("vy", deltas_vy, cols_vy, rel_tol=0.15)
    analyze_fd_plateau("wz", deltas_wz, cols_wz, rel_tol=0.15)
    analyze_fd_plateau("L",  deltas_L, cols_L, rel_tol=0.15)
    def make_forward_tip_fn(fwd_model, name="Cosserat"):
        def forward_y_fn(p8):
            p7 = pose8_quat_to_pose7_rotvec(p8)
            y = np.asarray(fwd_model(p7), float).reshape(3,)
            info = getattr(fwd_model, "last_info", None)
            if info is not None and not info.get("success", True):
                print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
            return y
        return forward_y_fn
    # ------------------------------------------------------------
    # nominal state
    # ------------------------------------------------------------
    r_src = start_point[:3]
    rvec_src = start_point[3:6]
    p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

    # ------------------------------------------------------------
    # choose nominal perturbations
    # ------------------------------------------------------------
    dt = 1.0
    dr = 5e-3
    dtheta = 3e-1
    dL = 1e-3

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        dtheta / dt, dtheta / dt, dtheta / dt,
        dL / dt
    ], dtype=float)

    # ------------------------------------------------------------
    # build cold and warm forward handles
    # ------------------------------------------------------------
    cold6d = ColdForward6D(fwd_cos)
    warm6d = WarmForward6D(fwd_cos)

    forward_tip_cold = make_forward_tip_fn(cold6d, "Cosserat cold")
    forward_tip_warm = make_forward_tip_fn(warm6d, "Cosserat warm")

    # optional: prime the warm cache once
    _ = forward_tip_warm(p8_nom)

    # ------------------------------------------------------------
    # benchmark both
    # ------------------------------------------------------------
    cold_stats = benchmark_jacobian_mode(
        "COLD",
        forward_tip_cold,
        p8_nom,
        dt=dt,
        eps_u=eps_u,
        dL=dL,
        n_out=3,
        n_repeat=3,
    )

    warm_stats = benchmark_jacobian_mode(
        "WARM",
        forward_tip_warm,
        p8_nom,
        dt=dt,
        eps_u=eps_u,
        dL=dL,
        n_out=3,
        n_repeat=3,
    )

    compare_jacobian_modes(cold_stats, warm_stats)