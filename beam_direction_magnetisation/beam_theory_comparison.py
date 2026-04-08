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

    L_cmd = 0.075
    N_nodes = 10
    USE_LUMEN = True

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

    start_point = np.asarray(get_point(0, 20, base_point, pivot_point), dtype=float)
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
    # build Cosserat wrapper
    # ------------------------------------------------------------
    # if USE_LUMEN is False, give dummy lumen so the constructor does not choke
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

    # # ------------------------------------------------------------
    # # build DER wrapper
    # # ------------------------------------------------------------
    # fwd_der = DEREnergyMinForwardWithLumen(
    #     p0_ur=p0_ur,
    #     q0_ur=q0_ur,
    #     Kinv_fun=Kinv_fun,
    #     EA_wire=EA_wire,
    #     EA_tip=EA_tip,
    #     M_ref_local=M_ref_local,
    #     m_body=m_body,
    #     lumen_C=lumen_C,
    #     lumen_R=lumen_R,
    #     N_nodes=N_nodes,
    #     maxiter=30,
    #     amp_init=2e-4,
    #     enforce_inextensibility=True,
    #     use_lumen=USE_LUMEN,
    #     L_tip_full=mag_len,
    #     L_tip_min=0.01,
    #     ref_twist=None,
    #     use_continuation=False,
    #     N_coarse=9,
    # )

    # # ------------------------------------------------------------
    # # single comparison
    # # ------------------------------------------------------------
    # r_src = start_point[:3]
    # rvec_src = start_point[3:6]
    # p7 = np.hstack([r_src, rvec_src, L_cmd])

    # tip_cos, tip_der = compare_models(
    #     fwd_cos,
    #     fwd_der,
    #     p7,
    #     p0_ur=p0_ur,
    #     q0_ur=q0_ur,
    #     L_cmd=L_cmd,
    #     lumen_C=lumen_C,
    # )
    # tip = fwd_der(p7)

    # from beam_direction_magnetisation.node_optimisation import dipole_field_points
    # from beam_direction_magnetisation.magnetism.magnetic_methods import dipole_field_from_source

    # kin = fwd_der.last_info["dbg"]["kin"]
    # pts = kin["mid"].T
    # m_src = fwd_der.last_info["m_src"]
    # r_src_used = fwd_der.last_info["r_src"]

    # B_cos = dipole_field_from_source(pts, r_src_used, m_src, r_min=1e-6)
    # B_der = dipole_field_points(pts.T, r_src_used, m_src, r_min=1e-6).T

    # err = np.linalg.norm(B_cos - B_der, axis=1)

    # print("\n" + "=" * 80)
    # print("FIELD LAW COMPARISON ON IDENTICAL POINTS")
    # print("=" * 80)
    # print("max |B_cos - B_der| =", np.max(err))
    # print("mean|B_cos - B_der| =", np.mean(err))
    # print("max |B_cos|         =", np.max(np.linalg.norm(B_cos, axis=1)))
    # print("max |B_der|         =", np.max(np.linalg.norm(B_der, axis=1)))
    # print("rel max error       =", np.max(err) / (np.max(np.linalg.norm(B_cos, axis=1)) + 1e-12))
    # mag = fwd_der.last_info["dbg"]["mag"]
    # kin = fwd_der.last_info["dbg"]["kin"]

    # pts = kin["mid"].T                    # (n_seg,3)
    # B_same = dipole_field_from_source(pts, r_src_used, m_src, r_min=1e-6)   # (n_seg,3)

    # M_world = mag["M_world"].T            # (n_seg,3)
    # ell_i = mag["ell_i"]

    # mdotB_same = np.sum(M_world * B_same, axis=1)
    # Wm_same_midpoint = np.sum(-mdotB_same * ell_i)

    # print("DER Wm stored        =", mag["wm_seg"].sum())
    # print("DER Wm recomputed    =", Wm_same_midpoint)
    # print("difference           =", Wm_same_midpoint - mag["wm_seg"].sum())
    # # ------------------------------------------------------------
    # # length sweep comparison
    # # ------------------------------------------------------------
    # print("\n" + "=" * 80)
    # print("LENGTH SWEEP COMPARISON")
    # print("=" * 80)

    # L_vals = np.linspace(0.045, L_cmd, 6)
    # tips_cos = []
    # tips_der = []

    # for L_ins in L_vals:
    #     base_point_i = np.array([
    #         pivot_point[0] - (L_ins + 0.08),
    #         pivot_point[1],
    #         -0.1,
    #         np.pi, 0.001, 0.001
    #     ], float)

    #     start_point_i = np.asarray(get_point(0, 60, base_point_i, pivot_point), dtype=float)
    #     start_point_i[2] = -0.1

    #     r_src_i = start_point_i[:3]
    #     rvec_src_i = start_point_i[3:6]
    #     p7_i = np.hstack([r_src_i, rvec_src_i, L_ins])

    #     tip_cos_i = fwd_cos(p7_i)
    #     tip_der_i = fwd_der(p7_i)

    #     tips_cos.append(tip_cos_i.copy())
    #     tips_der.append(tip_der_i.copy())

    #     print(
    #         f"L={L_ins:.6f} | "
    #         f"Cos tip={tip_cos_i} | "
    #         f"DER tip={tip_der_i} | "
    #         f"diff={np.linalg.norm(tip_der_i - tip_cos_i):.6e}"
    #     )

    # tips_cos = np.asarray(tips_cos)
    # tips_der = np.asarray(tips_der)

    # plt.figure(figsize=(9, 6))
    # plt.plot(L_vals, tips_cos[:, 0], "-o", label="Cos x")
    # plt.plot(L_vals, tips_cos[:, 1], "-o", label="Cos y")
    # plt.plot(L_vals, tips_der[:, 0], "--s", label="DER x")
    # plt.plot(L_vals, tips_der[:, 1], "--s", label="DER y")
    # plt.xlabel("Insertion length [m]")
    # plt.ylabel("Tip position [m]")
    # plt.title("Cosserat vs DER: tip position vs insertion")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    def integrate_pose8_body(p8, u7, dt):
        """
        p8 = [x,y,z, qw,qx,qy,qz, L]
        u7 = [vx,vy,vz, wx,wy,wz, dL]  (wx..wz in BODY frame)
        """
        p8 = np.asarray(p8, float).copy()
        u7 = np.asarray(u7, float).ravel()

        # translation + length
        p8[0:3] += dt * u7[0:3]
        p8[7]   += dt * u7[6]

        # quaternion update: q_new = q ⊗ δq(dt*ω_body)
        q = quat_wxyz_normalize(p8[3:7])
        dphi = dt * u7[3:6]
        dq = small_rot_quat_wxyz(dphi)
        q_new = quat_wxyz_mul(q, dq)         # body-frame increment
        p8[3:7] = quat_wxyz_normalize(q_new)

        return p8
    import time
    def numerical_B_y_wrt_u(p8, forward_y_fn, dt, eps_u, n_out):
        import time
        t0 = time.perf_counter()

        p8 = np.asarray(p8, float).ravel()
        B = np.zeros((n_out, 7), float)

        tint = 0.0
        tfwd = 0.0

        for i in range(6):
            du = np.zeros(7); du[i] = eps_u[i]

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

        delta_L = 1e-3
        p_plus = p8.copy();  p_plus[7]  += delta_L
        p_minus = p8.copy(); p_minus[7] -= delta_L

        a = time.perf_counter()
        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
        B[:, 6] = dt * dy_dL

        t1 = time.perf_counter()
        print(f"[TIME] numerical_B total: {(t1-t0)*1e3:.2f} ms")
        print(f"[TIME] integrate total: {(tint)*1e3:.2f} ms")
        print(f"[TIME] forward total: {(tfwd)*1e3:.2f} ms")

        return B
    def numerical_B_y_wrt_u_xy_wz_L_forward(p8, forward_y_fn, dt, eps_u, n_out, delta_L=1e-3):
        import time
        t0 = time.perf_counter()

        p8 = np.asarray(p8, float).ravel()
        B_red = np.zeros((n_out, 4), float)

        tint = 0.0
        tfwd = 0.0

        a = time.perf_counter()
        y0 = np.asarray(forward_y_fn(p8), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        selected = [0, 1, 5]   # vx, vy, wz

        for j, i in enumerate(selected):
            du = np.zeros(7)
            du[i] = eps_u[i]

            a = time.perf_counter()
            p_plus = integrate_pose8_body(p8, du, dt)
            b = time.perf_counter()
            tint += (b - a)

            a = time.perf_counter()
            y_plus = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
            b = time.perf_counter()
            tfwd += (b - a)

            B_red[:, j] = (y_plus - y0) / (eps_u[i])

        p_plus = p8.copy()
        p_plus[7] += delta_L

        a = time.perf_counter()
        y_plus = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        dy_dL = (y_plus - y0) / delta_L
        B_red[:, 3] = dt * dy_dL

        t1 = time.perf_counter()
        print(f"[TIME] numerical_B forward total: {(t1-t0)*1e3:.2f} ms")
        print(f"[TIME] integrate total: {(tint)*1e3:.2f} ms")
        print(f"[TIME] forward total: {(tfwd)*1e3:.2f} ms")

        return B_red
    def numerical_B_y_wrt_u_xy_wz_L(p8, forward_y_fn, dt, eps_u, n_out, delta_L=1e-3):
        """
        Compute Jacobian only w.r.t.
            vx, vy, wz, and L

        Returns:
            B_red: shape (n_out, 4)
                columns = [dY/dvx, dY/dvy, dY/dwz, dt*dY/dL]
        """
        t0 = time.perf_counter()

        p8 = np.asarray(p8, float).ravel()
        B_red = np.zeros((n_out, 4), float)

        tint = 0.0
        tfwd = 0.0

        # Only keep vx, vy, wz
        selected = [0, 1, 5]   # vx, vy, wz

        for j, i in enumerate(selected):
            du = np.zeros(7)
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

        # L sensitivity
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
        print(f"[TIME] numerical_B total: {(t1-t0)*1e3:.2f} ms")
        print(f"[TIME] integrate total: {(tint)*1e3:.2f} ms")
        print(f"[TIME] forward total: {(tfwd)*1e3:.2f} ms")

        return B_red
    def pose8_quat_to_pose7_rotvec(p8):
        p8 = np.asarray(p8, float).ravel()
        t = p8[0:3]
        q = quat_wxyz_normalize(p8[3:7])
        L = p8[7]
        rvec = quat_wxyz_to_rotvec(q)
        return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

    def make_forward_tip_fn_checked(fwd_model, name="model"):
        def forward_y_fn(p8):
            p7 = pose8_quat_to_pose7_rotvec(p8)
            y = np.asarray(fwd_model(p7), float).reshape(3,)
            info = getattr(fwd_model, "last_info", None)
            if info is not None and not info.get("success", True):
                print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
            return y
        return forward_y_fn

    p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

    forward_tip_cos = make_forward_tip_fn_checked(fwd_cos, "Cosserat")
    import time
    import numpy as np

    def benchmark_forward(forward_y_fn, p8_nom, n_repeat=5):
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

    def benchmark_jacobians_cos_only(p8_nom, forward_tip_cos, dt, eps_u, n_out=3, n_repeat=3):
        full_times = []
        red_times = []

        B_full_last = None
        B_red_last = None

        for _ in range(n_repeat):
            t0 = time.perf_counter()
            B_full_last = numerical_B_y_wrt_u_xy_wz_L(
                p8_nom, forward_tip_cos, dt=dt, eps_u=eps_u, n_out=n_out
            )
            t1 = time.perf_counter()
            full_times.append((t1 - t0) * 1e3)

            t0 = time.perf_counter()
            B_red_last = numerical_B_y_wrt_u_xy_wz_L_forward(
                p8_nom, forward_tip_cos, dt=dt, eps_u=eps_u, n_out=n_out
            )
            t1 = time.perf_counter()
            red_times.append((t1 - t0) * 1e3)

        print("\n[JACOBIAN BENCHMARK: COSSERAT ONLY]")
        print(f"full Jacobian mean time    : {np.mean(full_times):.2f} ms")
        print(f"reduced Jacobian mean time : {np.mean(red_times):.2f} ms")
        print(f"speedup factor             : {np.mean(full_times)/np.mean(red_times):.2f}x")

        return {
            "full_times_ms": np.array(full_times),
            "reduced_times_ms": np.array(red_times),
            "B_full": B_full_last,
            "B_red": B_red_last,
        }
    dt = 0.05
    dr = 1e-4
    dtheta = 5e-4
    dL = 1e-4

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        dtheta / dt, dtheta / dt, dtheta / dt,
        dL / dt
    ], dtype=float)
    forward_tip_cos = make_forward_tip_fn_checked(fwd_cos, "Cosserat")

    benchmark_forward(forward_tip_cos, p8_nom, n_repeat=5)

    bench = benchmark_jacobians_cos_only(
        p8_nom,
        forward_tip_cos,
        dt=dt,
        eps_u=eps_u,
        n_out=3,
        n_repeat=3,
    )
    B_full = bench["B_full"]
    B_red = bench["B_red"]

    # B_full_subset = B_full[:, [0, 1, 5, 6]]

    print("\n[CONSISTENCY CHECK: FULL vs REDUCED]")
    print("B_full_subset =\n", B_full)
    print("B_red         =\n", B_red)
    print("difference    =\n", B_red - B_full)
    print("subset diff Fro norm =", np.linalg.norm(B_red - B_full))
    print("subset diff max abs  =", np.max(np.abs(B_red - B_full)))