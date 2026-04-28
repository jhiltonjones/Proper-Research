import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import (quat_derivative_body, quat_normalize, 
                                                                              quat_to_rot, quat_to_R,
                                                                              T_to_p_quat_wxyz)
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
import pandas as pd
import os
# ------------------------------------------------------------
# pretty printing
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
# pose helpers
# ------------------------------------------------------------
def integrate_pose8_body(p8, u7, dt):
    """
    p8 = [x,y,z, qw,qx,qy,qz, L]
    u7 = [vx,vy,vz, wx,wy,wz, dL]
    angular increment is interpreted in BODY frame.
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
    """
    p8 = [x,y,z,qw,qx,qy,qz,L]
    p7 = [x,y,z,rx,ry,rz,L]
    """
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = quat_wxyz_normalize(p8[3:7])
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)


def quat_wxyz_to_R(q):
    q = np.asarray(q, float).ravel()
    q = q / (np.linalg.norm(q) + 1e-12)
    # scipy expects [x, y, z, w]
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def dipole_from_pose_wxyz(q_src, m_body):
    R = quat_wxyz_to_R(q_src)
    return R @ np.asarray(m_body, float).reshape(3,)


# ------------------------------------------------------------
# deterministic / cold wrappers
# ------------------------------------------------------------
class ColdForwardP7:
    """
    Wraps a p7->tip model and clears warm-start/cache before each call if possible.
    """
    def __init__(self, fwd_model):
        self.fwd = fwd_model
        self.last_info = None

    def __call__(self, p7):
        # wrappers you already have often use _last or similar
        if hasattr(self.fwd, "_last"):
            try:
                self.fwd._last["q_init"] = None
                self.fwd._last["L"] = None
            except Exception:
                pass
        if hasattr(self.fwd, "_sol_prev"):
            try:
                self.fwd._sol_prev = None
            except Exception:
                pass

        y = self.fwd(p7)
        self.last_info = getattr(self.fwd, "last_info", None)
        return y


class WarmForwardP7:
    """
    Leaves internal warm-start/cache untouched.
    """
    def __init__(self, fwd_model):
        self.fwd = fwd_model
        self.last_info = None

    def __call__(self, p7):
        y = self.fwd(p7)
        self.last_info = getattr(self.fwd, "last_info", None)
        return y


def make_forward_tip_fn_from_p7_model(fwd_model, name="model"):
    """
    Returns forward_y_fn(p8) -> tip(3,)
    for wrappers that already accept p7.
    """
    def forward_y_fn(p8):
        p7 = pose8_quat_to_pose7_rotvec(p8)
        y = np.asarray(fwd_model(p7), float).reshape(3,)
        info = getattr(fwd_model, "last_info", None)
        if info is not None and (not info.get("success", True)):
            print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
        return y
    return forward_y_fn
# ------------------------------------------------------------
# Jacobians by central differences
# ------------------------------------------------------------
def numerical_B_y_wrt_u_central(p8, forward_y_fn, dt, eps_u, n_out=3, delta_L=None):
    """
    Full Jacobian wrt [vx, vy, vz, wx, wy, wz, dL].
    Returns shape (n_out, 7).
    """
    p8 = np.asarray(p8, float).ravel()
    B = np.zeros((n_out, 7), float)

    if delta_L is None:
        delta_L = dt * eps_u[6]

    for i in range(6):
        du = np.zeros(7, float)
        du[i] = eps_u[i]

        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)

        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

        B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

    # length channel
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[7]  += delta_L
    p_minus[7] -= delta_L

    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
    B[:, 6] = dt * dy_dL
    return B


def numerical_B_y_wrt_u_xy_wz_L_central(p8, forward_y_fn, dt, eps_u, n_out=3, delta_L=None):
    """
    Reduced Jacobian wrt [vx, vy, wz, dL].
    Returns shape (n_out, 4).
    """
    p8 = np.asarray(p8, float).ravel()
    B_red = np.zeros((n_out, 4), float)

    if delta_L is None:
        delta_L = dt * eps_u[6]

    selected = [0, 1, 5]  # vx, vy, wz

    for j, i in enumerate(selected):
        du = np.zeros(7, float)
        du[i] = eps_u[i]

        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)

        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

        B_red[:, j] = (y_plus - y_minus) / (2.0 * eps_u[i])

    # length channel
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[7]  += delta_L
    p_minus[7] -= delta_L

    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
    B_red[:, 3] = dt * dy_dL
    return B_red


def benchmark_forward(forward_y_fn, p8_nom, n_repeat=3):
    times = []
    y_last = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        y_last = forward_y_fn(p8_nom)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return {
        "times_ms": np.array(times),
        "y_last": np.asarray(y_last, float).copy(),
    }


def benchmark_jacobian_mode(label, forward_tip_fn, p8_nom, dt, eps_u, dL, n_out=3, n_repeat=3):
    forward_times = []
    full_times = []
    red_times = []

    B_full_last = None
    B_red_last = None

    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = forward_tip_fn(p8_nom)
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
    print(f"full jac mean [ms]      : {np.mean(full_times):.2f}")
    print(f"reduced jac mean [ms]   : {np.mean(red_times):.2f}")

    return {
        "label": label,
        "forward_times_ms": np.array(forward_times),
        "full_times_ms": np.array(full_times),
        "reduced_times_ms": np.array(red_times),
        "B_full": B_full_last,
        "B_red": B_red_last,
    }

# ------------------------------------------------------------
# Direct model adapters: p8 -> tip
# ------------------------------------------------------------
def make_forward_tip_fn_der_direct_min(
    *,
    p0_ur, q0_ur,
    Kinv_fun,
    EA_wire, EA_tip,
    M_ref_local,
    m_body,
    lumen_C, lumen_R,
    use_lumen,
    N_nodes,
    mag_len,
    name="der_direct_min",
):
    def forward_y_fn(p8):
        p8 = np.asarray(p8, float).ravel()
        r_src = p8[0:3]
        q_src = quat_wxyz_normalize(p8[3:7])
        L_cmd = float(p8[7])

        L_model, wire_len, tip_len = effective_lengths(
            L_cmd, L_tip_full=mag_len, L_tip_min=0.01
        )
        m_src = dipole_from_pose_wxyz(q_src, m_body)

        p_opt, theta_opt, info = solve_nodes_twist_min(
            p0=p0_ur,
            q0=q0_ur,
            L=L_model,
            N=N_nodes,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src,
            m_src=m_src,
            mu_tip=np.linalg.norm(M_ref_local),
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=use_lumen,
            maxiter=300,
            M_ref_local=M_ref_local,
            ref_twist=None,
            enforce_inextensibility=True,
            amp_init=2e-4,
            q_init=None,
        )

        if not info.get("success", True):
            print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
        return p_opt[:, -1]
    return forward_y_fn


def make_forward_tip_fn_der_residual(
    *,
    p0_ur, q0_ur,
    Kinv_fun,
    EA_wire, EA_tip,
    M_ref_local,
    m_body,
    lumen_C, lumen_R,
    use_lumen,
    N_nodes,
    mag_len,
    q_init_seed=None,
    name="der_residual",
):
    def forward_y_fn(p8):
        p8 = np.asarray(p8, float).ravel()
        r_src = p8[0:3]
        q_src = quat_wxyz_normalize(p8[3:7])
        L_cmd = float(p8[7])

        L_model, wire_len, tip_len = effective_lengths(
            L_cmd, L_tip_full=mag_len, L_tip_min=0.01
        )
        m_src = dipole_from_pose_wxyz(q_src, m_body)

        p_opt, theta_opt, info = solve_nodes_twist_residual(
            p0=p0_ur,
            q0=q0_ur,
            L=L_model,
            N=N_nodes,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src,
            m_src=m_src,
            mu_tip=np.linalg.norm(M_ref_local),
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=use_lumen,
            maxiter=200,
            M_ref_local=M_ref_local,
            ref_twist=None,
            q_init=q_init_seed,
        )

        if not info.get("success", True):
            print(f"[WARN] {name} failed: {info.get('message', 'unknown')}")
        return p_opt[:, -1]
    return forward_y_fn
def vec3_or_nan(v):
    if v is None:
        return [np.nan, np.nan, np.nan]
    v = np.asarray(v, float).reshape(-1)
    if v.size < 3:
        return [np.nan, np.nan, np.nan]
    return [float(v[0]), float(v[1]), float(v[2])]

def make_forward_tip_fn_cosserat_direct(
    *,
    p0_ur, q0_ur,
    Kinv_fun,
    m_body,
    mag_len,
    name="cosserat_direct",
):
    def forward_y_fn(p8):
        p8 = np.asarray(p8, float).ravel()
        r_src = p8[0:3]
        q_src = quat_wxyz_normalize(p8[3:7])
        L_cmd = float(p8[7])

        L_model, wire_len, tip_len = effective_lengths(
            L_cmd, L_tip_full=mag_len, L_tip_min=0.01
        )

        model = CosseratForwardModel(
            p0=p0_ur,
            q0=q0_ur,
            Kinv_fun=Kinv_fun,
            m_local_fun=make_m_local_fun_wire_tip(
                wire_len, len_tip=tip_len, mode="axial", alpha_end=0.0
            ),
            m_moment=0.0,
            wire_len=wire_len,
        )

        out = model.forward(
            L=L_model,
            r_src=r_src,
            q_src=q_src,
            wire_len=wire_len,
            m_body=m_body,
        )

        if not out.get("solved", True):
            print(f"[WARN] {name} failed: {out.get('message', 'unknown')}")
        return np.asarray(out["p_tip"], float).reshape(3,)
    return forward_y_fn
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot

    np.set_printoptions(precision=6, suppress=True)

    # ============================================================
    # Imports
    # ============================================================
    from proper_research.parameters import default_magnet_params, default_beam_params
    from proper_research.robot.transformations import get_point
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from beam_direction_magnetisation.quarternions.quarternions_functions import (
        T_to_p_quat_wxyz,
    )

    from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
    from proper_research.control.mpc_boundary import resample_polyline

    # Wrappers / solvers
    from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen
    from beam_direction_magnetisation.node_class import (
        DEREnergyMinForwardWithLumen,
        set_axes_equal_3d,
        make_initial_nodes_straight,
        rod_section_stiffness,
        make_Kbt_inv_profile,
    )

    # Direct solvers / helpers
    from beam_direction_magnetisation.node_optimisation import (
        solve_nodes_twist_min,
        solve_nodes_twist_residual,
        magnetic_energy_from_kinematics,
        dipole_field_points,
    )
    from beam_direction_magnetisation.magnetism.magnetic_methods import dipole_field_from_source

    from beam_direction_magnetisation.cosserat_w_minimal_energy import (
        CosseratForwardModel,
        effective_lengths,
        make_m_local_fun_wire_tip,
    )

    # ============================================================
    # Flags
    # ============================================================
    RUN_COS_WRAPPER = True
    RUN_DER_WRAPPER = True
    RUN_DER_DIRECT_MIN = False
    RUN_DER_RESIDUAL = False
    RUN_DIRECT_COSSERAT = True

    USE_LUMEN = False

    # ============================================================
    # Helpers
    # ============================================================
    def rotvec_pose7_from_start_point(start_point, L_cmd):
        r_src = np.asarray(start_point[:3], float)
        rvec_src = np.asarray(start_point[3:6], float)
        return np.hstack([r_src, rvec_src, float(L_cmd)])

    def summarize_result(name, res, p0_ur, q0_ur, L_cmd):
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        tip = res.get("tip", None)
        p_center = res.get("p_centerline", None)
        info = res.get("info", {})
        success = info.get("success", None)

        print("success             :", success)
        if "message" in info:
            print("message             :", info["message"])
        if "W" in info:
            print("W                   :", info["W"])

        if tip is not None:
            print("tip                 :", tip)

        if p_center is not None:
            print("centerline shape    :", p_center.shape)
            p_straight = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, p_center.shape[1])
            tip_straight = p_straight[:, -1]
            print("straight tip        :", tip_straight)
            print("tip deflection      :", tip - tip_straight)
            print("tip deflection norm :", np.linalg.norm(tip - tip_straight))

    def nodewise_error(a, b):
        if (a is None) or (b is None):
            return None
        if a.shape != b.shape:
            return None
        return np.linalg.norm(a - b, axis=0)

    def plot_centerlines(results, p0_ur, q0_ur, L_cmd, r_src, lumen_C=None, title="Model comparison"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        n_plot = None
        for res in results.values():
            if res["p_centerline"] is not None:
                n_plot = res["p_centerline"].shape[1]
                break

        if n_plot is not None:
            p_straight = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, n_plot)
            ax.plot(
                p_straight[0], p_straight[1], p_straight[2],
                "--", linewidth=1.5, label="straight rod"
            )

        for name, res in results.items():
            p = res["p_centerline"]
            if p is None:
                continue
            ax.plot(
                p[0], p[1], p[2],
                "-o", linewidth=2, markersize=4, label=name
            )

        ax.scatter([p0_ur[0]], [p0_ur[1]], [p0_ur[2]], s=80, marker="o", label="base")
        ax.scatter([r_src[0]], [r_src[1]], [r_src[2]], s=90, marker="s", label="magnet")

        if lumen_C is not None:
            ax.plot(lumen_C[:, 0], lumen_C[:, 1], lumen_C[:, 2], "k:", linewidth=1.5, label="lumen")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        set_axes_equal_3d(ax)
        plt.tight_layout()
        plt.show()
    def dipole_from_pose(q_src, m_body):
        R = quat_to_R(q_src)
        return R @ m_body  # (3,)
    def compare_tips(results):
        names = list(results.keys())
        print("\n" + "=" * 80)
        print("TIP COMPARISON")
        print("=" * 80)
        for name in names:
            print(f"{name:24s}: {results[name]['tip']}")

        print("\nPairwise tip differences:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni, nj = names[i], names[j]
                ti, tj = results[ni]["tip"], results[nj]["tip"]
                if (ti is None) or (tj is None):
                    continue
                print(
                    f"{ni:24s} vs {nj:24s} | "
                    f"norm = {np.linalg.norm(ti - tj):.6e}"
                )

    # ============================================================
    # Common case setup
    # ============================================================
    beam_params = default_beam_params()
    mag_params = default_magnet_params()

    L_cmd = 0.01
    N_nodes = 5

    pivot_point = np.array([
        0.7981328220229531, -0.7112731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.06),
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

    # Source dipole in body frame
    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)
    m_src = dipole_from_pose(q_src_ur, m_body)
    mag_len = beam_params.length_of_mag
    mu_line = beam_params.mag * beam_params.A_cs
    M_ref_local = np.array([0.0, 0.0, mu_line], float)

    # Effective lengths for direct solvers
    L_model, wire_len, tip_len = effective_lengths(
        L_cmd,
        L_tip_full=mag_len,
        L_tip_min=0.01,
    )

    wire = rod_section_stiffness(r=200e-6, E=50e6, nu=0.4)
    tip = rod_section_stiffness(r=beam_params.r, E=beam_params.E, nu=0.49)

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

    print("\n" + "=" * 80)
    print("COMMON CASE SETUP")
    print("=" * 80)
    print(f"L_cmd            : {L_cmd:.6e}")
    print(f"L_model          : {L_model:.6e}")
    print(f"wire_len         : {wire_len:.6e}")
    print(f"tip_len          : {tip_len:.6e}")
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

    # ============================================================
    # Optional lumen
    # ============================================================
    lumen_C = None
    lumen_R = None

    if USE_LUMEN:
        Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
        t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

        lumen_C = make_lumen_centerline_turning(
            p_start=p0_ur,
            t0=t0,
            length=0.03,
            n_pts=130,
            bend_axis=np.array([0.0, 0.0, 1.0]),
            bend_angle=np.deg2rad(0.0),
            bend_start=0.005,
            bend_end=0.03,
        )
        lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
        lumen_R = np.full(len(lumen_C), 0.004)

    if lumen_C is None:
        lumen_C_safe = np.zeros((2, 3), float)
        lumen_R_safe = np.full(2, 1.0, float)
    else:
        lumen_C_safe = lumen_C
        lumen_R_safe = lumen_R

    # ============================================================
    # Build wrappers
    # ============================================================
    fwd_cos = None
    if RUN_COS_WRAPPER:
        fwd_cos = EnergyMinForwardWithLumen(
            p0_ur=p0_ur,
            q0_ur=q0_ur,
            Kinv_fun=Kinv_fun,
            u_star=np.zeros(3),
            m_body=m_body,
            lumen_C=lumen_C_safe,
            lumen_R=lumen_R_safe,
            N_nodes=N_nodes,
            maxiter=30,
            L0_init=0.01,
            dL_internal=0.01,
            use_lumen_jac=USE_LUMEN,
            L_tip_full=mag_len,
            L_tip_min=0.01,
        )

    fwd_der = None
    if RUN_DER_WRAPPER:
        fwd_der = DEREnergyMinForwardWithLumen(
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
            maxiter=60,
            amp_init=2e-4,
            enforce_inextensibility=True,
            use_lumen=USE_LUMEN,
            L_tip_full=mag_len,
            L_tip_min=0.01,
            ref_twist=None,
            use_continuation=False,
            N_coarse=9,
        )
    # ============================================================
    # Sweep over magnet angle and save results
    # ============================================================
    angles_deg = np.arange(0, 5, 5)   # 0, -5, -10, ..., -90
    all_rows = []

    for angle_deg in angles_deg:
        print("\n" + "#" * 100)
        print(f"RUNNING ANGLE = {angle_deg} deg")
        print("#" * 100)

        # --------------------------------------------------------
        # Recompute source pose for this angle
        # --------------------------------------------------------
        start_point = np.asarray(get_point(0, angle_deg, base_point, pivot_point), dtype=float)
        start_point[2] = -0.1

        T_ur_mag = ur_pose6_to_T(start_point)
        r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

        m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)
        m_src = dipole_from_pose(q_src_ur, m_body)

        # common p7 input for wrappers
        p7 = rotvec_pose7_from_start_point(start_point, L_cmd)

        results = {}

        # ---- 1) Cosserat wrapper ----
        if fwd_cos is not None:
            try:
                tip_cos = fwd_cos(p7)
                results["cos_wrapper"] = dict(
                    tip=np.asarray(tip_cos, float).copy(),
                    p_centerline=None if fwd_cos.last_p_centerline is None else fwd_cos.last_p_centerline.copy(),
                    info=dict(success=True, message="wrapper run complete"),
                )
            except Exception as e:
                print(f"cos_wrapper failed at angle {angle_deg}: {e}")
                results["cos_wrapper"] = dict(tip=None, p_centerline=None, info=dict(success=False, message=str(e)))

        # ---- 2) DER wrapper ----
        if fwd_der is not None:
            try:
                tip_der = fwd_der(p7)
                info = fwd_der.last_info if fwd_der.last_info is not None else {}
                results["der_wrapper"] = dict(
                    tip=np.asarray(tip_der, float).copy(),
                    p_centerline=None if fwd_der.last_p_centerline is None else fwd_der.last_p_centerline.copy(),
                    info=dict(
                        success=bool(info.get("success", True)),
                        message=str(info.get("message", "")),
                        W=float(info.get("W", np.nan)) if "W" in info else np.nan,
                        dbg=info.get("dbg", None),
                        wire_len=info.get("wire_len", wire_len),
                        tip_len=info.get("tip_len", tip_len),
                        m_src=info.get("m_src", None),
                        r_src=info.get("r_src", None),
                        q_opt=info.get("q_opt", None),
                    ),
                )
            except Exception as e:
                print(f"der_wrapper failed at angle {angle_deg}: {e}")
                results["der_wrapper"] = dict(tip=None, p_centerline=None, info=dict(success=False, message=str(e)))

        # ---- 3) Direct DER energy minimization ----
        info_min_direct = None
        if RUN_DER_DIRECT_MIN:
            try:
                p_min, theta_min, info_min = solve_nodes_twist_min(
                    p0=p0_ur,
                    q0=q0_ur,
                    L=L_model,
                    N=N_nodes,
                    wire_len=wire_len,
                    tip_len=tip_len,
                    Kinv_fun=Kinv_fun,
                    r_src=r_src_ur,
                    m_src=m_src,
                    mu_tip=mu_line,
                    EA_wire=EA_wire,
                    EA_tip=EA_tip,
                    lumen_C=lumen_C,
                    lumen_R=lumen_R,
                    use_lumen=USE_LUMEN,
                    maxiter=300,
                    M_ref_local=M_ref_local,
                    ref_twist=None,
                    enforce_inextensibility=True,
                    amp_init=2e-4,
                    q_init=None,
                )
                info_min_direct = info_min
                results["der_direct_min"] = dict(
                    tip=p_min[:, -1].copy(),
                    p_centerline=p_min.copy(),
                    info=info_min,
                )
            except Exception as e:
                print(f"der_direct_min failed at angle {angle_deg}: {e}")
                results["der_direct_min"] = dict(tip=None, p_centerline=None, info=dict(success=False, message=str(e)))

        # ---- 4) Direct DER residual solver ----
        if RUN_DER_RESIDUAL:
            try:
                q_init_res = None
                if info_min_direct is not None and "q_opt" in info_min_direct:
                    q_init_res = info_min_direct["q_opt"]

                p_res, theta_res, info_res = solve_nodes_twist_residual(
                    p0=p0_ur,
                    q0=q0_ur,
                    L=L_model,
                    N=N_nodes,
                    wire_len=wire_len,
                    tip_len=tip_len,
                    Kinv_fun=Kinv_fun,
                    r_src=r_src_ur,
                    m_src=m_src,
                    mu_tip=mu_line,
                    EA_wire=EA_wire,
                    EA_tip=EA_tip,
                    lumen_C=lumen_C,
                    lumen_R=lumen_R,
                    use_lumen=USE_LUMEN,
                    maxiter=200,
                    M_ref_local=M_ref_local,
                    ref_twist=None,
                    q_init=q_init_res,
                )
                results["der_residual"] = dict(
                    tip=p_res[:, -1].copy(),
                    p_centerline=p_res.copy(),
                    info=info_res,
                )
            except Exception as e:
                print(f"der_residual failed at angle {angle_deg}: {e}")
                results["der_residual"] = dict(tip=None, p_centerline=None, info=dict(success=False, message=str(e)))

        # ---- 5) Direct continuous Cosserat BVP ----
        if RUN_DIRECT_COSSERAT:
            try:
                model = CosseratForwardModel(
                    p0=p0_ur,
                    q0=q0_ur,
                    Kinv_fun=Kinv_fun,
                    m_local_fun=make_m_local_fun_wire_tip(
                        wire_len,
                        len_tip=tip_len,
                        mode="axial",
                        alpha_end=0.0,
                    ),
                    m_moment=0.0,
                    wire_len=wire_len,
                )

                out = model.forward(
                    L=L_model,
                    r_src=r_src_ur,
                    q_src=q_src_ur,
                    wire_len=wire_len,
                    m_body=m_body,
                )

                p_cos_direct = out["profiles"]["p"]
                results["cos_bvp_direct"] = dict(
                    tip=out["p_tip"].copy(),
                    p_centerline=p_cos_direct.copy(),
                    info=dict(
                        success=bool(out["solved"]),
                        message=str(out["message"]),
                        B_tip=float(out["B_tip"]),
                        F_net=np.asarray(out["F_net"], float).copy(),
                        T_net=np.asarray(out["T_net"], float).copy(),
                        theta_y=float(out["theta_y"]),
                        theta_z=float(out["theta_z"]),
                        theta_total=float(out["theta_total"]),
                    ),
                )
            except Exception as e:
                print(f"cos_bvp_direct failed at angle {angle_deg}: {e}")
                results["cos_bvp_direct"] = dict(tip=None, p_centerline=None, info=dict(success=False, message=str(e)))

        # --------------------------------------------------------
        # Save one summary row for this angle
        # --------------------------------------------------------
        row = {
            "angle_deg": float(angle_deg),
            "magnet_x": float(r_src_ur[0]),
            "magnet_y": float(r_src_ur[1]),
            "magnet_z": float(r_src_ur[2]),
        }

        for model_name, res in results.items():
            tip = res.get("tip", None)
            info = res.get("info", {})

            tx, ty, tz = vec3_or_nan(tip)

            row[f"{model_name}_success"] = info.get("success", np.nan)
            row[f"{model_name}_tip_x"] = tx
            row[f"{model_name}_tip_y"] = ty
            row[f"{model_name}_tip_z"] = tz
            row[f"{model_name}_message"] = info.get("message", "")

            if "W" in info:
                row[f"{model_name}_W"] = info.get("W", np.nan)
            if "B_tip" in info:
                row[f"{model_name}_B_tip"] = info.get("B_tip", np.nan)
            if "theta_y" in info:
                row[f"{model_name}_theta_y"] = info.get("theta_y", np.nan)
            if "theta_z" in info:
                row[f"{model_name}_theta_z"] = info.get("theta_z", np.nan)
            if "theta_total" in info:
                row[f"{model_name}_theta_total"] = info.get("theta_total", np.nan)

        # pairwise tip differences
        model_keys = list(results.keys())
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                ni = model_keys[i]
                nj = model_keys[j]
                ti = results[ni]["tip"]
                tj = results[nj]["tip"]

                if (ti is None) or (tj is None):
                    row[f"tipdiff_{ni}_vs_{nj}"] = np.nan
                else:
                    row[f"tipdiff_{ni}_vs_{nj}"] = float(np.linalg.norm(np.asarray(ti) - np.asarray(tj)))

        all_rows.append(row)
    df = pd.DataFrame(all_rows)

    out_dir = "comparison_outputs"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "model_comparison_angle_sweep_scrap.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved CSV to:")
    print(csv_path)
    print("\nColumns:")
    print(df.columns.tolist())
    # ============================================================
    # Common p7 input for wrappers
    # ============================================================
    p7 = rotvec_pose7_from_start_point(start_point, L_cmd)

    # ============================================================
    # Run all models
    # ============================================================
    results = {}

    # ---- 1) Cosserat wrapper ----
    if fwd_cos is not None:
        tip_cos = fwd_cos(p7)
        results["cos_wrapper"] = dict(
            tip=np.asarray(tip_cos, float).copy(),
            p_centerline=None if fwd_cos.last_p_centerline is None else fwd_cos.last_p_centerline.copy(),
            info=dict(
                success=True,
                message="wrapper run complete",
            ),
        )

    # ---- 2) DER wrapper ----
    info_min_direct = None
    if fwd_der is not None:
        tip_der = fwd_der(p7)
        info = fwd_der.last_info if fwd_der.last_info is not None else {}
        results["der_wrapper"] = dict(
            tip=np.asarray(tip_der, float).copy(),
            p_centerline=None if fwd_der.last_p_centerline is None else fwd_der.last_p_centerline.copy(),
            info=dict(
                success=bool(info.get("success", True)),
                message=str(info.get("message", "")),
                W=float(info.get("W", np.nan)) if "W" in info else np.nan,
                dbg=info.get("dbg", None),
                wire_len=info.get("wire_len", wire_len),
                tip_len=info.get("tip_len", tip_len),
                m_src=info.get("m_src", None),
                r_src=info.get("r_src", None),
                q_opt=info.get("q_opt", None),
            ),
        )

    # ---- 3) Direct DER energy minimization ----
    if RUN_DER_DIRECT_MIN:
        p_min, theta_min, info_min = solve_nodes_twist_min(
            p0=p0_ur,
            q0=q0_ur,
            L=L_model,
            N=N_nodes,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src_ur,
            m_src=m_src,   # replace with world-frame source dipole if your solver expects that
            mu_tip=mu_line,
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=USE_LUMEN,
            maxiter=50,
            M_ref_local=M_ref_local,
            ref_twist=None,
            enforce_inextensibility=True,
            amp_init=2e-4,
            q_init=None,
        )
        info_min_direct = info_min
        results["der_direct_min"] = dict(
            tip=p_min[:, -1].copy(),
            p_centerline=p_min.copy(),
            info=info_min,
        )

    # ---- 4) Direct DER residual solver ----
    if RUN_DER_RESIDUAL:
        q_init_res = None
        if info_min_direct is not None and "q_opt" in info_min_direct:
            q_init_res = info_min_direct["q_opt"]

        p_res, theta_res, info_res = solve_nodes_twist_residual(
            p0=p0_ur,
            q0=q0_ur,
            L=L_model,
            N=N_nodes,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src_ur,
            m_src=m_src,   # replace with world-frame source dipole if expected
            mu_tip=mu_line,
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=USE_LUMEN,
            maxiter=200,
            M_ref_local=M_ref_local,
            ref_twist=None,
            q_init=q_init_res,
        )
        results["der_residual"] = dict(
            tip=p_res[:, -1].copy(),
            p_centerline=p_res.copy(),
            info=info_res,
        )

    # ---- 5) Direct continuous Cosserat BVP ----
    if RUN_DIRECT_COSSERAT:
        model = CosseratForwardModel(
            p0=p0_ur,
            q0=q0_ur,
            Kinv_fun=Kinv_fun,
            m_local_fun=make_m_local_fun_wire_tip(
                wire_len,
                len_tip=tip_len,
                mode="axial",
                alpha_end=0.0,
            ),
            m_moment=0.0,
            wire_len=wire_len,
        )

        out = model.forward(
            L=L_model,
            r_src=r_src_ur,
            q_src=q_src_ur,
            wire_len=wire_len,
            m_body=m_body,
        )

        p_cos_direct = out["profiles"]["p"]
        results["cos_bvp_direct"] = dict(
            tip=out["p_tip"].copy(),
            p_centerline=p_cos_direct.copy(),
            info=dict(
                success=bool(out["solved"]),
                message=str(out["message"]),
                B_tip=float(out["B_tip"]),
                F_net=np.asarray(out["F_net"], float).copy(),
                T_net=np.asarray(out["T_net"], float).copy(),
                theta_y=float(out["theta_y"]),
                theta_z=float(out["theta_z"]),
                theta_total=float(out["theta_total"]),
            ),
        )

        if hasattr(model, "_last_energy_choice"):
            results["cos_bvp_direct"]["info"]["energy_choice"] = model._last_energy_choice

    # ============================================================
    # Summaries
    # ============================================================
    for name, res in results.items():
        summarize_result(name, res, p0_ur, q0_ur, L_model)

    compare_tips(results)

    # ============================================================
    # Pairwise nodewise errors where possible
    # ============================================================
    print("\n" + "=" * 80)
    print("NODEWISE ERRORS")
    print("=" * 80)
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            err = nodewise_error(results[ni]["p_centerline"], results[nj]["p_centerline"])
            if err is None:
                print(f"{ni:24s} vs {nj:24s}: shapes differ or missing")
            else:
                print(
                    f"{ni:24s} vs {nj:24s} | "
                    f"mean = {np.mean(err):.6e} | max = {np.max(err):.6e}"
                )

    # ============================================================
    # Plot all centerlines
    # ============================================================
    plot_centerlines(
        results,
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        L_cmd=L_model,
        r_src=r_src_ur,
        lumen_C=lumen_C,
        title="All 4 rod models comparison",
    )

    # ============================================================
    # Optional magnetic field comparison on identical points
    # ============================================================
    if "der_wrapper" in results:
        der_info = results["der_wrapper"]["info"]
        dbg = der_info.get("dbg", None)
        if dbg is not None and "kin" in dbg and "mag" in dbg:
            kin = dbg["kin"]
            mag = dbg["mag"]
            pts = kin["mid"].T

            m_src_used = der_info.get("m_src", None)
            r_src_used = der_info.get("r_src", None)

            if (m_src_used is not None) and (r_src_used is not None):
                B_cos = dipole_field_from_source(pts, r_src_used, m_src_used, r_min=1e-6)
                B_der = dipole_field_points(pts.T, r_src_used, m_src_used, r_min=1e-6).T

                err = np.linalg.norm(B_cos - B_der, axis=1)

                print("\n" + "=" * 80)
                print("FIELD LAW COMPARISON ON IDENTICAL POINTS")
                print("=" * 80)
                print("max |B_cos - B_der| =", np.max(err))
                print("mean|B_cos - B_der| =", np.mean(err))
                print("rel max error       =", np.max(err) / (np.max(np.linalg.norm(B_cos, axis=1)) + 1e-12))

                M_world = mag["M_world"].T
                ell_i = mag["ell_i"]
                mdotB_same = np.sum(M_world * B_cos, axis=1)
                Wm_same_midpoint = np.sum(-mdotB_same * ell_i)

                print("DER Wm stored        =", mag["wm_seg"].sum())
                print("DER Wm recomputed    =", Wm_same_midpoint)
                print("difference           =", Wm_same_midpoint - mag["wm_seg"].sum())

    import numpy as np
    import pandas as pd
    import os

    
    # ------------------------------------------------------------
    # Jacobian configuration
    # ------------------------------------------------------------
    dt = 1.0
    dr = 5e-3
    dtheta = 3e-1
    dL = 1e-3

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        dtheta / dt, dtheta / dt, dtheta / dt,
        dL / dt,
    ], dtype=float)

    row_labels = ["tip_x", "tip_y", "tip_z"]
    col_labels_full = ["vx", "vy", "vz", "wx", "wy", "wz", "L"]
    col_labels_red = ["vx", "vy", "wz", "L"]

    p8_nom = np.hstack([r_src_ur, q_src_ur, float(L_cmd)])


    # ------------------------------------------------------------
    # Helper: build all Jacobian forward functions
    # ------------------------------------------------------------
    def build_jacobian_forward_functions(
        *,
        fwd_cos,
        fwd_der,
        RUN_DER_DIRECT_MIN,
        RUN_DER_RESIDUAL,
        RUN_DIRECT_COSSERAT,
        results,
        p0_ur,
        q0_ur,
        Kinv_fun,
        EA_wire,
        EA_tip,
        M_ref_local,
        m_body,
        lumen_C,
        lumen_R,
        USE_LUMEN,
        N_nodes,
        mag_len,
    ):
        jac_forward_fns = {}

        # -------------------------
        # Wrapper models
        # -------------------------
        if fwd_cos is not None:
            fwd_cos_cold = WarmForwardP7(fwd_cos)
            jac_forward_fns["cos_wrapper"] = make_forward_tip_fn_from_p7_model(
                fwd_cos_cold,
                "cos_wrapper",
            )

        if fwd_der is not None:
            fwd_der_cold = WarmForwardP7(fwd_der)
            jac_forward_fns["der_wrapper"] = make_forward_tip_fn_from_p7_model(
                fwd_der_cold,
                "der_wrapper",
            )

        # -------------------------
        # Direct DER energy minimisation
        # -------------------------
        if RUN_DER_DIRECT_MIN:
            jac_forward_fns["der_direct_min"] = make_forward_tip_fn_der_direct_min(
                p0_ur=p0_ur,
                q0_ur=q0_ur,
                Kinv_fun=Kinv_fun,
                EA_wire=EA_wire,
                EA_tip=EA_tip,
                M_ref_local=M_ref_local,
                m_body=m_body,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                use_lumen=USE_LUMEN,
                N_nodes=N_nodes,
                mag_len=mag_len,
            )

        # -------------------------
        # Direct DER residual
        # -------------------------
        if RUN_DER_RESIDUAL:
            q_seed = None
            if (
                "der_direct_min" in results
                and isinstance(results["der_direct_min"], dict)
                and "info" in results["der_direct_min"]
                and isinstance(results["der_direct_min"]["info"], dict)
                and "q_opt" in results["der_direct_min"]["info"]
            ):
                q_seed = results["der_direct_min"]["info"]["q_opt"]

            jac_forward_fns["der_residual"] = make_forward_tip_fn_der_residual(
                p0_ur=p0_ur,
                q0_ur=q0_ur,
                Kinv_fun=Kinv_fun,
                EA_wire=EA_wire,
                EA_tip=EA_tip,
                M_ref_local=M_ref_local,
                m_body=m_body,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                use_lumen=USE_LUMEN,
                N_nodes=N_nodes,
                mag_len=mag_len,
                q_init_seed=q_seed,
            )

        # -------------------------
        # Direct Cosserat
        # -------------------------
        if RUN_DIRECT_COSSERAT:
            jac_forward_fns["cos_bvp_direct"] = make_forward_tip_fn_cosserat_direct(
                p0_ur=p0_ur,
                q0_ur=q0_ur,
                Kinv_fun=Kinv_fun,
                m_body=m_body,
                mag_len=mag_len,
            )

        return jac_forward_fns


    # ------------------------------------------------------------
    # Helper: compute Jacobians for all models
    # ------------------------------------------------------------
    def compute_all_jacobians(
        *,
        jac_forward_fns,
        p8_nom,
        dt,
        eps_u,
        dL,
        n_out=3,
        n_repeat=1,
        row_labels=None,
        col_labels_full=None,
        col_labels_red=None,
        print_tables=True,
    ):
        jac_stats = {}

        for name, forward_fn in jac_forward_fns.items():
            print("\n" + "-" * 80)
            print(f"Computing Jacobian for: {name}")
            print("-" * 80)

            stats = benchmark_jacobian_mode(
                label=name,
                forward_tip_fn=forward_fn,
                p8_nom=p8_nom,
                dt=dt,
                eps_u=eps_u,
                dL=dL,
                n_out=n_out,
                n_repeat=n_repeat,
            )
            jac_stats[name] = stats

            if print_tables:
                print_jacobian_table(
                    stats["B_full"],
                    row_labels=row_labels,
                    col_labels=col_labels_full,
                    title=f"{name} full Jacobian",
                )

                print_jacobian_table(
                    stats["B_red"],
                    row_labels=row_labels,
                    col_labels=col_labels_red,
                    title=f"{name} reduced Jacobian",
                )

        return jac_stats


    # ------------------------------------------------------------
    # Helper: pairwise comparisons
    # ------------------------------------------------------------
    def compare_all_jacobians(
        *,
        jac_stats,
        row_labels,
        col_labels_full,
        col_labels_red,
    ):
        jac_names = list(jac_stats.keys())

        for i in range(len(jac_names)):
            for j in range(i + 1, len(jac_names)):
                ni = jac_names[i]
                nj = jac_names[j]

                print("\n" + "=" * 80)
                print(f"JACOBIAN COMPARISON: {ni} vs {nj}")
                print("=" * 80)

                print_jacobian_comparison(
                    jac_stats[ni]["B_full"],
                    jac_stats[nj]["B_full"],
                    row_labels=row_labels,
                    col_labels=col_labels_full,
                    name_a=f"{ni}_full",
                    name_b=f"{nj}_full",
                )

                print_jacobian_comparison(
                    jac_stats[ni]["B_red"],
                    jac_stats[nj]["B_red"],
                    row_labels=row_labels,
                    col_labels=col_labels_red,
                    name_a=f"{ni}_red",
                    name_b=f"{nj}_red",
                )


    # ------------------------------------------------------------
    # Helper: save Jacobians to CSV
    # ------------------------------------------------------------
    def save_jacobians_to_csv(
        jac_stats,
        out_dir="jacobian_outputs_10nodes",
        row_labels=None,
        col_labels_full=None,
        col_labels_red=None,
    ):
        os.makedirs(out_dir, exist_ok=True)

        for name, stats in jac_stats.items():
            B_full = np.asarray(stats["B_full"], float)
            B_red = np.asarray(stats["B_red"], float)

            df_full = pd.DataFrame(B_full, index=row_labels, columns=col_labels_full)
            df_red = pd.DataFrame(B_red, index=row_labels, columns=col_labels_red)

            df_full.to_csv(os.path.join(out_dir, f"{name}_jacobian_full.csv"))
            df_red.to_csv(os.path.join(out_dir, f"{name}_jacobian_reduced.csv"))

        # Also save one summary table if timing/norm metadata exists
        summary_rows = []
        for name, stats in jac_stats.items():
            row = {"model": name}

            if "elapsed_mean" in stats:
                row["elapsed_mean"] = stats["elapsed_mean"]
            if "elapsed_std" in stats:
                row["elapsed_std"] = stats["elapsed_std"]
            if "B_full" in stats:
                row["full_fro_norm"] = float(np.linalg.norm(stats["B_full"]))
            if "B_red" in stats:
                row["red_fro_norm"] = float(np.linalg.norm(stats["B_red"]))

            summary_rows.append(row)

        if len(summary_rows) > 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, "jacobian_summary.csv"),
                index=False,
            )

        print(f"\nSaved Jacobian CSV files to: {out_dir}")


    # ------------------------------------------------------------
    # Main Jacobian run
    # ------------------------------------------------------------
    jac_forward_fns = build_jacobian_forward_functions(
        fwd_cos=fwd_cos,
        fwd_der=fwd_der,
        RUN_DER_DIRECT_MIN=RUN_DER_DIRECT_MIN,
        RUN_DER_RESIDUAL=RUN_DER_RESIDUAL,
        RUN_DIRECT_COSSERAT=RUN_DIRECT_COSSERAT,
        results=results,
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        EA_wire=EA_wire,
        EA_tip=EA_tip,
        M_ref_local=M_ref_local,
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        USE_LUMEN=USE_LUMEN,
        N_nodes=N_nodes,
        mag_len=mag_len,
    )

    jac_stats = compute_all_jacobians(
        jac_forward_fns=jac_forward_fns,
        p8_nom=p8_nom,
        dt=dt,
        eps_u=eps_u,
        dL=dL,
        n_out=3,
        n_repeat=1,   # increase to 3 if you want timing averages
        row_labels=row_labels,
        col_labels_full=col_labels_full,
        col_labels_red=col_labels_red,
        print_tables=True,
    )

    compare_all_jacobians(
        jac_stats=jac_stats,
        row_labels=row_labels,
        col_labels_full=col_labels_full,
        col_labels_red=col_labels_red,
    )

    save_jacobians_to_csv(
        jac_stats,
        out_dir="jacobian_outputs",
        row_labels=row_labels,
        col_labels_full=col_labels_full,
        col_labels_red=col_labels_red,
    )
# # ------------------------------------------------------------
# # Jacobian setup
# # ------------------------------------------------------------
# dt = 1.0
# dr = 5e-3
# dtheta = 3e-1
# dL = 1e-3

# eps_u = np.array([
#     dr / dt, dr / dt, dr / dt,
#     dtheta / dt, dtheta / dt, dtheta / dt,
#     dL / dt
# ], dtype=float)

# row_labels = ["tip_x", "tip_y", "tip_z"]
# col_labels_full = ["vx", "vy", "vz", "wx", "wy", "wz", "L"]
# col_labels_red = ["vx", "vy", "wz", "L"]

# p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

# jac_forward_fns = {}

# # wrapper models: cold versions are fairest for Jacobians
# if fwd_cos is not None:
#     fwd_cos_cold = WarmForwardP7(fwd_cos)
#     jac_forward_fns["cos_wrapper"] = make_forward_tip_fn_from_p7_model(fwd_cos_cold, "cos_wrapper")

# if fwd_der is not None:
#     fwd_der_cold = WarmForwardP7(fwd_der)
#     jac_forward_fns["der_wrapper"] = make_forward_tip_fn_from_p7_model(fwd_der_cold, "der_wrapper")

# # direct models
# if RUN_DER_DIRECT_MIN:
#     jac_forward_fns["der_direct_min"] = make_forward_tip_fn_der_direct_min(
#         p0_ur=p0_ur,
#         q0_ur=q0_ur,
#         Kinv_fun=Kinv_fun,
#         EA_wire=EA_wire,
#         EA_tip=EA_tip,
#         M_ref_local=M_ref_local,
#         m_body=m_body,
#         lumen_C=lumen_C,
#         lumen_R=lumen_R,
#         use_lumen=USE_LUMEN,
#         N_nodes=N_nodes,
#         mag_len=mag_len,
#     )

# if RUN_DER_RESIDUAL:
#     q_seed = None
#     if "der_direct_min" in results and "q_opt" in results["der_direct_min"]["info"]:
#         q_seed = results["der_direct_min"]["info"]["q_opt"]
#     jac_forward_fns["der_residual"] = make_forward_tip_fn_der_residual(
#         p0_ur=p0_ur,
#         q0_ur=q0_ur,
#         Kinv_fun=Kinv_fun,
#         EA_wire=EA_wire,
#         EA_tip=EA_tip,
#         M_ref_local=M_ref_local,
#         m_body=m_body,
#         lumen_C=lumen_C,
#         lumen_R=lumen_R,
#         use_lumen=USE_LUMEN,
#         N_nodes=N_nodes,
#         mag_len=mag_len,
#         q_init_seed=q_seed,
#     )

# if RUN_DIRECT_COSSERAT:
#     jac_forward_fns["cos_bvp_direct"] = make_forward_tip_fn_cosserat_direct(
#         p0_ur=p0_ur,
#         q0_ur=q0_ur,
#         Kinv_fun=Kinv_fun,
#         m_body=m_body,
#         mag_len=mag_len,
#     )
# # ------------------------------------------------------------
# # Compute Jacobians for all models
# # ------------------------------------------------------------
# jac_stats = {}

# for name, forward_fn in jac_forward_fns.items():
#     stats = benchmark_jacobian_mode(
#         label=name,
#         forward_tip_fn=forward_fn,
#         p8_nom=p8_nom,
#         dt=dt,
#         eps_u=eps_u,
#         dL=dL,
#         n_out=3,
#         n_repeat=1,   # raise to 3 for timing averages
#     )
#     jac_stats[name] = stats

#     print_jacobian_table(
#         stats["B_full"],
#         row_labels=row_labels,
#         col_labels=col_labels_full,
#         title=f"{name} full Jacobian"
#     )

#     print_jacobian_table(
#         stats["B_red"],
#         row_labels=row_labels,
#         col_labels=col_labels_red,
#         title=f"{name} reduced Jacobian"
#     )
# # ------------------------------------------------------------
# # Pairwise Jacobian comparisons
# # ------------------------------------------------------------
# jac_names = list(jac_stats.keys())

# for i in range(len(jac_names)):
#     for j in range(i + 1, len(jac_names)):
#         ni = jac_names[i]
#         nj = jac_names[j]

#         print("\n" + "=" * 80)
#         print(f"JACOBIAN COMPARISON: {ni} vs {nj}")
#         print("=" * 80)

#         print_jacobian_comparison(
#             jac_stats[ni]["B_full"],
#             jac_stats[nj]["B_full"],
#             row_labels=row_labels,
#             col_labels=col_labels_full,
#             name_a=f"{ni}_full",
#             name_b=f"{nj}_full",
#         )

#         print_jacobian_comparison(
#             jac_stats[ni]["B_red"],
#             jac_stats[nj]["B_red"],
#             row_labels=row_labels,
#             col_labels=col_labels_red,
#             name_a=f"{ni}_red",
#             name_b=f"{nj}_red",
#         )