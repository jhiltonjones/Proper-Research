from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    make_lumen_centerline_turning,
    CosseratForwardModel,
    dipole_from_pose,
    u0_from_bvp,
    solve_quasistatic_insertion,
    lumen_violation_profile,
)
from beam_direction_magnetisation.post_processing.post_processing import (
    plot_energy_only_3d,
    make_lumen_centerline_double_turn,
)
from proper_research.control.no_path_plan_mpc import DeterministicForward6D, numerical_B_y_wrt_u, integrate_pose8_body
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
from proper_research.control.mpc_boundary import resample_polyline
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, effective_lengths
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.spatial.transform import Rotation as Rot
def controller_B_matrix(p8, forward6d_obj, dt, eps_u):
    """
    Build the controller's local input-output Jacobian:
        B = dy/du
    using the exact same routine as MPC.
    """
    forward6d_obj.start_step()
    B = numerical_B_y_wrt_u(
        p8=np.asarray(p8, float).copy(),
        forward_y_fn=forward6d_obj,
        dt=float(dt),
        eps_u=np.asarray(eps_u, float),
        n_out=6,
    )
    return np.asarray(B, float)
def eval_one_step_truth(p8, du, dt, forward6d_truth):
    """
    True one-step plant response using the same input integration as the controller.
    Returns:
        y0, y1, dy_true, p1
    """
    p8 = np.asarray(p8, float).copy()
    du = np.asarray(du, float).reshape(7,)

    forward6d_truth.start_step()
    y0 = np.asarray(forward6d_truth(p8, commit=False), float).reshape(6,)

    p1 = integrate_pose8_body(p8, du, dt)
    forward6d_truth.start_step()
    y1 = np.asarray(forward6d_truth(p1, commit=False), float).reshape(6,)

    dy_true = y1 - y0
    return y0, y1, dy_true, p1
def sample_control_perturbations(eps_u, n_dirs=20, seed=0, scale=0.25):
    """
    Random perturbations in control space.
    scale=0.25 means 25% of eps_u magnitude per random direction.
    """
    eps_u = np.asarray(eps_u, float).reshape(7,)
    rng = np.random.default_rng(seed)

    U = []
    for _ in range(n_dirs):
        d = rng.standard_normal(7)
        d /= (np.linalg.norm(d) + 1e-12)
        du = scale * eps_u * d
        U.append(du)
    return U
def axis_control_perturbations(eps_u, scale=0.25):
    eps_u = np.asarray(eps_u, float).reshape(7,)
    U = []
    for i in range(7):
        du_p = np.zeros(7); du_p[i] = scale * eps_u[i]
        du_m = np.zeros(7); du_m[i] = -scale * eps_u[i]
        U.append(du_p)
        U.append(du_m)
    return U
def compare_controller_jacobians_at_point(
    p8_nom,
    dt,
    eps_u,
    forward6d_truth,
    forward6d_bc,
    forward6d_wrong,
    lumen_C,
    lumen_R,
    du_list,
):
    """
    Fair comparison:
      - truth plant = forward6d_truth
      - B_bc built using controller's own finite-difference wrt u
      - B_wrong built the same way, but with no-BC Jacobian model
    """
    p8_nom = np.asarray(p8_nom, float).copy()

    # Nominal truth output / centerline / clearance
    forward6d_truth.start_step()
    y0 = np.asarray(forward6d_truth(p8_nom, commit=False), float).reshape(6,)
    C0 = None if forward6d_truth.last_p_centerline is None else np.asarray(forward6d_truth.last_p_centerline, float).copy()
    clr0 = None if C0 is None else min_clearance_from_centerline(C0, lumen_C, lumen_R)

    # Controller Jacobians
    B_bc = controller_B_matrix(p8_nom, forward6d_bc, dt, eps_u)
    B_wrong = controller_B_matrix(p8_nom, forward6d_wrong, dt, eps_u)

    rows = []
    for k, du in enumerate(du_list):
        du = np.asarray(du, float).reshape(7,)

        _, y1, dy_true, p1 = eval_one_step_truth(p8_nom, du, dt, forward6d_truth)

        dy_bc = B_bc @ du
        dy_wrong = B_wrong @ du

        err_bc = np.linalg.norm(dy_true - dy_bc)
        err_wrong = np.linalg.norm(dy_true - dy_wrong)

        denom = np.linalg.norm(dy_true) + 1e-12
        rel_bc = err_bc / denom
        rel_wrong = err_wrong / denom

        forward6d_truth.start_step()
        _ = forward6d_truth(p1, commit=False)
        C1 = None if forward6d_truth.last_p_centerline is None else np.asarray(forward6d_truth.last_p_centerline, float).copy()
        clr1 = None if C1 is None else min_clearance_from_centerline(C1, lumen_C, lumen_R)

        rows.append({
            "k": k,
            "clearance_nom": clr0,
            "clearance_test": clr1,
            "du_norm": float(np.linalg.norm(du)),
            "dy_true_norm": float(np.linalg.norm(dy_true)),
            "err_bc": float(err_bc),
            "err_wrong": float(err_wrong),
            "rel_err_bc": float(rel_bc),
            "rel_err_wrong": float(rel_wrong),
        })

    return B_bc, B_wrong, rows
def unit(v, eps=1e-12):
    v = np.asarray(v, float).ravel()
    return v / (np.linalg.norm(v) + eps)


def tangent_from_centerline(C):
    C = np.asarray(C, float)
    if C.shape[0] == 3:
        t = C[:, -1] - C[:, -2]
    else:
        t = C[-1, :] - C[-2, :]
    return unit(t)


def min_clearance_from_centerline(C, lumen_C, lumen_R):
    """
    Approximate minimum clearance:
      clearance_i = min_j ||beam_point_i - lumen_center_j|| - lumen_R[j]
    Returns minimum over beam points.
    """
    C = np.asarray(C, float)
    if C.shape[0] == 3:
        P = C.T   # (N,3)
    else:
        P = C

    L = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()

    min_clear = np.inf
    for p in P:
        d = np.linalg.norm(L - p[None, :], axis=1) - R
        min_clear = min(min_clear, np.min(d))
    return float(min_clear)


def eval_forward6(fwd6_obj, p8, commit=False):
    """
    Deterministic 6D output:
      y = [tip_xyz(3), tip_tangent(3)]
    Also returns centerline and min clearance.
    """
    fwd6_obj.start_step()
    y = np.asarray(fwd6_obj(p8, commit=commit), float).reshape(6,)
    C = None if fwd6_obj.last_p_centerline is None else np.asarray(fwd6_obj.last_p_centerline, float).copy()
    return y, C


def fd_jacobian(fwd6_obj, p8_nom, step_vec):
    """
    Central finite-difference Jacobian of y wrt p8.
    p8 = [x,y,z,qw,qx,qy,qz,L]
    step_vec gives perturbation size per input dimension.
    """
    p8_nom = np.asarray(p8_nom, float).copy()
    step_vec = np.asarray(step_vec, float).reshape(-1)
    ny = 6
    nu = p8_nom.size
    J = np.zeros((ny, nu), float)

    y0, C0 = eval_forward6(fwd6_obj, p8_nom, commit=False)

    for i in range(nu):
        h = step_vec[i]
        dp = np.zeros_like(p8_nom)
        dp[i] = h

        p_plus = p8_nom.copy()
        p_minus = p8_nom.copy()
        p_plus[i] += h
        p_minus[i] -= h

        # keep quaternion roughly normalized if perturbing quaternion entries
        if 3 <= i <= 6:
            p_plus[3:7] /= np.linalg.norm(p_plus[3:7]) + 1e-12
            p_minus[3:7] /= np.linalg.norm(p_minus[3:7]) + 1e-12

        y_plus, _ = eval_forward6(fwd6_obj, p_plus, commit=False)
        y_minus, _ = eval_forward6(fwd6_obj, p_minus, commit=False)

        J[:, i] = (y_plus - y_minus) / (2.0 * h)

    return J, y0, C0


def sample_perturbations(step_vec, n_dirs=20, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    U = []
    for _ in range(n_dirs):
        d = rng.standard_normal(len(step_vec))
        d /= np.linalg.norm(d) + 1e-12
        du = scale * step_vec * d
        U.append(du)
    return U


def linear_prediction_errors(fwd6_truth, p8_nom, J_bc, J_wrong, du_list, lumen_C, lumen_R):
    """
    Compare true nonlinear delta y against linear predictions.
    """
    y0, C0 = eval_forward6(fwd6_truth, p8_nom, commit=False)
    clr0 = None if C0 is None else min_clearance_from_centerline(C0, lumen_C, lumen_R)

    rows = []
    for k, du in enumerate(du_list):
        p_test = p8_nom + du

        # renormalize quaternion
        p_test[3:7] /= np.linalg.norm(p_test[3:7]) + 1e-12

        y_true, C_true = eval_forward6(fwd6_truth, p_test, commit=False)
        dy_true = y_true - y0

        dy_bc = J_bc @ du
        dy_wrong = J_wrong @ du

        err_bc = np.linalg.norm(dy_true - dy_bc)
        err_wrong = np.linalg.norm(dy_true - dy_wrong)

        denom = np.linalg.norm(dy_true) + 1e-12
        rel_bc = err_bc / denom
        rel_wrong = err_wrong / denom

        clr_true = None if C_true is None else min_clearance_from_centerline(C_true, lumen_C, lumen_R)

        rows.append({
            "k": k,
            "clearance_nom": clr0,
            "clearance_test": clr_true,
            "dy_true_norm": np.linalg.norm(dy_true),
            "err_bc": err_bc,
            "err_wrong": err_wrong,
            "rel_err_bc": rel_bc,
            "rel_err_wrong": rel_wrong,
        })
    return rows
beam_params = default_beam_params()
mag_params = default_magnet_params()

L_min_energy = 0.04
mag_len = beam_params.length_of_mag
m_body = np.array([mag_params.mag_epm, 0.0, 0.0])

pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1,
    -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)

start_point = np.array([
    0.7281328220229531, -0.7555298925316631, -0.09,
    -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)

# rotate start pose 180 deg about local Z
t = start_point[:3]
rvec = start_point[3:]
R0 = Rot.from_rotvec(rvec)
R_spin_localZ = Rot.from_rotvec([0, 0, np.deg2rad(-35)])
R_new = R0 * R_spin_localZ
start_point = np.hstack([t, R_new.as_rotvec()])

# pivot pose
T_ur_pivot = ur_pose6_to_T(pivot_point)
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

# source magnet pose
T_ur_mag = ur_pose6_to_T(start_point)
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

# initial lumen tangent
Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
t0 = Rbase @ np.array([-1.0, 0.0, 0.0])
L_ins = L_min_energy
r_src = start_point[:3]
rvec_src = start_point[3:6]

q_xyzw = Rot.from_rotvec(rvec_src).as_quat()
q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
p8 = np.hstack([r_src, q_wxyz, L_ins])
s_straight = 0.03
lumen_C = make_lumen_centerline_turning(
    p_start=p0_ur,
    t0=t0,
    length=0.06 + s_straight,
    n_pts=130,
    bend_axis=np.array([0.0, 0.0, 1.0]),
    bend_angle=np.deg2rad(90.0),
    bend_start=0.0 + s_straight,
    bend_end=0.03 + s_straight,
)
lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
lumen_R = np.full(len(lumen_C), 0.003)

forward_model = EnergyMinForwardWithLumen(
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    u_star=np.zeros(3),
    m_body=m_body,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    N_nodes=35,
    maxiter=70,
    L0_init=0.01,
    dL_internal=0.002,
    L_tip_full=0.04,
    L_tip_min=0.01,
    use_lumen_jac=True
)
forward_model_wrong = EnergyMinForwardWithLumen(
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    u_star=np.zeros(3),
    m_body=m_body,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    N_nodes=35,
    maxiter=70,
    L0_init=0.01,
    dL_internal=0.002,
    L_tip_full=0.04,
    L_tip_min=0.01,
    use_lumen_jac=False
)
forward6d = DeterministicForward6D(forward_model)
forward6d_wrong = DeterministicForward6D(forward_model_wrong)
forward6d_truth = forward6d
dt = 0.01

dr = 5e-3
dtheta = np.deg2rad(50.0)
dL = 1e-3

eps_u = np.array([
    dr / dt, dr / dt, dr / dt,
    dtheta / dt, dtheta / dt, dtheta / dt,
    dL / dt
], dtype=float)
B_bc, B_wrong, rows = compare_controller_jacobians_at_point(
    p8_nom=p8,
    dt=dt,
    eps_u=eps_u,
    forward6d_truth=forward6d_truth,
    forward6d_bc=forward6d,
    forward6d_wrong=forward6d_wrong,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    du_list=sample_control_perturbations(eps_u, n_dirs=10, seed=1, scale=0.2),
)

print("Nominal minimum clearance [m]:", rows[0]["clearance_nom"] if rows else None)
print("||B_bc - B_wrong||_F =", np.linalg.norm(B_bc - B_wrong))

for r in rows[:5]:
    print(r)

mean_rel_bc = np.mean([r["rel_err_bc"] for r in rows])
mean_rel_wrong = np.mean([r["rel_err_wrong"] for r in rows])

print("Mean relative prediction error, BC Jacobian    :", mean_rel_bc)
print("Mean relative prediction error, wrong Jacobian:", mean_rel_wrong)
def plot_prediction_error_bars(rows):
    ks = [r["k"] for r in rows]
    e1 = [r["rel_err_bc"] for r in rows]
    e2 = [r["rel_err_wrong"] for r in rows]

    plt.figure(figsize=(10, 4))
    w = 0.4
    ks = np.asarray(ks)
    plt.bar(ks - w/2, e1, width=w, label="BC-aware Jacobian")
    plt.bar(ks + w/2, e2, width=w, label="No-BC Jacobian")
    plt.xlabel("Perturbation index")
    plt.ylabel("Relative linearisation error")
    plt.title("One-step prediction error at nominal operating point")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_prediction_error_bars(rows)



# def run_clearance_sweep(
#     base_p8,
#     sweep_idx,
#     sweep_values,
#     fwd_truth,
#     fwd_wrong,
#     step_vec,
#     lumen_C,
#     lumen_R,
#     n_dirs=20,
#     seed=0,
# ):
#     out = []

#     for val in sweep_values:
#         p8_nom = base_p8.copy()
#         p8_nom[sweep_idx] = val

#         # keep quaternion normalized
#         p8_nom[3:7] /= np.linalg.norm(p8_nom[3:7]) + 1e-12

#         try:
#             J_bc, y0, C0 = fd_jacobian(fwd_truth, p8_nom, step_vec)
#             J_wrong, _, _ = fd_jacobian(fwd_wrong, p8_nom, step_vec)

#             clr0 = min_clearance_from_centerline(C0, lumen_C, lumen_R)

#             du_list = sample_perturbations(step_vec, n_dirs=n_dirs, seed=seed, scale=1.0)
#             rows = linear_prediction_errors(
#                 fwd6_truth=fwd_truth,
#                 p8_nom=p8_nom,
#                 J_bc=J_bc,
#                 J_wrong=J_wrong,
#                 du_list=du_list,
#                 lumen_C=lumen_C,
#                 lumen_R=lumen_R,
#             )

#             mean_bc = np.mean([r["rel_err_bc"] for r in rows])
#             mean_wrong = np.mean([r["rel_err_wrong"] for r in rows])
#             max_bc = np.max([r["rel_err_bc"] for r in rows])
#             max_wrong = np.max([r["rel_err_wrong"] for r in rows])

#             out.append({
#                 "sweep_value": float(val),
#                 "clearance_nom": float(clr0),
#                 "mean_rel_err_bc": float(mean_bc),
#                 "mean_rel_err_wrong": float(mean_wrong),
#                 "max_rel_err_bc": float(max_bc),
#                 "max_rel_err_wrong": float(max_wrong),
#                 "J_diff_F": float(np.linalg.norm(J_bc - J_wrong)),
#             })

#             print(f"val={val:.5f}, clearance={clr0:.6f}, mean_bc={mean_bc:.4f}, mean_wrong={mean_wrong:.4f}")

#         except Exception as e:
#             print(f"Skipping sweep value {val}: {e}")

#     return out
# # Example: sweep source y-position around nominal
# base_p8 = p8.copy()
# y_nom = base_p8[1]
# sweep_values = np.linspace(y_nom - 0.008, y_nom + 0.008, 9)

# sweep_rows = run_clearance_sweep(
#     base_p8=base_p8,
#     sweep_idx=1,  # y-position
#     sweep_values=sweep_values,
#     fwd_truth=fwd_truth,
#     fwd_wrong=fwd_wrong_jac,
#     step_vec=step_vec,
#     lumen_C=lumen_C,
#     lumen_R=lumen_R,
#     n_dirs=15,
#     seed=2,
# )
# def plot_error_vs_clearance(sweep_rows):
#     clr = np.array([r["clearance_nom"] for r in sweep_rows], float)
#     e_bc = np.array([r["mean_rel_err_bc"] for r in sweep_rows], float)
#     e_wrong = np.array([r["mean_rel_err_wrong"] for r in sweep_rows], float)
#     jdiff = np.array([r["J_diff_F"] for r in sweep_rows], float)

#     order = np.argsort(clr)
#     clr = clr[order]
#     e_bc = e_bc[order]
#     e_wrong = e_wrong[order]
#     jdiff = jdiff[order]

#     plt.figure(figsize=(7, 5))
#     plt.plot(clr, e_bc, "-o", label="BC-aware Jacobian")
#     plt.plot(clr, e_wrong, "-o", label="No-BC Jacobian")
#     plt.xlabel("Nominal minimum clearance [m]")
#     plt.ylabel("Mean relative one-step prediction error")
#     plt.title("Linearisation accuracy vs clearance")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(7, 5))
#     plt.plot(clr, jdiff, "-o")
#     plt.xlabel("Nominal minimum clearance [m]")
#     plt.ylabel(r"$||J_{BC} - J_{noBC}||_F$")
#     plt.title("Jacobian difference vs clearance")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# plot_error_vs_clearance(sweep_rows)