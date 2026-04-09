from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    make_lumen_centerline_turning,
    CosseratForwardModel,
    dipole_from_pose,
    u0_from_bvp,
    solve_quasistatic_insertion,
    lumen_violation_profile, magnetic_tip_wrench_about_interface
)

from beam_direction_magnetisation.post_processing.post_processing import (
    plot_energy_only_3d,
    make_lumen_centerline_double_turn,
)
from proper_research.robot.transformations import get_point

from proper_research.control.no_path_plan_mpc import DeterministicForward6D
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
from proper_research.control.mpc_boundary import resample_polyline
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, effective_lengths, WarmForwardP8TipTangent
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt
def eval_branch(fwd6, p8, label="", x_ref=None):
    # fwd6.start_step()
    y = np.asarray(fwd6(p8, commit=True), float).reshape(6,)
    pcl = np.asarray(fwd6.last_p_centerline, float)

    tip = y[:3]
    tan = y[3:6]

    print(f"\n===== {label} =====")
    print("p8 =", p8)
    print("y  =", y)
    print("tip =", tip)
    print("tan =", tan)
    print("pcl first =", pcl[:, 0])
    print("pcl last  =", pcl[:, -1])
    print("dist tip to first =", np.linalg.norm(tip - pcl[:, 0]))
    print("dist tip to last  =", np.linalg.norm(tip - pcl[:, -1]))

    if x_ref is not None:
        x_ref = np.asarray(x_ref, float).reshape(6,)
        tip_err = np.linalg.norm(tip - x_ref[:3])
        tan_dot = float(np.dot(tan, x_ref[3:6]))
        print("tip_err_vs_ref =", tip_err)
        print("tan_dot_vs_ref =", tan_dot)

    return {
        "p8": p8.copy(),
        "y": y.copy(),
        "tip": tip.copy(),
        "tan": tan.copy(),
        "pcl": pcl.copy(),
    }
def dipole_field_points(x, r_src, m_src, mu0_over_4pi=1e-7, r_min=1e-6):
    """
    x: (3,M) field points
    r_src: (3,)
    m_src: (3,)
    Returns B: (3,M)
    """
    x = np.asarray(x, float)
    r_src = np.asarray(r_src, float).reshape(3,1)
    m_src = np.asarray(m_src, float).reshape(3,1)

    r = x - r_src
    rn = np.linalg.norm(r, axis=0)
    rn = np.maximum(rn, r_min)

    rhat = r / rn[None, :]
    mdotr = np.sum(m_src * rhat, axis=0)

    B = mu0_over_4pi * (
        (3.0 * rhat * mdotr[None, :] - m_src) / (rn[None, :]**3)
    )
    return B
def debug_forward_solution(
    *,
    forward_model,
    p0_ur,
    q0_ur,
    r_src_ur,
    q_src_ur,
    m_body,
    L_ins,
):
    info = forward_model.last_info
    if info is None:
        print("[DEBUG] No last_info available.")
        return

    p = np.asarray(forward_model.last_p_centerline, float)
    s = np.asarray(info["s"], float)
    q = forward_model.last_hist[-1]["q"]

    L_model, wire_len, tip_len = effective_lengths(
        L_ins,
        L_tip_full=forward_model.L_tip_full,
        L_tip_min=forward_model.L_tip_min,
    )

    print("\n" + "=" * 90)
    print("FORWARD SOLUTION DEBUG")
    print("=" * 90)
    print(f"L_ins                : {L_ins:.6e}")
    print(f"L_model              : {L_model:.6e}")
    print(f"wire_len             : {wire_len:.6e}")
    print(f"tip_len              : {tip_len:.6e}")
    print(f"n_centerline_pts     : {p.shape[1] if p.ndim == 2 else 'unknown'}")
    print(f"s range              : [{s.min():.6e}, {s.max():.6e}]")
    print(f"tip position         : {p[:, -1]}")
    print(f"base position        : {p[:, 0]}")
    print(f"tip deflection       : {p[:, -1] - p[:, 0]}")
    print(f"tip deflection norm  : {np.linalg.norm(p[:, -1] - p[:, 0]):.6e}")

    # ------------------------------------------------------------------
    # segment strains from solver info
    # ------------------------------------------------------------------
    u_flat = np.asarray(info["u_flat_opt"], float)
    u_seg = u_flat.reshape(len(s) - 1, 3)

    s_mid = 0.5 * (s[:-1] + s[1:])
    kappa_twist = np.abs(u_seg[:, 0])
    kappa_b = np.linalg.norm(u_seg[:, 1:3], axis=1)

    wire_mask = s_mid < wire_len
    tip_mask = s_mid >= wire_len

    print("\n--- curvature summary ---")
    print(f"max |twist|          : {np.max(kappa_twist):.6e}")
    print(f"max |bend|           : {np.max(kappa_b):.6e}")
    print(f"mean |bend|          : {np.mean(kappa_b):.6e}")
    if np.any(wire_mask):
        print(f"wire mean |bend|     : {np.mean(kappa_b[wire_mask]):.6e}")
        print(f"wire max  |bend|     : {np.max(kappa_b[wire_mask]):.6e}")
    if np.any(tip_mask):
        print(f"tip  mean |bend|     : {np.mean(kappa_b[tip_mask]):.6e}")
        print(f"tip  max  |bend|     : {np.max(kappa_b[tip_mask]):.6e}")

    # ------------------------------------------------------------------
    # magnetic wrench at wire-tip interface
    # ------------------------------------------------------------------
    wrench = magnetic_tip_wrench_about_interface(
        p, q, s,
        r_src=r_src_ur,
        m_src=dipole_from_pose(q_src_ur, m_body),
        m_local_fun=make_m_local_fun_wire_tip(
            wire_len, len_tip=tip_len, mode="axial", alpha_end=0.0, eps=1e-3
        ),
        m_moment=0.0,
        wire_len=wire_len,
    )

    print("\n--- magnetic interface wrench ---")
    print("F_tip                :", wrench["F_tip"])
    print("Tau_tip              :", wrench["Tau_tip"])
    print("M_force              :", wrench["M_force"])
    print("M_total              :", wrench["M_total"])
    print(f"|F_tip|              : {np.linalg.norm(wrench['F_tip']):.6e}")
    print(f"|Tau_tip|            : {np.linalg.norm(wrench['Tau_tip']):.6e}")
    print(f"|M_total|            : {np.linalg.norm(wrench['M_total']):.6e}")

    # ------------------------------------------------------------------
    # lumen violation / contact
    # ------------------------------------------------------------------
    try:
        viol = lumen_violation_profile(p, forward_model.lumen_C, forward_model.lumen_R)
        viol = np.asarray(viol, float)
        print("\n--- lumen contact ---")
        print(f"max violation        : {np.max(viol):.6e}")
        print(f"mean violation       : {np.mean(viol):.6e}")
        print(f"num violating pts    : {int(np.sum(viol > 0.0))}")
    except Exception as e:
        print("\n--- lumen contact ---")
        print("Could not evaluate lumen violation profile:", e)

    # ------------------------------------------------------------------
    # field sampling along centerline
    # ------------------------------------------------------------------
    try:
        m_src_world = dipole_from_pose(q_src_ur, m_body)

        B = dipole_field_points(p, r_src_ur, m_src_world)   # (3, Npts)
        Bmag = np.linalg.norm(B, axis=0)

        # estimate tangent from centerline
        dp = np.diff(p, axis=1)
        ds = np.linalg.norm(dp, axis=0) + 1e-12
        t = dp / ds[None, :]

        # field-angle to local tangent
        Bmid = 0.5 * (B[:, :-1] + B[:, 1:])
        Bmid_mag = np.linalg.norm(Bmid, axis=0) + 1e-12
        cosang = np.sum(t * Bmid, axis=0) / Bmid_mag
        ang_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

        print("\n--- field along rod ---")
        print(f"max |B|              : {np.max(Bmag):.6e}")
        print(f"mean|B|              : {np.mean(Bmag):.6e}")
        print(f"tip  |B|             : {Bmag[-1]:.6e}")
        print(f"base |B|             : {Bmag[0]:.6e}")
        print(f"mean angle(t,B) [deg]: {np.mean(ang_deg):.6f}")
        print(f"min  angle(t,B) [deg]: {np.min(ang_deg):.6f}")
        print(f"max  angle(t,B) [deg]: {np.max(ang_deg):.6f}")
    except Exception as e:
        print("\n--- field along rod ---")
        print("Could not sample magnetic field along rod:", e)

    print("=" * 90)
def debug_geometry_split(forward_model, L_ins):
    info = forward_model.last_info
    if info is None:
        return

    s = np.asarray(info["s"], float)
    L_model, wire_len, tip_len = effective_lengths(
        L_ins,
        L_tip_full=forward_model.L_tip_full,
        L_tip_min=forward_model.L_tip_min,
    )
    s_mid = 0.5 * (s[:-1] + s[1:])

    wire_mask = s_mid < wire_len
    tip_mask = s_mid >= wire_len

    print("\n--- model geometry split ---")
    print(f"L_model              : {L_model:.6e}")
    print(f"wire_len             : {wire_len:.6e}")
    print(f"tip_len              : {tip_len:.6e}")
    print(f"wire segments        : {int(np.sum(wire_mask))}")
    print(f"tip segments         : {int(np.sum(tip_mask))}")
def plot_wire_tip_split(centerline, wire_len, s):
    p = np.asarray(centerline, float)
    s = np.asarray(s, float)

    wire_mask = s <= wire_len
    tip_mask = s >= wire_len

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p[0, wire_mask], p[1, wire_mask], p[2, wire_mask], linewidth=3, label="wire")
    ax.plot(p[0, tip_mask], p[1, tip_mask], p[2, tip_mask], linewidth=3, label="magnetic tip")

    ax.legend()
    ax.set_title("Centerline split into wire and tip")
    plt.show()
    
def plot_bending_profile_from_info(info, wire_len, title="Bending curvature profile"):
    s = np.asarray(info["s"], float)
    u_flat = np.asarray(info["u_flat_opt"], float)
    u_seg = u_flat.reshape(len(s)-1, 3)

    s_mid = 0.5 * (s[:-1] + s[1:])
    kappa_b = np.linalg.norm(u_seg[:, 1:3], axis=1)   # bend only
    kappa_twist = np.abs(u_seg[:, 0])

    plt.figure(figsize=(7,4))
    plt.plot(s_mid, kappa_b, label="bending curvature |kappa_b|")
    plt.plot(s_mid, kappa_twist, label="twist |u_t|", alpha=0.6)
    plt.axvline(wire_len, linestyle="--", label="wire-tip interface")
    plt.xlabel("s [m]")
    plt.ylabel("strain magnitude [1/m]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    wire_mask = s_mid < wire_len
    tip_mask = s_mid >= wire_len

    print("\n[CURVATURE SUMMARY]")
    if np.any(wire_mask):
        print("wire: mean bend =", np.mean(kappa_b[wire_mask]),
              "max bend =", np.max(kappa_b[wire_mask]))
    if np.any(tip_mask):
        print("tip : mean bend =", np.mean(kappa_b[tip_mask]),
              "max bend =", np.max(kappa_b[tip_mask]))

    if np.any(wire_mask) and np.any(tip_mask):
        ratio = (np.mean(kappa_b[tip_mask]) + 1e-12) / (np.mean(kappa_b[wire_mask]) + 1e-12)
        print("tip/wire mean bending ratio =", ratio)
def plot_wire_tip_split(centerline, wire_len, s):
    p = np.asarray(centerline, float)
    s = np.asarray(s, float)

    wire_mask = s <= wire_len
    tip_mask = s >= wire_len

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p[0, wire_mask], p[1, wire_mask], p[2, wire_mask], linewidth=3, label="wire")
    ax.plot(p[0, tip_mask], p[1, tip_mask], p[2, tip_mask], linewidth=3, label="magnetic tip")

    ax.legend()
    ax.set_title("Centerline split into wire and tip")
    plt.show()
def make_Kbt_inv_profile(EI_wire, EI_tip, GJ_wire, GJ_tip, bend_soft=1.0, tors_soft=1.0):
    def Kbt_inv_profile(s, len_wire):
        s = np.asarray(s, float)
        mask_tip = (s >= len_wire)

        EI_s = np.where(mask_tip, EI_tip, EI_wire)
        GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)

        Kinv = np.zeros((3, 3, s.size), float)
        Kinv[0, 0, :] = tors_soft / GJ_s
        Kinv[1, 1, :] = bend_soft / EI_s
        Kinv[2, 2, :] = bend_soft / EI_s
        return Kinv

    return Kbt_inv_profile
def rod_section_stiffness(r, E, nu):
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    EA = E * A
    EI = E * I
    GJ = G * J

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": EA,
        "EI": EI,
        "GJ": GJ,
    }
beam_params = default_beam_params()
mag_params = default_magnet_params()

L_cmd = 0.065
N_nodes = 10
USE_LUMEN = True

pivot_point = np.array([
    0.7681328220229531, -0.7112731669220016, -0.1,
    np.pi, 0.001, 0.001
], float)

base_point = np.array([
    pivot_point[0] - (L_cmd + 0.15),
    pivot_point[1],
    -0.1,
    np.pi, 0.001, 0.001
], float)

start_point = np.asarray(get_point(0, 20, base_point, pivot_point), dtype=float)
start_point[2] = -0.1
m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)
# pivot pose
T_ur_pivot = ur_pose6_to_T(pivot_point)
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

# source magnet pose
T_ur_mag = ur_pose6_to_T(start_point)
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

# initial lumen tangent
Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

s_straight = 0.03
lumen_C = make_lumen_centerline_turning(
    p_start=p0_ur,
    t0=t0,
    length=0.06 + s_straight,
    n_pts=130,
    bend_axis=np.array([0.0, 0.0, 1.0]),
    bend_angle=np.deg2rad(40.0),
    bend_start=0.0 + s_straight,
    bend_end=0.03 + s_straight,
)
lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
lumen_R = np.full(len(lumen_C), 0.004)
L_model, wire_len_model, tip_len_model = effective_lengths(L_cmd)
print(
    f"[INIT] L_ins={L_cmd:.3f} -> "
    f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
)
wire = rod_section_stiffness(
    r=200e-6,
    E=50e6,
    nu=0.4,
)
# tip = rod_section_stiffness(
#     r=2e-3,
#     E=3e6,
#     nu=0.49,
# )
tip = rod_section_stiffness(
    r=beam_params.r,
    E=beam_params.E,
    nu=0.49,
)
EA_wire = wire["EA"]
EI_wire = wire["EI"]
GJ_wire = wire["GJ"]

EA_tip = tip["EA"]
EI_tip = tip["EI"]
GJ_tip = tip["GJ"]
Kinv_fun = make_Kbt_inv_profile(
    EI_wire=EI_wire,
    EI_tip=EI_tip,
    GJ_wire=GJ_wire,
    GJ_tip=GJ_tip,
    bend_soft=1.0,
    tors_soft=1.0,
)
forward_model = EnergyMinForwardWithLumen(
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    Kinv_fun=Kinv_fun,
    u_star=np.zeros(3),
    m_body=m_body,
    lumen_C=np.asarray(lumen_C, float),
    lumen_R=np.asarray(lumen_R, float),
    N_nodes=8,
    maxiter=30,
    L0_init=0.01,
    dL_internal=0.002,
    use_lumen_jac=False,
    L_tip_full=tip_len_model,
    L_tip_min=0.01,
)

print("t0 =", t0)
print("z variation in lumen =", lumen_C[:,2].min(), lumen_C[:,2].max())
# forward_model_wrong = EnergyMinForwardWithLumen(
#     p0_ur=p0_ur,
#     q0_ur=q0_ur,
#     Kinv_fun=Kbt_inv_profile,
#     u_star=np.zeros(3),
#     m_body=m_body,
#     lumen_C=lumen_C,
#     lumen_R=lumen_R,
#     N_nodes=35,
#     maxiter=70,
#     L0_init=0.01,
#     dL_internal=0.002,
#     L_tip_full=0.04,
#     L_tip_min=0.01,
#     use_lumen_jac=False
# )
fwd6 = WarmForwardP8TipTangent(forward_model)
# fwd6_wrong = DeterministicForward6D(forward_model_wrong)
L_ins = L_cmd
r_src = start_point[:3]
rvec_src = start_point[3:6]

q_xyzw = Rot.from_rotvec(rvec_src).as_quat()
q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
p8 = np.hstack([r_src, q_wxyz, L_ins])

# fwd6.start_step()
y = fwd6(p8, commit=True)

tip_pos = y[:3]
tip_tangent = y[3:6]
centreline = fwd6.last_p_centerline
debug_geometry_split(forward_model, L_ins)

debug_forward_solution(
    forward_model=forward_model,
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    r_src_ur=r_src_ur,
    q_src_ur=q_src_ur,
    m_body=m_body,
    L_ins=L_ins,
)
print("tip position:", tip_pos)
print("tip tangent:", tip_tangent)
print("centreline shape:", None if centreline is None else centreline.shape)
print("tip position:", tip_pos)
print("tip tangent:", tip_tangent)
print("centreline shape:", None if centreline is None else centreline.shape)

p_mag = np.hstack([r_src_ur, q_src_ur, 0.0])
L_model, wire_len, tip_len = effective_lengths(
    L_ins,
    L_tip_full=forward_model.L_tip_full,
    L_tip_min=forward_model.L_tip_min,
)
info = forward_model.last_info
s = np.asarray(info["s"], float)
p = np.asarray(forward_model.last_p_centerline, float)

# if your info doesn't store q yet, you'll need to store it from the solve
# ideally add q to info or to the wrapper
q = forward_model.last_hist[-1]["q"]

wrench = magnetic_tip_wrench_about_interface(
    p, q, s,
    r_src=r_src_ur,
    m_src=dipole_from_pose(q_src_ur, m_body),
    m_local_fun=make_m_local_fun_wire_tip(
        wire_len, len_tip=tip_len, mode="axial", alpha_end=0.0, eps=1e-3
    ),
    m_moment=0.0,
    wire_len=wire_len,
)

print("F_tip    =", wrench["F_tip"])
print("Tau_tip  =", wrench["Tau_tip"])
print("M_force  =", wrench["M_force"])
print("M_total  =", wrench["M_total"])
print("|M_total| =", np.linalg.norm(wrench["M_total"]))
plot_bending_profile_from_info(
    forward_model.last_info,
    wire_len=wire_len,
    title="Wire vs tip bending"
)
plot_wire_tip_split(
    forward_model.last_p_centerline,
    wire_len=wire_len,
    s=forward_model.last_info["s"]
)

plot_energy_only_3d(
    centreline,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    p0=p0_ur,
    tip=tip_pos,
    tip_from_centerline=centreline[:, -1],
    p_mag=p_mag,
    mag_axis="x",
    mag_arrow_len=0.02,
    show=True,
)
p0_good = np.array([
    5.51141362e-01, -7.11278590e-01, -1.00000000e-01,
    5.39964856e-04,  9.99999749e-01,  3.22682731e-04,  3.26505816e-04,
    7.49543617e-02
], float)

p0_bad = np.array([
    5.47115615e-01, -7.31269877e-01, -1.00000000e-01,
    5.39263042e-04,  9.99999297e-01, -1.00666627e-03,  3.20264949e-04,
    7.69276531e-02
], float)

x_meas_like = np.array([
    0.69389495, -0.71370919, -0.1,
    -0.99693916, 0.0781813, 0.0
], float)

res_good = eval_branch(fwd6, p0_good, label="GOOD pose", x_ref=x_meas_like)
res_bad  = eval_branch(fwd6, p0_bad,  label="BAD pose",  x_ref=x_meas_like)