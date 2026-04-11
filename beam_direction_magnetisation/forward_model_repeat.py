# ============================================================
# REPEATABILITY TESTS FOR COSSERAT FORWARD MODEL
# ============================================================
import time
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

def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = quat_wxyz_normalize(p8[3:7])
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

def integrate_pose8_body(p8, u7, dt):
    """
    p8 = [x,y,z, qw,qx,qy,qz, L]
    u7 = [vx,vy,vz, wx,wy,wz, dL]
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

def clear_forward_cache(fwd_model):
    """
    Best-effort cache reset so each solve is as cold as possible.
    """
    if hasattr(fwd_model, "_last"):
        try:
            for k in list(fwd_model._last.keys()):
                fwd_model._last[k] = None
        except Exception:
            pass

    # also clear common public mirrors if present
    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if hasattr(fwd_model, name):
            try:
                setattr(fwd_model, name, None)
            except Exception:
                pass

def run_forward_once(fwd_model, p8, cold=False):
    """
    Evaluate tip at pose8. Returns (tip, info, elapsed_ms).
    """
    if cold:
        clear_forward_cache(fwd_model)

    p7 = pose8_quat_to_pose7_rotvec(p8)

    t0 = time.perf_counter()
    tip = np.asarray(fwd_model(p7), float).reshape(3,)
    t1 = time.perf_counter()

    info = getattr(fwd_model, "last_info", None)
    elapsed_ms = (t1 - t0) * 1e3
    return tip, info, elapsed_ms

def summarize_repeatability(name, tips, times_ms, infos=None):
    tips = np.asarray(tips, float)
    mean_tip = np.mean(tips, axis=0)
    err = np.linalg.norm(tips - mean_tip[None, :], axis=1)

    print("\n" + "=" * 80)
    print(f"REPEATABILITY SUMMARY: {name}")
    print("=" * 80)
    print("mean tip                :", mean_tip)
    print("std tip xyz             :", np.std(tips, axis=0))
    print("max deviation from mean :", np.max(err))
    print("mean deviation from mean:", np.mean(err))
    print("mean solve time [ms]    :", np.mean(times_ms))
    print("min  solve time [ms]    :", np.min(times_ms))
    print("max  solve time [ms]    :", np.max(times_ms))

    if infos is not None and len(infos) > 0:
        success_flags = []
        energies = []
        messages = {}
        for info in infos:
            if info is None:
                continue
            success_flags.append(bool(info.get("success", True)))
            if "W" in info:
                energies.append(float(info["W"]))
            msg = str(info.get("message", ""))
            messages[msg] = messages.get(msg, 0) + 1

        if len(success_flags) > 0:
            print("success count           :", np.sum(success_flags), "/", len(success_flags))
        if len(energies) > 0:
            print("energy mean             :", np.mean(energies))
            print("energy std              :", np.std(energies))
            print("energy min/max          :", np.min(energies), np.max(energies))
        if len(messages) > 0:
            print("messages:")
            for k, v in messages.items():
                print(f"  {v:3d} x {k}")

def repeatability_test_same_input(fwd_model, p8_nom, n_repeat=10, cold=False, label=""):
    tips = []
    infos = []
    times_ms = []

    for k in range(n_repeat):
        tip, info, t_ms = run_forward_once(fwd_model, p8_nom, cold=cold)
        tips.append(tip)
        infos.append(info)
        times_ms.append(t_ms)

        msg = "" if info is None else info.get("message", "")
        succ = True if info is None else info.get("success", True)

        print(
            f"[{label}] run {k:02d} | "
            f"tip={tip} | "
            f"time={t_ms:.2f} ms | "
            f"success={succ} | "
            f"msg={msg}"
        )

    summarize_repeatability(label, tips, times_ms, infos)
    return {
        "tips": np.asarray(tips),
        "times_ms": np.asarray(times_ms),
        "infos": infos,
    }

def repeatability_test_perturbed(
    fwd_model,
    p8_nom,
    *,
    dt,
    channel,
    delta,
    n_repeat=8,
    cold=True,
    channel_name=None,
):
    """
    Test repeatability at p8+delta and p8-delta for one channel.
    For channel < 6, delta is in physical state units over one step:
        x/y/z in m, rotations in rad
    For channel == 6, delta is in L units [m].
    """
    if channel_name is None:
        names = ["vx", "vy", "vz", "wx", "wy", "wz", "L"]
        channel_name = names[channel]

    if channel < 6:
        du = np.zeros(7, float)
        du[channel] = delta / dt
        p_plus = integrate_pose8_body(p8_nom, +du, dt)
        p_minus = integrate_pose8_body(p8_nom, -du, dt)
    else:
        p_plus = np.asarray(p8_nom, float).copy()
        p_minus = np.asarray(p8_nom, float).copy()
        p_plus[7] += delta
        p_minus[7] -= delta

    print("\n" + "=" * 80)
    print(f"PERTURBED REPEATABILITY TEST: {channel_name}, delta={delta:.3e}")
    print("=" * 80)

    plus = repeatability_test_same_input(
        fwd_model, p_plus, n_repeat=n_repeat, cold=cold,
        label=f"{channel_name} +delta"
    )
    minus = repeatability_test_same_input(
        fwd_model, p_minus, n_repeat=n_repeat, cold=cold,
        label=f"{channel_name} -delta"
    )

    mean_plus = np.mean(plus["tips"], axis=0)
    mean_minus = np.mean(minus["tips"], axis=0)
    secant = (mean_plus - mean_minus) / (2.0 * delta)

    print("\n[PERTURBATION SECANT ESTIMATE]")
    print("mean tip (+)            :", mean_plus)
    print("mean tip (-)            :", mean_minus)
    print("secant dtip/ddelta      :", secant)
    print("secant norm             :", np.linalg.norm(secant))

    return {
        "plus": plus,
        "minus": minus,
        "secant": secant,
    }

def plot_repeatability_cloud(results, title="Repeatability cloud"):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label, data in results.items():
        tips = np.asarray(data["tips"])
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], label=label, s=40)

        mean_tip = np.mean(tips, axis=0)
        ax.scatter([mean_tip[0]], [mean_tip[1]], [mean_tip[2]], marker="x", s=100)

    ax.set_title(title)
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip y")
    ax.set_zlabel("tip z")
    ax.legend()
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()
    # global setup
    # ------------------------------------------------------------
beam_params = default_beam_params()
mag_params = default_magnet_params()

L_cmd = 0.065
N_nodes = 10
USE_LUMEN = False

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

    s_straight = 0.03
    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.06 + s_straight,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(0.0),
        bend_start=0.0 + s_straight,
        bend_end=0.03 + s_straight,
    )
    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
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
# nominal p8 state
# ------------------------------------------------------------
p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

# ------------------------------------------------------------
# 1) repeatability at identical input
# ------------------------------------------------------------
warm_nom = repeatability_test_same_input(
    fwd_cos, p8_nom, n_repeat=10, cold=False, label="nominal warm"
)

cold_nom = repeatability_test_same_input(
    fwd_cos, p8_nom, n_repeat=10, cold=True, label="nominal cold"
)

plot_repeatability_cloud(
    {
        "warm nominal": warm_nom,
        "cold nominal": cold_nom,
    },
    title="Cosserat repeatability at nominal input"
)

# ------------------------------------------------------------
# 2) repeatability under perturbations
# Choose deltas from your FD sweep plateau regions
# ------------------------------------------------------------
dt_rep = 1.0
delta_vy = 5e-3
delta_wz = 3e-1
delta_L  = 1e-3

rep_vy = repeatability_test_perturbed(
    fwd_cos,
    p8_nom,
    dt=dt_rep,
    channel=1,          # vy
    delta=delta_vy,
    n_repeat=8,
    cold=True,
    channel_name="vy",
)

rep_wz = repeatability_test_perturbed(
    fwd_cos,
    p8_nom,
    dt=dt_rep,
    channel=5,          # wz
    delta=delta_wz,
    n_repeat=8,
    cold=True,
    channel_name="wz",
)

rep_L = repeatability_test_perturbed(
    fwd_cos,
    p8_nom,
    dt=dt_rep,
    channel=6,          # L
    delta=delta_L,
    n_repeat=8,
    cold=True,
    channel_name="L",
)
warm_nom = repeatability_test_same_input(
fwd_cos, p8_nom, n_repeat=10, cold=False, label="nominal warm"
)

cold_nom = repeatability_test_same_input(
fwd_cos, p8_nom, n_repeat=10, cold=True, label="nominal cold"
)