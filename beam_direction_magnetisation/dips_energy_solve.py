# ============================================================
# BRANCH MAP + MULTI-START ANALYSIS FOR COSSERAT FORWARD MODEL
# ============================================================
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from proper_research.parameters import default_magnet_params, default_beam_params
from proper_research.robot.transformations import get_point
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
    quat_wxyz_normalize,
    quat_wxyz_mul,
    quat_wxyz_to_rotvec,
    small_rot_quat_wxyz,
)
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
from proper_research.control.mpc_boundary import resample_polyline

from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen
from beam_direction_magnetisation.node_class import (
    set_axes_equal_3d,
    make_initial_nodes_straight,
    rod_section_stiffness,
    make_Kbt_inv_profile,
)

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# BASIC HELPERS
# ============================================================
def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = quat_wxyz_normalize(p8[3:7])
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)


def integrate_pose8_body(p8, u7, dt):
    """
    p8 = [x,y,z,qw,qx,qy,qz,L]
    u7 = [vx,vy,vz,wx,wy,wz,dL]
    """
    p8 = np.asarray(p8, float).copy()
    u7 = np.asarray(u7, float).ravel()

    p8[0:3] += dt * u7[0:3]
    p8[7] += dt * u7[6]

    q = quat_wxyz_normalize(p8[3:7])
    dphi = dt * u7[3:6]
    dq = small_rot_quat_wxyz(dphi)
    p8[3:7] = quat_wxyz_normalize(quat_wxyz_mul(q, dq))
    return p8


def clear_forward_cache(fwd_model):
    if hasattr(fwd_model, "_last"):
        try:
            for k in list(fwd_model._last.keys()):
                fwd_model._last[k] = None
        except Exception:
            pass

    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if hasattr(fwd_model, name):
            try:
                setattr(fwd_model, name, None)
            except Exception:
                pass


def run_forward_once(fwd_model, p8, cold=False):
    if cold:
        clear_forward_cache(fwd_model)

    p7 = pose8_quat_to_pose7_rotvec(p8)

    t0 = time.perf_counter()
    tip = np.asarray(fwd_model(p7), float).reshape(3,)
    t1 = time.perf_counter()

    info = getattr(fwd_model, "last_info", None)
    elapsed_ms = (t1 - t0) * 1e3
    return tip, info, elapsed_ms


def extract_contact_scalar(info):
    """
    Best-effort contact metric extraction.
    Returns np.nan if unavailable.
    """
    if info is None:
        return np.nan

    dbg = info.get("dbg", {})
    if "Wc" in dbg:
        return float(dbg["Wc"])

    con = dbg.get("contact", {})
    if "Wc" in con:
        return float(con["Wc"])

    return np.nan


def extract_energy(info):
    if info is None:
        return np.nan
    return float(info.get("W", np.nan))


def extract_success(info):
    if info is None:
        return False
    return bool(info.get("success", False))


def extract_message(info):
    if info is None:
        return "no info"
    return str(info.get("message", ""))


# ============================================================
# REPEATABILITY / MULTI-START CLUSTERING HELPERS
# ============================================================
def summarize_runs(label, tips, infos, times_ms):
    tips = np.asarray(tips, float)
    mean_tip = np.mean(tips, axis=0)
    dev = np.linalg.norm(tips - mean_tip[None, :], axis=1)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {label}")
    print("=" * 80)
    print("mean tip                :", mean_tip)
    print("std tip xyz             :", np.std(tips, axis=0))
    print("max deviation from mean :", np.max(dev))
    print("mean solve time [ms]    :", np.mean(times_ms))
    print("min/max solve time [ms] :", np.min(times_ms), np.max(times_ms))

    succ = np.array([extract_success(info) for info in infos], dtype=bool)
    W = np.array([extract_energy(info) for info in infos], float)

    print("success count           :", np.sum(succ), "/", len(succ))
    print("energy mean/std         :", np.nanmean(W), np.nanstd(W))
    print("energy min/max          :", np.nanmin(W), np.nanmax(W))


def cluster_by_tip_and_energy(solutions, tip_tol=1e-4, energy_tol=1e-6):
    """
    Simple greedy clustering.
    Each solution is a dict with keys:
      - tip
      - energy
      - info
      - time_ms
      - label
    """
    clusters = []

    for sol in solutions:
        assigned = False
        for c in clusters:
            tip_err = np.linalg.norm(sol["tip"] - c["rep"]["tip"])
            dW = abs(sol["energy"] - c["rep"]["energy"])

            if tip_err < tip_tol and dW < energy_tol:
                c["members"].append(sol)
                assigned = True
                break

        if not assigned:
            clusters.append({
                "rep": sol,
                "members": [sol],
            })

    return clusters


def print_cluster_summary(clusters):
    print("\n" + "=" * 80)
    print("MULTI-START CLUSTER SUMMARY")
    print("=" * 80)

    for k, c in enumerate(clusters):
        rep = c["rep"]
        members = c["members"]
        tips = np.array([m["tip"] for m in members], float)
        energies = np.array([m["energy"] for m in members], float)

        print(f"\nCluster {k}:")
        print("  count         :", len(members))
        print("  rep tip       :", rep["tip"])
        print("  mean tip      :", np.mean(tips, axis=0))
        print("  std tip       :", np.std(tips, axis=0))
        print("  mean energy   :", np.mean(energies))
        print("  min/max energy:", np.min(energies), np.max(energies))
        print("  labels        :", [m["label"] for m in members])


# ============================================================
# BRANCH MAP
# ============================================================
def run_branch_map(
    fwd_model,
    p8_nom,
    *,
    vy_vals,
    wz_vals,
    L_vals,
    dt=1.0,
):
    """
    For each grid point:
      - cold solve from scratch
      - warm solve using path continuation along loop order
    """
    ny, nw, nL = len(vy_vals), len(wz_vals), len(L_vals)

    tip_diff = np.full((ny, nw, nL), np.nan)
    energy_diff = np.full((ny, nw, nL), np.nan)
    time_cold = np.full((ny, nw, nL), np.nan)
    time_warm = np.full((ny, nw, nL), np.nan)
    same_success = np.zeros((ny, nw, nL), dtype=bool)
    same_msg = np.zeros((ny, nw, nL), dtype=bool)
    contact_diff = np.full((ny, nw, nL), np.nan)

    warm_initialized = False

    for i, vy in enumerate(vy_vals):
        for j, wz in enumerate(wz_vals):
            for k, L in enumerate(L_vals):
                u = np.zeros(7, float)
                u[1] = vy / dt
                u[5] = wz / dt

                p8_here = integrate_pose8_body(p8_nom, u, dt)
                p8_here[7] = L

                # cold
                tip_c, info_c, t_c = run_forward_once(fwd_model, p8_here, cold=True)

                # warm
                if not warm_initialized:
                    clear_forward_cache(fwd_model)
                    warm_initialized = True
                tip_w, info_w, t_w = run_forward_once(fwd_model, p8_here, cold=False)

                tip_diff[i, j, k] = np.linalg.norm(tip_w - tip_c)
                energy_diff[i, j, k] = abs(extract_energy(info_w) - extract_energy(info_c))
                time_cold[i, j, k] = t_c
                time_warm[i, j, k] = t_w
                same_success[i, j, k] = (extract_success(info_w) == extract_success(info_c))
                same_msg[i, j, k] = (extract_message(info_w) == extract_message(info_c))
                contact_diff[i, j, k] = abs(extract_contact_scalar(info_w) - extract_contact_scalar(info_c))

                print(
                    f"[grid i={i} j={j} k={k}] "
                    f"vy={vy:+.4e}, wz={wz:+.4e}, L={L:.4e} | "
                    f"||tip_w-c||={tip_diff[i,j,k]:.3e} | "
                    f"|dW|={energy_diff[i,j,k]:.3e} | "
                    f"tc={t_c:.1f} ms | tw={t_w:.1f} ms"
                )

    return {
        "vy_vals": np.asarray(vy_vals),
        "wz_vals": np.asarray(wz_vals),
        "L_vals": np.asarray(L_vals),
        "tip_diff": tip_diff,
        "energy_diff": energy_diff,
        "time_cold": time_cold,
        "time_warm": time_warm,
        "same_success": same_success,
        "same_msg": same_msg,
        "contact_diff": contact_diff,
    }


def plot_branch_heatmap(branch_data, L_index=0, field="tip_diff", title=None):
    vy_vals = branch_data["vy_vals"]
    wz_vals = branch_data["wz_vals"]
    Z = branch_data[field][:, :, L_index]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[wz_vals[0], wz_vals[-1], vy_vals[0], vy_vals[-1]],
    )
    plt.xlabel("wz perturbation")
    plt.ylabel("vy perturbation")
    plt.colorbar(im, label=field)
    if title is None:
        title = f"{field} at L={branch_data['L_vals'][L_index]:.4f}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# MULTI-START ANALYSIS AT A FIXED COMMAND
# ============================================================
def make_multistart_perturbations():
    """
    Returns a list of labels and state perturbations in pose8-space.
    """
    starts = []

    # straight/default cold
    starts.append(("cold_default", None))

    # pose perturbations
    starts.append(("vy_plus_small", ("vy", +5e-3)))
    starts.append(("vy_minus_small", ("vy", -5e-3)))
    starts.append(("wz_plus_small", ("wz", +1e-1)))
    starts.append(("wz_minus_small", ("wz", -1e-1)))

    # larger ones
    starts.append(("vy_plus_big", ("vy", +1e-2)))
    starts.append(("vy_minus_big", ("vy", -1e-2)))
    starts.append(("wz_plus_big", ("wz", +3e-1)))
    starts.append(("wz_minus_big", ("wz", -3e-1)))

    return starts


def build_pose8_from_named_perturbation(p8_nom, tag, dt=1.0):
    if tag is None:
        return np.asarray(p8_nom, float).copy()

    kind, val = tag
    p8 = np.asarray(p8_nom, float).copy()

    if kind == "vy":
        u = np.zeros(7, float)
        u[1] = val / dt
        p8 = integrate_pose8_body(p8, u, dt)
    elif kind == "wz":
        u = np.zeros(7, float)
        u[5] = val / dt
        p8 = integrate_pose8_body(p8, u, dt)
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")

    return p8


def run_multistart_map(fwd_model, p8_nom, dt=1.0):
    starts = make_multistart_perturbations()
    sols = []

    for label, tag in starts:
        p8_seed = build_pose8_from_named_perturbation(p8_nom, tag, dt=dt)

        # force cold solve from this seed command point
        tip, info, t_ms = run_forward_once(fwd_model, p8_seed, cold=True)

        sols.append({
            "label": label,
            "tip": tip.copy(),
            "energy": extract_energy(info),
            "time_ms": t_ms,
            "info": info,
        })

        print(
            f"[multistart] {label:>14s} | "
            f"tip={tip} | W={extract_energy(info):.6e} | "
            f"time={t_ms:.1f} ms | success={extract_success(info)}"
        )

    return sols


def plot_multistart_tips(solutions):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for sol in solutions:
        tip = sol["tip"]
        ax.scatter([tip[0]], [tip[1]], [tip[2]], s=60)
        ax.text(tip[0], tip[1], tip[2], sol["label"], fontsize=8)

    ax.set_title("Multi-start tip clusters")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip y")
    ax.set_zlabel("tip z")
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
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

    wire = rod_section_stiffness(r=200e-6, E=50e6, nu=0.4)
    tip = rod_section_stiffness(r=beam_params.r, E=beam_params.E, nu=0.49)

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=wire["EI"],
        EI_tip=tip["EI"],
        GJ_wire=wire["GJ"],
        GJ_tip=tip["GJ"],
        bend_soft=1.0,
        tors_soft=1.0,
    )

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
            bend_angle=np.deg2rad(40.0),
            bend_start=0.0 + s_straight,
            bend_end=0.03 + s_straight,
        )
        lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
        lumen_R = np.full(len(lumen_C), 0.004)

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

    p8_nom = np.hstack([r_src_ur, q_src_ur, L_cmd])

    # -------------------------------
    # 1) warm-vs-cold branch map
    # -------------------------------
    vy_vals = np.linspace(-1e-2, 1e-2, 7)
    wz_vals = np.linspace(-3e-1, 3e-1, 7)
    L_vals = np.array([L_cmd - 1e-3, L_cmd, L_cmd + 1e-3])

    branch_data = run_branch_map(
        fwd_cos,
        p8_nom,
        vy_vals=vy_vals,
        wz_vals=wz_vals,
        L_vals=L_vals,
        dt=1.0,
    )

    plot_branch_heatmap(branch_data, L_index=1, field="tip_diff", title="||tip_warm - tip_cold||")
    plot_branch_heatmap(branch_data, L_index=1, field="energy_diff", title="|W_warm - W_cold|")
    plot_branch_heatmap(branch_data, L_index=1, field="time_cold", title="cold solve time [ms]")
    plot_branch_heatmap(branch_data, L_index=1, field="time_warm", title="warm solve time [ms]")

    # -------------------------------
    # 2) multi-start map
    # -------------------------------
    sols = run_multistart_map(fwd_cos, p8_nom, dt=1.0)
    clusters = cluster_by_tip_and_energy(sols, tip_tol=1e-4, energy_tol=1e-6)
    print_cluster_summary(clusters)
    plot_multistart_tips(sols)