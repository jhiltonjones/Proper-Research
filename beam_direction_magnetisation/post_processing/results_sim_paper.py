import csv
import numpy as np
def filter_series_by_k(series, k_min=None, k_max=None):
    k = np.asarray(series["k"], int)
    mask = np.ones_like(k, dtype=bool)

    if k_min is not None:
        mask &= (k >= k_min)
    if k_max is not None:
        mask &= (k <= k_max)

    out = {}
    for key, val in series.items():
        arr = np.asarray(val)
        if arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[key] = arr[mask]
        else:
            out[key] = val
    return out
def closest_point_polyline(C, x):
    """
    Return closest point on polyline C to point x.
    Outputs:
      i_seg: segment index (0..M-2)
      u:     segment parameter in [0,1]
      c:     closest point (3,)
      d2:    squared distance
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3,)
    V = C[1:] - C[:-1]                 # (M-1,3)
    W = x.reshape(1,3) - C[:-1]        # (M-1,3)
    VV = np.sum(V*V, axis=1) + 1e-15
    u = np.sum(W*V, axis=1) / VV
    u = np.clip(u, 0.0, 1.0)
    P = C[:-1] + u.reshape(-1,1)*V
    d2 = np.sum((P - x.reshape(1,3))**2, axis=1)
    i_seg = int(np.argmin(d2))
    return i_seg, float(u[i_seg]), P[i_seg], float(d2[i_seg])

def unit(v, eps=1e-12):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)
def tip_in_vessel_frame(x_tip, C, R, s_path, t_prev=None, n_prev=None):
    """
    Returns dict with:
      idx_seg, u, s, c (closest point), t,n,b, x_perp,y_perp,rho, clearance, R_here
    """
    i_seg, u, c, d2 = closest_point_polyline(C, x_tip)

    # arc-length at closest point
    s = float(s_path[i_seg] + u * (s_path[i_seg+1] - s_path[i_seg]))

    # tangent
    t = unit(C[i_seg+1] - C[i_seg])
    if t_prev is not None and np.dot(t, t_prev) < 0.0:
        t = -t

    # normal/binormal: propagate prev normal if given; otherwise build from fixed reference
    if n_prev is None:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n = unit(ref - np.dot(ref, t)*t)
    else:
        n = unit(n_prev - np.dot(n_prev, t)*t)
        if np.linalg.norm(n) < 1e-9:
            # fallback
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ref, t)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            n = unit(ref - np.dot(ref, t)*t)

    b = unit(np.cross(t, n))
    n = unit(np.cross(b, t))  # re-orthonormalize

    r = np.asarray(x_tip, float).reshape(3,) - np.asarray(c, float).reshape(3,)
    x_perp = float(np.dot(n, r))
    y_perp = float(np.dot(b, r))
    rho = float(np.hypot(x_perp, y_perp))

    # radius at this segment (linear interp if you want; simplest: take R[i_seg])
    R_here = float(R[i_seg]) if np.ndim(R) > 0 else float(R)
    clearance = float(R_here - rho)

    return dict(
        idx_seg=i_seg, u=u, s=s, c=c, t=t, n=n, b=b,
        x_perp=x_perp, y_perp=y_perp, rho=rho, clearance=clearance, R=R_here
    )
def arc_length_param(C):
    """C: (M,3) -> s: (M,) cumulative arc-length."""
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s
def load_log_csv(path):
    with open(path, "r") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if len(rows) == 0:
        raise ValueError(f"No rows found in {path}")

    def _f(rr, key, default="nan"):
        v = rr.get(key, default)
        if v is None or v == "":
            return float(default)
        return float(v)

    def _first_float(rr, keys, default=np.nan):
        for key in keys:
            if key in rr and rr[key] not in ("", None):
                try:
                    return float(rr[key])
                except ValueError:
                    pass
        return float(default)

    # extract arrays
    tip = np.array([
        [
            _first_float(rr, ["y_x", "tip_x"]),
            _first_float(rr, ["y_y", "tip_y"]),
            _first_float(rr, ["y_z", "tip_z"]),
        ]
        for rr in rows
    ], float)

    status = [rr.get("status", "") for rr in rows]
    infeas = np.array([int(float(rr.get("infeasible", -1))) for rr in rows], int)

    # Prediction error in metres.
    # Supports old and new CSV headers.
    pred1 = np.array([
        _first_float(
            rr,
            [
                "pred1_err_xy_m",
                "pred1_err_xyz_m",
                "pred1_err",
                "pred1_err_xy",
            ],
            default=np.nan,
        )
        for rr in rows
    ], float)

    # If only mm column exists, convert to metres
    if np.all(~np.isfinite(pred1)):
        pred1 = np.array([
            1e-3 * _first_float(
                rr,
                [
                    "pred1_err_xy_mm",
                    "pred1_err_xyz_mm",
                ],
                default=np.nan,
            )
            for rr in rows
        ], float)

    k = np.array([int(float(rr.get("k", i))) for i, rr in enumerate(rows)], int)

    # i_ref is present in your CSV, but this makes it robust.
    i_ref = np.array([
        int(float(rr.get("i_ref", rr.get("i_ref_mpc", -1))))
        for rr in rows
    ], int)

    jac_gain = np.array([
        _first_float(
            rr,
            [
                "jac_gain_actual_over_jac",
                "jac_gain",
                "gain_actual_over_jac",
            ],
            default=np.nan,
        )
        for rr in rows
    ], float)

    jac_angle = np.array([
        _first_float(
            rr,
            [
                "jac_angle_actual_deg",
                "jac_angle_deg",
                "angle_jac_actual_deg",
            ],
            default=np.nan,
        )
        for rr in rows
    ], float)

    jac_pred_mm = np.array([
        _first_float(
            rr,
            [
                "jac_pred_mm",
                "jac_norm_pred_mm",
                "norm_jac_mm",
            ],
            default=np.nan,
        )
        for rr in rows
    ], float)

    jac_actual_mm = np.array([
        _first_float(
            rr,
            [
                "jac_actual_mm",
                "jac_norm_actual_mm",
                "norm_actual_mm",
            ],
            default=np.nan,
        )
        for rr in rows
    ], float)
    return dict(
        tip=tip,
        status=status,
        infeas=infeas,
        pred1=pred1,      # metres
        k=k,
        i_ref=i_ref,

        jac_gain=jac_gain,
        jac_angle=jac_angle,
        jac_pred_mm=jac_pred_mm,
        jac_actual_mm=jac_actual_mm,

        rows=rows,
    )
def plot_prediction_error(series, dt=None, title_prefix="Prediction error", k_min=None, k_max=None):
    """
    Plots pred1 error (meters) from analyze_run(...)->series.
    Expects:
      series["pred1_err"] : (N,) in meters
      series["k"]         : (N,) optional
      series["i_ref"]     : (N,) optional
    """
    pred1 = np.asarray(series.get("pred1_err", []), float).ravel()
    if pred1.size == 0:
        print("[plot_prediction_error] No pred1_err found in series.")
        return

    k = np.asarray(series.get("k", np.arange(pred1.size)), int).ravel()
    i_ref = np.asarray(series.get("i_ref", np.full(pred1.size, -1, int)), int).ravel()

    # --- filter by k range ---
    mask = np.ones_like(k, dtype=bool)
    if k_min is not None:
        mask &= (k >= k_min)
    if k_max is not None:
        mask &= (k <= k_max)

    pred1 = pred1[mask]
    k = k[mask]
    i_ref = i_ref[mask]

    if pred1.size == 0:
        print("[plot_prediction_error] No samples left after k filtering.")
        return

    pred1_mm = 1e3 * pred1

    # --- Plot 1: vs step k ---
    plt.figure()
    plt.plot(k, pred1_mm, marker="o", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Predicition error [mm]")
    if dt is not None:
        plt.title(f"{title_prefix} vs step (dt={dt:.3f}s)")
    else:
        plt.title(f"{title_prefix} vs step")
    plt.grid(True, alpha=0.3)
    plt.xlim(k.min(), k.max())
    plt.show()

    # --- Plot 2: vs centerline index ---
    valid = (i_ref >= 0) & np.isfinite(pred1_mm)
    if np.any(valid):
        plt.figure()
        plt.plot(i_ref[valid], pred1_mm[valid], marker="o", linewidth=1)
        plt.xlabel("Centerline index")
        plt.ylabel("Prediction error [mm]")
        plt.title(f"{title_prefix} vs centerline index")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("[plot_prediction_error] i_ref not available (or all invalid), skipped index plot.")

    print(
        "[pred1_err] "
        f"RMS={np.sqrt(np.nanmean(pred1_mm**2)):.3f} mm, "
        f"mean={np.nanmean(pred1_mm):.3f} mm, "
        f"p95={np.nanpercentile(pred1_mm,95):.3f} mm, "
        f"max={np.nanmax(pred1_mm):.3f} mm"
    )
def analyze_run(log_csv_path, lumen_C, lumen_R, dt, contact_delta=0.0):
    data = load_log_csv(log_csv_path)
    tip = data["tip"]

    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float) if np.ndim(lumen_R) else float(lumen_R)
    s_path = arc_length_param(C)

    N = tip.shape[0]
    s = np.zeros(N, float)
    xperp = np.zeros(N, float)
    yperp = np.zeros(N, float)
    rho = np.zeros(N, float)
    clearance = np.zeros(N, float)
    idx = np.zeros(N, int)

    t_prev = None
    n_prev = None
    for k in range(N):
        out = tip_in_vessel_frame(tip[k], C, R, s_path, t_prev=t_prev, n_prev=n_prev)
        s[k] = out["s"]
        xperp[k] = out["x_perp"]
        yperp[k] = out["y_perp"]
        rho[k] = out["rho"]
        clearance[k] = out["clearance"]
        idx[k] = out["idx_seg"]
        t_prev = out["t"]
        n_prev = out["n"]

    # contact / penetration
    pen = np.maximum(0.0, -clearance)  # penetration in meters
    contact = clearance <= contact_delta

    # progress
    ds = np.diff(s, prepend=s[0])
    backtrack = ds < 0.0
    stall = ds < 1e-5  # 0.01 mm, tune

    summary = dict(
        N=int(N),
        T=float(N*dt),

        # clearance/contact
        clearance_min=float(np.min(clearance)),
        clearance_p05=float(np.percentile(clearance, 5)),
        clearance_mean=float(np.mean(clearance)),
        contact_frac=float(np.mean(contact)),
        pen_max_mm=float(1e3*np.max(pen)),
        pen_rms_mm=float(1e3*np.sqrt(np.mean(pen**2))),
        pen_int_mm_s=float(1e3*np.sum(pen)*dt),

        # centering (radial)
        rho_mean_mm=float(1e3*np.mean(rho)),
        rho_p95_mm=float(1e3*np.percentile(rho, 95)),

        # progress
        s_final=float(s[-1]),
        s_min=float(np.min(s)),
        s_backtrack_frac=float(np.mean(backtrack)),
        stall_frac=float(np.mean(stall)),

        # MPC health
        infeas_frac=float(np.mean(data["infeas"] > 0)),
        pred1_err_rms_mm=float(1e3*np.sqrt(np.nanmean(data["pred1"]**2))),
        
    )
    series = dict(
        tip=tip, s=s, ds=ds,
        xperp=xperp, yperp=yperp, rho=rho,
        clearance=clearance, pen=pen, contact=contact,
        idx=idx,

        pred1_err=data["pred1"],   # metres
        k=data["k"],
        i_ref=data["i_ref"],

        jac_gain=data["jac_gain"],
        jac_angle=data["jac_angle"],
        jac_pred_mm=data["jac_pred_mm"],
        jac_actual_mm=data["jac_actual_mm"],

        status=np.array(data["status"], dtype=object),
        infeas=data["infeas"],
    )
    return summary, series
def plot_jacobian_gain_compare_many(
    runs,
    k_min=None,
    k_max=None,
    title="Jacobian gain comparison",
    plot_vs_i_ref=True,
):
    """
    Plots actual/Jacobian-predicted movement gain.

    gain = ||actual tip displacement|| / ||Jacobian-predicted displacement||

    Ideal value is 1.
    Values < 1 mean the Jacobian over-predicts movement.
    Values > 1 mean the Jacobian under-predicts movement.
    """

    plt.figure(figsize=(11, 5))

    stats = []

    for label, series in runs:
        s = filter_series_by_k(series, k_min=k_min, k_max=k_max)

        gain = np.asarray(s.get("jac_gain", []), float).ravel()
        if gain.size == 0:
            print(f"[plot_jacobian_gain_compare_many] {label} has no jac_gain data.")
            continue

        k = np.asarray(s.get("k", np.arange(gain.size)), int).ravel()
        valid = np.isfinite(gain)

        if not np.any(valid):
            print(f"[plot_jacobian_gain_compare_many] {label} has no finite jac_gain samples.")
            continue

        plt.plot(
            k[valid],
            gain[valid],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )

        stats.append((label, gain[valid]))

    plt.axhline(1.0, linestyle="--", linewidth=1, label="ideal gain = 1")
    plt.xlabel("Step")
    plt.ylabel(r"Jacobian gain $\|\Delta p_\mathrm{actual}\| / \|\Delta p_\mathrm{Jac}\|$")
    plt.title(title + " vs step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if plot_vs_i_ref:
        plt.figure(figsize=(11, 5))
        plotted_any = False

        for label, series in runs:
            s = filter_series_by_k(series, k_min=k_min, k_max=k_max)

            gain = np.asarray(s.get("jac_gain", []), float).ravel()
            if gain.size == 0:
                continue

            i_ref = np.asarray(s.get("i_ref", np.full(gain.size, -1, int)), int).ravel()

            valid = (i_ref >= 0) & np.isfinite(gain)

            if np.any(valid):
                plt.plot(
                    i_ref[valid],
                    gain[valid],
                    marker="o",
                    linewidth=1.5,
                    markersize=3,
                    label=label,
                )
                plotted_any = True

        if plotted_any:
            plt.axhline(1.0, linestyle="--", linewidth=1, label="ideal gain = 1")
            plt.xlabel("i_ref centreline index")
            plt.ylabel(r"Jacobian gain $\|\Delta p_\mathrm{actual}\| / \|\Delta p_\mathrm{Jac}\|$")
            plt.title(title + " vs centreline index")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
            print("[plot_jacobian_gain_compare_many] i_ref not available for any series.")

    for label, gain in stats:
        gain = gain[np.isfinite(gain)]

        if gain.size == 0:
            continue

        print(
            f"[{label} jac_gain] "
            f"mean={np.mean(gain):.3f}, "
            f"median={np.median(gain):.3f}, "
            f"p05={np.percentile(gain, 5):.3f}, "
            f"p95={np.percentile(gain, 95):.3f}, "
            f"mean_abs_err_from_1={np.mean(np.abs(gain - 1.0)):.3f}"
        )
import numpy as np
import matplotlib.pyplot as plt

def _make_ring_points(c, t, R, n=32):
    """Return ring points around centerline point c with normal plane orthonormal to tangent t."""
    t = t / (np.linalg.norm(t) + 1e-12)

    # pick a stable reference not parallel to t
    ref = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(ref, t)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], float)

    n1 = ref - np.dot(ref, t) * t
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(t, n1)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)

    ang = np.linspace(0, 2*np.pi, n, endpoint=True)
    ring = c.reshape(1,3) + R*(np.cos(ang).reshape(-1,1)*n1.reshape(1,3) +
                              np.sin(ang).reshape(-1,1)*n2.reshape(1,3))
    return ring

def plot_world_centerline_tube_and_many_tips(
    lumen_C,
    lumen_R,
    runs,
    ring_step=5,
    ring_n=24,
    title="World view comparison",
    show_start_end=True,
):
    """
    runs: list of tuples:
        [
            ("bc", series_bc["tip"]),
            ("no bc", series_nobc["tip"]),
            ("adv weight", series_adv_weight["tip"]),
            ("above bc", series_above_bc["tip"]),
        ]
    """

    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float) if np.ndim(lumen_R) else np.full(C.shape[0], float(lumen_R))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # centerline
    ax.plot(
        C[:, 0],
        C[:, 1],
        C[:, 2],
        linewidth=2.5,
        label="Centerline",
    )

    # tip paths
    for label, tip_xyz in runs:
        tip_xyz = np.asarray(tip_xyz, float)

        if tip_xyz.size == 0:
            print(f"[plot_world_centerline_tube_and_many_tips] {label} has no tip data.")
            continue

        ax.plot(
            tip_xyz[:, 0],
            tip_xyz[:, 1],
            tip_xyz[:, 2],
            linewidth=2,
            label=label,
        )

        if show_start_end:
            ax.scatter(
                tip_xyz[0, 0],
                tip_xyz[0, 1],
                tip_xyz[0, 2],
                s=35,
                marker="o",
            )
            ax.scatter(
                tip_xyz[-1, 0],
                tip_xyz[-1, 1],
                tip_xyz[-1, 2],
                s=55,
                marker="^",
            )

    # tube rings
    for i in range(0, C.shape[0] - 1, int(ring_step)):
        t = C[i + 1] - C[i]
        ring = _make_ring_points(C[i], t, R[i], n=int(ring_n))
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            ring[:, 2],
            linewidth=1,
            alpha=0.25,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
def plot_prediction_error_compare_many(
    runs,
    dt=None,
    title_prefix="Prediction error comparison",
    k_min=None,
    k_max=None,
):
    """
    runs: list of tuples:
        [
            ("bc", series_bc),
            ("no bc", series_nobc),
            ("adv weight", series_adv_weight),
            ("above bc", series_above_bc),
        ]
    """

    plt.figure(figsize=(11, 5))

    stats = []

    for label, series in runs:
        s = filter_series_by_k(series, k_min=k_min, k_max=k_max)

        pred1 = np.asarray(s.get("pred1_err", []), float).ravel()
        if pred1.size == 0:
            print(f"[plot_prediction_error_compare_many] {label} has no pred1_err data.")
            continue

        k = np.asarray(s.get("k", np.arange(pred1.size)), int).ravel()
        pred1_mm = 1e3 * pred1

        plt.plot(k, pred1_mm, marker="o", linewidth=1.5, markersize=3, label=label)

        stats.append((label, pred1_mm))

    plt.xlabel("Step")
    plt.ylabel("Predicition Error")

    if dt is not None:
        plt.title(f"{title_prefix} vs step (dt={dt:.3f}s)")
    else:
        plt.title(f"{title_prefix} vs step")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot vs centerline index ---
    plt.figure(figsize=(11, 5))
    plotted_any = False

    for label, series in runs:
        s = filter_series_by_k(series, k_min=k_min, k_max=k_max)

        pred1 = np.asarray(s.get("pred1_err", []), float).ravel()
        if pred1.size == 0:
            continue

        i_ref = np.asarray(s.get("i_ref", np.full(pred1.size, -1, int)), int).ravel()
        pred1_mm = 1e3 * pred1

        valid = (i_ref >= 0) & np.isfinite(pred1_mm)

        if np.any(valid):
            plt.plot(
                i_ref[valid],
                pred1_mm[valid],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=label,
            )
            plotted_any = True

    if plotted_any:
        plt.xlabel("Centreline index")
        plt.ylabel("Prediction Error [mm]")
        plt.title(f"{title_prefix} vs centerline index")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
        print("[plot_prediction_error_compare_many] i_ref not available for any series.")

    # --- Print summary statistics ---
    for label, pred1_mm in stats:
        print(
            f"[{label}] "
            f"RMS={np.sqrt(np.nanmean(pred1_mm**2)):.3f} mm, "
            f"mean={np.nanmean(pred1_mm):.3f} mm, "
            f"p95={np.nanpercentile(pred1_mm, 95):.3f} mm, "
            f"max={np.nanmax(pred1_mm):.3f} mm"
        )
def plot_world_centerline_tube_and_two_tips(
    lumen_C,
    lumen_R,
    tip_xyz_1,
    tip_xyz_2,
    label1="Run 1",
    label2="Run 2",
    ring_step=5,
    ring_n=24,
    title="World view comparison",
):
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float) if np.ndim(lumen_R) else np.full(C.shape[0], float(lumen_R))
    tip_xyz_1 = np.asarray(tip_xyz_1, float)
    tip_xyz_2 = np.asarray(tip_xyz_2, float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # centerline
    ax.plot(C[:, 0], C[:, 1], C[:, 2], linewidth=2, label="centerline")

    # two tip paths
    ax.plot(tip_xyz_1[:, 0], tip_xyz_1[:, 1], tip_xyz_1[:, 2], linewidth=2, label=label1)
    ax.plot(tip_xyz_2[:, 0], tip_xyz_2[:, 1], tip_xyz_2[:, 2], linewidth=2, label=label2)

    # mark starts and ends
    ax.scatter(tip_xyz_1[0, 0], tip_xyz_1[0, 1], tip_xyz_1[0, 2], s=40, marker="o", label=f"{label1} start")
    ax.scatter(tip_xyz_1[-1, 0], tip_xyz_1[-1, 1], tip_xyz_1[-1, 2], s=60, marker="^", label=f"{label1} end")

    ax.scatter(tip_xyz_2[0, 0], tip_xyz_2[0, 1], tip_xyz_2[0, 2], s=40, marker="o", label=f"{label2} start")
    ax.scatter(tip_xyz_2[-1, 0], tip_xyz_2[-1, 1], tip_xyz_2[-1, 2], s=60, marker="^", label=f"{label2} end")

    # tube rings
    for i in range(0, C.shape[0] - 1, int(ring_step)):
        t = C[i + 1] - C[i]
        ring = _make_ring_points(C[i], t, R[i], n=int(ring_n))
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=1, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()
    plt.show()
def plot_clearance_compare_many(
    runs,
    k_min=None,
    k_max=None,
    title="Clearance comparison",
):
    """
    runs: list of tuples:
        [
            ("bc", series_bc),
            ("no bc", series_nobc),
            ("adv weight", series_adv_weight),
            ("above bc", series_above_bc),
        ]
    """

    plt.figure(figsize=(11, 5))

    for label, series in runs:
        s = filter_series_by_k(series, k_min=k_min, k_max=k_max)

        if "clearance" not in s:
            print(f"[plot_clearance_compare_many] {label} has no clearance data.")
            continue

        k = np.asarray(s["k"], int)
        clearance_mm = 1e3 * np.asarray(s["clearance"], float)

        plt.plot(k, clearance_mm, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Wall Distance [mm]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_summary_table(summaries):
    """
    summaries: list of tuples:
        [
            ("CC", summary_bc),
            ("No CC", summary_nobc),
            ...
        ]
    """

    keys = [
        "N",
        "T",
        "clearance_min",
        "clearance_p05",
        "clearance_mean",
        "contact_frac",
        "pen_max_mm",
        "pen_rms_mm",
        "pen_int_mm_s",
        "rho_mean_mm",
        "rho_p95_mm",
        "s_final",
        "s_backtrack_frac",
        "stall_frac",
        "infeas_frac",
        "pred1_err_rms_mm",
    ]

    print("\nSummary comparison:")
    print("-" * 120)

    header = "metric".ljust(24)
    for label, _ in summaries:
        header += label.rjust(24)
    print(header)

    print("-" * 120)

    for key in keys:
        row = key.ljust(24)
        for _, summary in summaries:
            val = summary.get(key, np.nan)

            if isinstance(val, float):
                row += f"{val:24.6g}"
            else:
                row += f"{str(val):>24}"

        print(row)

    print("-" * 120)
if __name__ == "__main__":
    lumen_C = np.load(
        "/Users/jackhilton-jones/Proper-Research/mpc_3np1s_nobc_120/lumen_C.npy"
    )
    lumen_R = np.load(
        "/Users/jackhilton-jones/Proper-Research/mpc_3np1s_nobc_120/lumen_R.npy"
    )

    dt = 0.02

    summary_bc, series_bc = analyze_run(
        "/Users/jackhilton-jones/Proper-Research/mpc_3np1s_bc_120/log.csv",
        lumen_C,
        lumen_R,
        dt=dt,
    )

    summary_nobc, series_nobc = analyze_run(
        "/Users/jackhilton-jones/Proper-Research/mpc_3np1s_nobc_120/log.csv",
        lumen_C,
        lumen_R,
        dt=dt,
    )

    # summary_adv_weight, series_adv_weight = analyze_run(
    #     "/Users/jackhilton-jones/Proper-Research/mpc_1np1s_wadv_30/log.csv",
    #     lumen_C,
    #     lumen_R,
    #     dt=dt,
    # )

    # summary_above_bc, series_above_bc = analyze_run(
    #     "/Users/jackhilton-jones/Proper-Research/mpc_1np1s_above_60/log.csv",
    #     lumen_C,
    #     lumen_R,
    #     dt=dt,
    # )

    print("CC summary:")
    print(summary_bc)
    print()

    print("No CC summary:")
    print(summary_nobc)
    print()

    # print("Advancement weight summary:")
    # print(summary_adv_weight)
    # print()

    # print("Above magnet CC summary:")
    # print(summary_above_bc)
    # print()

    k_min = 5
    k_max = 400

    # Store all four runs in one list.
    runs = [
        ("CC", series_bc),
        ("No CC", series_nobc),
        # ("CC + advancement weight", series_adv_weight),
        # ("Above magnet + CC", series_above_bc),
    ]

    # Filtered versions for trajectory plot.
    trajectory_runs = [
        (label, filter_series_by_k(series, k_min=k_min, k_max=k_max)["tip"])
        for label, series in runs
    ]

    plot_world_centerline_tube_and_many_tips(
        lumen_C,
        lumen_R,
        trajectory_runs,
        ring_step=5,
        ring_n=24,
        title=f"Trajectory of a 30 degree curved vessel",
    )

    plot_prediction_error_compare_many(
        runs,
        dt=dt,
        title_prefix="MPC 1-step prediction error",
        k_min=k_min,
        k_max=k_max,
    )

    plot_clearance_compare_many(
        runs,
        k_min=k_min,
        k_max=k_max,
        title=f"Wall Distance comparison",
    )
    summary_runs = [
        ("CC", summary_bc),
        ("No CC", summary_nobc),
        # ("CC + adv weight", summary_adv_weight),
        # ("Above magnet + CC", summary_above_bc),
    ]
    plot_jacobian_gain_compare_many(
        runs,
        k_min=k_min,
        k_max=k_max,
        title="Jacobian movement gain comparison",
    )
    print_summary_table(summary_runs)
