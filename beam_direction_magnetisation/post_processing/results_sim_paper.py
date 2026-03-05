import csv
import numpy as np

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

    def _f(rr, key, default="nan"):
        v = rr.get(key, default)
        if v is None or v == "":
            return float(default)
        return float(v)

    # extract arrays
    tip = np.array([[float(rr["y_x"]), float(rr["y_y"]), float(rr["y_z"])] for rr in rows], float)
    status = [rr.get("status", "") for rr in rows]
    infeas = np.array([int(rr.get("infeasible", -1)) for rr in rows], int)

    # support both old/new CSV names
    pred1 = np.array([
        _f(rr, "pred1_err", rr.get("pred1_err_xy", "nan"))
        for rr in rows
    ], float)

    # (optional) keep k and i_ref too, useful for plotting/debug
    k = np.array([int(rr.get("k", i)) for i, rr in enumerate(rows)], int)
    i_ref = np.array([int(rr.get("i_ref", -1)) for rr in rows], int)

    return dict(
        tip=tip,
        status=status,
        infeas=infeas,
        pred1=pred1,      # meters
        k=k,
        i_ref=i_ref,
        rows=rows
    )
def plot_prediction_error(series, dt=None, title_prefix="Prediction error"):
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

    pred1_mm = 1e3 * pred1

    # --- Plot 1: vs step k ---
    plt.figure()
    plt.plot(k, pred1_mm, marker="o", linewidth=1)
    plt.xlabel("k (step)")
    plt.ylabel("pred1_err [mm]")
    if dt is not None:
        t = np.asarray(k, float) * float(dt)
        # lightweight annotation in title only
        plt.title(f"{title_prefix} vs step (dt={dt:.3f}s)")
    else:
        plt.title(f"{title_prefix} vs step")
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Plot 2: vs centerline index ---
    valid = (i_ref >= 0) & np.isfinite(pred1_mm)
    if np.any(valid):
        plt.figure()
        plt.plot(i_ref[valid], pred1_mm[valid], marker="o", linewidth=1)
        plt.xlabel("i_ref (centerline index)")
        plt.ylabel("pred1_err [mm]")
        plt.title(f"{title_prefix} vs centerline index")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("[plot_prediction_error] i_ref not available (or all invalid), skipped index plot.")

    # Optional quick stats print
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

        # add these from CSV
        pred1_err=data["pred1"],   # meters
        k=data["k"],
        i_ref=data["i_ref"],
        status=np.array(data["status"], dtype=object),
        infeas=data["infeas"],
    )
    return summary, series
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

def plot_world_centerline_tube_and_tip(lumen_C, lumen_R, tip_xyz,
                                       ring_step=5, ring_n=24, title="World view"):
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float) if np.ndim(lumen_R) else np.full(C.shape[0], float(lumen_R))
    tip_xyz = np.asarray(tip_xyz, float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # centerline
    ax.plot(C[:,0], C[:,1], C[:,2], linewidth=2, label="centerline")

    # tip path
    ax.plot(tip_xyz[:,0], tip_xyz[:,1], tip_xyz[:,2], linewidth=2, label="tip path")

    # tube boundary (rings)
    for i in range(0, C.shape[0]-1, int(ring_step)):
        t = C[i+1] - C[i]
        ring = _make_ring_points(C[i], t, R[i], n=int(ring_n))
        ax.plot(ring[:,0], ring[:,1], ring[:,2], linewidth=1, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
if __name__ == "__main__":
    lumen_C = np.load("/Users/jackhilton-jones/Proper-Research/mpc_run_test_90_contact/lumen_C.npy")
    lumen_R = np.load("/Users/jackhilton-jones/Proper-Research/mpc_run_test_90_contact/lumen_R.npy")
    summary, series = analyze_run(
        "/Users/jackhilton-jones/Proper-Research/mpc_run_test_90_contact/log.csv",
        lumen_C, lumen_R, dt=0.01
    )
    print(summary)

    plot_world_centerline_tube_and_tip(
        lumen_C, lumen_R, series["tip"],
        ring_step=5, ring_n=24,
        title="Tip path vs vessel (world)"
    )

    # NEW
    plot_prediction_error(series, dt=0.01, title_prefix="MPC 1-step prediction error")