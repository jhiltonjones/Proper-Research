import numpy as np
import csv
import matplotlib.pyplot as plt
import json
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_to_rotvec, quat_wxyz_normalize, quat_wxyz_mul,small_rot_quat_wxyz 
def debug_step_pose7_no_targets(
    k,
    p_now,
    x_now,
    info,
    mpc,
    i_ref=None,
    print_horizon=4,
    do_nl_rollout=True,
):
    """
    Reference-free debug print for 6D-output MPC:
        y = [x, y, z, tx, ty, tz]

    Expected keys in `info`:
        status, infeasible, u0, x_now, p_prev, x_prev,
        X_pred, U_seq, X_aff_last, Mc_last, X_nom_last,
        pred1_err_xy, pred1_err_xyz, tan1_err,
        mpc_debug (optional rich term/constraint breakdown)

    Prints:
      - current tip position/tangent
      - current pose in pose7 rotvec form
      - applied control
      - 1-step prediction mismatch
      - predicted linear horizon
      - nominal nonlinear horizon at SQP linearization point
      - internal affine consistency check
      - optional nonlinear open-loop rollout along planned U
      - objective / penalty / constraint debug breakdown from info["mpc_debug"]
    """

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _arr(x, default=None, dtype=float):
        if x is None:
            return default
        return np.asarray(x, dtype=dtype)

    def _scalar(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return default

    def _fmt_vec(v, prec=4, scale=1.0):
        v = np.asarray(v, float).ravel()
        body = ",".join([f"{scale*vi:+.{prec}f}" for vi in v])
        return f"[{body}]"

    def _fmt_stats(name, arr, scale=1.0, unit="", prec=4, indent="      "):
        a = np.asarray(arr, float).ravel()
        if a.size == 0 or not np.any(np.isfinite(a)):
            print(f"{indent}{name}: n/a")
            return
        af = a[np.isfinite(a)]
        print(
            f"{indent}{name}: "
            f"min={scale*np.min(af):.{prec}f}{unit}  "
            f"mean={scale*np.mean(af):.{prec}f}{unit}  "
            f"max={scale*np.max(af):.{prec}f}{unit}"
        )

    def _print_section(title):
        print(title)

    def _print_horizon_block(title, X, ph):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[0] == 0:
            print(f"   {title}: unavailable")
            return
        print(f"   {title}")
        for i in range(min(ph, X.shape[0])):
            p = X[i, :3]
            if X.shape[1] >= 6:
                t = X[i, 3:6]
                print(
                    f"   {i:02d}: "
                    f"pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]  "
                    f"tan=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]"
                )
            else:
                print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]")

    # ------------------------------------------------------------------
    # horizon sizes
    # ------------------------------------------------------------------
    Np = int(getattr(mpc, "Np", 0))
    ph = min(int(print_horizon), Np) if Np > 0 else int(print_horizon)

    # ------------------------------------------------------------------
    # current output
    # ------------------------------------------------------------------
    x_now6 = info.get("x_now", None)
    if x_now6 is None:
        x_now6 = np.asarray(x_now, float).ravel()
    else:
        x_now6 = np.asarray(x_now6, float).ravel()

    pos_now = x_now6[:3] if x_now6.size >= 3 else np.full(3, np.nan)
    tan_now = x_now6[3:6] if x_now6.size >= 6 else np.full(3, np.nan)

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------
    status = info.get("status", "?")
    infeas = int(info.get("infeasible", -1))
    pred_err_xy = _scalar(info.get("pred1_err_xy", np.nan))
    pred_err_xyz = _scalar(info.get("pred1_err_xyz", np.nan))
    tan1_err = _scalar(info.get("tan1_err", np.nan))

    print(
        f"k={k:04d} "
        f"tip=[{pos_now[0]:+.4f},{pos_now[1]:+.4f},{pos_now[2]:+.4f}] "
        f"tan=[{tan_now[0]:+.3f},{tan_now[1]:+.3f},{tan_now[2]:+.3f}] "
        f"status={status} infeas={infeas} "
        f"pred1_xy={pred_err_xy:.4e}"
    )
    if i_ref is not None:
        print(f"   path index i_ref={int(i_ref)}")

    # ------------------------------------------------------------------
    # pose
    # ------------------------------------------------------------------
    p7 = pose8_quat_to_pose7_rotvec(p_now)
    rvec = p7[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))
    print(
        f"   p_now: "
        f"x={p7[0]:+.3f} y={p7[1]:+.3f} z={p7[2]:+.3f}  "
        f"rotvec=[{rvec[0]:+.3f},{rvec[1]:+.3f},{rvec[2]:+.3f}] "
        f"|theta|={theta_deg:.1f}deg  L={p7[6]:.3f}"
    )

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------
    u0 = np.asarray(info.get("u0", np.zeros(getattr(mpc, "m", 7))), float).ravel()
    if u0.size >= 7:
        print(
            "   u0: "
            f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
            f"omega_body=[{u0[3]:+.4f},{u0[4]:+.4f},{u0[5]:+.4f}]  "
            f"dL={u0[6]:+.5f}"
        )
    else:
        print(f"   u0: {_fmt_vec(u0, prec=4)}")

    # ------------------------------------------------------------------
    # one-step errors
    # ------------------------------------------------------------------
    if np.isfinite(pred_err_xy) or np.isfinite(pred_err_xyz) or np.isfinite(tan1_err):
        print(
            f"   one-step errors: "
            f"xy={pred_err_xy:.4e}  xyz={pred_err_xyz:.4e}  tan={tan1_err:.4e}"
        )

    # ------------------------------------------------------------------
    # predicted horizon
    # ------------------------------------------------------------------
    X_pred = info.get("X_pred", None)
    U_seq = info.get("U_seq", None)

    if X_pred is None:
        print("   X_pred missing -> infeasible/failed QP or not stored.")
    else:
        X_pred = np.asarray(X_pred, float)
        if X_pred.ndim == 1:
            if Np > 0:
                X_pred = X_pred.reshape(Np, -1)
            else:
                X_pred = X_pred.reshape(1, -1)

        if np.all(np.isfinite(X_pred)):
            if Np <= 0:
                Np = X_pred.shape[0]
                ph = min(ph, Np)
            _print_horizon_block("Horizon: predicted (linear MPC)", X_pred, ph)
        else:
            print("   X_pred contains NaNs -> infeasible/failed QP.")

    # ------------------------------------------------------------------
    # affine consistency check
    # ------------------------------------------------------------------
    Mc_last = info.get("Mc_last", None)
    X_aff_last = info.get("X_aff_last", None)

    if (Mc_last is not None) and (X_aff_last is not None) and (U_seq is not None) and (X_pred is not None):
        Mc_last = np.asarray(Mc_last, float)
        X_aff_last = np.asarray(X_aff_last, float)
        U_seq = np.asarray(U_seq, float)

        n_out = int(getattr(mpc, "n", X_pred.shape[1]))
        U_vec = U_seq.reshape(-1, 1)
        Mc0 = Mc_last[0:n_out, :]
        x1_from_blocks = (X_aff_last[0:n_out, :] + Mc0 @ U_vec).reshape(n_out,)
        xpred0 = X_pred[0, :n_out]
        print("   |(X_aff0 + Mc0U) - X_pred[0]| =", float(np.linalg.norm(x1_from_blocks - xpred0)))

    # ------------------------------------------------------------------
    # nominal nonlinear at SQP linearization point
    # ------------------------------------------------------------------
    X_nom_last = info.get("X_nom_last", None)
    if X_nom_last is not None:
        X_nom_last = np.asarray(X_nom_last, float)
        if X_nom_last.ndim == 2 and X_nom_last.shape[0] >= 1:
            _print_horizon_block("Horizon: nominal nonlinear (SQP lin point eval)", X_nom_last, ph)

    # ------------------------------------------------------------------
    # nonlinear open-loop rollout
    # ------------------------------------------------------------------
    if do_nl_rollout and (U_seq is not None):
        U_seq = np.asarray(U_seq, float)
        p_start = info.get("p_prev", None)
        if (U_seq.ndim == 2) and np.all(np.isfinite(U_seq)) and (p_start is not None):
            P_nl, X_nl = rollout_open_loop_from_plan_commit(mpc, p_start, U_seq)
            _print_horizon_block("Horizon: NL rollout COMMIT (plant-like along planned U)", X_nl, ph)

            if X_pred is not None and X_pred.shape[0] >= 1:
                ph3 = min(ph, X_nl.shape[0], X_pred.shape[0])
                e_pos_m, e_tan_deg = horizon_pred_errors(X_nl, X_pred, ph=ph3)
                print("   Horizon errors (NL commit rollout vs linear pred):")
                for i in range(ph3):
                    print(f"   {i:02d}: pos={1e3*e_pos_m[i]:.3f}mm  tan={e_tan_deg[i]:.2f}deg")

    # ------------------------------------------------------------------
    # MPC debug breakdown
    # ------------------------------------------------------------------
    mdbg = info.get("mpc_debug", None)
    if mdbg is not None:
        weights = mdbg.get("weights", {})
        costs = mdbg.get("costs", {})
        penalties = mdbg.get("penalties", {})
        constraints = mdbg.get("constraints", {})

        _print_section("   --------------------------------------------------")
        _print_section("   MPC DEBUG: weights")
        for k_, v_ in weights.items():
            if np.isscalar(v_) or isinstance(v_, (float, int, np.floating, np.integer)):
                print(f"   {k_:>24s}: {float(v_): .6e}")
            else:
                arr = np.asarray(v_, float).ravel()
                if arr.size <= 8:
                    print(f"   {k_:>24s}: {arr}")
                else:
                    print(
                        f"   {k_:>24s}: "
                        f"min={np.min(arr):.6e} mean={np.mean(arr):.6e} max={np.max(arr):.6e}"
                    )

        _print_section("   MPC DEBUG: objective term costs")
        for k_, v_ in costs.items():
            print(f"   {k_:>24s}: {_scalar(v_): .6e}")

        # -----------------------
        # penalties
        # -----------------------
        if "advance" in penalties:
            adv = penalties["advance"]
            _print_section("   MPC DEBUG: advancement")
            print(f"   {'weight':>24s}: {float(adv.get('weight', np.nan)): .6e}")
            _fmt_stats(
                "directional_progress",
                adv.get("directional_progress", []),
                scale=1.0,
                unit="",
                prec=6,
                indent="   ",
            )
            print(f"   {'s_des_eff':>24s}: {float(adv.get('s_des_eff', np.nan)): .6e}")
        if "track" in penalties:
            trk = penalties["track"]
            _print_section("   MPC DEBUG: tracking penalty")
            print(f"   {'Q_diag':>24s}: {np.asarray(trk.get('Q_diag', []), float)}")
            print(f"   {'idx_ref':>24s}: {np.asarray(trk.get('idx_ref', []), int)}")
            print(f"   {'cost':>24s}: {float(trk.get('cost', np.nan)): .6e}")
            _fmt_stats("pos_err_norm", trk.get("pos_err_norm", []), scale=1e3, unit="mm", prec=3, indent="   ")
        if "tangent" in penalties:
            tan = penalties["tangent"]
            _print_section("   MPC DEBUG: tangent penalty")
            print(f"   {'theta_ref_deg':>24s}: {float(tan.get('theta_ref_deg', np.nan)): .6f}")
            _fmt_stats("theta_pred_deg", tan.get("theta_pred_deg", []), unit="deg", prec=3, indent="   ")
            _fmt_stats("clearance", tan.get("clearance_m", []), scale=1e3, unit="mm", prec=3, indent="   ")
            _fmt_stats("weight", tan.get("weights", []), unit="", prec=6, indent="   ")
            _fmt_stats("residual", tan.get("residual", []), unit="", prec=6, indent="   ")

        # -----------------------
        # constraints
        # -----------------------
        def _print_soft_constraint_block(name, block, residual_key=None, extra_keys=()):
            if block is None:
                return
            _print_section(f"   MPC DEBUG: {name}")

            if "weight" in block:
                print(f"   {'weight':>24s}: {float(block['weight']): .6e}")
            if "eps" in block:
                print(f"   {'eps':>24s}: {float(block['eps']): .6e}")

            if residual_key is not None and residual_key in block:
                arr = np.asarray(block[residual_key], float)
                arr_abs = np.abs(arr.reshape(-1))
                _fmt_stats(f"{residual_key}_abs", arr_abs, indent="   ", prec=6)

            if "residual_norm" in block:
                _fmt_stats("residual_norm", block["residual_norm"], indent="   ", prec=6)
            if "rho_lat" in block:
                _fmt_stats("rho_lat", block["rho_lat"], scale=1e3, unit="mm", indent="   ", prec=3)

            if "slack" in block:
                _fmt_stats("slack", block["slack"], indent="   ", prec=6)
            if "violation" in block:
                _fmt_stats("violation", block["violation"], indent="   ", prec=6)
            if "violation_n" in block:
                _fmt_stats("violation_n", block["violation_n"], indent="   ", prec=6)
            if "violation_b" in block:
                _fmt_stats("violation_b", block["violation_b"], indent="   ", prec=6)

            for key in extra_keys:
                if key in block and np.isscalar(block[key]):
                    print(f"   {key:>24s}: {float(block[key]): .6e}")

        _print_soft_constraint_block(
            "standoff soft constraint",
            constraints.get("standoff_soft", None),
            residual_key="residual",
            extra_keys=("residual_abs_max", "slack_max", "violation_max"),
        )
        _print_soft_constraint_block(
            "inline soft constraint",
            constraints.get("inline_soft", None),
            residual_key=None,
            extra_keys=("rho_lat_max", "slack_max", "violation_max"),
        )
        _print_soft_constraint_block(
            "dipole soft constraint",
            constraints.get("dipole_soft", None),
            residual_key=None,
            extra_keys=("residual_norm_max", "slack_max", "violation_max"),
        )

        theta_hard = constraints.get("theta_hard", None)
        if theta_hard is not None:
            _print_section("   MPC DEBUG: hard theta constraint")
            print(f"   {'theta_max_deg':>24s}: {float(theta_hard.get('theta_max_deg', np.nan)): .6f}")
            if "active_k" in theta_hard:
                print(f"   {'active_k':>24s}: {np.asarray(theta_hard['active_k'], int)}")
            _fmt_stats("clearance", theta_hard.get("clearance_m", []), scale=1e3, unit="mm", prec=3, indent="   ")
            _fmt_stats("theta_deg", theta_hard.get("theta_deg", []), unit="deg", prec=3, indent="   ")
            _fmt_stats("margin", theta_hard.get("margin", []), unit="", prec=6, indent="   ")

    print("--------------------------------------------------------------------")
def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)
def rollout_open_loop_from_plan_commit(mpc, p_start, U_seq):
    """
    Roll forward the nonlinear 'plant' using U_seq, committing solver state each step.
    Returns:
        P_nl: (Np, n_p)  parameter trajectory (p1..pNp)
        X_nl: (Np, n)    output trajectory
    """
    p = np.asarray(p_start, float).copy()
    U_seq = np.asarray(U_seq, float)

    # Freeze baseline ONCE for this open-loop simulation
    if hasattr(mpc.forward_tip_fn, "start_step"):
        mpc.forward_tip_fn.start_step()

    P_nl = []
    X_nl = []

    for i in range(U_seq.shape[0]):
        p = mpc._clamp_p(integrate_pose8_body(p, U_seq[i], mpc.dt))
        # IMPORTANT: commit=True so warm-start continues along the trajectory
        try:
            x = np.asarray(mpc.forward_tip_fn(p, commit=True), float).reshape(mpc.n,)
        except TypeError:
            # if forward doesn't accept commit kwarg
            x = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)

        P_nl.append(p.copy())
        X_nl.append(x.copy())

    return np.vstack(P_nl), np.vstack(X_nl)
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
def horizon_pred_errors(X_nl, X_pred, ph=None):
    """
    X_nl, X_pred: (Np,6) arrays (pos xyz + tangent tx ty tz)
    Returns:
      e_pos_m: (ph,) position L2 error in meters
      e_tan_deg: (ph,) tangent angle error in degrees
    """
    X_nl = np.asarray(X_nl, float)
    X_pred = np.asarray(X_pred, float)
    Np = min(X_nl.shape[0], X_pred.shape[0])
    ph = Np if ph is None else min(int(ph), Np)

    e_pos = np.zeros(ph, float)
    e_tan = np.zeros(ph, float)
    for i in range(ph):
        e_pos[i] = np.linalg.norm(X_nl[i, 0:3] - X_pred[i, 0:3])
        e_tan[i] = _angle_deg(X_nl[i, 3:6], X_pred[i, 3:6])
    return e_pos, e_tan

def _angle_deg(u, v, eps=1e-12):
    u = np.asarray(u, float).reshape(3,)
    v = np.asarray(v, float).reshape(3,)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un < eps or vn < eps:
        return np.nan
    c = float(np.clip(np.dot(u/un, v/vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
def save_step_artifacts(
    *,
    k: int,
    frames_dir,
    log_csv_path,
    u0, p_now, y_now,
    i_ref: int,
    info: dict,
    centerline_tip,
    lumen_C, lumen_R, p0_ur,
    tip_pos, tip_tan, fixed_limits,
    tip_from_centerline=None,
):
    u0 = np.asarray(u0, float).ravel()

    title = (f"k={k:04d} i_ref={i_ref} "
             f"u0=[{u0[0]:+.3f},{u0[1]:+.3f},{u0[2]:+.3f},"
             f"{u0[3]:+.2f},{u0[4]:+.2f},{u0[5]:+.2f},{u0[6]:+.3f}] "
             f"status={info.get('status','?')} infeas={info.get('infeasible',-1)}")

    # ---------- 1) save plot frame ----------
    # plot_energy_only_3d(
    #     centerline_tip,
    #     lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
    #     tip=tip_pos,
    #     tip_from_centerline=tip_from_centerline,
    #     p_mag=p_now,              # <-- FIXED
    #     mag_axis="x",
    #     mag_arrow_len=0.02,
    #     title=title,              # <-- USE IT
    #     show=False,
    #     fixed_limits=fixed_limits,
    #     zoom_out=1.5,
    # )

    # ---------- compute contact-angle metrics ----------
    Cc = np.asarray(lumen_C, float)
    tip_pos = np.asarray(tip_pos, float).reshape(3,)
    tip_tan = np.asarray(tip_tan, float).reshape(3,)

    i_seg, u_seg, c_cl, t_v = _closest_centerline_tangent(Cc, tip_pos)
   
    tip_vessel_angle_deg = _angle_deg(tip_tan, t_v)

    r = tip_pos - c_cl
    etan = float(np.dot(r, t_v))
    r_perp = r - etan * t_v
    rho = float(np.linalg.norm(r_perp))

    clearance = np.nan
    if lumen_R is not None:
        Rr = np.asarray(lumen_R, float).ravel()
        idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr.size - 1))
        clearance = float(Rr[idx_v] - rho)

    # ---------- save frame ----------
    fig = plt.gcf()
    fig_path = frames_dir / f"frame_{k:06d}.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2) append one row to CSV ----------
    p_now = np.asarray(p_now, float).ravel()
    y_now = np.asarray(y_now, float).ravel()

    row = [
        int(k), int(i_ref),
        *u0.tolist(),
        *p_now.tolist(),
        *y_now.tolist(),
        str(info.get("status","")),
        int(info.get("infeasible", -1)),
        float(info.get("pred1_err_xy", np.nan)),
        float(tip_vessel_angle_deg),
        int(i_seg),
        float(u_seg),
        float(rho),
        float(clearance),
    ]
    with open(log_csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _closest_centerline_tangent(Cc, x):
    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x)
    i_seg = int(np.clip(i_seg, 0, Cc.shape[0]-2))
    t = Cc[i_seg+1, :3] - Cc[i_seg, :3]     # segment tangent at closest point
    t /= (np.linalg.norm(t) + 1e-12)
    return i_seg, float(u_seg), c_closest.reshape(3,), t
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
def setup_output_dirs(out_root, lumen_C: np.ndarray, lumen_R: np.ndarray, mpc, u_max, p_min, p_max):
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_root / "log.csv"
    log_meta_path = out_root / "meta.json"

    np.save(out_root / "lumen_C.npy", lumen_C)
    np.save(out_root / "lumen_R.npy", lumen_R)

    meta = dict(
        dt=float(mpc.dt),
        Np=int(mpc.Np),
        u_max=u_max.tolist(),
        p_min=p_min.tolist(),
        p_max=p_max.tolist(),
    )
    with open(log_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(log_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "i_ref",
            "u0_vx", "u0_vy", "u0_vz", "u0_wx", "u0_wy", "u0_wz", "u0_dL",
            "p_x", "p_y", "p_z", "p_qw", "p_qx", "p_qy", "p_qz", "p_L",
            "y_x", "y_y", "y_z", "y_tx", "y_ty", "y_tz",
            "status", "infeasible", "pred1_err"
        ])

    return frames_dir, log_csv_path