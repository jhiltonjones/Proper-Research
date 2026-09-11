#!/usr/bin/env python3
"""Repeated live comparison of the three offline controllers on one plan.

    naive_inverse_jacobian   resolved-rate, frozen beam Jacobian
    mpc_lti                  BeamOutputTrackingMPC, ONE Jacobian frozen at
                             reference sample 0
    mpc_ltv_offline          BeamOutputTrackingMPC, one Jacobian per reference
                             sample (relinearised along the planned trajectory)

mpc_lti and mpc_ltv_offline are given the SAME relinearised schedule (computed
once from planning_context's contact-free model bundle); mpc_lti just repeats
sample 0.  So LTI vs LTV isolates the value of time-varying linearisation, and
inverse-Jacobian vs LTI isolates resolved-rate vs receding-horizon QP.

Each (controller, rep) run is a FRESH SUBPROCESS (clean RTDE / camera / advancer
/ memory).  A run does:
  * robot -> reference joints
  * beam  -> --beam-len-mm  (advancer, camera-measured)
  * close_loop_path_follow.main() with dry_run=False, control_insertion=True

Everything lands under --out-dir:
  runs/<controller>_rep<k>_<stamp>/     (the full close_loop_path_follow output)
  ltv_schedule.npz                      (cached; reused on re-run / by --analyze-only)
  results.csv / results.json            (one row per run)
  summary.json                          (mean +/- std per controller)
  compare_paths.png / compare_bars.png

    # run it  (default: 3 reps x {inv,lti,ltv}, horizon 30)
    python -m proper_research.hardware.online.compare_controllers_live

    # more reps / a different plan / bigger horizon
    python -m proper_research.hardware.online.compare_controllers_live \
        --reps 5 --horizon 40 --plan-dir plans/<...>/time_parameterized_configuration_path

    # just rebuild the tables/plots from a finished run
    python -m proper_research.hardware.online.compare_controllers_live \
        --analyze-only close_loop_logs/compare_<stamp>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = "plans/triangle_8mm_2026-09-10/time_parameterized_configuration_path"
REF_JOINTS = np.array(
    [-0.85634357, -1.94584002, -1.76176286, -1.03319450, 1.56234264, -2.05640871]
)
KIND = {"inv": "naive_inverse_jacobian", "lti": "mpc_lti", "ltv": "mpc_ltv_offline"}
RUN_TIMEOUT_S = 240.0
# ur_rtde's RTDEControlInterface wedges (FK failures / segfaults) if a new
# control client connects within ~1 min of the previous one closing; give the
# controller's RTDE server time to release the control-script slot between runs.
INTER_RUN_SETTLE_S = 30.0
MAX_ATTEMPTS_PER_RUN = 3


def _joints_now():
    """Lightweight receive-only joint read (no control interface)."""
    try:
        import rtde_receive

        rr = rtde_receive.RTDEReceiveInterface("192.168.56.101")
        q = np.array(rr.getActualQ())
        try:
            rr.disconnect()
        except Exception:
            pass
        return q
    except Exception:
        return None


@contextlib.contextmanager
def _quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


# ==========================================================================
# hardware helpers (used inside a worker subprocess)
# ==========================================================================
def move_to_ref() -> bool:
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    r = URRTDERobot("192.168.56.101", frequency=125.0)
    try:
        for att in (1, 2, 3, 4):
            try:
                if r.get_safety_mode() not in (1, 2) or r.is_protective_stopped():
                    r.reconnect()
                    time.sleep(0.8)
                r.move_j(list(REF_JOINTS), speed=0.35, acceleration=0.35)
                q = np.array(r.get_joints())
                if np.max(np.abs(q - REF_JOINTS)) < 0.02:
                    print(f"[reset] at reference joints (dq={np.max(np.abs(q - REF_JOINTS)):.4f})")
                    return True
            except Exception as exc:  # noqa: BLE001
                print(f"[reset] att{att}: {exc!r}")
                try:
                    r.reconnect()
                except Exception:
                    pass
                time.sleep(1.0)
        return False
    finally:
        r.close()


def set_beam_length(target_m: float, tol_mm: float = 0.6, max_iter: int = 8):
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig

    scfg = StateStreamConfig(exposure=29.0)
    with _quiet():
        mapper = NewFrameTipMapper(scfg)
    T_R_B = mapper.T_R_B
    cam = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/cmp_len.png",
            roi_polygon_path=scfg.roi_polygon_path,
            manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: target_m, frame_processor=mapper,
    )
    cam.start()
    adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))
    adv.start()
    time.sleep(2.5)

    def meas(n: int = 25, tmo: float = 8.0):
        tips = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < tmo and len(tips) < n:
            est, _ = cam.latest(0.5)
            if est is not None:
                tips.append(np.asarray(est.tip_position_m))
            time.sleep(0.1)
        if not tips:
            return None
        t = T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0))
        return float(np.linalg.norm(t[:2]))

    try:
        for it in range(max_iter):
            L = meas()
            if L is None:
                print("[beam] no camera length")
                return None
            err_mm = 1e3 * (target_m - L)
            print(f"[beam] iter {it}: {1e3 * L:.1f} mm (target {1e3 * target_m:.0f}, err {err_mm:+.1f})")
            if abs(err_mm) <= tol_mm:
                return L
            adv.submit_distance(float(np.clip(err_mm, -4.0, 4.0)))
            t0 = time.monotonic()
            while time.monotonic() - t0 < 20:
                fb = adv.feedback()
                if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
                    break
                time.sleep(0.1)
            time.sleep(0.5)
        return meas()
    finally:
        try:
            adv.stop()
        except Exception:
            pass
        cam.stop()


# ==========================================================================
# relinearised Jacobian schedule (mpc_lti / mpc_ltv_offline share this)
# ==========================================================================
def build_ltv_schedule(plan_dir: str, cache: Path) -> np.ndarray:
    if cache.exists():
        sch = np.load(cache)["schedule"]
        print(f"[schedule] cache {cache} {sch.shape}")
        return sch
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.controllers.beam_jacobian_providers import from_model_bundle
    from proper_research.controllers.mpc_variants import precompute_schedule

    print("[schedule] building relinearised Jacobian schedule (one beam solve per sample)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    with _quiet():
        _, bundle, controller_pack, _ = build_planning_context()
        provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
        t0 = time.perf_counter()
        sch = precompute_schedule(reference=reference, jacobian_provider=provider)
    print(f"[schedule] {sch.shape} in {time.perf_counter() - t0:.0f}s -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, schedule=sch)
    return sch


# ==========================================================================
# worker: one run in its own process
# ==========================================================================
def _worker(kind_key: str, rep: int, out_dir: Path, sched_npz: Path,
            plan_dir: str, horizon: int, beam_len_mm: float,
            feedforward: bool = False, force_mpc_feedforward: bool = False,
            ff_trim_base_switch_at_hold: bool = False,
            mpc_ff_input_increment_weight: float | None = None,
            mpc_input_tracking_weight_override: float | None = None,
            mpc_input_increment_weight_override: float | None = None,
            mpc_solver_time_limit_s: float | None = None,
            mpc_directional_damping: float = 0.0,
            mpc_directional_damping_floor: float = 0.01) -> int:
    import proper_research.hardware.online.close_loop_path_follow as pf

    # Always move to reference first: besides resetting drift it "primes" the
    # RTDEControlInterface so pf.main()'s fresh control connection survives the
    # few-second beam-model build that happens right after it connects.
    if not move_to_ref():
        print("[worker] cannot reach reference joints")
        return 2
    time.sleep(3.0)  # let move_to_ref's control client fully close
    L = set_beam_length(beam_len_mm * 1e-3)
    if L is None or abs(L - beam_len_mm * 1e-3) > 1.5e-3:
        print(f"[worker] beam length not set (got {L})")
        return 3
    print(f"[worker] beam {1e3 * L:.1f} mm; running {KIND[kind_key]}")
    time.sleep(1.0)

    if kind_key in ("lti", "ltv"):
        pf._SCHEDULE_OVERRIDE = np.load(sched_npz)["schedule"]
        pf.CONFIG.mpc_prediction_horizon = int(horizon)
        pf.CONFIG.mpc_freeze_index = 0
    pf.CONFIG.dry_run = False
    pf.CONFIG.reference_source = "plan_dir"
    pf.CONFIG.plan_dir = plan_dir
    pf.CONFIG.control_hz = 10.0
    pf.CONFIG.controller_kind = KIND[kind_key]
    pf.CONFIG.control_insertion = True
    pf.CONFIG.advancer_dry_run = False
    pf.CONFIG.initial_insertion_m = beam_len_mm * 1e-3
    pf.CONFIG.exposure = 29.0
    pf.CONFIG.max_control_steps = 800
    pf.CONFIG.feedforward_joint_trajectory = bool(feedforward)
    pf.CONFIG.force_mpc_feedforward = bool(force_mpc_feedforward)
    pf.CONFIG.ff_trim_base_switch_at_hold = bool(ff_trim_base_switch_at_hold)
    pf.CONFIG.mpc_ff_input_increment_weight = mpc_ff_input_increment_weight
    pf.CONFIG.mpc_input_tracking_weight_override = mpc_input_tracking_weight_override
    pf.CONFIG.mpc_input_increment_weight_override = mpc_input_increment_weight_override
    pf.CONFIG.mpc_solver_time_limit_s = mpc_solver_time_limit_s
    pf.CONFIG.mpc_directional_damping = float(mpc_directional_damping)
    pf.CONFIG.mpc_directional_damping_floor = float(mpc_directional_damping_floor)
    pf.CONFIG.output_root = str(out_dir / "runs")
    pf.CONFIG.run_name = f"{kind_key}_rep{rep}"
    try:
        pf.main()
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] pf.main() raised {exc!r}")
        return 4
    return 0


# ==========================================================================
# metrics + analysis
# ==========================================================================
def run_metrics(run_dir: Path, kind_key: str, rep: int) -> dict:
    a = np.genfromtxt(run_dir / "tip_trajectory.csv", delimiter=",", names=True)
    err = np.atleast_1d(a["err_norm_mm"])
    t = np.atleast_1d(a["t_s"])
    hold = np.atleast_1d(a["terminal_hold"]).astype(bool)
    track = ~hold
    try:
        summ = json.loads((run_dir / "summary.json").read_text())
    except Exception:
        summ = {}
    return {
        "controller": KIND[kind_key], "key": kind_key, "rep": rep,
        "run_dir": str(run_dir),
        "stop_reason": summ.get("stop_reason", "?"),
        "steps": int(summ.get("control_steps", len(err))),
        "n_track": int(track.sum()),
        "rms_track_mm": float(np.sqrt(np.mean(err[track] ** 2))) if track.any() else float("nan"),
        "mean_track_mm": float(np.mean(err[track])) if track.any() else float("nan"),
        "max_track_mm": float(np.max(err[track])) if track.any() else float("nan"),
        "p95_track_mm": float(np.percentile(err[track], 95)) if track.any() else float("nan"),
        "mean_hold_mm": float(np.mean(err[hold])) if hold.any() else float("nan"),
        "final_mm": float(err[-1]),
        "duration_s": float(t[-1]),
    }


def _newest_run(out_dir: Path, kind_key: str, rep: int) -> Path | None:
    hits = sorted((out_dir / "runs").glob(f"{kind_key}_rep{rep}_*"))
    return hits[-1] if hits else None


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    path.write_text(",".join(cols) + "\n" + "\n".join(
        ",".join(str(r[c]) for c in cols) for r in rows) + "\n")


def analyse(out_dir: Path):
    rows = json.loads((out_dir / "results.json").read_text())
    rows = [r for r in rows if np.isfinite(r.get("rms_track_mm", np.nan))]
    if not rows:
        print("no usable results")
        return
    keys = []
    for r in rows:
        if r["key"] not in keys:
            keys.append(r["key"])
    METRICS = ["rms_track_mm", "mean_track_mm", "max_track_mm", "p95_track_mm", "mean_hold_mm", "final_mm"]
    agg = {}
    hdr = f"{'controller':22s} {'n':>2s}  " + "  ".join(f"{m.replace('_track_mm','').replace('_mm',''):>13s}" for m in METRICS)
    print("\n" + hdr)
    for k in keys:
        sub = [r for r in rows if r["key"] == k]
        agg[k] = {"n": len(sub)}
        line = f"{KIND[k]:22s} {len(sub):>2d}  "
        for m in METRICS:
            vals = np.array([r[m] for r in sub], float)
            agg[k][m] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                         "vals": [float(x) for x in vals]}
            line += f"  {np.nanmean(vals):5.2f}+-{np.nanstd(vals):<4.2f}"
        print(line)
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    bp = np.array([0.525575, -0.670028, -0.016567])
    u = np.array([-1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0])
    col = {"inv": "tab:blue", "lti": "tab:orange", "ltv": "tab:red"}

    fig, ax = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 5), squeeze=False)
    for j, k in enumerate(keys):
        drew = False
        for r in [x for x in rows if x["key"] == k]:
            csv = Path(r["run_dir"]) / "tip_trajectory.csv"
            if not csv.exists():
                continue
            a = np.genfromtxt(csv, delimiter=",", names=True)
            tip = np.c_[a["tip_x_m"], a["tip_y_m"], a["tip_z_m"]]
            des = np.c_[a["des_x_m"], a["des_y_m"], a["des_z_m"]]
            tuv = np.c_[(tip - bp) @ u, (tip - bp) @ v] * 1e3
            duv = np.c_[(des - bp) @ u, (des - bp) @ v] * 1e3
            if not drew:
                ax[0][j].plot(duv[:, 1], duv[:, 0], "k--", lw=1.3, alpha=.7, label="desired")
                drew = True
            ax[0][j].plot(tuv[:, 1], tuv[:, 0], "-", color=col[k], alpha=.65, label=f"rep {r['rep']}")
        ax[0][j].set_title(KIND[k]); ax[0][j].set_aspect("equal"); ax[0][j].grid(alpha=.3)
        ax[0][j].set_xlabel("tip B.y (mm)"); ax[0][j].set_ylabel("tip B.x (mm)")
        ax[0][j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "compare_paths.png", dpi=110)

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, m in enumerate(["rms_track_mm", "max_track_mm", "mean_hold_mm"]):
        ax2[i].bar([KIND[k] for k in keys], [agg[k][m]["mean"] for k in keys],
                   yerr=[agg[k][m]["std"] for k in keys], capsize=5,
                   color=[col[k] for k in keys])
        ax2[i].set_title(m.replace("_mm", " [mm]")); ax2[i].grid(alpha=.3, axis="y")
        ax2[i].tick_params(axis="x", labelrotation=12)
    fig2.tight_layout(); fig2.savefig(out_dir / "compare_bars.png", dpi=110)
    print(f"\nwrote {out_dir}/compare_paths.png  {out_dir}/compare_bars.png  {out_dir}/summary.json")


# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--controllers", default="inv,lti,ltv")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--plan-dir", default=str(REPO / DEFAULT_PLAN))
    p.add_argument("--beam-len-mm", type=float, default=30.0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--analyze-only", default=None)
    p.add_argument("--feedforward", action="store_true",
                   help="servo the PLANNED joints + a feedback trim (trajectory "
                        "tracking) instead of pure tip-error feedback")
    p.add_argument("--force-mpc-feedforward", action="store_true",
                   help="with --feedforward, also run mpc_lti/mpc_ltv_offline in "
                        "feedforward mode instead of the normal auto-disable "
                        "(diagnostic: known-worse terminal convergence, run to "
                        "collect instrumented data on why)")
    p.add_argument("--ff-trim-base-switch-at-hold", action="store_true",
                   help="diagnostic fix #2: once the terminal hold is reached, "
                        "trim off q_meas instead of the frozen planned reference "
                        "(pure feedback during hold, FF during transit)")
    p.add_argument("--mpc-ff-input-increment-weight", type=float, default=None,
                   help="diagnostic fix #1: override ConfigurationMPCConfig's "
                        "input_increment_weight (default 1e-3) for FF-mode MPC "
                        "only, e.g. 1e-4 or 0.0")
    p.add_argument("--mpc-input-tracking-weight", type=float, default=None,
                   help="override ConfigurationMPCConfig's input_tracking_weight "
                        "(R, default 1e-2) unconditionally (both FF and pure "
                        "feedback) -- e.g. near-0 to test MPC's unconstrained "
                        "behaviour against inverse-Jacobian's. CAUTION: can "
                        "increase solver iterations/time on an ill-conditioned "
                        "problem -- watch solve time, not just tracking error.")
    p.add_argument("--mpc-input-increment-weight", type=float, default=None,
                   help="same as --mpc-input-tracking-weight, for "
                        "input_increment_weight (Rd, default 1e-3), applied "
                        "unconditionally.")
    p.add_argument("--mpc-solver-time-limit-s", type=float, default=None,
                   help="real-time fairness cap: OSQP's own wall-clock cutoff "
                        "per solve (0 = disabled/OSQP default). On an "
                        "ill-conditioned problem ADMM iterations can blow up "
                        "and overrun the control period, causing the "
                        "wallclock-progress reference to skip samples -- set "
                        "to a fraction of dt (e.g. 0.08 at 10Hz) so MPC is "
                        "never structurally handicapped vs inverse-Jacobian's "
                        "near-instant solve. See beam-lateral-authority-limit "
                        "memory for the diagnosis.")
    p.add_argument("--mpc-directional-damping", type=float, default=0.0,
                   help="anisotropic input regularisation gain (0 = disabled): "
                        "penalises only the near-null singular direction of "
                        "each horizon step's local Jacobian, mirroring "
                        "inverse-Jacobian's damped-least-squares law "
                        "directionally instead of via a uniform input weight "
                        "(which was confirmed to plateau -- see "
                        "beam-lateral-authority-limit memory).")
    p.add_argument("--mpc-directional-damping-floor", type=float, default=0.01,
                   help="bounded-weight refinement: caps the near-null "
                        "direction's penalty weight to ~1/floor^2 instead of "
                        "unbounded 1/sigma^2 (units match the Jacobian's "
                        "singular values, comparable to inverse-Jacobian's "
                        "damping=0.05). The unbounded version wrecked OSQP's "
                        "solve time live -- see beam-lateral-authority-limit "
                        "memory. Only used when --mpc-directional-damping > 0.")
    # hidden worker entry point
    p.add_argument("--_worker", nargs=16, default=None,
                   metavar=("KIND", "REP", "OUT_DIR", "SCHED_NPZ", "PLAN_DIR", "HORIZON",
                            "BEAMLEN", "FF", "FORCE_MPC_FF", "FF_HOLD_SWITCH", "RD_WEIGHT",
                            "R_OVERRIDE", "RD_OVERRIDE", "TIME_LIMIT", "DIR_DAMPING",
                            "DIR_DAMPING_FLOOR"))
    args = p.parse_args()

    if args._worker:
        (kind_key, rep, out_dir, sched, plan_dir, horizon, blen, ff, force_mpc_ff,
         ff_hold_switch, rd_weight, r_override, rd_override,
         time_limit, dir_damping, dir_damping_floor) = args._worker
        sys.exit(_worker(kind_key, int(rep), Path(out_dir), Path(sched),
                         plan_dir, int(horizon), float(blen), bool(int(ff)),
                         bool(int(force_mpc_ff)), bool(int(ff_hold_switch)),
                         None if rd_weight == "none" else float(rd_weight),
                         None if r_override == "none" else float(r_override),
                         None if rd_override == "none" else float(rd_override),
                         None if time_limit == "none" else float(time_limit),
                         float(dir_damping), float(dir_damping_floor)))

    if args.analyze_only:
        analyse(Path(args.analyze_only))
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or (REPO / "close_loop_logs" / f"compare_{stamp}"))
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items() if k != "_worker"}, indent=2))
    print(f"[compare] out -> {out_dir}")

    order = [k.strip() for k in args.controllers.split(",") if k.strip()]
    for k in order:
        if k not in KIND:
            raise SystemExit(f"unknown controller {k!r}; use inv,lti,ltv")

    sched_npz = out_dir / "ltv_schedule.npz"
    if any(k in ("lti", "ltv") for k in order):
        build_ltv_schedule(args.plan_dir, sched_npz)

    results: list[dict] = []
    first = True
    for rep in range(1, args.reps + 1):
        for k in order:
            print(f"\n{'=' * 72}\n  rep {rep}/{args.reps}   {KIND[k]}\n{'=' * 72}", flush=True)
            cmd = [sys.executable, "-m", "proper_research.hardware.online.compare_controllers_live",
                   "--_worker", k, str(rep), str(out_dir), str(sched_npz),
                   args.plan_dir, str(args.horizon), str(args.beam_len_mm),
                   "1" if args.feedforward else "0",
                   "1" if args.force_mpc_feedforward else "0",
                   "1" if args.ff_trim_base_switch_at_hold else "0",
                   ("none" if args.mpc_ff_input_increment_weight is None
                    else str(args.mpc_ff_input_increment_weight)),
                   ("none" if args.mpc_input_tracking_weight is None
                    else str(args.mpc_input_tracking_weight)),
                   ("none" if args.mpc_input_increment_weight is None
                    else str(args.mpc_input_increment_weight)),
                   ("none" if args.mpc_solver_time_limit_s is None
                    else str(args.mpc_solver_time_limit_s)),
                   str(args.mpc_directional_damping),
                   str(args.mpc_directional_damping_floor)]
            rc, rd = None, None
            for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
                if not first:
                    print(f"[compare] settle {INTER_RUN_SETTLE_S:.0f}s (RTDE)...", flush=True)
                    time.sleep(INTER_RUN_SETTLE_S)
                first = False
                try:
                    rc = subprocess.run(cmd, cwd=str(REPO), timeout=RUN_TIMEOUT_S).returncode
                except subprocess.TimeoutExpired:
                    print(f"[compare] {KIND[k]} rep{rep} attempt {attempt} TIMED OUT")
                    rc = 99
                rd = _newest_run(out_dir, k, rep)
                ok = rc == 0 and rd is not None and (rd / "tip_trajectory.csv").exists() \
                    and (rd / "summary.json").exists()
                if ok:
                    break
                print(f"[compare] {KIND[k]} rep{rep} attempt {attempt} failed (rc={rc}); "
                      f"{'retrying' if attempt < MAX_ATTEMPTS_PER_RUN else 'giving up'}")
            if rc == 0 and rd is not None and (rd / "tip_trajectory.csv").exists():
                try:
                    m = run_metrics(rd, k, rep)
                    results.append(m)
                    print(f"[metrics] {KIND[k]} rep{rep}: rms={m['rms_track_mm']:.2f} "
                          f"max={m['max_track_mm']:.2f} hold={m['mean_hold_mm']:.2f} "
                          f"final={m['final_mm']:.2f} mm ({m['stop_reason']})")
                except Exception as exc:  # noqa: BLE001
                    print(f"[compare] metrics failed for {rd}: {exc!r}")
            else:
                print(f"[compare] {KIND[k]} rep{rep} FAILED (rc={rc}, dir={rd})")
                results.append({"controller": KIND[k], "key": k, "rep": rep,
                                "run_dir": str(rd) if rd else "", "stop_reason": f"rc{rc}",
                                "rms_track_mm": float("nan"), "mean_track_mm": float("nan"),
                                "max_track_mm": float("nan"), "p95_track_mm": float("nan"),
                                "mean_hold_mm": float("nan"), "final_mm": float("nan"),
                                "duration_s": float("nan"), "steps": 0, "n_track": 0})
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))
            _write_csv(out_dir / "results.csv", results)

    print("\n[compare] returning robot to reference...")
    try:
        move_to_ref()
    except Exception as exc:  # noqa: BLE001
        print(f"[compare] final move_to_ref: {exc!r}")
    analyse(out_dir)
    print(f"\n[compare] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
