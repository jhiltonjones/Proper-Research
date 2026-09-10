#!/usr/bin/env python3
"""Play the inverse-configuration trajectory OPEN LOOP and measure the tip.

No vision feedback: the robot is servoJ'd straight through the planned joint
waypoints ``reference.state[i][:6]`` and the advancer follows the planned
insertion rate.  The gap between the camera tip and ``desired_position_m[i]``
is then the pure PLANT-vs-MODEL mismatch of the offline plan -- it tells you
whether the closed-loop controllers miss the triangle corners because

  * the plan / forward model is optimistic about how far the beam bends
    (open loop misses the corners by a similar or larger margin), or
  * the time parameterisation is too fast for the beam to settle
    (open loop improves markedly when replayed slower -- try --speed-scale 2),
  * or it is a controller-tuning / authority issue
    (open loop tracks the corners fine, closed loop does not).

    # nominal speed, 1 rep
    python -m proper_research.hardware.online.open_loop_playback

    # half speed, 2 reps, a different plan
    python -m proper_research.hardware.online.open_loop_playback \
        --speed-scale 2.0 --reps 2 \
        --plan-dir plans/<...>/time_parameterized_configuration_path

    # overlay against a finished closed-loop compare run + redo the plot
    python -m proper_research.hardware.online.open_loop_playback \
        --analyze-only close_loop_logs/openloop_<stamp> \
        --compare-dir close_loop_logs/compare_tri
"""
from __future__ import annotations

import argparse
import io
import contextlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = "plans/triangle_8mm_2026-09-10/time_parameterized_configuration_path"
REF_JOINTS = np.array(
    [-0.85634357, -1.94584002, -1.76176286, -1.03319450, 1.56234264, -2.05640871]
)


@contextlib.contextmanager
def _quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


# reuse the hardened hardware helpers
from proper_research.hardware.online.compare_controllers_live import (  # noqa: E402
    move_to_ref,
    set_beam_length,
)
from proper_research.hardware.online.close_loop_path_follow import (  # noqa: E402
    _fit_planner_to_robot,
    _find_shape_npz,
)

CSV_HEADER = (
    "step,t_s,ref_index,tip_x_m,tip_y_m,tip_z_m,"
    "des_x_m,des_y_m,des_z_m,err_norm_mm,"
    "q1,q2,q3,q4,q5,q6,insertion_m,terminal_hold\n"
)


def _one_playback(plan_dir: str, speed_scale: float, out_dir: Path, rep: int) -> Path | None:
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    ref = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    states = np.asarray(ref.state, dtype=float)           # (N, 7) [q1..q6, insertion]
    des_planner = np.asarray(ref.desired_position_m, dtype=float)  # (N, 3) planner frame
    n = states.shape[0]
    dt_plan = float(ref.sample_period_s)
    dt_ctrl = dt_plan * float(speed_scale)
    L0 = float(states[0, 6])
    print(f"[open-loop] {n} samples, plan dt {dt_plan:.3f}s -> playback dt {dt_ctrl:.3f}s "
          f"(x{speed_scale:.2f}), L0 {1e3 * L0:.1f} mm")

    # ---- reset robot + beam ----
    if not move_to_ref():
        print("[open-loop] cannot reach reference joints")
        return None
    time.sleep(3.0)
    if abs(np.max(np.abs(states[0, :6] - REF_JOINTS))) > 1e-3:
        print("[open-loop] WARNING: plan start joints != REF_JOINTS")
    Lm = set_beam_length(L0)
    if Lm is None or abs(Lm - L0) > 1.5e-3:
        print(f"[open-loop] beam length not set (got {Lm})")
        return None
    print(f"[open-loop] beam at {1e3 * Lm:.1f} mm")
    time.sleep(1.0)

    scfg = StateStreamConfig(exposure=29.0)
    with _quiet():
        mapper = NewFrameTipMapper(scfg)
    T_R_B = mapper.T_R_B
    ins_state = {"v": L0}
    cam = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/openloop.png",
            roi_polygon_path=scfg.roi_polygon_path,
            manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: ins_state["v"], frame_processor=mapper,
    )
    cam.start()
    adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False,
                                          max_rate_m_s=2.5e-3))
    adv.start()
    robot = URRTDERobot("192.168.56.101", frequency=500.0)

    run_dir = out_dir / "runs" / f"openloop_x{speed_scale:g}_rep{rep}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj = (run_dir / "tip_trajectory.csv").open("w")
    traj.write(CSV_HEADER)

    def tip_R():
        est, _ = cam.latest(0.5)
        return None if est is None else np.asarray(est.tip_position_m, dtype=float)

    abort = ""
    try:
        # warm up vision + planner->robot fit
        t_end = time.monotonic() + 10.0
        first = None
        while time.monotonic() < t_end:
            first = tip_R()
            if first is not None:
                break
            time.sleep(0.1)
        if first is None:
            abort = "no_vision_warmup"
            raise RuntimeError(abort)
        npz = _find_shape_npz(Path(plan_dir))
        if npz is not None:
            r_fit, t_fit = _fit_planner_to_robot(npz, first, T_R_B)
        else:
            r_fit, t_fit = np.eye(3), np.zeros(3)
        des_R = des_planner @ r_fit.T + t_fit
        print(f"[open-loop] planner->R fit |t|={np.linalg.norm(t_fit):.3f} det={np.linalg.det(r_fit):+.3f}")

        t0 = time.monotonic()
        next_tick = t0
        hold_steps = int(round(3.0 / dt_ctrl))
        total = (n - 1) + hold_steps
        for step in range(total):
            i = min(step + 1, n - 1)
            terminal = step >= (n - 1)
            q_cmd = states[i, :6]
            ins_rate_plan = (states[i, 6] - states[i - 1, 6]) / dt_plan if i > 0 else 0.0
            lk = min(0.2, max(0.05, 1.2 * dt_ctrl))  # servoJ requires lookahead in [0.03, 0.2]
            robot.servo_j(list(q_cmd), time_s=dt_ctrl, lookahead_time=lk, gain=200)
            # keep the true travelled distance the same -> scale the rate by 1/speed
            rate = 0.0 if terminal else float(ins_rate_plan / float(speed_scale))
            adv.submit_rate(rate, dt_ctrl)
            if not terminal:
                ins_state["v"] = float(np.clip(ins_state["v"] + rate * dt_ctrl, 0.005, 0.05))

            tp = tip_R()
            if tp is not None:
                d = des_R[i]
                e = d - tp
                b_z = np.asarray(T_R_B.rotation[:, 2])
                e_ip = e - float(np.dot(e, b_z)) * b_z
                err_mm = 1e3 * float(np.linalg.norm(e_ip))
                traj.write(
                    f"{step},{time.monotonic() - t0:.4f},{i},"
                    f"{tp[0]:.6f},{tp[1]:.6f},{tp[2]:.6f},"
                    f"{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{err_mm:.3f},"
                    + ",".join(f"{v:.6f}" for v in q_cmd)
                    + f",{ins_state['v']:.6f},{int(terminal)}\n"
                )
                traj.flush()
                if step % 10 == 0 or (terminal and step % 5 == 0):
                    print(f"[{step:4d}] i={i:3d}/{n - 1} |e|={err_mm:6.2f}mm ins={1e3 * ins_state['v']:.1f}mm"
                          + (" TERM" if terminal else ""))
            next_tick += dt_ctrl
            slp = next_tick - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            if step % 25 == 0:
                try:
                    if robot.is_protective_stopped():
                        abort = "protective_stop"
                        break
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        abort = abort or repr(exc)
        print(f"[open-loop] aborted: {abort}")
    finally:
        try:
            robot.servo_stop(2.0)
        except Exception:
            pass
        try:
            adv.submit_rate(0.0, dt_ctrl)
            adv.stop()
        except Exception:
            pass
        cam.stop()
        robot.close()
        traj.close()

    (run_dir / "summary.json").write_text(json.dumps({
        "kind": "open_loop", "speed_scale": speed_scale, "rep": rep,
        "plan_dir": plan_dir, "stop_reason": abort or "path_complete",
        "samples": n, "playback_dt_s": dt_ctrl,
    }, indent=2))
    return run_dir


def _metrics(run_dir: Path, speed_scale: float, rep: int) -> dict:
    a = np.genfromtxt(run_dir / "tip_trajectory.csv", delimiter=",", names=True)
    err = np.atleast_1d(a["err_norm_mm"])
    hold = np.atleast_1d(a["terminal_hold"]).astype(bool)
    track = ~hold
    summ = json.loads((run_dir / "summary.json").read_text())
    return {
        "key": f"openloop_x{speed_scale:g}", "controller": f"open loop x{speed_scale:g}",
        "rep": rep, "run_dir": str(run_dir), "stop_reason": summ.get("stop_reason", "?"),
        "rms_track_mm": float(np.sqrt(np.mean(err[track] ** 2))) if track.any() else float("nan"),
        "mean_track_mm": float(np.mean(err[track])) if track.any() else float("nan"),
        "max_track_mm": float(np.max(err[track])) if track.any() else float("nan"),
        "p95_track_mm": float(np.percentile(err[track], 95)) if track.any() else float("nan"),
        "mean_hold_mm": float(np.mean(err[hold])) if hold.any() else float("nan"),
        "final_mm": float(err[-1]),
    }


def analyse(out_dir: Path, compare_dir: Path | None):
    rows = json.loads((out_dir / "results.json").read_text())
    rows = [r for r in rows if np.isfinite(r.get("rms_track_mm", np.nan))]
    if not rows:
        print("no usable results")
        return
    bp = np.array([0.525575, -0.670028, -0.016567])
    u = np.array([-1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0])

    print(f"\n{'run':22s} {'rms':>7s} {'mean':>7s} {'max':>7s} {'p95':>7s} {'hold':>7s} {'final':>7s}   [mm]")
    keys = []
    for r in rows:
        if r["key"] not in keys:
            keys.append(r["key"])
    agg = {}
    for k in keys:
        sub = [r for r in rows if r["key"] == k]
        agg[k] = {}
        for m in ("rms_track_mm", "mean_track_mm", "max_track_mm", "p95_track_mm", "mean_hold_mm", "final_mm"):
            vals = np.array([r[m] for r in sub], float)
            agg[k][m] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        a = agg[k]
        print(f"{sub[0]['controller']:22s} "
              f"{a['rms_track_mm']['mean']:7.2f} {a['mean_track_mm']['mean']:7.2f} "
              f"{a['max_track_mm']['mean']:7.2f} {a['p95_track_mm']['mean']:7.2f} "
              f"{a['mean_hold_mm']['mean']:7.2f} {a['final_mm']['mean']:7.2f}")

    cmp_rows = []
    if compare_dir is not None and (compare_dir / "results.json").exists():
        cmp_rows = [r for r in json.loads((compare_dir / "results.json").read_text())
                    if np.isfinite(r.get("rms_track_mm", np.nan))]
        for k in sorted({r["key"] for r in cmp_rows}):
            sub = [r for r in cmp_rows if r["key"] == k]
            print(f"{'(closed) ' + sub[0]['controller']:22.22s} "
                  f"{np.nanmean([r['rms_track_mm'] for r in sub]):7.2f} "
                  f"{np.nanmean([r['mean_track_mm'] for r in sub]):7.2f} "
                  f"{np.nanmean([r['max_track_mm'] for r in sub]):7.2f} "
                  f"{np.nanmean([r['p95_track_mm'] for r in sub]):7.2f} "
                  f"{np.nanmean([r['mean_hold_mm'] for r in sub]):7.2f} "
                  f"{np.nanmean([r['final_mm'] for r in sub]):7.2f}")
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    drew_des = False
    for r in rows:
        a = np.genfromtxt(Path(r["run_dir"]) / "tip_trajectory.csv", delimiter=",", names=True)
        tip = np.c_[a["tip_x_m"], a["tip_y_m"], a["tip_z_m"]]
        des = np.c_[a["des_x_m"], a["des_y_m"], a["des_z_m"]]
        tuv = np.c_[(tip - bp) @ u, (tip - bp) @ v] * 1e3
        duv = np.c_[(des - bp) @ u, (des - bp) @ v] * 1e3
        if not drew_des:
            ax[0].plot(duv[:, 1], duv[:, 0], "k--", lw=1.4, label="desired (plan)")
            drew_des = True
        ax[0].plot(tuv[:, 1], tuv[:, 0], "-", lw=1.6, label=r["controller"] + f" rep{r['rep']}")
    for r in cmp_rows:
        csv = Path(r["run_dir"]) / "tip_trajectory.csv"
        if not csv.exists():
            continue
        a = np.genfromtxt(csv, delimiter=",", names=True)
        tip = np.c_[a["tip_x_m"], a["tip_y_m"], a["tip_z_m"]]
        tuv = np.c_[(tip - bp) @ u, (tip - bp) @ v] * 1e3
        ax[0].plot(tuv[:, 1], tuv[:, 0], "-", color="0.7", lw=0.8, alpha=0.6)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("tip B.y (in-plane, mm)"); ax[0].set_ylabel("tip B.x (axial, mm)")
    ax[0].set_title("open-loop tip path vs plan  (grey = closed-loop runs)")
    ax[0].legend(fontsize=8)

    for r in rows:
        a = np.genfromtxt(Path(r["run_dir"]) / "tip_trajectory.csv", delimiter=",", names=True)
        ax[1].plot(a["t_s"], a["err_norm_mm"], "-", label=r["controller"] + f" rep{r['rep']}")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("in-plane tracking error [mm]")
    ax[1].set_title("open-loop error vs plan"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "open_loop.png", dpi=110)
    print(f"\nwrote {out_dir}/open_loop.png  {out_dir}/summary.json")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", default=str(REPO / DEFAULT_PLAN))
    p.add_argument("--speed-scale", type=float, nargs="+", default=[1.0],
                   help="playback dt multiplier(s); >1 = slower.  e.g. --speed-scale 1 2")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--analyze-only", default=None)
    p.add_argument("--compare-dir", default=None,
                   help="a close_loop_logs/compare_* dir to overlay for reference")
    args = p.parse_args()

    if args.analyze_only:
        analyse(Path(args.analyze_only),
                Path(args.compare_dir) if args.compare_dir else None)
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or (REPO / "close_loop_logs" / f"openloop_{stamp}"))
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items()}, indent=2))

    results = []
    for scale in args.speed_scale:
        for rep in range(1, args.reps + 1):
            print(f"\n{'=' * 72}\n  open loop  x{scale:g}  rep {rep}/{args.reps}\n{'=' * 72}", flush=True)
            rd = _one_playback(args.plan_dir, float(scale), out_dir, rep)
            if rd is not None and (rd / "tip_trajectory.csv").exists():
                try:
                    results.append(_metrics(rd, float(scale), rep))
                    m = results[-1]
                    print(f"[metrics] x{scale:g} rep{rep}: rms={m['rms_track_mm']:.2f} "
                          f"max={m['max_track_mm']:.2f} final={m['final_mm']:.2f} mm ({m['stop_reason']})")
                except Exception as exc:  # noqa: BLE001
                    print(f"[metrics] failed for {rd}: {exc!r}")
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))
            time.sleep(20)

    print("\n[open-loop] returning robot to reference...")
    try:
        move_to_ref()
    except Exception:
        pass
    analyse(out_dir, Path(args.compare_dir) if args.compare_dir else None)
    print(f"\n[open-loop] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
