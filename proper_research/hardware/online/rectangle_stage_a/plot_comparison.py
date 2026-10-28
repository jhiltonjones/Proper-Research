#!/usr/bin/env python3
"""Compare multiple Stage-A runs: prints the standard metrics table and
writes overlay plots (tip error vs time, servo error vs time, tip
trajectory in-plane) so you can look at, not just read, what happened.

Metrics (matching the tables used throughout the 2026-09-17 rectangle
investigation):
  E_RMS(active), E_max(active), E_final   -- beam-tracking, terminal_hold
                                              ticks excluded from RMS/max
                                              (passive magnet-retreat/beam-
                                              relaxation ticks are not
                                              evidence of active control)
  E_servo_RMS, E_servo_max                -- ||q_cmd - q_meas||, execution-
                                              layer fidelity (only present
                                              for accumulator_seam runs --
                                              i.e. everything through
                                              execution layer C)
  P(rate sat)                             -- fraction of ticks with any
                                              joint at the max_joint_step_rad
                                              seam-level clip
  R_q                                     -- command roughness (RMS of the
                                              second difference of q_cmd)
  P(err>4mm)                              -- fraction of active ticks with
                                              beam error above 4mm

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.plot_comparison \\
        --run close_loop_logs/.../openloopff_accumC_<ts> \\
        --run close_loop_logs/.../inv_2dof_trim_kp0.6_kn0_<ts> \\
        --run close_loop_logs/.../mpc_ltv_accumC_<ts> \\
        --labels open-loop-C,INV-2DOF-trim,MPC-LTV \\
        --out-dir /tmp/stage_a_comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: str) -> dict:
    rows = [json.loads(l) for l in open(f"{run_dir}/path_follow.jsonl")]
    active = [r for r in rows if not r.get("terminal_hold")]
    err_active = np.array([r["error_norm_mm"] for r in active])
    err_all = np.array([r["error_norm_mm"] for r in rows])
    t_all = np.array([r["t_s"] for r in rows])

    servo = [r.get("q_servo_error_rad") for r in rows]
    has_servo = any(s is not None for s in servo)
    servo_norm = (
        np.linalg.norm(np.array([s for s in servo if s is not None]), axis=1)
        if has_servo else np.array([])
    )

    delta_q = np.array([r["q_target_delta_rad"] for r in rows])
    max_step = float(np.max(np.abs(delta_q)))  # heuristic: the seam's own clip bound
    rate_sat = np.any(np.abs(delta_q) >= max_step - 1.0e-5, axis=1) if max_step > 0 else np.zeros(len(rows), dtype=bool)

    q_cmd = np.array([r["q_target_rad"] for r in rows])
    d1 = np.diff(q_cmd, axis=0)
    d2 = np.diff(d1, axis=0)
    R_q = float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=1)))) if len(d2) else float("nan")

    tip_mm = np.array([r["tip_mm"] for r in rows])
    des_mm = np.array([r["desired_mm"] for r in rows])

    return dict(
        rows=rows, t_all=t_all, err_all=err_all, err_active=err_active,
        servo_norm=servo_norm, has_servo=has_servo, rate_sat=rate_sat,
        R_q=R_q, tip_mm=tip_mm, des_mm=des_mm,
    )


def print_table(runs: dict[str, dict]) -> None:
    header = (f"{'label':22s} {'E_RMS':>8} {'E_max':>8} {'E_final':>8} "
              f"{'Eserv_RMS':>10} {'Eserv_max':>10} {'P(rate_sat)':>12} "
              f"{'R_q(mrad)':>10} {'P(>4mm)':>8}")
    print(header)
    print("-" * len(header))
    for label, d in runs.items():
        e_rms = np.sqrt(np.mean(d["err_active"] ** 2))
        e_max = np.max(d["err_active"])
        e_final = d["err_all"][-1]
        p_spike = np.mean(d["err_active"] > 4.0)
        p_rate_sat = np.mean(d["rate_sat"])
        if d["has_servo"]:
            serv_rms = f"{np.sqrt(np.mean(d['servo_norm']**2))*1000:.2f}"
            serv_max = f"{np.max(d['servo_norm'])*1000:.2f}"
        else:
            serv_rms = serv_max = "n/a"
        print(f"{label:22s} {e_rms:8.3f} {e_max:8.3f} {e_final:8.3f} "
              f"{serv_rms:>10} {serv_max:>10} {p_rate_sat:12.3f} "
              f"{d['R_q']*1000:10.3f} {p_spike:8.3f}")


def write_plots(runs: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, d in runs.items():
        ax.plot(d["t_all"], d["err_all"], label=label, lw=1.2)
    ax.set_xlabel("t (s)"); ax.set_ylabel("beam tip error (mm)")
    ax.set_title("Beam-tracking error vs time")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "error_vs_time.png", dpi=150)
    plt.close(fig)

    have_servo = {k: v for k, v in runs.items() if v["has_servo"]}
    if have_servo:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for label, d in have_servo.items():
            ax.plot(np.arange(len(d["servo_norm"])) * 0.1, d["servo_norm"] * 1000, label=label, lw=1.2)
        ax.set_xlabel("t (s)"); ax.set_ylabel("||q_cmd - q_meas|| (mrad)")
        ax.set_title("Execution-layer servo error vs time")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "servo_error_vs_time.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    any_des = next(iter(runs.values()))["des_mm"]
    ax.plot(any_des[:, 0], any_des[:, 1], "k--", lw=1.0, label="planned (desired)")
    for label, d in runs.items():
        ax.plot(d["tip_mm"][:, 0], d["tip_mm"][:, 1], lw=1.2, label=label)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Tip trajectory (in-plane)")
    ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "tip_trajectory.png", dpi=150)
    plt.close(fig)

    print(f"[plot] wrote error_vs_time.png, "
          f"{'servo_error_vs_time.png, ' if have_servo else ''}"
          f"tip_trajectory.png -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True, dest="runs",
                   help="a run directory (repeat for each condition, e.g. "
                        "--run path/to/run1 --run path/to/run2)")
    p.add_argument("--labels", default=None,
                   help="comma-separated labels matching --run order (default: dir names)")
    p.add_argument("--out-dir", default=None, help="write comparison plots here")
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise ValueError(f"--labels has {len(labels)} entries but --run was given {len(args.runs)} times")

    runs = {label: load_run(run_dir) for label, run_dir in zip(labels, args.runs)}
    print_table(runs)
    if args.out_dir:
        write_plots(runs, Path(args.out_dir))


if __name__ == "__main__":
    main()
