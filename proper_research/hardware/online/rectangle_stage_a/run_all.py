#!/usr/bin/env python3
"""Run the whole Stage-A pipeline against one plan: open-loop-C acceptance
test, then INV-2DOF-trim, then MPC-LTV (genuine schedule), then the
comparison table + plots. Each stage is a separate subprocess (not an
in-process call) so `close_loop_path_follow.CONFIG`'s module-level state
never leaks between stages the way it would if this script imported and
called each `main()` directly.

This does NOT run the offline planner itself -- point --plan-dir at a
`time_parameterized_configuration_path` directory `run_time_parameterization.py`
already produced. See this package's README for the full plan-to-plots
pipeline including that step.

Stops after any stage fails (a live run aborting is not necessarily a bug --
see the README on `tcp_out_of_workspace` -- but this script does not try to
guess whether it's safe to continue past one).

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_all \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/stage_a_$(date +%Y%m%d) \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy

Skip a stage with --skip-open-loop / --skip-inv / --skip-mpc (e.g. to
re-plot an existing set of runs, or re-run just the one that aborted).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n{'='*88}\n$ {' '.join(cmd)}\n{'='*88}")
    subprocess.run(cmd, check=True)


def latest_run_dir(out_dir: str, run_name_prefix: str) -> str:
    candidates = sorted(Path(out_dir, "runs").glob(f"{run_name_prefix}*"))
    if not candidates:
        raise FileNotFoundError(f"no run directory matching {run_name_prefix}* under {out_dir}/runs")
    return str(candidates[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--schedule-cache", required=True,
                   help="genuine LTV Jacobian schedule cache (built once, ~90-150s, "
                        "reused by run_mpc_ltv.py and q_ablation.py)")
    p.add_argument("--kp", type=float, default=0.6)
    p.add_argument("--kn", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--skip-open-loop", action="store_true")
    p.add_argument("--skip-inv", action="store_true")
    p.add_argument("--skip-mpc", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--run-q-ablation", action="store_true",
                   help="also run the offline (no robot) Q-ablation on the MPC run")
    args = p.parse_args()

    py = sys.executable
    pkg = "proper_research.hardware.online.rectangle_stage_a"

    if not args.skip_open_loop:
        run([py, "-m", f"{pkg}.run_open_loop_c",
             "--plan-dir", args.plan_dir, "--out-dir", args.out_dir])

    if not args.skip_inv:
        run([py, "-m", f"{pkg}.run_inv_2dof_trim",
             "--plan-dir", args.plan_dir, "--out-dir", args.out_dir,
             "--kp", str(args.kp), "--kn", str(args.kn)])

    if not args.skip_mpc:
        run([py, "-m", f"{pkg}.run_mpc_ltv",
             "--plan-dir", args.plan_dir, "--out-dir", args.out_dir,
             "--schedule-cache", args.schedule_cache, "--horizon", str(args.horizon)])

    open_loop_dir = latest_run_dir(args.out_dir, "openloopff_accumC")
    inv_dir = latest_run_dir(args.out_dir, f"inv_2dof_trim_kp{args.kp:g}_kn{args.kn:g}")
    mpc_dir = latest_run_dir(args.out_dir, "mpc_ltv_accumC")

    if not args.skip_plots:
        run([py, "-m", f"{pkg}.plot_comparison",
             "--run", open_loop_dir, "--run", inv_dir, "--run", mpc_dir,
             "--labels", f"open-loop-C,INV-2DOF-trim(kp{args.kp:g}),MPC-LTV",
             "--out-dir", str(Path(args.out_dir) / "comparison_plots")])

    if args.run_q_ablation:
        run([py, "-m", f"{pkg}.q_ablation",
             "--plan-dir", args.plan_dir, "--mpc-run-dir", mpc_dir,
             "--schedule-cache", args.schedule_cache,
             "--out", str(Path(args.out_dir) / "q_ablation.npz")])

    print(f"\n[run_all] done. runs: \n  {open_loop_dir}\n  {inv_dir}\n  {mpc_dir}")
    if not args.skip_plots:
        print(f"[run_all] comparison plots: {Path(args.out_dir) / 'comparison_plots'}")


if __name__ == "__main__":
    main()
