"""Unified single-rep live launcher for the 2x2 {SJ,FJ} x {MPC,INV-7} grid.

Wraps whichever of `run_mpc_delay_aware_insertion_anchor.py` /
`run_inv_7dof_delay_aware.py` is selected, after applying the z-raise patch
(if any) and picking the SJ or FJ schedule file. Always run through
`run_2x2_grid.sh`, which resets insertion before each call -- do not call
this directly back-to-back without a reset in between (insertion drifts
across runs).

Must run each invocation as a fresh process (see the module docstring note
below) -- `python3 -m ...run_controller_rep --controller mpc --jacobian fj ...`

Usage:
    python3 -m proper_research.hardware.online.zshift_grid_toolkit.run_controller_rep \\
        --controller {mpc,inv7} --jacobian {sj,fj} \\
        --plan-dir <plan_dir> --sj-schedule <sj.npy> --fj-schedule <fj.npy> \\
        --out-dir close_loop_logs/myrun --run-name-prefix ushape_zraise30mm \\
        --rep 1 --zraise-mm 30 --insertion-tol-mm 0.3
"""
import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--controller", required=True, choices=["mpc", "inv7"])
    p.add_argument("--jacobian", required=True, choices=["sj", "fj"])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--sj-schedule", required=True)
    p.add_argument("--fj-schedule", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name-prefix", required=True,
                    help="run directories are named <prefix>_<controller>_<jacobian>_rep<rep>_<timestamp>")
    p.add_argument("--rep", required=True)
    p.add_argument("--zraise-mm", type=float, default=0.0)
    p.add_argument("--insertion-tol-mm", type=float, default=0.3)
    p.add_argument("--kp", type=float, default=1.0, help="INV-7 only")
    args = p.parse_args()

    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(args.zraise_mm)

    schedule_path = args.sj_schedule if args.jacobian == "sj" else args.fj_schedule
    run_name = f"{args.run_name_prefix}_{args.controller}_{args.jacobian.upper()}_rep{args.rep}"

    if args.controller == "mpc":
        sys.argv = [
            "run_mpc_delay_aware_insertion_anchor.py",
            "--plan-dir", args.plan_dir,
            "--out-dir", args.out_dir,
            "--run-name", run_name,
            "--schedule-cache", schedule_path,
            "--insertion-tol-mm", str(args.insertion_tol_mm),
        ]
        print(f"[run_controller_rep] argv: {sys.argv}")
        from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import main as run_main
    else:
        sys.argv = [
            "run_inv_7dof_delay_aware.py",
            "--plan-dir", args.plan_dir,
            "--out-dir", args.out_dir,
            "--run-name", run_name,
            "--schedule-cache", schedule_path,
            "--insertion-tol-mm", str(args.insertion_tol_mm),
            "--kp", str(args.kp),
        ]
        print(f"[run_controller_rep] argv: {sys.argv}")
        from proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware import main as run_main

    run_main()


if __name__ == "__main__":
    main()
