"""Open-loop (feedforward-only) sanity-check launcher, z-raise aware.

Run this FIRST for any new plan, before spending time on SJ/FJ schedules or
closed-loop reps: it's the fast, cheap gate that confirms the raised (or
unraised) plan tracks at a similar RMS to the historical baseline for that
shape before committing to the full 2x2 grid.

Usage:
    python3 -m proper_research.hardware.online.zshift_grid_toolkit.run_open_loop \\
        --plan-dir plans/ushape_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --run-name ushape_zraise30mm_openloop \\
        --zraise-mm 30 --insertion-tol-mm 0.3
"""
import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--zraise-mm", type=float, default=0.0)
    p.add_argument("--insertion-tol-mm", type=float, default=0.3)
    args = p.parse_args()

    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(args.zraise_mm)

    sys.argv = [
        "run_open_loop_c.py",
        "--plan-dir", args.plan_dir,
        "--out-dir", args.out_dir,
        "--run-name", args.run_name,
        "--insertion-tol-mm", str(args.insertion_tol_mm),
    ]
    print(f"[run_open_loop] argv: {sys.argv}")
    from proper_research.hardware.online.rectangle_stage_a.run_open_loop_c import main as run_main
    run_main()


if __name__ == "__main__":
    main()
