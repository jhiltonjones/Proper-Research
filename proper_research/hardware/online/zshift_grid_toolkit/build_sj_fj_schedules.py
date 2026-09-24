"""Build the genuine (SJ) and frozen (FJ) Jacobian schedules for a plan.

SJ = one genuine `from_model_bundle(jacobian_mode="accurate")` beam solve
per reference sample (the honest, per-tick-varying linearization).
FJ = SJ's own index-0 value repeated across every sample
(`np.repeat(sj[0:1], N, axis=0)`) -- i.e. "the model never updates after
the first tick." This is the exact FJ construction used throughout the
raised-rectangle z-shift study, kept identical here for comparability.

Usage:
    python3 -m proper_research.hardware.online.zshift_grid_toolkit.build_sj_fj_schedules \\
        --plan-dir plans/ushape_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path \\
        --sj-out /tmp/ushape_zraise30mm_schedule_SJ.npy \\
        --fj-out /tmp/ushape_zraise30mm_schedule_FJ.npy \\
        --zraise-mm 30

Takes ~15-20 minutes for a ~200-sample plan (one beam solve per sample).
"""
import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--sj-out", required=True)
    p.add_argument("--fj-out", required=True)
    p.add_argument("--zraise-mm", type=float, default=0.0,
                    help="must match the value used to build the plan itself (0 for an unraised plan)")
    args = p.parse_args()

    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(args.zraise_mm)

    from proper_research.controllers import mpc_variants
    from proper_research.controllers.beam_jacobian_providers import from_model_bundle
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )

    print(f"[build_sj_fj_schedules] plan_dir={args.plan_dir} zraise_mm={args.zraise_mm}")
    reference = load_configuration_reference(args.plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate"
    )

    print(f"[build_sj_fj_schedules] building genuine SJ schedule "
          f"({reference.sample_count} beam solves, ~5-8s each)...")
    sj = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True
    )
    np.save(args.sj_out, sj)
    print(f"[build_sj_fj_schedules] SJ schedule {sj.shape} saved -> {args.sj_out}")

    fj = np.repeat(sj[0:1], sj.shape[0], axis=0)
    np.save(args.fj_out, fj)
    print(f"[build_sj_fj_schedules] FJ schedule {fj.shape} saved -> {args.fj_out} "
          f"(every row == SJ row 0)")


if __name__ == "__main__":
    main()
