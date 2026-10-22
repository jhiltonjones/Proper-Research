from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
)
from proper_research.simulation.simulations.controller_factory_optimized import (
    build_controller_optimized,
)
from proper_research.simulation.simulations.initial_conditions import (
    make_initial_poses,
)
from proper_research.simulation.simulations.logging_utils import (
    setup_output_dirs,
)
from proper_research.simulation.simulations.model_factory_optimized import (
    build_model_bundle_optimized,
)
from proper_research.simulation.simulations.runner_optimized import (
    run_simulation_optimized,
)


def run_experiment_optimized(
    exp_cfg,
    design_cfg: ControllerDesignConfig | None = None,
    *,
    sensitivity_workers: int = 1,
    result_detail: str = "contact",
    save_plots: bool = False,
    full_logging: bool = False,
    validate_nonlinear_candidate: bool = False,
    collect_full_diagnostics: bool = False,
    verbose: bool = True,
):
    """
    Run one fixed-mode experiment with the optimized model and controller.

    The experiment's solver_mode remains fixed, so separate LTI, LTV one-shot
    and SQP-full configurations remain separate research conditions.
    """
    exp_cfg.validate()
    design_cfg = design_cfg or ControllerDesignConfig()

    if getattr(exp_cfg.controller, "controller_kind", "mpc") != "mpc":
        raise ValueError(
            "The optimized simulation entry currently supports MPC experiments."
        )

    out_root = Path(exp_cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pivot_point, start_point, L0, dt = make_initial_poses()
    bundle = build_model_bundle_optimized(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=exp_cfg.lumen,
        plant_contact=bool(exp_cfg.model.plant_contact),
        result_detail=result_detail,
        store_history=False,
        store_vectors_in_info=False,
        sensitivity_workers=int(sensitivity_workers),
        copy_cached_results=True,
    )

    plant_model = bundle.models["plant"]
    jacobian_model = bundle.models[exp_cfg.model.jacobian_variant]

    controller_pack = build_controller_optimized(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant_model,
        jacobian_model=jacobian_model,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=design_cfg,
        validate_nonlinear_candidate=validate_nonlinear_candidate,
        collect_full_diagnostics=collect_full_diagnostics,
        verbose_controller=False,
    )

    controller = controller_pack["controller"]
    p0 = controller_pack["p0"]
    p_min = controller_pack["p_min"]
    p_max = controller_pack["p_max"]
    u_max = controller_pack["u_max"]
    forward6d = controller_pack["forward6d_plant"]

    frames_dir, _ = setup_output_dirs(
        out_root=out_root,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        mpc=controller,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        save_plots=save_plots,
    )
    log_csv_path = out_root / "optimized_log.csv"
    if log_csv_path.exists():
        log_csv_path.unlink()

    metadata = {
        "run_name": str(exp_cfg.run_name),
        "solver_mode": str(exp_cfg.controller.solver_mode),
        "Np": int(exp_cfg.controller.Np),
        "N_sqp": int(exp_cfg.controller.N_sqp),
        "rollout_steps": int(exp_cfg.controller.rollout_steps),
        "sensitivity_workers": int(sensitivity_workers),
        "result_detail": result_detail,
        "validate_nonlinear_candidate": bool(
            validate_nonlinear_candidate
        ),
        "collect_full_diagnostics": bool(collect_full_diagnostics),
        "reference_mode": str(design_cfg.reference_mode),
        "q_contour": float(design_cfg.q_contour),
        "q_lag": float(design_cfg.q_lag),
        "contouring_axis_weights": list(design_cfg.contouring_axis_weights),
        "ref_lookahead_m": float(design_cfg.ref_lookahead_m),
        "ref_stride_m": float(design_cfg.ref_stride_m),
        "ref_search_backward_m": float(design_cfg.ref_search_backward_m),
        "ref_search_forward_m": float(design_cfg.ref_search_forward_m),
        "ref_max_progress_per_step_m": float(
            design_cfg.ref_max_progress_per_step_m
        ),
        "progress_constraint_mode": design_cfg.progress_constraint_mode,
        "enable_progress_slack": bool(design_cfg.enable_progress_slack),
        "enable_hard_progress_constraint": bool(
            design_cfg.enable_hard_progress_constraint
        ),
        "progress_request_m": float(design_cfg.progress_request_m),
        "progress_reward_weight": float(design_cfg.progress_reward_weight),
        "progress_slack_quadratic_weight": float(
            design_cfg.progress_slack_quadratic_weight
        ),
        "progress_slack_linear_weight": float(
            design_cfg.progress_slack_linear_weight
        ),
        "progress_slack_max_fraction": float(
            design_cfg.progress_slack_max_fraction
        ),
        "pid": int(os.getpid()),
    }
    with (out_root / "optimized_run_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    history = run_simulation_optimized(
        mpc=controller,
        forward6d=forward6d,
        p0_ur=bundle.p0_ur,
        p0=p0,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        lumen_path=bundle.lumen_path,
        s_path=bundle.s_path,
        frames_dir=frames_dir,
        log_csv_path=log_csv_path,
        max_steps=int(exp_cfg.controller.max_steps),
        rollout_steps=int(exp_cfg.controller.rollout_steps),
        solver_mode=str(exp_cfg.controller.solver_mode),
        save_plots=save_plots,
        full_logging=full_logging,
        verbose=verbose,
        enable_tip_tangent_stop=True,
        tip_tangent_stop_deg=40.0,
    )
    return {
        "history": history,
        "controller_pack": controller_pack,
        "bundle": bundle,
        "log_csv_path": log_csv_path,
    }


def run_experiment_grid_optimized(
    experiments: Iterable,
    design_cfg: ControllerDesignConfig | None = None,
    **run_kwargs,
):
    """Run an existing iterable of ExperimentConfig objects sequentially."""
    results = []
    for experiment in experiments:
        results.append(
            run_experiment_optimized(
                experiment,
                design_cfg=design_cfg,
                **run_kwargs,
            )
        )
    return results


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--factory must have the form package.module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an optimized magnetic-beam MPC experiment without the "
            "hierarchical controller."
        )
    )
    parser.add_argument(
        "--factory",
        required=True,
        help=(
            "Python factory package.module:function returning either exp_cfg "
            "or (exp_cfg, design_cfg)."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--result-detail",
        choices=("none", "contact", "full"),
        default="contact",
    )
    parser.add_argument(
        "--validate-candidate",
        action="store_true",
        help="Enable expensive nonlinear post-QP candidate validation.",
    )
    parser.add_argument("--full-logging", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    produced = _load_factory(args.factory)()
    if isinstance(produced, tuple):
        exp_cfg, design_cfg = produced
    else:
        exp_cfg = produced
        design_cfg = ControllerDesignConfig()

    run_experiment_optimized(
        exp_cfg,
        design_cfg=design_cfg,
        sensitivity_workers=args.workers,
        result_detail=args.result_detail,
        save_plots=args.save_plots,
        full_logging=args.full_logging,
        validate_nonlinear_candidate=args.validate_candidate,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
