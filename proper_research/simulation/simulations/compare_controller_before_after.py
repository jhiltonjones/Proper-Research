from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np

from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
    build_controller as build_legacy_controller,
)
from proper_research.simulation.simulations.controller_factory_optimized import (
    build_controller_optimized,
)
from proper_research.simulation.simulations.initial_conditions import (
    make_initial_poses,
)
from proper_research.simulation.simulations.model_factory_optimized import (
    build_model_bundle_optimized,
)


def _load_factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _run_controller(
    controller,
    *,
    steps: int,
    solver_mode: str,
    rollout_steps: int,
    suppress_stdout: bool,
):
    rows = []
    sink = io.StringIO()
    output_context = (
        contextlib.redirect_stdout(sink)
        if suppress_stdout
        else contextlib.nullcontext()
    )
    with output_context:
        for k in range(int(steps)):
            started = time.perf_counter()
            p, x, info = controller.step(
                x_meas=None,
                rollout_steps=int(rollout_steps),
                solver_mode=solver_mode,
            )
            wall_s = time.perf_counter() - started
            u0 = np.asarray(
                info.get("u0", np.full(controller.m, np.nan)),
                dtype=float,
            ).reshape(-1)
            rows.append(
                {
                    "k": k,
                    "wall_s": wall_s,
                    "status": str(info.get("status", "")),
                    "infeasible": int(info.get("infeasible", 0)),
                    "p": np.asarray(p, dtype=float).copy(),
                    "x": np.asarray(x, dtype=float).copy(),
                    "u0": u0.copy(),
                    "forward_calls": int(info.get("forward_calls_step", -1)),
                    "jacobian_calls": int(info.get("jacobian_calls_step", -1)),
                    "qp_build_s": float(
                        info.get("qp_build_time_total_s", np.nan)
                    ),
                    "osqp_s": float(info.get("osqp_time_total_s", np.nan)),
                }
            )
            if int(info.get("infeasible", 0)):
                break
    return rows


def compare_controllers(
    exp_cfg,
    design_cfg: ControllerDesignConfig | None = None,
    *,
    sensitivity_workers: int = 1,
    steps: int = 20,
    output_dir: Path = Path("controller_before_after"),
    suppress_legacy_stdout: bool = True,
):
    """
    Compare legacy and optimized controllers while both use the optimized
    forward and Jacobian models.
    """
    exp_cfg.validate()
    design_cfg = design_cfg or ControllerDesignConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot_point, start_point, L0, dt = make_initial_poses()
    bundle = build_model_bundle_optimized(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=exp_cfg.lumen,
        plant_contact=bool(exp_cfg.model.plant_contact),
        sensitivity_workers=int(sensitivity_workers),
        result_detail="contact",
        store_history=False,
        store_vectors_in_info=False,
    )
    plant = bundle.models["plant"]
    jacobian = bundle.models[exp_cfg.model.jacobian_variant]

    legacy_pack = build_legacy_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant,
        jacobian_model=jacobian,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=design_cfg,
    )
    optimized_pack = build_controller_optimized(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant,
        jacobian_model=jacobian,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=design_cfg,
        validate_nonlinear_candidate=False,
        collect_full_diagnostics=False,
    )

    solver_mode = str(exp_cfg.controller.solver_mode)
    rollout_steps = int(exp_cfg.controller.rollout_steps)

    legacy = _run_controller(
        legacy_pack["controller"],
        steps=steps,
        solver_mode=solver_mode,
        rollout_steps=rollout_steps,
        suppress_stdout=suppress_legacy_stdout,
    )
    optimized = _run_controller(
        optimized_pack["controller"],
        steps=steps,
        solver_mode=solver_mode,
        rollout_steps=rollout_steps,
        suppress_stdout=False,
    )

    count = min(len(legacy), len(optimized))
    output_rows = []
    for k in range(count):
        before = legacy[k]
        after = optimized[k]
        output_rows.append(
            {
                "k": k,
                "legacy_wall_ms": 1.0e3 * before["wall_s"],
                "optimized_wall_ms": 1.0e3 * after["wall_s"],
                "speedup": before["wall_s"] / max(after["wall_s"], 1.0e-15),
                "tip_difference_m": float(
                    np.linalg.norm(before["x"][:3] - after["x"][:3])
                ),
                "pose_difference": float(
                    np.linalg.norm(before["p"] - after["p"])
                ),
                "u0_difference_inf": float(
                    np.max(np.abs(before["u0"] - after["u0"]))
                ),
                "legacy_status": before["status"],
                "optimized_status": after["status"],
                "legacy_infeasible": before["infeasible"],
                "optimized_infeasible": after["infeasible"],
                "optimized_qp_build_ms": 1.0e3 * after["qp_build_s"],
                "optimized_osqp_ms": 1.0e3 * after["osqp_s"],
                "optimized_forward_calls": after["forward_calls"],
                "optimized_jacobian_calls": after["jacobian_calls"],
            }
        )

    csv_path = output_dir / "controller_before_after.csv"
    if output_rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(output_rows[0]),
            )
            writer.writeheader()
            writer.writerows(output_rows)

    legacy_times = [row["wall_s"] for row in legacy[1:]] or [
        row["wall_s"] for row in legacy
    ]
    optimized_times = [row["wall_s"] for row in optimized[1:]] or [
        row["wall_s"] for row in optimized
    ]
    median_legacy = statistics.median(legacy_times)
    median_optimized = statistics.median(optimized_times)
    summary = {
        "solver_mode": solver_mode,
        "steps_compared": count,
        "sensitivity_workers": int(sensitivity_workers),
        "legacy_median_ms": 1.0e3 * median_legacy,
        "optimized_median_ms": 1.0e3 * median_optimized,
        "median_speedup": median_legacy / max(median_optimized, 1.0e-15),
        "maximum_tip_difference_m": max(
            (row["tip_difference_m"] for row in output_rows),
            default=np.nan,
        ),
        "maximum_u0_difference_inf": max(
            (row["u0_difference_inf"] for row in output_rows),
            default=np.nan,
        ),
        "legacy_infeasible_steps": sum(
            row["infeasible"] for row in legacy
        ),
        "optimized_infeasible_steps": sum(
            row["infeasible"] for row in optimized
        ),
    }
    with (output_dir / "controller_before_after_summary.json").open(
        "w"
    ) as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factory",
        required=True,
        help=(
            "package.module:function returning exp_cfg or "
            "(exp_cfg, design_cfg)"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("controller_before_after"),
    )
    parser.add_argument(
        "--show-legacy-output",
        action="store_true",
    )
    args = parser.parse_args()

    produced = _load_factory(args.factory)()
    if isinstance(produced, tuple):
        exp_cfg, design_cfg = produced
    else:
        exp_cfg = produced
        design_cfg = ControllerDesignConfig()

    compare_controllers(
        exp_cfg,
        design_cfg,
        sensitivity_workers=args.workers,
        steps=args.steps,
        output_dir=args.output_dir,
        suppress_legacy_stdout=not args.show_legacy_output,
    )


if __name__ == "__main__":
    main()
