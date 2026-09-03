#!/usr/bin/env python3
"""Layer 1 analysis — offline inverse configuration.

What this layer does
--------------------
For every arc-length node ``s`` along the lumen centreline it solves a small
weighted least-squares problem in ``chi = [q1..q6, L]``:

    minimise  ||(p(chi) - p*(s)) / tol_p||^2
            + ||(t(chi) - t*(s)) / (2 sin(tol_t / 2))||^2
            + ||sqrt(w_c) (chi - chi_prev) / step||^2
            + ||sqrt(w_0) (chi - chi_centre) / half_span||^2

subject to a box that is the intersection of the absolute joint limits with a
trust region around the previous node, plus — when an exclusion radius is
configured — a hard source-magnet/lumen keep-out inequality.

Two consequences drive everything downstream and this script measures both:

1. Task error is a **cost**, so the solver drives it far below tolerance.
   That is why L1 typically spends only a small fraction of its error budget.
2. Each accepted node is **committed** into the beam solver before the next
   node is solved, so the path is history dependent.  Step halving on failure,
   node-to-node jumps, and Jacobian conditioning are the observable symptoms of
   that continuation struggling, and they are all reported here.

Usage
-----
    python analyse_layer1_inverse.py --inverse-dir results/offline_inverse_configuration_60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from ._stack_common import *  # noqa: F403
    from . import _stack_common as C
except ImportError:  # run as a plain script
    import _stack_common as C
    from _stack_common import *  # noqa: F403


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inverse-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def analyse_inverse_layer(inverse_dir: Path) -> tuple[C.Report, dict]:
    inverse_dir = Path(inverse_dir).expanduser().resolve()
    table = C.read_table(inverse_dir / "inverse_configuration_path.csv")
    payload = C.read_json(inverse_dir / "inverse_configuration_summary.json")
    configuration = dict(payload.get("configuration", {}))
    summary = dict(payload.get("summary", {}))

    position_tolerance = float(configuration.get("position_tolerance_m", math.nan))
    tangent_tolerance = float(configuration.get("tangent_tolerance_rad", math.nan))
    if not np.isfinite(position_tolerance) or not np.isfinite(tangent_tolerance):
        raise ValueError(
            "inverse_configuration_summary.json does not carry finite "
            "position_tolerance_m / tangent_tolerance_rad."
        )

    s = table.column("s_m")
    states = table.state_matrix("q{i}_rad", "insertion_m")
    if states is None:
        raise ValueError("Inverse CSV is missing q1_rad..q6_rad / insertion_m.")
    position_error = table.column("position_error_m")
    tangent_error = table.column("tangent_error_rad")
    feasible = table.column("feasible", 1.0) > 0.5
    solver_success = table.column("solver_success", 1.0) > 0.5

    report = C.Report("Layer 1 — offline inverse configuration")
    metrics: dict = {
        "layer": 1,
        "name": "offline_inverse_configuration",
        "source_dir": str(inverse_dir),
        "node_count": int(table.row_count),
        "position_tolerance_m": position_tolerance,
        "tangent_tolerance_rad": tangent_tolerance,
        "summary_from_planner": summary,
    }

    # ---------------------------------------------------------------- task
    position_budget = C.budget_report(
        "tip position", position_error, position_tolerance, unit="um", display_scale=1e6
    )
    tangent_budget = C.budget_report(
        "tip tangent", tangent_error, tangent_tolerance, unit="deg",
        display_scale=180.0 / math.pi,
    )
    metrics["position_budget"] = position_budget.as_dict()
    metrics["tangent_budget"] = tangent_budget.as_dict()

    report.head("What the layer achieved against its tolerances")
    report.text(
        "Task error is an objective term here, not a constraint, so a healthy "
        "L1 solve sits well inside its budget. A budget near 1 means the "
        "least-squares solve is saturating and the node is only marginally "
        "reachable."
    )
    report.table(
        ["quantity", "tolerance", "max", "p95", "rms", "budget p95", "over tolerance"],
        [
            [
                "tip position",
                f"{1e6 * position_tolerance:.1f} um",
                f"{1e6 * position_budget.maximum:.2f} um",
                f"{1e6 * position_budget.p95:.2f} um",
                f"{1e6 * position_budget.rms:.2f} um",
                f"{position_budget.utilisation_p95:.2f}x",
                f"{100 * position_budget.over_budget_fraction:.1f}%",
            ],
            [
                "tip tangent",
                f"{math.degrees(tangent_tolerance):.1f} deg",
                f"{math.degrees(tangent_budget.maximum):.2f} deg",
                f"{math.degrees(tangent_budget.p95):.2f} deg",
                f"{math.degrees(tangent_budget.rms):.2f} deg",
                f"{tangent_budget.utilisation_p95:.2f}x",
                f"{100 * tangent_budget.over_budget_fraction:.1f}%",
            ],
        ],
    )
    if position_budget.over_budget_fraction > 0:
        report.finding(
            "fail",
            "Some nodes exceed the position tolerance",
            f"{100 * position_budget.over_budget_fraction:.1f}% of nodes are outside "
            f"{1e6 * position_tolerance:.1f} um (worst {1e6 * position_budget.maximum:.1f} um). "
            "A downstream global solve that treats this tolerance as a hard constraint "
            "starts from an infeasible seed, which disables its feasible-seed fallback.",
        )
    else:
        report.finding(
            "ok",
            "Every node is inside the position tolerance",
            f"worst {1e6 * position_budget.maximum:.2f} um of "
            f"{1e6 * position_tolerance:.1f} um.",
        )
    if tangent_budget.utilisation_max < 0.5:
        report.finding(
            "info",
            "The tangent tolerance is slack",
            f"peak tangent error is only {tangent_budget.utilisation_max:.2f} of the "
            f"{math.degrees(tangent_tolerance):.0f} deg allowance, so the task is "
            "effectively position-constrained only. Anything downstream that "
            "trades against 'the task tolerance' is really trading against position.",
        )

    # -------------------------------------------------------- feasibility
    infeasible = np.flatnonzero(~feasible)
    metrics["feasibility"] = {
        "all_nodes_feasible": bool(np.all(feasible)),
        "feasible_fraction": C.fraction(feasible),
        "solver_success_fraction": C.fraction(solver_success),
        "first_infeasible_node": int(infeasible[0]) if infeasible.size else None,
        "infeasible_node_indices": infeasible.tolist(),
        "reached_end": summary.get("all_nodes_feasible"),
        "termination_reason": summary.get("termination_reason"),
    }
    if infeasible.size:
        report.finding(
            "fail",
            f"{infeasible.size} node(s) infeasible",
            f"first at node {int(infeasible[0])} (s = {s[infeasible[0]]:.4f} m). "
            "The global layer can only run in refine_complete mode on a fully "
            "feasible seed.",
        )

    # ------------------------------------------------------ continuation
    report.head("Continuation health")
    ds = np.diff(s)
    step_stats = {
        "median_step_m": float(np.median(ds)) if ds.size else math.nan,
        "minimum_step_m": C.minimum(ds),
        "maximum_step_m": C.maximum(ds),
        "step_reductions": int(np.sum(ds < 0.999 * np.median(ds))) if ds.size else 0,
    }
    delta_q = table.column("delta_q_norm_rad", math.nan)
    delta_L = table.column("delta_insertion_m", math.nan)
    attempts = table.column("attempt_count", math.nan)
    joint_step_limit = np.asarray(
        configuration.get("maximum_joint_step_rad", [math.nan] * 6), dtype=float
    )
    insertion_step_limit = float(
        configuration.get("maximum_insertion_step_m", math.nan)
    )
    metrics["continuation"] = {
        **step_stats,
        "delta_q_norm_max_rad": C.maximum(delta_q),
        "delta_insertion_max_m": C.maximum(delta_L),
        "mean_attempts_per_node": float(np.nanmean(attempts)) if attempts.size else math.nan,
        "max_attempts": C.maximum(attempts),
        "multistart_nodes": int(np.sum(C.finite(attempts) > 1)),
        "joint_step_limit_rad": joint_step_limit.tolist(),
        "insertion_step_limit_m": insertion_step_limit,
        "joint_step_utilisation_max": (
            C.maximum(delta_q) / float(np.min(joint_step_limit))
            if np.all(np.isfinite(joint_step_limit)) and np.min(joint_step_limit) > 0
            else math.nan
        ),
        "insertion_step_utilisation_max": (
            C.maximum(delta_L) / insertion_step_limit
            if np.isfinite(insertion_step_limit) and insertion_step_limit > 0
            else math.nan
        ),
    }
    report.bullets(
        [
            f"Arc-length step: median {1e3 * step_stats['median_step_m']:.3f} mm, "
            f"range {1e3 * step_stats['minimum_step_m']:.3f}–{1e3 * step_stats['maximum_step_m']:.3f} mm "
            f"({step_stats['step_reductions']} shortened steps, i.e. retries after a failed node).",
            f"Largest node-to-node joint jump {C.maximum(delta_q):.4f} rad, insertion jump "
            f"{1e3 * C.maximum(delta_L):.3f} mm.",
            f"Multistart was needed at {metrics['continuation']['multistart_nodes']} node(s); "
            f"worst case {C.maximum(attempts):.0f} attempts.",
        ]
    )
    if metrics["continuation"]["multistart_nodes"] > 0.1 * table.row_count:
        report.finding(
            "warn",
            "Frequent multistart",
            f"{metrics['continuation']['multistart_nodes']} of {table.row_count} nodes needed "
            "more than one initial guess, which usually means the continuation is being "
            "pushed near a fold or a singularity.",
        )

    # ------------------------------------------------------ conditioning
    condition = table.column("jacobian_condition", math.nan)
    rank = table.column("jacobian_effective_rank", math.nan)
    metrics["conditioning"] = {
        "condition_max": C.maximum(condition),
        "condition_median": C.percentile(condition, 50.0),
        "condition_p95": C.percentile(condition, 95.0),
        "minimum_effective_rank": C.minimum(rank),
        "rank_deficient_nodes": int(np.sum(C.finite(rank) < 5)),
    }
    if metrics["conditioning"]["rank_deficient_nodes"]:
        report.finding(
            "warn",
            "Rank-deficient task Jacobians",
            f"{metrics['conditioning']['rank_deficient_nodes']} node(s) have effective rank "
            "below 5 on the tolerance-scaled task Jacobian. Note the tangent block is "
            "projected onto the unit sphere, so rank 5 (not 6) is the healthy maximum.",
        )

    # --------------------------------------------------------- exclusion
    margin = table.column("source_magnet_lumen_margin_m", math.nan)
    radius = table.column("source_magnet_lumen_exclusion_radius_m", math.nan)
    satisfied = table.column("source_magnet_lumen_constraint_satisfied", 1.0) > 0.5
    exclusion_enabled = bool(np.any(np.isfinite(margin)))
    metrics["head_exclusion"] = {
        "enabled": exclusion_enabled,
        "radius_m": C.percentile(radius, 50.0) if exclusion_enabled else None,
        "minimum_margin_m": C.minimum(margin) if exclusion_enabled else None,
        "violating_nodes": int(np.sum(~satisfied)),
        "active_nodes": (
            int(np.sum(C.finite(margin) <= 1e-9)) if exclusion_enabled else 0
        ),
    }
    if exclusion_enabled:
        active = metrics["head_exclusion"]["active_nodes"]
        report.head("Source-magnet / lumen keep-out")
        report.bullets(
            [
                f"Exclusion radius {1e3 * metrics['head_exclusion']['radius_m']:.1f} mm.",
                f"Minimum margin {1e3 * metrics['head_exclusion']['minimum_margin_m']:.4f} mm "
                f"across {table.row_count} nodes.",
                f"{active} node(s) sit on the constraint boundary (margin <= 1 nm).",
            ]
        )
        if active > 0:
            report.finding(
                "info",
                "The keep-out constraint is load-bearing",
                f"{active} node(s) are exactly on the boundary, so this constraint is "
                "actively shaping the L1 solution. Any later layer that drops it will "
                "move the magnet into the lumen.",
            )
        if metrics["head_exclusion"]["violating_nodes"]:
            report.finding(
                "fail",
                "Keep-out violated at some nodes",
                f"{metrics['head_exclusion']['violating_nodes']} node(s).",
            )

    # --------------------------------------------- geometric path quality
    scale = C.DEFAULT_CONFIGURATION_SCALE
    _, first, second, delta = C.path_derivatives(s, states, scale)
    metrics["path_quality"] = {
        "configuration_scale": scale.tolist(),
        "max_abs_first": C.max_abs(first, axis=0).tolist() if first.size else [],
        "rms_first": np.sqrt(np.mean(first * first, axis=0)).tolist() if first.size else [],
        "max_abs_second": C.max_abs(second, axis=0).tolist() if second.size else [],
        "rms_second": np.sqrt(np.mean(second * second, axis=0)).tolist() if second.size else [],
        "first_norm_max": C.maximum(np.linalg.norm(first, axis=1)) if first.size else math.nan,
        "second_norm_max": C.maximum(np.linalg.norm(second, axis=1)) if second.size else math.nan,
        "scaled_path_length": (
            float(np.sum(np.linalg.norm(delta / scale[None, :], axis=1))) if delta.size else 0.0
        ),
    }
    report.head("Geometric path quality (time-parameterisation proxies)")
    report.text(
        "A time parameteriser sees `q_dot = q'(s) s_dot` and "
        "`q_ddot = q''(s) s_dot^2 + q'(s) s_ddot`, so these per-coordinate "
        "derivatives — not the tip path — set how fast the path can be traversed."
    )
    report.table(
        ["coordinate", "max |dchi/ds| (scaled)", "rms", "max |d2chi/ds2| (scaled)", "rms"],
        [
            [
                C.COORDINATE_NAMES[i],
                metrics["path_quality"]["max_abs_first"][i],
                metrics["path_quality"]["rms_first"][i],
                metrics["path_quality"]["max_abs_second"][i],
                metrics["path_quality"]["rms_second"][i],
            ]
            for i in range(states.shape[1])
        ],
    )
    peak_first = np.asarray(metrics["path_quality"]["max_abs_first"], dtype=float)
    if peak_first.size == 7:
        dominant = int(np.argmax(peak_first))
        report.finding(
            "info",
            f"{C.COORDINATE_NAMES[dominant]} dominates the first-derivative demand",
            f"peak scaled |dchi/ds| = {peak_first[dominant]:.1f}, "
            f"{peak_first[dominant] / max(1e-12, np.median(peak_first)):.0f}x the median "
            "coordinate. This is the axis most likely to bind during time "
            "parameterisation, so it is the one whose smoothness matters most.",
        )

    # ------------------------------------------------------- tip geometry
    tip = table.vector("tip")
    desired = table.vector("desired")
    if tip is not None:
        metrics["tip_path"] = C.polyline_jaggedness(tip, s)
        if desired is not None:
            metrics["desired_path"] = C.polyline_jaggedness(desired, s)
    metrics["arrays"] = {
        "s_m": s.tolist(),
        "position_error_m": position_error.tolist(),
        "tangent_error_rad": tangent_error.tolist(),
    }
    report.metrics = metrics
    return report, metrics


def _plots(metrics: dict, table: C.Table, output_dir: Path, dpi: int) -> None:
    figure, axes = C.new_figure(3, 1, (12, 10))
    if figure is None:
        return
    s = table.column("s_m")
    tol = metrics["position_tolerance_m"]
    axes[0].plot(s, 1e6 * table.column("position_error_m"), color=C.PALETTE["achieved"])
    axes[0].axhline(1e6 * tol, color=C.PALETTE["limit"], linestyle=":", label="tolerance")
    axes[0].set_ylabel("tip position error [um]")
    axes[0].set_title("L1 task error against tolerance")
    axes[0].legend()
    axes[1].plot(
        s, np.degrees(table.column("tangent_error_rad")), color=C.PALETTE["achieved"]
    )
    axes[1].axhline(
        math.degrees(metrics["tangent_tolerance_rad"]),
        color=C.PALETTE["limit"], linestyle=":", label="tolerance",
    )
    axes[1].set_ylabel("tip tangent error [deg]")
    axes[1].legend()
    margin = table.column("source_magnet_lumen_margin_m", math.nan)
    if np.any(np.isfinite(margin)):
        axes[2].plot(s, 1e3 * margin, color=C.PALETTE["magnet"])
        axes[2].axhline(0.0, color=C.PALETTE["limit"], linestyle=":")
        axes[2].set_ylabel("keep-out margin [mm]")
    axes[2].set_xlabel("lumen arc length s [m]")
    C.save_figure(figure, output_dir / "layer1_task_and_constraints.png", dpi)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir or (args.inverse_dir / "stack_analysis")
    report, metrics = analyse_inverse_layer(args.inverse_dir)
    table = C.read_table(Path(args.inverse_dir) / "inverse_configuration_path.csv")
    paths = report.write(Path(output_dir), "layer1_inverse")
    if not args.no_plots:
        _plots(metrics, table, Path(output_dir), args.dpi)
    print(f"Layer 1 analysis -> {paths['report']}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
