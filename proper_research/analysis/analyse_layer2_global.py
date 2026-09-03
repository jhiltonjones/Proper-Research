#!/usr/bin/env python3
"""Layer 2 analysis — global constrained configuration smoothing.

Why this script exists alongside `analyse_global_configuration_improvement.py`
-----------------------------------------------------------------------------
That script answers "did the configuration path get smoother?" — and it does
that well. This one answers the question that follows from it:

    the joints got smoother, so why did the beam path get worse?

The mechanism is in the formulation, not in the code. The global objective is
a pure quadratic in ``chi = [q1..q6, L]``:

    0.5 * || D1 chi ||^2_W1  +  0.5 * || D2 chi ||^2_W2
  + 0.5 * || chi - seed ||^2_Wseed  +  0.5 * || chi - centre ||^2_Wcentre

There is **no task-space term at all**. Tip position and tangent enter only as
hard inequality constraints at the nodes:

    || p(chi_j) - p*(s_j) ||  <=  position_tolerance_m
    angle( t(chi_j), t*(s_j) )  <=  tangent_tolerance_rad

A constraint is free until it binds. Any tip motion that stays inside the
tolerance ball costs the optimiser exactly nothing, while the joint smoothing
it buys pays. So the optimiser will spend the entire error budget, and the tip
migrates from wherever L1 put it (deep inside the ball, because L1 *minimised*
that error) out to the ball's surface — independently at each node. A path that
hops around inside a tolerance tube is exactly what "more jagged" looks like.

This script therefore measures four things the smoothness report does not:

1. **Budget migration** — the distribution of ``error / tolerance`` before and
   after. This is the direct, quantitative form of the jaggedness complaint.
2. **Tip-path jaggedness** — turning angles and second differences of the
   achieved tip polyline, seed vs optimised.
3. **The binding-axis trade** — which coordinates were smoothed and which were
   made worse. Smoothing is a zero-sum trade under a fixed objective, and the
   axis that pays is often the one that binds downstream.
4. **Convergence forensics and dropped constraints** — whether the returned
   path is a converged solution or an intermediate iterate, and whether any
   constraint enforced by L1 has silently disappeared at L2.

Usage
-----
    python analyse_layer2_global.py \
        --global-dir results/global_configuration_full_debug \
        --inverse-dir results/offline_inverse_configuration_60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from . import _stack_common as C
except ImportError:
    import _stack_common as C


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--global-dir", required=True, type=Path)
    parser.add_argument(
        "--inverse-dir",
        type=Path,
        default=None,
        help="Optional: enables the dropped-constraint and seed-provenance checks.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def _seed_tip_positions(table: C.Table) -> np.ndarray | None:
    """The global CSV stores seed *configurations*, not seed tip positions.

    We cannot recompute the seed's tip without the beam model, so the honest
    fallback is the inverse CSV's saved tip. Returns None when unavailable.
    """
    return None


def analyse_global_layer(
    global_dir: Path, inverse_dir: Path | None = None
) -> tuple[C.Report, dict]:
    global_dir = Path(global_dir).expanduser().resolve()
    table = C.read_table(global_dir / "global_configuration_path.csv")
    payload = C.read_json(global_dir / "global_configuration_summary.json")
    configuration = dict(payload.get("configuration", {}))
    summary = dict(payload.get("summary", {}))

    dense_path = global_dir / "global_configuration_dense_validation.csv"
    dense: C.Table | None = None
    if dense_path.exists() and dense_path.stat().st_size > 0:
        try:
            dense = C.read_table(dense_path)
        except ValueError:
            dense = None

    inverse_table = None
    inverse_configuration: dict = {}
    if inverse_dir is not None:
        inverse_dir = Path(inverse_dir).expanduser().resolve()
        inverse_table = C.read_table(inverse_dir / "inverse_configuration_path.csv")
        inverse_payload = C.read_json(inverse_dir / "inverse_configuration_summary.json")
        inverse_configuration = dict(inverse_payload.get("configuration", {}))

    position_tolerance = float(configuration.get("position_tolerance_m", math.nan))
    tangent_tolerance = float(configuration.get("tangent_tolerance_rad", math.nan))
    scale = np.asarray(
        configuration.get("configuration_scale", C.DEFAULT_CONFIGURATION_SCALE),
        dtype=float,
    ).reshape(-1)

    s = table.column("s_m")
    optimised = table.state_matrix("q{i}_rad", "insertion_m")
    seed_keys = [f"seed_q{i}" for i in range(1, 7)] + ["seed_L"]
    seed = (
        np.stack([table.columns[key] for key in seed_keys], axis=1)
        if table.has(*seed_keys)
        else None
    )
    achieved = table.vector("achieved")
    desired = table.vector("desired")
    position_error = table.column("position_error_m")
    tangent_error = table.column("tangent_error_rad")

    report = C.Report("Layer 2 — global constrained configuration smoothing")
    metrics: dict = {
        "layer": 2,
        "name": "global_constrained_configuration",
        "source_dir": str(global_dir),
        "node_count": int(table.row_count),
        "mode": configuration.get("mode"),
        "position_tolerance_m": position_tolerance,
        "tangent_tolerance_rad": tangent_tolerance,
        "summary_from_planner": summary,
    }

    # ------------------------------------------------- convergence first
    report.head("Did the solve actually converge?")
    iterations = summary.get("optimizer_iterations")
    max_iterations = configuration.get("maximum_iterations")
    violation = summary.get("normalized_constraint_violation")
    optimizer_success = summary.get("optimizer_success")
    globally_feasible = summary.get("globally_feasible")
    refinement_rounds = summary.get("refinement_rounds")
    max_rounds = configuration.get("maximum_refinement_rounds")
    dense_enabled = configuration.get("dense_validation_enabled")
    hit_iteration_cap = (
        isinstance(iterations, (int, float))
        and isinstance(max_iterations, (int, float))
        and iterations >= max_iterations
    )
    metrics["convergence"] = {
        "optimizer_success": optimizer_success,
        "globally_feasible": globally_feasible,
        "dense_validation_feasible": summary.get("dense_validation_feasible"),
        "termination_reason": summary.get("termination_reason"),
        "iterations": iterations,
        "maximum_iterations": max_iterations,
        "hit_iteration_cap": bool(hit_iteration_cap),
        "normalized_constraint_violation": violation,
        "constraint_tolerance": configuration.get("constraint_tolerance"),
        "refinement_rounds": refinement_rounds,
        "maximum_refinement_rounds": max_rounds,
        "dense_validation_enabled": dense_enabled,
        "restoration_slack_maximum": summary.get("restoration_slack_maximum"),
        "selected_candidate_feasible": summary.get("selected_candidate_feasible"),
    }
    report.table(
        ["field", "value"],
        [
            ["optimizer_success", optimizer_success],
            ["globally_feasible", globally_feasible],
            ["termination_reason", summary.get("termination_reason")],
            ["iterations / maximum", f"{iterations} / {max_iterations}"],
            ["refinement rounds / maximum", f"{refinement_rounds} / {max_rounds}"],
            ["normalized constraint violation", violation],
            ["constraint tolerance", configuration.get("constraint_tolerance")],
        ],
    )
    if hit_iteration_cap:
        report.finding(
            "fail",
            "The optimiser stopped at its iteration cap, not at a solution",
            f"trust-constr ran {iterations} of a permitted {max_iterations} iterations. "
            "What was written out is an intermediate iterate. Interior iterates of a "
            "trust-region method routinely look rough — do not diagnose path quality "
            "from a run that did not converge. Raise maximum_iterations first.",
        )
    if isinstance(violation, (int, float)) and violation > float(
        configuration.get("constraint_tolerance", 1e-8)
    ):
        report.finding(
            "fail",
            "The returned path violates its hard constraints",
            f"normalized violation {violation:.3g} against a tolerance of "
            f"{configuration.get('constraint_tolerance')}. Part of the tip error you are "
            "seeing is constraint violation, not a permitted trade inside tolerance.",
        )
    if dense_enabled is False:
        report.finding(
            "warn",
            "Dense validation was disabled",
            "globally_feasible is computed as node feasibility AND dense feasibility, "
            "and the dense term is False whenever validation is off — so globally_feasible "
            "can never be True in this configuration, regardless of solution quality.",
        )
    if isinstance(max_rounds, (int, float)) and max_rounds == 0 and dense_enabled:
        report.finding(
            "warn",
            "No refinement rounds were allowed",
            "maximum_refinement_rounds = 0, so between-node failures found by dense "
            "validation could not be repaired by inserting a node. The run terminates "
            "with 'maximum_refinement_rounds_reached' whether or not it was close.",
        )

    # --------------------------------------------- the budget migration
    report.head("Where the tip error went: budget migration")
    report.text(
        "L1 treats task error as a cost and drives it toward zero; L2 treats the "
        "same quantity as a hard constraint, which is free until it binds. The "
        "table below is the quantitative form of 'the beam path looks more jagged': "
        "the tip has moved from the middle of the tolerance ball to its surface."
    )
    after_budget = C.budget_report(
        "L2 optimised", position_error, position_tolerance, unit="um", display_scale=1e6
    )
    metrics["position_budget_after"] = after_budget.as_dict()
    rows = [
        [
            "L2 optimised (node)",
            f"{1e6 * after_budget.maximum:.2f}",
            f"{1e6 * after_budget.p95:.2f}",
            f"{1e6 * after_budget.rms:.2f}",
            f"{after_budget.utilisation_p95:.2f}x",
            f"{100 * after_budget.near_boundary_fraction:.1f}%",
            f"{100 * after_budget.over_budget_fraction:.1f}%",
        ]
    ]
    before_budget = None
    if inverse_table is not None:
        before_budget = C.budget_report(
            "L1 seed", inverse_table.column("position_error_m"),
            position_tolerance, unit="um", display_scale=1e6,
        )
        metrics["position_budget_before"] = before_budget.as_dict()
        rows.insert(
            0,
            [
                "L1 seed (node)",
                f"{1e6 * before_budget.maximum:.2f}",
                f"{1e6 * before_budget.p95:.2f}",
                f"{1e6 * before_budget.rms:.2f}",
                f"{before_budget.utilisation_p95:.2f}x",
                f"{100 * before_budget.near_boundary_fraction:.1f}%",
                f"{100 * before_budget.over_budget_fraction:.1f}%",
            ],
        )
    if dense is not None and dense.has("position_error_m"):
        dense_budget = C.budget_report(
            "L2 between nodes", dense.column("position_error_m"),
            position_tolerance, unit="um", display_scale=1e6,
        )
        metrics["position_budget_dense"] = dense_budget.as_dict()
        rows.append(
            [
                "L2 optimised (between nodes)",
                f"{1e6 * dense_budget.maximum:.2f}",
                f"{1e6 * dense_budget.p95:.2f}",
                f"{1e6 * dense_budget.rms:.2f}",
                f"{dense_budget.utilisation_p95:.2f}x",
                f"{100 * dense_budget.near_boundary_fraction:.1f}%",
                f"{100 * dense_budget.over_budget_fraction:.1f}%",
            ]
        )
    report.table(
        ["stage", "max [um]", "p95 [um]", "rms [um]", "budget p95",
         ">=80% of budget", "over tolerance"],
        rows,
    )
    if before_budget is not None and np.isfinite(before_budget.rms):
        growth = after_budget.rms / max(before_budget.rms, 1e-30)
        metrics["position_budget_rms_growth"] = float(growth)
        level = "warn" if growth > 1.5 else "info"
        report.finding(
            level,
            f"Typical tip error grew {growth:.1f}x while joints were smoothed",
            f"rms {1e6 * before_budget.rms:.2f} um -> {1e6 * after_budget.rms:.2f} um "
            f"(p95 {1e6 * before_budget.p95:.2f} -> {1e6 * after_budget.p95:.2f} um). "
            "This is the formulation working as written, not a bug: tip error inside "
            "tolerance is free, joint roughness is not.",
        )
    if dense is not None and "position_budget_dense" in metrics:
        dense_over = metrics["position_budget_dense"]["over_budget_fraction"]
        if dense_over and dense_over > 0:
            report.finding(
                "fail",
                "The path leaves tolerance between nodes",
                f"{100 * dense_over:.1f}% of dense samples exceed the position tolerance "
                "even though the constraint is imposed only at nodes. Node-only "
                "enforcement does not imply a feasible continuous path.",
            )

    # ----------------------------------------------- tip-path jaggedness
    report.head("Tip-path jaggedness")
    if achieved is not None:
        after_jag = C.polyline_jaggedness(achieved, s)
        metrics["tip_jaggedness_after"] = after_jag
        jag_rows = [
            [
                "L2 optimised",
                after_jag.get("turning_angle_rms_deg", math.nan),
                after_jag.get("turning_angle_p95_deg", math.nan),
                after_jag.get("total_absolute_turning_deg", math.nan),
                after_jag.get("tortuosity", math.nan),
            ]
        ]
        if inverse_table is not None:
            inverse_tip = inverse_table.vector("tip")
            if inverse_tip is not None:
                before_jag = C.polyline_jaggedness(
                    inverse_tip, inverse_table.column("s_m")
                )
                metrics["tip_jaggedness_before"] = before_jag
                jag_rows.insert(
                    0,
                    [
                        "L1 seed",
                        before_jag.get("turning_angle_rms_deg", math.nan),
                        before_jag.get("turning_angle_p95_deg", math.nan),
                        before_jag.get("total_absolute_turning_deg", math.nan),
                        before_jag.get("tortuosity", math.nan),
                    ],
                )
        if desired is not None:
            centre_jag = C.polyline_jaggedness(desired, s)
            metrics["tip_jaggedness_centreline"] = centre_jag
            jag_rows.append(
                [
                    "desired centreline",
                    centre_jag.get("turning_angle_rms_deg", math.nan),
                    centre_jag.get("turning_angle_p95_deg", math.nan),
                    centre_jag.get("total_absolute_turning_deg", math.nan),
                    centre_jag.get("tortuosity", math.nan),
                ]
            )
        report.text(
            "Turning angle is the angle between consecutive chords of the tip "
            "polyline. It is the metric that matches the eye: a path can have a "
            "tiny total excursion and still look ragged if it changes direction "
            "at every node. The desired centreline row is the floor — any excess "
            "over it was introduced by the planner."
        )
        report.table(
            ["path", "turning rms [deg]", "turning p95 [deg]",
             "total turning [deg]", "tortuosity"],
            jag_rows,
        )
        if "tip_jaggedness_before" in metrics:
            b = metrics["tip_jaggedness_before"].get("turning_angle_rms_deg", math.nan)
            a = metrics["tip_jaggedness_after"].get("turning_angle_rms_deg", math.nan)
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                metrics["tip_turning_growth"] = float(a / b)
                if a > 1.2 * b:
                    report.finding(
                        "warn",
                        f"Tip path is {a / b:.1f}x more jagged than the seed's",
                        f"turning-angle rms {b:.2f} deg -> {a:.2f} deg. Confirms the "
                        "complaint numerically; see the budget-migration table for why.",
                    )
                else:
                    report.finding(
                        "ok",
                        "Tip-path jaggedness is not materially worse",
                        f"turning-angle rms {b:.2f} deg -> {a:.2f} deg.",
                    )

    # ------------------------------------------------ the smoothing trade
    report.head("Which coordinates paid for the smoothing")
    if seed is not None and optimised is not None:
        _, first_before, second_before, delta_before = C.path_derivatives(s, seed, scale)
        _, first_after, second_after, delta_after = C.path_derivatives(s, optimised, scale)
        trade_rows = []
        regressions = []
        for i in range(optimised.shape[1]):
            b1 = float(C.max_abs(first_before[:, i])) if first_before.size else math.nan
            a1 = float(C.max_abs(first_after[:, i])) if first_after.size else math.nan
            b2 = float(C.max_abs(second_before[:, i])) if second_before.size else math.nan
            a2 = float(C.max_abs(second_after[:, i])) if second_after.size else math.nan
            d1 = C.improvement_percent(b1, a1)
            d2 = C.improvement_percent(b2, a2)
            trade_rows.append(
                [C.COORDINATE_NAMES[i], b1, a1, f"{d1:+.1f}%", b2, a2, f"{d2:+.1f}%"]
            )
            if np.isfinite(d2) and d2 < 0:
                regressions.append((C.COORDINATE_NAMES[i], d2, b2, a2))
        metrics["smoothing_trade"] = [
            {
                "coordinate": C.COORDINATE_NAMES[i],
                "max_abs_first_before": trade_rows[i][1],
                "max_abs_first_after": trade_rows[i][2],
                "max_abs_second_before": trade_rows[i][4],
                "max_abs_second_after": trade_rows[i][5],
            }
            for i in range(len(trade_rows))
        ]
        report.table(
            ["coordinate", "max|d1| before", "after", "change",
             "max|d2| before", "after", "change"],
            trade_rows,
        )
        # Only report a regression that is material: a coordinate whose 'before'
        # value was numerically negligible can show a huge percentage change that
        # means nothing.
        after_peaks = [row[5] for row in trade_rows if np.isfinite(row[5])]
        material = 0.05 * max(after_peaks) if after_peaks else 0.0
        regressions = [item for item in regressions if item[3] >= material]
        if regressions:
            worst = min(regressions, key=lambda item: item[1])
            report.finding(
                "warn",
                f"Second-derivative demand on {worst[0]} got worse by "
                f"{min(abs(worst[1]), 9999):.0f}%",
                f"{worst[2]:.3g} -> {worst[3]:.3g}. Under a fixed quadratic objective, "
                "smoothing is a trade: the coordinate whose term contributes least to "
                "the cost is the one that pays. Check whether this coordinate is the "
                "one that binds during time parameterisation — if it is, the smoothing "
                "made the traversal slower, not faster. Raise its "
                "second_difference_weight, or reduce its configuration_scale, to "
                "rebalance.",
            )
        metrics["seed_change"] = {
            "max_abs_change_per_coordinate": C.max_abs(
                optimised - seed, axis=0
            ).tolist(),
            "max_scaled_change_norm": float(
                np.max(np.linalg.norm((optimised - seed) / scale[None, :], axis=1))
            ),
        }

    # -------------------------------------------- constraints dropped at L2
    report.head("Constraints that exist at L1 but not at L2")
    dropped: list[str] = []
    if inverse_table is not None:
        inverse_margin = inverse_table.column("source_magnet_lumen_margin_m", math.nan)
        if np.any(np.isfinite(inverse_margin)):
            radius = C.percentile(
                inverse_table.column("source_magnet_lumen_exclusion_radius_m", math.nan),
                50.0,
            )
            global_magnet = table.vector("magnet_pose")
            inverse_desired = inverse_table.vector("desired")
            centreline = inverse_desired if inverse_desired is not None else desired
            if global_magnet is not None and centreline is not None and np.isfinite(radius):
                distance = C.nearest_polyline_distance(global_magnet, centreline)
                approx_margin = distance - radius
                metrics["head_exclusion_regression"] = {
                    "radius_m": float(radius),
                    "inverse_minimum_margin_m": C.minimum(inverse_margin),
                    "global_reconstructed_minimum_margin_m": C.minimum(approx_margin),
                    "global_violating_nodes": int(np.sum(C.finite(approx_margin) < 0.0)),
                    "reconstruction_is_approximate": True,
                }
                dropped.append("source-magnet / lumen keep-out")
                if C.minimum(approx_margin) < 0.0:
                    report.finding(
                        "fail",
                        "The keep-out constraint is violated after smoothing",
                        f"reconstructed minimum margin {1e3 * C.minimum(approx_margin):.3f} mm "
                        f"against an L1 minimum of {1e3 * C.minimum(inverse_margin):.3e} mm. "
                        "The global layer imposes no exclusion constraint at all, so a "
                        "constraint that was active and load-bearing at L1 simply "
                        "disappeared. This matters far more than the tip jaggedness: "
                        "re-impose it in the global layer, or clamp the smoothing to "
                        "states that preserve it.",
                    )
    if dropped:
        report.bullets(
            [
                f"**{name}** — enforced as a hard inequality at L1, absent from the L2 "
                "objective and constraint set."
                for name in dropped
            ]
        )
    else:
        report.text(
            "No L1 constraint diagnostics were available to cross-check; pass "
            "`--inverse-dir` to enable this section."
        )
    metrics["constraints_dropped_at_l2"] = dropped

    report.head("What to change, in order")
    report.bullets(
        [
            "Fix convergence before anything else — an unconverged iterate is not "
            "evidence about the formulation.",
            "Re-impose any constraint listed above as dropped.",
            "If the tip path still matters after that, add an explicit task-space "
            "term. The cheapest is a small quadratic pull toward the centre of the "
            "tolerance ball using the linearised tip, which costs one extra "
            "block-diagonal term in the same QP; the alternative is to shrink the "
            "tolerance the global layer is allowed to spend, e.g. 0.5x the physical "
            "tolerance, leaving the rest as margin for the controller.",
            "Re-balance the per-coordinate smoothing weights so the axis that binds "
            "downstream is not the one that pays.",
        ]
    )
    report.metrics = metrics
    return report, metrics


def _plots(
    metrics: dict, table: C.Table, inverse_table: C.Table | None,
    dense: C.Table | None, output_dir: Path, dpi: int,
) -> None:
    figure, axes = C.new_figure(3, 1, (12, 11))
    if figure is None:
        return
    s = table.column("s_m")
    tol = metrics["position_tolerance_m"]
    if inverse_table is not None:
        axes[0].plot(
            inverse_table.column("s_m"),
            inverse_table.column("position_error_m") / tol,
            "--", color=C.PALETTE["before"], label="L1 seed",
        )
    axes[0].plot(
        s, table.column("position_error_m") / tol,
        color=C.PALETTE["achieved"], label="L2 optimised",
    )
    if dense is not None and dense.has("position_error_m", "s_m"):
        axes[0].scatter(
            dense.column("s_m"), dense.column("position_error_m") / tol,
            s=9, alpha=0.5, color=C.PALETTE["desired"], label="L2 between nodes",
        )
    axes[0].axhline(1.0, color=C.PALETTE["limit"], linestyle=":", label="tolerance")
    axes[0].set_ylabel("position error / tolerance")
    axes[0].set_title("Tolerance-budget migration — the jaggedness, quantified")
    axes[0].legend()

    achieved = table.vector("achieved")
    desired = table.vector("desired")
    if achieved is not None and desired is not None:
        offset = achieved - desired
        for index, axis_name in enumerate("xyz"):
            axes[1].plot(s, 1e6 * offset[:, index], label=f"L2 {axis_name}")
        if inverse_table is not None:
            inverse_tip = inverse_table.vector("tip")
            inverse_desired = inverse_table.vector("desired")
            if inverse_tip is not None and inverse_desired is not None:
                axes[1].plot(
                    inverse_table.column("s_m"),
                    1e6 * np.linalg.norm(inverse_tip - inverse_desired, axis=1),
                    "--", color=C.PALETTE["before"], label="L1 |error|",
                )
        axes[1].axhline(1e6 * tol, color=C.PALETTE["limit"], linestyle=":")
        axes[1].axhline(-1e6 * tol, color=C.PALETTE["limit"], linestyle=":")
        axes[1].set_ylabel("tip offset from centreline [um]")
        axes[1].set_title("Per-axis wander inside the tolerance tube")
        axes[1].legend(ncol=4, fontsize=8)

    seed_keys = [f"seed_q{i}" for i in range(1, 7)] + ["seed_L"]
    if table.has(*seed_keys):
        seed = np.stack([table.columns[key] for key in seed_keys], axis=1)
        optimised = table.state_matrix("q{i}_rad", "insertion_m")
        scale = C.DEFAULT_CONFIGURATION_SCALE
        _, _, second_before, _ = C.path_derivatives(s, seed, scale)
        _, _, second_after, _ = C.path_derivatives(s, optimised, scale)
        if second_before.size:
            axes[2].semilogy(
                s[1:-1], np.linalg.norm(second_before, axis=1),
                "--", color=C.PALETTE["before"], label="seed",
            )
            axes[2].semilogy(
                s[1:-1], np.linalg.norm(second_after, axis=1),
                color=C.PALETTE["achieved"], label="global",
            )
        axes[2].set_ylabel(r"$\|d^2(\chi/\mathrm{scale})/ds^2\|$")
        axes[2].set_title("Configuration second-derivative demand")
        axes[2].legend()
    axes[2].set_xlabel("lumen arc length s [m]")
    C.save_figure(figure, output_dir / "layer2_budget_and_jaggedness.png", dpi)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir or (args.global_dir / "stack_analysis")
    report, metrics = analyse_global_layer(args.global_dir, args.inverse_dir)
    paths = report.write(Path(output_dir), "layer2_global")
    if not args.no_plots:
        table = C.read_table(Path(args.global_dir) / "global_configuration_path.csv")
        inverse_table = (
            C.read_table(Path(args.inverse_dir) / "inverse_configuration_path.csv")
            if args.inverse_dir is not None
            else None
        )
        dense_path = Path(args.global_dir) / "global_configuration_dense_validation.csv"
        dense = None
        if dense_path.exists() and dense_path.stat().st_size > 0:
            try:
                dense = C.read_table(dense_path)
            except ValueError:
                dense = None
        _plots(metrics, table, inverse_table, dense, Path(output_dir), args.dpi)
    print(f"Layer 2 analysis -> {paths['report']}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
