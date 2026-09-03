#!/usr/bin/env python3
"""End-to-end analysis of the four-layer control stack.

Runs each layer's analysis and then does the thing no single-layer report can:
puts the layers side by side on one axis, so you can see where the tip error is
created, where it is spent, and where it is absorbed.

    L1 inverse configuration   solves for chi(s) with task error as a COST
    L2 global smoothing        re-solves for chi(s) with task error as a CONSTRAINT
    L3 time parameterisation   turns chi(s) into chi(t) under actuator limits
    L4 MPC                     tracks chi(t) with beam feedback

The two facts that make the stack legible:

* **Layers 1 and 2 optimise the same quantity with opposite semantics.** L1
  minimises tip error; L2 merely requires it to stay inside a ball. Error is
  therefore expected to *grow* across L2 — up to the tolerance, and no further.
* **The tolerances are not the same size at every layer.** A micron of planner
  roughness is meaningless against a millimetre-scale controller tolerance.
  The budget column below is what makes them comparable.

Usage
-----
    python analyse_stack.py \
        --inverse-dir   results/offline_inverse_configuration_60 \
        --global-dir    results/global_configuration_full_debug \
        --reference-dir results/time_parameterized_configuration_path \
        --mpc-dir       results/..._mpc_simulation \
        --output-dir    results/stack_analysis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from . import _stack_common as C
    from .analyse_layer1_inverse import analyse_inverse_layer
    from .analyse_layer2_global import analyse_global_layer
    from .analyse_layer3_time_parameterization import analyse_time_layer
    from .analyse_layer4_mpc import analyse_mpc_layer
except ImportError:
    import _stack_common as C
    from analyse_layer1_inverse import analyse_inverse_layer
    from analyse_layer2_global import analyse_global_layer
    from analyse_layer3_time_parameterization import analyse_time_layer
    from analyse_layer4_mpc import analyse_mpc_layer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inverse-dir", type=Path, default=None)
    parser.add_argument("--global-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--baseline-reference-dir", type=Path, default=None)
    parser.add_argument("--mpc-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def analyse_stack(
    *,
    inverse_dir: Path | None,
    global_dir: Path | None,
    reference_dir: Path | None,
    baseline_reference_dir: Path | None,
    mpc_dir: Path | None,
    output_dir: Path,
    plots: bool = True,
    dpi: int = 170,
) -> C.Report:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layers: dict[str, dict] = {}
    reports: dict[str, C.Report] = {}
    failures: dict[str, str] = {}

    def run(name: str, function, *args) -> None:
        try:
            report, metrics = function(*args)
        except Exception as error:  # a missing layer must not kill the rest
            failures[name] = f"{type(error).__name__}: {error}"
            return
        reports[name] = report
        layers[name] = metrics
        stem = {"L1": "layer1_inverse", "L2": "layer2_global",
                "L3": "layer3_time", "L4": "layer4_mpc"}[name]
        report.write(output_dir, stem)

    if inverse_dir is not None:
        run("L1", analyse_inverse_layer, inverse_dir)
    if global_dir is not None:
        run("L2", analyse_global_layer, global_dir, inverse_dir)
    if reference_dir is not None:
        run("L3", analyse_time_layer, reference_dir, baseline_reference_dir, None)
    if mpc_dir is not None:
        run("L4", analyse_mpc_layer, mpc_dir, reference_dir)

    report = C.Report("Control stack — end-to-end analysis")
    report.metrics = {"layers": layers, "unavailable_layers": failures}

    report.head("The stack on one axis")
    report.text(
        "Each row is a layer's tip-position error against *its own* tolerance. "
        "The budget column is the comparable one: it is the fraction of the "
        "allowance each layer spends. Error growing from L1 to L2 is the design "
        "working as written — L1 minimises this quantity, L2 only bounds it — "
        "but growth beyond a budget of 1 is a constraint violation, not a trade."
    )
    rows = []
    if "L1" in layers:
        b = layers["L1"]["position_budget"]
        rows.append(["L1 inverse (nodes)", b["tolerance"], b["rms"], b["p95"],
                     b["maximum"], b["utilisation_p95"], "cost"])
    if "L2" in layers:
        b = layers["L2"].get("position_budget_after")
        if b:
            rows.append(["L2 global (nodes)", b["tolerance"], b["rms"], b["p95"],
                         b["maximum"], b["utilisation_p95"], "hard constraint"])
        d = layers["L2"].get("position_budget_dense")
        if d:
            rows.append(["L2 global (between nodes)", d["tolerance"], d["rms"],
                         d["p95"], d["maximum"], d["utilisation_p95"], "unconstrained"])
    if "L3" in layers and "planned_position_error" in layers["L3"]:
        p = layers["L3"]["planned_position_error"]
        rows.append(["L3 timed reference", math.nan, p["rms_m"], math.nan,
                     p["max_m"], math.nan, "inherited"])
    if "L4" in layers:
        b = layers["L4"]["position_budget"]
        rows.append(["L4 MPC (closed loop)", b["tolerance"], b["rms"], b["p95"],
                     b["maximum"], b["utilisation_p95"], "cost + feedback"])
    report.table(
        ["layer", "tolerance [m]", "rms [m]", "p95 [m]", "max [m]",
         "budget p95", "task error is a..."],
        rows,
    )

    # Scale comparison: is upstream roughness even visible downstream?
    if "L2" in layers and "L4" in layers:
        upstream = layers["L2"].get("position_budget_after", {}).get("maximum", math.nan)
        downstream_tolerance = layers["L4"]["position_budget"]["tolerance"]
        if np.isfinite(upstream) and downstream_tolerance > 0:
            ratio = upstream / downstream_tolerance
            report.metrics["upstream_roughness_vs_downstream_tolerance"] = float(ratio)
            level = "info" if ratio < 0.1 else "warn"
            report.finding(
                level,
                f"Planner tip roughness is {100 * ratio:.1f}% of the controller's tolerance",
                f"peak L2 tip excursion {1e6 * upstream:.1f} um against a controller "
                f"tolerance of {1e3 * downstream_tolerance:.2f} mm. "
                + (
                    "At this ratio the planner's tip jaggedness cannot be the cause of "
                    "any tracking problem you observe — look for the real cause in the "
                    "constraint and convergence findings instead."
                    if ratio < 0.1
                    else "This is large enough to matter downstream."
                ),
            )

    report.head("Per-layer verdicts")
    for name in ("L1", "L2", "L3", "L4"):
        if name not in reports:
            if name in failures:
                report.lines.append(f"**{name}** — not analysed: {failures[name]}")
                report.lines.append("")
            continue
        layer_report = reports[name]
        report.lines.append(f"**{name} — {layer_report.title}**")
        report.lines.append("")
        for finding in layer_report.findings:
            report.lines.append(finding.line())
            if finding.level in {"fail", "warn"}:
                report.findings.append(
                    C.Finding(finding.level, f"[{name}] {finding.title}", finding.detail)
                )
        report.lines.append("")

    report.head("What each layer is for, and how to tell it is doing its job")
    report.table(
        ["layer", "decides", "healthy when", "failure signature"],
        [
            ["L1 inverse", "chi(s) node by node, with continuation",
             "task error far below tolerance; few multistarts; keep-out margin > 0",
             "step halving, rank-deficient Jacobians, nodes at the tolerance"],
            ["L2 global", "a smoother chi(s) inside the same tolerance",
             "converged, feasible at nodes AND between them, all L1 constraints "
             "still enforced",
             "iteration cap hit, violation > tolerance, dense samples outside "
             "tolerance, constraints silently dropped"],
            ["L3 time", "s_dot(s) under actuator limits",
             "some actuator at a limit almost everywhere",
             "low saturation (something else is limiting), non-monotone s(t)"],
            ["L4 MPC", "velocity commands from measured beam feedback",
             "tracking well inside tolerance, every QP solved within one sample "
             "period",
             "QP overruns, saturated acceleration, one-step prediction error "
             "comparable to per-step motion"],
        ],
    )

    if plots:
        _plot_stack(layers, output_dir, dpi)
    report.write(output_dir, "stack")
    return report


def _plot_stack(layers: dict, output_dir: Path, dpi: int) -> None:
    figure, axes = C.new_figure(1, 1, (11, 5))
    if figure is None:
        return
    axis = axes[0]
    names, values, tolerances = [], [], []
    for key, label, getter in (
        ("L1", "L1 inverse", lambda m: m.get("position_budget")),
        ("L2", "L2 global", lambda m: m.get("position_budget_after")),
        ("L2", "L2 dense", lambda m: m.get("position_budget_dense")),
        ("L4", "L4 MPC", lambda m: m.get("position_budget")),
    ):
        if key not in layers:
            continue
        block = getter(layers[key])
        if not block:
            continue
        names.append(label)
        values.append(block["rms"])
        tolerances.append(block["tolerance"])
    if not names:
        return
    positions = np.arange(len(names))
    axis.bar(positions, 1e6 * np.asarray(values), color=C.PALETTE["achieved"],
             label="rms tip error")
    axis.plot(positions, 1e6 * np.asarray(tolerances), "o--",
              color=C.PALETTE["limit"], label="that layer's tolerance")
    axis.set_yscale("log")
    axis.set_xticks(positions, names)
    axis.set_ylabel("tip position [um, log scale]")
    axis.set_title("Error budget across the stack")
    axis.legend()
    C.save_figure(figure, output_dir / "stack_error_budget.png", dpi)


def main() -> None:
    args = _arguments()
    report = analyse_stack(
        inverse_dir=args.inverse_dir,
        global_dir=args.global_dir,
        reference_dir=args.reference_dir,
        baseline_reference_dir=args.baseline_reference_dir,
        mpc_dir=args.mpc_dir,
        output_dir=args.output_dir,
        plots=not args.no_plots,
        dpi=args.dpi,
    )
    print(f"Stack analysis -> {Path(args.output_dir) / 'stack_report.md'}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
