"""Plot the four-controller closed-loop comparison.

Reads controller_comparison[_<tag>].json / _steps.csv from
controller_comparison_study.py and writes controller_comparison.png into
--study-dir. Reusable -- pass --tags for whichever tags exist.

Both Jacobian models are drawn when present: the contact plant Jacobian as a
solid line / solid bar, the contact-free ("no_contact") model as a dashed line /
hatched bar, in the same per-controller colour.

Usage
-----
    python -m proper_research.experiments.plot_controller_comparison \\
        --study-dir results/contact_study_radius_p30_m70
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CONTROLLER_COLORS = {
    "naive_inverse_jacobian": "tab:red",
    "mpc_lti": "tab:blue",
    "mpc_ltv_offline": "tab:green",
    "mpc_ltv_sqp_online": "tab:purple",
}
CONTROLLER_ORDER = list(CONTROLLER_COLORS.keys())

# (linestyle, bar hatch) per Jacobian model
MODEL_STYLE = {
    "contact": ("-", None),
    "no_contact": ("--", "///"),
}
MODEL_ORDER = ["contact", "no_contact"]
MODEL_LABEL = {"contact": "contact Jacobian", "no_contact": "contact-free Jacobian"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the controller comparison.")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=("",))
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--only-radius-mm", type=float, nargs="+", default=None,
                        help="keep only these lumen radii (mm); default keeps all")
    parser.add_argument("--clip-error-mm", type=float, default=None,
                        help="clip the y-axis of the error-vs-arc panels to this "
                             "value (diverging runs otherwise flatten the rest)")
    return parser.parse_args()


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_all(study_dir: Path, tags):
    records: list[dict] = []
    steps: list[dict] = []
    for tag in tags:
        suffix = f"_{tag}" if tag else ""
        jpath = study_dir / f"controller_comparison{suffix}.json"
        cpath = study_dir / f"controller_comparison{suffix}_steps.csv"
        if jpath.exists():
            records.extend(json.loads(jpath.read_text()))
        if cpath.exists():
            with cpath.open() as handle:
                steps.extend(list(csv.DictReader(handle)))
    return records, steps


def controller_order(present) -> list[str]:
    ordered = [c for c in CONTROLLER_ORDER if c in present]
    ordered += sorted(c for c in present if c not in CONTROLLER_ORDER)
    return ordered


def model_order(present) -> list[str]:
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += sorted(m for m in present if m not in MODEL_ORDER)
    return ordered


def main() -> int:
    args = _arguments()
    records, steps = load_all(args.study_dir, args.tags)
    if not records:
        print("no controller_comparison*.json found for the given --tags")
        return 1

    if args.only_radius_mm is not None:
        keep = [round(v, 4) for v in args.only_radius_mm]
        records = [r for r in records
                   if any(abs(as_float(r["radius_mm"]) - k) < 1e-3 for k in keep)]
        if not records:
            print(f"no records at radius {args.only_radius_mm} mm")
            return 1

    radii = sorted({round(as_float(r["radius_mm"]), 4) for r in records}, reverse=True)
    controllers = controller_order({r["controller"] for r in records})
    models = model_order({r.get("jacobian_model", "contact") for r in records})

    steps_by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in steps:
        steps_by_cell[row["cell"]].append(row)
    for cell in steps_by_cell:
        steps_by_cell[cell].sort(key=lambda r: int(as_float(r["k"])))

    def find(radius_mm, controller, model):
        return next(
            (r for r in records
             if abs(as_float(r["radius_mm"]) - radius_mm) < 1e-6
             and r["controller"] == controller
             and r.get("jacobian_model", "contact") == model),
            None,
        )

    n_radii = max(len(radii), 1)
    fig, axes = plt.subplots(3, max(n_radii, 2), figsize=(6.5 * max(n_radii, 2), 13.5))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Controller comparison, closed loop, contact plant{suffix}\n"
        "solid = contact Jacobian, dashed/hatched = contact-free Jacobian",
        fontsize=13, y=0.997)

    n_groups = max(len(controllers) * len(models), 1)
    width = 0.8 / n_groups
    x = np.arange(len(radii))

    def bar_group(ax, metric_key, as_percent_of_samples=False):
        slot = 0
        for controller in controllers:
            color = CONTROLLER_COLORS.get(controller, "grey")
            for model in models:
                _ls, hatch = MODEL_STYLE.get(model, ("-", None))
                vals = []
                for r in radii:
                    rec = find(r, controller, model) or {}
                    if as_percent_of_samples:
                        samples = max(rec.get("samples", 1), 1)
                        vals.append(100.0 * rec.get("solve_failures", 0) / samples)
                    else:
                        vals.append(rec.get(metric_key, np.nan))
                ax.bar(x + slot * width, vals, width, color=color, hatch=hatch,
                       edgecolor="black", linewidth=0.3)
                slot += 1
        ax.set_xticks(x + width * (n_groups - 1) / 2)
        ax.set_xticklabels([f"{r:.2f}" for r in radii])
        ax.set_xlabel("lumen radius [mm]")

    # row 0: tip position error vs arc length, one subplot per radius
    for col, radius_mm in enumerate(radii):
        ax = axes[0, col]
        for controller in controllers:
            color = CONTROLLER_COLORS.get(controller, "grey")
            for model in models:
                rec = find(radius_mm, controller, model)
                if rec is None:
                    continue
                rows = steps_by_cell.get(rec["cell"], [])
                if not rows:
                    continue
                arc = np.array([as_float(r["arc_mm"]) for r in rows])
                err = np.array([as_float(r["position_error_mm"]) for r in rows])
                ls, _hatch = MODEL_STYLE.get(model, ("-", None))
                ax.plot(arc, err, lw=1.3, color=color, ls=ls)
        ax.set_xlabel("arc length [mm]")
        ax.set_ylabel("tip position error [mm]" if col == 0 else "")
        ax.set_title(f"(a{col+1}) radius = {radius_mm:.2f} mm", fontsize=10)
        if args.clip_error_mm is not None:
            ax.set_ylim(0, args.clip_error_mm)
        if col == 0:
            handles = [Line2D([0], [0], color=CONTROLLER_COLORS.get(c, "grey"), lw=1.3)
                       for c in controllers]
            handles += [Line2D([0], [0], color="black", lw=1.3,
                               ls=MODEL_STYLE.get(m, ("-", None))[0])
                        for m in models]
            labels = list(controllers) + [MODEL_LABEL.get(m, m) for m in models]
            ax.legend(handles, labels, fontsize=7)
    for col in range(len(radii), axes.shape[1]):
        axes[0, col].axis("off")

    legend_handles = [Patch(facecolor=CONTROLLER_COLORS.get(c, "grey"), label=c)
                      for c in controllers]
    legend_handles += [Patch(facecolor="white", edgecolor="black",
                             hatch=MODEL_STYLE.get(m, ("-", None))[1],
                             label=MODEL_LABEL.get(m, m))
                       for m in models]

    # row 1, col 0: rms grouped bar by radius x controller x model
    ax = axes[1, 0]
    bar_group(ax, "rms_beam_position_error_mm")
    ax.set_ylabel("rms tip position error [mm]")
    ax.set_title("(b) whole-path rms accuracy", fontsize=10)

    # row 1, col 1: max error
    if axes.shape[1] > 1:
        ax = axes[1, 1]
        bar_group(ax, "maximum_beam_position_error_mm")
        ax.set_ylabel("max tip position error [mm]")
        ax.set_title("(c) whole-path max accuracy", fontsize=10)
    for col in range(2, axes.shape[1]):
        axes[1, col].axis("off")

    # row 2, col 0: mean solve time (log y)
    ax = axes[2, 0]
    bar_group(ax, "mean_solve_time_ms")
    ax.set_yscale("log")
    ax.set_ylabel("mean solve time [ms]")
    ax.set_title("(d) controller cost per step", fontsize=10)

    # row 2, col 1: solve-failure rate
    if axes.shape[1] > 1:
        ax = axes[2, 1]
        bar_group(ax, "solve_failures", as_percent_of_samples=True)
        ax.set_ylabel("solve failure rate [%]")
        ax.set_title("(e) solver reliability", fontsize=10)
    for col in range(2, axes.shape[1]):
        axes[2, col].axis("off")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8, frameon=False)

    fig.subplots_adjust(top=0.92, bottom=0.09, left=0.05, right=0.98,
                        hspace=0.4, wspace=0.28)
    if args.only_radius_mm is not None:
        tag = "_".join(f"{v:g}".replace(".", "p") for v in sorted(radii, reverse=True))
        out = args.study_dir / f"controller_comparison_radius_{tag}mm.png"
    else:
        out = args.study_dir / "controller_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
