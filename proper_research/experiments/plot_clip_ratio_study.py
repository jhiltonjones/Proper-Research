"""Plot Test C: naive-controller clip ratio, along the path and by segment.

Reads one or more ``test_c_clip_<tag>.json`` / ``_steps.csv`` pairs written by
``clip_ratio_study.py`` and writes ``test_c_clip_ratio.png`` into
``--study-dir``. Reusable -- pass ``--tags`` for whichever tags exist.

Usage
-----
    python -m proper_research.experiments.plot_clip_ratio_study \\
        --study-dir results/contact_study_radius_p30_m70 --tags quick full
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
from matplotlib import cm


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Test C (clip ratio study).")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=("",),
                         help="Which test_c_clip_<tag> files to load and merge.")
    parser.add_argument("--title-suffix", default="")
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
        jpath = study_dir / f"test_c_clip{suffix}.json"
        cpath = study_dir / f"test_c_clip{suffix}_steps.csv"
        if jpath.exists():
            records.extend(json.loads(jpath.read_text()))
        if cpath.exists():
            with cpath.open() as handle:
                steps.extend(list(csv.DictReader(handle)))
    return records, steps


def main() -> int:
    args = _arguments()
    records, steps = load_all(args.study_dir, args.tags)
    if not records:
        print("no test_c_clip*.json found for the given --tags", flush=True)
        return 1

    cells = sorted({r["cell"] for r in records})
    steps_by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in steps:
        steps_by_cell[row["cell"]].append(row)
    for cell in steps_by_cell:
        steps_by_cell[cell].sort(key=lambda r: int(as_float(r["k"])))

    # Short, readable per-cell labels: "3.50mm/contact" instead of the full
    # "radius_x1__contact__naive_inverse_jacobian_quick" cell id.
    short_name: dict[str, str] = {}
    for rec in records:
        controller_name = rec.get("controller", "naive_inverse_jacobian")
        cell = rec.get("cell", "")
        prefix = f"{rec['anatomy']}__{rec['jacobian_model']}__{controller_name}"
        tag = cell[len(prefix):].lstrip("_") if cell.startswith(prefix) else ""
        label = f"{rec['radius_mm']:.2f}mm/{rec['jacobian_model']}"
        if tag:
            label += f" [{tag}]"
        short_name[cell] = label

    def short(cell: str) -> str:
        return short_name.get(cell, cell)

    colors = cm.tab10(np.linspace(0, 1, max(len(cells), 2)))
    cmap = {c: colors[i % len(colors)] for i, c in enumerate(cells)}

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"Test C — naive-controller clip ratio{suffix}", fontsize=13, y=0.995)

    # (a) alpha_mag vs arc length, one line per cell, bend segments shaded
    ax = axes[0, 0]
    segments_seen: dict[str, tuple[float, float]] = {}
    for cell in cells:
        rows = steps_by_cell.get(cell, [])
        if not rows:
            continue
        arc = np.array([as_float(r["arc_mm"]) for r in rows])
        alpha = np.array([as_float(r["alpha_mag"]) for r in rows])
        ax.plot(arc, alpha, lw=1.1, color=cmap[cell], label=short(cell), alpha=0.85)
        for r in rows:
            seg = r["segment"]
            a = as_float(r["arc_mm"])
            if seg.startswith("bend"):
                lo, hi = segments_seen.get(seg, (a, a))
                segments_seen[seg] = (min(lo, a), max(hi, a))
    for seg, (lo, hi) in segments_seen.items():
        ax.axvspan(lo, hi, color="crimson", alpha=0.06)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"$\alpha = \|command_{clipped}\| / \|command_{raw}\|$")
    ax.set_title("(a) surviving command magnitude along the path (bend shaded)", fontsize=10)
    ax.legend(fontsize=6, ncol=1, loc="lower left")

    # (b) tip position error vs arc length, same x-axis
    ax = axes[0, 1]
    for cell in cells:
        rows = steps_by_cell.get(cell, [])
        if not rows:
            continue
        arc = np.array([as_float(r["arc_mm"]) for r in rows])
        perr = np.array([as_float(r["position_error_mm"]) for r in rows])
        ax.plot(arc, perr, lw=1.1, color=cmap[cell], label=short(cell), alpha=0.85)
    for seg, (lo, hi) in segments_seen.items():
        ax.axvspan(lo, hi, color="crimson", alpha=0.06)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("tip position error [mm]")
    ax.set_title("(b) tracking error along the path — where clipping costs accuracy", fontsize=10)
    ax.legend(fontsize=6, loc="upper left")

    # (c) clip activity rate by segment, grouped by cell
    ax = axes[1, 0]
    seg_order = []
    for rec in records:
        for label in rec.get("by_segment", {}):
            if label not in seg_order:
                seg_order.append(label)
    width = 0.8 / max(len(cells), 1)
    x = np.arange(len(seg_order))
    for i, cell in enumerate(cells):
        rec = next((r for r in records if r["cell"] == cell), None)
        if rec is None:
            continue
        vals = [rec.get("by_segment", {}).get(seg, {}).get("clip_activity_rate", np.nan)
                for seg in seg_order]
        ax.bar(x + i * width, [100 * v if v == v else 0 for v in vals], width,
               color=cmap[cell], label=short(cell))
    ax.set_xticks(x + width * (len(cells) - 1) / 2)
    ax.set_xticklabels(seg_order, rotation=20, ha="right")
    ax.set_ylabel("clip activity rate [%]")
    ax.set_title("(c) fraction of steps clipped, by path segment", fontsize=10)
    ax.legend(fontsize=6, ncol=1)

    # (d) alpha_mag distribution (box) by segment, one panel row per cell group via color
    ax = axes[1, 1]
    box_data, box_labels, box_colors = [], [], []
    for cell in cells:
        rows = steps_by_cell.get(cell, [])
        for seg in seg_order:
            vals = [as_float(r["alpha_mag"]) for r in rows if r["segment"] == seg]
            if not vals:
                continue
            box_data.append(vals)
            box_labels.append(f"{short(cell)}\n{seg}")
            box_colors.append(cmap[cell])
    if box_data:
        bp = ax.boxplot(box_data, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(box_labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel(r"$\alpha$ distribution")
    ax.set_title("(d) alpha_mag spread by cell x segment", fontsize=10)

    # (e) which bound is active, by segment (stacked), first cell shown per radius/jacobian combo
    ax = axes[2, 0]
    x = np.arange(len(seg_order))
    width = 0.8 / max(len(cells), 1)
    for i, cell in enumerate(cells):
        rec = next((r for r in records if r["cell"] == cell), None)
        if rec is None:
            continue
        vel = np.array([rec.get("by_segment", {}).get(seg, {}).get("at_vel_rate", 0) or 0
                        for seg in seg_order])
        acc = np.array([rec.get("by_segment", {}).get(seg, {}).get("at_acc_rate", 0) or 0
                        for seg in seg_order])
        state = np.array([rec.get("by_segment", {}).get(seg, {}).get("at_state_rate", 0) or 0
                          for seg in seg_order])
        xi = x + i * width
        ax.bar(xi, 100 * vel, width, color="tab:blue", label="velocity" if i == 0 else None)
        ax.bar(xi, 100 * acc, width, bottom=100 * vel, color="tab:orange",
               label="acceleration" if i == 0 else None)
        ax.bar(xi, 100 * state, width, bottom=100 * (vel + acc), color="tab:green",
               label="state box" if i == 0 else None)
    ax.set_xticks(x + width * (len(cells) - 1) / 2)
    ax.set_xticklabels(seg_order, rotation=20, ha="right")
    ax.set_ylabel("rate bound was active [%] (stacked)")
    ax.set_title("(e) which constraint clips, by segment (bars = cells in order)", fontsize=10)
    ax.legend(fontsize=7)

    # (f) whole-path rms/max accuracy per cell
    ax = axes[2, 1]
    rms = [r.get("rms_mm", np.nan) for r in records]
    mx = [r.get("max_mm", np.nan) for r in records]
    x = np.arange(len(records))
    w = 0.35
    ax.bar(x - w / 2, rms, w, color="steelblue", label="rms [mm]")
    ax.bar(x + w / 2, mx, w, color="crimson", alpha=0.7, label="max [mm]")
    ax.set_xticks(x)
    ax.set_xticklabels([short(r["cell"]) for r in records], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("beam tip position error [mm]")
    ax.set_title("(f) whole-path tracking accuracy per cell", fontsize=10)
    ax.legend(fontsize=8)

    fig.subplots_adjust(top=0.94, bottom=0.10, left=0.06, right=0.98,
                        hspace=0.55, wspace=0.22)
    out = args.study_dir / "test_c_clip_ratio.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
