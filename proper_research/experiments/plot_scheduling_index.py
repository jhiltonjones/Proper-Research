"""Plot Test B: the scheduling index S and the contraction margin mu.

Reads the output of ``scheduling_index_study.py`` from ``--study-dir`` and
writes ``test_b_scheduling_index.png`` into the same directory. Reusable
across studies -- point it at any directory that has
``scheduling_index_S.json`` + ``scheduling_index_S_samples.csv``.

Usage
-----
    python -m proper_research.experiments.plot_scheduling_index \\
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
from matplotlib import cm


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Test B (scheduling index S / mu).")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="",
                         help="Appended to the figure suptitle, e.g. the vessel name.")
    parser.add_argument("--rotation-gap-threshold", type=float, default=0.45,
                         help="gap_rel above which a state counts as 'diverged' in panel (f).")
    return parser.parse_args()


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    args = _arguments()
    study_dir: Path = args.study_dir
    summary = json.loads((study_dir / "scheduling_index_S.json").read_text())
    with (study_dir / "scheduling_index_S_samples.csv").open() as handle:
        sample_rows = list(csv.DictReader(handle))

    summary = sorted(summary, key=lambda r: -r["radius_mm"])
    radii = [row["radius_mm"] for row in summary]
    by_radius: dict[float, list[dict]] = defaultdict(list)
    for row in sample_rows:
        by_radius[round(as_float(row["radius_mm"]), 4)].append(row)
    for radius_mm in by_radius:
        by_radius[radius_mm].sort(key=lambda r: as_float(r["arc_mm"]))

    colors = cm.viridis(np.linspace(0.05, 0.9, len(radii)))
    cmap = {r: c for r, c in zip(radii, colors)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"Test B — scheduling index S and contraction margin mu{suffix}",
                 fontsize=13, y=0.995)

    rr = np.array(radii, dtype=float)

    # (a) S vs radius -- the headline confirmation that S_free is radius-invariant
    ax = axes[0, 0]
    s_free = np.array([row["S_free"] for row in summary])
    s_contact = np.array([row["S_contact"] for row in summary])
    s_succ = np.array([row["S_succ"] for row in summary])
    ax.plot(rr, s_free, "o-", lw=2, label="S_free (offline-schedule model)")
    ax.plot(rr, s_contact, "s--", lw=1.6, label="S_contact")
    ax.plot(rr, s_succ, "^:", lw=1.2, label="S_succ (free)", alpha=0.8)
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel("S")
    ax.set_title("(a) S vs radius — is the schedule radius-blind?", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # (b) mu_min / mu_median vs radius -- the P2 contraction test
    ax = axes[0, 1]
    mu_min = np.array([row["mu_min"] for row in summary])
    mu_med = np.array([row["mu_median"] for row in summary])
    ax.plot(rr, mu_min, "o-", lw=2, color="crimson", label="mu_min")
    ax.plot(rr, mu_med, "s--", lw=1.6, color="steelblue", label="mu_median")
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.fill_between(rr, mu_min, 0, where=(mu_min < 0), color="crimson", alpha=0.12)
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel(r"$\mu = \lambda_{min}(sym(J_{contact} J_{free}^+))$")
    ax.set_title("(b) contraction margin — crosses zero where LTI/LTV stop being valid", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # (c) fraction of mu samples negative, and gap_rel median/max vs radius
    ax = axes[0, 2]
    frac_neg = np.array([row["frac_mu_negative"] for row in summary])
    gap_med = np.array([row["gap_rel_median"] for row in summary])
    ax2 = ax.twinx()
    ax.plot(rr, 100 * frac_neg, "o-", lw=2, color="crimson", label="frac(mu<0)")
    ax2.plot(rr, 100 * gap_med, "s--", lw=1.6, color="teal", label="median gap_rel")
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel("fraction of path with mu < 0  [%]", color="crimson")
    ax2.set_ylabel("median relative Jacobian gap  [%]", color="teal")
    ax.set_title("(c) how much of the path has lost contraction", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.invert_xaxis()

    # (d) mu(s) along the path, per radius
    ax = axes[1, 0]
    for radius_mm in radii:
        rows = [r for r in by_radius[radius_mm] if r["mu"] not in ("", "nan")]
        arc = np.array([as_float(r["arc_mm"]) for r in rows])
        mu = np.array([as_float(r["mu"]) for r in rows])
        finite = np.isfinite(mu)
        ax.plot(arc[finite], mu[finite], "o-", ms=3, lw=1.3, color=cmap[radius_mm],
                label=f"{radius_mm:.2f} mm")
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"$\mu(s)$")
    ax.set_title("(d) where along the path contraction is lost", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # (e) gap_rel(s) along the path, per radius (finer trace than panel d's mu)
    ax = axes[1, 1]
    for radius_mm in radii:
        rows = by_radius[radius_mm]
        arc = np.array([as_float(r["arc_mm"]) for r in rows])
        gap = np.array([as_float(r["gap_rel"]) for r in rows])
        order = np.argsort(arc)
        ax.plot(arc[order], 100 * gap[order], lw=1.3, color=cmap[radius_mm],
                label=f"{radius_mm:.2f} mm")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("relative Jacobian gap [%]")
    ax.set_title("(e) gap along the path (finer trace, feeds panel d's mu)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # (f) summary bar: S is flat, mu is not
    ax = axes[1, 2]
    width = 0.35
    x = np.arange(len(radii))
    s_free_norm = s_free / max(s_free.max(), 1e-12)
    ax.bar(x - width / 2, s_free_norm, width, color="steelblue", label="S_free / max(S_free)")
    mu_plot = np.clip(-mu_min, 0, None)
    ax.bar(x + width / 2, mu_plot, width, color="crimson", label="max(0, -mu_min)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in radii])
    ax.set_xlabel("lumen radius [mm]")
    ax.set_title("(f) S stays flat; the contraction violation is what moves", fontsize=10)
    ax.legend(fontsize=8)

    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.05, right=0.98,
                        hspace=0.38, wspace=0.30)
    out = study_dir / "test_b_scheduling_index.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
