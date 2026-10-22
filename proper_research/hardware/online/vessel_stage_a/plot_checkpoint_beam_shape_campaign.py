#!/usr/bin/env python3
"""Turn checkpoint_beam_shape_campaign.py's checkpoint_results.json into the
contact-vs-no-contact comparison figures: per-checkpoint spatial overlays
(measured vs both model predictions) and residual-vs-arclength curves,
plus a printed summary table and overall mean residual for each model.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.plot_checkpoint_beam_shape_campaign \\
        --results checkpoint_campaign_logs/myrun/checkpoint_results.json \\
        --out-dir checkpoint_campaign_logs/myrun
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resample_by_arclength(pts: np.ndarray, n: int = 40):
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0), np.zeros(n)
    s_query = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(s_query, s, pts[:, k])
    return out, s_query / total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-resample", type=int, default=40)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    N = args.n_resample
    n_cp = len(data)
    ncols = min(3, n_cp)
    nrows = math.ceil(n_cp / ncols)

    fig_spatial, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    summary = []
    for i, cp in enumerate(data):
        meas = np.asarray(cp["beam_R_measured"])
        pred_c = np.asarray(cp["pred_contact"])
        pred_nc = np.asarray(cp["pred_nocontact"])

        meas_rs, frac = resample_by_arclength(meas, N)
        predc_rs, _ = resample_by_arclength(pred_c, N)
        prednc_rs, _ = resample_by_arclength(pred_nc, N)
        resid_c = np.linalg.norm(meas_rs - predc_rs, axis=1) * 1000.0
        resid_nc = np.linalg.norm(meas_rs - prednc_rs, axis=1) * 1000.0

        summary.append(dict(
            label=cp["label"], target_L_mm=cp.get("target_L_mm"), L_chord_mm=cp["L_chord_m"] * 1000.0,
            resid_c_base_mm=float(resid_c[0]), resid_c_tip_mm=float(resid_c[-1]), resid_c_mean_mm=float(resid_c.mean()),
            resid_nc_base_mm=float(resid_nc[0]), resid_nc_tip_mm=float(resid_nc[-1]), resid_nc_mean_mm=float(resid_nc.mean()),
        ))

        ax = axes.flat[i]
        ax.plot(meas[:, 0], meas[:, 1], "-", color="black", lw=2, label="measured")
        ax.plot(pred_c[:, 0], pred_c[:, 1], "--", color="tab:red", lw=1.5, label="predicted (contact)")
        ax.plot(pred_nc[:, 0], pred_nc[:, 1], "--", color="tab:blue", lw=1.5, label="predicted (no-contact)")
        ax.scatter(*meas[0, :2], color="green", marker="s", s=40, zorder=5, label="base")
        ax.scatter(*meas[-1, :2], color="black", marker="o", s=40, zorder=5, label="measured tip")
        title = f"{cp['label']}\nL_chord={cp['L_chord_m'] * 1000:.1f}mm"
        if cp.get("target_L_mm") is not None:
            title += f" (target {cp['target_L_mm']:.0f}mm)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.axis("equal")
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n_cp, nrows * ncols):
        axes.flat[j].axis("off")
    fig_spatial.suptitle("Measured vs predicted (contact / no-contact) beam shape per checkpoint", fontsize=13)
    fig_spatial.tight_layout(rect=[0, 0, 1, 0.96])
    spatial_path = out_dir / "checkpoint_campaign_spatial.png"
    fig_spatial.savefig(spatial_path, dpi=150)
    print(f"saved -> {spatial_path}")

    # residual vs arclength
    fig_r, ax_r = plt.subplots(1, 2, figsize=(14, 6))
    for cp in data:
        meas = np.asarray(cp["beam_R_measured"])
        pred_c = np.asarray(cp["pred_contact"])
        pred_nc = np.asarray(cp["pred_nocontact"])
        meas_rs, frac = resample_by_arclength(meas, N)
        predc_rs, _ = resample_by_arclength(pred_c, N)
        prednc_rs, _ = resample_by_arclength(pred_nc, N)
        resid_c = np.linalg.norm(meas_rs - predc_rs, axis=1) * 1000.0
        resid_nc = np.linalg.norm(meas_rs - prednc_rs, axis=1) * 1000.0
        ax_r[0].plot(frac, resid_c, "-o", ms=3, label=cp["label"])
        ax_r[1].plot(frac, resid_nc, "-o", ms=3, label=cp["label"])
    ax_r[0].set_title("CONTACT model"); ax_r[0].set_xlabel("s/L (0=base, 1=tip)"); ax_r[0].set_ylabel("|measured - predicted| (mm)")
    ax_r[1].set_title("NO-CONTACT model"); ax_r[1].set_xlabel("s/L (0=base, 1=tip)"); ax_r[1].set_ylabel("|measured - predicted| (mm)")
    ax_r[0].legend(fontsize=7); ax_r[1].legend(fontsize=7)
    fig_r.suptitle("Residual vs arc-length fraction, contact vs no-contact model")
    fig_r.tight_layout()
    residual_path = out_dir / "checkpoint_campaign_residuals.png"
    fig_r.savefig(residual_path, dpi=150)
    print(f"saved -> {residual_path}")

    # bar chart: mean tip residual per checkpoint, contact vs no-contact
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    labels = [s["label"] for s in summary]
    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width / 2, [s["resid_c_tip_mm"] for s in summary], width, label="contact model", color="tab:red")
    ax_bar.bar(x + width / 2, [s["resid_nc_tip_mm"] for s in summary], width, label="no-contact model", color="tab:blue")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel("tip residual |measured - predicted| (mm)")
    ax_bar.set_title("Tip residual per checkpoint: contact vs no-contact model")
    ax_bar.legend()
    fig_bar.tight_layout()
    bar_path = out_dir / "checkpoint_campaign_tip_residual_bar.png"
    fig_bar.savefig(bar_path, dpi=150)
    print(f"saved -> {bar_path}")

    print("\n=== SUMMARY ===")
    header = f"{'label':<25} {'target_L':>9} {'L_chord':>8} | {'base_c':>7} {'tip_c':>7} {'mean_c':>7} | {'base_nc':>8} {'tip_nc':>8} {'mean_nc':>8}"
    print(header)
    for s in summary:
        tl = f"{s['target_L_mm']:.0f}mm" if s["target_L_mm"] is not None else "n/a"
        print(f"{s['label']:<25} {tl:>9} {s['L_chord_mm']:8.2f} | "
              f"{s['resid_c_base_mm']:7.2f} {s['resid_c_tip_mm']:7.2f} {s['resid_c_mean_mm']:7.2f} | "
              f"{s['resid_nc_base_mm']:8.2f} {s['resid_nc_tip_mm']:8.2f} {s['resid_nc_mean_mm']:8.2f}")

    mean_resid_c = float(np.mean([s["resid_c_mean_mm"] for s in summary]))
    mean_resid_nc = float(np.mean([s["resid_nc_mean_mm"] for s in summary]))
    mean_tip_c = float(np.mean([s["resid_c_tip_mm"] for s in summary]))
    mean_tip_nc = float(np.mean([s["resid_nc_tip_mm"] for s in summary]))
    print(f"\nOverall mean residual (all points along beam): CONTACT={mean_resid_c:.2f}mm  NO-CONTACT={mean_resid_nc:.2f}mm")
    print(f"Overall mean TIP residual:                      CONTACT={mean_tip_c:.2f}mm  NO-CONTACT={mean_tip_nc:.2f}mm")

    with open(out_dir / "checkpoint_campaign_summary.json", "w") as f:
        json.dump({
            "per_checkpoint": summary,
            "mean_residual_mm": {"contact": mean_resid_c, "no_contact": mean_resid_nc},
            "mean_tip_residual_mm": {"contact": mean_tip_c, "no_contact": mean_tip_nc},
        }, f, indent=2)


if __name__ == "__main__":
    main()
