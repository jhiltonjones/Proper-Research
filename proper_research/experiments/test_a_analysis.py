"""Test A analysis + explainability figures. Reusable across vessels.

Consumes <output-dir>/offline/jacobian_divergence.csv (+ _summary.json)
produced by ``run_contact_study.py --stage offline``.

Writes, into <output-dir>:
  test_a_jacobian_divergence.png   9-panel explainability figure
  test_a_beam_shape_lumen.png      beam shape vs lumen, nominal + tightening sweep
  test_a_metrics.json              threshold numbers used in the write-up
  test_a_summary_table.md          the markdown result table

Usage
-----
    python -m proper_research.experiments.test_a_analysis \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --nominal-radius-mm 3.5 \\
        --first-angle-deg 30 --second-angle-deg -70 \\
        --title-suffix "+30/-70 double-bend vessel"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from proper_research.experiments.pinned_planning_context import (  # noqa: E402
    add_geometry_arguments,
    resolve_planning_context_from_args,
    resolve_reference_dir,
)

GAP_CONTACT_THRESHOLD = 0.01   # 1 % relative Frobenius gap => "in contact"
ROTATION_THRESHOLD_DEG = 45.0  # row-space rotation => structural, not disturbance
ACTUATOR_LABELS = ["q1", "q2", "q3", "q4", "q5", "q6", "L"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test A: explainability figures from the offline Jacobian "
                    "divergence CSV, reusable across vessels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Same --output-dir passed to run_contact_study.py --stage offline.")
    parser.add_argument("--reference", type=Path, default=None,
                         help="Explicit time-parameterised reference dir. "
                              "Default: auto-detect under the vessel out_root.")
    parser.add_argument("--r-beam-mm", type=float, default=1.0,
                         help="Beam radius (ContactParams.r_beam), mm. Default 1.0.")
    parser.add_argument("--title-suffix", default="",
                         help='e.g. "+30/-70 double-bend vessel", used in figure titles.')
    add_geometry_arguments(parser)
    return parser.parse_args()


# --------------------------------------------------------------------------
def load_rows(offline_csv: Path) -> list[dict]:
    with offline_csv.open() as handle:
        return list(csv.DictReader(handle))


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def group_by_radius(rows: list[dict], nominal_radius_mm: float) -> dict[float, dict[str, np.ndarray]]:
    """radius_mm -> {metric: array over strided samples}, sorted by sample."""
    buckets: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        scale = as_float(row["radius_scale"])
        buckets[round(scale * nominal_radius_mm, 3)].append(row)

    out: dict[float, dict[str, np.ndarray]] = {}
    keys = [
        "sample", "frobenius_relative", "spectral_relative",
        "principal_angle_1_deg", "principal_angle_2_deg", "principal_angle_3_deg",
        "gain_ratio_min", "gain_ratio_max",
        "condition_contact", "condition_contact_free",
        *[f"per_actuator_relative_{i}" for i in range(1, 8)],
    ]
    for radius_mm, bucket in buckets.items():
        bucket.sort(key=lambda r: as_float(r["sample"]))
        col = {k: np.array([as_float(r.get(k, "nan")) for r in bucket]) for k in keys}
        for extra in ("contact_active", "gap_min_m", "contacting_node_fraction"):
            if extra in bucket[0]:
                col[extra] = np.array([as_float(r.get(extra, "nan")) for r in bucket])
        out[radius_mm] = col
    return dict(sorted(out.items(), reverse=True))


def arc_mm_for_samples(ref, samples: np.ndarray) -> np.ndarray:
    s = np.asarray(ref.path_coordinate_m, dtype=float)
    idx = np.clip(samples.astype(int), 0, s.size - 1)
    return 1.0e3 * (s[idx] - s[0])


# --------------------------------------------------------------------------
# figure 1 — explainability
# --------------------------------------------------------------------------
def figure_divergence(by_radius, ref, output_dir: Path, nominal_radius_mm: float,
                      r_beam_mm: float, title_suffix: str) -> tuple[dict, str]:
    radii = list(by_radius.keys())
    colors = cm.viridis(np.linspace(0.05, 0.9, len(radii)))
    cmap = {r: c for r, c in zip(radii, colors)}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    suffix = f", {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Test A — contact vs contact-free tip Jacobian{suffix}\n"
        f"lumen radius swept {radii[-1]:.2f}-{radii[0]:.2f} mm (nominal {nominal_radius_mm:.1f} mm, "
        f"beam radius {r_beam_mm:.1f} mm); {len(next(iter(by_radius.values()))['sample'])} states/radius",
        fontsize=13, y=0.995,
    )

    metrics = {}
    rr = np.array(radii)

    ax = axes[0, 0]
    med = np.array([np.nanmedian(by_radius[r]["frobenius_relative"]) for r in radii])
    rms = np.array([np.sqrt(np.nanmean(by_radius[r]["frobenius_relative"] ** 2)) for r in radii])
    mx = np.array([np.nanmax(by_radius[r]["frobenius_relative"]) for r in radii])
    ax.plot(rr, 100 * med, "o-", label="median", lw=2)
    ax.plot(rr, 100 * rms, "s--", label="rms", lw=1.5)
    ax.plot(rr, 100 * mx, "^:", label="max", lw=1.2, alpha=0.8)
    ax.axvline(nominal_radius_mm, color="grey", ls=":", lw=1)
    ax.axvline(r_beam_mm, color="crimson", ls="-", lw=1)
    ax.text(r_beam_mm, ax.get_ylim()[1], " beam", color="crimson", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel(r"relative Frobenius gap $\|\Delta J\|/\|J\|$ [%]")
    ax.set_title("(a) dose-response vs lumen radius", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    ax = axes[0, 1]
    for r in radii:
        arc = arc_mm_for_samples(ref, by_radius[r]["sample"])
        ax.plot(arc, 100 * by_radius[r]["frobenius_relative"], lw=1.4,
                color=cmap[r], label=f"{r:.2f} mm")
    ax.set_xlabel("arc length along planned path [mm]")
    ax.set_ylabel("relative gap [%]")
    ax.set_title("(b) gap along the planned path", fontsize=10)
    ax.legend(fontsize=7, title="lumen R", ncol=2)

    ax = axes[0, 2]
    for r in radii:
        arc = arc_mm_for_samples(ref, by_radius[r]["sample"])
        ax.plot(arc, by_radius[r]["principal_angle_3_deg"], lw=1.4, color=cmap[r],
                label=f"{r:.2f} mm")
    ax.axhline(ROTATION_THRESHOLD_DEG, color="crimson", ls="--", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("largest principal angle [deg]")
    ax.set_title("(c) principal-angle rotation along path", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    kc_med = np.array([np.nanmedian(by_radius[r]["condition_contact"]) for r in radii])
    kc_max = np.array([np.nanmax(by_radius[r]["condition_contact"]) for r in radii])
    kn_med = np.array([np.nanmedian(by_radius[r]["condition_contact_free"]) for r in radii])
    ax.plot(rr, kc_med, "o-", label=r"$\kappa(J_{contact})$ median", lw=2)
    ax.plot(rr, kc_max, "^:", label=r"$\kappa(J_{contact})$ max", lw=1.2)
    ax.plot(rr, kn_med, "s--", label=r"$\kappa(J_{free})$ median", lw=1.5, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel("condition number")
    ax.set_title("(d) conditioning vs lumen radius", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    ax = axes[1, 1]
    arc0 = arc_mm_for_samples(ref, next(iter(by_radius.values()))["sample"])
    grid = np.vstack([by_radius[r]["principal_angle_3_deg"] for r in radii])
    im = ax.pcolormesh(arc0, np.arange(len(radii)), grid, cmap="magma",
                       vmin=0, vmax=90, shading="nearest")
    ax.set_yticks(np.arange(len(radii)))
    ax.set_yticklabels([f"{r:.2f}" for r in radii])
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("lumen radius [mm]")
    ax.set_title("(e) rotation heatmap: radius x arc length", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    width = 0.8 / len(radii)
    x = np.arange(7)
    for k, r in enumerate(radii):
        pa = np.array([np.nanmax(by_radius[r][f"per_actuator_relative_{i}"]) for i in range(1, 8)])
        ax.bar(x + k * width, pa, width, color=cmap[r], label=f"{r:.2f} mm")
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(ACTUATOR_LABELS)
    ax.set_xlabel("actuator column of J")
    ax.set_ylabel("max relative column gap")
    ax.set_title("(f) per-actuator gap (max over path)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 0]
    g_med = np.array([np.nanmedian(by_radius[r]["gain_ratio_min"]) for r in radii])
    g_lo = np.array([np.nanpercentile(by_radius[r]["gain_ratio_min"], 10) for r in radii])
    ax.plot(rr, g_med, "o-", lw=2, label="median")
    ax.plot(rr, g_lo, "v:", lw=1.2, label="10th pct")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel(r"$\sigma_{min}(J_{contact})/\sigma_{min}(J_{free})$")
    ax.set_title("(g) weakest-direction gain vs radius", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    ax = axes[2, 1]
    frac_contact = np.array([
        np.mean(by_radius[r]["frobenius_relative"] > GAP_CONTACT_THRESHOLD) for r in radii
    ])
    frac_rot = np.array([
        np.mean(by_radius[r]["principal_angle_3_deg"] > ROTATION_THRESHOLD_DEG) for r in radii
    ])
    ax.plot(rr, 100 * frac_contact, "o-", lw=2,
            label=f"gap > {GAP_CONTACT_THRESHOLD*100:.0f}%")
    ax.plot(rr, 100 * frac_rot, "s--", lw=2,
            label=f"principal angle > {ROTATION_THRESHOLD_DEG:.0f}°")
    ax.set_xlabel("lumen radius [mm]")
    ax.set_ylabel("fraction of planned path [%]")
    ax.set_title("(h) contact / rotation fraction vs radius", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    ax = axes[2, 2]
    all_clear, all_gap = [], []
    for r in radii:
        clearance = r - r_beam_mm
        n = by_radius[r]["frobenius_relative"].size
        all_clear.append(np.full(n, clearance))
        all_gap.append(by_radius[r]["frobenius_relative"])
    all_clear = np.concatenate(all_clear)
    all_gap = np.concatenate(all_gap)
    ax.scatter(all_clear, 100 * all_gap, s=8, alpha=0.35, color="teal")
    for r in radii:
        clearance = r - r_beam_mm
        ax.plot(clearance, 100 * np.nanmedian(by_radius[r]["frobenius_relative"]),
                "o", color="crimson", ms=7)
    ax.set_yscale("log")
    ax.set_xlabel("nominal wall clearance  R - r_beam  [mm]")
    ax.set_ylabel("relative gap [%]")
    ax.set_title("(i) gap vs nominal wall clearance", fontsize=10)

    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.055, right=0.99,
                        hspace=0.40, wspace=0.30)
    out = output_dir / "test_a_jacobian_divergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    metrics["radii_mm"] = radii
    metrics["gap_median_pct"] = (100 * med).round(4).tolist()
    metrics["gap_rms_pct"] = (100 * rms).round(4).tolist()
    metrics["gap_max_pct"] = (100 * mx).round(4).tolist()
    metrics["principal_angle_max_deg"] = [
        float(np.nanmax(by_radius[r]["principal_angle_3_deg"])) for r in radii
    ]
    metrics["principal_angle_median_deg"] = [
        float(np.nanmedian(by_radius[r]["principal_angle_3_deg"])) for r in radii
    ]
    metrics["kappa_contact_median"] = kc_med.round(1).tolist()
    metrics["kappa_contact_max"] = kc_max.round(1).tolist()
    metrics["kappa_free_median"] = float(np.nanmedian(kn_med))
    metrics["gain_ratio_min_median"] = g_med.round(4).tolist()
    metrics["frac_path_in_contact_pct"] = (100 * frac_contact).round(1).tolist()
    metrics["frac_path_rotated_pct"] = (100 * frac_rot).round(1).tolist()
    return metrics, str(out)


# --------------------------------------------------------------------------
# figure 2 — beam shape vs lumen
# --------------------------------------------------------------------------
def figure_beam_shape(by_radius, args, exp_cfg, pack, out_root,
                      nominal_radius_mm: float, title_suffix: str) -> str:
    from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
    from proper_research.experiments import run_contact_study as RCS
    from proper_research.simulation.simulations.model_factory import build_model_bundle
    from proper_research.simulation.simulations.initial_conditions import make_initial_poses
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (  # noqa: E501
        load_configuration_reference,
    )

    ref = load_configuration_reference(
        resolve_reference_dir(out_root, getattr(args, 'reference', None)),
        require_planned_beam_feasible=False,
    )
    states = np.asarray(ref.state, dtype=float)
    arc = 1.0e3 * (np.asarray(ref.path_coordinate_m, float)
                   - float(ref.path_coordinate_m[0]))
    pivot, start, L0, dt = make_initial_poses()

    radii = list(by_radius.keys())
    nominal = min(radii, key=lambda r: abs(r - nominal_radius_mm))
    apex_local = int(np.nanargmax(by_radius[nominal]["frobenius_relative"]))
    apex_sample = int(by_radius[nominal]["sample"][apex_local])
    show_samples = np.linspace(0, len(states) - 1, 6).astype(int)

    def beam_xyz(bundle, state):
        ad = build_diagnostic_adapter(beam_model=bundle.models["contact"],
                                      controller_pack=pack)
        ad.forward_output(np.asarray(state, float), commit=True)
        pc = ad.beam_output_fn.forward_adapter.last_p_centerline
        return None if pc is None else np.asarray(pc, float).T

    bundles = {}
    for r in radii:
        scale = r / nominal_radius_mm
        spec = RCS.AnatomySpec(name=f"r{r:.2f}", radius_scale=scale)
        lumen = RCS.scale_lumen_config(exp_cfg.lumen, spec)
        bundles[r] = build_model_bundle(pivot_point=pivot, L0=L0, lumen_cfg=lumen,
                                        plant_contact=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))
    suffix = f", {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Test A — beam shape vs lumen wall{suffix}", fontsize=13, y=0.99)

    def draw_lumen(ax, bundle, label, color):
        C = np.asarray(bundle.lumen_C, float)
        R = np.asarray(bundle.lumen_R, float)
        t = np.gradient(C[:, :2], axis=0)
        t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        w1 = C[:, :2] + nrm * R[:, None]
        w2 = C[:, :2] - nrm * R[:, None]
        ax.plot(1e3 * C[:, 0], 1e3 * C[:, 1], "--", color=color, lw=1, alpha=0.7)
        ax.plot(1e3 * w1[:, 0], 1e3 * w1[:, 1], "-", color=color, lw=1.3, label=label)
        ax.plot(1e3 * w2[:, 0], 1e3 * w2[:, 1], "-", color=color, lw=1.3)

    ax = axes[0]
    draw_lumen(ax, bundles[nominal], f"lumen wall ({nominal:.2f} mm)", "0.4")
    dp = np.asarray(ref.desired_position_m, float)
    ax.plot(1e3 * dp[:, 0], 1e3 * dp[:, 1], ":", color="tab:blue", lw=1.2,
            label="planned tip path")
    cols = cm.plasma(np.linspace(0, 0.9, len(show_samples)))
    for s, c in zip(show_samples, cols):
        b = beam_xyz(bundles[nominal], states[s])
        if b is None:
            continue
        ax.plot(1e3 * b[:, 0], 1e3 * b[:, 1], "-o", color=c, ms=3, lw=1.6,
                label=f"s = {arc[s]:.1f} mm")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"(a) nominal lumen {nominal:.2f} mm — beam sweeps the plan", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    rc = cm.viridis(np.linspace(0.05, 0.9, len(radii)))
    for r, c in zip(radii, rc):
        draw_lumen(ax, bundles[r], f"R = {r:.2f} mm", c)
        b = beam_xyz(bundles[r], states[apex_sample])
        if b is not None:
            ax.plot(1e3 * b[:, 0], 1e3 * b[:, 1], "-o", color=c, ms=3, lw=1.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"(b) apex (s = {arc[apex_sample]:.1f} mm): beam pressed "
                 "into the wall as R tightens", fontsize=10)
    ax.legend(fontsize=7)

    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.99, wspace=0.18)
    out = args.output_dir / "test_a_beam_shape_lumen.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
def main() -> int:
    args = _arguments()
    offline_csv = args.output_dir / "offline" / "jacobian_divergence.csv"
    offline_json = args.output_dir / "offline" / "jacobian_divergence_summary.json"
    if not offline_csv.exists():
        print(f"missing {offline_csv} - run run_contact_study.py --stage offline first",
              file=sys.stderr)
        return 1
    exp_cfg, bundle0, pack, out_root = resolve_planning_context_from_args(args)
    nominal_radius_mm = args.nominal_radius_mm
    if nominal_radius_mm is None:
        nominal_radius_mm = 1000.0 * float(np.mean(np.asarray(bundle0.lumen_R, dtype=float)))
        print(f"[geometry] --nominal-radius-mm not given; using the built bundle's own "
              f"radius: {nominal_radius_mm:.4f} mm", flush=True)

    rows = load_rows(offline_csv)
    by_radius = group_by_radius(rows, nominal_radius_mm)
    summary = json.loads(offline_json.read_text()) if offline_json.exists() else {}
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (  # noqa: E501
        load_configuration_reference,
    )
    ref = load_configuration_reference(
        resolve_reference_dir(out_root, getattr(args, 'reference', None)),
        require_planned_beam_feasible=False,
    )

    metrics, fig1 = figure_divergence(
        by_radius, ref, args.output_dir, nominal_radius_mm, args.r_beam_mm,
        args.title_suffix,
    )
    print("wrote", fig1)

    try:
        fig2 = figure_beam_shape(
            by_radius, args, exp_cfg, pack, out_root, nominal_radius_mm,
            args.title_suffix,
        )
        print("wrote", fig2)
        metrics["beam_shape_figure"] = fig2
    except Exception as exc:  # keep the metrics even if the model build fails
        print(f"beam-shape figure failed: {exc!r}", file=sys.stderr)
        metrics["beam_shape_figure_error"] = repr(exc)

    (args.output_dir / "test_a_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", args.output_dir / "test_a_metrics.json")

    radii = metrics["radii_mm"]
    lines = [
        "| lumen radius | clearance R-r_beam | gap median / rms / max [%] | "
        "principal angle median / max [deg] | kappa(J_contact) median / max | "
        "sigma_min ratio (median) | path in contact / rotated [%] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(radii):
        lines.append(
            f"| {r:.2f} mm | {r - args.r_beam_mm:.2f} mm | "
            f"{metrics['gap_median_pct'][i]:.2f} / {metrics['gap_rms_pct'][i]:.1f} / "
            f"{metrics['gap_max_pct'][i]:.1f} | "
            f"{metrics['principal_angle_median_deg'][i]:.1f} / "
            f"{metrics['principal_angle_max_deg'][i]:.1f} | "
            f"{metrics['kappa_contact_median'][i]:.0f} / {metrics['kappa_contact_max'][i]:.0f} | "
            f"{metrics['gain_ratio_min_median'][i]:.3f} | "
            f"{metrics['frac_path_in_contact_pct'][i]:.0f} / {metrics['frac_path_rotated_pct'][i]:.0f} |"
        )
    (args.output_dir / "test_a_summary_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", args.output_dir / "test_a_summary_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
