"""The contact-Jacobian counterpart to check_free_jacobian_path_variation.py.

That script shows J_free's own per-state trace along the path is large but
exactly radius-invariant. This script computes the SAME trace for J_contact,
at the same radii, and overlays both: unlike J_free, J_contact's path-
variation trace is NOT radius-invariant -- it grows with radius, specifically
where the two Jacobians already disagree (the bend). Panel (b) isolates that
extra, contact-only variation directly.

Usage
-----
    python -m proper_research.experiments.check_contact_jacobian_path_variation \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.15 1.0 0.70 0.50 0.40 --stride 4 \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5
"""
from __future__ import annotations

import argparse
import sys
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

from proper_research.experiments import run_contact_study as RCS  # noqa: E402
from proper_research.experiments.pinned_planning_context import (  # noqa: E402
    add_geometry_arguments,
    resolve_planning_context_from_args,
    resolve_reference_dir,
)


def spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), 2))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether J_contact's path variation depends on radius "
                    "(unlike J_free's).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--radius-scales", type=float, nargs="+",
                         default=(1.15, 1.0, 0.70, 0.50, 0.40))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--title-suffix", default="")
    add_geometry_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (  # noqa: E501
        load_configuration_reference,
    )
    from proper_research.controllers import beam_jacobian_providers as providers
    from proper_research.simulation.simulations.model_factory import build_model_bundle
    from proper_research.simulation.simulations.initial_conditions import make_initial_poses

    exp_cfg, bundle0, pack0, out_root = resolve_planning_context_from_args(args)
    reference = load_configuration_reference(
        resolve_reference_dir(out_root, args.reference),
        require_planned_beam_feasible=False,
    )
    states = np.asarray(reference.state, dtype=float)[:: max(1, args.stride)]
    arc_m = np.asarray(reference.path_coordinate_m, dtype=float)[:: max(1, args.stride)]
    arc_mm = 1.0e3 * (arc_m - float(arc_m[0]))
    pivot, start, L0, dt = make_initial_poses()

    contact_traces: dict[float, np.ndarray] = {}
    free_traces: dict[float, np.ndarray] = {}
    radius_mm_by_scale: dict[float, float] = {}
    for scale in args.radius_scales:
        spec = RCS.AnatomySpec(name=f"r{scale:g}", radius_scale=scale)
        lumen = RCS.scale_lumen_config(exp_cfg.lumen, spec)
        bundle = build_model_bundle(pivot_point=pivot, L0=L0, lumen_cfg=lumen, plant_contact=True)
        radius_mm_by_scale[scale] = 1000.0 * float(np.mean(np.asarray(bundle.lumen_R, dtype=float)))
        contact_provider, free_provider = providers.contact_and_contact_free(
            bundle=bundle, controller_pack=pack0,
        )
        Jc = np.stack([np.asarray(contact_provider(s), dtype=float).reshape(3, 7) for s in states])
        Jf = np.stack([np.asarray(free_provider(s), dtype=float).reshape(3, 7) for s in states])
        Jc0, Jf0 = Jc[0], Jf[0]
        nc0, nf0 = spectral_norm(Jc0), spectral_norm(Jf0)
        contact_traces[scale] = np.array([spectral_norm(J - Jc0) for J in Jc]) / nc0
        free_traces[scale] = np.array([spectral_norm(J - Jf0) for J in Jf]) / nf0
        print(f"radius_scale={scale:g} ({radius_mm_by_scale[scale]:.2f} mm): "
              f"S_contact={contact_traces[scale].max():.4f}  "
              f"S_free={free_traces[scale].max():.10f}")

    ref_free = free_traces[args.radius_scales[0]]
    for scale in args.radius_scales[1:]:
        d = float(np.max(np.abs(free_traces[scale] - ref_free)))
        print(f"max|S_free_trace(r={scale:g}) - S_free_trace(r={args.radius_scales[0]:g})| = {d:.3e}")

    colors = cm.viridis(np.linspace(0.05, 0.9, len(args.radius_scales)))
    cmap = {s: c for s, c in zip(args.radius_scales, colors)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"J_contact's path variation is NOT radius-invariant, "
                 f"unlike J_free's{suffix}", fontsize=12, y=0.99)

    ax = axes[0]
    for scale in args.radius_scales:
        ax.plot(arc_mm, contact_traces[scale], "-o", ms=3, lw=1.6,
                color=cmap[scale], label=f"{radius_mm_by_scale[scale]:.2f} mm")
    ax.plot(arc_mm, ref_free, "--", lw=2.0, color="black", alpha=0.8,
            label="J_free (all radii identical)")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"$\|J(s) - J(0)\| \,/\, \|J(0)\|$")
    ax.set_title("(a) contact trace spreads out by radius; free trace does not", fontsize=10)
    ax.legend(fontsize=8, title="lumen radius")

    ax = axes[1]
    for scale in args.radius_scales:
        extra = contact_traces[scale] - ref_free
        ax.plot(arc_mm, extra, "-o", ms=3, lw=1.6, color=cmap[scale],
                label=f"{radius_mm_by_scale[scale]:.2f} mm")
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"contact trace $-$ free trace")
    ax.set_title("(b) the contact-only extra variation, isolated", fontsize=10)
    ax.legend(fontsize=8, title="lumen radius")

    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.07, right=0.98, wspace=0.28)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "test_b_contact_jacobian_path_variation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
