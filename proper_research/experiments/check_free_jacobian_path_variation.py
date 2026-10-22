"""Does J_free vary along the path, and does that variation depend on radius?

Answers a specific confusion Test B's "S_free is radius-invariant" headline
can cause: that finding is about radius, not about the path. This script
plots the FULL per-state trace (not just its max = S_free) at several radii,
confirms the traces are bit-for-bit identical across radius, and plots the
physical drivers (magnet position, insertion length) alongside it so the
large, real, path-driven variation is visible directly.

Usage
-----
    python -m proper_research.experiments.check_free_jacobian_path_variation \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.15 1.0 0.40 --stride 4 \\
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
        description="Check whether J_free's path variation depends on radius.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--radius-scales", type=float, nargs="+", default=(1.15, 1.0, 0.40))
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
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

    exp_cfg, bundle0, pack0, out_root = resolve_planning_context_from_args(args)
    reference = load_configuration_reference(
        resolve_reference_dir(out_root, args.reference),
        require_planned_beam_feasible=False,
    )
    states = np.asarray(reference.state, dtype=float)[:: max(1, args.stride)]
    arc_m = np.asarray(reference.path_coordinate_m, dtype=float)[:: max(1, args.stride)]
    arc_mm = 1.0e3 * (arc_m - float(arc_m[0]))

    dh = pack0["robot_dh"]
    T_F_M = np.asarray(pack0["T_F_M"], dtype=float).reshape(4, 4)

    traces: dict[float, np.ndarray] = {}
    for scale in args.radius_scales:
        spec = RCS.AnatomySpec(name=f"r{scale:g}", radius_scale=scale)
        lumen = RCS.scale_lumen_config(exp_cfg.lumen, spec)
        pivot, start, L0, dt = make_initial_poses()
        bundle = build_model_bundle(pivot_point=pivot, L0=L0, lumen_cfg=lumen, plant_contact=True)
        _, free_provider = providers.contact_and_contact_free(bundle=bundle, controller_pack=pack0)

        Jf = np.stack([np.asarray(free_provider(s), dtype=float).reshape(3, 7) for s in states])
        J0 = Jf[0]
        n0 = spectral_norm(J0)
        trace = np.array([spectral_norm(J - J0) for J in Jf]) / n0
        traces[scale] = trace
        print(f"radius_scale={scale:g}: S_free (max of trace) = {trace.max():.10f}, "
              f"trace[0]={trace[0]:.6f}, n_states={len(trace)}")

    ref_trace = traces[args.radius_scales[0]]
    for scale in args.radius_scales[1:]:
        max_abs_diff = float(np.max(np.abs(traces[scale] - ref_trace)))
        print(f"max|trace(r={scale:g}) - trace(r={args.radius_scales[0]:g})| = {max_abs_diff:.3e}")

    magnet_pos = []
    for state in states:
        T_R_M = urik.forward_kinematics(state[:6], dh, T_F_M).T_R_target
        magnet_pos.append(np.asarray(T_R_M[:3, 3], dtype=float))
    magnet_pos = np.stack(magnet_pos)
    magnet_travel_mm = 1.0e3 * np.linalg.norm(magnet_pos - magnet_pos[0], axis=1)
    insertion_mm = 1.0e3 * states[:, 6]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"Does J_free vary along the path? Yes -- radius is the only thing it's blind to{suffix}",
                 fontsize=12, y=0.99)

    ax = axes[0]
    colors = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]
    for scale, color in zip(args.radius_scales, colors):
        ax.plot(arc_mm, traces[scale], "-o", ms=3, lw=1.6, color=color,
                label=f"radius x{scale:g}", alpha=0.8)
    ax.axhline(ref_trace.max(), color="crimson", ls="--", lw=1,
               label=f"S_free = {ref_trace.max():.4f}")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"$\|J_{free}(s) - J_{free}(0)\| \,/\, \|J_{free}(0)\|$")
    ax.set_title("(a) J_free's path variation (radii overlaid, exactly)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax2 = ax.twinx()
    l1, = ax.plot(arc_mm, magnet_travel_mm, "-o", ms=3, lw=1.6, color="darkorange",
                  label="magnet travel since start [mm]")
    l2, = ax2.plot(arc_mm, insertion_mm, "-s", ms=3, lw=1.6, color="steelblue",
                   label="insertion length L [mm]")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("magnet position change [mm]", color="darkorange")
    ax2.set_ylabel("insertion length [mm]", color="steelblue")
    ax.set_title("(b) what's actually moving: magnet pose + insertion length", fontsize=10)
    ax.legend(handles=[l1, l2], fontsize=8, loc="upper left")

    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.07, right=0.90, wspace=0.4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "test_b_free_jacobian_path_variation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
