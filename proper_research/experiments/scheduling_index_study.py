"""Test B, generalized: the Jacobian-variation index S across a radius sweep.

Reusable, geometry-agnostic version of the one-off
``results/contact_study_radius_p30_m50/scripts/compute_S.py``. Works on
whatever vessel ``proper_research.planning.planning_context`` currently builds
(``planning_context.py`` selects the geometry), so re-running this after
switching to a new vessel just needs a new ``--output-dir``.

The question
------------
``S`` is the index the offline LTI/LTV scheduling rungs use to decide "how much
does the Jacobian change along this plan": ``S = max_j ||J_j - J_0||_2 /
||J_0||_2`` over the reference, evaluated with the **contact-free** provider
(the model the offline schedule is actually built from — it has no lumen
query, so it cannot see the radius sweep at all). Test A's prediction was that
tightening the lumen should not move S; this script measures that directly,
plus the quantity that *does* move: ``mu_min``, the P2 contraction margin
between the contact-aware plant and the contact-free model the schedule
linearises around.

What it computes, per radius
-----------------------------
S_free        max_j ||J_j - J_0||_2 / ||J_0||_2         (contact-free provider)
S_contact     same, contact-aware provider
S_succ        sum_j ||J_{j+1} - J_j||_2 / ||J_0||_2      (path length in J-space, contact-free)
S_meanrel     mean_j ||J_j - Jbar||_2 / ||Jbar||_2        (contact-free, about the mean)
mu(s)         lambda_min(sym(J_contact(s) J_free(s)^+))   sampled along the path
mu_min/median statistics of mu(s) -- the P2 contraction test

Outputs (into --output-dir)
----------------------------
scheduling_index_S.json          one dict per radius: summary numbers (as before)
scheduling_index_S_samples.csv   per-state rows for every radius: arc length,
                                  gap_rel, mu (on the mu-stride subsample) --
                                  what plot_scheduling_index.py plots

Usage
-----
    python -m proper_research.experiments.scheduling_index_study \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.15 1.0 0.70 0.50 0.40 \\
        --stride 4 --mu-stride 5

    python -m proper_research.experiments.plot_scheduling_index \\
        --study-dir results/contact_study_radius_p30_m70
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent / "controllers"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from proper_research.experiments import run_contact_study as RCS  # noqa: E402
from proper_research.experiments.pinned_planning_context import (  # noqa: E402
    add_geometry_arguments,
    resolve_planning_context_from_args,
    resolve_reference_dir,
)


def _spectral_norm(matrix: Array) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), 2))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test B: the Jacobian-scheduling index S across a radius sweep, "
                    "reusable across vessels (the vessel comes from planning_context.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Where to write scheduling_index_S.json / _samples.csv.")
    parser.add_argument("--reference", type=Path, default=None,
                         help="Override the reference directory (default: the "
                              "planning_context out_root's time-parameterized path).")
    parser.add_argument("--radius-scales", type=float, nargs="+",
                         default=(1.0, 0.75, 0.55, 0.40),
                         help="Lumen radius multipliers (centreline held fixed).")
    parser.add_argument("--stride", type=int, default=4,
                         help="Subsample the reference for the J-schedule (S is a max/sum measure).")
    parser.add_argument("--mu-stride", type=int, default=5,
                         help="Further subsample (of the strided states) for mu_min, which "
                              "needs an extra pseudo-inverse + eig per state.")
    parser.add_argument("--jacobian-mode", default="fast",
                         help="Passed through to beam_jacobian_providers (default 'fast').")
    parser.add_argument("--damping", type=float, default=1e-9,
                         help="Tikhonov damping for the free-Jacobian pseudo-inverse in mu_min.")
    add_geometry_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    from proper_research.simulation.simulations.model_factory import build_model_bundle
    from proper_research.simulation.simulations.initial_conditions import make_initial_poses
    from proper_research.controllers import beam_jacobian_providers as providers

    harness = RCS._load("compare_controllers")
    base_module = harness.load_base_module()

    exp_cfg, bundle0, pack0, out_root = resolve_planning_context_from_args(args)
    RCS._assert_plant_contact(bundle0, "the baseline planning context")
    reference_dir = resolve_reference_dir(out_root, args.reference)
    reference = base_module.load_configuration_reference(
        reference_dir, require_planned_beam_feasible=False,
    )
    nominal_radius_mm = 1000.0 * float(np.mean(np.asarray(bundle0.lumen_R, dtype=float)))
    pivot, start, L0, dt = make_initial_poses()

    states = np.asarray(reference.state, dtype=float)[:: max(1, int(args.stride))]
    sample_indices = np.arange(0, reference.sample_count, max(1, int(args.stride)))[: states.shape[0]]
    arc_m = np.asarray(reference.path_coordinate_m, dtype=float)[sample_indices]
    arc_mm = 1.0e3 * (arc_m - float(reference.path_coordinate_m[0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for scale in args.radius_scales:
        name = f"radius_x{scale:g}".replace(".", "p")
        started = time.perf_counter()
        spec = RCS.AnatomySpec(name=name, radius_scale=float(scale))
        lumen = RCS.scale_lumen_config(exp_cfg.lumen, spec)
        bundle = build_model_bundle(pivot_point=pivot, L0=L0, lumen_cfg=lumen, plant_contact=True)
        radius_mm = round(1000.0 * float(np.mean(np.asarray(bundle.lumen_R, dtype=float))), 4)

        contact_provider, free_provider = providers.contact_and_contact_free(
            bundle=bundle, controller_pack=pack0, jacobian_mode=args.jacobian_mode,
        )

        Jf = np.stack([np.asarray(free_provider(s), dtype=float).reshape(3, 7) for s in states])
        Jc = np.stack([np.asarray(contact_provider(s), dtype=float).reshape(3, 7) for s in states])
        J0f, J0c = Jf[0], Jc[0]
        n0f, n0c = _spectral_norm(J0f), _spectral_norm(J0c)

        S_free = max(_spectral_norm(J - J0f) for J in Jf) / n0f
        S_contact = max(_spectral_norm(J - J0c) for J in Jc) / n0c
        S_succ = sum(
            _spectral_norm(Jf[i + 1] - Jf[i]) for i in range(len(Jf) - 1)
        ) / n0f
        Jbar = Jf.mean(axis=0)
        nbar = _spectral_norm(Jbar)
        S_meanrel = float(np.mean([_spectral_norm(J - Jbar) for J in Jf])) / nbar

        mu_stride = max(1, int(args.mu_stride))
        mu_indices = np.arange(0, len(states), mu_stride)
        mu_values: list[float] = []
        gap_rel_all = np.array([
            float(np.linalg.norm(jc - jn) / max(float(np.linalg.norm(jc)), 1e-30))
            for jc, jn in zip(Jc, Jf)
        ])
        for i in mu_indices:
            jc, jn = Jc[i], Jf[i]
            pinv = jn.T @ np.linalg.solve(
                jn @ jn.T + float(args.damping) * np.eye(3), np.eye(3)
            )
            mix = jc @ pinv
            mu = float(np.min(np.linalg.eigvalsh(0.5 * (mix + mix.T))))
            mu_values.append(mu)
            sample_rows.append(dict(
                anatomy=name, radius_scale=scale, radius_mm=radius_mm,
                sample=int(sample_indices[i]), arc_mm=float(arc_mm[i]),
                gap_rel=float(gap_rel_all[i]), mu=mu,
            ))
        # also log gap_rel at every strided state (finer than mu) for the trace plot
        for i in range(len(states)):
            if i in set(mu_indices.tolist()):
                continue
            sample_rows.append(dict(
                anatomy=name, radius_scale=scale, radius_mm=radius_mm,
                sample=int(sample_indices[i]), arc_mm=float(arc_mm[i]),
                gap_rel=float(gap_rel_all[i]), mu=math.nan,
            ))

        mu_values_arr = np.asarray(mu_values, dtype=float)
        row = dict(
            anatomy=name, radius_scale=float(scale), radius_mm=radius_mm,
            nominal_radius_mm=round(nominal_radius_mm, 4),
            n_states=int(len(states)), n_mu_states=int(len(mu_values)),
            S_free=S_free, S_contact=S_contact, S_succ=S_succ, S_meanrel=S_meanrel,
            mu_min=float(np.min(mu_values_arr)), mu_median=float(np.median(mu_values_arr)),
            mu_mean=float(np.mean(mu_values_arr)),
            frac_mu_negative=float(np.mean(mu_values_arr < 0.0)),
            gap_rel_median=float(np.median(gap_rel_all)),
            gap_rel_max=float(np.max(gap_rel_all)),
            elapsed_s=float(time.perf_counter() - started),
        )
        summary_rows.append(row)
        print(json.dumps(row), flush=True)

        # write incrementally so a killed run still leaves partial results
        (args.output_dir / "scheduling_index_S.json").write_text(
            json.dumps(summary_rows, indent=2)
        )
        if sample_rows:
            header = list(sample_rows[0].keys())
            with (args.output_dir / "scheduling_index_S_samples.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(sample_rows)

    print("wrote", args.output_dir / "scheduling_index_S.json")
    print("wrote", args.output_dir / "scheduling_index_S_samples.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
