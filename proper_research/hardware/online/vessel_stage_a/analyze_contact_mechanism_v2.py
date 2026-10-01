#!/usr/bin/env python3
"""Whole-beam contact-mechanism analysis (2026-09-28, v2).

Supersedes the tip-only approximation in `analyze_contact_mechanism.py` for
the NEXT round of contact/no-contact/frozen vessel closed-loop MPC hardware
runs, once they're re-recorded with the new per-tick log fields added this
session (see `delay_aware_mpc.py`/`worker_process.py`/
`process_isolated_adapter.py`'s 2026-09-28 instrumentation-pass comments):

  - `jacobian_used` (3,7): the EXACT Jacobian the live QP used that tick,
    not a post-hoc schedule-index reconstruction (that approach measured
    ~80-97% median relative error against the live-logged condition number
    in this session's v1 analysis -- this field fixes that at the source).
  - `predicted_states` (N,7): the full MPC horizon state prediction.
  - `magnet_xyz_m`, `dual_y`, `qp_objective`/`qp_status`/`qp_iterations`/
    `qp_primal_residual`/`qp_dual_residual`: additional per-tick QP
    diagnostics, all new.

v1's central finding (this session, 2026-09-28) was that NEITHER the
contact-aware nor the no-contact run ever brought the TIP within 2mm of the
wall. The user's hypothesis for the rerun: contact may be happening on the
beam's STEM (some interior arc-length point), which pure tip-clearance
can't see at all. This script's core addition is exactly that: reconstruct
the FULL beam centreline every tick (model-inferred at the MEASURED
(q, L) state -- there is no live vision ground truth for the whole beam,
see this session's decision to accept that tradeoff rather than invest in
live full-shape vision) and search the WHOLE beam for the closest wall
approach, not just the tip.

Two DELIBERATELY SEPARATE contact labels (do not conflate, per user spec):
  - `c_min <= PHYSICAL_CONTACT_TOL_M` (default 0.2mm): physical contact.
  - `c_min <= CONTACT_BAND_M` (default 0.5mm, matches this project's own
    `--mpc-wall-avoidance-margin-mm` default): contact-MODEL-ACTIVATION
    region -- the contact-aware model may already be doing something
    different here even before physical contact.

Usage (once new-schema data exists)
------------------------------------
    python -m proper_research.hardware.online.vessel_stage_a.analyze_contact_mechanism_v2 \\
        --contact-run close_loop_logs/myrun/<new contact run dir> \\
        --nocontact-run close_loop_logs/myrun/<new no-contact run dir>

NOTE: this script has NOT been run end-to-end against real data -- there is
none yet (that's the whole point of the rerun it's written for). It IS
validated against the new IPC/field plumbing directly (see this session's
`smoke_new_fields.py`-style check: real `MPCWorkerHandle.try_solve()` calls
confirmed `jacobian_used` shape (3,7) finite, `predicted_states` shape
(N,7), `dual_y` shape matching the expected constraint-row count, all
without touching hardware). Field names/shapes below are believed correct
against the actual plumbing just implemented, not just the investigation
that preceded it -- but this file itself has not been exercised against a
real `predicted_beam_positions.jsonl`. Treat the exact key names as
provisional until run once against real output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from proper_research.hardware.online.vessel_stage_a.analyze_contact_mechanism import (
    LUMEN_FILE, R_BEAM_M, Z_RAISE_M, build_adapters, load_lumen, predict_centreline,
)

REPO = Path("/home/jack/Proper-Research")
CONTACT_BAND_M = 0.5e-3          # matches --mpc-wall-avoidance-margin-mm default
PHYSICAL_CONTACT_TOL_M = 0.2e-3  # measurement/reconstruction tolerance, per user spec


# ===========================================================================
# loading (new schema)
# ===========================================================================
def load_run(run_dir: Path) -> list[dict[str, Any]]:
    """`predicted_beam_positions.jsonl` is the per-tick record of record for
    this analysis (it carries `z_meas`, `jacobian_used`, `predicted_states`,
    `u0`/command -- see `process_isolated_adapter.py`'s `row` dict). One
    dict per control tick, in tick order."""
    path = run_dir / "predicted_beam_positions.jsonl"
    rows = [json.loads(line) for line in path.open()]
    for i, r in enumerate(rows):
        if r.get("jacobian_used") is None or r.get("predicted_states") is None:
            raise ValueError(
                f"{run_dir}: row {i} is missing 'jacobian_used'/'predicted_states' -- "
                "this run predates the 2026-09-28 instrumentation pass, or was a "
                "deadline-miss/error tick with no solve. This script requires the "
                "new schema; rerun with the updated controller code."
            )
    return rows


# ===========================================================================
# whole-beam contact geometry
# ===========================================================================
def beam_contact_profile(adapter, lumen_query, q6: np.ndarray, L_m: float):
    """Forward-solves the FULL centreline at the measured (q6, L_m) state
    (model-inferred, not vision-measured -- see module docstring), then
    finds the closest wall approach anywhere along it, not just at the tip.

    Returns:
      c_min_m: float -- min over s of (local wall radius - dist-to-
        centreline - r_beam), the same sign convention as v1's tip-only
        d_w (positive = free lumen, 0 = beam surface at the wall, negative
        = geometric penetration).
      s_frac: float in [0, 1] -- arc-length fraction (from the beam base)
        of the closest-approach node.
      tangent_c: (3,) unit vector -- LOCAL beam tangent AT the closest-
        approach node (central difference, not the tip chord).
      normal_c: (3,) unit vector -- LOCAL wall outward normal at that same
        node (from LumenQuery, radially outward = direction of decreasing
        clearance).
      centreline: (M,3) -- the full solved centreline, for anyone who wants
        the raw shape (e.g. to plot it inside the vessel).
    """
    centreline = predict_centreline(adapter, q6, L_m)  # (M, 3), base -> tip
    seg = np.diff(centreline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1] if arc[-1] > 1e-9 else 1.0

    M = centreline.shape[0]
    delta, Rloc, _, grad_delta, _ = lumen_query.closest_many_with_gradients(centreline, window=None)
    c = Rloc - delta - R_BEAM_M  # (M,), same sign convention as v1's d_w
    # 2026-09-30 fix (first real end-to-end run against live data, per this
    # module's own "not yet exercised" caveat): node 0 of `centreline` is
    # the beam's RIGID ANCHOR (== the fixed beam-base pivot the whole
    # session's magnet-exclusion geometry is built from) -- it does not
    # move with q/insertion at all, so its clearance to the nearest lumen
    # sample is a CONSTANT, tick-independent artifact of the anchor's own
    # fixed position, not a measurement of beam-wall contact. Found via a
    # real replay where every single tick of a run reported an identical
    # c_min with s_frac==0.0 -- the anchor was trivially winning every
    # search. Excluded from the closest-approach search; genuine contact
    # is only meaningful over the deflectable part of the beam.
    if M > 1:
        i_min = int(np.argmin(c[1:])) + 1
    else:
        i_min = int(np.argmin(c))
    c_min = float(c[i_min])
    s_frac = float(arc[i_min] / total_len)
    normal_c = grad_delta[i_min]

    if i_min == 0:
        tangent_c = centreline[1] - centreline[0]
    elif i_min == M - 1:
        tangent_c = centreline[-1] - centreline[-2]
    else:
        tangent_c = centreline[i_min + 1] - centreline[i_min - 1]  # central difference
    norm = np.linalg.norm(tangent_c)
    tangent_c = tangent_c / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    return c_min, s_frac, tangent_c, normal_c, centreline, c, arc, total_len


# ===========================================================================
# per-run analysis
# ===========================================================================
def analyze_run(run_dir: Path, adapter, lumen_query, label: str) -> dict[str, Any]:
    rows = load_run(run_dir)
    n = len(rows)

    c_min_mm = np.empty(n)
    s_c_frac = np.empty(n)
    theta_wall_deg_stem = np.empty(n)   # at s_c, NOT the tip
    tip_m = np.empty((n, 3))
    z_meas_all = np.empty((n, 7))
    u0_all = np.empty((n, 7))
    jacobian_used_all = np.empty((n, 3, 7))

    print(f"[v2] {label}: forward-solving {n} full centrelines (measured q,L per tick)...")
    for i, r in enumerate(rows):
        z_meas = np.asarray(r["z_meas"], dtype=float)
        z_meas_all[i] = z_meas
        u0_all[i] = np.asarray(r["u0"], dtype=float)
        jacobian_used_all[i] = np.asarray(r["jacobian_used"], dtype=float)
        tip_m[i] = np.asarray(r["measured_beam_position_m"], dtype=float)

        c_min, s_frac, tangent_c, normal_c, *_ = beam_contact_profile(
            adapter, lumen_query, z_meas[:6], float(z_meas[6]),
        )
        c_min_mm[i] = c_min * 1000.0
        s_c_frac[i] = s_frac
        cosang = float(np.clip(abs(np.dot(tangent_c, normal_c)), -1.0, 1.0))
        theta_n = np.degrees(np.arccos(cosang))
        theta_wall_deg_stem[i] = 90.0 - theta_n
        if (i + 1) % 100 == 0:
            print(f"[v2]   {label}: {i + 1}/{n}")

    physical_contact = c_min_mm <= (PHYSICAL_CONTACT_TOL_M * 1000.0)
    activation_region = c_min_mm <= (CONTACT_BAND_M * 1000.0)

    # Tip-frame wall geometry + REAL-Jacobian normal/tangential decomposition
    # (fixes v1's ~80-97% schedule-index error -- this uses jacobian_used,
    # the exact per-tick matrix the live QP solved with).
    delta, Rloc, _, grad_delta_tip, _ = lumen_query.closest_many_with_gradients(tip_m, window=None)
    d_w_tip_mm = (Rloc - delta - R_BEAM_M) * 1000.0
    n_tip = grad_delta_tip  # (n, 3)

    state = z_meas_all  # (n, 7) = [q(6), L(1)], matches jacobian_used's column layout
    dstate = np.diff(state, axis=0)
    dx_meas = np.diff(tip_m, axis=0)
    dx_pred = np.einsum("nij,nj->ni", jacobian_used_all[:-1], dstate)
    e_J_real_mm = np.linalg.norm(dx_meas - dx_pred, axis=1) * 1000.0

    n_k = n_tip[:-1]
    dx_meas_n_mm = np.einsum("ni,ni->n", n_k, dx_meas) * 1000.0
    dx_pred_n_mm = np.einsum("ni,ni->n", n_k, dx_pred) * 1000.0
    dx_meas_t_mm = np.linalg.norm(dx_meas - dx_meas_n_mm[:, None] * 1e-3 * n_k, axis=1) * 1000.0
    dx_pred_t_mm = np.linalg.norm(dx_pred - dx_pred_n_mm[:, None] * 1e-3 * n_k, axis=1) * 1000.0

    Jn_tip = np.einsum("ni,nij->nj", n_tip, jacobian_used_all)
    Jn_tip_norm = np.linalg.norm(Jn_tip, axis=1)
    Jt_tip = jacobian_used_all - np.einsum("ni,nj->nij", n_tip, Jn_tip)
    Jt_tip_norm = np.linalg.norm(Jt_tip.reshape(n, -1), axis=1)
    sigmas = np.linalg.svd(jacobian_used_all[:, :, :6], compute_uv=False)
    cond_J_real = sigmas[:, 0] / np.maximum(sigmas[:, -1], 1e-12)

    return dict(
        label=label, n=n, c_min_mm=c_min_mm, s_c_frac=s_c_frac,
        theta_wall_deg_stem=theta_wall_deg_stem,
        physical_contact=physical_contact, activation_region=activation_region,
        d_w_tip_mm=d_w_tip_mm, e_J_real_mm=e_J_real_mm,
        dx_meas_n_mm=dx_meas_n_mm, dx_pred_n_mm=dx_pred_n_mm,
        dx_meas_t_mm=dx_meas_t_mm, dx_pred_t_mm=dx_pred_t_mm,
        Jn_tip_norm=Jn_tip_norm, Jt_tip_norm=Jt_tip_norm, cond_J_real=cond_J_real,
    )


def summarize(m: dict[str, Any]) -> dict[str, Any]:
    phys = m["physical_contact"]
    act = m["activation_region"]

    def _mean(arr, mask):
        sel = np.asarray(arr)[mask[: len(arr)]] if len(mask) >= len(arr) else np.asarray(arr)[mask]
        return float(np.mean(sel)) if sel.size else float("nan")

    return dict(
        label=m["label"], n_ticks=int(m["n"]),
        min_c_min_mm=float(np.min(m["c_min_mm"])),
        frac_ticks_physical_contact=float(np.mean(phys)),
        frac_ticks_activation_region=float(np.mean(act)),
        n_physical_contact_ticks=int(np.sum(phys)),
        n_activation_region_ticks=int(np.sum(act)),
        s_c_frac_range_during_activation=(
            [float(np.min(m["s_c_frac"][act])), float(np.max(m["s_c_frac"][act]))]
            if act.any() else None
        ),
        mean_theta_wall_deg_during_activation=_mean(m["theta_wall_deg_stem"], act),
        mean_e_J_real_mm_during_activation=_mean(m["e_J_real_mm"], act),
        mean_e_J_real_mm_free_space=_mean(m["e_J_real_mm"], ~act),
        mean_Jn_tip_norm_during_activation=_mean(m["Jn_tip_norm"], act),
        mean_Jn_tip_norm_free_space=_mean(m["Jn_tip_norm"], ~act),
        mean_dx_meas_n_mm_during_activation=_mean(m["dx_meas_n_mm"], act),
        mean_dx_pred_n_mm_during_activation=_mean(m["dx_pred_n_mm"], act),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--contact-run", required=True)
    p.add_argument("--nocontact-run", required=True)
    p.add_argument("--out-dir", default=str(
        REPO / "close_loop_logs/myrun/contact_mechanism_analysis_v2"
    ))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[v2] loading lumen + building model adapters...")
    lumen_C, lumen_R, provenance = load_lumen()
    from proper_research.simulation.magnetic_beam.contact import LumenQuery
    lq = LumenQuery(lumen_C, lumen_R)
    adapter_contact, adapter_nocontact = build_adapters()

    mc = analyze_run(Path(args.contact_run), adapter_contact, lq, "contact")
    mn = analyze_run(Path(args.nocontact_run), adapter_nocontact, lq, "nocontact")
    sc, sn = summarize(mc), summarize(mn)

    stats = {"contact": sc, "nocontact": sn, "contact_band_m": CONTACT_BAND_M,
              "physical_contact_tol_m": PHYSICAL_CONTACT_TOL_M, "r_beam_m": R_BEAM_M,
              "lumen_provenance": provenance}
    (out_dir / "stats_v2.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))
    print(f"\n[v2] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
