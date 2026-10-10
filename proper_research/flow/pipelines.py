"""STEP 2.4 -- the two flow pipelines, and the cheap D1 comparison.

Mirrors ``controllers/beam_jacobian_providers.contact_and_contact_free``: one
model, two providers that differ only in how the four raw channels are used.

    scalar_aggregate   collapses 4 -> 1   (the current vision-replacement path)
    differential       4 -> (lateral_y, lateral_z, common_mode, quadrupole)

``compare_sensor_pipelines`` returns ``sigma_min``, ``rank`` and the condition
number for each over a set of offsets.  It is the D1 gate and runs in
milliseconds -- no closed loop, no QP, no beam simulation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .flow_jacobian import FlowMeasurementProvider
from .sensor_model import (
    SensorRing,
    channel_decomposition,
    common_mode,
)

Array = np.ndarray


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
def scalar_aggregate_provider(ring: SensorRing) -> FlowMeasurementProvider:
    """The current pipeline: the four channels are averaged to one scalar.

    ``s_bar = mean(s)``.  Its gradient is parallel to ``e`` (radial only) and
    vanishes at ``e = 0`` -- rank 1 away from the setpoint, rank 0 at it.
    """
    return FlowMeasurementProvider(
        name="scalar_aggregate",
        ring=ring,
        source="mean of the four channels (4 -> 1)",
        channel_names=("common_mode",),
        measurement_fn=lambda e: np.atleast_1d(common_mode(ring.read_phasor(e))),
    )


def differential_provider(ring: SensorRing) -> FlowMeasurementProvider:
    """The proposed pipeline: the four orthogonal channel combinations are kept.

    ``(lateral_y, lateral_z)`` are linear in ``e`` with ``e``-independent gain
    ``2 u'(a)`` -- rank 2 everywhere, best-conditioned at the setpoint.
    """
    return FlowMeasurementProvider(
        name="differential",
        ring=ring,
        source="four orthogonal channel combinations (4 -> 4)",
        channel_names=("lateral_y", "lateral_z", "common_mode", "quadrupole"),
        measurement_fn=lambda e: np.array(
            list(channel_decomposition(ring.read_phasor(e)).values())
        ),
    )


def sensor_pipelines(ring: SensorRing) -> tuple[FlowMeasurementProvider,
                                                FlowMeasurementProvider]:
    """Both providers over one flow model -- the analogue of contact_and_contact_free."""
    return scalar_aggregate_provider(ring), differential_provider(ring)


# --------------------------------------------------------------------------
# D1 comparison
# --------------------------------------------------------------------------
def compare_sensor_pipelines(
    scalar: FlowMeasurementProvider,
    differential: FlowMeasurementProvider,
    offsets: Iterable[Array],
) -> dict:
    """sigma_min / rank / condition number for each pipeline over the offsets.

    Offsets are 2-vectors in metres.  The two differential *offset* rows
    (lateral_y, lateral_z) are what carry the estimate, so the differential
    conditioning is reported on that 2x2 sub-block as well as on the full map.
    """
    offsets = [np.asarray(o, dtype=float).reshape(2) for o in offsets]
    rows = []
    for e in offsets:
        e_over_R = float(np.linalg.norm(e) / scalar.ring.vessel_R)
        sc = scalar.conditioning(e)
        df = differential.conditioning(e)
        # differential estimate sub-block: rows lateral_y, lateral_z
        m_full = differential.normalised_jacobian(e)
        sub = m_full[:2, :]
        sv_sub = np.linalg.svd(sub, compute_uv=False)
        rows.append({
            "e_over_R": e_over_R,
            "scalar_sigma_min": sc["sigma_min"],
            "scalar_rank": sc["rank"],
            "scalar_condition": sc["condition_number"],
            "differential_sigma_min": float(sv_sub.min()),
            "differential_rank": int(np.sum(sv_sub > 1e-12)),
            "differential_condition": float(sv_sub.max() / sv_sub.min())
            if sv_sub.min() > 1e-12 else np.inf,
        })
    a_over_R = scalar.ring.a_over_R
    return {
        "scalar": scalar.describe(),
        "differential": differential.describe(),
        "a_over_R": a_over_R,
        "crossover_e_over_R": 2.0 * a_over_R,
        "analytic_scalar_sigma_min": "4 |e|/R",
        "analytic_differential_sigma_min": "8 a/R (Poiseuille) = 2 Gamma(alpha, a/R)",
        "rows": rows,
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test() -> bool:
    from .flow_field import FlowField, Waveform, Gamma
    ok = True

    vessel_R = 4e-3
    a_over_R = 0.2
    ring = SensorRing(a_ring=a_over_R * vessel_R, vessel_R=vessel_R,
                      field=FlowField(alpha=1e-4, waveform=Waveform.single_harmonic()))
    scalar, diff = sensor_pipelines(ring)

    offsets = [np.array([f * vessel_R, 0.0]) for f in (0.0, 0.01, 0.05, 0.1, 0.4)]
    report = compare_sensor_pipelines(scalar, diff, offsets)

    # scalar sigma_min ~ 4|e|/R and rank falls to 0 at e = 0
    worst = 0.0
    for row in report["rows"]:
        worst = max(worst, abs(row["scalar_sigma_min"] - 4.0 * row["e_over_R"]))
    t1 = worst < 1e-5 and report["rows"][0]["scalar_rank"] == 0
    print(f"  [{'ok  ' if t1 else 'FAIL'}] scalar sigma_min == 4|e|/R, rank 0 at e=0  "
          f"worst {worst:.2e}")
    ok &= t1

    # differential sigma_min == 8 a/R, constant, rank 2 everywhere
    vals = [row["differential_sigma_min"] for row in report["rows"]]
    ranks = [row["differential_rank"] for row in report["rows"]]
    t2 = max(abs(v - 8 * a_over_R) for v in vals) < 1e-6 and all(r == 2 for r in ranks)
    print(f"  [{'ok  ' if t2 else 'FAIL'}] differential sigma_min == 8a/R, rank 2 always  "
          f"range [{min(vals):.6f}, {max(vals):.6f}]")
    ok &= t2

    # crossover
    t3 = abs(report["crossover_e_over_R"] - 2 * a_over_R) < 1e-12
    print(f"  [{'ok  ' if t3 else 'FAIL'}] crossover at |e| = 2a")
    ok &= t3

    # Womersley: differential still rank 2, scalar still singular at e=0
    ring_w = SensorRing(a_ring=a_over_R * vessel_R, vessel_R=vessel_R,
                        field=FlowField(alpha=5.0, waveform=Waveform.single_harmonic()))
    sc_w, df_w = sensor_pipelines(ring_w)
    rep_w = compare_sensor_pipelines(sc_w, df_w, [np.array([0.0, 0.0]),
                                                 np.array([0.05 * vessel_R, 0.0])])
    sig = rep_w["rows"][0]["differential_sigma_min"]
    t4 = (rep_w["rows"][0]["scalar_rank"] == 0
          and rep_w["rows"][0]["differential_rank"] == 2
          and abs(sig - 2 * Gamma(5.0, a_over_R)) < 1e-4)
    print(f"  [{'ok  ' if t4 else 'FAIL'}] Womersley: differential rank 2 at e=0, "
          f"sigma_min {sig:.4f} == 2 Gamma {2*Gamma(5.0, a_over_R):.4f}")
    ok &= t4

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    ok = _self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
