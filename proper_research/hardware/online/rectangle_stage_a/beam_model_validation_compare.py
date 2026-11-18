"""Offline comparison: measured (q, L) vs beam-model-predicted tip position,
for the bounded static beam-model validation procedure
(BEAM_MODEL_VALIDATION_PROCEDURE_2026-11-18.md).

Reports THREE separated error numbers per sample -- never conflated:
  1. joint positioning error  = ||measured_q - planned_q||          (rad)
  2. insertion positioning error = |measured_L - planned_L|          (m)
  3. model (planar tip) error = ||model_predicted_tip_xy - measured_tip_xy||
     in the camera's 2D measurement plane only -- NOT a 3D tip error, per
     this project's documented observable-dimensions limitation (Z is
     reconstructed from the fixed pivot + chord geometry, not an
     independent per-sample measurement).

Uses the FIXED, already-established frame registration (planner_to_live_R_fit/
t_fit) -- never refit per sample. Frames used, stated explicitly:
  - measured_q, planned_q: robot joint space (UR10e joint angles, rad).
  - measured_L, planned_L: insertion length (m), advancer-axis scalar.
  - measured_tip: robot base frame R (metres), from the live camera chain
    (NewFrameTipMapper's pivot + chord reconstruction).
  - model-predicted tip: planner frame P (metres), as beam_jacobian_providers'
    model naturally outputs -- rotated/translated into frame R via the SAME
    fixed (r_fit, t_fit) registration before comparison, so both tip
    quantities compared below are in frame R.

This script is offline-only: it takes already-measured values (from a JSON
file or passed directly) -- it does not connect to any hardware itself.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationSample:
    idx: int
    planned_q: np.ndarray       # (6,) rad
    planned_L_m: float
    measured_q: np.ndarray      # (6,) rad
    measured_L_m: float
    measured_tip_R: np.ndarray  # (3,) m, robot base frame, from camera chain


def joint_positioning_error(sample: ValidationSample) -> float:
    return float(np.linalg.norm(sample.measured_q - sample.planned_q))


def insertion_positioning_error(sample: ValidationSample) -> float:
    return float(abs(sample.measured_L_m - sample.planned_L_m))


def model_planar_tip_error(
    sample: ValidationSample, r_fit: np.ndarray, t_fit: np.ndarray,
    model_predict_tip_fn,
) -> float:
    """model_predict_tip_fn(q, L) -> predicted tip in PLANNER frame P (m).
    Rotated into frame R via the fixed registration, then compared to the
    measured tip's first two (planar, camera-observable) components only."""
    tip_planner = np.asarray(
        model_predict_tip_fn(sample.measured_q, sample.measured_L_m), dtype=float
    ).reshape(3)
    tip_R_predicted = r_fit @ tip_planner + t_fit
    # Planar (camera image-plane) comparison only -- first two components of
    # the robot-frame tip vector, consistent with this procedure's documented
    # observable-dimensions limitation (Z not independently measured).
    return float(
        np.linalg.norm(tip_R_predicted[:2] - sample.measured_tip_R[:2])
    )


def run_comparison(samples_path: str, registration_path: str, model: str = "mock") -> list[dict]:
    with open(samples_path) as f:
        raw_samples = json.load(f)
    with open(registration_path) as f:
        reg = json.load(f)
    r_fit = np.asarray(reg["planner_to_live_R_fit"], dtype=float)
    t_fit = np.asarray(reg["planner_to_live_t_fit_m"], dtype=float)

    if model == "mock":
        # Offline self-test stand-in: identity-ish forward model (planner-frame
        # tip == measured_q[0:3]-derived toy point). Real execution must swap
        # this for beam_jacobian_providers.from_model_bundle(...)'s actual
        # forward-prediction call -- not wired here since that requires the
        # live model bundle/plan context this offline self-test doesn't build.
        def model_predict_tip_fn(q, L):
            return np.array([q[0] * 0.1, q[1] * 0.1, L])
    else:
        raise NotImplementedError(
            "Real model path (beam_jacobian_providers.from_model_bundle) must be "
            "wired in at execution time with the plan's actual model bundle -- "
            "not available in this offline self-test."
        )

    results = []
    for raw in raw_samples:
        sample = ValidationSample(
            idx=raw["idx"],
            planned_q=np.asarray(raw["planned_q"], dtype=float),
            planned_L_m=float(raw["planned_L_m"]),
            measured_q=np.asarray(raw["measured_q"], dtype=float),
            measured_L_m=float(raw["measured_L_m"]),
            measured_tip_R=np.asarray(raw["measured_tip_R"], dtype=float),
        )
        results.append({
            "idx": sample.idx,
            "joint_positioning_error_rad": joint_positioning_error(sample),
            "insertion_positioning_error_m": insertion_positioning_error(sample),
            "model_planar_tip_error_m": model_planar_tip_error(
                sample, r_fit, t_fit, model_predict_tip_fn
            ),
        })
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", required=True, help="JSON file of measured samples")
    p.add_argument("--registration", required=True, help="JSON file with planner_to_live_R_fit/t_fit (the FIXED registration -- never refit per sample)")
    p.add_argument("--model", default="mock", choices=["mock"])
    args = p.parse_args()
    results = run_comparison(args.samples, args.registration, model=args.model)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
