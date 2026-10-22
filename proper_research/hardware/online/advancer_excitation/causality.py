#!/usr/bin/env python3
"""Offline (no robot) causality checks for the advancer-excitation
hypothesis, run AFTER analyze.py has produced a resampled-at-10Hz vibration
record.

Part 1 -- does the measured advancer-induced vibration alone, fed through
the INV controller's own J^+ kp mapping (measurement fixed, only the beam
position changes), produce robot commands resembling the live jagged
motion?

    dz_vib,k = -J^+ kp r_k        (r_k from analyze.py's --analysis npz,
                                    J^+ the exact damped pinv
                                    TwoDOFTrimController uses)

for kp in {0.3, 0.6, 1.0}, reporting R_q (RMS trim magnitude), P(rate-limit
would be active), peak joint increment, and the dominant frequency of the
resulting command's spectrum (should land near the excitation test's f_osc
if the mechanism is real).

Part 2 -- spectral/causal-chain comparison against an EXISTING closed-loop
run: PSD of the tracking error e(t) and the feedback command u0[:6], plus
their cross-correlation at lag 0/1/2 ticks. If the excitation test's f_osc
shows up in both spectra and the cross-correlation peaks near lag 0-2, that
is direct evidence for the causal chain advancer -> beam oscillation ->
tracking error -> feedback command -> robot jaggedness.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.causality \\
        --analysis advancer_excitation_logs/rectangle_2026-09-18/analysis.npz \\
        --L-index 4 \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --closed-loop-run close_loop_logs/rectangle_inv_2dof_trim_2026-09-17/inv_2dof_trim_accumC_kn0_kp0.6_20260917T175242Z
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import common
from .analyze import psd_peak, resample_uniform

MAX_JOINT_STEP_RAD = 0.010  # rectangle_stage_a.common.MAX_JOINT_STEP_RAD -- the rate bound
DT = 0.1
DAMPING = 5.0e-2  # TwoDOFTrimController's default -- see inverse_jacobian_2dof_trim.py


def build_fixed_jacobian(plan_dir: str, q0: np.ndarray | None, l0_m: float) -> np.ndarray:
    """One evaluation of the same state-dependent Jacobian source the live
    controllers use, at the (fixed, by construction) test pose."""
    from proper_research.controllers.beam_jacobian_providers import from_model_bundle
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.hardware.online.rectangle_stage_a import common as rsa_common

    if q0 is None:
        q0, l0_from_plan = rsa_common.load_plan_initial_state(plan_dir)
        if l0_m is None:
            l0_m = l0_from_plan
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    state = np.concatenate([np.asarray(q0, dtype=float).reshape(6), [float(l0_m)]])
    J = np.asarray(jac_provider(state), dtype=float).reshape(3, 7)
    return J


def replay_vibration_through_inv(r_k: np.ndarray, J: np.ndarray, kp: float) -> np.ndarray:
    """r_k: (K,3) vibration residual. Returns dz_vib: (K,6) position trims,
    dz_vib,k = -J^+ (kp * r_k) -- see module docstring."""
    Jq_pinv = common.damped_pinv_joint_jacobian(J[:, :6], damping=DAMPING)
    return -(Jq_pinv @ (kp * r_k).T).T  # (K,6)


def part1_kp_sweep(r_k: np.ndarray, J: np.ndarray, *, kps=(0.3, 0.6, 1.0)) -> None:
    print("\n=== PART 1: vibration-driven INV command (measurement-only, robot state fixed) ===")
    header = f"{'kp':>5} {'R_q(mrad)':>10} {'P(rate_active)':>15} {'peak(mrad)':>11} {'f_dom(Hz)':>10}"
    print(header)
    for kp in kps:
        dz = replay_vibration_through_inv(r_k, J, kp)
        r_q = float(np.sqrt(np.mean(np.sum(dz ** 2, axis=1)))) * 1e3  # mrad, vector RMS
        active_frac = float(np.mean(np.any(np.abs(dz) > MAX_JOINT_STEP_RAD, axis=1)))
        peak = float(np.max(np.abs(dz))) * 1e3
        t_k = np.arange(dz.shape[0]) * DT
        mag = np.linalg.norm(dz, axis=1)
        f_dom, _p, _f, _psd = psd_peak(t_k, mag)
        print(f"{kp:5.2f} {r_q:10.4f} {active_frac:15.3f} {peak:11.4f} {f_dom:10.3f}")
    print("Decisive pattern to look for: R_q(kp=1.0) > R_q(kp=0.6) > R_q(kp=0.3), and "
          "f_dom landing near the excitation test's own f_osc for this length.")


def part2_spectral_comparison(closed_loop_run_dir: str, f_osc_reference: float) -> None:
    print(f"\n=== PART 2: spectral comparison against {closed_loop_run_dir} ===")
    rows = [json.loads(l) for l in open(f"{closed_loop_run_dir}/path_follow.jsonl")]
    t = np.array([r["t_s"] for r in rows], dtype=float)
    e_norm = np.array([r["error_norm_mm"] for r in rows], dtype=float)
    u0 = np.array([r["u0"] for r in rows], dtype=float)[:, :6]
    u_norm = np.linalg.norm(u0, axis=1)

    f_e, _p_e, _f1, _psd1 = psd_peak(t, e_norm, exclude_dc_hz=0.05)
    f_u, _p_u, _f2, _psd2 = psd_peak(t, u_norm, exclude_dc_hz=0.05)
    print(f"dominant freq of e(t) (tracking error):     {f_e:.3f} Hz")
    print(f"dominant freq of |u0[:6]| (feedback cmd):    {f_u:.3f} Hz")
    print(f"excitation-test f_osc (reference):           {f_osc_reference:.3f} Hz")
    print(f"agreement e vs excitation:  {'YES' if abs(f_e - f_osc_reference) < 0.5 else 'no'} "
          f"(|delta|={abs(f_e - f_osc_reference):.3f} Hz)")
    print(f"agreement u0 vs excitation: {'YES' if abs(f_u - f_osc_reference) < 0.5 else 'no'} "
          f"(|delta|={abs(f_u - f_osc_reference):.3f} Hz)")

    e_c = e_norm - e_norm.mean()
    u_c = u_norm - u_norm.mean()
    print("\ncross-correlation C(l) = corr(e_k, |u0|_{k+l}):")
    for lag in (0, 1, 2):
        if lag == 0:
            a, b = e_c, u_c
        else:
            a, b = e_c[:-lag], u_c[lag:]
        denom = np.std(a) * np.std(b)
        c = float(np.mean(a * b) / denom) if denom > 1e-12 else float("nan")
        print(f"  lag={lag}: {c:+.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis", required=True, help="analyze.py's --out npz")
    p.add_argument("--L-index", type=int, default=4, help="which length's r_k to use (1-4; "
                    "default 4 = L_max, the worst-case/most-conservative choice)")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--l0-mm", type=float, default=None,
                    help="insertion length to evaluate J at (default: plan's own initial state)")
    p.add_argument("--kps", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    p.add_argument("--closed-loop-run", default=None,
                    help="an existing close_loop_path_follow run dir (has path_follow.jsonl) "
                         "for the part-2 spectral comparison; omit to skip part 2")
    args = p.parse_args()

    data = np.load(args.analysis)
    key = f"L{args.L_index}_r_k"
    if key not in data:
        raise SystemExit(f"{args.analysis} has no {key} -- did analyze.py run on that length's "
                          f"hold_post data? available keys: {list(data.keys())}")
    r_k = data[key]
    print(f"[causality] loaded r_k: {r_k.shape} from {args.analysis} (L_index={args.L_index})")

    J = build_fixed_jacobian(args.plan_dir, None, args.l0_mm)
    print(f"[causality] J (3x7) cond={np.linalg.cond(J[:, :6]):.1f}")

    part1_kp_sweep(r_k, J, kps=tuple(args.kps))

    if args.closed_loop_run:
        f_osc_ref = float("nan")
        f_key = f"L{args.L_index}_f_osc"
        if f_key in data:
            f_osc_ref = float(np.nanmean(data[f_key]))
        part2_spectral_comparison(args.closed_loop_run, f_osc_ref)


if __name__ == "__main__":
    main()
