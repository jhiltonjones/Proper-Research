#!/usr/bin/env python3
"""Precompute the magnetic force/torque actuation-preference SCHEDULE a
built vessel plan needs to turn on `magnetic_force_preference` (MPC, via
`run_mpc_delay_aware_vessel.py`) or `force_preference_gradient_fn` (resolved-
rate, via `run_inverse_jacobian_online_vessel.py`) -- see those modules'
own docstrings, and `simulation/magnetic_beam/force_preference.py` for the
underlying physics.

2026-10-16: uses the FROZEN-SHAPE (closed-form) Jacobian
(`force_preference.frozen_shape_wrench_pose7_jacobian`), not the exact,
beam-reshape-aware one (`magnetic_wrench_pose7_jacobian`) this script used
originally. One beam solve per reference sample (same cost as the ordinary
position-tracking schedule), not 14 -- tractable to build in one sitting
instead of hours.

This trades accuracy for that speed, and the trade is NOT small: checked
directly against the exact version at a representative pose, the frozen
Jacobian's torque sensitivity was off by ~150% (not just noisy -- the
beam's own elastic reshaping in response to the magnet moving dominates,
and often has the OPPOSITE sign from the frozen, field-only effect; see
this session's discussion). Accepted deliberately here because this term
is only ever a SOFT tie-breaker among already-feasible/already-tracked
solutions (it never overrides the primary task-tracking cost) -- a wrong-
direction nudge from an imperfect gradient is not dangerous the way a
wrong safety constraint would be, just potentially unhelpful. Revisit with
the exact version (slower, correct) if this term's live behaviour ever
needs to be trusted beyond "probably does more good than harm."

Output .npz (one array per reference sample, keyed exactly as
`DelayAwareBeamOutputTrackingMPC._configure_magnetic_force_preference`'s
own kwargs expect, so a caller can `**dict(np.load(path))`-unpack this
directly, modulo the four extra plain-float/3-vector keys it also needs
(force_desired_N/torque_desired_Nm/lambda_force/lambda_torque), which are
provided at RUN time, not baked into this schedule -- the same wrench
sensitivity supports any desired F_des/tau_des/weights without recomputing):
    force_jacobians   (S,3,7)  d(F_total)/d[q1..q6,L] at each sample (col 6 == 0)
    torque_jacobians  (S,3,7)  d(tau_total)/d[q1..q6,L] at each sample (col 6 == 0)
    nominal_force_N   (S,3)    resultant force at each sample's nominal state
    nominal_torque_Nm (S,3)    resultant torque at each sample's nominal state

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.compute_magnetic_force_preference_schedule \\
        --plan-dir plans/<plan_root>/time_parameterized_configuration_path \\
        --lumen-file <same lumen file the plan was built from> \\
        --insertion-max-mm <same value build_vessel_plan.py used> \\
        --output schedules/<plan_root>_force_preference.npz \\
        [--contact / --no-contact]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.simulation.magnetic_beam.force_preference import (
    solved_beam_segments, frozen_shape_wrench, frozen_shape_wrench_pose7_jacobian,
)
from proper_research.simulation.magnetic_beam.magnetism import dipole_from_pose
from proper_research.simulation.magnetic_beam.kinematics import quat_from_rotvec_ur
from proper_research.simulation.magnetic_beam.controller_adapters import (
    assemble_tip_robot_jacobian,
)


def _p7_from_state(adapter, state: np.ndarray) -> np.ndarray:
    T_R_M = adapter.magnet_transform(state)
    return np.concatenate(
        (T_R_M[:3, 3], Rot.from_matrix(T_R_M[:3, :3]).as_rotvec(), [float(state[6])])
    )


def build_schedule(
    *, adapter, reference_state: np.ndarray,
    eps_pos_m: float = 2.0e-4, eps_rot_rad: float = 2.0e-3,
) -> dict[str, np.ndarray]:
    """``reference_state``: (S,7) -- the SAME array `reference.state` is.
    One beam solve per sample (`solved_beam_segments`), reused for both the
    nominal wrench and the closed-form (no further solve) Jacobian -- see
    module docstring for the frozen-shape approximation this makes.
    """
    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("adapter.model is None -- cannot compute the magnetic wrench.")
    S = reference_state.shape[0]
    force_jacobians = np.zeros((S, 3, 7), dtype=float)
    torque_jacobians = np.zeros((S, 3, 7), dtype=float)
    nominal_force_N = np.zeros((S, 3), dtype=float)
    nominal_torque_Nm = np.zeros((S, 3), dtype=float)

    t0 = time.perf_counter()
    for i in range(S):
        state_i = np.asarray(reference_state[i], dtype=float).reshape(7)
        p7_i = _p7_from_state(adapter, state_i)
        snapshot = solved_beam_segments(model, p7_i)  # the ONE solve this sample pays for
        r_src = p7_i[0:3]
        m_src = dipole_from_pose(quat_from_rotvec_ur(p7_i[3:6]), snapshot.m_body)
        about_point_m = snapshot.r_mid_world[:, 0]  # same convention as total_magnetic_wrench's default
        wrench = frozen_shape_wrench(snapshot, r_src, m_src, about_point_m=about_point_m)
        nominal_force_N[i] = wrench.force_N
        nominal_torque_Nm[i] = wrench.torque_Nm

        dF_dp7, dtau_dp7 = frozen_shape_wrench_pose7_jacobian(
            snapshot, p7_i, about_point_m=about_point_m,
            eps_pos_m=eps_pos_m, eps_rot_rad=eps_rot_rad,
        )
        J_arm = adapter.magnet_geometric_jacobian(state_i)  # (6,6) [v;omega]
        force_jacobians[i] = assemble_tip_robot_jacobian(dF_dp7, J_arm, insertion_jacobian=1.0)
        torque_jacobians[i] = assemble_tip_robot_jacobian(dtau_dp7, J_arm, insertion_jacobian=1.0)

        if i % 25 == 0 or i == S - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1.0e-9)
            eta_s = (S - 1 - i) / max(rate, 1.0e-9)
            print(f"[force_pref schedule] {i + 1}/{S} ({elapsed:.1f}s elapsed, "
                  f"ETA {eta_s:.0f}s)", flush=True)

    return dict(
        force_jacobians=force_jacobians, torque_jacobians=torque_jacobians,
        nominal_force_N=nominal_force_N, nominal_torque_Nm=nominal_torque_Nm,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True,
                    help="the plan's time_parameterized_configuration_path directory")
    p.add_argument("--lumen-file", required=True,
                    help="the SAME lumen file build_vessel_plan.py used for this plan")
    p.add_argument("--insertion-max-mm", type=float, required=True,
                    help="the SAME value build_vessel_plan.py used for this plan")
    p.add_argument("--jacobian-mode", choices=("fast", "accurate"), default="accurate")
    contact_group = p.add_mutually_exclusive_group()
    contact_group.add_argument("--contact", dest="contact", action="store_true", default=True,
                                help="the SAME --contact/--no-contact build_vessel_plan.py used "
                                     "for this plan (default: contact-aware)")
    contact_group.add_argument("--no-contact", dest="contact", action="store_false")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from proper_research.planning.vessel_context import build_vessel_planning_context
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )

    insertion_max_m = args.insertion_max_mm / 1000.0
    print(f"[force_pref schedule] rebuilding the SAME model config build_vessel_plan.py used "
          f"(lumen_file={args.lumen_file}, insertion_max_m={insertion_max_m:.4f}, "
          f"jacobian_mode={args.jacobian_mode!r}, contact={args.contact})")
    _exp_cfg, _bundle, controller_pack, _out_root, _centreline, _lumen_R, _provenance = (
        build_vessel_planning_context(
            lumen_file=args.lumen_file, insertion_max_m=insertion_max_m,
            jacobian_mode=args.jacobian_mode, plant_contact=args.contact,
        )
    )
    adapter = controller_pack["plant_diagnostic_joint_adapter"]

    reference = load_configuration_reference(args.plan_dir, require_planned_beam_feasible=False)
    reference_state = np.asarray(reference.state, dtype=float)
    print(f"[force_pref schedule] {reference_state.shape[0]} reference samples "
          f"(frozen-shape closed-form Jacobian -- one beam solve per sample)")

    schedule = build_schedule(adapter=adapter, reference_state=reference_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **schedule)
    print(f"[force_pref schedule] wrote {args.output} "
          f"(force_jacobians={schedule['force_jacobians'].shape}, "
          f"torque_jacobians={schedule['torque_jacobians'].shape})")


if __name__ == "__main__":
    main()
