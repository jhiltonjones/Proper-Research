#!/usr/bin/env python3
"""Offline (no robot) counterfactual QP re-solves for an mpc_ltv_offline run:
does state/posture tracking through Q cause a saturating, jagged command?
The MPC analogue of the INV kn ablation, but done as exact re-solves rather
than an algebraic decomposition -- once the QP is constrained, "task part"
vs "Q part" of a saturated solution isn't a unique split.

Three conditions, all built from the SAME reference/schedule/mpc_config/
beam_config as the run being analysed (Q/Qp/R/Rd/N otherwise identical):

  baseline   as-run (state_tracking_weight from --mpc-run-dir's summary.json,
             default 1.0)
  Q~=0       state_tracking_weight scaled down by --eps-q (default 0.001x).
             Literal Q=0 is checked separately and is expected to be
             DARE-infeasible for this system (confirmed 2026-09-17: closed-
             loop spectral radius 1.000136, just barely unstable -- nothing
             regulates the null-space configuration, so the terminal cost-to-
             go has a marginal mode). This is a REAL finding: Q is
             structurally load-bearing for the terminal cost to be well-
             posed here, not just a soft preference.
  Q_no_null  Q replaced by its row-space-only projection Q_R = M^T P_R M
             (M = sqrt(diag(Q0)), P_R the row-space projector of the scaled
             Jacobian J@M^-1 via SVD), plus a small residual --eps-null
             fraction of the null-space part (pure Q_N=0 is ALSO DARE-
             infeasible -- confirmed separately). Schedule-wide for the DARE
             terminal cost; a single representative Q_R (index 0) for the
             non-terminal Qbar stage blocks, matching how baseline's own
             non-terminal Q is already stage-constant (not a fidelity
             reduction -- an extension of the existing simplification; see
             the module-level comment in NullAwareMPC for why per-tick
             stage-varying Q would need a full OSQP re-setup per tick).

Every logged tick is replayed at the EXACT (measured_state, control_index,
previous_input, measured_beam_position) the real robot visited -- a pure
counterfactual holding the real state trajectory fixed, not a closed-loop
resimulation. Baseline reproduction of the logged u0 is checked and printed
but is NOT expected to be exact -- an internal disturbance/output-residual
estimator state can't be perfectly cold-started from a log; the comparison
between the three conditions (all replayed through the identical pipeline)
is the scientifically valid part regardless of that gap.

2026-09-17 rectangle result: BOTH ablations made P(u active) and roughness
WORSE, not better (baseline 0.626 -> Q~=0 0.737 -> Q_no_null 0.737 for
P(u_active); R_u 63.4 -> 80.5 -> 86.1 mrad/s) -- falsifying the "state/
posture tracking causes the saturation" hypothesis for MPC. See this
package's README for the full table and the fallback conclusion (Qp/task-
tracking demand + horizon dynamics + hard rate limits, not Q).

IMPORTANT: this script requires a Jacobian schedule cache built by
run_mpc_ltv.py (`--schedule-cache`) -- pass the SAME path here.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.q_ablation \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --mpc-run-dir close_loop_logs/my_run/runs/mpc_ltv_accumC_<timestamp> \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy \\
        --out /tmp/q_ablation_results.npz
"""
import argparse
import json
import math
from dataclasses import replace

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig,
    load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
    BeamOutputTrackingMPC,
)

VLIM_JOINTS = 0.10
INSERTION_RATE_LIMIT = 2.0e-3
ACCEL_JOINTS = 0.40
DT = 0.1


def build_configs(horizon: int) -> tuple[ConfigurationMPCConfig, BeamOutputMPCConfig]:
    vlim = tuple([VLIM_JOINTS] * 6 + [INSERTION_RATE_LIMIT])
    alim = tuple([ACCEL_JOINTS] * 6 + [10.0 * INSERTION_RATE_LIMIT])
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=horizon,
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]),
        state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=vlim, acceleration_limit=alim,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=True, directional_damping=0.0, directional_damping_floor=0.01,
    )
    return mpc_config, beam_config


def load_logged_ticks(run_dir: str, initial_insertion_m: float):
    rows = [json.loads(l) for l in open(f"{run_dir}/path_follow.jsonl")]
    K = len(rows)
    q_meas = [np.asarray(r["q_meas_rad"], dtype=float) for r in rows]
    tip_m = [np.asarray(r["tip_mm"], dtype=float) * 1e-3 for r in rows]
    desired_m = [np.asarray(r["desired_mm"], dtype=float) * 1e-3 for r in rows]
    ref_index = [int(r["ref_index"]) for r in rows]
    u0_logged = [np.asarray(r["u0"], dtype=float) for r in rows]
    prev_input_seq = [np.zeros(7)] + u0_logged[:-1]
    insertion_at_solve = [initial_insertion_m] + [float(r["insertion_length_m"]) for r in rows[:-1]]
    measured_states = [
        np.concatenate([q_meas[k], [insertion_at_solve[k]]]) for k in range(K)
    ]
    return dict(
        K=K, measured_states=measured_states, tip_m=tip_m, desired_m=desired_m,
        ref_index=np.array(ref_index), prev_input_seq=np.array(prev_input_seq),
        u0_logged=np.array(u0_logged),
    )


def replay(controller, ticks: dict, label: str) -> np.ndarray:
    K = ticks["K"]
    u0s = []
    for k in range(K):
        step = controller.solve(
            measured_state=ticks["measured_states"][k],
            measured_beam_position=ticks["tip_m"][k],
            control_index=int(ticks["ref_index"][k]),
            previous_input=ticks["prev_input_seq"][k],
        )
        u0s.append(np.asarray(step.command, dtype=float).reshape(7))
    u0s = np.array(u0s)
    print(f"[q_ablation] {label}: replayed {K} ticks")
    return u0s


def build_null_projectors(schedule: np.ndarray, mpc_config: ConfigurationMPCConfig):
    Q0 = np.diag(1.0 / np.asarray(mpc_config.state_error_scale, dtype=float) ** 2)
    m_diag = np.sqrt(np.diag(Q0))
    m_inv = 1.0 / m_diag
    sc = schedule.shape[0]
    Q_R = np.empty((sc, 7, 7))
    Q_N = np.empty((sc, 7, 7))
    ranks = []
    for idx in range(sc):
        J_tilde = schedule[idx] * m_inv[None, :]
        _, S, Vt = np.linalg.svd(J_tilde, full_matrices=True)
        rank = int(np.sum(S > 1.0e-9 * max(S[0], 1.0)))
        ranks.append(rank)
        V_row = Vt[:rank]
        P_R = V_row.T @ V_row
        P_N = np.eye(7) - P_R
        Q_R[idx] = (m_diag[:, None] * P_R) * m_diag[None, :]
        Q_N[idx] = (m_diag[:, None] * P_N) * m_diag[None, :]
    print(f"[q_ablation] row-space rank across schedule: min={min(ranks)} max={max(ranks)} (expect 3)")
    return Q_R, Q_N


class NullAwareMPC(BeamOutputTrackingMPC):
    """BeamOutputTrackingMPC with Q replaced by a (possibly schedule-varying)
    externally supplied matrix. See the module docstring for the terminal-
    cost-vs-stage-cost fidelity tradeoff this makes.
    """

    def __init__(self, *, q_schedule: np.ndarray, q_representative: np.ndarray, **kwargs):
        self._q_schedule = q_schedule
        super().__init__(**kwargs)
        n = self.n
        for j in range(self.N - 1):  # leave the terminal block (zeroed under DARE) alone
            self.Qbar[j * n:(j + 1) * n, j * n:(j + 1) * n] = q_representative
        base_hessian = 2.0 * (
            self.S.T @ self.Qbar @ self.S + self.Rbar + self.D.T @ self.Rdbar @ self.D
        )
        base_hessian += float(self.config.hessian_regularization) * np.eye(self.nu)
        self.H = 0.5 * (base_hessian + base_hessian.T)
        self._base_hessian = np.asarray(self.H, dtype=float).copy()
        if self.backend == "osqp":
            # NOT self._setup_osqp() -- that establishes a sparsity pattern
            # from self.H alone, which is missing the dense G.T@Qpbar@G
            # (task-tracking) contribution solve() adds at runtime, and
            # OSQP's incremental .update(Px=...) cannot introduce entries
            # outside the pattern fixed at setup. This was a real bug here
            # once (exactly-zero commands at some ticks) -- use the same
            # full-dense-pattern setup the unmodified class uses.
            self._setup_variable_hessian_osqp()

    def _build_ltv_riccati_schedule(self, *, Q, Qp, R, Rd):
        # Copy of BeamOutputTrackingMPC._build_ltv_riccati_schedule with the
        # constant `Q` replaced by `self._q_schedule[index]`.
        final_input = np.asarray(self.reference.input[-1], dtype=float)
        input_scale = np.asarray(self.config.input_error_scale, dtype=float).reshape(self.m)
        self.terminal_equilibrium_normalized_input = float(
            np.max(np.abs(final_input) / input_scale)
        )
        if (
            self.terminal_equilibrium_normalized_input
            > float(self.beam_config.terminal_equilibrium_tolerance)
        ):
            raise ValueError("final reference sample is not an equilibrium")

        sample_count = self.reference.sample_count
        augmented_size = 2 * self.n
        P_schedule = np.empty((sample_count, augmented_size, augmented_size), dtype=float)
        K_schedule = np.empty((sample_count, self.m, augmented_size), dtype=float)

        effective_weights = np.empty((sample_count, self.n, self.n), dtype=float)
        for index, jacobian in enumerate(self.reference_position_jacobians):
            effective_weights[index] = self._symmetric(
                self._q_schedule[index] + jacobian.T @ Qp @ jacobian
            )

        A_aug, B_aug, Q_final, N_aug, R_aug = self._augmented_riccati_matrices(
            Q_effective=effective_weights[-1], R=R, Rd=Rd,
        )
        from scipy.linalg import solve_discrete_are
        P_final = solve_discrete_are(A_aug, B_aug, Q_final, R_aug, s=N_aug)
        P_schedule[-1] = self._symmetric(P_final)

        final_control_hessian = self._symmetric(R_aug + B_aug.T @ P_schedule[-1] @ B_aug)
        final_control_gradient = B_aug.T @ P_schedule[-1] @ A_aug + N_aug.T
        K_schedule[-1] = np.linalg.solve(final_control_hessian, final_control_gradient)
        closed_loop = A_aug - B_aug @ K_schedule[-1]
        self.terminal_closed_loop_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(closed_loop)))
        )
        if self.terminal_closed_loop_spectral_radius >= 1.0:
            raise RuntimeError(
                f"final DARE solution is not stabilizing: spectral radius="
                f"{self.terminal_closed_loop_spectral_radius:.6f}"
            )

        for index in range(sample_count - 2, -1, -1):
            _, _, Q_aug, _, _ = self._augmented_riccati_matrices(
                Q_effective=effective_weights[index], R=R, Rd=Rd,
            )
            P_next = P_schedule[index + 1]
            control_hessian = self._symmetric(R_aug + B_aug.T @ P_next @ B_aug)
            control_gradient = B_aug.T @ P_next @ A_aug + N_aug.T
            gain = np.linalg.solve(control_hessian, control_gradient)
            P_current = (
                Q_aug + A_aug.T @ P_next @ A_aug
                - (A_aug.T @ P_next @ B_aug + N_aug) @ gain
            )
            P_schedule[index] = self._symmetric(P_current)
            K_schedule[index] = gain

        if not np.all(np.isfinite(P_schedule)):
            raise FloatingPointError("backward LTV Riccati schedule non-finite")
        return P_schedule, K_schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--mpc-run-dir", required=True,
                   help="the specific timestamped run directory, e.g. "
                        ".../runs/mpc_ltv_accumC_20260917T181546Z")
    p.add_argument("--schedule-cache", required=True,
                   help="the SAME --schedule-cache path passed to run_mpc_ltv.py")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--initial-insertion-m", type=float, default=0.025)
    p.add_argument("--eps-q", type=float, default=0.001,
                   help="Q~=0 condition: state_tracking_weight scale factor")
    p.add_argument("--eps-null", type=float, default=0.01,
                   help="Q_no_null condition: residual null-space fraction")
    p.add_argument("--out", default=None, help="npz path to save all three replays")
    args = p.parse_args()

    reference = load_configuration_reference(args.plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(args.schedule_cache)
    mpc_config, beam_config = build_configs(args.horizon)
    ticks = load_logged_ticks(args.mpc_run_dir, args.initial_insertion_m)
    print(f"[q_ablation] loaded {ticks['K']} logged ticks from {args.mpc_run_dir}")

    print("\n[q_ablation] building baseline controller...")
    baseline = BeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule,
    )
    u0_baseline = replay(baseline, ticks, "baseline")
    diff = np.linalg.norm(u0_baseline - ticks["u0_logged"], axis=1)
    rel = diff / (np.linalg.norm(ticks["u0_logged"], axis=1) + 1e-9)
    print(f"[q_ablation] baseline vs logged u0: mean|diff|={diff.mean():.5f} "
          f"median_rel={np.median(rel):.4f} (an internal disturbance-estimator "
          f"state can't be perfectly cold-started from a log -- moderate "
          f"agreement is expected, see module docstring)")

    print(f"\n[q_ablation] checking literal Q=0 (expected DARE-infeasible)...")
    try:
        BeamOutputTrackingMPC(
            reference=reference, config=replace(mpc_config, state_tracking_weight=0.0),
            beam_config=beam_config, reference_position_jacobians=schedule,
        )
        print("[q_ablation] Q=0 literal: unexpectedly succeeded")
    except RuntimeError as exc:
        print(f"[q_ablation] Q=0 literal: DARE-infeasible as expected -- {exc}")

    print(f"\n[q_ablation] building Q~=0 controller (weight x{args.eps_q})...")
    mpc_config_q0 = replace(mpc_config, state_tracking_weight=mpc_config.state_tracking_weight * args.eps_q)
    controller_q0 = BeamOutputTrackingMPC(
        reference=reference, config=mpc_config_q0, beam_config=beam_config,
        reference_position_jacobians=schedule,
    )
    print(f"[q_ablation] Q~=0 terminal closed-loop spectral radius = "
          f"{controller_q0.terminal_closed_loop_spectral_radius:.4f}")
    u0_q0 = replay(controller_q0, ticks, "Q~=0")

    print(f"\n[q_ablation] building Q_no_null controller (eps_null={args.eps_null})...")
    Q_R, Q_N = build_null_projectors(schedule, mpc_config)
    Q_no_null_schedule = Q_R + args.eps_null * Q_N
    controller_null = NullAwareMPC(
        q_schedule=Q_no_null_schedule, q_representative=Q_no_null_schedule[0],
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule,
    )
    print(f"[q_ablation] Q_no_null terminal closed-loop spectral radius = "
          f"{controller_null.terminal_closed_loop_spectral_radius:.4f}")
    u0_no_null = replay(controller_null, ticks, "Q_no_null")

    if args.out:
        np.savez(
            args.out,
            u0_baseline=u0_baseline, u0_logged=ticks["u0_logged"],
            u0_q0=u0_q0, u0_no_null=u0_no_null,
            ref_index=ticks["ref_index"], tip_m=np.array(ticks["tip_m"]),
            desired_m=np.array(ticks["desired_m"]), prev_input_seq=ticks["prev_input_seq"],
            schedule=schedule,
        )
        print(f"\n[q_ablation] saved -> {args.out}")


if __name__ == "__main__":
    main()
