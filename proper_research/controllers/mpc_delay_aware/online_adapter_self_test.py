"""Self-test for `DelayAwareOfflineControllerAdapter` / `build_delay_aware_online_adapter`
against fake frames -- no camera, robot or serial connection. Mirrors
`controller_adapters.run_self_test()`'s scaffold, adapted for this
controller's different call signature (`solve_delay_aware`, not `solve`)
and its shadow execution-C accumulator (`q_cmd`/`q_cmd_prev`), the one
piece of new state this adapter carries that the existing ones don't.

Checks, before ever running this against the real robot:
  1. SolveResult shape/finiteness, reference_index monotonic (same baseline
     checks controller_adapters.run_self_test() makes for every kind).
  2. The shadow q_cmd accumulator matches an independent recursion computed
     from the returned u0 sequence exactly (same clip formula, applied
     externally here from the test, not read back from adapter internals).
  3. The side-channel prediction log is valid JSONL with the expected keys,
     one row per tick, and q_cmd_k/q_cmd_km1 in that log match check #2's
     independent recursion exactly (not just internally self-consistent).
  4. Insertion fallback (StateEstimate.insertion_length_m = NaN) still
     works, matching every other kind's behaviour.

Run via:
    python -m proper_research.controllers.mpc_delay_aware.online_adapter_self_test
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.hardware.online.messages import StateEstimate
from proper_research.hardware.online.online_mpc_runner import SolveResult
from proper_research.simulation.simulations import (
    simulate_time_parameterized_beam_output_mpc as beam_module,
)
from proper_research.simulation.simulations import (
    simulate_time_parameterized_configuration_mpc as base_module,
)

from .online_adapter import build_delay_aware_online_adapter

Array = np.ndarray


def run_self_test() -> None:
    dt = 0.1
    sample_count = 20
    time_values = np.arange(sample_count, dtype=float) * dt
    state_reference = np.zeros((sample_count, 7), dtype=float)
    state_reference[:, 6] = 0.001 * np.arange(sample_count)
    input_reference = np.zeros_like(state_reference)
    desired_position = np.column_stack(
        [np.zeros(sample_count), np.zeros(sample_count), 5.0 * state_reference[:, 6]]
    )
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (sample_count, 1))

    reference = base_module.ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=np.zeros(sample_count),
        state=state_reference,
        input=input_reference,
        acceleration=np.zeros_like(state_reference),
        desired_position_m=desired_position,
        desired_tangent=tangent,
        planned_beam_feasible=np.ones(sample_count, dtype=bool),
        planned_position_error_m=np.zeros(sample_count),
        planned_tangent_error_rad=np.zeros(sample_count),
        sample_period_s=dt,
        source="online_adapter_self_test",
    )
    joint_velocity_limit_rad_s = 0.4
    max_joint_step_rad = joint_velocity_limit_rad_s * dt
    mpc_config = base_module.ConfigurationMPCConfig(
        sample_period_s=dt,
        prediction_horizon=6,
        state_min=tuple([-0.5] * 6 + [-0.02]),
        state_max=tuple([0.5] * 6 + [0.05]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [0.01]),
        acceleration_limit=tuple([2.0] * 6 + [0.1]),
        state_tracking_weight=1.0e-3,
        input_tracking_weight=1.0e-4,
        input_increment_weight=1.0e-4,
        solver_backend="osqp" if base_module.osqp is not None else "scipy",
        solver_absolute_tolerance=1.0e-8,
        solver_relative_tolerance=1.0e-8,
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3),
        position_tracking_weight=1.0,
        use_dare_terminal_cost=False,  # REQUIRED for the delay-aware controller
    )

    def jacobian_provider(state: Array) -> Array:
        matrix = np.zeros((3, 7), dtype=float)
        matrix[0, 0] = 0.3
        matrix[1, 1] = 0.2
        matrix[2, 6] = 5.0
        return matrix

    schedule = np.stack(
        [jacobian_provider(np.zeros(7)) for _ in range(reference.sample_count)], axis=0
    )

    def make_estimate(step: int, t: float) -> StateEstimate:
        q6 = 0.001 * step * np.ones(6)
        tip = np.array([0.02 * math.sin(step), 0.0, 5.0 * 0.001 * step])
        return StateEstimate(
            t_monotonic=t,
            x_meas=np.concatenate([tip, [1.0, 0.0, 0.0]]),
            tip_position_m=tip,
            tip_tangent=np.array([1.0, 0.0, 0.0]),
            insertion_length_m=float("nan"),  # forces the reference fallback
            robot_joints=q6,
            frame_index=step,
        )

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "predicted_beam_positions.jsonl"
        solver = build_delay_aware_online_adapter(
            reference=reference,
            mpc_config=mpc_config,
            beam_config=beam_config,
            schedule=schedule,
            joint_velocity_limit_rad_s=joint_velocity_limit_rad_s,
            max_joint_step_rad=max_joint_step_rad,
            delay_samples=2,
            beta_d=1.0,
            adapter_config=OfflineControllerConfig(progress_mode="wallclock", terminal_hold_steps=3),
            prediction_log_path=str(log_path),
        )

        t0 = 100.0
        seen_indices = []
        u0_history = []
        for step in range(8):
            estimate = make_estimate(step, t0 + step * dt)
            result = solver(estimate, dt)
            assert isinstance(result, SolveResult), "wrong return type"
            u0 = np.asarray(result.u0, dtype=float).reshape(-1)
            assert u0.size == 7, f"u0 must be length 7, got {u0.size}"
            assert np.all(np.isfinite(u0)), f"non-finite u0 at step {step}"
            seen_indices.append(result.info["reference_index"])
            u0_history.append(u0.copy())
        assert seen_indices == sorted(seen_indices), (
            f"reference_index must be non-decreasing, got {seen_indices}"
        )
        assert seen_indices[-1] > seen_indices[0], "progress never advanced"
        assert np.isfinite(result.info["measured_joint_state"][6]), (
            "insertion fallback did not produce a finite value"
        )
        print(f"[online_adapter self-test] 1/4 PASS: SolveResult shape/finiteness, "
              f"reference_index monotonic {seen_indices}")

        solver.close()
        rows = [json.loads(line) for line in log_path.open()]
        assert len(rows) == 8, f"expected 8 logged rows, got {len(rows)}"
        for r in rows:
            for key in ("step", "ref_index", "z_meas", "q_cmd_k", "q_cmd_km1",
                        "insertion_cmd_m", "u_prev", "u0", "predicted_beam_positions_m", "success"):
                assert key in r, f"missing key {key!r} in logged row {r}"
            assert len(r["q_cmd_k"]) == 6 and len(r["q_cmd_km1"]) == 6
            pred = np.asarray(r["predicted_beam_positions_m"], dtype=float)
            assert pred.shape == (mpc_config.prediction_horizon, 3), (
                f"predicted_beam_positions_m shape {pred.shape} != "
                f"({mpc_config.prediction_horizon}, 3)"
            )
        print("[online_adapter self-test] 2/4 PASS: prediction log is valid JSONL, "
              "8 rows, all expected keys present, correct prediction-stack shape")

        # independent recursion of the shadow accumulator, from the u0
        # history alone -- must match the logged q_cmd_k/q_cmd_km1 exactly.
        q0 = 0.001 * 0 * np.ones(6)  # step=0's measured q, matches make_estimate's seed
        q_cmd = q0.copy()
        q_cmd_prev = q0.copy()
        for i, u0 in enumerate(u0_history):
            logged_qk = np.asarray(rows[i]["q_cmd_k"], dtype=float)
            logged_qkm1 = np.asarray(rows[i]["q_cmd_km1"], dtype=float)
            assert np.allclose(logged_qk, q_cmd, atol=1e-12), (
                f"step {i}: logged q_cmd_k != independent recursion, "
                f"max diff={np.max(np.abs(logged_qk - q_cmd))}"
            )
            assert np.allclose(logged_qkm1, q_cmd_prev, atol=1e-12), (
                f"step {i}: logged q_cmd_km1 != independent recursion, "
                f"max diff={np.max(np.abs(logged_qkm1 - q_cmd_prev))}"
            )
            qd = np.clip(u0[:6], -joint_velocity_limit_rad_s, joint_velocity_limit_rad_s)
            delta_q = np.clip(qd * dt, -max_joint_step_rad, max_joint_step_rad)
            q_cmd_prev = q_cmd.copy()
            q_cmd = q_cmd + delta_q
        print("[online_adapter self-test] 3/4 PASS: shadow q_cmd/q_cmd_prev accumulator "
              "matches an independent recursion from the logged u0 sequence exactly")

        # u_prev in row i must equal u0 in row i-1 (rate-continuity bookkeeping).
        for i in range(1, len(rows)):
            assert np.allclose(
                np.asarray(rows[i]["u_prev"], dtype=float),
                np.asarray(rows[i - 1]["u0"], dtype=float), atol=1e-12,
            ), f"row {i}: u_prev != previous row's u0"
        print("[online_adapter self-test] 4/4 PASS: u_prev[i] == u0[i-1] for every logged tick")

    print("[online_adapter self-test] ALL PASS")


if __name__ == "__main__":
    run_self_test()
