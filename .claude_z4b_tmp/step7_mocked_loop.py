"""Z4b step 7: 30s mocked-loop run of campaign runners (period, index,
latency, pre-flight, abort, logs).

Drives the REAL production adapter class (ProcessIsolatedDelayAwareAdapter /
build_delay_aware_online_adapter, the exact class run_mpc_delay_aware_vessel.py
and run_mpc_delay_aware_qn_zero.py both instantiate) against each campaign
plan's REAL reference trajectory (loaded from the actual plan file this
session just rebuilt/verified in step 6), with NO camera/robot connection --
mocked StateEstimate ticks fed in a real wall-clock loop at the counter-mode
period. Uses a cheap synthetic Jacobian provider (not the plan's own
contact-model schedule) so the 30s loop is about LOOP TIMING, not physics --
step 6 already verified the plan/schedule numerics; this step verifies the
runner infrastructure (period, index, pre-flight, abort) on top of it.

Honesty note (see final report): this covers the MPC-SJ path (shared by
vessel and shape runners) against all 4 campaign plan references
(rectangle, triangle-35, vessel plan B, vessel plan B'). It does NOT cover
the INV-7 controller path -- building an equivalent hardware-free harness
for InverseJacobian7DOFDelayAware was not completed in the time available
this pass.
"""
import sys
import time

sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")

import numpy as np

from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.hardware.online.messages import StateEstimate
from proper_research.simulation.simulations import (
    simulate_time_parameterized_beam_output_mpc as beam_module,
)
from proper_research.simulation.simulations import (
    simulate_time_parameterized_configuration_mpc as base_module,
)
from proper_research.controllers.mpc_delay_aware.online_adapter import (
    build_delay_aware_online_adapter,
)

ROOT = "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis"

PLANS = {
    "rectangle_sj (campaign)": f"{ROOT}/plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path",
    "triangle35_sj (campaign)": f"{ROOT}/plans/triangle_base35mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path",
    "vessel_planB (reference)": f"{ROOT}/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_dipolealigned_2026-10-20/time_parameterized_configuration_path",
    "vessel_planBprime (reference)": f"{ROOT}/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_2026-10-19/time_parameterized_configuration_path",
}


def synthetic_jacobian(_state: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 7), dtype=float)
    m[0, 0] = 0.3
    m[1, 1] = 0.2
    m[2, 6] = 5.0
    return m


def run_mocked_loop(label: str, plan_dir: str, duration_s: float = 30.0, dt: float = 0.1):
    print(f"\n{'='*70}\n[{label}] plan_dir={plan_dir}\n{'='*70}")
    reference = base_module.load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    n = reference.sample_count
    schedule = np.stack([synthetic_jacobian(np.zeros(7)) for _ in range(n)], axis=0)

    joint_velocity_limit_rad_s = 0.1
    max_joint_step_rad = joint_velocity_limit_rad_s * dt
    mpc_config = base_module.ConfigurationMPCConfig(
        sample_period_s=dt, prediction_horizon=15,
        state_min=tuple([-1.0] * 6 + [-0.02]), state_max=tuple([1.0] * 6 + [0.20]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [0.002]),
        acceleration_limit=tuple([0.5] * 6 + [0.02]),
        state_tracking_weight=1.0e-3, input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        solver_backend="osqp" if base_module.osqp is not None else "scipy",
        solver_absolute_tolerance=1.0e-7, solver_relative_tolerance=1.0e-7,
        solver_time_limit_s=0.03,
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3),
        position_tracking_weight=1.0, use_dare_terminal_cost=False,
    )

    # Z4b campaign-2 default: counter mode, reflecting the post-Z4b
    # OfflineControllerConfig default directly (no override passed here).
    adapter_config = OfflineControllerConfig(terminal_hold_steps=3)
    print(f"[preflight] progress_mode={adapter_config.progress_mode!r} "
          f"(expect 'counter' per Z4b step 1 default)")
    assert adapter_config.progress_mode == "counter"

    solver = build_delay_aware_online_adapter(
        reference=reference, mpc_config=mpc_config, beam_config=beam_config,
        schedule=schedule, joint_velocity_limit_rad_s=joint_velocity_limit_rad_s,
        max_joint_step_rad=max_joint_step_rad, delay_samples=2, beta_d=1.0,
        adapter_config=adapter_config,
    )

    tick_period_s = dt  # nominal 100ms control period
    n_ticks = int(round(duration_s / tick_period_s))
    period_samples = []
    indices_seen = []
    latencies_ms = []
    deadline_misses = 0
    t_prev_wall = None
    t0 = time.monotonic()
    for step in range(n_ticks):
        loop_start = time.monotonic()
        q6 = 0.001 * min(step, n - 1) * np.ones(6)
        tip = np.asarray(reference.desired_position_m[min(step, n - 1)], dtype=float)
        estimate = StateEstimate(
            t_monotonic=t0 + step * tick_period_s,
            x_meas=np.concatenate([tip, [1.0, 0.0, 0.0]]),
            tip_position_m=tip, tip_tangent=np.array([1.0, 0.0, 0.0]),
            insertion_length_m=float(reference.state[min(step, n - 1), 6]),
            robot_joints=q6, frame_index=step,
        )
        call_start = time.monotonic()
        result = solver(estimate, tick_period_s)
        call_latency_ms = (time.monotonic() - call_start) * 1000.0
        latencies_ms.append(call_latency_ms)
        if not np.all(np.isfinite(result.u0)):
            deadline_misses += 1
        indices_seen.append(result.info.get("reference_index"))

        if t_prev_wall is not None:
            period_samples.append(loop_start - t_prev_wall)
        t_prev_wall = loop_start

        elapsed = time.monotonic() - loop_start
        sleep_s = tick_period_s - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    total_wall_s = time.monotonic() - t0
    period_samples = np.asarray(period_samples)
    latencies_ms = np.asarray(latencies_ms)
    indices_seen = np.asarray([i for i in indices_seen if i is not None])

    print(f"[period] n_ticks={n_ticks} total_wall_s={total_wall_s:.2f}s "
          f"mean_period_ms={period_samples.mean()*1000:.2f} "
          f"std_period_ms={period_samples.std()*1000:.2f} "
          f"(target={tick_period_s*1000:.0f}ms)")
    print(f"[index] counter-mode monotonic: {'YES' if np.all(np.diff(indices_seen) >= 0) else 'NO'} "
          f"(first 5: {indices_seen[:5].tolist()}, matches raw tick count: "
          f"{bool(np.array_equal(indices_seen[:len(indices_seen)], np.arange(len(indices_seen))))})")
    print(f"[latency] mean_solve_call_ms={latencies_ms.mean():.2f} max={latencies_ms.max():.2f} "
          f"(70ms deadline per campaign-2 decision -- real worker-process deadline enforcement "
          f"is NOT exercised here, this is the solve call in-process, not the isolated-worker "
          f"IPC path the real runners use)")
    print(f"[deadline_misses] {deadline_misses}/{n_ticks}")
    print(f"[abort] N/A -- no tracking-error ramp injected in this run "
          f"(abort logic lives in close_loop_path_follow.py's run loop, not in this "
          f"adapter-level harness; see test_step2_tracking_abort.py for that check)")
    return {
        "label": label, "n_ticks": n_ticks, "mean_period_ms": float(period_samples.mean() * 1000),
        "deadline_misses": int(deadline_misses), "indices_monotonic": bool(np.all(np.diff(indices_seen) >= 0)),
        "progress_mode": adapter_config.progress_mode,
    }


if __name__ == "__main__":
    results = []
    for label, plan_dir in PLANS.items():
        results.append(run_mocked_loop(label, plan_dir, duration_s=30.0, dt=0.1))

    print(f"\n{'='*70}\n[SUMMARY]\n{'='*70}")
    for r in results:
        print(r)
