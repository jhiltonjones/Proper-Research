"""SEPARATE, clearly-labeled sub-test: drives the real adapter to the
ACTUAL configured persistent-failure abort threshold (consecutive_deadline_
misses >= 5, traced directly in close_loop_path_follow.py:2029). The 3
sustained real-cadence/real-deadline reps in sustained_worker_deadline_
test.py naturally produced misses (15/90, 16.7%) but never a streak longer
than 2 -- this threshold needs deliberate exercise. Reuses the SAME
artificial-deadline-tightening technique as the mocked execution self-test
(adapter._deadline_s temporarily set to 1ms -- the exact same real solve
against the exact same real poll-based check, not a fabricated delay),
kept completely separate from, and not a replacement for, the normal-
scheduling sustained-cadence results.

SCOPE, STATED EXPLICITLY (do not lose this in a chat message -- it lives
here, in the code, on purpose): this test confirms that
`ProcessIsolatedDelayAwareAdapter`'s OWN `consecutive_deadline_misses`
counter correctly reaches the real production abort-threshold VALUE
(5) that `close_loop_path_follow.py:2029` checks
(`if consecutive_misses >= 5: abort_reason = ...`). It does NOT execute
that file's own, much larger online control loop, and it does NOT
exercise that file's actual `abort_reason`/`break` statement -- neither
is instantiated anywhere in this script. A passing result here is
evidence the COUNTER is correct; it is not evidence the ONLINE LOOP's
own abort wiring has been exercised end to end.
"""
import sys, json, time
MAIN = "/home/jack/Proper-Research"
sys.path.insert(0, MAIN)

_FORBIDDEN_HW_IDENTIFIERS = ("URRtde", "RobotJointStream", "CameraSource",
                             "rtde_control", "rtde_receive", "RTDEControlInterface", "RTDEReceiveInterface")
with open(__file__, "r") as _f:
    _own_source = _f.read()
for _ident in _FORBIDDEN_HW_IDENTIFIERS:
    _occ = _own_source.count(_ident)
    assert _occ <= 1, (
        f"HARDWARE GUARD FAILED: {_ident!r} appears {_occ} times in this file -- "
        f"expected at most 1 (this guard's own tuple). This script must never import or "
        f"construct a real robot/camera interface."
    )
print("[hardware guard] confirmed: no robot/camera interface imported or constructed.")

import numpy as np
import math

from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.hardware.online.messages import StateEstimate
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.hardware.online.rectangle_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

import proper_research.controllers.mpc_delay_aware.worker_process as wp_module
assert wp_module.__file__ == MAIN + "/proper_research/controllers/mpc_delay_aware/worker_process.py"
assert ".claude/worktrees" not in wp_module.__file__

ABORT_THRESHOLD = 5


def build_worker_and_adapter():
    RUN_DIR = f"{MAIN}/close_loop_logs/myrun/k2_instrumented_repeat_20261116T113659Z_20261116T113717Z"
    PLAN_DIR = f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path"
    SCHEDULE_PATH = f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy"
    cm = json.load(open(f"{RUN_DIR}/controller_metadata.json"))
    sm = json.load(open(f"{RUN_DIR}/summary.json"))
    cfg = sm["config"]
    dt = 1.0 / cfg["control_hz"]
    horizon = cm["horizon"]
    mpc_config_kwargs = dict(
        sample_period_s=dt, prediction_horizon=int(horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]), state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([cfg["joint_velocity_limit_rad_s"]] * 6 + [cfg["insertion_rate_limit_m_s"]]),
        acceleration_limit=tuple([cfg["joint_acceleration_limit_rad_s2"]] * 6 + [10.0 * cfg["insertion_rate_limit_m_s"]]),
    )
    increment_scale = tuple(
        dt * a for a in ([cfg["joint_acceleration_limit_rad_s2"]] * 6 + [10.0 * cfg["insertion_rate_limit_m_s"]])
    )
    nullspace_r700_config_kwargs = dict(
        mpc_config_kwargs, input_tracking_weight=cm["input_tracking_weight"],
        input_increment_weight=cm["input_increment_weight"], input_increment_scale=increment_scale,
    )
    beam_config_kwargs = dict(
        position_error_scale_m=tuple([float(cfg["mpc_position_error_scale_mm"]) * 1e-3] * 3),
        position_tracking_weight=float(cfg["mpc_position_tracking_weight"]),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    tcp_dh = urik.corrected_dh_from_config(urik.CONFIG)
    tcp_T_F_TCP = urik.pose6_to_T(urik.CONFIG.offline_active_T_flange_tcp_pose6)
    tcp_ref_states_q = np.asarray(reference.state[:, :6], dtype=float)
    S = tcp_ref_states_q.shape[0]
    tcp_jacobians = np.zeros((S, 3, 6)); tcp_positions = np.zeros((S, 3))
    for i in range(S):
        res = urik.forward_kinematics(tcp_ref_states_q[i], tcp_dh, tcp_T_F_TCP)
        tcp_positions[i] = res.T_R_target[:3, 3]
        tcp_jacobians[i] = urik.geometric_jacobian(tcp_ref_states_q[i], tcp_dh, tcp_T_F_TCP)[:3, :6]
    margin_m = 15.0 * 1e-3
    xyz_min = np.asarray(common.WORKSPACE_XYZ_MIN_M, dtype=float) + margin_m
    xyz_max = np.asarray(common.WORKSPACE_XYZ_MAX_M, dtype=float) - margin_m
    tcp_workspace_kwargs = dict(position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
                                 xyz_min=xyz_min, xyz_max=xyz_max)
    worker = MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=SCHEDULE_PATH,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=cm["delay_samples"], beta_d=cm["beta_d"], single_threaded=True,
        enable_nullspace_r700=True, nullspace_r700_config_kwargs=nullspace_r700_config_kwargs,
        insertion_state_anchor_weight=cm["insertion_state_anchor_weight"],
        tcp_workspace_kwargs=tcp_workspace_kwargs,
    )
    config = OfflineControllerConfig()
    config.progress_mode = "counter"
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=reference, worker=worker, deadline_s=0.070, config=config,
        joint_velocity_limit_rad_s=cfg["joint_velocity_limit_rad_s"], max_joint_step_rad=0.1,
        controller_kind="nullspace_r700", controller_label="abort_threshold_subtest",
    )
    return worker, adapter, dt


def main():
    per_tick = json.load(open(f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_per_tick.json"))["marginfix"]
    by_step = {r["step"]: r for r in per_tick}
    row = by_step[198]  # arbitrary real, valid problem -- doesn't matter which under a 1ms deadline

    print("[abort_threshold_subtest] spawning real worker...")
    worker, adapter, dt = build_worker_and_adapter()
    print(f"[abort_threshold_subtest] worker ready. Driving {ABORT_THRESHOLD + 1} consecutive misses via "
          f"adapter._deadline_s=1ms (same real solve/real poll check, not a fabricated delay).")
    adapter._deadline_s = 0.001
    trace = []
    threshold_reached_at = None
    for rep in range(1, ABORT_THRESHOLD + 2):
        adapter._counter = row["control_index"]
        adapter._q_cmd = np.asarray(row["q_cmd"], dtype=float).copy()
        adapter._q_cmd_prev = np.asarray(row["q_cmd_prev"], dtype=float).copy()
        adapter._ins_cmd = float(row["insertion_cmd"])
        adapter._prev_input = np.asarray(row["previous_input"], dtype=float).copy()
        z = np.asarray(row["z_meas"], dtype=float)
        estimate = StateEstimate(
            t_monotonic=time.monotonic(), x_meas=np.zeros(6),
            tip_position_m=np.asarray(row["measured_beam_position"]),
            tip_tangent=np.zeros(3), insertion_length_m=row["insertion_cmd"], robot_joints=z[:6],
        )
        result = adapter(estimate, dt)
        cm = result.info["consecutive_deadline_misses"]
        trace.append((rep, result.info["success"], cm))
        print(f"  rep={rep}: success={result.info['success']} consecutive_deadline_misses={cm}  "
              f"{'<-- REAL ABORT THRESHOLD REACHED' if cm >= ABORT_THRESHOLD and threshold_reached_at is None else ''}")
        if cm >= ABORT_THRESHOLD and threshold_reached_at is None:
            threshold_reached_at = rep

    worker.close()
    print(f"\n[abort_threshold_subtest] threshold ({ABORT_THRESHOLD}) first reached at rep={threshold_reached_at}")
    assert threshold_reached_at is not None, "abort threshold was never reached -- counter logic may be broken"
    assert threshold_reached_at == ABORT_THRESHOLD, (
        f"expected threshold reached exactly at rep={ABORT_THRESHOLD} (consecutive_deadline_misses "
        f"increments by exactly 1 per genuine miss), got rep={threshold_reached_at}"
    )
    print("[abort_threshold_subtest] PASSED: consecutive_deadline_misses reaches the real "
          f"production abort threshold ({ABORT_THRESHOLD}) exactly when expected, under repeated "
          "genuine misses. NOTE: this confirms the ADAPTER's own counter correctly reaches the "
          "value close_loop_path_follow.py:2029 checks (`consecutive_misses >= 5`); it does not "
          "itself execute that file's full online control loop or its actual abort/break "
          "statement, which lives in a separate, much larger module not instantiated here.")


if __name__ == "__main__":
    main()
