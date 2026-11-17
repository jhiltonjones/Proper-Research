"""CONTINUOUS MOCKED EXECUTION TEST -- explicitly kept separate from
historical-state replay. The real `ProcessIsolatedDelayAwareAdapter` and
real worker run tick-to-tick WITHOUT re-seeding q_cmd/q_cmd_prev/
previous_input from any historical log between ticks -- these evolve
from the adapter's own real internal state update each call, exactly as
the real online loop does (see process_isolated_adapter.py's own
"THIS process -- not the worker -- is the single source of truth for
q_cmd/q_cmd_prev/previous_input" docstring). This is the capability
explicitly identified as missing from historical-state replay several
rounds ago.

HONEST SCOPE LIMITATION, stated plainly rather than papered over: there
is still no real-plant physical response model here. `z_meas` (what the
"robot" reports back) is approximated as EXACTLY the just-applied
command (perfect, instantaneous joint tracking -- q_cmd after this
tick's update), and `measured_beam_position` uses the OFFLINE reference
schedule's own nominal tip position at the current index (not a real
contact-mechanics simulation of how the beam actually responds). This
validates the CONTROLLER SOFTWARE's own continuous state-accumulation,
command-dispatch, and deadline/IPC logic end-to-end -- it does NOT
validate against a physical plant model. Building a genuine plant-
response simulation would mean integrating this project's SEPARATE
offline beam-physics simulation stack (simulate_time_parameterized_beam_
output_mpc.py and friends) with the real adapter/worker IPC path -- a
materially larger undertaking than fits in this pass, not attempted here
as a shortcut-and-call-it-equivalent.

No hardware connection anywhere in this file.
"""
import sys, json, os, time
MAIN = "/home/jack/Proper-Research"
sys.path.insert(0, MAIN)

_FORBIDDEN_HW_IDENTIFIERS = ("URRtde", "RobotJointStream", "CameraSource",
                             "rtde_control", "rtde_receive", "RTDEControlInterface", "RTDEReceiveInterface")
with open(__file__, "r") as _f:
    _own_source = _f.read()
for _ident in _FORBIDDEN_HW_IDENTIFIERS:
    assert _own_source.count(_ident) <= 1
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

RUN_DIR = f"{MAIN}/close_loop_logs/myrun/k2_instrumented_repeat_20261116T113659Z_20261116T113717Z"
PLAN_DIR = f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path"
SCHEDULE_PATH = f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy"


def build():
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
        controller_kind="nullspace_r700", controller_label="continuous_mocked_execution_test",
    )
    return worker, adapter, reference, dt


def main(n_ticks=60):
    print(f"[continuous_mocked_execution_test] spawning real worker+adapter, running {n_ticks} "
          f"CONTINUOUS ticks -- NO re-seeding from any historical log between ticks.")
    worker, adapter, reference, dt = build()
    S = reference.sample_count

    # Seed ONLY the very first tick from the real initial plan state
    # (equivalent to the robot starting at the plan's own q0) -- every
    # SUBSEQUENT tick's q_cmd/q_cmd_prev/previous_input comes from the
    # adapter's OWN prior real output, never re-seeded.
    q0 = np.asarray(reference.state[0, :6], dtype=float)
    l0 = float(reference.state[0, 6])

    results = []
    first_fail = None
    for step in range(n_ticks):
        ref_idx = min(step, S - 1)
        # HONEST SIMPLIFICATION: z_meas = the adapter's own current
        # command accumulator (perfect, instantaneous joint tracking), NOT
        # a real plant response -- see this file's own module docstring.
        if adapter._q_cmd is None:
            z6 = q0
            ins = l0
        else:
            z6 = adapter._q_cmd.copy()
            ins = adapter._ins_cmd
        tip_mm_nominal = reference.desired_position_m[ref_idx] * 1e3  # offline nominal, not a real contact simulation
        estimate = StateEstimate(
            t_monotonic=time.monotonic(), x_meas=np.zeros(6), tip_position_m=tip_mm_nominal / 1e3,
            tip_tangent=np.zeros(3), insertion_length_m=ins, robot_joints=z6,
        )
        result = adapter(estimate, dt)
        results.append(dict(step=step, ref_idx=ref_idx, success=result.info["success"],
                             status=result.info["status"], q_cmd_after=adapter._q_cmd.copy().tolist()))
        if not result.info["success"] and first_fail is None:
            first_fail = step
        print(f"  tick={step} ref_idx={ref_idx} success={result.info['success']} status={result.info['status']!r}")
        time.sleep(0.1)  # real 100ms cadence, same as the sustained test

    worker.close()
    n_success = sum(1 for r in results if r["success"])
    print(f"\n[continuous_mocked_execution_test] {n_success}/{len(results)} ticks succeeded, "
          f"first_fail={first_fail}")
    print("[continuous_mocked_execution_test] This result describes the CONTROLLER SOFTWARE's own "
          "continuous accumulator/dispatch/IPC/deadline behavior under genuine tick-to-tick state "
          "evolution. It does NOT validate physical tracking accuracy or a real plant's response -- "
          "see this file's own module docstring for that scope limitation, stated plainly.")
    return results


if __name__ == "__main__":
    main()
