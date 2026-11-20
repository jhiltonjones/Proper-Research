"""Sustained real-worker deadline test, 100ms cadence, REAL 70ms deadline,
normal OS scheduling (no priority/affinity changes). Uses the REAL
MPCWorkerHandle (spawns a genuine worker subprocess) and REAL
ProcessIsolatedDelayAwareAdapter -- no reimplementation. No robot/camera
interface is ever constructed (same hardware guard as the mocked
execution self-test).

PREDEFINED repetition count, stated before running: N_REPS = 3 sustained
runs, each covering the same representative 30-tick window of real,
historically-recorded marginfix-plan state (steps 185-214, covering the
plan's two measured 70ms-deadline exceedances at steps 198/204). Every
result from every run is reported, including any miss, not a cherry-
picked subset.

This is a MOCKED EXECUTION TEST (checks deadline handling/bookkeeping
mechanics), not a historical-state replay (which only checks solvability/
timing on recorded inputs) and does NOT establish physical tracking
accuracy or full-path feasibility on real hardware.

Run location: this dev machine (not necessarily the intended lab
computer) -- see the investigation report for which numbers came from
here vs. what's prepared for the user to run on the actual target
machine.
"""
import sys, json, time, os
MAIN = "/home/jack/Proper-Research"
sys.path.insert(0, MAIN)
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

# RUNTIME hardware guard (same technique as the mocked execution self-test).
_FORBIDDEN_HW_IDENTIFIERS = ("URRtde", "RobotJointStream", "CameraSource",
                             "rtde_control", "rtde_receive", "RTDEControlInterface", "RTDEReceiveInterface")
with open(__file__, "r") as _f:
    _own_source = _f.read()
for _ident in _FORBIDDEN_HW_IDENTIFIERS:
    _occ = _own_source.count(_ident)
    assert _occ <= 2, f"HARDWARE GUARD FAILED: {_ident!r} appears {_occ} times -- expected <=2 (docstring/tuple only)"
print("[hardware guard] confirmed: no robot/camera interface imported or constructed.")

N_REPS = 3  # PREDEFINED before running, per the coordinator's explicit instruction
CADENCE_S = 0.100  # real online control-loop rate
DEADLINE_S = 0.070  # real, unmodified deadline
ABORT_CONSECUTIVE_MISSES = 5  # real threshold, traced directly in close_loop_path_follow.py:2029


def scheduling_diagnostics():
    diag = {}
    try:
        diag["sched_getaffinity"] = sorted(os.sched_getaffinity(0))
    except Exception as e:
        diag["sched_getaffinity"] = f"unavailable: {e}"
    try:
        diag["load_average"] = os.getloadavg()
    except Exception as e:
        diag["load_average"] = f"unavailable: {e}"
    try:
        diag["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    except Exception as e:
        diag["nice"] = f"unavailable: {e}"
    try:
        diag["cpu_count"] = os.cpu_count()
    except Exception as e:
        diag["cpu_count"] = f"unavailable: {e}"
    return diag


def build_worker():
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
    return worker, reference, cfg, dt


def instrument_adapter_timing(adapter):
    """Non-invasive timing wrapper around the REAL Connection primitives
    try_solve uses internally (send/poll/recv) -- records separate
    timestamps WITHOUT changing try_solve's logic, return values, or
    control flow at all. The one-outstanding-request/deadline/seq-match
    decisions remain 100% the real, unmodified try_solve code."""
    conn = adapter._worker._parent_conn
    timing_log = []
    orig_send, orig_poll, orig_recv = conn.send, conn.poll, conn.recv

    def timed_send(obj):
        t0 = time.monotonic()
        result = orig_send(obj)
        timing_log.append(("send", time.monotonic() - t0))
        return result

    def timed_poll(timeout=None):
        t0 = time.monotonic()
        result = orig_poll(timeout)
        timing_log.append(("poll", time.monotonic() - t0))
        return result

    def timed_recv():
        t0 = time.monotonic()
        result = orig_recv()
        timing_log.append(("recv", time.monotonic() - t0))
        return result

    conn.send, conn.poll, conn.recv = timed_send, timed_poll, timed_recv
    return timing_log


def main():
    import os as _os  # moved to the top: referenced earlier in this function than its old import site
    print(f"[sustained_test] PREDEFINED repetition count: N_REPS={N_REPS}, cadence={CADENCE_S*1e3:.0f}ms, "
          f"deadline={DEADLINE_S*1e3:.0f}ms (real, unmodified), real abort threshold="
          f"{ABORT_CONSECUTIVE_MISSES} consecutive misses (traced in close_loop_path_follow.py:2029)")
    print("[sustained_test] NORMAL OS scheduling -- no priority/affinity changes for this baseline.")
    print(f"[sustained_test] scheduling diagnostics (this process, at startup): {scheduling_diagnostics()}")

    per_tick = json.load(open(f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_per_tick.json"))["marginfix"]
    by_step = {r["step"]: r for r in per_tick}
    test_steps = [s for s in range(185, 215) if s in by_step]  # representative window incl. the 2 measured exceedances

    print("\n[sustained_test] spawning REAL MPCWorkerHandle...")
    worker, reference, cfg, dt = build_worker()
    print(f"[sustained_test] worker ready in {(worker.t_worker_ready - worker.t_spawn_start)*1e3:.0f}ms")
    _parent_sched = scheduling_diagnostics()
    _worker_sched = worker.worker_scheduling_diagnostics
    # 2026-11-16: if the run label claims nice=0, verify the WORKER (not
    # just the parent) is actually at nice=0 before proceeding -- a
    # spawned subprocess is not guaranteed to inherit the exact same
    # value, and a silently-still-elevated worker must never be
    # presented as part of a "nice=0" comparison.
    _run_label_check = _os.environ.get("SUSTAINED_TEST_RUN_LABEL", "")
    if "nice0" in _run_label_check.lower():
        _worker_nice_actual = _worker_sched.get("nice") if isinstance(_worker_sched, dict) else None
        if _worker_nice_actual != 0:
            worker.close()
            raise SystemExit(
                f"REFUSING TO PROCEED: run label {_run_label_check!r} claims nice=0, but the "
                f"WORKER subprocess is actually at nice={_worker_nice_actual!r}. This run will "
                f"NOT be reported as a nice=0 comparison."
            )
        print("[nice=0 check] WORKER confirmed at nice=0 (verified, not assumed).")
    print(f"[sustained_test] PARENT scheduling diagnostics: {_parent_sched}")
    print(f"[sustained_test] WORKER scheduling diagnostics (captured independently inside the "
          f"spawned subprocess -- NOT assumed to match the parent): {_worker_sched}")

    config = OfflineControllerConfig()
    config.progress_mode = "counter"
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=reference, worker=worker, deadline_s=DEADLINE_S, config=config,
        joint_velocity_limit_rad_s=cfg["joint_velocity_limit_rad_s"], max_joint_step_rad=0.1,
        controller_kind="nullspace_r700", controller_label="sustained_test",
    )
    timing_log = instrument_adapter_timing(adapter)

    all_runs = []
    for rep in range(1, N_REPS + 1):
        print(f"\n[sustained_test] === REP {rep}/{N_REPS} ===")
        run_results = []
        for step in test_steps:
            row = by_step[step]
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
            timing_log.clear()
            t_call0 = time.monotonic()
            result = adapter(estimate, dt)
            t_call1 = time.monotonic()

            send_s = sum(d for k, d in timing_log if k == "send")
            poll_s = sum(d for k, d in timing_log if k == "poll")
            recv_s = sum(d for k, d in timing_log if k == "recv")
            total_adapter_ms = (t_call1 - t_call0) * 1e3

            run_results.append(dict(
                step=step, control_index=row["control_index"],
                prep_send_ms=send_s * 1e3, response_wait_ms=poll_s * 1e3, recv_deserialize_ms=recv_s * 1e3,
                total_adapter_ms=total_adapter_ms, success=result.info["success"], status=result.info["status"],
                worker_t_solve_ms=result.info.get("solve_time_s", float("nan")) * 1e3,
                consecutive_misses=result.info["consecutive_deadline_misses"],
            ))
            print(f"  step={step}: prep_send={send_s*1e3:.3f}ms response_wait={poll_s*1e3:.2f}ms "
                  f"recv={recv_s*1e3:.3f}ms total_adapter={total_adapter_ms:.2f}ms "
                  f"worker_t_solve={result.info.get('solve_time_s', float('nan'))*1e3:.2f}ms "
                  f"success={result.info['success']} status={result.info['status']!r}")
            # pace at the real 100ms cadence
            elapsed = time.monotonic() - t_call0
            if elapsed < CADENCE_S:
                time.sleep(CADENCE_S - elapsed)
        all_runs.append(run_results)

    worker.close()

    print("\n\n[sustained_test] === SUMMARY across all N_REPS runs, every tick reported ===")
    any_miss_anywhere = False
    for rep_idx, run_results in enumerate(all_runs, start=1):
        misses = [r for r in run_results if not r["success"]]
        if misses:
            any_miss_anywhere = True
        print(f"  rep {rep_idx}: {len(run_results)} ticks, {len(misses)} misses: "
              f"{[m['step'] for m in misses]}")
    print(f"\n  ANY miss across all {N_REPS} reps under REAL 70ms deadline / 100ms cadence / normal "
          f"scheduling: {any_miss_anywhere}")

    _os.makedirs(f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics", exist_ok=True)
    # 2026-11-16: filename ALWAYS includes the run label, never a bare
    # fixed name -- a prior round of this investigation accidentally
    # overwrote a real, already-collected result set by running this
    # script again at the old fixed path; this makes that mistake
    # structurally impossible going forward.
    _run_label_for_path = _os.environ.get("SUSTAINED_TEST_RUN_LABEL")
    if not _run_label_for_path:
        import datetime as _dt
        _run_label_for_path = "unlabeled_" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    _out_path = (
        f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/"
        f"sustained_worker_test_results_{_run_label_for_path}.json"
    )
    _json = __import__("json")
    _results_payload = {
        "run_label": _os.environ.get("SUSTAINED_TEST_RUN_LABEL", "unlabeled"),
        "parent_scheduling_diagnostics": _parent_sched,
        "worker_scheduling_diagnostics": _worker_sched,
        "runs": all_runs,
    }
    # 2026-11-16 (structural fix, not just a naming convention): REFUSE to
    # proceed if the target path already exists, regardless of why (label
    # reused by mistake, by a second person, or anything else) -- a prior
    # round of this investigation overwrote a real, already-collected
    # result set this way. A run-label-based filename alone does not
    # close this; checking existence at write time does.
    if _os.path.exists(_out_path):
        raise FileExistsError(
            f"REFUSING TO OVERWRITE: {_out_path} already exists. Choose a "
            f"different SUSTAINED_TEST_RUN_LABEL (or move/rename the "
            f"existing file first if it is genuinely safe to replace) and "
            f"rerun -- this script will never silently overwrite a prior "
            f"result file."
        )
    _json.dump(_results_payload, open(_out_path, "w"))
    print(f"saved full results (including parent+worker scheduling diagnostics) to {_out_path}")
    return all_runs


if __name__ == "__main__":
    main()
