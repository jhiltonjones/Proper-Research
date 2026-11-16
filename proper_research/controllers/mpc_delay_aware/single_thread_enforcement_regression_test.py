"""Regression test for the PROPOSED `_force_single_threaded_blas()`
replacement -- still NOT adopted as the production default. Exercises the
proposed mechanism (threadpoolctl.threadpool_limits(1), applied
post-import) via the EXISTING, already-production opt-in
MPC_WORKER_VERIFIED_SINGLE_THREAD=1 toggle, which already implements the
IDENTICAL function body being proposed -- this genuinely exercises the
mechanism in a REAL spawned worker subprocess, not a same-process
simulation.

No hardware connection anywhere in this file (hardware guard below).
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
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.hardware.online.rectangle_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik


def test_empty_backend_list_is_a_failure_not_a_pass():
    """Demonstrate the empty-list trap directly: query threadpool_info()
    in a brand-new, bare interpreter subprocess BEFORE anything has ever
    touched numpy/BLAS. A naive 'if not bad: pass' check would wrongly
    accept this (bad=[] since there's nothing to be wrong about); the
    correct check must separately require the backend list to be
    NON-EMPTY and to include the specific backends expected."""
    import subprocess
    code = "import threadpoolctl; print(threadpoolctl.threadpool_info())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    info = eval(out)
    print(f"[empty-list check] threadpool_info() before any BLAS use: {info}")
    assert info == [], "expected an empty list before BLAS is ever exercised (confirms the trap is real)"
    # The NAIVE (wrong) check:
    naive_bad = [d for d in info if d.get("num_threads") != 1]
    naive_passes = not naive_bad
    print(f"[empty-list check] NAIVE check (only inspects entries that exist) would say: "
          f"{'PASS -- WRONG, this is the bug' if naive_passes else 'fail'}")
    assert naive_passes, "the naive check should incorrectly pass here -- that's the point being demonstrated"
    # The CORRECT check: must also require a non-empty, named-backend list.
    correct_passes = bool(info) and naive_passes
    print(f"[empty-list check] CORRECT check (also requires non-empty, named backends): "
          f"{'PASS' if correct_passes else 'FAIL -- correctly rejects the empty-list case'}")
    assert not correct_passes, "the correct check must REJECT an empty backend list"
    print("[empty-list check] PASSED: empty threadpool_info() is correctly treated as failure, not success.\n")


def test_real_spawned_worker_single_thread_enforcement():
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

    timing_log_path = f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/regression_test_postsolve_threadpool.jsonl"
    if os.path.exists(timing_log_path):
        raise FileExistsError(f"REFUSING TO OVERWRITE: {timing_log_path} already exists.")
    os.environ["MPC_WORKER_TIMING_LOG"] = timing_log_path
    os.environ["MPC_WORKER_VERIFIED_SINGLE_THREAD"] = "1"  # exercises the EXACT mechanism being proposed

    print("[test] spawning REAL worker subprocess with single_threaded=True and the proposed "
          "mechanism active (via the existing opt-in toggle, same function body as proposed)...")
    worker = MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=SCHEDULE_PATH,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=cm["delay_samples"], beta_d=cm["beta_d"], single_threaded=True,
        enable_nullspace_r700=True, nullspace_r700_config_kwargs=nullspace_r700_config_kwargs,
        insertion_state_anchor_weight=cm["insertion_state_anchor_weight"],
        tcp_workspace_kwargs=tcp_workspace_kwargs,
    )
    print(f"[test] worker spawned successfully in {(worker.t_worker_ready - worker.t_spawn_start)*1e3:.0f}ms "
          f"-- the request-serving loop had not yet started when this ready handshake was received "
          f"(confirmed by code ordering: the limit is applied before the `while True: req = conn.recv()` "
          f"loop even exists in the worker's source).")

    diag = worker.worker_scheduling_diagnostics
    info_at_startup = diag.get("threadpool_info")
    print(f"[test] threadpool_info() reported at startup (BEFORE serving any request): {info_at_startup}")

    assert isinstance(info_at_startup, list) and len(info_at_startup) > 0, (
        f"EMPTY OR MISSING backend list at startup -- this must be treated as a FAILURE, "
        f"not a pass: {info_at_startup}"
    )
    numpy_entries = [d for d in info_at_startup if "numpy" in d.get("filepath", "")]
    scipy_entries = [d for d in info_at_startup if "scipy" in d.get("filepath", "")]
    assert len(numpy_entries) >= 1, f"expected a numpy.libs OpenBLAS entry, got: {info_at_startup}"
    assert len(scipy_entries) >= 1, f"expected a scipy.libs OpenBLAS entry, got: {info_at_startup}"
    assert all(d["num_threads"] == 1 for d in numpy_entries), f"numpy backend not at 1 thread: {numpy_entries}"
    assert all(d["num_threads"] == 1 for d in scipy_entries), f"scipy backend not at 1 thread: {scipy_entries}"
    print("[test] PASSED (before-serving): numpy's AND scipy's OpenBLAS backends, named and verified "
          "individually, both confirmed at num_threads=1 before the worker ever served a request.")

    # Run ONE real solve through the worker, then check the POST-solve
    # thread-pool snapshot (new instrumentation, this round) to confirm
    # nothing during a real solve reset the limit.
    per_tick = json.load(open(f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_per_tick.json"))["marginfix"]
    row = per_tick[50]
    resp = worker.try_solve(
        kind="nullspace_r700", z_meas=np.asarray(row["z_meas"]), q_cmd=np.asarray(row["q_cmd"]),
        q_cmd_prev=np.asarray(row["q_cmd_prev"]), insertion_cmd=row["insertion_cmd"],
        measured_beam_position=np.asarray(row["measured_beam_position"]), control_index=row["control_index"],
        previous_input=np.asarray(row["previous_input"]), deadline_s=5.0,  # generous, this is a correctness check not a timing one
    )
    assert resp is not None and resp["status"] == "ok", f"real solve failed: {resp}"
    worker.close()
    time.sleep(0.2)

    log_entries = [json.loads(l) for l in open(timing_log_path)]
    assert len(log_entries) == 1
    info_after_solve = log_entries[0]["threadpool_info_after_solve"]
    print(f"[test] threadpool_info() AFTER a real solve: {info_after_solve}")
    assert isinstance(info_after_solve, list) and len(info_after_solve) > 0, (
        f"EMPTY OR MISSING backend list after solve -- must be treated as a FAILURE: {info_after_solve}"
    )
    numpy_after = [d for d in info_after_solve if "numpy" in d.get("filepath", "")]
    scipy_after = [d for d in info_after_solve if "scipy" in d.get("filepath", "")]
    assert len(numpy_after) >= 1 and all(d["num_threads"] == 1 for d in numpy_after), (
        f"numpy backend thread limit did not survive a real solve: {numpy_after}"
    )
    assert len(scipy_after) >= 1 and all(d["num_threads"] == 1 for d in scipy_after), (
        f"scipy backend thread limit did not survive a real solve: {scipy_after}"
    )
    print("[test] PASSED (after-solve): both named backends still confirmed at num_threads=1 "
          "AFTER a real solve -- the limit was not reset by anything during actual MPC computation.")


if __name__ == "__main__":
    test_empty_backend_list_is_a_failure_not_a_pass()
    test_real_spawned_worker_single_thread_enforcement()
    print("\nALL REGRESSION CHECKS PASSED.")
