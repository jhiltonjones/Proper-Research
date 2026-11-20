"""EXPLORATORY DIAGNOSTIC, not a regression test -- reports numbers, has no
pass/fail assertions. A single-hypothesis timing probe, companion to
`tcp_fix_full_replay_timing.py` (same replay methodology, see that
file's own docstring for the full historical-state-replay caveats).

Tests ONE concrete, cheap, safe-offline hypothesis for the occasional
70ms-deadline exceedances seen in the full replay: that Python's garbage
collector, running during the long (200+-tick) sustained in-process
sequential replay, is responsible. Disables `gc` entirely at the top of
this script and reruns the marginfix plan's full replay.

RESULT (2026-11-16, see the investigation report for the full numbers):
disabling GC did NOT reduce the spikes -- max total time and the exceedance
count were both equal-or-worse with GC disabled than with it enabled. This
rules out GC as the cause. Combined with the mocked-execution test's own
finding (the exact same "slow" tick inputs complete in ~20ms when run in
an isolated, freshly-spawned worker rather than as part of a sustained
sequential run), the evidence points to transient OS-level scheduling
contention on this shared, non-realtime development machine, not a
deterministic cost in the matrix-update or solver code.

Usage
-----
    python3 -m proper_research.studies.mpc_ablations.tcp_fix_gc_disabled_timing_probe

Dependencies: numpy, osqp; no git, no hardware. Marginfix plan only (the
original plan was not separately re-tested under this probe, since the
hypothesis is about general Python-process behavior, not plan-specific).
"""

import sys, json, time, gc
MAIN = "/home/jack/Proper-Research"
sys.path.insert(0, MAIN)  # CANONICAL checkout, fixed in place this round

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

gc.disable()  # item-4 hypothesis test: does disabling the GC remove the timing spikes?
import numpy as np
import math

from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
    InsertionAnchoredTaskNullspaceMPC,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.hardware.online.rectangle_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik


def build_controller(plan_dir, schedule_path, cm, cfg):
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
    s = float(cfg["mpc_position_error_scale_mm"]) * 1.0e-3
    beam_config_kwargs = dict(
        position_error_scale_m=(s, s, s), position_tracking_weight=float(cfg["mpc_position_tracking_weight"]),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(schedule_path)

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

    null_mpc_config = ConfigurationMPCConfig(**nullspace_r700_config_kwargs)
    beam_config = BeamOutputMPCConfig(**beam_config_kwargs)
    controller = InsertionAnchoredTaskNullspaceMPC(
        gamma=0.0, insertion_state_anchor_weight=cm["insertion_state_anchor_weight"],
        reference=reference, config=null_mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=cm["delay_samples"], beta_d=cm["beta_d"],
        tcp_workspace=tcp_workspace_kwargs,
    )
    return controller, horizon


def tick_inputs(rows_by_step, step):
    row = rows_by_step[step]
    prev_row = rows_by_step[step - 1]
    prev2_row = rows_by_step[step - 2]
    control_index = row["ref_index"]
    q_cmd = np.asarray(prev_row["q_target_rad"], dtype=float)
    q_cmd_prev = np.asarray(prev2_row["q_target_rad"], dtype=float)
    insertion_cmd = float(prev_row["insertion_length_m"])
    previous_input = np.asarray(prev_row["u0"], dtype=float)
    measured_beam_position = np.asarray(row["tip_mm"], dtype=float) / 1e3
    z_meas = np.concatenate([np.asarray(row["q_meas_rad"], dtype=float), [insertion_cmd]])
    return dict(
        control_index=control_index, q_cmd=q_cmd, q_cmd_prev=q_cmd_prev,
        insertion_cmd=insertion_cmd, previous_input=previous_input,
        measured_beam_position=measured_beam_position, z_meas=z_meas,
    )


def run_full_replay(label, plan_dir, schedule_path, run_dir, start_step=1):
    cm = json.load(open(f"{run_dir}/controller_metadata.json"))
    sm = json.load(open(f"{run_dir}/summary.json"))
    cfg = sm["config"]
    rows = [json.loads(l) for l in open(f"{run_dir}/path_follow.jsonl")]
    rows_by_step = {r["step"]: r for r in rows}
    max_step = max(rows_by_step.keys())

    controller, horizon = build_controller(plan_dir, schedule_path, cm, cfg)
    print(f"\n=== {label}: FULL replay, steps {start_step}..{max_step}, horizon={horizon}, "
          f"sample_count={controller.reference.sample_count} (FIXED code) ===")

    DEADLINE_MS = 70.0
    results = []
    first_fail = None
    for step in range(start_step, max_step + 1):
        if step - 1 not in rows_by_step or step - 2 not in rows_by_step:
            continue
        inp = tick_inputs(rows_by_step, step)

        # Isolate matrix-update time vs solver time directly (not just total).
        t0 = time.perf_counter()
        x_exec = np.concatenate([inp["q_cmd"], inp["q_cmd_prev"], [inp["insertion_cmd"]]])
        instantaneous, estimated = controller._estimate_output_residual(
            measured_state=inp["z_meas"], measured_beam_position=inp["measured_beam_position"],
            control_index=inp["control_index"],
        )
        hessian, linear_cost, beam_terms = controller._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=inp["previous_input"], control_index=inp["control_index"],
            estimated_residual=estimated,
        )
        lower, upper = controller._constraint_bounds_exec(
            x_exec=x_exec, previous_input=inp["previous_input"], control_index=inp["control_index"],
        )
        t1 = time.perf_counter()  # end of matrix/bound build (includes TCP Jacobian linearization)
        warm_start = controller._feasible_warm_start(
            input_reference=beam_terms["input_reference"], previous_input=inp["previous_input"],
        )
        update_kwargs = dict(Px=controller._upper_values(hessian), q=linear_cost, l=lower, u=upper)
        if (controller._magnet_excl_radius is not None or controller._magnet_ws_z_min is not None
                or controller._tcp_ws_min is not None):
            update_kwargs["Ax"] = controller._a_values(controller.A)
        t2 = time.perf_counter()  # end of Ax-values extraction (the fix's own added cost)
        controller._solver.update(**update_kwargs)
        controller._solver.warm_start(x=warm_start)
        result = controller._solver.solve()
        t3 = time.perf_counter()  # end of OSQP solve
        status = str(result.info.status).lower()
        success = status in {"solved", "solved inaccurate"}
        if success:
            controller._warm_start = np.asarray(result.x, dtype=float).copy()

        matrix_build_ms = (t1 - t0) * 1e3
        ax_extract_ms = (t2 - t1) * 1e3
        solver_ms = (t3 - t2) * 1e3
        total_ms = (t3 - t0) * 1e3
        results.append(dict(
            step=step, control_index=inp["control_index"], success=success, status=status,
            iters=int(result.info.iter), matrix_build_ms=matrix_build_ms,
            ax_extract_ms=ax_extract_ms, solver_ms=solver_ms, total_ms=total_ms,
            q_cmd=inp["q_cmd"].tolist(), q_cmd_prev=inp["q_cmd_prev"].tolist(),
            previous_input=inp["previous_input"].tolist(), insertion_cmd=inp["insertion_cmd"],
            z_meas=inp["z_meas"].tolist(), measured_beam_position=inp["measured_beam_position"].tolist(),
        ))
        if not success and first_fail is None:
            first_fail = step
            print(f"  FIRST FAILURE at step={step} control_index={inp['control_index']} "
                  f"status={status} iters={result.info.iter}")
            # Confirm hold/abort semantics: do NOT force success, just record and
            # continue observing (mirrors the real loop holding q_cmd and counting
            # consecutive misses) -- but stop the replay here since downstream
            # rows reflect the REAL run's own held/frozen state after its failure,
            # not a valid counterfactual continuation for a fixed controller.
            break

    n = len(results)
    n_success = sum(r["success"] for r in results)
    print(f"  Ticks replayed: {n}  successes: {n_success}  failures: {n - n_success}")
    if first_fail is None:
        print(f"  COMPLETED to the end of the real log's recorded ticks WITHOUT any QP failure.")
    all_total = [r["total_ms"] for r in results]
    all_matrix = [r["matrix_build_ms"] for r in results]
    all_ax = [r["ax_extract_ms"] for r in results]
    all_solver = [r["solver_ms"] for r in results]
    over = sum(1 for t in all_total if t > DEADLINE_MS)
    print(f"  matrix-build ms: mean={np.mean(all_matrix):.2f} median={np.median(all_matrix):.2f} max={np.max(all_matrix):.2f}")
    print(f"  Ax-extract ms:   mean={np.mean(all_ax):.2f} median={np.median(all_ax):.2f} max={np.max(all_ax):.2f}")
    print(f"  solver ms:       mean={np.mean(all_solver):.2f} median={np.median(all_solver):.2f} max={np.max(all_solver):.2f}")
    print(f"  TOTAL ms:        mean={np.mean(all_total):.2f} median={np.median(all_total):.2f} max={np.max(all_total):.2f}")
    print(f"  ticks over {DEADLINE_MS}ms deadline: {over}/{n}")
    return dict(label=label, results=results, first_fail=first_fail, n=n, n_success=n_success,
                over_deadline=over)


print("osqp/controller code loaded from:", InsertionAnchoredTaskNullspaceMPC.__module__,
      InsertionAnchoredTaskNullspaceMPC.__mro__[0].__module__)
import proper_research.controllers.mpc_delay_aware.delay_aware_mpc as dam
print("delay_aware_mpc module file:", dam.__file__)
assert dam.__file__ == MAIN + "/proper_research/controllers/mpc_delay_aware/delay_aware_mpc.py", (
    f"SAFETY CHECK FAILED: not using the canonical checkout's module! got {dam.__file__}"
)
assert ".claude/worktrees" not in dam.__file__, "SAFETY CHECK FAILED: resolved to a worktree path!"

# --- marginfix plan (230 samples) ---
r1 = run_full_replay(
    "MARGINFIX plan (230 samples)",
    f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path",
    f"{MAIN}/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy",
    f"{MAIN}/close_loop_logs/myrun/k2_instrumented_repeat_20261116T113659Z_20261116T113717Z",
)

print("\n\n=== SUMMARY (gc.disable() variant, marginfix plan only) ===")
print(f"{r1['label']}: {r1['n_success']}/{r1['n']} ticks succeeded, "
      f"first_fail={r1['first_fail']}, over_deadline={r1['over_deadline']}/{r1['n']}")

import json as _json
_out_path = f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_gc_disabled_per_tick.json"
if _os.path.exists(_out_path):
    raise FileExistsError(
        f"REFUSING TO OVERWRITE: {_out_path} already exists. Move/rename it first if it is "
        f"genuinely safe to replace -- this script will never silently overwrite a prior result."
    )
import os as _os
_json.dump({"marginfix_gc_disabled": r1["results"]}, open(_out_path, "w"))
print(f"saved per-tick data to {_out_path}")
