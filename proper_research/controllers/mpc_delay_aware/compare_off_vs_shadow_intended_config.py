"""2026-11-19 (third review round), points 1+5: off vs shadow comparison
on the INTENDED hardware configuration -- the REAL plan used by both
official K2 STOP hardware runs (identified by hash cross-reference, see
module docstring below), NOT the September zraise30mm historical replay.

PLAN/SCHEDULE IDENTITY (verified, not assumed):
  plan:     plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/
            time_parameterized_configuration_path/time_parameterized_configuration_path.npz
            sha256 62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613
            -- matches the K2 operator procedure doc's recorded hash for
            "Original", i.e. the SAME plan both official K2 STOP runs
            (N=15, N=3) used. Verified bit-identical to
            plans/rectangle_10x15mm_skipglobal_ae1_2026-11-10/... (the
            source this session's Z4b-step-6 rebuild reproduced
            bit-for-bit) -- confirmed again at the top of main() below,
            not just asserted here.
  schedule: plans/rectangle_10x15mm_skipglobal_ae1_2026-11-10/
            jacobian_schedule_qn_zero.npy, sha256 computed and printed at
            runtime -- a genuine beam-model schedule (214,3,7), matching
            this exact plan's sample count.

NO real recorded hardware telemetry exists in this worktree for this
EXACT plan (the two official K2 STOP runs' own log directories are not
present here -- searched, confirmed absent). This comparison therefore
drives the real worker/adapter with the PLAN'S OWN nominal reference
trajectory as the "measured" state (q_meas=q_ref[k], tip=p_nom[k] exactly,
zero deviation) -- NOT a substitute for real noisy measurement, stated
plainly as a limitation. This has one deliberate, useful consequence per
the review's point 6: zero deviation means zero excitation, so ZERO
jacobian-adaptation acceptances are the EXPECTED, CORRECT outcome here,
not a failure -- demonstrating that "insufficient information" is handled
as a valid, non-alarming result, not silently inflated into a bogus
g_hat.

SETTINGS VERIFIED AGAINST THE K2 BASELINE (not silently inherited):
  - beam_base_xyz_m: plan's own recorded value vs common.py's current
    calibrated default -- confirmed IDENTICAL (checked at runtime below).
  - TCP-workspace-robust-margin-mm=15, horizon=15, max-tracking-error-mm=7:
    match the K2 operator procedure doc's own recorded settings exactly
    (set explicitly below, not left to a possibly-different code default).
  - magnet-exclusion-in-QP, dipole-alignment: BOTH are NEW since the K2
    runs (added 2026-11-18, after the K2 round) and default ON/weighted
    in the current rectangle runner -- EXPLICITLY DISABLED here
    (--disable-magnet-exclusion-in-qp equivalent, dipole weight=0) so
    this comparison matches the ACTUAL validated K2 baseline, not a
    silently-changed current default.
  - servo_stream_hz=50.0, max_joint_step_rad=0.01, control_hz=10.0: current
    common.py defaults, unchanged since before the K2 round -- no drift.

ORDER-BALANCED DESIGN: 4 repetitions, ABBA order (off, shadow, shadow,
off) -- counterbalances any linear warm-up/thermal trend across the run
sequence, rather than always running one condition first.

Run: python3 -m proper_research.controllers.mpc_delay_aware.compare_off_vs_shadow_intended_config
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online.messages import StateEstimate
from proper_research.hardware.online.rectangle_stage_a import common

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_ae1_2026-11-10/jacobian_schedule_qn_zero.npy"
EXPECTED_PLAN_SHA256 = "62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613"
HORIZON = 15
DT = 0.1
REAL_DEADLINE_S = 0.070
ORDER = ["off", "shadow", "shadow", "off"]  # ABBA, counterbalanced


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spawn(jac_mode: str) -> MPCWorkerHandle:
    mpc_config_kwargs = dict(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([common.JOINT_VELOCITY_LIMIT_RAD_S] * 6 + [0.002]),
        acceleration_limit=tuple([common.JOINT_ACCELERATION_LIMIT_RAD_S2] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=False,  # planned-input penalty, matching the reviewed K2 baseline
    )
    beam_config_kwargs = dict(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    jac_kwargs = None if jac_mode == "off" else dict(mode=jac_mode, excitation_floor_m=5e-4, forgetting_lambda=0.98)
    return MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=SCHEDULE_PATH,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True, nullspace_r700_config_kwargs=mpc_config_kwargs,
        # magnet_exclusion/dipole deliberately NOT passed here -- off, matching
        # the K2 baseline these features postdate.
        jacobian_adaptation_kwargs=jac_kwargs,
    )


def _run_one(rep: int, mode: str, ref) -> dict:
    worker = _spawn(mode)
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=worker, deadline_s=REAL_DEADLINE_S,
        joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
        max_joint_step_rad=common.MAX_JOINT_STEP_RAD,
        controller_kind="nullspace_r700", controller_label=f"compare_intended_rep{rep}_{mode}",
    )
    S = ref.sample_count
    solver_ms, wait_ms, turnaround_ms, statuses, consecutive_trace = [], [], [], [], []
    accepted = rejected = 0
    n_diag_present = 0
    jac_scale_values = []
    tick_start = time.monotonic()
    for i in range(S):
        q_ref6 = np.asarray(ref.state[i, :6], dtype=float)
        L_ref = float(ref.state[i, 6])
        tip_nom = np.asarray(ref.desired_position_m[i], dtype=float)
        estimate = StateEstimate(
            t_monotonic=time.monotonic(), x_meas=np.concatenate([tip_nom, np.zeros(3)]),
            tip_position_m=tip_nom, tip_tangent=np.array([1.0, 0.0, 0.0]),
            insertion_length_m=L_ref, robot_joints=q_ref6, frame_index=i,
        )
        t_send = time.monotonic()
        result = adapter(estimate, DT)
        total_ms = (time.monotonic() - t_send) * 1e3
        info = result.info
        statuses.append(info["status"])
        consecutive_trace.append(info["consecutive_deadline_misses"])
        if info["status"] == "ok":
            solver_ms.append(info["solve_time_s"] * 1e3)
            wait_ms.append(max(0.0, total_ms - info["solve_time_s"] * 1e3))
        turnaround_ms.append(total_ms)
        # 2026-11-19 fix verification: `info["jacobian_adaptation_diagnostics"]`
        # / `info["jacobian_scale_applied"]` now exist (see process_isolated_
        # adapter.py's 2026-11-19 fix) -- confirm they are ACTUALLY present
        # on this real tick's info dict, not just in the discarded raw resp.
        diag = info.get("jacobian_adaptation_diagnostics")
        if diag is not None:
            n_diag_present += 1
            if diag["accepted"]:
                accepted += 1
            else:
                rejected += 1
        jac_scale_values.append(info.get("jacobian_scale_applied"))
        elapsed = time.monotonic() - tick_start
        target = (i + 1) * DT
        if target > elapsed:
            time.sleep(target - elapsed)
    worker.close()
    n_misses = sum(1 for s in statuses if s != "ok")
    jac_scale_nonzero = [v for v in jac_scale_values if v is not None and v != 0.0]
    return {
        "rep": rep, "mode": mode, "n_ticks": S, "n_misses": n_misses,
        "max_consecutive_deadline_misses": max(consecutive_trace, default=0),
        "would_abort": max(consecutive_trace, default=0) >= 5,
        "solver_time_ms_mean": float(np.mean(solver_ms)) if solver_ms else float("nan"),
        "solver_time_ms_max": float(np.max(solver_ms)) if solver_ms else float("nan"),
        "response_wait_ms_mean": float(np.mean(wait_ms)) if wait_ms else float("nan"),
        "total_turnaround_ms_mean": float(np.mean(turnaround_ms)),
        "total_turnaround_ms_max": float(np.max(turnaround_ms)),
        "n_diag_present_in_info": n_diag_present,
        "n_accepted": accepted, "n_rejected": rejected,
        "n_jacobian_scale_nonzero": len(jac_scale_nonzero),  # must be 0 for mode in ("off","shadow")
    }


def main() -> None:
    plan_sha = _sha256(PLAN_DIR + "/time_parameterized_configuration_path.npz")
    assert plan_sha == EXPECTED_PLAN_SHA256, (
        f"PLAN HASH MISMATCH: got {plan_sha}, expected {EXPECTED_PLAN_SHA256} "
        f"(the recorded K2 'Original' plan hash) -- refusing to proceed with the wrong plan"
    )
    schedule_sha = _sha256(SCHEDULE_PATH)
    print(f"[compare_intended] plan sha256={plan_sha} (VERIFIED == K2 Original)")
    print(f"[compare_intended] schedule sha256={schedule_sha}")

    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    sched = np.load(SCHEDULE_PATH)
    assert sched.shape[0] == ref.sample_count, "schedule/plan sample-count mismatch"
    assert np.all(np.isfinite(sched))

    current_beam_base = common._calibrated_default_initial_poses_for_guard()
    plan_meta = json.loads(Path(PLAN_DIR).parent.joinpath("shape_rectangle", "plan_shape_summary.json").read_text())
    plan_beam_base = np.asarray(plan_meta["beam_base_xyz_m"])
    assert np.allclose(current_beam_base, plan_beam_base, atol=1e-6), (
        f"beam base drift detected: plan={plan_beam_base.tolist()} vs current={current_beam_base.tolist()}"
    )
    print(f"[compare_intended] beam_base_xyz_m VERIFIED unchanged vs plan's own recorded value: "
          f"{plan_beam_base.tolist()}")
    print(f"[compare_intended] magnet-exclusion-in-QP and dipole-alignment: DISABLED "
          f"(both postdate the K2 baseline being reproduced here)")
    print(f"[compare_intended] {ref.sample_count} reference samples, order={ORDER}")

    all_reps = []
    for rep, mode in enumerate(ORDER, start=1):
        print(f"\n[compare_intended] --- rep {rep}/{len(ORDER)}: mode={mode} ---")
        r = _run_one(rep, mode, ref)
        all_reps.append(r)
        print(f"  misses={r['n_misses']}/{r['n_ticks']} max_consecutive={r['max_consecutive_deadline_misses']} "
              f"would_abort={r['would_abort']}")
        print(f"  solver_ms(mean/max)={r['solver_time_ms_mean']:.2f}/{r['solver_time_ms_max']:.2f} "
              f"wait_ms(mean)={r['response_wait_ms_mean']:.2f} "
              f"turnaround_ms(mean/max)={r['total_turnaround_ms_mean']:.2f}/{r['total_turnaround_ms_max']:.2f}")
        print(f"  diagnostics present in info dict: {r['n_diag_present_in_info']}/{r['n_ticks']} ticks "
              f"(accepted={r['n_accepted']}, rejected={r['n_rejected']}) "
              f"jacobian_scale_applied nonzero count={r['n_jacobian_scale_nonzero']} (must be 0)")

    out = Path("close_loop_logs/myrun/compare_off_vs_shadow_intended_config_result.json")
    out.write_text(json.dumps(all_reps, indent=2))
    print(f"\n[compare_intended] ALL {len(ORDER)} repetitions retained -> {out}")


if __name__ == "__main__":
    main()
