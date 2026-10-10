"""Same-state MPC counterfactual replay (Tasks 2 and 6).

Reconstructs the EXACT online MPC object (`ExactQNZeroTaskNullspaceDelayAwareMPC`,
the "exact_qn0_R700_vessel" variant every closed-loop MPC run in this
investigation actually used -- confirmed from controller_metadata.json's
`controller_variant` field) for a given hardware run, in-process (not through
the hardware's own process-isolated worker, which exists only to protect a
live RTDE/camera connection from a slow solve -- irrelevant here), so that its
`reference_position_jacobians` schedule can be swapped for an arbitrary
alternate Jacobian immediately before a `solve_delay_aware` call.

Per-tick ground truth comes from each run's OWN `predicted_beam_positions.jsonl`
(the adapter's own prediction/solve log -- NOT path_follow.jsonl), which
already contains the exact (z_meas, q_cmd_k, q_cmd_km1, insertion_cmd_m,
u_prev, u0, jacobian_used) the real tick was solved with. This makes
"solve the same tick again" a direct, non-inferential replay: every quantity
`solve_delay_aware` needs except the controller's own internal
`_filtered_output_residual` EMA state is read straight from this log.

`_filtered_output_residual` is replayed by calling the controller's own
`_estimate_output_residual` (no QP solve, cheap) at every tick from 0..k-1
using each tick's own logged z_meas/measured_beam_position/ref_index BEFORE
the REAL (un-swapped) schedule -- i.e. the disturbance estimate is always
built from what actually, physically happened, never from a counterfactual
model. Only the Jacobian the QP itself is built from (the horizon window used
inside `_beam_prediction_terms_exec`, index offset +1..+N) is swapped for the
counterfactual call.

Sanity check (run this file directly): for several ticks across 4 runs
(mix of contact/no-contact, 210mm/255mm), solve with the run's OWN historical
schedule (no swap) and compare against the logged u0. This is the single
most important check in this module -- see the printed summary table.

Read-only with respect to all proper_research/** source and all hardware
logs; writes only into tables/ when run as __main__.

Public API
----------
    load_tick_log(run_dir) -> list[dict]          one dict per logged tick
    build_controller(run_dir, meta=None) -> (controller, schedule_path, tick_log)
    replay_residual_state(controller, tick_log, k, schedule_path) -> None
        (mutates controller._filtered_output_residual in place to its
        value just before tick k's solve, by replaying ticks 0..k-1)
    solve_same_state(controller, tick_log, k, J_override=None) -> step
        (DelayAwareBeamOutputMPCStep; .command is u0, a (7,) array)
    workspace_margins(magnet_xyz_m, meta) -> dict[str, float]
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")

import numpy as np

import manifest  # noqa: E402
import loader  # noqa: E402
import live_jac  # noqa: E402

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (  # noqa: E402
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (  # noqa: E402
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.exact_qn_zero_mpc import (  # noqa: E402
    ExactQNZeroTaskNullspaceDelayAwareMPC,
)

REPO = "/home/jack/Proper-Research"
SCHED_DIR = f"{REPO}/plans/stage3_design/mpc_schedules"
SCHED_CANDIDATES = {
    True: [
        f"{SCHED_DIR}/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy",
        f"{SCHED_DIR}/vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy",
    ],
    False: [
        f"{SCHED_DIR}/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy",
    ],
}

# literal constants copied from run_mpc_delay_aware_vessel.py's
# spawn_and_warm_worker (the only caller that builds these kwargs for every
# run in this investigation -- position_error_scale_mm/position_tracking_weight
# are hardcoded there, not CLI-controlled)
_INPUT_TRACKING_WEIGHT = 7.0
_INPUT_INCREMENT_WEIGHT = 1.0e-3
_POSITION_ERROR_SCALE_MM = 0.5
_POSITION_TRACKING_WEIGHT = 1.0
_INSERTION_MAX_M = 0.110
_JOINT_VELOCITY_LIMIT_RAD_S = 0.3  # common.py default; margin-insensitive for in-range states, see module docstring
_JOINT_ACCEL_LIMIT_RAD_S2 = 2.0
_INSERTION_RATE_LIMIT_M_S = 0.002
_CONTROL_HZ = 10.0


def load_tick_log(run_dir: str) -> list[dict]:
    rows = []
    with open(f"{run_dir}/predicted_beam_positions.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_run_meta(dirname_or_rm):
    if isinstance(dirname_or_rm, dict):
        return dirname_or_rm
    for rm in manifest.RUNS:
        if rm["dirname"] == dirname_or_rm:
            return rm
    raise KeyError(dirname_or_rm)


def _pick_schedule_file(contact: bool, tick_log: list[dict]) -> str:
    """Empirically match the run's own logged jacobian_used (at several
    ticks, offset by +1 from ref_index per solve_delay_aware's
    _reference_indices(control_index, future=True)) against every candidate
    schedule file, rather than inferring the path from code/CLI defaults."""
    candidates = SCHED_CANDIDATES[contact]
    probe_idxs = np.linspace(0, len(tick_log) - 2, min(6, len(tick_log) - 1)).astype(int)
    best = None
    for path in candidates:
        sched = np.load(path)
        max_err = 0.0
        for k in probe_idxs:
            row = tick_log[int(k)]
            ju = np.asarray(row["jacobian_used"], dtype=float)
            idx = min(row["ref_index"] + 1, sched.shape[0] - 1)
            max_err = max(max_err, float(np.linalg.norm(sched[idx] - ju)))
        if best is None or max_err < best[1]:
            best = (path, max_err)
    assert best[1] < 1e-6, f"no schedule candidate matched (best max_err={best[1]:.3e}): {best[0]}"
    return best[0]


def build_controller(run_dir_or_dirname, meta: dict | None = None, force_schedule: np.ndarray | None = None):
    """Returns (controller, schedule_path, tick_log). `controller` is a
    FRESH ExactQNZeroTaskNullspaceDelayAwareMPC with _filtered_output_residual
    still at its post-__init__ default (None) -- call replay_residual_state
    before using it for any tick k > 0.

    `force_schedule`: pass an explicit (S,3,7) schedule array (e.g. a
    frozen-Jacobian run's single matrix tiled across every reference sample)
    to skip the empirical schedule-file match entirely -- needed for the H5
    frozen ablations, whose runs were never driven by one of SCHED_CANDIDATES'
    files. schedule_path is reported as "<forced>" in this case."""
    if "/" in str(run_dir_or_dirname):
        run_dir = run_dir_or_dirname
        dirname = run_dir.rstrip("/").split("/")[-1]
    else:
        dirname = run_dir_or_dirname
        run_dir = f"{manifest.BASE}/{dirname}"
    rm = _find_run_meta(dirname)
    tick_log = load_tick_log(run_dir)

    with open(f"{run_dir}/controller_metadata.json") as f:
        cm = json.load(f)
    if meta is None:
        meta = cm

    if force_schedule is not None:
        schedule_path = "<forced>"
        schedule = np.asarray(force_schedule, dtype=float)
    else:
        contact = bool(cm["contact"])
        schedule_path = _pick_schedule_file(contact, tick_log)
        schedule = np.load(schedule_path)

    reference = load_configuration_reference(cm["plan_dir"], require_planned_beam_feasible=False)

    dt = 1.0 / _CONTROL_HZ
    increment_scale = tuple(
        dt * a for a in ([_JOINT_ACCEL_LIMIT_RAD_S2] * 6 + [10.0 * _INSERTION_RATE_LIMIT_M_S])
    )
    mpc_config_kwargs = dict(
        sample_period_s=dt, prediction_horizon=int(cm["horizon"]),
        state_min=tuple([-2.0 * np.pi] * 6 + [0.0]),
        state_max=tuple([2.0 * np.pi] * 6 + [_INSERTION_MAX_M]),
        velocity_limit=tuple([_JOINT_VELOCITY_LIMIT_RAD_S] * 6 + [cm["insertion_rate_limit_m_s"]]),
        acceleration_limit=tuple([_JOINT_ACCEL_LIMIT_RAD_S2] * 6 + [10.0 * cm["insertion_rate_limit_m_s"]]),
        input_tracking_weight=float(cm["input_tracking_weight"]),
        input_increment_weight=float(cm["input_increment_weight"]),
        input_increment_scale=increment_scale,
        solver_time_limit_s=0.0,
        accept_time_limit_solution=False,
        diagnostic_zero_input_reference_in_R=False,
    )
    mpc_config = ConfigurationMPCConfig(**mpc_config_kwargs)

    s = _POSITION_ERROR_SCALE_MM * 1.0e-3
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(s, s, s), position_tracking_weight=_POSITION_TRACKING_WEIGHT,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )

    mag_jacobians = np.empty((reference.sample_count, 3, 6), dtype=float)
    mag_positions = np.empty((reference.sample_count, 3), dtype=float)
    q_all = np.asarray(reference.state[:, :6], dtype=float)
    for i in range(reference.sample_count):
        mag_positions[i] = loader.magnet_xyz_batch(q_all[i:i + 1])[0]
        mag_jacobians[i] = loader.magnet_position_jacobian(q_all[i])

    magnet_exclusion_kwargs = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        lumen_C_m=np.asarray(cm["magnet_exclusion_point_R"], dtype=float),
        radius_m=float(cm["magnet_exclusion_radius_m"]),
    )
    magnet_workspace_kwargs = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        z_min_m=float(cm["magnet_z_bounds_m"][0]), z_max_m=float(cm["magnet_z_bounds_m"][1]),
    )

    controller = ExactQNZeroTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=int(cm["delay_samples"]), beta_d=float(cm["beta_d"]),
        magnet_exclusion=magnet_exclusion_kwargs, magnet_workspace=magnet_workspace_kwargs,
    )
    return controller, schedule_path, tick_log


def replay_residual_state(controller, tick_log: list[dict], k: int) -> None:
    """Advance controller._filtered_output_residual from its fresh (None)
    state through ticks 0..k-1, using ONLY each tick's own logged
    (z_meas, measured_beam_position, ref_index) -- never a counterfactual
    model, and no QP solve. Idempotent only if called on a fresh controller;
    callers that need multiple k's on one controller should rebuild it."""
    for j in range(k):
        row = tick_log[j]
        z_meas = np.asarray(row["z_meas"], dtype=float)
        beam_pos = np.asarray(row["measured_beam_position_m"], dtype=float)
        controller._estimate_output_residual(
            measured_state=z_meas, measured_beam_position=beam_pos, control_index=row["ref_index"],
        )


def solve_same_state(controller, tick_log: list[dict], k: int, J_override: np.ndarray | None = None):
    """Solve tick k's exact decision again. If J_override is given (a (3,7)
    matrix), every horizon-stage slot of controller.reference_position_
    jacobians for this one solve is set to J_override (constant-over-horizon
    counterfactual, matching the "one model for the whole prediction,
    evaluated where the plant actually is" convention mpc_variants.py's own
    live-relinearizing SQP variant uses) -- then restored to the real
    schedule immediately after, so repeated calls on the same controller for
    different k/J_override never leak state into each other."""
    row = tick_log[k]
    z_meas = np.asarray(row["z_meas"], dtype=float)
    q_cmd = np.asarray(row["q_cmd_k"], dtype=float)
    q_cmd_prev = np.asarray(row["q_cmd_km1"], dtype=float)
    insertion_m = float(row["insertion_cmd_m"])
    beam_pos = np.asarray(row["measured_beam_position_m"], dtype=float)
    control_index = int(row["ref_index"])
    previous_input = np.asarray(row["u_prev"], dtype=float)

    saved = None
    if J_override is not None:
        saved = controller.reference_position_jacobians.copy()
        future_idx = controller._reference_indices(control_index, future=True)
        controller.reference_position_jacobians[future_idx] = np.asarray(J_override, dtype=float)

    try:
        step = controller.solve_delay_aware(
            z_meas=z_meas, q_cmd=q_cmd, q_cmd_prev=q_cmd_prev, insertion_m=insertion_m,
            measured_beam_position=beam_pos, control_index=control_index, previous_input=previous_input,
        )
    finally:
        if saved is not None:
            controller.reference_position_jacobians[:] = saved
    return step


def workspace_margins(magnet_xyz_m: np.ndarray, cm: dict) -> dict:
    """Signed margins (mm, positive = feasible side) for the magnet-exclusion
    radius and z-workspace bound the QP itself enforces (see
    controller_metadata.json's own fields)."""
    base = np.asarray(cm["magnet_exclusion_point_R"], dtype=float)
    d_xy = float(np.linalg.norm(magnet_xyz_m[:2] - base[:2]))
    excl_margin_mm = (d_xy - float(cm["magnet_exclusion_radius_m"])) * 1e3
    z = float(magnet_xyz_m[2])
    z_min, z_max = cm["magnet_z_bounds_m"]
    return dict(
        exclusion_margin_mm=excl_margin_mm,
        z_margin_to_min_mm=(z - z_min) * 1e3,
        z_margin_to_max_mm=(z_max - z) * 1e3,
    )


if __name__ == "__main__":
    probe_runs = [
        "mpc_closedloop_contact_floor210_baseline_v2_20261007T180726Z",      # 210, contact
        "mpc_closedloop_nocontact_2026-10-07_v3_20261007T164416Z",            # 210, no-contact
        "mpc_closedloop_contact_floor255_matched_v1_20261008T114346Z",        # 255, contact (matched)
        "mpc_closedloop_nocontact_floor255_20261007T183458Z",                 # 255, no-contact
    ]
    print(f"{'run':<55} {'tick':>5} {'||u0_logged||':>13} {'||u0_replay-u0_logged||':>24} {'rel_err':>9}")
    worst_rel = 0.0
    for dirname in probe_runs:
        t0 = time.time()
        controller, sched_path, tick_log = build_controller(dirname)
        n = len(tick_log)
        probe_ticks = sorted(set(np.linspace(5, n - 2, 5).astype(int).tolist()))
        for k in probe_ticks:
            replay_residual_state(controller, tick_log, k)
            step = solve_same_state(controller, tick_log, k, J_override=None)
            u0_logged = np.asarray(tick_log[k]["u0"], dtype=float)
            diff = float(np.linalg.norm(step.command - u0_logged))
            scale = max(float(np.linalg.norm(u0_logged)), 1e-9)
            rel = diff / scale
            worst_rel = max(worst_rel, rel)
            print(f"{dirname:<55} {k:>5} {scale:>13.5f} {diff:>24.3e} {rel:>9.3%}")
        print(f"  [{dirname}] schedule={sched_path.split('/')[-1]} built+probed in {time.time()-t0:.1f}s")
    print(f"\nworst relative command error across all probes: {worst_rel:.3%}")
