#!/usr/bin/env python3
"""Live closed-loop MPC for the vessel-navigation model-necessity study
(2026-09-23): the two frozen LTV+d conditions (J_nc vs J_c), exact
Q_N=0, R=700R0.

Adapted from the validated `rectangle_stage_a/run_mpc_delay_aware_
insertion_anchor.py` (same execution-C architecture, same process-
isolated worker, same planner<->live frame-mismatch fix, same
insertion-offset safety monitor -- all copied UNCHANGED, not re-derived)
with three vessel-specific additions:

1. Exact Q_N=0: the worker is spawned with `enable_exact_qn_zero=True`
   (`ExactQNZeroTaskNullspaceDelayAwareMPC`, P_N=P_R=0 at every stage --
   NOT gamma=0, which leaves P_N fully active; see this project's
   `paper/model_necessity_study.tex` Sec. mn-exp4 for why that
   distinction is load-bearing). Verified offline to reproduce the
   study's own DecoupledProjectorMPC(projector=None) bit-for-bit before
   being wired in here.

2. Vessel-aware, contact-toggleable schedule: `--contact`/`--no-contact`
   selects `from_model_bundle(..., contact=True/False)` built from
   `build_vessel_planning_context` (the real digitized lumen), not the
   generic free-space `build_planning_context()` the rectangle runner
   uses. Also raises the beam base/magnet z by 3cm to match the
   physically raised rig, mirroring the offline study's own
   Z_RAISE_M monkeypatch of `make_initial_poses`.

3. Raised vision reference pose: `pf.StateStreamConfig` is monkeypatched
   (same established pattern this package already uses for
   `pf.build_offline_solver`/`pf._make_output_dir`) so the LIVE camera's
   3D reconstruction plane matches the raised rig too -- found live
   2026-09-23 that without this, tip z stays pinned at the old
   (un-raised) height, which is harmless for the open-loop harness's own
   error metric (z is explicitly projected out there) but would corrupt
   the closed-loop MPC's disturbance estimator and Qpbar cost, which see
   the RAW, unprojected 3-vector.

4. Independent magnet-z safety monitor (added 2026-09-23 after a live
   Q_N=0 run drove three joints to their velocity limit simultaneously
   for 1+ second, raising the magnet 12.6cm in 1.4s -- well past this
   project's own documented ~8cm safe-rise envelope -- while
   `close_loop_path_follow.py`'s generic `tcp_out_of_workspace` check
   never fired. Root cause not fully confirmed, but the check reads
   `RobotJointStream.latest_pose()` i.e. `getActualTCPPose()`, which
   depends on the robot controller's OWN configured TCP offset possibly
   not matching this project's magnet-mount model. This monitor does not
   depend on that: it recomputes the magnet position itself every tick
   from the MEASURED joints, via the same verified DH+T_F_M forward-
   kinematics model this project's planning already relies on, and
   aborts (same `info["abort_reason"]` hook as the insertion-offset
   monitor) if the magnet leaves [z_start-floor_margin, z_start+rise_limit]
   -- see `process_isolated_adapter.py`'s `magnet_z_bounds_m`.

5. Planner<->robot frame registration disabled (added 2026-09-23 after a
   second live near-miss the same night): `close_loop_path_follow.py`'s
   `_load_plan_reference` searches plan_dir's ancestry for ANY file named
   `shape_centreline.npz` to compute a rigid registration fit -- a
   mechanism written for free-space shape plans, whose abstract local
   coordinates genuinely need it. It silently found an UNRELATED stale
   file from a different (square-shape) planning run sharing the same
   output tree, and used its un-raised `tip0.z` to compute a spurious
   +30mm z registration offset on top of the vessel reference's already-
   correct target -- the controller was chasing an unreachable target
   30mm above the true beam plane. Bypassed with an exact identity
   transform: the vessel reference is already in robot frame (built via
   real FK throughout `build_vessel_planning_context`), so no
   registration is needed or meaningful here.

6. Independent magnet-exclusion-radius safety monitor (added 2026-09-23,
   same night, third finding): the offline plan's magnet-exclusion-radius
   constraint (minimum distance from the magnet to the vessel lumen,
   set from the same empirically validated closest-safe-approach joint
   state `plan_vessel_path.py` used) is baked into the offline reference
   trajectory but was never enforced ONLINE. With Q_N=0 removing all
   posture anchoring, the redundant joint DOF are free to wander far from
   that reference configuration while still tracking the tip well --
   confirmed live: both the no-contact and contact runs violated the
   exclusion radius by 25-33mm mid-run, undetected by any other monitor.
   Recomputes min-distance(magnet, lumen centreline) every tick from the
   measured joints and aborts if it drops below the validated radius --
   see `process_isolated_adapter.py`'s `magnet_exclusion_radius_m`.

   REWORKED 2026-09-28 (same day as the in-QP wiring below): the
   exclusion point-set is now the fixed BEAM-BASE point (210.43mm, this
   project's originally-established floor -- see
   `vessel_magnet_initial_position_2026-09-27.json` and
   `run_open_loop_vessel.py`'s `BEAM_BASE_PIVOT_XYZ`), not the vessel
   lumen centreline. The vessel-centreline version was itself only a few
   hours old and depended on an empirically-captured "closest-safe-
   approach" reference-joints file that turned out fragile (a near-
   singular Jacobian artifact in its offline schedule, and a frozen-
   Jacobian run that aborted on it at 143/525 ticks); this reverts to the
   simpler, longer-established, already-validated-elsewhere single-point
   floor. `compute_magnet_exclusion` no longer reads a reference-joints
   file or builds the vessel planning context at all -- see its
   docstring. `--magnet-exclusion-reference-joints` is kept as an
   accepted-but-unused CLI flag purely so existing invocations don't
   break; a note is printed if it's passed.

6b. NEW 2026-09-28: magnet z-workspace bounds wired directly into the QP
   (`delay_aware_mpc.py`'s `_configure_magnet_workspace`), the same
   proactive upgrade fix 6's in-QP wiring gave the point-exclusion
   constraint, applied to the existing post-hoc `_MAGNET_Z_BOUNDS_M`
   monitor's own numbers. Closes a blind spot the point-exclusion
   constraint alone does NOT cover: a frozen-Jacobian run swung the arm's
   flange to z=0.262m (~25cm above normal operating height) while every
   QP solve that run reported success -- none of the QP's existing
   constraints (per-joint box, rate limits, point exclusion) bound
   general Cartesian/TCP-workspace excursions, so this was only ever
   catchable after the fact by the external `tcp_out_of_workspace`
   monitor. See `_configure_magnet_workspace`'s docstring.

Everything else is frozen identically to the validated rectangle run:
R=700*R0, original Qp/Rd/u_ref, d=2, beta_d=1, N=15, V_f=0, insertion
authority (s_u,L=5mm/s, |u_L|<=2mm/s), execution-C, process isolation,
insertion-offset safety monitor (5mm default).

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \\
        --plan-dir uprgrade_configuration/.../vessel_lumen/time_parameterized_configuration_path \\
        --lumen-file vessel_lumen_robot_frame_raised3cm_trimmed.json \\
        --insertion-max-mm 55 \\
        --contact \\
        --out-dir close_loop_logs/myrun --schedule-cache /tmp/vessel_c_schedule.npy
"""
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online import controller_adapters
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import common

_real_build_offline_solver = controller_adapters.build_offline_solver
_real_make_output_dir = pf._make_output_dir
_LAST_OUTPUT_DIR: Path | None = None
_METADATA_WRITTEN = False

# --- fix 3: raised vision reference pose (see module docstring) -----------
import proper_research.hardware.online.state_stream as state_stream_mod

_RealStateStreamConfig = state_stream_mod.StateStreamConfig


def _RaisedStateStreamConfig(**kwargs):
    """Drop-in for StateStreamConfig with T_robot_beam_pose6's z raised by
    common.Z_RAISE_M -- called exactly like the real dataclass (positional
    defaults preserved), only the pose default changes.

    2026-09-29: also defaults marker_min_count/marker_max_count to 2 -- the
    physical rig's middle (tangent_start) marker was permanently removed,
    so only base+tip are ever visible now. StateStreamConfig's own
    _tip_and_tangent_start_px already handles the 2-marker case correctly
    (falls back to a zero chord tangent, tip position unaffected) -- see
    its docstring -- and this project's live vessel MPC only tracks tip
    POSITION (n_out=3), never tangent, so losing the tangent is a no-op
    for control. Only the DEFAULT changes (explicit marker_min_count/
    marker_max_count kwargs, if ever passed, still win), so this is
    isolated to this script's own patched StateStreamConfig, not the
    shared class default other (unmodified) rigs elsewhere in this
    project still rely on for a real 3-4-marker tangent."""
    kwargs.setdefault("marker_min_count", 2)
    kwargs.setdefault("marker_max_count", 2)
    cfg = _RealStateStreamConfig(**kwargs)
    if "T_robot_beam_pose6" not in kwargs:
        pose6 = list(cfg.T_robot_beam_pose6)
        pose6[2] += common.Z_RAISE_M
        cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
    return cfg


pf.StateStreamConfig = _RaisedStateStreamConfig

# --- fix 5: disable the planner<->robot frame registration for vessel plans
# (found live 2026-09-23, second incident of the night): `_load_plan_reference`
# calls `_find_shape_npz(plan_dir)`, which searches plan_dir and its parents
# for ANY file named shape_centreline.npz -- a mechanism written for
# plan_shape_path.py's free-space shapes, whose abstract shape-local
# coordinates genuinely need a live rigid-registration fit to robot frame.
# Vessel plans (plan_vessel_path.py) never produce their own such file, so
# this search silently picked up an UNRELATED stale file from a different
# (square-shape) planning run sharing the same output directory tree --
# its tip0 was the OLD, UN-RAISED height, producing a spurious t_fit.z=
# +0.03m applied ON TOP of the vessel reference's already-correct
# (already-raised) desired_position_m. The controller was relentlessly
# chasing an unreachable target 30mm above the true beam plane -- this
# alone plausibly explains the aggressive joint motion seen in both
# no-contact runs tonight, independent of any real Jacobian mismatch.
# The vessel plan's desired_position_m is ALREADY in robot frame (built via
# real FK throughout build_vessel_planning_context) -- no registration is
# needed or meaningful here, so this bypasses it with an exact identity
# transform rather than trying to find/build a correct vessel-specific
# equivalent of shape_centreline.npz.
def _identity_fit_planner_to_robot(npz_path, start_tip_R, T_R_B):
    return np.eye(3), np.zeros(3)


pf._fit_planner_to_robot = _identity_fit_planner_to_robot

# _load_plan_reference still calls _find_shape_npz(plan_dir) BEFORE
# _fit_planner_to_robot and raises FileNotFoundError if nothing turns up --
# harmless today only because a stale file happens to exist nearby. Make
# that not load-bearing: always resolve to this file itself (guaranteed to
# exist), since _fit_planner_to_robot above never reads its contents.
pf._find_shape_npz = lambda plan_dir: Path(__file__)

# --- fix 4: independent magnet-z safety monitor (see module docstring) ----
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)
from proper_research.planning.planning_context import make_robot_config

_MAGNET_RISE_LIMIT_M = 0.04  # tightened 2026-09-23 after the live near-miss (was 0.08)
# NOT a bare JointSpaceRobotConfig() -- its tcp_to_magnet_pose6 default
# (None) falls back to ur_magnet_ik_jacobian_validation.CONFIG's STALE
# 0.47m tool offset (see that class's own docstring warning). Caught live
# 2026-09-23: a bare-default version of this exact check disagreed with
# the already-validated adapter_c.magnet_transform by >100mm on all three
# axes for the same joints -- a safety check computing the WRONG magnet
# position is worse than no check. make_robot_config() is the SAME
# factory build_vessel_planning_context/build_controller use, with the
# correct tcp_to_magnet_pose6 explicitly set.
_robot_kin = _resolve_robot_kinematics(make_robot_config())


def _magnet_transform_fn(q6: np.ndarray) -> np.ndarray:
    """Magnet xyz from measured joints -- the SAME verified DH+T_F_M model
    used throughout this project's planning/offline work, independent of
    the robot controller's own configured TCP offset (see module
    docstring's fix 4)."""
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)

# Frozen controller spec -- module-level constants, not CLI flags, so this
# run cannot silently drift from the offline-validated formulation.
_GAMMA = 0.0  # unused by ExactQNZeroTaskNullspaceDelayAwareMPC; kept for metadata clarity
_INPUT_TRACKING_WEIGHT = 7.0          # 700 * default 1.0e-2 = R700
_INPUT_INCREMENT_WEIGHT = 1.0e-3
_INSERTION_RATE_LIMIT_M_S = 2.0e-3

_DEADLINE_MS = 70.0
_INSERTION_OFFSET_ABORT_M = 0.005  # 5mm, matches the validated rectangle default
_WORKER: MPCWorkerHandle | None = None
_CONTACT: bool = True
_MAGNET_Z_BOUNDS_M: tuple | None = None  # set in main() from the plan's own start pose
_MAGNET_EXCLUSION_LUMEN_C_M = None  # set in main(); single beam-base point for the exclusion check
_MAGNET_EXCLUSION_RADIUS_M: float | None = None  # set in main()

# --- fix 6 (reworked 2026-09-28): magnet-to-beam-base exclusion, fixed
# 210.43mm floor -- same point/radius `run_open_loop_vessel.py`'s
# BEAM_BASE_PIVOT_XYZ / capture_live_start_position.py's
# BEAM_BASE_XYZ_RAISED30MM already use elsewhere in this package (raised
# +30mm frame, matching this script's own zraise_patch.apply(30.0)
# convention). See module docstring's fix 6 for why this replaced the
# vessel-centreline version.
# 2026-09-29 fix: this used to hardcode the +30mm raised-workspace beam-
# base point (z=0.013433), regardless of --z-raise-mm (which didn't exist
# yet) -- silently checking the magnet-exclusion/z-workspace QP constraints
# against a raised point even for an unraised plan, a 30mm safety-margin
# error. Now recomputed in main() from common.Z_RAISE_M after --z-raise-mm
# is applied (see build_vessel_plan.py / run_open_loop_vessel.py's own
# 2026-09-29 fixes for the same bug elsewhere in this package).
_BEAM_BASE_PIVOT_XY_ROT = np.array([0.525575, -0.670028])
_BEAM_BASE_PIVOT_Z_UNRAISED = -0.016567
_BEAM_BASE_PIVOT_XYZ_R = np.array([[0.525575, -0.670028, 0.013433]])  # overwritten in main()
_BEAM_BASE_EXCLUSION_RADIUS_M = 0.21043  # overwritten in main() from --beam-base-exclusion-floor-mm
_MAGNET_CONSTRAINTS_IN_QP: bool = False  # set in main(); True unless --disable-magnet-exclusion-in-qp


def _wrapped_make_output_dir(cfg):
    global _LAST_OUTPUT_DIR
    out = _real_make_output_dir(cfg)
    _LAST_OUTPUT_DIR = out
    return out


pf._make_output_dir = _wrapped_make_output_dir


def build_magnet_exclusion_schedule(reference) -> tuple[np.ndarray, np.ndarray]:
    """Analytic magnet-position Jacobian + nominal magnet position at every
    reference sample -- the schedule the in-QP magnet-exclusion constraint
    linearizes about (see `delay_aware_mpc.py`'s `_configure_magnet_
    exclusion` docstring). Same `_robot_kin`/`geometric_jacobian` this
    module already uses for `_magnet_transform_fn`, just evaluated over the
    whole reference trajectory instead of one live pose -- position Jacobian
    only (rows 0:3 of the 6x6 spatial Jacobian), since the magnet is rigidly
    on the end effector and its position never depends on insertion L."""
    q = np.asarray(reference.state[:, :6], dtype=float)
    jacobians = np.empty((q.shape[0], 3, 6), dtype=float)
    positions = np.empty((q.shape[0], 3), dtype=float)
    for i in range(q.shape[0]):
        fk = urik.forward_kinematics(q[i], _robot_kin.dh, _robot_kin.T_F_M)
        positions[i] = fk.T_R_target[:3, 3]
        jacobians[i] = urik.geometric_jacobian(q[i], _robot_kin.dh, _robot_kin.T_F_M)[:3, :]
    return jacobians, positions


def spawn_and_warm_worker(
    *, plan_dir: str, schedule_cache: str, control_hz: float, prediction_horizon: int,
    joint_velocity_limit_rad_s: float, insertion_rate_limit_m_s: float,
    joint_acceleration_limit_rad_s2: float, position_error_scale_mm: float,
    position_tracking_weight: float, insertion_max_m: float,
    magnet_exclusion_kwargs: dict | None = None,
    magnet_workspace_kwargs: dict | None = None,
    solver_time_limit_s: float = 0.0,
) -> MPCWorkerHandle:
    """Spawn + warm the MPC worker BEFORE any RTDE/camera connection opens."""
    dt = 1.0 / control_hz
    increment_scale = tuple(
        dt * a for a in ([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s])
    )
    exact_qn0_config_kwargs = dict(
        sample_period_s=dt, prediction_horizon=int(prediction_horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [0.0]), state_max=tuple([2.0 * math.pi] * 6 + [insertion_max_m]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [insertion_rate_limit_m_s]),
        acceleration_limit=tuple([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s]),
        input_tracking_weight=_INPUT_TRACKING_WEIGHT,
        input_increment_weight=_INPUT_INCREMENT_WEIGHT,
        input_increment_scale=increment_scale,
        # 2026-09-29 fix: this script never set this before, so OSQP's own
        # `time_limit` stayed at the library default (disabled) regardless
        # of how many iterations an ill-conditioned tick needed -- measured
        # live spikes to 1300+ iterations / 64ms+ solve time against a
        # typical ~120-150 iterations / ~8-20ms, occasionally exceeding the
        # 70ms tick deadline outright. OSQP's ADMM iterate is a valid
        # (if less-converged) solution at any point, so bounding solve
        # time directly -- rather than trying to eliminate the underlying
        # ill-conditioning -- trades a possibly-slightly-suboptimal command
        # on a rare hard tick for NEVER missing the deadline on that tick's
        # account. Confirmed the actual IPC/logging overhead added
        # 2026-09-28 is NOT the cause (measured directly: ~0.02ms, 3
        # orders of magnitude below the deadline) before adding this.
        solver_time_limit_s=float(solver_time_limit_s),
        # See ConfigurationMPCConfig.accept_time_limit_solution's docstring
        # (2026-09-29 fix): without this, setting solver_time_limit_s only
        # made time-limited solves get classified as outright failures
        # faster, not usable ones -- defeating the whole point. Tied
        # directly to solver_time_limit_s being on, since there's no
        # reason to set one without the other for this script.
        accept_time_limit_solution=float(solver_time_limit_s) > 0.0,
    )
    s = float(position_error_scale_mm) * 1.0e-3
    beam_config_kwargs = dict(
        position_error_scale_m=(s, s, s), position_tracking_weight=float(position_tracking_weight),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    print("[vessel-mpc] spawning + warming process-isolated MPC worker "
          "(BEFORE any hardware connection); building exact-Q_N=0/R700...")
    worker = MPCWorkerHandle(
        plan_dir=plan_dir, schedule_path=schedule_cache,
        mpc_config_kwargs=exact_qn0_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_exact_qn_zero=True, exact_qn_zero_config_kwargs=exact_qn0_config_kwargs,
        magnet_exclusion_kwargs=magnet_exclusion_kwargs,
        magnet_workspace_kwargs=magnet_workspace_kwargs,
    )
    spawn_ms = (worker.t_worker_ready - worker.t_spawn_start) * 1e3
    print(f"[vessel-mpc] worker ready (spawn+import+construct+warm-up+checks = {spawn_ms:.0f}ms)")
    print(f"[vessel-mpc] CONTROLLER CONFIRMED: exact Q_N=0 (P_N=P_R=0), R700 "
          f"(input_tracking_weight={_INPUT_TRACKING_WEIGHT}), contact={_CONTACT}, "
          f"safety abort at |L-Lref|>{_INSERTION_OFFSET_ABORT_M*1e3:.1f}mm, "
          f"magnet-exclusion-in-qp={'ON' if magnet_exclusion_kwargs is not None else 'OFF'}, "
          f"magnet-workspace-in-qp={'ON' if magnet_workspace_kwargs is not None else 'OFF'}")
    worker.require_frame_transform()
    print("[vessel-mpc] worker will refuse solves until set_frame_transform() is called "
          "(happens once preflight computes the live registration)")
    return worker


def _wrapped_build_offline_solver(kind, **kwargs):
    global _METADATA_WRITTEN
    if kind != "mpc_delay_aware":
        return _real_build_offline_solver(kind, **kwargs)
    assert _LAST_OUTPUT_DIR is not None, "output dir not yet created"
    assert _WORKER is not None, "worker must be spawned+warmed BEFORE pf.main() -- see main()"
    cfg = pf.CONFIG

    transform = pf._PLANNER_TO_LIVE_TRANSFORM
    if transform is None:
        raise RuntimeError(
            "pf._PLANNER_TO_LIVE_TRANSFORM is not set -- _load_plan_reference() must run "
            "before build_offline_solver() is called."
        )
    r_fit, t_fit = transform

    if not _METADATA_WRITTEN:
        meta = {
            "controller_variant": "exact_qn0_R700_vessel",
            "contact": _CONTACT,
            "gamma": _GAMMA,
            "input_tracking_weight": _INPUT_TRACKING_WEIGHT,
            "input_increment_weight": _INPUT_INCREMENT_WEIGHT,
            "insertion_offset_abort_m": _INSERTION_OFFSET_ABORT_M,
            "state_scale_q_rad": math.radians(0.5),
            "state_scale_L_m": 0.25e-3,
            "input_scale_q_rad_s": 0.05,
            "input_scale_L_m_s": 5.0e-3,
            "delay_samples": 2,
            "beta_d": 1.0,
            "horizon": int(cfg.mpc_prediction_horizon),
            "terminal_cost": False,
            "insertion_rate_limit_m_s": _INSERTION_RATE_LIMIT_M_S,
            "plan_dir": str(cfg.plan_dir),
            "z_raise_m": common.Z_RAISE_M,
            "magnet_z_bounds_m": list(_MAGNET_Z_BOUNDS_M) if _MAGNET_Z_BOUNDS_M else None,
            "magnet_exclusion_point_R": _MAGNET_EXCLUSION_LUMEN_C_M[0].tolist()
                if _MAGNET_EXCLUSION_LUMEN_C_M is not None else None,
            "magnet_exclusion_radius_m": _MAGNET_EXCLUSION_RADIUS_M,
            "magnet_exclusion_source": f"beam_base_fixed_{_BEAM_BASE_EXCLUSION_RADIUS_M*1e3:.2f}mm",
            "magnet_constraints_in_qp": _MAGNET_CONSTRAINTS_IN_QP,
            "planner_to_live_R_fit": r_fit.tolist(),
            "planner_to_live_t_fit_m": t_fit.tolist(),
            "planner_to_live_t_fit_norm_m": float(np.linalg.norm(t_fit)),
        }
        (_LAST_OUTPUT_DIR / "controller_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[vessel-mpc] wrote controller_metadata.json -> {_LAST_OUTPUT_DIR}")
        _METADATA_WRITTEN = True

    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=kwargs["reference"], worker=_WORKER, deadline_s=_DEADLINE_MS / 1e3,
        joint_velocity_limit_rad_s=float(cfg.joint_velocity_limit_rad_s),
        max_joint_step_rad=float(cfg.max_joint_step_rad),
        config=kwargs.get("adapter_config"),
        prediction_log_path=str(_LAST_OUTPUT_DIR / "predicted_beam_positions.jsonl"),
        controller_kind="exact_qn0",
        controller_label=f"mpc_delay_aware_exact_qn0_vessel_{'contact' if _CONTACT else 'nocontact'}",
        insertion_offset_abort_m=_INSERTION_OFFSET_ABORT_M,
        magnet_transform_fn=_magnet_transform_fn,
        magnet_z_bounds_m=_MAGNET_Z_BOUNDS_M,
        magnet_exclusion_lumen_C_m=_MAGNET_EXCLUSION_LUMEN_C_M,
        magnet_exclusion_radius_m=_MAGNET_EXCLUSION_RADIUS_M,
    )
    _WORKER.set_frame_transform(r_fit, t_fit)
    print(f"[vessel-mpc] worker frame transform applied: "
          f"|t_fit|={np.linalg.norm(t_fit)*1e3:.1f}mm det(R_fit)={np.linalg.det(r_fit):+.3f}")

    p_des_P = np.asarray(
        load_configuration_reference(cfg.plan_dir, require_planned_beam_feasible=False).desired_position_m,
        dtype=float,
    )
    p_des_R = p_des_P @ r_fit.T + t_fit
    p_des_harness = np.asarray(kwargs["reference"].desired_position_m, dtype=float)
    discrepancy = float(np.max(np.linalg.norm(p_des_R - p_des_harness, axis=1)))
    print(f"[vessel-mpc] frame-fix check: max_i||R_fit@p_des_i^P+t_fit - p_des_i^harness|| "
          f"= {discrepancy*1e3:.6f}mm (should be numerical noise)")
    assert discrepancy < 1e-6, (
        f"worker/harness target discrepancy after frame fix is {discrepancy*1e3:.3f}mm, "
        f"not numerical noise -- refusing to proceed live"
    )
    return adapter


pf.build_offline_solver = _wrapped_build_offline_solver


# Captured ONCE, at import time, BEFORE any patching -- calling this wrapper
# must never re-resolve `initial_conditions_mod.make_initial_poses` by name,
# since that attribute gets reassigned to this very wrapper below (a lazy
# re-import inside the wrapper recurses into itself infinitely; hit and
# confirmed live 2026-09-23, same pitfall as this project's earlier offline
# vessel-planning monkeypatches -- see vessel-planning-raised-base memory).
import proper_research.simulation.simulations.initial_conditions as _initial_conditions_mod
import proper_research.planning.planning_context as _planning_context_mod

_ORIG_MAKE_INITIAL_POSES = _initial_conditions_mod.make_initial_poses


def _raised_make_initial_poses():
    p, s, L, dt = _ORIG_MAKE_INITIAL_POSES()
    p = np.array(p, dtype=float).copy()
    s = np.array(s, dtype=float).copy()
    p[2] += common.Z_RAISE_M
    s[2] += common.Z_RAISE_M
    return p, s, L, dt


def compute_magnet_exclusion(*, reference_joints_path: str | None = None):
    """REWORKED 2026-09-28 (see module docstring's fix 6): the magnet
    exclusion is now a FIXED single point/radius -- the beam base,
    210.43mm -- not a computation against the vessel lumen or the
    empirically-captured reference-joints file. No longer builds the
    vessel planning context or does any FK/IK work; this is now a pure
    constant lookup, kept as a function (rather than inlined at the call
    site) only so `main()`'s call site and log message stay unchanged in
    shape.

    `reference_joints_path`: accepted-but-UNUSED, purely so existing
    `--magnet-exclusion-reference-joints <file>` invocations (this flag
    is still accepted by argparse, see `main()`) don't need to change --
    a note is printed if a path is actually passed, so it's obvious this
    is now a no-op rather than silently ignored."""
    if reference_joints_path is not None:
        print(f"[vessel-mpc] NOTE: --magnet-exclusion-reference-joints "
              f"({reference_joints_path}) is no longer used -- the magnet exclusion "
              f"is now a fixed beam-base point/radius, not derived from this file "
              f"(see module docstring's fix 6)")
    print(f"[vessel-mpc] magnet exclusion: beam-base point "
          f"{_BEAM_BASE_PIVOT_XYZ_R[0].tolist()} -> {_BEAM_BASE_EXCLUSION_RADIUS_M*1e3:.2f}mm "
          f"(fixed floor, not vessel-centreline-derived)")
    return _BEAM_BASE_PIVOT_XYZ_R.copy(), _BEAM_BASE_EXCLUSION_RADIUS_M


def build_or_load_schedule(
    plan_dir: str, cache_path: str, *, lumen_file: str, insertion_max_mm: float, contact: bool,
) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[vessel-mpc] loading cached schedule from {cache_path}")
        return np.load(cache_path)
    print(f"[vessel-mpc] building genuine from_model_bundle Jacobian schedule "
          f"(contact={contact}, ~90-150s)...")

    _initial_conditions_mod.make_initial_poses = _raised_make_initial_poses
    _planning_context_mod.make_initial_poses = _raised_make_initial_poses

    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=lumen_file, insertion_max_m=insertion_max_mm * 1.0e-3,
    )
    # jacobian_mode="accurate", NOT the default "fast": "fast" mode was
    # found (2026-09-23, this same vessel plan) to have a single-tick
    # numerical estimator artifact -- a spurious singular-value spike
    # (~79 vs ~1.0-1.1 at every neighboring tick) while the true tip
    # output changes smoothly there -- confirmed by direct comparison
    # against "accurate" mode at the same state. See
    # paper/model_necessity_study.tex Sec. mn-exp3's methodological note.
    jac_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack, contact=contact, jacobian_mode="accurate",
    )
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[vessel-mpc] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    global _DEADLINE_MS, _WORKER, _INSERTION_OFFSET_ABORT_M, _CONTACT, _MAGNET_Z_BOUNDS_M
    global _MAGNET_EXCLUSION_LUMEN_C_M, _MAGNET_EXCLUSION_RADIUS_M, _MAGNET_CONSTRAINTS_IN_QP
    global _BEAM_BASE_PIVOT_XYZ_R, _BEAM_BASE_EXCLUSION_RADIUS_M
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--z-raise-mm", type=float, default=0.0,
                    help="rigid z-shift applied to the beam-base pivot, the offline-"
                         "planner pivot/start-point, AND the live vision reconstruction "
                         "plane -- 0.0 (default) for an unraised/original-height "
                         "workspace, 30.0 to reproduce the 2026-09-27/28 raised-"
                         "workspace runs. MUST match whether --lumen-file (and the plan "
                         "it built) was itself built raised or unraised -- a mismatch "
                         "here silently misreads real camera detections AND checks the "
                         "magnet-exclusion/z-workspace QP safety constraints against the "
                         "wrong beam-base point, by the z-raise amount.")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="mpc_delay_aware_vessel_accumC")
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--insertion-max-mm", type=float, required=True)
    p.add_argument("--magnet-exclusion-reference-joints", default=None,
                    help="DEPRECATED/UNUSED as of 2026-09-28 -- the magnet exclusion is now a "
                         "fixed beam-base point/210.43mm radius, not derived from this file. "
                         "Still accepted (no longer required) purely so existing invocations "
                         "don't break; a note is printed if passed. See module docstring's fix 6.")
    p.add_argument("--beam-base-exclusion-floor-mm", type=float, default=210.43,
                    help="the TRUE magnet-to-beam-base exclusion floor (before "
                         "--magnet-exclusion-tolerance-mm is subtracted to get the live "
                         "abort/QP threshold). Default 210.43mm matches the project's "
                         "established value; 2026-09-29 made overridable after a v3-vessel "
                         "centreline-tracking run aborted at gap=205.4mm < 205.4mm -- i.e. "
                         "crossed the live threshold by a hair, and post-hoc wall-clearance "
                         "analysis of that run showed zero actual lumen-wall penetration, "
                         "consistent with the shortfall being linearization/tracking-lag "
                         "error (see delay_aware_mpc.py's _configure_magnet_exclusion "
                         "docstring) rather than a real large safety breach.")
    contact_group = p.add_mutually_exclusive_group(required=True)
    contact_group.add_argument("--contact", dest="contact", action="store_true")
    contact_group.add_argument("--no-contact", dest="contact", action="store_false")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--deadline-ms", type=float, default=70.0)
    p.add_argument("--solver-time-limit-s", type=float, default=0.03,
                    help="OSQP's own wall-clock cutoff per solve (0 = disabled/OSQP "
                         "default, the behaviour every run before 2026-09-29). At an "
                         "ill-conditioned tick, OSQP's ADMM iterations can spike into "
                         "the thousands, occasionally exceeding --deadline-ms outright "
                         "(measured: 2000+ iterations, 50-58ms solve, against a "
                         "typical ~150-300/~7-10ms). OSQP's iterate is a valid, if "
                         "less-converged, solution at any point, so bounding solve "
                         "time directly is safe -- it trades solution quality on a "
                         "rare hard tick for guaranteeing the deadline is never missed "
                         "on that tick's account. Default 0.03s (NOT 0.05 -- measured "
                         "live 2026-09-29 that non-OSQP per-tick overhead alone can "
                         "spike to ~28ms, so 0.05+0.028=0.078s exceeded the 70ms "
                         "deadline even with the cap active; 0.03s leaves real margin "
                         "against that overhead, not just against --deadline-ms.")
    p.add_argument("--insertion-offset-abort-mm", type=float, default=5.0)
    p.add_argument("--insertion-tol-mm", type=float, default=3.0)
    p.add_argument("--magnet-rise-limit-mm", type=float, default=_MAGNET_RISE_LIMIT_M * 1e3,
                    help="independent safety monitor: abort if the magnet's own FK-computed z "
                         "rises more than this above its start position -- see module "
                         "docstring's fix 4 for why this exists")
    p.add_argument("--magnet-floor-margin-mm", type=float, default=40.0,
                    help="independent safety monitor: abort if the magnet's own FK-computed z "
                         "drops more than this below its start position (symmetric with "
                         "--magnet-rise-limit-mm by default, 2026-09-23: normal small "
                         "oscillation legitimately spans ~5mm around start, so a tight floor "
                         "produced spurious trips on otherwise clean runs)")
    p.add_argument("--magnet-exclusion-tolerance-mm", type=float, default=5.0,
                    help="independent safety monitor: subtract this from the computed "
                         "magnet-exclusion radius before using it as the live abort threshold "
                         "-- 2026-09-23: the known-safe open-loop baseline legitimately dips to "
                         "~1.5mm inside the raw computed radius (solver/FK discretization, not "
                         "a real violation), while the two actual incidents were 25-37mm "
                         "violations -- comfortable margin for a small tolerance here")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--disable-magnet-exclusion-in-qp", action="store_true",
                    help="disable BOTH linearized magnet safety constraints wired directly into "
                         "the QP (2026-09-28): the magnet-to-beam-base point exclusion and the "
                         "magnet z-workspace bounds. The independent post-hoc measured-joint "
                         "safety monitors (--magnet-exclusion-tolerance-mm, --magnet-rise-limit-mm/"
                         "--magnet-floor-margin-mm) stay on regardless of this flag; see "
                         "delay_aware_mpc.py's _configure_magnet_exclusion/_configure_magnet_"
                         "workspace docstrings for why these were added")
    args = p.parse_args()

    _BEAM_BASE_EXCLUSION_RADIUS_M = args.beam_base_exclusion_floor_mm / 1000.0
    common.Z_RAISE_M = args.z_raise_mm / 1000.0
    _BEAM_BASE_PIVOT_XYZ_R = np.array([[
        _BEAM_BASE_PIVOT_XY_ROT[0], _BEAM_BASE_PIVOT_XY_ROT[1],
        _BEAM_BASE_PIVOT_Z_UNRAISED + common.Z_RAISE_M,
    ]])
    print(f"[vessel-mpc] z_raise_mm = {args.z_raise_mm} "
          f"beam_base_pivot_xyz = {_BEAM_BASE_PIVOT_XYZ_R[0].tolist()}")

    _DEADLINE_MS = args.deadline_ms
    _INSERTION_OFFSET_ABORT_M = args.insertion_offset_abort_mm * 1e-3
    _CONTACT = bool(args.contact)

    schedule = build_or_load_schedule(
        args.plan_dir, args.schedule_cache, lumen_file=args.lumen_file,
        insertion_max_mm=args.insertion_max_mm, contact=_CONTACT,
    )
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[vessel-mpc] schedule: {schedule.shape} contact={_CONTACT} "
          f"d=2, beta_d=1.0, V_f=0, N={args.horizon}, process-isolated, "
          f"deadline={args.deadline_ms:.0f}ms, insertion-offset-abort={args.insertion_offset_abort_mm:.1f}mm")

    _MAGNET_EXCLUSION_LUMEN_C_M, exclusion_radius_true_m = compute_magnet_exclusion(
        reference_joints_path=args.magnet_exclusion_reference_joints,
    )
    _MAGNET_EXCLUSION_RADIUS_M = exclusion_radius_true_m - args.magnet_exclusion_tolerance_mm * 1e-3
    print(f"[vessel-mpc] magnet exclusion live abort threshold: {_MAGNET_EXCLUSION_RADIUS_M*1e3:.1f}mm "
          f"(true radius {exclusion_radius_true_m*1e3:.1f}mm - "
          f"{args.magnet_exclusion_tolerance_mm:.1f}mm tolerance)")

    # Loaded early (cheap, no motion -- same file `preflight()` itself reads
    # first) so the magnet-z bounds are known BEFORE any reset motion runs,
    # letting the reset itself be checked against them (and the exclusion
    # radius, already computed above) rather than only guarding the
    # closed-loop run that follows it.
    q0, l0 = common.load_plan_initial_state(args.plan_dir)
    magnet_z_start = float(_magnet_transform_fn(q0)[2])
    magnet_rise_limit_m = args.magnet_rise_limit_mm * 1e-3
    # Floor margin: originally a tight 5mm (matching "may never go lower"
    # read literally), which produced two false/near-false positives live
    # 2026-09-23 -- a zero-reset-noise trip on tick 0, and later a clean
    # 44-tick run (tracking error 0.12-1.36mm, no instability) tripped by
    # ordinary ~5mm oscillation around start. Widened to match the rise
    # limit (symmetric by default) after confirming via that same clean
    # run that normal operation legitimately spans close to 5mm either
    # side of start -- a tighter floor was not adding real protection,
    # only false trips on otherwise-good runs.
    _MAGNET_Z_FLOOR_MARGIN_M = args.magnet_floor_margin_mm * 1e-3
    _MAGNET_Z_BOUNDS_M = (
        magnet_z_start - _MAGNET_Z_FLOOR_MARGIN_M, magnet_z_start + magnet_rise_limit_m,
    )
    print(f"[vessel-mpc] magnet-z safety monitor: start_z={magnet_z_start*1e3:.1f}mm "
          f"bounds=[{_MAGNET_Z_BOUNDS_M[0]*1e3:.1f},{_MAGNET_Z_BOUNDS_M[1]*1e3:.1f}]mm "
          f"(rise_limit={args.magnet_rise_limit_mm:.1f}mm, floor_margin="
          f"{_MAGNET_Z_FLOOR_MARGIN_M*1e3:.1f}mm)")

    magnet_exclusion_kwargs = None
    magnet_workspace_kwargs = None
    _MAGNET_CONSTRAINTS_IN_QP = not args.disable_magnet_exclusion_in_qp
    if not args.disable_magnet_exclusion_in_qp:
        # Same schedule (magnet position + its position-Jacobian at every
        # reference sample) feeds BOTH new in-QP constraints -- one real
        # FK/Jacobian pass over the reference trajectory, computed once,
        # not duplicated per constraint.
        _reference_for_schedule = load_configuration_reference(
            args.plan_dir, require_planned_beam_feasible=False,
        )
        mag_jacobians, mag_positions = build_magnet_exclusion_schedule(_reference_for_schedule)
        magnet_exclusion_kwargs = dict(
            position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
            lumen_C_m=_MAGNET_EXCLUSION_LUMEN_C_M, radius_m=_MAGNET_EXCLUSION_RADIUS_M,
        )
        print(f"[vessel-mpc] magnet-exclusion constraint will be wired INTO the QP "
              f"(radius={_MAGNET_EXCLUSION_RADIUS_M*1e3:.1f}mm, same radius as the post-hoc "
              f"monitor) -- pass --disable-magnet-exclusion-in-qp to turn this off")
        magnet_workspace_kwargs = dict(
            position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
            z_min_m=_MAGNET_Z_BOUNDS_M[0], z_max_m=_MAGNET_Z_BOUNDS_M[1],
        )
        print(f"[vessel-mpc] magnet z-workspace constraint will be wired INTO the QP "
              f"(z in [{_MAGNET_Z_BOUNDS_M[0]*1e3:.1f},{_MAGNET_Z_BOUNDS_M[1]*1e3:.1f}]mm, "
              f"same bounds as the post-hoc monitor) -- pass --disable-magnet-exclusion-in-qp "
              f"to turn this off too")
    else:
        print("[vessel-mpc] --disable-magnet-exclusion-in-qp set: magnet exclusion and "
              "z-workspace bounds stay ONLY post-hoc monitors, not visible to the optimizer")

    _WORKER = spawn_and_warm_worker(
        plan_dir=args.plan_dir, schedule_cache=args.schedule_cache,
        control_hz=common.CONTROL_HZ, prediction_horizon=args.horizon,
        joint_velocity_limit_rad_s=args.joint_velocity_limit_rad_s,
        insertion_rate_limit_m_s=_INSERTION_RATE_LIMIT_M_S,
        joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
        insertion_max_m=args.insertion_max_mm * 1.0e-3,
        magnet_exclusion_kwargs=magnet_exclusion_kwargs,
        magnet_workspace_kwargs=magnet_workspace_kwargs,
        solver_time_limit_s=args.solver_time_limit_s,
    )

    if args.skip_preflight:
        pass  # q0, l0 already loaded above
    else:
        # The reset-to-start move_j itself is now checked: the straight-line
        # joint-space path it will take is sampled and refused (before any
        # motion starts) if it would cross the magnet-exclusion radius or
        # magnet-z bounds -- move_j has no real-time interruption once
        # started, and a joint-space-linear path between two safe endpoints
        # is not guaranteed to stay safe in between. See
        # common.reset_to_plan_initial_safe / _validate_reset_path_safe.
        q0, l0 = common.preflight(
            args.plan_dir, insertion_tol_mm=args.insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=_MAGNET_EXCLUSION_LUMEN_C_M,
            magnet_exclusion_radius_m=_MAGNET_EXCLUSION_RADIUS_M,
            magnet_z_bounds_m=_MAGNET_Z_BOUNDS_M,
        )

    cfg = pf.CONFIG
    cfg.controller_kind = "mpc_delay_aware"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"
    cfg.mpc_prediction_horizon = args.horizon
    cfg.mpc_use_dare_terminal_cost = False

    cfg.feedforward_joint_trajectory = False
    cfg.accumulator_seam = True
    cfg.servo_stream_hz = args.servo_stream_hz
    cfg.force_mpc_feedforward = False

    cfg.joint_velocity_limit_rad_s = args.joint_velocity_limit_rad_s
    cfg.max_joint_step_rad = args.max_joint_step_rad
    cfg.joint_acceleration_limit_rad_s2 = common.JOINT_ACCELERATION_LIMIT_RAD_S2
    cfg.control_insertion = True
    cfg.control_hz = common.CONTROL_HZ
    cfg.max_control_steps = args.max_control_steps
    cfg.initial_insertion_m = l0

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name
    cfg.dry_run = args.dry_run

    print(
        f"[vessel-mpc] plan={args.plan_dir}\n"
        f"[vessel-mpc] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[vessel-mpc] horizon={cfg.mpc_prediction_horizon} d=2 beta_d=1.0 V_f=0 "
        f"servo_stream_hz={cfg.servo_stream_hz}\n"
        f"[vessel-mpc] CONTROLLER = exact Q_N=0, R700, contact={_CONTACT}, "
        f"z_raise={common.Z_RAISE_M*1e3:.0f}mm, "
        f"safety abort |L-Lref|>{args.insertion_offset_abort_mm:.1f}mm"
    )
    try:
        pf.main()
    finally:
        _WORKER.close()


if __name__ == "__main__":
    main()
