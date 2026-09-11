"""Trace a square / triangle tip path online with an offline joint-space controller.

Same servoJ + advancer closed loop as :mod:`close_loop_tip_control`, but the
target is a *moving* timed reference instead of one fixed point, driven through
``build_offline_solver`` (the exact seam the simulation comparison uses).

Two ways to get the timed reference
----------------------------------
``reference_source = "plan_dir"``
    Load a real offline-planner output -- the directory written by

        run_inverse_head_exclusion.py
          -> run_global_configuration.py
          -> run_time_parameterization.py

    via ``load_configuration_reference``.  This is the "all the framework wired
    together" path: geometry -> inverse configuration -> global smoothing ->
    time parameterisation -> online tracking.  The planner works in its own
    model/world frame, so the reference's ``desired_position_m`` must already be
    in the hardware robot-base frame for the tracking error to mean anything --
    verify that once, supervised, before trusting a run.

``reference_source = "synthetic"``  (default)
    Build the shape directly in the hardware robot-base frame, in the beam
    bending plane (B.y sideways via the joints, B.x axial via the advancer),
    centred on the *current* tip.  No planner, no frame ambiguity -- runnable
    immediately, and it exercises exactly the same online stack.

Recording
---------
* ``path_follow.jsonl``      -- one JSON object per control tick
* ``tip_trajectory.csv``     -- flat table: tip, reference, error, q, insertion
* ``path_follow_plot.png``   -- tip vs reference in the shape plane + error(t)

    python -m proper_research.hardware.online.close_loop_path_follow
"""

from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import asdict, dataclass, replace as replace_dc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.close_loop_tip_control import (
    AnalyticalBeamJacobianProvider,
    SimpleBeamJacobianProvider,
    _make_output_dir,
    _mpc_config,
    _translational_jacobian_fd,
)
from proper_research.hardware.online.controller_adapters import (
    OfflineControllerConfig,
    build_offline_solver,
)
from proper_research.hardware.online.messages import now_monotonic
from proper_research.hardware.online.state_stream import (
    NewFrameTipMapper,
    RobotJointStream,
    StateStreamConfig,
    _DebugLineFilter,
)


# =============================================================================
# USER CONFIGURATION -- edit here; there are no terminal arguments
# =============================================================================


@dataclass
class PathFollowConfig:
    dry_run: bool = False

    # --- where the timed reference comes from ---------------------
    reference_source: str = "plan_dir"      # "synthetic" | "plan_dir"
    # 2026-09-10: feasible closed 8 mm apex-at-start triangle from
    # plan_shape_path.py with the calibrated model (E=2.5 MPa, N52 900 A m^2).
    # Beam starts at 30 mm (== initial_conditions L0); insertion 30-39 mm.
    # (plans/square_2mm_2026-09-10/ is the smaller square alternative.)
    plan_dir: str = "plans/triangle_8mm_2026-09-10/time_parameterized_configuration_path"
    # The offline planner runs in its own world frame (legacy pivot, beam
    # horizontal along -X).  shape_centreline.npz (written next to the plan by
    # plan_shape_path.py) carries tip0 / u_axis / v_axis in that frame; we fit
    # a rigid planner->robot-base transform from it + the measured start tip so
    # the plan's desired_position_m lands in R.  Leave blank to auto-locate.
    plan_frame_npz: str = ""

    # --- synthetic shape (reference_source == "synthetic") -------
    shape: str = "triangle"                    # "square" | "triangle"
    shape_size_mm: float = 6.0               # half the side/height; shape starts at the current tip
    shape_speed_mm_s: float = 1.5            # tip traversal speed along the path
    shape_laps: int = 1
    shape_settle_s: float = 1.0              # hold at the first corner before moving
    shape_end_hold_s: float = 2.0            # hold at the last corner after the lap
    # In-plane axes of the beam frame B the shape is drawn in.
    #   u -> plotted as the horizontal axis   (default B.y: sideways, joints)
    #   v -> plotted as the vertical axis     (default B.x: axial, advancer)
    shape_axis_u: str = "b_y"
    shape_axis_v: str = "b_x"

    # --- controller ----------------------------------------------
    #   "naive_inverse_jacobian" : InverseJacobianBeamController (resolved-rate)
    #   "mpc_lti"                : BeamOutputTrackingMPC, one frozen Jacobian
    #   "mpc_ltv_offline"        : one Jacobian per reference sample (== mpc_lti
    #                              here, since the beam Jacobian is frozen)
    controller_kind: str = "naive_inverse_jacobian"
    position_gain: float = 0.6
    damping: float = 5.0e-2
    nullspace_gain: float = 0.0
    feedforward: bool = False
    control_insertion: bool = True

    # --- MPC (controller_kind == "mpc_lti" / "mpc_ltv_offline") ---
    mpc_prediction_horizon: int = 12
    mpc_freeze_index: int = 0             # reference sample the LTI model linearises at
    mpc_position_error_scale_mm: float = 0.5
    mpc_position_tracking_weight: float = 1.0
    mpc_use_dare_terminal_cost: bool = True

    # --- feedforward (trajectory tracking) ---------------------
    # False : pure feedback -- servo q_meas + the controller step toward the tip
    #         target.  Ignores the planned joint trajectory; with an accurate
    #         plan this is WORSE than open-loop (2026-09-10 triangle: inv RMS
    #         1.70, mpc_lti 1.47, vs open-loop 1.31 mm).
    # True  : servo the PLANNED joints reference.state[i][:6] plus the
    #         controller's CORRECTION as a trim, planned insertion rate to the
    #         advancer.  For MPC the planned input reference.input[i] is
    #         subtracted from the command first (the MPC already feeds it
    #         forward internally) so it is not double-counted.  2026-09-10:
    #         inv+FF RMS 1.06 mm / max 2.07 -- the best of all controllers.
    feedforward_joint_trajectory: bool = True
    feedforward_correction_scale: float = 1.0   # multiplies the feedback step in FF mode
    # Diagnostic escape hatch: MPC kinds normally have feedforward_joint_trajectory
    # force-disabled in main() (see the comment there -- known worse terminal
    # convergence, ~1.4-1.6mm).  Set this True to run mpc_lti/mpc_ltv_offline WITH
    # feedforward anyway, e.g. to collect instrumented data diagnosing *why* it is
    # worse instead of just avoiding the mode.
    force_mpc_feedforward: bool = False
    # 2026-09-11 fix #2 (diagnosed root cause: MPC's u0 does not shrink to zero
    # as the tracking error shrinks/grows the way inverse-Jacobian's does --
    # corr(|u0|,error) collapses to ~0.03-0.14 at the hold vs inverse-Jacobian's
    # 0.98-1.0 -- so once FF bakes that "sticky" u0 into a trim on the FROZEN
    # terminal reference position, nothing pulls it back toward the true
    # residual). When True: once terminal_hold is reached, fall back to pure
    # feedback (q_target = q_meas + delta_q, exactly the non-FF branch) instead
    # of trimming the frozen ref_state[target_index] -- feedforward stays in
    # charge during transit (where it already works), feedback takes over for
    # the static hold (where it's needed). See beam-lateral-authority-limit
    # memory for the diagnosis this is based on.
    ff_trim_base_switch_at_hold: bool = False
    # 2026-09-11 fix #1 (same diagnosis, different lever): ConfigurationMPCConfig's
    # input_increment_weight (Rd) penalises u0 changing from the controller's
    # OWN previous output -- confirmed NOT a hard velocity-limit saturation
    # (max per-axis |u0| at hold was 0.062 rad/s vs the 0.10 rad/s bound), so
    # this is a genuine soft-cost lever. None = leave ConfigurationMPCConfig's
    # default (1e-3); set to a smaller value (e.g. 1e-4 or 0.0) to make MPC's
    # u0 respond more to the CURRENT residual instead of "stay near last tick" --
    # only applied when feedforward_joint_trajectory is True, so it never
    # touches MPC's already-winning pure-feedback behaviour.
    mpc_ff_input_increment_weight: float | None = None
    # 2026-09-11: general (NOT FF-gated) overrides for the same two QP cost
    # terms, to test "does MPC's constraint-aware horizon beat inverse-Jacobian
    # once control-effort regularisation is minimised, so its UNCONSTRAINED
    # behaviour matches resolved-rate and only the box constraints + lookahead
    # can differentiate them". Both None = ConfigurationMPCConfig defaults
    # (input_tracking_weight=1e-2, input_increment_weight=1e-3) unchanged.
    # CAUTION: driving these toward 0 removes positive-definite structure from
    # the QP Hessian -- on an ill-conditioned problem (this beam's Jacobian
    # condition number is ~1.5e5-1.7e5 depending on shape) that can mean MORE
    # solver iterations/time, not fewer; watch solver_time_s/iterations, not
    # just tracking error, when using this.
    mpc_input_tracking_weight_override: float | None = None
    mpc_input_increment_weight_override: float | None = None
    # 2026-09-11: real-time-fairness safety net -- confirmed via
    # target_index-jump-per-tick analysis that mpc_ltv_offline overran the
    # 100ms/10Hz control budget on 73% of ticks on a hard (ill-conditioned)
    # shape (mean solve 160ms, max 419ms), causing the WALLCLOCK-based
    # reference progress to silently skip 1.6 samples/tick on average instead
    # of 1 -- an unfair comparison against inverse-Jacobian's near-instant
    # (~0.1ms) solve. mpc_lti was NOT affected (0% overrun) on the same run,
    # so this is solver-time-specific, not a blanket MPC problem. None = no
    # override (ConfigurationMPCConfig.solver_time_limit_s stays 0 = OSQP
    # default/unbounded); set to a fraction of dt (e.g. 0.08 at 10Hz) to force
    # OSQP to return its best solution within budget instead of overrunning.
    mpc_solver_time_limit_s: float | None = None
    # 2026-09-11: explicit anisotropic input regularisation (see
    # BeamOutputMPCConfig.directional_damping's docstring for the full
    # diagnosis) -- penalises only the near-null singular direction of each
    # horizon step's local Jacobian, unlike the isotropic
    # mpc_input_tracking_weight_override which plateaus without reaching
    # inverse-Jacobian's directional-damping behaviour. 0.0 = disabled.
    mpc_directional_damping: float = 0.0
    mpc_directional_damping_floor: float = 0.01

    # --- control loop -------------------------------------------
    control_hz: float = 10.0
    servo_lookahead_s: float = 0.20
    servo_gain: int = 200
    # 2026-09-11: briefly raised 0.006 -> 0.012 -> 0.018 while diagnosing why
    # pure-feedback inverse-Jacobian doesn't reach the 20mm triangle's corners
    # (see beam-lateral-authority-limit memory: the old 0.006 cap was pinned
    # on 75% of ticks). Reverted back to the original 0.006 alongside
    # joint_velocity_limit_rad_s/joint_acceleration_limit_rad_s2 per user
    # request, to keep a consistent baseline for the open-loop/inv/mpc_lti/
    # mpc_ltv x FF/no-FF comparison -- the corner-reaching fix that actually
    # works cleanly is feedforward, not loosened limits (which helped pure
    # feedback partially but measurably hurt feedforward).
    max_joint_step_rad: float = 0.006
    max_control_steps: int = 600
    max_state_age_s: float = 0.50
    warmup_timeout_s: float = 20.0
    settle_ticks_before_start: int = 10
    project_error_to_beam_plane: bool = True

    # --- beam frame B in R (kept in sync with robotics_frame_measurement_validation) --
    beam_axial_axis_R: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    beam_plane_normal_axis_R: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # --- frozen beam Jacobian ----------------------------------
    jacobian_source: str = "analytical_beam"   # "analytical_beam" | "kinematic_scalar"
    magnet_tip_coupling: float = 1.0
    insertion_axial_gain: float = 1.0
    # 2026-09-10 calibration: source-magnet dipole direction in the magnet body
    # frame that gives a world -X (beam-axial) dipole at the reference pose.
    # Same value as robotics_frame_measurement_validation / close_loop_tip_control.
    dipole_unit_in_magnet_body: tuple[float, float, float] = (-0.932073, 0.361306, 0.026427)

    # --- advancer (insertion) ---------------------------------
    advancer_port: str = "/dev/ttyACM0"
    advancer_dry_run: bool = False

    # --- limits ----------------------------------------------
    # 2026-09-11: velocity/accel/step-cap were all raised today while
    # diagnosing why pure-feedback inverse-Jacobian doesn't reach the 20mm
    # triangle's corners (see beam-lateral-authority-limit memory). Confirmed
    # the raise helps pure feedback somewhat but HURTS feedforward (RMS
    # 1.61->2.14, visible overshoot at one corner) -- feedforward's trim is
    # normally small and was already well-tuned against the tighter original
    # limits, so loosening them just removes useful damping on the few ticks
    # where the trim spikes. Reverted all three back to original here per
    # user request, to keep a consistent, FF-validated baseline for the
    # upcoming open-loop/inv/mpc_lti/mpc_ltv x FF/no-FF x 3-rep comparison.
    joint_velocity_limit_rad_s: float = 0.10
    joint_acceleration_limit_rad_s2: float = 0.40
    insertion_rate_limit_m_s: float = 2.0e-3

    workspace_xyz_min_m: tuple[float, float, float] = (0.20, -1.20, -0.30)
    workspace_xyz_max_m: tuple[float, float, float] = (1.10, -0.20, 0.70)

    # --- robot / vision (mirrors StateStreamConfig) ----------
    robot_ip: str = "192.168.56.101"
    reader_poll_hz: float = 60.0
    robot_max_age_s: float = 0.15
    # Must equal initial_conditions.make_initial_poses()[2] (the offline planner
    # start length).  2026-09-10 bigger-square study: 30 mm.
    initial_insertion_m: float = 0.030

    cam_index: int = 0
    exposure: float = 29.0                 # matches the 2026-09-10 calibration sweeps
    gain: float = 0.0
    grab_period_s: float = 0.004
    reconstruct_period_s: float = 0.01
    image_filename: str = "/dev/shm/proper_pathfollow_frame.png"

    output_root: str = "close_loop_logs"
    run_name: str = "path_follow"


CONFIG = PathFollowConfig()

# A comparison harness may set this to an ndarray of shape [reference_samples, 3, 7]
# (d(tip_R)/d[q1..q6, insertion] relinearised at each reference sample).  When
# set, mpc_lti / mpc_ltv_offline use it instead of freezing / recomputing off the
# frozen jac_provider.  None -> normal behaviour.
_SCHEDULE_OVERRIDE = None

_AXIS_COLUMN = {"b_x": 0, "b_y": 1, "b_z": 2}


# =============================================================================
# synthetic shape reference -- built directly in the robot base frame
# =============================================================================


def _shape_corners_2d(shape: str, size_m: float) -> np.ndarray:
    """Closed polyline (last point == first) of the shape, centred on the origin."""
    if shape == "square":
        unit = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
        )
    elif shape == "triangle":
        unit = np.array(
            [[0.0, 1.0], [-0.8660254, -0.5], [0.8660254, -0.5], [0.0, 1.0]]
        )
    else:
        raise ValueError(f"shape must be 'square' or 'triangle'; got {shape!r}")
    return unit * float(size_m)


def _resample_polyline(corners: np.ndarray, ds: float) -> np.ndarray:
    """Walk the polyline at ~``ds`` spacing; keeps every corner exactly."""
    ds = max(float(ds), 1.0e-5)
    points = [corners[0].copy()]
    for start, end in zip(corners[:-1], corners[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        segments = max(1, int(math.ceil(length / ds)))
        for i in range(1, segments + 1):
            points.append(start + segment * (i / segments))
    return np.asarray(points, dtype=float)


def _build_shape_reference(
    cfg: PathFollowConfig,
    *,
    tip0_R: np.ndarray,
    T_R_B: Any,
    z0: np.ndarray,
    dt: float,
) -> Any:
    """A :class:`ConfigurationReference` whose ``desired_position_m`` traces the
    shape in the (u, v) plane of the beam frame, centred on ``tip0_R``."""
    from proper_research.simulation.simulations import (
        simulate_time_parameterized_configuration_mpc as base_module,
    )

    u_axis = np.asarray(T_R_B.rotation[:, _AXIS_COLUMN[cfg.shape_axis_u]], dtype=float)
    v_axis = np.asarray(T_R_B.rotation[:, _AXIS_COLUMN[cfg.shape_axis_v]], dtype=float)
    axial_R = np.asarray(T_R_B.rotation[:, 0], dtype=float)
    axial_R = axial_R / (np.linalg.norm(axial_R) + 1e-12)

    ds = max(cfg.shape_speed_mm_s * 1.0e-3 * dt, 1.0e-4)
    corners = _shape_corners_2d(cfg.shape, cfg.shape_size_mm * 1.0e-3)
    corners = corners - corners[0]                              # start AT the current tip
    lap = _resample_polyline(corners, ds)                       # (M, 2), closed
    one_lap = lap[1:]                                           # drop the repeated seam
    path2d = np.vstack([lap[:1]] + [one_lap] * max(1, cfg.shape_laps))

    settle = max(0, int(round(cfg.shape_settle_s / dt)))
    end_hold = max(1, int(round(cfg.shape_end_hold_s / dt)))
    path2d = np.vstack(
        [np.tile(path2d[:1], (settle, 1)), path2d, np.tile(path2d[-1:], (end_hold, 1))]
    )

    count = path2d.shape[0]
    desired = (
        tip0_R[None, :]
        + path2d[:, 0:1] * u_axis[None, :]
        + path2d[:, 1:2] * v_axis[None, :]
    )
    seg = np.linalg.norm(np.diff(desired, axis=0), axis=1)
    path_coordinate = np.concatenate([[0.0], np.cumsum(seg)])

    z0 = np.asarray(z0, dtype=float).reshape(7)
    reference = base_module.ConfigurationReference(
        time_s=np.arange(count, dtype=float) * dt,
        path_coordinate_m=path_coordinate,
        state=np.tile(z0, (count, 1)),
        input=np.zeros((count, 7)),
        acceleration=np.zeros((count, 7)),
        desired_position_m=desired,
        desired_tangent=np.tile(axial_R, (count, 1)),
        planned_beam_feasible=np.ones(count, dtype=bool),
        planned_position_error_m=np.zeros(count),
        planned_tangent_error_rad=np.zeros(count),
        sample_period_s=float(dt),
        source=f"synthetic_{cfg.shape}",
    )
    reference.validate(require_planned_beam_feasible=False)
    return reference


def _find_shape_npz(plan_dir: Path) -> Optional[Path]:
    """Locate shape_centreline.npz written by plan_shape_path.py near the plan."""
    for base in (plan_dir, *plan_dir.parents[:3]):
        hits = sorted(base.glob("**/shape_centreline.npz"))
        if hits:
            return hits[0]
    return None


def _fit_planner_to_robot(
    npz_path: Path, start_tip_R: np.ndarray, T_R_B: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R_fit, t_fit): planner-world point -> robot-base point.

    Aligns the planner's (tip0, +u forward, +v in-plane) triad with the
    hardware (measured start tip, B.x axial, B.y in-plane) triad.  x_R =
    R_fit @ x_planner + t_fit.
    """
    data = np.load(npz_path)
    tip0_p = np.asarray(data["tip0"], dtype=float).reshape(3)
    u_p = np.asarray(data["u_axis"], dtype=float).reshape(3)
    v_p = np.asarray(data["v_axis"], dtype=float).reshape(3)
    u_p = u_p / (np.linalg.norm(u_p) + 1e-12)
    v_p = v_p - np.dot(v_p, u_p) * u_p
    v_p = v_p / (np.linalg.norm(v_p) + 1e-12)
    planner_basis = np.column_stack([u_p, v_p, np.cross(u_p, v_p)])

    b_x = np.asarray(T_R_B.rotation[:, 0], dtype=float)  # axial  <- planner +u
    b_y = np.asarray(T_R_B.rotation[:, 1], dtype=float)  # in-plane <- planner +v
    b_z = np.asarray(T_R_B.rotation[:, 2], dtype=float)
    robot_basis = np.column_stack([b_x, b_y, b_z])

    r_fit = robot_basis @ planner_basis.T
    t_fit = np.asarray(start_tip_R, dtype=float).reshape(3) - r_fit @ tip0_p
    return r_fit, t_fit


def _load_plan_reference(
    cfg: PathFollowConfig, dt: float, start_tip_R: np.ndarray, T_R_B: Any
) -> Any:
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )

    plan_dir = Path(cfg.plan_dir).expanduser()
    if not plan_dir.exists():
        raise FileNotFoundError(f"plan_dir does not exist: {plan_dir}")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    if not math.isclose(reference.sample_period_s, dt, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"plan sample period {reference.sample_period_s:.4f}s != control dt "
            f"{dt:.4f}s. Set control_hz = {1.0 / reference.sample_period_s:.3f} "
            "or re-run time parameterisation at this rate."
        )

    npz_path = (
        Path(cfg.plan_frame_npz).expanduser()
        if cfg.plan_frame_npz
        else _find_shape_npz(plan_dir)
    )
    if npz_path is None or not npz_path.exists():
        raise FileNotFoundError(
            "shape_centreline.npz not found -- needed to map the planner frame "
            "into the robot base frame. Set plan_frame_npz explicitly."
        )
    r_fit, t_fit = _fit_planner_to_robot(npz_path, start_tip_R, T_R_B)
    des_planner = np.asarray(reference.desired_position_m, dtype=float)
    des_R = des_planner @ r_fit.T + t_fit
    tan_planner = np.asarray(reference.desired_tangent, dtype=float)
    tan_R = tan_planner @ r_fit.T
    tan_R = tan_R / (np.linalg.norm(tan_R, axis=1, keepdims=True) + 1e-12)

    from dataclasses import replace

    print(
        f"[path] planner->R fit: |t_fit|={np.linalg.norm(t_fit):.3f} m, "
        f"det(R_fit)={np.linalg.det(r_fit):+.3f}\n"
        f"[path]   plan des[0] {np.round(1e3 * des_R[0], 1).tolist()} mm  "
        f"des[-1] {np.round(1e3 * des_R[-1], 1).tolist()} mm  (robot frame)"
    )
    return replace(reference, desired_position_m=des_R, desired_tangent=tan_R)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    import sys

    cfg = CONFIG
    if cfg.controller_kind not in ("naive_inverse_jacobian", "mpc_lti", "mpc_ltv_offline"):
        raise NotImplementedError(
            f"controller_kind={cfg.controller_kind!r} not supported; use "
            "'naive_inverse_jacobian', 'mpc_lti' or 'mpc_ltv_offline'."
        )
    if (
        cfg.feedforward_joint_trajectory
        and cfg.controller_kind != "naive_inverse_jacobian"
        and not cfg.force_mpc_feedforward
    ):
        # The MPC formulation (state prediction from the measured state, Delta-u
        # cost, DARE terminal) assumes the "servo q_meas + u0" update.  Servoing
        # the planned joints + a correction fights that -- tested 2026-09-10,
        # mpc_*+FF terminal error ~1.6 mm.  Pure feedback for MPC (it already
        # tracks the planned state in its cost).  Set cfg.force_mpc_feedforward
        # to run it anyway (diagnostic runs).
        print(f"[path] feedforward_joint_trajectory auto-disabled for {cfg.controller_kind}")
        cfg = replace_dc(cfg, feedforward_joint_trajectory=False)
    elif cfg.feedforward_joint_trajectory and cfg.controller_kind != "naive_inverse_jacobian":
        print(
            f"[path] force_mpc_feedforward=True: leaving feedforward_joint_trajectory=True "
            f"for {cfg.controller_kind} (known-bad mode, running for diagnostics)"
        )
    dt = 1.0 / cfg.control_hz
    real_stdout = sys.stdout
    sys.stdout = _DebugLineFilter(real_stdout)
    output_dir = _make_output_dir(cfg)
    log_path = output_dir / "path_follow.jsonl"

    mode = "DRY-RUN (no motion)" if cfg.dry_run else "LIVE (servoJ per tick)"
    print(f"[path] {mode}  control @ {cfg.control_hz:.1f} Hz  ref={cfg.reference_source}")
    print(f"[path] log -> {log_path}")

    stream_cfg = StateStreamConfig(
        robot_ip=cfg.robot_ip,
        cam_index=cfg.cam_index,
        exposure=cfg.exposure,
        gain=cfg.gain,
        grab_period_s=cfg.grab_period_s,
        reconstruct_period_s=cfg.reconstruct_period_s,
        image_filename=cfg.image_filename,
        beam_axial_axis_R=cfg.beam_axial_axis_R,
        beam_plane_normal_axis_R=cfg.beam_plane_normal_axis_R,
    )
    mapper = NewFrameTipMapper(stream_cfg)
    beam_axis_R = np.asarray(mapper.T_R_B.rotation[:, 0], dtype=float)
    print(
        f"[path] T_R_B  B.x(axial)={np.round(mapper.T_R_B.rotation[:, 0], 3).tolist()}  "
        f"B.y(sideways)={np.round(mapper.T_R_B.rotation[:, 1], 3).tolist()}  "
        f"B.z(camera)={np.round(mapper.T_R_B.rotation[:, 2], 3).tolist()}"
    )

    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    robot = URRTDERobot(cfg.robot_ip, frequency=500.0)
    reader = RobotJointStream(cfg.robot_ip, poll_hz=cfg.reader_poll_hz, receive_only=True)

    def _assert_robot_ready() -> None:
        mode_ = robot.get_robot_mode()
        safety = robot.get_safety_mode()
        pstop = robot.is_protective_stopped()
        if pstop or mode_ != 7 or safety not in (1, 2):
            raise RuntimeError(
                f"robot not ready (mode={mode_}, safety={safety}, protective_stop={pstop}). "
                "Clear any protective stop on the pendant (Remote Control), then retry."
            )

    insertion_m = float(cfg.initial_insertion_m)

    camera = CameraSource(
        CameraConfig(
            cam_index=cfg.cam_index,
            exposure=cfg.exposure,
            gain=cfg.gain,
            grab_period_s=cfg.grab_period_s,
            reconstruct_period_s=cfg.reconstruct_period_s,
            image_filename=cfg.image_filename,
            roi_polygon_path=stream_cfg.roi_polygon_path,
            manual_boundary_path=stream_cfg.manual_boundary_path,
            pivot_hint=tuple(stream_cfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(stream_cfg.T_robot_beam_pose6, dtype=float),
        robot_joints_getter=lambda: reader.latest_joints(cfg.robot_max_age_s),
        robot_pose_getter=lambda: reader.latest_pose(cfg.robot_max_age_s),
        insertion_length_getter=lambda: insertion_m,
        frame_processor=mapper,
    )

    advancer = None
    if cfg.control_insertion:
        from proper_research.hardware.online.advancer_sink import (
            AdvancerSink,
            AdvancerSinkConfig,
        )

        advancer = AdvancerSink(
            AdvancerSinkConfig(
                port=cfg.advancer_port,
                dry_run=cfg.advancer_dry_run,
                max_rate_m_s=cfg.insertion_rate_limit_m_s,
            )
        )
        print(
            f"[path] advancer {'(dry-run)' if cfg.advancer_dry_run else 'LIVE'} "
            f"on {cfg.advancer_port}"
        )

    stop_flag = {"stop": False}
    prev_handler = signal.signal(
        signal.SIGINT, lambda *_: stop_flag.__setitem__("stop", True)
    )

    steps = 0
    ins_trim = 0.0   # accumulated feedback insertion correction (FF mode)
    abort_reason = ""
    started_moving = False
    jac_provider: Any = None
    solver: Any = None
    reference: Any = None
    log_file = log_path.open("w", encoding="utf-8")
    traj_file = (output_dir / "tip_trajectory.csv").open("w", encoding="utf-8", newline="")
    traj_file.write(
        "step,t_s,ref_index,tip_x_m,tip_y_m,tip_z_m,"
        "des_x_m,des_y_m,des_z_m,err_norm_mm,"
        "q1,q2,q3,q4,q5,q6,insertion_m,terminal_hold\n"
    )
    t_zero = now_monotonic()
    tip_log: list[np.ndarray] = []
    des_log: list[np.ndarray] = []
    err_log: list[float] = []
    tsec_log: list[float] = []

    try:
        _assert_robot_ready()
        reader.start()
        if advancer is not None:
            advancer.start()
        camera.start()
        print("[path] camera + reader up; waiting for first tip + joints...")

        deadline = now_monotonic() + cfg.warmup_timeout_s
        estimate = None
        while now_monotonic() < deadline and not stop_flag["stop"]:
            estimate, _age = camera.latest(cfg.max_state_age_s)
            if estimate is not None and estimate.robot_joints is not None:
                break
            estimate = None
            if not camera.healthy:
                raise RuntimeError(f"camera unhealthy during warmup: {camera.last_error!r}")
            time.sleep(0.05)
        if estimate is None:
            raise RuntimeError(
                "no fresh tip+joints within warmup_timeout_s "
                f"(vis_fail={mapper.detection_failures}, {mapper.last_error!r})"
            )

        start_tips = []
        while len(start_tips) < cfg.settle_ticks_before_start and not stop_flag["stop"]:
            est, _age = camera.latest(cfg.max_state_age_s)
            if est is not None:
                start_tips.append(np.asarray(est.tip_position_m, dtype=float))
            time.sleep(dt)
        start_tip = np.mean(np.vstack(start_tips), axis=0)
        q0 = np.asarray(estimate.robot_joints, dtype=float).reshape(6)
        z0 = np.concatenate([q0, [insertion_m]])

        # --- timed reference -----------------------------------
        if cfg.reference_source == "plan_dir":
            reference = _load_plan_reference(cfg, dt, start_tip, mapper.T_R_B)
            print(
                f"[path] loaded plan reference: {reference.sample_count} samples, "
                f"{reference.duration_s:.1f}s, source={reference.source!r}"
            )
        else:
            reference = _build_shape_reference(
                cfg, tip0_R=start_tip, T_R_B=mapper.T_R_B, z0=z0, dt=dt
            )
            print(
                f"[path] synthetic {cfg.shape}: {reference.sample_count} samples "
                f"({reference.duration_s:.1f}s), size +/-{cfg.shape_size_mm:.0f}mm, "
                f"speed {cfg.shape_speed_mm_s:.1f}mm/s, laps {cfg.shape_laps}"
            )
        des0 = reference.desired_position_m[0]
        des_last = reference.desired_position_m[-1]
        print(
            f"[path] start tip (mm) = {np.round(1e3 * start_tip, 2).tolist()}\n"
            f"[path] ref[0]     (mm) = {np.round(1e3 * des0, 2).tolist()}\n"
            f"[path] ref[-1]    (mm) = {np.round(1e3 * des_last, 2).tolist()}"
        )

        # --- frozen beam Jacobian -----------------------------
        if cfg.jacobian_source == "analytical_beam":
            print("[path] building analytic beam Jacobian (forward-model solve + FK)...")
            tcp_pose6 = np.asarray(
                robot.get_tcp_pose() if robot.get_tcp_pose() is not None else q0 * 0.0,
                dtype=float,
            )
            jac_provider = AnalyticalBeamJacobianProvider(
                robot=robot,
                q0=q0,
                tcp_pose6=tcp_pose6,
                insertion_m=insertion_m,
                mapper=mapper,
                dipole_unit_in_magnet_body=cfg.dipole_unit_in_magnet_body,
            )
        else:
            print("[path] building kinematic-scalar Jacobian (7 FK calls)...")
            jac_provider = SimpleBeamJacobianProvider(
                robot=robot,
                q0=q0,
                beam_axis_R=beam_axis_R,
                magnet_tip_coupling=cfg.magnet_tip_coupling,
                insertion_axial_gain=cfg.insertion_axial_gain,
            )

        if isinstance(jac_provider, AnalyticalBeamJacobianProvider):
            m_B = 1e3 * jac_provider.magnet_in_B_m
            col_norms = np.linalg.norm(jac_provider.j_beam_full[:, 0:3], axis=0)
            print(
                f"[path] model magnet in B = [{m_B[0]:+.0f}, {m_B[1]:+.0f}, {m_B[2]:+.0f}] mm "
                f"(coaxial: B.y/B.z ~0 -> magnet on the beam axis, ~280 mm ahead)\n"
                f"[path] |d(tip)/d(magnet)| per axis  B.z/out={col_norms[0]:.3f}  "
                f"B.y/in-plane={col_norms[1]:.3f}  B.x/axial={col_norms[2]:.3f} mm/mm "
                f"(axial should be ~0 for a coaxial axial-dipole magnet)"
            )
        j_full = jac_provider(z0)
        jv_robot = _translational_jacobian_fd(robot, q0)
        with np.printoptions(precision=5, suppress=True, linewidth=160):
            print("[path] ===== FROZEN BEAM JACOBIAN  d(tip_R xyz)/d[q1..q6, insertion] =====")
            print("[path] rows = tip R.x, R.y, R.z   cols = q1..q6 (rad), insertion (m)")
            for r, name in enumerate(("R.x", "R.y", "R.z")):
                print(f"[path]   {name}: {j_full[r]}")
            if isinstance(jac_provider, AnalyticalBeamJacobianProvider):
                print("[path] --- J = J_beam[:, 0:3] @ Jv_robot ,  J[:,6] = J_beam[:,6] ---")
                print("[path] J_beam (3x7) d(tip)/d[magnet_xyz(3), magnet_rot(3), insertion]:")
                for r, name in enumerate(("R.x", "R.y", "R.z")):
                    print(f"[path]   {name}: {jac_provider.j_beam_full[r]}")
            print("[path] Jv_robot (3x6) d(TCP_pos)/d(q):")
            for r, name in enumerate(("R.x", "R.y", "R.z")):
                print(f"[path]   {name}: {jv_robot[r]}")
            print("[path] ================================================================")
        print(f"[path] cond(J[:, :6]) = {jac_provider.last_condition:.2f}")

        # --- offline controller through the standard seam ----
        terminal_hold_steps = max(1, int(round(cfg.control_hz * cfg.shape_end_hold_s)))
        from dataclasses import replace as _dc_replace

        mpc_config = _mpc_config(cfg, dt)
        beam_config = None
        if cfg.controller_kind in ("mpc_lti", "mpc_ltv_offline"):
            from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
                BeamOutputMPCConfig,
            )

            mpc_config = _dc_replace(
                mpc_config, prediction_horizon=int(cfg.mpc_prediction_horizon)
            )
            if (
                cfg.feedforward_joint_trajectory
                and cfg.mpc_ff_input_increment_weight is not None
            ):
                print(
                    f"[path] fix #1: overriding input_increment_weight "
                    f"{mpc_config.input_increment_weight:g} -> "
                    f"{cfg.mpc_ff_input_increment_weight:g} (FF mode)"
                )
                mpc_config = _dc_replace(
                    mpc_config,
                    input_increment_weight=float(cfg.mpc_ff_input_increment_weight),
                )
            if cfg.mpc_input_tracking_weight_override is not None:
                print(
                    f"[path] overriding input_tracking_weight "
                    f"{mpc_config.input_tracking_weight:g} -> "
                    f"{cfg.mpc_input_tracking_weight_override:g}"
                )
                mpc_config = _dc_replace(
                    mpc_config,
                    input_tracking_weight=float(
                        cfg.mpc_input_tracking_weight_override
                    ),
                )
            if cfg.mpc_input_increment_weight_override is not None:
                print(
                    f"[path] overriding input_increment_weight "
                    f"{mpc_config.input_increment_weight:g} -> "
                    f"{cfg.mpc_input_increment_weight_override:g}"
                )
                mpc_config = _dc_replace(
                    mpc_config,
                    input_increment_weight=float(
                        cfg.mpc_input_increment_weight_override
                    ),
                )
            if cfg.mpc_solver_time_limit_s is not None:
                print(
                    f"[path] setting OSQP solver_time_limit_s -> "
                    f"{cfg.mpc_solver_time_limit_s:g}s (real-time fairness cap)"
                )
                mpc_config = _dc_replace(
                    mpc_config,
                    solver_time_limit_s=float(cfg.mpc_solver_time_limit_s),
                )
            s = float(cfg.mpc_position_error_scale_mm) * 1.0e-3
            beam_config = BeamOutputMPCConfig(
                position_error_scale_m=(s, s, s),
                position_tracking_weight=float(cfg.mpc_position_tracking_weight),
                use_dare_terminal_cost=bool(cfg.mpc_use_dare_terminal_cost),
                directional_damping=float(cfg.mpc_directional_damping),
                directional_damping_floor=float(cfg.mpc_directional_damping_floor),
            )

        # Optional externally-supplied Jacobian schedule (shape [samples, 3, 7]).
        # A comparison harness sets `close_loop_path_follow._SCHEDULE_OVERRIDE`
        # to a relinearised-along-the-reference schedule so mpc_lti (frozen at
        # sample 0) and mpc_ltv_offline (full schedule) are compared on the
        # SAME model, differing only in time-variation.  Ignored by
        # naive_inverse_jacobian (which relinearises live off `jac_provider`).
        schedule_override = globals().get("_SCHEDULE_OVERRIDE")
        if schedule_override is not None and beam_config is not None:
            schedule_override = np.asarray(schedule_override, dtype=float)
            if schedule_override.shape != (reference.sample_count, 3, 7):
                raise ValueError(
                    f"_SCHEDULE_OVERRIDE shape {schedule_override.shape} != "
                    f"({reference.sample_count}, 3, 7)"
                )
            print(f"[path] using external Jacobian schedule {schedule_override.shape}")
        else:
            schedule_override = None

        solver = build_offline_solver(
            cfg.controller_kind,
            reference=reference,
            jacobian_provider=jac_provider,
            mpc_config=mpc_config,
            beam_config=beam_config,
            schedule=schedule_override,
            freeze_index=int(cfg.mpc_freeze_index),
            adapter_config=OfflineControllerConfig(
                progress_mode="wallclock",
                terminal_hold_steps=terminal_hold_steps,
            ),
            position_gain=cfg.position_gain,
            damping=cfg.damping,
            nullspace_gain=cfg.nullspace_gain,
            feedforward=cfg.feedforward,
            allow_undeclared_jacobian=True,
        )
        print(
            f"[path] controller = {cfg.controller_kind}"
            + (
                f" (horizon={cfg.mpc_prediction_horizon}, freeze@{cfg.mpc_freeze_index})"
                if beam_config is not None
                else ""
            )
            + "; closing the loop"
        )

        # --- persist the frozen Jacobian (+ MPC schedule, if any) as structured
        # data, once per run, so a comparison harness can diagnose controller
        # internals after the fact instead of only reading console prints.
        try:
            jac_dump = {
                "controller_kind": cfg.controller_kind,
                "feedforward_joint_trajectory": bool(cfg.feedforward_joint_trajectory),
                "jacobian_source": cfg.jacobian_source,
                "q0_rad": np.asarray(q0, dtype=float).reshape(-1).tolist(),
                "insertion0_m": float(insertion_m),
                "j_full_3x7": np.asarray(j_full, dtype=float).tolist(),
                "jv_robot_3x6": np.asarray(jv_robot, dtype=float).tolist(),
                "jacobian_condition": float(jac_provider.last_condition),
            }
            if isinstance(jac_provider, AnalyticalBeamJacobianProvider):
                jac_dump["magnet_in_B_mm"] = (1e3 * jac_provider.magnet_in_B_m).tolist()
                jac_dump["j_beam_full_3x7"] = np.asarray(
                    jac_provider.j_beam_full, dtype=float
                ).tolist()
            if schedule_override is not None:
                jac_dump["schedule_override_shape"] = list(schedule_override.shape)
                jac_dump["schedule_override"] = schedule_override.tolist()
            if beam_config is not None:
                jac_dump["mpc_prediction_horizon"] = int(cfg.mpc_prediction_horizon)
                jac_dump["mpc_freeze_index"] = int(cfg.mpc_freeze_index)
                jac_dump["mpc_position_error_scale_mm"] = float(
                    cfg.mpc_position_error_scale_mm
                )
                jac_dump["mpc_position_tracking_weight"] = float(
                    cfg.mpc_position_tracking_weight
                )
                jac_dump["mpc_use_dare_terminal_cost"] = bool(
                    cfg.mpc_use_dare_terminal_cost
                )
            jac_dump_path = output_dir / "frozen_jacobian.json"
            with jac_dump_path.open("w", encoding="utf-8") as handle:
                json.dump(jac_dump, handle, indent=1)
            print(f"[path] frozen Jacobian (+schedule) -> {jac_dump_path}")
        except Exception as exc:  # pragma: no cover -- diagnostics must never break the run
            print(f"[path] WARNING: failed to dump frozen_jacobian.json: {exc!r}")

        safety_check_every = max(1, int(round(cfg.control_hz)))
        next_tick = now_monotonic()
        terminal_since = None

        while not stop_flag["stop"] and steps < cfg.max_control_steps:
            now = now_monotonic()
            if next_tick > now:
                time.sleep(min(next_tick - now, dt))
                continue
            next_tick = max(next_tick + dt, now)

            if not camera.healthy:
                abort_reason = f"camera_unhealthy: {camera.last_error!r}"
                break
            if steps % safety_check_every == 0:
                try:
                    if reader.is_protective_stopped() or reader.safety_mode() not in (1, 2):
                        abort_reason = "robot_protective_stop_or_unsafe"
                        break
                except Exception:
                    pass

            estimate, age = camera.latest(cfg.max_state_age_s)
            if estimate is None:
                abort_reason = f"stale_vision(age={age:.3f}s)"
                break
            if estimate.robot_joints is None:
                abort_reason = "no_joints_on_estimate"
                break

            pose = reader.latest_pose(cfg.robot_max_age_s)
            if pose is not None:
                low = np.asarray(cfg.workspace_xyz_min_m)
                high = np.asarray(cfg.workspace_xyz_max_m)
                if np.any(pose[:3] < low) or np.any(pose[:3] > high):
                    abort_reason = f"tcp_out_of_workspace({np.round(pose[:3], 3).tolist()})"
                    break

            result = solver(estimate, dt)
            command = np.asarray(result.u0, dtype=float).reshape(7)
            info = result.info
            ref_index = int(info.get("reference_index", 0))
            terminal_hold = bool(info.get("terminal_hold", False))

            if not cfg.control_insertion:
                command[6] = 0.0
            if not np.all(np.isfinite(command)):
                abort_reason = "nonfinite_command"
                break

            tip = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)
            # the inverse-Jacobian controller chases desired_position_m[ref_index + 1]
            target_index = min(ref_index + 1, reference.sample_count - 1)
            desired = np.asarray(reference.desired_position_m, dtype=float)[target_index]
            error = desired - tip
            if cfg.project_error_to_beam_plane:
                b_z = np.asarray(mapper.T_R_B.rotation[:, 2], dtype=float)
                error = error - float(np.dot(error, b_z)) * b_z
            error_mm = 1.0e3 * float(np.linalg.norm(error))
            q = np.asarray(estimate.robot_joints, dtype=float).reshape(6)

            ref_state = np.asarray(reference.state, dtype=float)
            ref_input = np.asarray(reference.input, dtype=float)
            # The MPC command already contains the planned (feedforward) input
            # in its cost; the resolved-rate command is pure correction.  In FF
            # mode, subtract the planned input for MPC so we don't double-count.
            is_mpc = cfg.controller_kind in ("mpc_lti", "mpc_ltv_offline")
            corr = command.copy()
            if cfg.feedforward_joint_trajectory and is_mpc:
                corr = command - ref_input[target_index]

            ff_ins_rate = 0.0
            if (
                cfg.feedforward_joint_trajectory
                and reference.sample_count > 1
                and not terminal_hold
            ):
                # 2026-09-11 bug fix: target_index/prev_ix both freeze at the
                # last reference sample during the terminal hold, so this used
                # to keep re-computing the SAME (generally nonzero -- the
                # planned trajectory can have residual insertion slope right
                # before the hold starts) rate every tick for the whole hold,
                # driving a real, unintended insertion drift (+0.3 to +1.7mm
                # measured across FF runs -- see beam-lateral-authority-limit
                # memory). A held target should not keep advancing insertion.
                prev_ix = max(0, target_index - 1)
                ff_ins_rate = float(
                    (ref_state[target_index, 6] - ref_state[prev_ix, 6]) / dt
                )

            if advancer is not None:
                advancer.submit_rate(
                    (float(corr[6]) if cfg.feedforward_joint_trajectory else float(command[6]))
                    + ff_ins_rate, dt
                )
            if cfg.feedforward_joint_trajectory:
                ins_trim += float(corr[6]) * dt
                insertion_m = float(np.clip(ref_state[target_index, 6] + ins_trim, 0.005, 0.20))
            elif advancer is not None:
                insertion_m = float(
                    np.clip(insertion_m + float(command[6]) * dt, 0.005, 0.20)
                )

            qd = np.clip(
                (corr[:6] if cfg.feedforward_joint_trajectory else command[:6]),
                -cfg.joint_velocity_limit_rad_s,
                cfg.joint_velocity_limit_rad_s,
            )
            delta_q = np.clip(qd * dt, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
            if cfg.feedforward_joint_trajectory and not (
                cfg.ff_trim_base_switch_at_hold and terminal_hold
            ):
                # servo the PLANNED joints + the controller correction as a trim
                q_target = ref_state[target_index, :6] + cfg.feedforward_correction_scale * delta_q
            else:
                # fix #2 (or FF off): pure feedback off the measured joints --
                # either FF was never on, or ff_trim_base_switch_at_hold kicked
                # in because the planned reference has stopped advancing.
                q_target = q + delta_q

            t_servo = now_monotonic()
            if not cfg.dry_run:
                robot.servo_j(
                    q_target,
                    time_s=2.0 * dt,
                    lookahead_time=cfg.servo_lookahead_s,
                    gain=cfg.servo_gain,
                )
                if np.any(np.abs(delta_q) > 1e-9):
                    started_moving = True
            servo_ms = 1e3 * (now_monotonic() - t_servo)

            steps += 1
            t_rel = now_monotonic() - t_zero
            tip_log.append(tip.copy())
            des_log.append(desired.copy())
            err_log.append(error_mm)
            tsec_log.append(t_rel)

            row = {
                "step": steps,
                "t_s": round(t_rel, 4),
                "state_age_s": round(float(age), 4),
                "ref_index": ref_index,
                "target_index": int(target_index),
                "progress_index": int(info.get("progress_index", ref_index)),
                "terminal_hold": terminal_hold,
                "infeasible": bool(result.infeasible),
                "tip_mm": [round(float(v), 3) for v in 1e3 * tip],
                "desired_mm": [round(float(v), 3) for v in 1e3 * desired],
                "error_mm": [round(float(v), 3) for v in 1e3 * error],
                "error_norm_mm": round(error_mm, 3),
                "u0": [round(float(v), 6) for v in command],
                "u0_correction": [round(float(v), 6) for v in corr],
                "ff_insertion_rate_m_s": round(ff_ins_rate, 6),
                "insertion_length_m": round(float(insertion_m), 5),
                "q_meas_rad": [round(float(v), 6) for v in q],
                "q_target_delta_rad": [round(float(v), 6) for v in delta_q],
                "servo_ms": round(servo_ms, 1),
                "dry_run": cfg.dry_run,
                # --- controller-internals diagnostics (2026-09-11) ---
                "controller_kind": cfg.controller_kind,
                "feedforward": bool(cfg.feedforward_joint_trajectory),
                "solver_status": info.get("status"),
                "solver_success": info.get("success"),
                "solver_iterations": info.get("iterations"),
                "solver_time_s": info.get("solve_time_s"),
                "objective": info.get("objective"),
                "primal_residual": info.get("primal_residual"),
                "dual_residual": info.get("dual_residual"),
                "planned_input": info.get("planned_input"),
                "predicted_input_0": info.get("predicted_input_0"),
                "predicted_state_0": info.get("predicted_state_0"),
                "predicted_beam_position_0_m": info.get("predicted_beam_position_0_m"),
                "predicted_beam_error_0_m": info.get("predicted_beam_error_0_m"),
                "horizon_len": info.get("horizon_len"),
                "jacobian_condition": round(float(jac_provider.last_condition), 2),
            }
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            traj_file.write(
                f"{steps},{t_rel:.4f},{ref_index},"
                f"{tip[0]:.6f},{tip[1]:.6f},{tip[2]:.6f},"
                f"{desired[0]:.6f},{desired[1]:.6f},{desired[2]:.6f},"
                f"{error_mm:.3f},"
                + ",".join(f"{v:.6f}" for v in q)
                + f",{insertion_m:.6f},{int(terminal_hold)}\n"
            )
            traj_file.flush()

            if steps % 10 == 0 or (terminal_hold and steps % 5 == 0):
                print(
                    f"[{steps:4d}] ref={ref_index:4d}/{reference.sample_count - 1} "
                    f"|e|={error_mm:6.2f}mm "
                    f"e=[{1e3*error[0]:+6.2f} {1e3*error[1]:+6.2f} {1e3*error[2]:+6.2f}] "
                    f"|qd|={np.linalg.norm(command[:6]):.4f} "
                    f"ins={1e3*insertion_m:5.1f}mm{' TERM' if terminal_hold else ''}"
                )

            if terminal_hold:
                terminal_since = terminal_since or now_monotonic()
                if now_monotonic() - terminal_since >= cfg.shape_end_hold_s:
                    abort_reason = "path_complete"
                    break
        else:
            abort_reason = abort_reason or "max_control_steps"
    except Exception as exc:  # noqa: BLE001
        abort_reason = f"exception: {exc!r}"
        raise
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        if not cfg.dry_run:
            try:
                robot.servo_stop(2.0)
            except Exception as exc:  # noqa: BLE001
                print(f"[path] servo_stop error: {exc!r}")
        camera.stop()
        if advancer is not None:
            try:
                advancer.submit_rate(0.0, dt)
                advancer.stop()
            except Exception:
                pass
        try:
            reader.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        log_file.close()
        traj_file.close()
        sys.stdout = real_stdout

    _write_plot(cfg, output_dir, mapper, tip_log, des_log, err_log, tsec_log)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "stop_reason": abort_reason or ("stopped" if stop_flag["stop"] else "done"),
        "control_steps": steps,
        "started_moving": started_moving,
        "reference_samples": None if reference is None else int(reference.sample_count),
        "jacobian_condition": (
            None if jac_provider is None else round(jac_provider.last_condition, 2)
        ),
        "final_error_mm": round(err_log[-1], 3) if err_log else None,
        "max_error_mm": round(max(err_log), 3) if err_log else None,
        "vision_detection_failures": mapper.detection_failures,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== PATH-FOLLOW SUMMARY ===")
    for key, value in summary.items():
        if key == "config":
            continue
        print(f"  {key:24s}: {value}")
    print(f"  log                     : {log_path}")
    print(f"  csv                     : {output_dir / 'tip_trajectory.csv'}")
    print(f"  plot                    : {output_dir / 'path_follow_plot.png'}")


def _write_plot(
    cfg: PathFollowConfig,
    output_dir: Path,
    mapper: Any,
    tip_log: list,
    des_log: list,
    err_log: list,
    tsec_log: list,
) -> None:
    if not tip_log:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[path] plot skipped (matplotlib unavailable: {exc!r})")
        return

    tip = np.vstack(tip_log)
    des = np.vstack(des_log)
    u_axis = np.asarray(mapper.T_R_B.rotation[:, _AXIS_COLUMN[cfg.shape_axis_u]], dtype=float)
    v_axis = np.asarray(mapper.T_R_B.rotation[:, _AXIS_COLUMN[cfg.shape_axis_v]], dtype=float)
    origin = des[0]
    tip_u = 1e3 * (tip - origin) @ u_axis
    tip_v = 1e3 * (tip - origin) @ v_axis
    des_u = 1e3 * (des - origin) @ u_axis
    des_v = 1e3 * (des - origin) @ v_axis

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax0.plot(des_u, des_v, "-", color="tab:blue", lw=2, label="reference")
    ax0.plot(tip_u, tip_v, "-", color="tab:red", lw=1.2, label="tip (measured)")
    ax0.plot(tip_u[0], tip_v[0], "ko", ms=5, label="start")
    ax0.set_xlabel(f"{cfg.shape_axis_u} (mm)")
    ax0.set_ylabel(f"{cfg.shape_axis_v} (mm)")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)
    ax0.set_title(f"{cfg.shape} in beam plane")

    ax1.plot(tsec_log, err_log, "-", color="tab:red", lw=1.2)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("tracking error |e| (mm)")
    ax1.grid(alpha=0.3)
    ax1.set_title("tip tracking error")

    fig.tight_layout()
    out = output_dir / "path_follow_plot.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[path] wrote {out}")


if __name__ == "__main__":
    main()
