"""Part 3 (first increment): close the loop on a static tip-position target.

Smallest useful closed loop -- fast tip-only vision in the new frame + the
existing joint-space controller (``InverseJacobianBeamController`` ==
``naive_inverse_jacobian``) driving the UR joints toward ONE fixed tip target
near the current position.

    tip (vision)  ─┐
    joints (rtde_receive) ─┼─▶ controller.solve() @ control_hz ─▶ qd[6]
    frozen beam Jacobian  ─┘                                        │
                                          q_target = q_meas + qd*T  │
                                                                    ▼
                       robot.servo_j(q_target, time=T, lookahead=...)

The command goes out through **servoJ**, not speedJ.  speedJ needs a hard
real-time streaming thread (``initPeriod``/``waitPeriod`` every ~8 ms); with a
loaded Python process (vision) that thread gets starved and the controller
protective-stops with **C271A1**.  servoJ is a position servo with a lookahead
buffer -- it is called *blocking* once per control tick and tolerates an
irregular rate, so no separate thread and no C271A1.  Readiness and safety mode
are checked from a separate ``rtde_receive`` connection.

The beam Jacobian is frozen at the start pose -- either the analytic beam
sensitivity (``jacobian_source="analytical_beam"``) or the UR translational
Jacobian * a scalar coupling.

SAFETY
------
* ``dry_run = True`` -> the whole loop runs and logs; nothing moves.
* joint speed and per-tick joint step are clamped; servoStop on any exit;
  the servoJ block itself decelerates the arm if the loop stops calling it.
* target defaults to startup tip + ``target_offset_R_mm`` (zero -> command ~0).
* measured-TCP workspace box, stale-vision abort, safety-mode watch, Ctrl+C.
* insertion is not commanded unless ``control_insertion = True``.

    python -m proper_research.hardware.online.close_loop_tip_control
"""

from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
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
class CloseLoopConfig:
    dry_run: bool = False                    # True -> compute + log only, no motion

    # --- target (robot base frame, metres) -------------------------
    target_offset_R_mm: tuple[float, float, float] = (0.0, -10.0, -5.0)
    tolerance_mm: float = 1.5
    converged_hold_ticks: int = 8
    # Drop the out-of-plane (B.z) component of the tip error before control.
    # The beam only moves in its bending plane, so d(tip)/d(magnet) has a
    # near-zero B.z row -> the damped pseudo-inverse amplifies out-of-plane
    # noise into large joint commands.  Projecting the error onto (B.x, B.y)
    # removes that.
    project_error_to_beam_plane: bool = True

    # --- control loop -------------------------------------------------
    control_hz: float = 10.0                # servoJ blocks for 1/control_hz per tick
    servo_lookahead_s: float = 0.20         # servoJ lookahead buffer (>= 1/control_hz)
    servo_gain: int = 200                   # servoJ proportional gain (100..2000)
    max_joint_step_rad: float = 0.006       # hard cap on |q_target - q_meas| per tick (~0.34 deg)
    max_control_steps: int = 120
    max_state_age_s: float = 0.50
    warmup_timeout_s: float = 20.0
    settle_ticks_before_target: int = 10

    # --- controller ------------------------------------------------
    #   "jacobian"      : InverseJacobianBeamController (uses the frozen beam Jacobian)
    #   "pid_cartesian" : PID on the tip error in R, assumes tip follows magnet
    #     1:1, maps the desired magnet velocity to joints with ONLY the robot
    #     translational Jacobian.  Bypasses the beam model entirely -- if this
    #     converges and "jacobian" does not, the beam Jacobian / its frame is
    #     the problem.
    controller_type: str = "jacobian"
    position_gain: float = 0.6            # jacobian mode
    damping: float = 5.0e-2               # jacobian mode
    nullspace_gain: float = 0.0
    control_insertion: bool = True

    pid_kp: float = 3.0                   # pid_cartesian: m/s of tip vel per m of error
    pid_ki: float = 0.6
    pid_kd: float = 0.05
    pid_integral_limit_m: float = 0.02
    pid_jv_damping: float = 1.0e-3        # damped pinv of the 3x6 robot Jacobian
    pid_kp_insertion: float = 0.8         # insertion rate (m/s) per m of AXIAL (B.x) tip error

    # --- advancer (insertion) -------------------------------------
    # control_insertion (above): steer the tip's axial (B.x) position with the
    # advancer.  advancer_dry_run keeps it a software integral (no serial writes)
    # even when steering.
    advancer_port: str = "/dev/ttyACM0"
    advancer_dry_run: bool = False

    # --- beam frame B in R (kept in sync with robotics_frame_measurement_validation) --
    # beam_axial_axis_R  : direction the beam grows (it stands up -> +R.z).
    # beam_plane_normal_axis_R : the CAMERA viewing axis == B.z (out of the 2D
    #   image plane).  To identify it physically: jog the magnet purely along
    #   ONE robot axis; the axis whose motion does NOT move the beam tip in the
    #   2D image is the camera axis.  A wrong choice makes an in-plane magnet
    #   move look like an out-of-plane one (your "why did B.z change" question).
    beam_axial_axis_R: tuple[float, float, float] = (0.0, 0.0, 1.0)
    beam_plane_normal_axis_R: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    # --- beam Jacobian d(tip)/d[q1..q6, insertion], frozen at the start pose --
    #   "analytical_beam"  : real analytic beam sensitivity
    #     (MagneticBeamForwardModel.jacobian_tip_actuation_tangent -> d(tip)/d(magnet
    #      world translation) ) composed with the UR translational Jacobian.
    #   "kinematic_scalar" : UR translational Jacobian * magnet_tip_coupling
    #     (crude fallback, no magnetics/elasticity).
    jacobian_source: str = "analytical_beam"
    magnet_tip_coupling: float = 1.0       # only used by "kinematic_scalar"
    insertion_axial_gain: float = 1.0      # d(tip)/d(insertion) for "kinematic_scalar"
    dipole_unit_in_magnet_body: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # --- limits ---------------------------------------------------
    joint_velocity_limit_rad_s: float = 0.10
    joint_acceleration_limit_rad_s2: float = 0.40
    insertion_rate_limit_m_s: float = 2.0e-3
    speed_j_acceleration_rad_s2: float = 0.8

    workspace_xyz_min_m: tuple[float, float, float] = (0.20, -1.20, -0.30)
    workspace_xyz_max_m: tuple[float, float, float] = (1.10, -0.20, 0.70)

    # --- robot / vision (mirrors StateStreamConfig) --------------
    robot_ip: str = "192.168.56.101"
    reader_poll_hz: float = 60.0            # receive-only joint/pose poll rate
    robot_max_age_s: float = 0.15
    initial_insertion_m: float = 0.040

    cam_index: int = 0
    exposure: float = 27.0
    gain: float = 0.0
    grab_period_s: float = 0.004
    reconstruct_period_s: float = 0.01
    image_filename: str = "/dev/shm/proper_closeloop_frame.png"

    output_root: str = "close_loop_logs"
    run_name: str = "close_loop_tip"


CONFIG = CloseLoopConfig()


# =============================================================================
# beam Jacobian provider: frozen UR translational Jacobian + one coupling
# =============================================================================


def _translational_jacobian_fd(robot: Any, q: np.ndarray, eps: float = 1.0e-4) -> np.ndarray:
    """d(TCP position)/d(joints) (3x6) by finite-differencing forward kinematics.

    ``getJacobian`` needs RTDE output registers this controller does not
    configure; ``getForwardKinematics`` always works.  ~112 ms for the 7 FK
    calls -> compute this ONCE, before the streaming thread starts, and hold it
    frozen (a local Jacobian, like ``mpc_lti``).
    """

    q = np.asarray(q, dtype=float).reshape(6)
    try:
        p0 = np.asarray(robot.get_forward_kinematics(q), dtype=float)[:3]
        jac = np.zeros((3, 6), dtype=float)
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            jac[:, i] = (np.asarray(robot.get_forward_kinematics(dq), dtype=float)[:3] - p0) / eps
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"getForwardKinematics failed ({exc!r}). The arm is probably in a "
            "protective stop or a program is running -- clear it on the pendant "
            "(Remote Control), then re-run."
        ) from exc
    return jac


class SimpleBeamJacobianProvider:
    """``z=[q1..q6, insertion] -> 3x7`` d(tip_xyz)/d(state) in the robot base frame."""

    def __init__(
        self,
        *,
        robot: Any,
        q0: np.ndarray,
        beam_axis_R: np.ndarray,
        magnet_tip_coupling: float,
        insertion_axial_gain: float,
    ) -> None:
        self._beam_axis_R = np.asarray(beam_axis_R, dtype=float).reshape(3)
        self._beam_axis_R /= float(np.linalg.norm(self._beam_axis_R)) + 1e-12
        self._coupling = float(magnet_tip_coupling)
        self._axial_gain = float(insertion_axial_gain)
        self.calls = 0
        self._jv = _translational_jacobian_fd(robot, np.asarray(q0, dtype=float)[:6])
        self.last_condition = float(np.linalg.cond(self._jv))

    def __call__(self, state: Any) -> np.ndarray:
        matrix = np.zeros((3, 7), dtype=float)
        matrix[:, :6] = self._coupling * self._jv
        matrix[:, 6] = self._axial_gain * self._beam_axis_R
        self.calls += 1
        return matrix

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "SimpleBeamJacobianProvider",
            "source": "frozen FD-FK translational Jacobian + scalar magnet/tip coupling",
            "contact_used_in_jacobian": False,
            "magnet_tip_coupling": self._coupling,
            "insertion_axial_gain": self._axial_gain,
        }


class AnalyticalBeamJacobianProvider:
    """Frozen ``d(tip_xyz)/d[q1..q6, insertion]`` from the analytic beam model.

    Builds a ``MagneticBeamForwardModel`` at the operating point (same machinery
    as ``robotics_frame_measurement_validation``), takes its analytic tip
    sensitivity to the magnet's world translation and to insertion
    (``jacobian_tip_actuation_tangent``), and composes the translation part with
    the UR translational Jacobian:

        J[:, :6] = J_beam[:, 0:3] @ Jv_robot        (3x6)
        J[:, 6]  = J_beam[:, 6]                     (insertion)

    Computed once, frozen (a local linearisation).  The magnet-rotation coupling
    (J_beam columns 3..5) is dropped -- fine for a small translation-only move.
    """

    def __init__(
        self,
        *,
        robot: Any,
        q0: np.ndarray,
        tcp_pose6: np.ndarray,
        insertion_m: float,
        mapper: Any,
        dipole_unit_in_magnet_body: tuple[float, float, float],
    ) -> None:
        from proper_research.hardware import robotics_frame_measurement_validation as rfmv
        import proper_research.hardware.beam_hardware_experiment_v2 as experiment_v2

        base = experiment_v2._base_module()
        base.configure_bounds_beam_paths(experiment_v2.CONFIG)

        # Use the SAME configured instance the validation script uses, so the
        # magnet offset / frame match whatever was tuned there.
        fv_cfg = rfmv.CONFIG
        T_R_TCP, T_TCP_M, T_R_M = rfmv.source_transform_from_tcp(
            np.asarray(tcp_pose6, dtype=float).reshape(6), fv_cfg
        )
        self.magnet_in_B_m = mapper.T_R_B.inverse().apply_points(T_R_M.translation)
        lumen = rfmv.build_lumen_in_shared_frames(
            mapper.manual_frame, mapper.calibration, mapper.T_R_B
        )
        # CRITICAL: the Cosserat model grows its rod along  R0 @ [-1, 0, 0].
        # With R0 = T_R_B.rotation (B.x = +R.z) that points DOWN, i.e. the model
        # beam is upside-down vs reality and its d(tip)/d(magnet) has the wrong
        # structure -> the controller diverges.  Rebuild R0 so the rod grows
        # along the real beam axis (+B.x), same fix as
        # robotics_frame_measurement_validation's align_model_base_with_measured_beam.
        beam_axial_R = np.asarray(mapper.T_R_B.rotation[:, 0], dtype=float)
        model_base_rotation_R = rfmv.model_base_rotation_from_beam(
            beam_axial_R, mapper.T_R_B.rotation[:, 2]
        )
        adapter, _meta = rfmv.build_forward_model_in_shared_frame(
            experiment_cfg=experiment_v2.CONFIG,
            T_R_B=mapper.T_R_B,
            lumen=lumen,
            model_base_rotation_R=model_base_rotation_R,
            dipole_unit_in_magnet_body=dipole_unit_in_magnet_body,
        )
        p8 = rfmv.transform_to_p8(T_R_M, float(insertion_m))
        p7, output = adapter.commit_nominal(p8)
        j_beam = np.asarray(
            adapter.raw_model.jacobian_tip_actuation_tangent(p7), dtype=float
        ).reshape(3, 7)

        # Sanity: where does the MODEL put the tip, vs where vision sees it?
        # (both in the beam frame B).  If these disagree a lot the frame or the
        # model orientation is wrong, not just the gain.
        model_tip_R = np.asarray(output, dtype=float).reshape(-1)[:3]
        self.model_tip_B_m = mapper.T_R_B.inverse().apply_points(model_tip_R)
        self.magnet_minus_base_B_m = self.magnet_in_B_m  # base is the B origin

        jv = _translational_jacobian_fd(robot, np.asarray(q0, dtype=float)[:6])
        self._jac = np.zeros((3, 7), dtype=float)
        self._jac[:, :6] = j_beam[:, 0:3] @ jv
        self._jac[:, 6] = j_beam[:, 6]
        self.j_beam_full = j_beam.copy()
        self.calls = 0
        self.last_condition = float(np.linalg.cond(self._jac[:, :6]))
        self.j_beam_translation = j_beam[:, 0:3].copy()
        self.j_beam_insertion = j_beam[:, 6].copy()
        self.source_position_R_m = np.asarray(T_R_M.translation, dtype=float)

    def __call__(self, state: Any) -> np.ndarray:
        self.calls += 1
        return self._jac.copy()

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "AnalyticalBeamJacobianProvider",
            "source": "MagneticBeamForwardModel.jacobian_tip_actuation_tangent @ Jv_robot",
            "contact_used_in_jacobian": False,
        }


# =============================================================================
# static-target reference + minimal MPC config for the controller
# =============================================================================


def _static_reference(target_R_m: np.ndarray, z0: np.ndarray, dt: float) -> Any:
    from proper_research.simulation.simulations import (
        simulate_time_parameterized_configuration_mpc as base_module,
    )

    target = np.asarray(target_R_m, dtype=float).reshape(3)
    z0 = np.asarray(z0, dtype=float).reshape(7)
    return base_module.ConfigurationReference(
        time_s=np.array([0.0, dt], dtype=float),
        path_coordinate_m=np.zeros(2),
        state=np.vstack([z0, z0]),
        input=np.zeros((2, 7)),
        acceleration=np.zeros((2, 7)),
        desired_position_m=np.vstack([target, target]),
        desired_tangent=np.tile(np.array([0.0, 0.0, 1.0]), (2, 1)),
        planned_beam_feasible=np.ones(2, dtype=bool),
        planned_position_error_m=np.zeros(2),
        planned_tangent_error_rad=np.zeros(2),
        sample_period_s=float(dt),
        source="close_loop_static_target",
    )


def _mpc_config(cfg: CloseLoopConfig, dt: float) -> Any:
    from proper_research.simulation.simulations import (
        simulate_time_parameterized_configuration_mpc as base_module,
    )

    vlim = tuple([cfg.joint_velocity_limit_rad_s] * 6 + [cfg.insertion_rate_limit_m_s])
    alim = tuple(
        [cfg.joint_acceleration_limit_rad_s2] * 6 + [10.0 * cfg.insertion_rate_limit_m_s]
    )
    return base_module.ConfigurationMPCConfig(
        sample_period_s=float(dt),
        prediction_horizon=1,
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]),
        state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=vlim,
        acceleration_limit=alim,
    )


def _make_output_dir(cfg: CloseLoopConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# main
# =============================================================================


def main() -> None:
    import sys

    cfg = CONFIG
    dt = 1.0 / cfg.control_hz
    real_stdout = sys.stdout
    sys.stdout = _DebugLineFilter(real_stdout)  # drop the vision pipeline's [DEBUG] spam
    output_dir = _make_output_dir(cfg)
    log_path = output_dir / "close_loop.jsonl"

    mode = "DRY-RUN (no motion)" if cfg.dry_run else "LIVE (servoJ per tick)"
    print(f"[loop] {mode}  control @ {cfg.control_hz:.1f} Hz")
    print(f"[loop] log -> {log_path}")

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
        f"[loop] T_R_B columns  B.x(axial)={np.round(mapper.T_R_B.rotation[:, 0], 3).tolist()}  "
        f"B.y(in-plane sideways)={np.round(mapper.T_R_B.rotation[:, 1], 3).tolist()}  "
        f"B.z(camera/out-of-plane)={np.round(mapper.T_R_B.rotation[:, 2], 3).tolist()}\n"
        f"[loop]   -> moving the TCP along R.x changes "
        f"{'B.z (OUT of the vision plane)' if abs(mapper.T_R_B.rotation[0, 2]) > 0.7 else 'B.y (in-plane)' if abs(mapper.T_R_B.rotation[0, 1]) > 0.7 else 'B.x (axial)'}"
    )

    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    # Two separate robot connections on purpose:
    #  * URRTDERobot (control) -- only servo_j() / forward-kinematics use it,
    #    from the single control-loop thread (no separate streaming thread).
    #  * RobotJointStream (receive-only) -- a bare rtde_receive poll for joints
    #    and pose, feeding the camera and the control loop.  If CameraSource
    #    called URRTDERobot.get_joints() every frame it would grab the control
    #    lock mid-cycle and the RT thread would fall behind -> C271A1.
    robot = URRTDERobot(cfg.robot_ip, frequency=500.0)  # e-series controller rate for servoJ
    reader = RobotJointStream(cfg.robot_ip, poll_hz=cfg.reader_poll_hz, receive_only=True)

    def _assert_robot_ready() -> None:
        mode_ = robot.get_robot_mode()
        safety = robot.get_safety_mode()
        pstop = robot.is_protective_stopped()
        if pstop or mode_ != 7 or safety not in (1, 2):
            raise RuntimeError(
                f"robot not ready (mode={mode_}, safety={safety}, protective_stop={pstop}). "
                "Clear any protective stop on the pendant (Remote Control), make sure no "
                "program is running, then retry."
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

    servo_commands = {"sent": 0}

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
            f"[loop] advancer {'(dry-run)' if cfg.advancer_dry_run else 'LIVE'} "
            f"on {cfg.advancer_port} -- steering the tip axial (B.x) position"
        )

    stop_flag = {"stop": False}
    prev_handler = signal.signal(
        signal.SIGINT, lambda *_: stop_flag.__setitem__("stop", True)
    )

    steps = 0
    converged_ticks = 0
    abort_reason = ""
    started_moving = False
    jac_provider: Any = None
    log_file = log_path.open("w", encoding="utf-8")
    # Flat CSV for graphing / post-processing.
    traj_file = (output_dir / "tip_trajectory.csv").open("w", encoding="utf-8", newline="")
    traj_file.write(
        "step,t_s,tip_x_m,tip_y_m,tip_z_m,target_x_m,target_y_m,target_z_m,"
        "err_norm_mm,q1,q2,q3,q4,q5,q6,insertion_m,converged\n"
    )
    t_zero = now_monotonic()

    try:
        _assert_robot_ready()
        reader.start()
        if advancer is not None:
            advancer.start()
        camera.start()
        print("[loop] camera + reader up; waiting for first tip + joints...")

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
        while len(start_tips) < cfg.settle_ticks_before_target and not stop_flag["stop"]:
            est, _age = camera.latest(cfg.max_state_age_s)
            if est is not None:
                start_tips.append(np.asarray(est.tip_position_m, dtype=float))
            time.sleep(dt)
        start_tip = np.mean(np.vstack(start_tips), axis=0)
        target_R = start_tip + np.asarray(cfg.target_offset_R_mm, dtype=float) / 1.0e3
        q0 = np.asarray(estimate.robot_joints, dtype=float).reshape(6)
        z0 = np.concatenate([q0, [insertion_m]])
        print(
            f"[loop] start tip (mm)  = {np.round(1e3 * start_tip, 2).tolist()}\n"
            f"[loop] target tip (mm) = {np.round(1e3 * target_R, 2).tolist()}  "
            f"(offset {cfg.target_offset_R_mm} mm, tol {cfg.tolerance_mm} mm)"
        )

        # Frozen local Jacobian -- built before the control loop, so its FK requests
        # (and the one-time forward-model solve) never contend with the 125 Hz
        # streaming thread.
        if cfg.jacobian_source == "analytical_beam":
            print("[loop] building analytic beam Jacobian (forward-model solve + FK)...")
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
            m_B = 1e3 * jac_provider.magnet_in_B_m
            base_R = np.asarray(mapper.T_R_B.translation, dtype=float)
            m_minus_base_R = 1e3 * (jac_provider.source_position_R_m - base_R)
            print(
                f"[loop] MAGNET position (all relative to the beam BASE):\n"
                f"[loop]   in beam frame B [B.x along beam, B.y sideways, B.z out-of-plane]\n"
                f"[loop]     = [{m_B[0]:+.0f}, {m_B[1]:+.0f}, {m_B[2]:+.0f}] mm   "
                f"<- your spec was [300, 0, 0]; B.z=0 means IN the 2D vision plane\n"
                f"[loop]   in robot base R [R.x, R.y, R.z]\n"
                f"[loop]     = [{m_minus_base_R[0]:+.0f}, {m_minus_base_R[1]:+.0f}, {m_minus_base_R[2]:+.0f}] mm   "
                f"<- R.z=+{m_minus_base_R[2]:.0f} because the beam grows along +R.z (it stands up),\n"
                f"[loop]        so '300 mm along the beam' == '300 mm higher in R.z'. Same point.\n"
                f"[loop]   model tip in B = "
                f"{np.round(1e3 * jac_provider.model_tip_B_m, 1).tolist()} mm   "
                f"measured tip in B = {np.round(1e3 * mapper.T_R_B.inverse().apply_points(start_tip), 1).tolist()} mm\n"
                f"[loop] |d(tip)/d(magnet_xyz)| rows (R.x/R.y/R.z) = "
                f"{np.round(np.linalg.norm(jac_provider.j_beam_translation, axis=1), 3).tolist()} "
                f"mm/mm  <- coaxial magnet at 300 mm has almost no bending authority"
            )
        else:
            print("[loop] building kinematic-scalar Jacobian (7 FK calls)...")
            jac_provider = SimpleBeamJacobianProvider(
                robot=robot,
                q0=q0,
                beam_axis_R=beam_axis_R,
                magnet_tip_coupling=cfg.magnet_tip_coupling,
                insertion_axial_gain=cfg.insertion_axial_gain,
            )
        print(f"[loop] cond(J[:, :6]) = {jac_provider.last_condition:.2f}")
        j_full = jac_provider(z0)
        with np.printoptions(precision=5, suppress=True, linewidth=160):
            print("[loop] ========= FROZEN BEAM JACOBIAN  d(tip_R xyz)/d[q1..q6, insertion] =========")
            print(f"[loop] rows = tip R.x, R.y, R.z   cols = q1..q6 (rad), insertion (m)")
            for r, name in enumerate(("R.x", "R.y", "R.z")):
                print(f"[loop]   {name}: {j_full[r]}")
            if isinstance(jac_provider, AnalyticalBeamJacobianProvider):
                print("[loop] --- factors: J = J_beam[:, 0:3] @ Jv_robot ,  J[:,6] = J_beam[:,6] ---")
                print(f"[loop] J_beam (3x7) d(tip)/d[magnet_xyz(3), magnet_rot(3), insertion]:")
                for r, name in enumerate(("R.x", "R.y", "R.z")):
                    print(f"[loop]   {name}: {jac_provider.j_beam_full[r]}")
            print("[loop] ============================================================================")
        print(
            f"[loop] frozen J rows (R.x/R.y/R.z) |.| = "
            f"{np.round(np.linalg.norm(j_full[:, :6], axis=1), 4).tolist()}  "
            f"(target is in R.y = beam B.y)"
        )

        # robot translational Jacobian, frozen (used by pid_cartesian and for diag)
        jv_robot = _translational_jacobian_fd(robot, q0)
        with np.printoptions(precision=5, suppress=True, linewidth=160):
            print(f"[loop] Jv_robot (3x6) d(TCP_pos)/d(q):")
            for r, name in enumerate(("R.x", "R.y", "R.z")):
                print(f"[loop]   {name}: {jv_robot[r]}")

        controller = None
        if cfg.controller_type == "jacobian":
            from proper_research.controllers.inverse_jacobian_controller import (
                build_inverse_jacobian_controller,
            )

            controller = build_inverse_jacobian_controller(
                reference=_static_reference(target_R, z0, dt),
                jacobian_provider=jac_provider,
                mpc_config=_mpc_config(cfg, dt),
                position_gain=cfg.position_gain,
                damping=cfg.damping,
                nullspace_gain=cfg.nullspace_gain,
                feedforward=False,
                allow_undeclared_jacobian=True,
            )
        elif cfg.controller_type == "pid_cartesian":
            jv_pinv = jv_robot.T @ np.linalg.inv(
                jv_robot @ jv_robot.T + (cfg.pid_jv_damping**2) * np.eye(3)
            )  # (6, 3) damped pseudo-inverse of d(TCP_pos)/d(q)
        else:
            raise ValueError(f"unknown controller_type {cfg.controller_type!r}")

        print(f"[loop] controller = {cfg.controller_type}; servoJ control; closing the loop")

        u_prev = np.zeros(7, dtype=float)
        pid_integral = np.zeros(3, dtype=float)
        pid_prev_error = None
        safety_check_every = max(1, int(round(cfg.control_hz)))  # ~1 Hz
        next_tick = now_monotonic()

        while not stop_flag["stop"] and steps < cfg.max_control_steps:
            # servoJ is non-blocking, so pace the loop here.
            now = now_monotonic()
            if next_tick > now:
                time.sleep(min(next_tick - now, dt))
                continue
            next_tick = max(next_tick + dt, now)

            if not camera.healthy:
                abort_reason = f"camera_unhealthy: {camera.last_error!r}"
                break
            # Slow safety supervision (receive-only connection, no RT contention).
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

            tip = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)
            error = target_R - tip
            if cfg.project_error_to_beam_plane:
                b_z = np.asarray(mapper.T_R_B.rotation[:, 2], dtype=float)
                error = error - float(np.dot(error, b_z)) * b_z
            error_mm = 1.0e3 * float(np.linalg.norm(error))
            q = np.asarray(estimate.robot_joints, dtype=float).reshape(6)
            z = np.concatenate([q, [insertion_m]])

            predicted_err_mm = float("nan")
            if cfg.controller_type == "jacobian":
                step = controller.solve(
                    measured_state=z,
                    measured_beam_position=tip,
                    control_index=0,
                    previous_input=u_prev,
                )
                command = np.asarray(step.command, dtype=float).reshape(7)
                predicted_err_mm = 1e3 * float(step.first_predicted_beam_error_m)
            else:  # pid_cartesian
                b_x = np.asarray(mapper.T_R_B.rotation[:, 0], dtype=float)  # beam axial
                e_axial = float(np.dot(error, b_x))                         # tip error along the beam
                e_lateral = error - e_axial * b_x                           # the part the magnet can steer
                if pid_prev_error is None:
                    pid_prev_error = e_lateral.copy()
                pid_integral = np.clip(
                    pid_integral + e_lateral * dt,
                    -cfg.pid_integral_limit_m,
                    cfg.pid_integral_limit_m,
                )
                deriv = (e_lateral - pid_prev_error) / dt
                pid_prev_error = e_lateral.copy()
                v_tip_R = (
                    cfg.pid_kp * e_lateral + cfg.pid_ki * pid_integral + cfg.pid_kd * deriv
                )  # desired lateral tip velocity in R (m/s) == desired magnet velocity
                command = np.zeros(7, dtype=float)
                command[:6] = jv_pinv @ v_tip_R
                # axial (B.x) error -> advancer insertion rate.
                if cfg.control_insertion:
                    command[6] = float(
                        np.clip(
                            cfg.pid_kp_insertion * e_axial,
                            -cfg.insertion_rate_limit_m_s,
                            cfg.insertion_rate_limit_m_s,
                        )
                    )

            if not cfg.control_insertion:
                command[6] = 0.0
            if not np.all(np.isfinite(command)):
                abort_reason = "nonfinite_command"
                break

            converged = error_mm <= cfg.tolerance_mm
            converged_ticks = converged_ticks + 1 if converged else 0
            if converged:
                command = np.zeros(7, dtype=float)  # hold in place
                pid_integral[:] = 0.0

            # advancer: integrate the insertion-rate command and dispatch it.
            if advancer is not None:
                advancer.submit_rate(float(command[6]), dt)
                insertion_m = float(
                    np.clip(insertion_m + float(command[6]) * dt, 0.005, 0.20)
                )

            # velocity command -> per-tick joint target, clamped hard.
            qd = np.clip(
                command[:6],
                -cfg.joint_velocity_limit_rad_s,
                cfg.joint_velocity_limit_rad_s,
            )
            delta_q = np.clip(qd * dt, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
            q_target = q + delta_q

            t_servo = now_monotonic()
            if not cfg.dry_run:
                # servoJ is non-blocking; time_s holds the target past the next
                # call, lookahead_time smooths the gaps.
                robot.servo_j(
                    q_target,
                    time_s=2.0 * dt,
                    lookahead_time=cfg.servo_lookahead_s,
                    gain=cfg.servo_gain,
                )
                servo_commands["sent"] += 1
                if np.any(np.abs(delta_q) > 1e-9):
                    started_moving = True
            servo_ms = 1e3 * (now_monotonic() - t_servo)

            u_prev = command.copy()
            steps += 1

            row = {
                "step": steps,
                "t_monotonic": now_monotonic(),
                "state_age_s": round(float(age), 4),
                "tip_mm": [round(float(v), 3) for v in 1e3 * tip],
                "target_mm": [round(float(v), 3) for v in 1e3 * target_R],
                "error_mm": [round(float(v), 3) for v in 1e3 * error],
                "error_norm_mm": round(error_mm, 3),
                "qd_cmd_rad_s": [round(float(v), 5) for v in command[:6]],
                "qd_cmd_norm_rad_s": round(float(np.linalg.norm(command[:6])), 5),
                "insertion_rate_cmd_m_s": round(float(command[6]), 6),
                "insertion_length_m": round(float(insertion_m), 5),
                "predicted_tip_error_mm": round(predicted_err_mm, 3),
                "converged": bool(converged),
                "q_target_delta_rad": [round(float(v), 6) for v in delta_q],
                "servo_ms": round(servo_ms, 1),
                "servo_commands_sent": servo_commands["sent"],
                "dry_run": cfg.dry_run,
            }
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            traj_file.write(
                f"{steps},{now_monotonic()-t_zero:.4f},"
                f"{tip[0]:.6f},{tip[1]:.6f},{tip[2]:.6f},"
                f"{target_R[0]:.6f},{target_R[1]:.6f},{target_R[2]:.6f},"
                f"{error_mm:.3f},"
                + ",".join(f"{v:.6f}" for v in q)
                + f",{insertion_m:.6f},{int(converged)}\n"
            )
            traj_file.flush()

            if steps % 5 == 0 or converged:
                ins_txt = (
                    f"ins={1e3*insertion_m:5.1f}mm(dL={1e3*command[6]:+5.2f}) "
                    if advancer is not None
                    else ""
                )
                print(
                    f"[{steps:4d}] |e|={error_mm:6.2f}mm "
                    f"e_mm=[{1e3*error[0]:+6.2f} {1e3*error[1]:+6.2f} {1e3*error[2]:+6.2f}] "
                    f"|qd|={np.linalg.norm(command[:6]):.4f}rad/s {ins_txt}"
                    f"{'CONVERGED' if converged else ''}"
                )

            if converged_ticks >= cfg.converged_hold_ticks:
                abort_reason = abort_reason or "converged"
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
                print(f"[loop] servo_stop error: {exc!r}")
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

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "stop_reason": abort_reason or ("stopped" if stop_flag["stop"] else "done"),
        "control_steps": steps,
        "started_moving": started_moving,
        "jacobian_condition": (
            None if jac_provider is None else round(jac_provider.last_condition, 2)
        ),
        "servo_commands_sent": servo_commands["sent"],
        "vision_detection_failures": mapper.detection_failures,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== CLOSE-LOOP SUMMARY ===")
    for key, value in summary.items():
        if key == "config":
            continue
        print(f"  {key:24s}: {value}")
    print(f"  log                     : {log_path}")


if __name__ == "__main__":
    main()
