"""Joint-space controller factory for the magnetic-beam simulator.

This module replaces only the pose-space ``build_controller`` path.  The beam
models are unchanged: they still receive a source-magnet pose and insertion
length.  The adapter inserts robot forward kinematics before the beam model and
the robot geometric Jacobian before the beam sensitivity.

Controller state and input
--------------------------
    state   = [q1, q2, q3, q4, q5, q6, insertion_length]
    control = [qd1, qd2, qd3, qd4, qd5, qd6, insertion_rate]

The Jacobian callback returned here is continuous.  ``MPCControllerTipXY``
forms the discrete stage matrix exactly once as ``B_k = dt * J_k``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations import controller_factory as legacy
from proper_research.simulation.simulations.joint_space_beam_mpc_adapter import (
    JointSpaceBeamMPCAdapter,
    controller_callbacks,
)


Array = np.ndarray


@dataclass(frozen=True)
class JointSpaceRobotConfig:
    """Robot, tool and joint-space limits used by the simulator.

    ``None`` means "reuse the corresponding value from
    ur_magnet_ik_jacobian_validation.CONFIG".  This keeps the simulator and
    the validated robot/magnet frame model on the same DH parameters and tool
    offset.
    """

    q_seed_rad: tuple[float, ...] | None = None
    active_tcp_pose6: tuple[float, ...] | None = None
    tcp_to_magnet_pose6: tuple[float, ...] | None = None
    joint_lower_bounds_rad: tuple[float, ...] | None = None
    joint_upper_bounds_rad: tuple[float, ...] | None = None
    insertion_min_m: float = 0.0
    insertion_max_m: float = 0.20
    debug_initial_ik: bool = True
    validate_chain_rule_at_start: bool = False
    chain_rule_joint_step_rad: float = 1.0e-6
    chain_rule_insertion_step_m: float = 1.0e-6

    def validate(self) -> None:
        for name in (
            "q_seed_rad",
            "active_tcp_pose6",
            "tcp_to_magnet_pose6",
            "joint_lower_bounds_rad",
            "joint_upper_bounds_rad",
        ):
            value = getattr(self, name)
            if value is not None:
                expected = 6
                array = np.asarray(value, dtype=float).reshape(-1)
                if array.size != expected or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must contain {expected} finite values.")
        if not np.isfinite(self.insertion_min_m):
            raise ValueError("insertion_min_m must be finite.")
        if not np.isfinite(self.insertion_max_m):
            raise ValueError("insertion_max_m must be finite.")
        if self.insertion_min_m >= self.insertion_max_m:
            raise ValueError("insertion_min_m must be below insertion_max_m.")
        if self.chain_rule_joint_step_rad <= 0.0:
            raise ValueError("chain_rule_joint_step_rad must be positive.")
        if self.chain_rule_insertion_step_m <= 0.0:
            raise ValueError("chain_rule_insertion_step_m must be positive.")


@dataclass(frozen=True)
class _ResolvedRobotKinematics:
    ik_cfg: Any
    dh: Any
    T_F_M: Array
    q_seed_rad: Array
    q_lower_rad: Array
    q_upper_rad: Array


def _finite6(value: Any, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 6 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values.")
    return result.copy()


def _resolve_robot_kinematics(
    robot_cfg: JointSpaceRobotConfig,
) -> _ResolvedRobotKinematics:
    robot_cfg.validate()
    source = urik.CONFIG
    ik_cfg = replace(source)
    ik_cfg.debug_enabled = bool(robot_cfg.debug_initial_ik)

    q_seed = _finite6(
        source.offline_actual_q_rad
        if robot_cfg.q_seed_rad is None
        else robot_cfg.q_seed_rad,
        "q_seed_rad",
    )
    q_lower = _finite6(
        source.joint_lower_bounds_rad
        if robot_cfg.joint_lower_bounds_rad is None
        else robot_cfg.joint_lower_bounds_rad,
        "joint_lower_bounds_rad",
    )
    q_upper = _finite6(
        source.joint_upper_bounds_rad
        if robot_cfg.joint_upper_bounds_rad is None
        else robot_cfg.joint_upper_bounds_rad,
        "joint_upper_bounds_rad",
    )
    if np.any(q_lower >= q_upper):
        raise ValueError("Every lower joint limit must be below its upper limit.")
    ik_cfg.joint_lower_bounds_rad = tuple(q_lower)
    ik_cfg.joint_upper_bounds_rad = tuple(q_upper)

    active_tcp_pose6 = _finite6(
        source.offline_active_T_flange_tcp_pose6
        if robot_cfg.active_tcp_pose6 is None
        else robot_cfg.active_tcp_pose6,
        "active_tcp_pose6",
    )

    if robot_cfg.tcp_to_magnet_pose6 is None:
        if bool(source.assume_tcp_is_magnet_frame):
            tcp_to_magnet_pose6 = np.zeros(6, dtype=float)
        elif source.T_tcp_magnet_pose6 is None:
            raise ValueError(
                "No TCP-to-magnet transform is configured. Set "
                "JointSpaceRobotConfig.tcp_to_magnet_pose6 or update the "
                "validated UR configuration."
            )
        else:
            tcp_to_magnet_pose6 = _finite6(
                source.T_tcp_magnet_pose6, "T_tcp_magnet_pose6"
            )
    else:
        tcp_to_magnet_pose6 = _finite6(
            robot_cfg.tcp_to_magnet_pose6, "tcp_to_magnet_pose6"
        )

    T_F_TCP = urik.pose6_to_T(active_tcp_pose6)
    T_TCP_M = urik.pose6_to_T(tcp_to_magnet_pose6)
    T_F_M = urik.validate_transform(T_F_TCP @ T_TCP_M, "T_F_M")

    return _ResolvedRobotKinematics(
        ik_cfg=ik_cfg,
        dh=urik.corrected_dh_from_config(ik_cfg),
        T_F_M=T_F_M,
        q_seed_rad=q_seed,
        q_lower_rad=q_lower,
        q_upper_rad=q_upper,
    )


def _pose8_from_transform(T_R_M: Any, insertion_m: float) -> Array:
    T_R_M = urik.validate_transform(T_R_M, "T_R_M")
    q_xyzw = Rot.from_matrix(T_R_M[:3, :3]).as_quat()
    q_wxyz = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float
    )
    return np.concatenate(
        (T_R_M[:3, 3], q_wxyz, np.array([float(insertion_m)]))
    )


def _pose7_from_transform(T_R_M: Any, insertion_m: float) -> Array:
    pose6 = urik.T_to_pose6(urik.validate_transform(T_R_M, "T_R_M"))
    return np.concatenate((pose6, np.array([float(insertion_m)])))


def _make_beam_forward_callback(model):
    """Wrap the existing pose8 ControllerForwardAdapter with a transform API."""
    forward_adapter = legacy.ControllerForwardAdapter(model)
    forward_adapter.start_step()

    def beam_output(T_R_M, insertion_m, *, commit=False):
        p8 = _pose8_from_transform(T_R_M, insertion_m)
        return np.asarray(
            forward_adapter(p8, commit=bool(commit)), dtype=float
        ).reshape(-1)

    beam_output.start_step = forward_adapter.start_step
    if callable(getattr(forward_adapter, "reset", None)):
        beam_output.reset = forward_adapter.reset
    beam_output.capture_cache_state = forward_adapter.capture_cache_state
    beam_output.restore_cache_state = forward_adapter.restore_cache_state
    beam_output.get_baseline_cache_copy = (
        forward_adapter.get_baseline_cache_copy
    )
    beam_output.set_baseline_cache = forward_adapter.set_baseline_cache
    beam_output.reset_to_initial_baseline = (
        forward_adapter.reset_to_initial_baseline
    )
    beam_output.forward_adapter = forward_adapter
    beam_output.model = model
    return beam_output, forward_adapter


def _make_beam_jacobian_callback(
    model,
    *,
    forward_adapter,
    jacobian_mode: str,
):
    """Return d[tip,tangent]/d[world translation, world rotation,L]."""
    if jacobian_mode not in {"fast", "accurate"}:
        raise ValueError("jacobian_mode must be 'fast' or 'accurate'.")

    def beam_jacobian(T_R_M, insertion_m):
        p7 = _pose7_from_transform(T_R_M, insertion_m)
        # The forward and Jacobian callbacks deliberately share one model so
        # the derivative is formed at the same equilibrium.  Preserve both
        # caches around this calculation: otherwise a Jacobian request changes
        # the warm start used by the next forward evaluation.
        snapshot = forward_adapter.capture_cache_state()
        try:
            model.set_cache(forward_adapter.get_baseline_cache_copy())
            model.solve(p7, commit=True, reuse_cache=True)

            if hasattr(model, "jacobian_output_actuation_tangent"):
                result = model.jacobian_output_actuation_tangent(
                    p7,
                    solve_if_needed=False,
                    mode=jacobian_mode,
                )
                return np.asarray(result, dtype=float).reshape(6, 7)

            result = model.jacobian_tip_actuation_tangent(
                p7,
                solve_if_needed=False,
                mode=jacobian_mode,
            )
            return np.asarray(result, dtype=float).reshape(3, 7)
        finally:
            forward_adapter.restore_cache_state(snapshot)

    return beam_jacobian


def _validate_design_vectors(design_cfg) -> None:
    if int(design_cfg.n_out) not in {3, 6}:
        raise ValueError("Joint-space beam control requires n_out=3 or n_out=6.")
    if int(design_cfg.n_u) != 7:
        raise ValueError(
            "Set ControllerDesignConfig.n_u=7 for "
            "[qd1..qd6, insertion_rate]."
        )
    required_vectors = {
        "w_u": 7,
        "w_du": 7,
        "u_max": 7,
        "trust_radius": 7,
        "trust_radius_min": 7,
        "trust_radius_max": 7,
    }
    for name, size in required_vectors.items():
        value = np.asarray(getattr(design_cfg, name), dtype=float).reshape(-1)
        if value.size != size or not np.all(np.isfinite(value)):
            raise ValueError(
                f"ControllerDesignConfig.{name} must contain {size} finite "
                "joint/insertion values."
            )

    # The rollout seed is a control rate, not an insertion displacement.  A
    # seed outside its hard input bound can make the input-bound and
    # trust-region boxes disjoint before the first QP solve.
    u_max = np.asarray(design_cfg.u_max, dtype=float).reshape(7)
    dL_guess = float(design_cfg.dL_guess)
    if not np.isfinite(dL_guess):
        raise ValueError("ControllerDesignConfig.dL_guess must be finite.")
    if abs(dL_guess) > u_max[6] + 1.0e-15:
        raise ValueError(
            "ControllerDesignConfig.dL_guess is an insertion rate and must "
            f"satisfy abs(dL_guess) <= u_max[6]; got {dL_guess} m/s and "
            f"u_max[6]={u_max[6]} m/s. Use dL_guess=0.0 for the initial "
            "joint-space smoke test."
        )


def build_controller(
    *,
    start_point: Array,
    L0: float,
    dt: float,
    plant_model,
    jacobian_model,
    lumen_C: Array,
    lumen_R: Array,
    run_cfg,
    design_cfg=None,
    robot_cfg: JointSpaceRobotConfig | None = None,
):
    """Build plant and Jacobian adapters around a joint-space controller."""
    run_cfg.validate()
    design_cfg = design_cfg or legacy.ControllerDesignConfig()
    design_cfg.validate()
    _validate_design_vectors(design_cfg)
    robot_cfg = robot_cfg or JointSpaceRobotConfig()
    robot_cfg.validate()

    dt = float(dt)
    L0 = float(L0)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if not robot_cfg.insertion_min_m <= L0 <= robot_cfg.insertion_max_m:
        raise ValueError(
            f"Initial insertion L0={L0} lies outside "
            f"[{robot_cfg.insertion_min_m}, {robot_cfg.insertion_max_m}]."
        )

    lumen_C = np.asarray(lumen_C, dtype=float)
    lumen_R = np.asarray(lumen_R, dtype=float).reshape(-1)
    if lumen_C.ndim != 2 or lumen_C.shape[0] < 2 or lumen_C.shape[1] < 3:
        raise ValueError("lumen_C must have shape (M, >=3), M >= 2.")
    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError("lumen_R must contain one radius per centreline node.")

    robot = _resolve_robot_kinematics(robot_cfg)
    T_R_M_start = urik.pose6_to_T(
        np.asarray(start_point, dtype=float).reshape(6)
    )
    ik_result = urik.inverse_kinematics_dls(
        T_R_target=T_R_M_start,
        q_seed_rad=robot.q_seed_rad,
        dh=robot.dh,
        T_F_target=robot.T_F_M,
        cfg=robot.ik_cfg,
    )
    if not ik_result.converged:
        raise RuntimeError(
            "Initial magnet-pose IK failed: "
            f"reason={ik_result.reason}, "
            f"position_error={ik_result.final_position_error_m:.6e} m, "
            f"orientation_error={ik_result.final_orientation_error_rad:.6e} rad. "
            "Use a reachable start_point and a nearby q_seed_rad."
        )

    q0 = np.asarray(ik_result.q_rad, dtype=float).reshape(6)
    p0 = np.concatenate((q0, np.array([L0])))
    p_min = np.concatenate(
        (robot.q_lower_rad, np.array([robot_cfg.insertion_min_m]))
    )
    p_max = np.concatenate(
        (robot.q_upper_rad, np.array([robot_cfg.insertion_max_m]))
    )

    def magnet_transform_fn(q6):
        return urik.forward_kinematics(
            q6, robot.dh, robot.T_F_M
        ).T_R_target

    def magnet_geometric_jacobian_fn(q6):
        return urik.geometric_jacobian(q6, robot.dh, robot.T_F_M)

    owned_plant_model = copy.deepcopy(plant_model)
    owned_jacobian_model = copy.deepcopy(jacobian_model)
    plant_beam_output, plant_forward_adapter = _make_beam_forward_callback(
        owned_plant_model
    )
    jacobian_beam_output, jacobian_forward_adapter = _make_beam_forward_callback(
        owned_jacobian_model
    )
    plant_beam_jacobian = _make_beam_jacobian_callback(
        owned_plant_model,
        forward_adapter=plant_forward_adapter,
        jacobian_mode="fast",
    )
    jacobian_beam_jacobian = _make_beam_jacobian_callback(
        owned_jacobian_model,
        forward_adapter=jacobian_forward_adapter,
        jacobian_mode="fast",
    )

    # The MPC may deliberately use only tip position (n_out=3), while the
    # runner always needs the complete [tip_xyz, tangent_xyz] output for
    # geometry plots and tangent-safety diagnostics.  Keep those interfaces
    # separate even though they share the same nonlinear plant/model cache.
    output_indices = tuple(range(int(design_cfg.n_out)))
    plant_adapter = JointSpaceBeamMPCAdapter(
        magnet_transform_fn=magnet_transform_fn,
        magnet_geometric_jacobian_fn=magnet_geometric_jacobian_fn,
        beam_output_fn=plant_beam_output,
        beam_output_jacobian_fn=plant_beam_jacobian,
        output_indices=output_indices,
        model=owned_plant_model,
    )
    plant_diagnostic_adapter = JointSpaceBeamMPCAdapter(
        magnet_transform_fn=magnet_transform_fn,
        magnet_geometric_jacobian_fn=magnet_geometric_jacobian_fn,
        beam_output_fn=plant_beam_output,
        beam_output_jacobian_fn=plant_beam_jacobian,
        output_indices=(0, 1, 2, 3, 4, 5),
        model=owned_plant_model,
    )
    jacobian_adapter = JointSpaceBeamMPCAdapter(
        magnet_transform_fn=magnet_transform_fn,
        magnet_geometric_jacobian_fn=magnet_geometric_jacobian_fn,
        beam_output_fn=jacobian_beam_output,
        beam_output_jacobian_fn=jacobian_beam_jacobian,
        output_indices=output_indices,
        model=owned_jacobian_model,
    )

    plant_callbacks = controller_callbacks(plant_adapter)
    plant_diagnostic_callbacks = controller_callbacks(
        plant_diagnostic_adapter
    )
    common_kwargs = dict(
        Jxy_fn=jacobian_adapter,
        forward_tip_fn=plant_callbacks["forward_tip_fn"],
        magnet_position_fn=plant_callbacks["magnet_position_fn"],
        magnet_position_jacobian_fn=plant_callbacks[
            "magnet_position_jacobian_fn"
        ],
        jacobian_returns_continuous=True,
        dt=dt,
        Np=int(run_cfg.Np),
        n_out=int(design_cfg.n_out),
        n_u=7,
        n_p=7,
        w_xy=np.asarray(design_cfg.w_tracking, dtype=float),
        w_u=np.asarray(design_cfg.w_u, dtype=float),
        w_du=np.asarray(design_cfg.w_du, dtype=float),
        u_max=np.asarray(design_cfg.u_max, dtype=float),
        p_min=p_min,
        p_max=p_max,
        model_mode="ltv",
        solver_mode=str(run_cfg.solver_mode),
        N_sqp=int(run_cfg.N_sqp),
        trust_radius=np.asarray(design_cfg.trust_radius, dtype=float),
        trust_radius_min=np.asarray(design_cfg.trust_radius_min, dtype=float),
        trust_radius_max=np.asarray(design_cfg.trust_radius_max, dtype=float),
        enable_hard_epm_tip_clearance=bool(
            design_cfg.enable_hard_epm_tip_clearance
        ),
        epm_tip_hard_min_m=float(design_cfg.epm_tip_hard_min_m),
        dL_index=6,
        dL_back_max=float(design_cfg.dL_back_max),
        dL_fwd_max=float(design_cfg.dL_fwd_max),
        enable_hard_tip_tangent_angle=bool(
            design_cfg.enable_hard_tip_tangent_angle
        ),
        tip_tangent_max_angle_deg=float(
            design_cfg.tip_tangent_max_angle_deg
        ),
        tip_tangent_nonlinear_tol_deg=float(
            design_cfg.tip_tangent_nonlinear_tol_deg
        ),
        tip_tangent_activation_clearance_m=float(
            design_cfg.tip_tangent_activation_clearance_m
        ),
        tip_tangent_activation_guard_m=float(
            design_cfg.tip_tangent_activation_guard_m
        ),
        tip_radius_m=float(design_cfg.tip_radius_m),
        qp_reg=float(design_cfg.qp_reg),
    )

    if run_cfg.controller_kind == "mpc":
        controller = legacy.MPCControllerTipXY(**common_kwargs)
    elif run_cfg.controller_kind == "inverse_jacobian":
        inverse_class = getattr(legacy, "InverseJacobianControllerTipXY", None)
        if inverse_class is None:
            raise RuntimeError(
                "The inverse-Jacobian controller class is not available from "
                "controller_factory."
            )
        controller = inverse_class(
            **common_kwargs,
            sequence_mode=run_cfg.inverse_sequence_mode,
            kp=run_cfg.inverse_kp,
            desired_step_max=run_cfg.inverse_desired_step_max_m,
            extra_damping=run_cfg.inverse_extra_damping,
        )
        if getattr(controller, "state_layout", None) != "joint_insertion":
            raise RuntimeError(
                "InverseJacobianControllerTipXY is still pose-space. Update "
                "its rollout to state + dt*[qd1..qd6,dL] before using this mode."
            )
    else:
        raise ValueError(f"Unknown controller_kind: {run_cfg.controller_kind!r}.")

    controller.lumen_C = lumen_C[:, :3].copy()
    controller.lumen_R = lumen_R.copy()
    legacy.configure_path_following(controller, design_cfg)
    controller.dL_guess = float(design_cfg.dL_guess)
    controller.set_initial_params(p0)

    chain_rule_validation = None
    if robot_cfg.validate_chain_rule_at_start:
        chain_rule_validation = jacobian_adapter.validate_chain_rule(
            p0,
            joint_step_rad=robot_cfg.chain_rule_joint_step_rad,
            insertion_step_m=robot_cfg.chain_rule_insertion_step_m,
        )

    T_R_M_check = magnet_transform_fn(q0)
    initial_position_error = float(
        np.linalg.norm(T_R_M_check[:3, 3] - T_R_M_start[:3, 3])
    )
    initial_orientation_error = float(
        Rot.from_matrix(
            T_R_M_start[:3, :3] @ T_R_M_check[:3, :3].T
        ).magnitude()
    )

    return {
        "controller": controller,
        "mpc": controller,
        "controller_kind": run_cfg.controller_kind,
        "p0": p0,
        "q0": q0,
        "p_min": p_min,
        "p_max": p_max,
        "u_max": np.asarray(design_cfg.u_max, dtype=float).copy(),
        # Full output for runner visualization/safety, independent of n_out.
        "forward6d_plant": plant_diagnostic_callbacks["forward_tip_fn"],
        "design_cfg": design_cfg,
        "J_fn": jacobian_adapter,
        "plant_joint_adapter": plant_adapter,
        "plant_diagnostic_joint_adapter": plant_diagnostic_adapter,
        "jacobian_joint_adapter": jacobian_adapter,
        "plant_model_owned": owned_plant_model,
        "jacobian_model_owned": owned_jacobian_model,
        "plant_forward_pose8_adapter": plant_forward_adapter,
        "jacobian_forward_pose8_adapter": jacobian_forward_adapter,
        "robot_dh": robot.dh,
        "T_F_M": robot.T_F_M.copy(),
        "initial_ik_result": ik_result,
        "initial_magnet_position_error_m": initial_position_error,
        "initial_magnet_orientation_error_rad": initial_orientation_error,
        "chain_rule_validation": chain_rule_validation,
    }


__all__ = ["JointSpaceRobotConfig", "build_controller"]
