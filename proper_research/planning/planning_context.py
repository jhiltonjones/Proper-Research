"""Shared construction for the offline inverse and global path planners.

This module extracts the active setup from the working experiment script.  It
constructs the same one-entry experiment grid, lumen, beam models, robot
configuration, controller pack, and inverse-planner configuration without
running inverse planning, global optimization, or the online simulation.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from proper_research.planning.offline_inverse_configuration import (
    InverseConfigurationPlannerConfig,
)
from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
)
from proper_research.simulation.simulations.controller_factory_joint_space import (
    JointSpaceRobotConfig,
    build_controller,
)
from proper_research.simulation.simulations.initial_conditions import (
    SOURCE_DIPOLE_BODY_AXIS,
    make_initial_poses,
)
from proper_research.simulation.simulations.model_factory import (
    build_model_bundle,
)
from proper_research.simulation.simulations.scenario import (
    LumenBend,
    LumenConfig,
    make_curvature_jacobian_grid,
)


DEFAULT_RUN_ROOT = Path("uprgrade_configuration")


def make_double_bend_lumen_config(
    first_angle_deg: float = 20.0,
    second_angle_deg: float = -40.0,
) -> LumenConfig:
    """Return the lumen geometry used by the saved inverse-planning run."""
    return LumenConfig(
        length=0.04,
        n_pts=240,
        n_ref_pts=100,
        radius=0.0025,
        ds_target=1.0e-3,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(first_angle_deg),
                bend_start=0.015,
                bend_end=0.02,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(second_angle_deg),
                bend_start=0.02,
                bend_end=0.035,
            ),
        ),
    )


def make_design_config() -> ControllerDesignConfig:
    """Return a fresh controller design configuration for this experiment."""
    return replace(
        ControllerDesignConfig(),
        n_out=3,
        n_p=7,
        w_tracking=(
            1000.0,
            1000.0,
            0.0,
        ),
        u_max=np.array(
            [
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.20,
            ],
            dtype=float,
        ),
        trust_radius=np.array(
            [
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.10,
            ],
            dtype=float,
        ),
        trust_radius_min=np.array(
            [
                0.005,
                0.005,
                0.005,
                0.005,
                0.005,
                0.005,
                0.0001,
            ],
            dtype=float,
        ),
        trust_radius_max=np.array(
            [
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.50,
                0.20,
            ],
            dtype=float,
        ),
        dL_guess=0.0,
        enable_hard_epm_tip_clearance=True,
        enable_hard_tip_tangent_angle=False,
    )


def make_robot_config() -> JointSpaceRobotConfig:
    """Return the UR/insertion configuration matched to the live hardware.

    ``q_seed_rad`` is the live joint vector (see
    ``initial_conditions.LIVE_JOINTS_RAD``); with the offline-FK-consistent
    ``start_point`` from ``make_initial_poses`` the initial magnet-pose IK
    returns exactly those six joint angles.

    ``tcp_to_magnet_pose6 = (0, 0, 0.03, 0, 0, 0)`` -- the source magnet sits
    30 mm below the TCP along TCP.z, the same transform
    ``robotics_frame_measurement_validation.CONFIG.T_tcp_magnet_pose6`` uses.
    (The stale ``urik.CONFIG.T_tcp_magnet_pose6`` is 0.47 m and must be
    overridden here.)  ``active_tcp_pose6 = 0`` because the robot's installed
    TCP offset is zero (flange == TCP).
    """
    from proper_research.simulation.simulations.initial_conditions import (
        LIVE_JOINTS_RAD,
        TCP_TO_MAGNET_POSE6,
    )

    return JointSpaceRobotConfig(
        q_seed_rad=tuple(LIVE_JOINTS_RAD),
        active_tcp_pose6=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tcp_to_magnet_pose6=tuple(TCP_TO_MAGNET_POSE6),
        insertion_min_m=0.0,
        insertion_max_m=0.05,
        debug_initial_ik=True,
        validate_chain_rule_at_start=True,
    )


def make_inverse_config() -> InverseConfigurationPlannerConfig:
    """Return the exact inverse-planner configuration from the saved run."""
    return InverseConfigurationPlannerConfig(
        position_tolerance_m=1.0e-3,
        tangent_tolerance_rad=np.deg2rad(80.0),

        initial_path_step_m=2.5e-4,
        minimum_path_step_m=1.25e-4,
        maximum_path_step_m=4.5e-3,
        grow_step_after_successes=999,

        maximum_function_evaluations=300,
        maximum_multistart_attempts=20,

        maximum_joint_step_rad=(0.075,) * 6,
        maximum_insertion_step_m=7.5e-4,

        continuity_weight=(
            5.0e-2,
            5.0e-2,
            5.0e-2,
            5.0e-2,
            5.0e-2,
            5.0e-2,
            1.0e-1,
        ),

        solve_initial_node=False,
        use_extrapolated_guess=True,
        insertion_non_decreasing=False,
        require_contact_model=True,
        finite_difference_validation_at_start=True,
        debug=True,
    )


def make_experiment_config(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
):
    """Recreate the single ExperimentConfig used by the working script."""
    # Radius-axis contact study (2026-09): centre point is the +30/-50 geometry,
    # which already has a complete L1->L2->L3 reference under full_control_stack.
    lumen_config = make_double_bend_lumen_config(
        first_angle_deg=30.0,
        second_angle_deg=-70.0,
    )

    experiments = make_curvature_jacobian_grid(
        run_root=Path(run_root),
        lumen_configs=(lumen_config,),
        jacobian_variants=("contact",),
        controller_kinds=("mpc",),
        solver_modes=("sqp_full",),
        inverse_sequence_modes=("rollout_ltv",),
        inverse_kp_values=(1,),
        inverse_desired_step_max_m=4.0e-3,
        inverse_extra_damping=0.0,
        rollout_steps_values=(10,),
        Np=15,
        N_sqp=50,
        max_steps=25,
        plant_contact=True,
        adaptive_rollout_enabled=False,
    )

    if len(experiments) != 1:
        raise RuntimeError(
            "Expected exactly one planning experiment, "
            f"but the grid produced {len(experiments)}."
        )

    exp_cfg = experiments[0]
    exp_cfg.validate()
    return exp_cfg


def build_planning_context(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
):
    """Build the model and controller objects needed by both offline stages.

    This function intentionally does not call ``solve_from_controller_pack``,
    ``optimize_from_saved_inverse_result``, or ``run_simulation``.

    Returns
    -------
    tuple
        ``(exp_cfg, bundle, controller_pack, out_root)``.
    """
    exp_cfg = make_experiment_config(run_root=Path(run_root))
    out_root = Path(exp_cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pivot_point, start_point, L0, dt = make_initial_poses()

    # The contact model needs a lumen that the (straight) nominal beam sits
    # well inside, or the analytic contact Jacobian fails FD validation and the
    # forward model bends.  The stock double-bend lumen curves away from the
    # straight beam.  Use a straight, wide-bore lumen down the beam axis; the
    # planners pass the real target centreline separately.
    bundle_lumen_cfg = LumenConfig(
        length=0.06, n_pts=240, n_ref_pts=100, radius=0.05, ds_target=1.0e-3, bends=()
    )

    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=bundle_lumen_cfg,
        plant_contact=exp_cfg.model.plant_contact,
        # Lab magnet on an extended bracket; dipole aligned with the beam axial
        # (+R.z) magnetisation direction at the reference pose.  The legacy body
        # -x default points the dipole ~horizontal and bends the model beam
        # ~20 deg where the real beam is straight.
        source_dipole_body_axis=SOURCE_DIPOLE_BODY_AXIS,
        # Build the contact lumen at the real beam base, not the legacy fixed
        # pose 270 mm away (which corrupts the contact solve).
        lumen_pivot_point=pivot_point,
    )

    plant_model = bundle.models["plant"]
    jacobian_model = bundle.models[
        exp_cfg.model.jacobian_variant
    ]

    controller_pack = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant_model,
        jacobian_model=jacobian_model,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=make_design_config(),
        robot_cfg=make_robot_config(),
    )

    return exp_cfg, bundle, controller_pack, out_root


__all__ = [
    "DEFAULT_RUN_ROOT",
    "build_planning_context",
    "make_design_config",
    "make_double_bend_lumen_config",
    "make_experiment_config",
    "make_inverse_config",
    "make_robot_config",
]
