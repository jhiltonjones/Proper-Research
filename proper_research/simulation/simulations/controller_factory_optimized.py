from __future__ import annotations

import copy

import numpy as np

from proper_research.simulation.magnetic_beam.controller_adapters import (
    ControllerForwardAdapter,
)
from proper_research.simulation.magnetic_beam.controller_output_adapters import (
    make_controller_output_jacobian_fn,
)
from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
    configure_path_following,
    make_pose8_from_start_point,
    make_pose_bounds,
)
from proper_research.simulation_controller.controller_optimized import (
    MPCControllerTipXYOptimized,
)
from proper_research.simulation_controller.persistent_osqp import (
    PersistentOSQPSettings,
)


def build_controller_optimized(
    *,
    start_point: np.ndarray,
    L0: float,
    dt: float,
    plant_model,
    jacobian_model,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    run_cfg,
    design_cfg: ControllerDesignConfig | None = None,
    copy_models: bool = True,
    validate_nonlinear_candidate: bool = False,
    collect_full_diagnostics: bool = False,
    verbose_controller: bool = False,
    qp_settings: PersistentOSQPSettings | None = None,
):
    """Build the fast controller with the same control law as the debug version."""
    run_cfg.validate()
    if getattr(run_cfg, "controller_kind", "mpc") != "mpc":
        raise ValueError(
            "build_controller_optimized currently supports controller_kind='mpc'."
        )

    design_cfg = design_cfg or ControllerDesignConfig()
    design_cfg.validate()

    lumen_C = np.asarray(lumen_C, dtype=float)
    lumen_R = np.asarray(lumen_R, dtype=float).reshape(-1)
    if lumen_C.ndim != 2 or lumen_C.shape[0] < 2 or lumen_C.shape[1] < 3:
        raise ValueError("lumen_C must have shape (M, >=3), M >= 2.")
    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError("lumen_R must contain one radius per centreline node.")

    p0 = make_pose8_from_start_point(start_point=start_point, L0=L0)
    p_min, p_max = make_pose_bounds(
        start_point=start_point,
        design_cfg=design_cfg,
    )

    # Independent caches preserve the intended plant/Jacobian separation.
    plant_owned = copy.deepcopy(plant_model) if copy_models else plant_model
    jacobian_owned = (
        copy.deepcopy(jacobian_model) if copy_models else jacobian_model
    )

    forward6d_plant = ControllerForwardAdapter(plant_owned)
    forward6d_plant.start_step()

    def forward_output_plant(p8, *, commit=False):
        y6 = np.asarray(
            forward6d_plant(p8, commit=commit),
            dtype=float,
        ).reshape(6)
        return y6[: design_cfg.n_out]

    forward_output_plant.start_step = forward6d_plant.start_step
    forward_output_plant.reset = forward6d_plant.reset
    forward_output_plant.adapter = forward6d_plant
    forward_output_plant.model = forward6d_plant.model

    J_fn = make_controller_output_jacobian_fn(
        jacobian_model=jacobian_owned,
        dt=float(dt),
        n_out=design_cfg.n_out,
        angular_velocity_frame="body",
        jacobian_mode="fast",
        copy_model=False,
    )

    controller = MPCControllerTipXYOptimized(
        Jxy_fn=J_fn,
        forward_tip_fn=forward_output_plant,
        dt=float(dt),
        Np=int(run_cfg.Np),
        n_out=design_cfg.n_out,
        n_u=design_cfg.n_u,
        n_p=design_cfg.n_p,
        w_xy=np.asarray(design_cfg.w_tracking, dtype=float),
        w_u=np.asarray(design_cfg.w_u, dtype=float),
        w_du=np.asarray(design_cfg.w_du, dtype=float),
        u_max=np.asarray(design_cfg.u_max, dtype=float),
        p_min=np.asarray(p_min, dtype=float),
        p_max=np.asarray(p_max, dtype=float),
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
        dL_index=int(design_cfg.dL_index),
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
        qp_settings=qp_settings,
        validate_nonlinear_candidate=validate_nonlinear_candidate,
        collect_full_diagnostics=collect_full_diagnostics,
        verbose_controller=verbose_controller,
    )

    controller.lumen_C = lumen_C[:, :3].copy()
    controller.lumen_R = lumen_R.copy()
    configure_path_following(controller, design_cfg)
    controller.dL_guess = float(design_cfg.dL_guess)
    controller.set_initial_params(p0)

    return {
        "controller": controller,
        "mpc": controller,
        "controller_kind": "mpc",
        "p0": p0,
        "p_min": p_min,
        "p_max": p_max,
        "u_max": np.asarray(design_cfg.u_max, dtype=float).copy(),
        "forward6d_plant": forward6d_plant,
        "design_cfg": design_cfg,
        "J_fn": J_fn,
        "plant_model_owned": plant_owned,
        "jacobian_model_owned": jacobian_owned,
    }


build_controller = build_controller_optimized
