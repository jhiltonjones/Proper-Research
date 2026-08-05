from __future__ import annotations

import copy

import numpy as np

from proper_research.simulation.magnetic_beam.controller_adapters import (
    ControllerForwardAdapter,
    make_controller_jacobian_fn,
)
from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
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
    """
    Build the optimized MPC controller around optimized beam models.

    LTI, LTV one-shot and full SQP remain selected by run_cfg.solver_mode.
    Hierarchical solver selection is intentionally absent.
    """
    run_cfg.validate()
    if getattr(run_cfg, "controller_kind", "mpc") != "mpc":
        raise ValueError(
            "build_controller_optimized currently supports controller_kind='mpc'."
        )

    if design_cfg is None:
        design_cfg = ControllerDesignConfig()

    lumen_C = np.asarray(lumen_C, dtype=float)
    lumen_R = np.asarray(lumen_R, dtype=float).reshape(-1)
    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(
            f"lumen_C must have shape (M, >=3), got {lumen_C.shape}."
        )
    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError(
            f"lumen_R length {lumen_R.size} does not match "
            f"lumen_C length {lumen_C.shape[0]}."
        )

    p0 = make_pose8_from_start_point(
        start_point=np.asarray(start_point, dtype=float),
        L0=float(L0),
    )
    p_min, p_max = make_pose_bounds(
        start_point=np.asarray(start_point, dtype=float),
        design_cfg=design_cfg,
    )

    # Keep independent plant and Jacobian caches. Copies happen once at startup,
    # not inside the controller hot path.
    plant_owned = copy.deepcopy(plant_model) if copy_models else plant_model
    jacobian_owned = (
        copy.deepcopy(jacobian_model) if copy_models else jacobian_model
    )

    forward6d_plant = ControllerForwardAdapter(plant_owned)
    forward6d_plant.start_step()

    def forward3d_plant(p8, *, commit=False):
        y6 = forward6d_plant(p8, commit=commit)
        return np.asarray(y6, dtype=float).reshape(-1)[: design_cfg.n_out]

    forward3d_plant.start_step = forward6d_plant.start_step
    forward3d_plant.reset = forward6d_plant.reset
    forward3d_plant.adapter = forward6d_plant
    forward3d_plant.model = forward6d_plant.model

    # copy_model=False because jacobian_owned is already an independent instance.
    J_fn = make_controller_jacobian_fn(
        jacobian_model=jacobian_owned,
        dt=float(dt),
        n_out=design_cfg.n_out,
        angular_velocity_frame="body",
        jacobian_mode="fast",
        copy_model=False,
    )

    common_kwargs = dict(
        Jxy_fn=J_fn,
        forward_tip_fn=forward3d_plant,
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
        trust_radius_min=np.asarray(
            design_cfg.trust_radius_min,
            dtype=float,
        ),
        trust_radius_max=np.asarray(
            design_cfg.trust_radius_max,
            dtype=float,
        ),
        enable_hard_epm_tip_clearance=bool(
            design_cfg.enable_hard_epm_tip_clearance
        ),
        epm_tip_hard_min_m=float(design_cfg.epm_tip_hard_min_m),
        dL_index=int(design_cfg.dL_index),
        dL_back_max=float(design_cfg.dL_back_max),
        dL_fwd_max=float(design_cfg.dL_fwd_max),
        qp_reg=float(design_cfg.qp_reg),
        qp_settings=qp_settings,
        validate_nonlinear_candidate=validate_nonlinear_candidate,
        collect_full_diagnostics=collect_full_diagnostics,
        verbose_controller=verbose_controller,
    )
    controller = MPCControllerTipXYOptimized(**common_kwargs)

    controller.lumen_C = lumen_C[:, :3].copy()
    controller.lumen_R = lumen_R.copy()
    controller.use_xy_ref_distance = bool(design_cfg.use_xy_ref_distance)
    controller.ref_lookahead_pts = int(design_cfg.ref_lookahead_pts)
    controller.allow_ref_backward = bool(design_cfg.allow_ref_backward)
    controller.ref_stride_pts = int(design_cfg.ref_stride_pts)
    controller.ref_stage_weights = np.linspace(
        float(design_cfg.ref_weight_start),
        float(design_cfg.ref_weight_end),
        controller.Np,
    )
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


# Compatibility alias for scripts that previously imported build_controller.
build_controller = build_controller_optimized
