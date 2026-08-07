from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from proper_research.simulation.magnetic_beam.controller_adapters import (
    ControllerForwardAdapter,
)
from proper_research.simulation.magnetic_beam.controller_output_adapters import (
    make_controller_output_jacobian_fn,
)
from proper_research.simulation.simulations.scenario import ControllerRunConfig
from proper_research.simulation_controller.controller import MPCControllerTipXY
from proper_research.simulation_controller.geometry import (
    pose7_rotvec_to_pose8_quat,
)
from proper_research.simulation_controller.inverse_jacobian_controller import (
    InverseJacobianControllerTipXY,
)


@dataclass(frozen=True)
class ControllerDesignConfig:
    """Fixed controller, path-reference and safety configuration.

    The default output is six-dimensional so the tangent is available for
    diagnostics or a hard safety constraint. Its three tracking weights are
    zero, therefore the objective tracks only tip position.

    For the lowest-cost position-only experiment use n_out=3 and provide only
    the first three tracking weights. In that mode the hard tangent constraint
    must remain disabled.
    """

    n_out: int = 6
    n_u: int = 7
    n_p: int = 8

    # [tip_xyz, tip_tangent]. Zero tangent weights mean no tangent tracking.
    w_tracking: tuple[float, ...] = (
        1000.0,
        1000.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    w_u: tuple[float, ...] = (
        1e-6,
        1e-6,
        1e-2,
        1e-2,
        1e-6,
        1e-6,
        1e-6,
    )
    w_du: tuple[float, ...] = (1e-8,) * 7

    u_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.5,                       # vx [m/s]
                0.5,                       # vy [m/s]
                0.0,                       # vz locked
                0.0,                       # wx locked
                0.5,                       # wy [rad/s]
                np.deg2rad(30.0) / 0.1,    # wz [rad/s]
                0.5,                       # insertion rate [m/s]
            ],
            dtype=float,
        )
    )
    trust_radius: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.5, 0.5, 0.0, 0.0, 0.5, np.deg2rad(30.0) / 0.1, 0.5],
            dtype=float,
        )
    )
    trust_radius_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.5, 0.5, 0.0, 0.0, 0.5, np.deg2rad(30.0) / 0.1, 0.5],
            dtype=float,
        )
    )
    trust_radius_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.025, 0.025, 0.0, 0.0, 0.05, 0.05, 0.025],
            dtype=float,
        )
    )

    L_min: float = 0.002
    L_max: float = 0.05
    dL_index: int = 6
    dL_back_max: float = 0.01
    dL_fwd_max: float = 0.05

    enable_hard_epm_tip_clearance: bool = True
    epm_tip_hard_min_m: float = 0.05
    qp_reg: float = 1e-9

    # Reference/path mode.
    reference_mode: str = "contouring"
    use_xy_ref_distance: bool = False
    allow_ref_backward: bool = False
    ref_lookahead_m: float = 4.0e-4
    ref_stride_m: float = 4.0e-4
    ref_search_backward_m: float = 5.0e-4
    ref_search_forward_m: float = 5.0e-3
    ref_max_progress_per_step_m: float = 1.0e-3
    ref_max_backward_per_step_m: float = 0.0
    ref_weight_start: float = 2.0
    ref_weight_end: float = 0.5

    # Contouring objective: strong normal error, weak along-path lag error.
    q_contour: float = 100.0
    q_lag: float = 2.0
    contouring_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Progress constraint is separate from the geometric reference. Leave off
    # while validating path tracking; turn on identically for both Jacobians in
    # the safe-progress feasibility experiment.
    enable_hard_progress_constraint: bool = False
    progress_request_m: float = 5.0e-4
    progress_reward_weight: float = 0.0

    # Tangent is modelled when n_out=6. It is not tracked unless w_tracking[3:6]
    # are positive. This safety constraint is independent of tracking weights.
    enable_hard_tip_tangent_angle: bool = False
    tip_tangent_max_angle_deg: float = 40.0
    tip_tangent_nonlinear_tol_deg: float = 0.25
    tip_tangent_activation_clearance_m: float = 1.0e-3
    tip_tangent_activation_guard_m: float = 0.5e-3
    tip_radius_m: float = 0.0

    dL_guess: float = 0.02

    def validate(self) -> None:
        if self.n_out not in {3, 6}:
            raise ValueError("n_out must be 3 or 6.")
        if len(self.w_tracking) != self.n_out:
            raise ValueError(
                f"w_tracking must have length n_out={self.n_out}; "
                f"got {len(self.w_tracking)}."
            )
        if len(self.w_u) != self.n_u or len(self.w_du) != self.n_u:
            raise ValueError("w_u and w_du must each have length n_u.")
        for name in ("u_max", "trust_radius", "trust_radius_min", "trust_radius_max"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if value.size != self.n_u:
                raise ValueError(f"{name} must have length n_u={self.n_u}.")
        if self.enable_hard_tip_tangent_angle and self.n_out < 6:
            raise ValueError(
                "Hard tip-tangent angle safety requires n_out=6. "
                "Tangent tracking weights may still remain zero."
            )
        if self.reference_mode not in {"point", "contouring"}:
            raise ValueError("reference_mode must be 'point' or 'contouring'.")
        if self.enable_hard_progress_constraint and self.reference_mode != "contouring":
            raise ValueError(
                "enable_hard_progress_constraint requires reference_mode='contouring'."
            )


def make_pose8_from_start_point(
    *,
    start_point: np.ndarray,
    L0: float,
) -> np.ndarray:
    start_point = np.asarray(start_point, dtype=float).reshape(6)
    p7 = np.concatenate((start_point, [float(L0)]))
    return pose7_rotvec_to_pose8_quat(p7)


def make_pose_bounds(
    *,
    start_point: np.ndarray,
    design_cfg: ControllerDesignConfig,
) -> tuple[np.ndarray, np.ndarray]:
    start_point = np.asarray(start_point, dtype=float).reshape(6)
    p_min = np.array(
        [
            0.2,
            -1.0,
            start_point[2],
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            design_cfg.L_min,
        ],
        dtype=float,
    )
    p_max = np.array(
        [
            0.8,
            1.5,
            start_point[2],
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            design_cfg.L_max,
        ],
        dtype=float,
    )
    return p_min, p_max


def configure_path_following(controller, design_cfg: ControllerDesignConfig) -> None:
    """Apply identical geometric reference settings to either controller class."""
    controller.reference_mode = str(design_cfg.reference_mode)
    controller.use_xy_ref_distance = bool(design_cfg.use_xy_ref_distance)
    controller.allow_ref_backward = bool(design_cfg.allow_ref_backward)
    controller.ref_lookahead_m = float(design_cfg.ref_lookahead_m)
    controller.ref_stride_m = float(design_cfg.ref_stride_m)
    controller.ref_search_backward_m = float(design_cfg.ref_search_backward_m)
    controller.ref_search_forward_m = float(design_cfg.ref_search_forward_m)
    controller.ref_max_progress_per_step_m = float(
        design_cfg.ref_max_progress_per_step_m
    )
    controller.ref_max_backward_per_step_m = float(
        design_cfg.ref_max_backward_per_step_m
    )
    controller.ref_stage_weights = np.linspace(
        float(design_cfg.ref_weight_start),
        float(design_cfg.ref_weight_end),
        int(controller.Np),
    )
    controller.q_contour = float(design_cfg.q_contour)
    controller.q_lag = float(design_cfg.q_lag)
    controller.contouring_axis_weights = np.asarray(
        design_cfg.contouring_axis_weights,
        dtype=float,
    )
    controller.enable_hard_progress_constraint = bool(
        design_cfg.enable_hard_progress_constraint
    )
    controller.progress_request_m = float(design_cfg.progress_request_m)
    controller.progress_reward_weight = float(design_cfg.progress_reward_weight)

    # Respect all configured output weights. Zero tangent weights produce zero
    # tangent cost; positive weights enable tangent tracking explicitly.
    controller.track_all_outputs = True


def build_controller(
    *,
    start_point: np.ndarray,
    L0: float,
    dt: float,
    plant_model,
    jacobian_model,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    run_cfg: ControllerRunConfig,
    design_cfg: ControllerDesignConfig | None = None,
):
    run_cfg.validate()
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

    forward6d_plant = ControllerForwardAdapter(copy.deepcopy(plant_model))
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
        jacobian_model=jacobian_model,
        dt=float(dt),
        n_out=design_cfg.n_out,
        angular_velocity_frame="body",
        jacobian_mode="fast",
        copy_model=True,
    )

    common_kwargs = dict(
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
    )

    if run_cfg.controller_kind == "mpc":
        controller = MPCControllerTipXY(**common_kwargs)
    elif run_cfg.controller_kind == "inverse_jacobian":
        controller = InverseJacobianControllerTipXY(
            **common_kwargs,
            sequence_mode=run_cfg.inverse_sequence_mode,
            kp=run_cfg.inverse_kp,
            desired_step_max=run_cfg.inverse_desired_step_max_m,
            extra_damping=run_cfg.inverse_extra_damping,
        )
    else:
        raise ValueError(f"Unknown controller_kind: {run_cfg.controller_kind!r}.")

    controller.lumen_C = lumen_C[:, :3].copy()
    controller.lumen_R = lumen_R.copy()
    configure_path_following(controller, design_cfg)
    controller.dL_guess = float(design_cfg.dL_guess)
    controller.set_initial_params(p0)

    return {
        "controller": controller,
        "mpc": controller,
        "controller_kind": run_cfg.controller_kind,
        "p0": p0,
        "p_min": p_min,
        "p_max": p_max,
        "u_max": np.asarray(design_cfg.u_max, dtype=float).copy(),
        "forward6d_plant": forward6d_plant,
        "design_cfg": design_cfg,
        "J_fn": J_fn,
        "jacobian_model_owned": J_fn.model,
    }
