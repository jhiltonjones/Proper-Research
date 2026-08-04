import copy
from dataclasses import dataclass

import numpy as np

from proper_research.simulation_controller.controller import MPCControllerTipXY
from proper_research.simulation.magnetic_beam.controller_adapters import (
    ControllerForwardAdapter,
)
from proper_research.simulation.simulations.scenario import ControllerRunConfig
# Put these helper functions wherever you currently have them.
# If they are still in the old simulation script, move them into geometry_utils.py
# and jacobian_utils.py as shown here.
from proper_research.simulation_controller.geometry import (
    pose7_rotvec_to_pose8_quat,
)

from proper_research.simulation.simulations.jacobian_utils import (
    analytic_J_robot_xy_yaw_dL,
    J_full_from_robot_reduced_tip_tangent,
)
from proper_research.simulation_controller.controller import (
    MPCControllerTipXY,
)

from proper_research.simulation_controller.inverse_jacobian_controller import (
    InverseJacobianControllerTipXY,
)
@dataclass(frozen=True)
class ControllerDesignConfig:
    """
    Fixed controller tuning parameters.

    ControllerRunConfig should contain experiment-run settings such as Np,
    N_sqp, solver_mode, rollout_steps and max_steps.

    ControllerDesignConfig contains controller tuning, bounds and weights.
    """

    n_out: int = 3
    n_u: int = 7
    n_p: int = 8

    w_tracking: tuple[float, float, float] = (
        1000.0,
        1000.0,
        0.0,
    )

    w_u: tuple[float, float, float, float, float, float, float] = (
        1e-6,
        1e-6,
        1e-2,
        1e-2,
        1e-6,
        1e-6,
        1e-6,
    )

    w_du: tuple[float, float, float, float, float, float, float] = (
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
    )
    wz_max = np.deg2rad(30.0) / 0.1  # 1.745 rad/s

    u_max = np.array(
        [
            0.5,       # vx [m/s]
            0.5,       # vy [m/s]
            0.0,       # vz locked
            0.0,       # wx locked
            0.5,       # wy [rad/s]
            wz_max,    # wz [rad/s]: 10 degrees per command
            0.5,       # insertion rate [m/s]
        ],
        dtype=float,
    )

    # trust_radius = np.array(
    #     [
    #         0.50,  # vx
    #         0.50,  # vy
    #         0.00,  # vz
    #         0.00,  # wx
    #         0.50,  # wy
    #         0.50,  # wz
    #         0.50,  # Ldot
    #     ],
    #     dtype=float,
    # )
    trust_radius = u_max
    trust_radius_max = trust_radius
    trust_radius_min = np.array(
        [
            0.025,
            0.025,
            0.000,
            0.000,
            0.050,
            0.050,
            0.025,
        ],
        dtype=float,
    )

    # trust_radius_max = np.array(
    #    [
    #         0.50,  # vx
    #         0.50,  # vy
    #         0.00,  # vz
    #         0.00,  # wx
    #         0.50,  # wy
    #         0.50,  # wz
    #         0.50,  # Ldot
    #     ],
    #     dtype=float,
    # )

    L_min: float = 0.01
    L_max: float = 0.05

    dL_index: int = 6
    dL_back_max: float = 1
    dL_fwd_max: float = 1

    enable_hard_epm_tip_clearance: bool = True
    epm_tip_hard_min_m: float = 0.05

    qp_reg: float = 1e-9

    use_xy_ref_distance: bool = True
    ref_lookahead_pts: int = 1
    allow_ref_backward: bool = False
    ref_stride_pts: int = 1
    ref_weight_start: float = 2.0
    ref_weight_end: float = 0.5

    dL_guess: float = 0.02


def make_pose8_from_start_point(
    *,
    start_point: np.ndarray,
    L0: float,
) -> np.ndarray:
    """
    Convert UR pose6 + insertion length into pose8.

    Input start_point:
        [x, y, z, rx, ry, rz]

    Output pose8:
        [x, y, z, qw, qx, qy, qz, L]
    """
    start_point = np.asarray(start_point, float).reshape(6)

    p0_pose7 = np.array(
        [
            start_point[0],
            start_point[1],
            start_point[2],
            start_point[3],
            start_point[4],
            start_point[5],
            float(L0),
        ],
        dtype=float,
    )

    return pose7_rotvec_to_pose8_quat(p0_pose7)


def make_pose_bounds(
    *,
    start_point: np.ndarray,
    design_cfg: ControllerDesignConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build controller pose bounds for pose8:

        [x, y, z, qw, qx, qy, qz, L]
    """
    start_point = np.asarray(start_point, float).reshape(6)

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
            +np.inf,
            +np.inf,
            +np.inf,
            +np.inf,
            design_cfg.L_max,
        ],
        dtype=float,
    )

    return p_min, p_max

"""
Drop-in bridge from the controller package to magnetic_beam2.

Replace the controller package's existing ``make_jacobian_fn`` with the
function below, then delete ``analytic_J_robot_full_with_diag``.  The beam
Jacobian is computed only by MagneticBeamForwardModel.
"""

from proper_research.simulation.magnetic_beam.controller_adapters import (
    make_controller_jacobian_fn,
)


# def make_jacobian_fn(
#     *,
#     jacobian_model,
#     dt: float,
#     n_out: int,
# ):
#     return make_controller_jacobian_fn(
#         jacobian_model=jacobian_model,
#         dt=dt,
#         n_out=n_out,
#         # Your controller update is q_next = q * dq(dt * omega_body).
#         angular_velocity_frame="body",
#         # Uses forward differences, exact matrix caching, and bounded
#         # quasi-Newton Hessian reuse from the magnetic_beam2 model.
#         jacobian_mode="fast",
#         # Create one private Jacobian-model instance once.  It is deliberately
#         # not copied again inside each J_fn call.
#         copy_model=True,
#     )
# def make_jacobian_fn(
#     *,
#     jacobian_model,
#     dt: float,
#     n_out: int,
# ):
#     """
#     Construct the one-step control-to-tip Jacobian

#         B = d p_tip,next / d u

#     for the seven-channel controller input:

#         [vx, vy, vz, wx, wy, wz, dL_rate].
#     """
#     dt = float(dt)

#     if dt <= 0.0:
#         raise ValueError(
#             f"dt must be positive, got {dt}."
#         )

#     if int(n_out) != 3:
#         raise ValueError(
#             "The current beam model supplies a 3D "
#             "tip-position Jacobian, so n_out must be 3."
#         )

#     last_diag: dict = {}

#     def J_fn(p8):
#         nonlocal last_diag

#         p8 = np.asarray(
#             p8,
#             float,
#         ).reshape(8)

#         forward6d_jac = (
#             ControllerForwardAdapter(
#                 copy.deepcopy(jacobian_model)
#             )
#         )

#         forward6d_jac.start_step()

#         J_control_state, beam_diag = (
#             analytic_J_robot_full_with_diag(
#                 p8,
#                 forward6d_jac,
#                 n_out=n_out,
#             )
#         )

#         last_diag = dict(beam_diag)

#         J_control_state = np.asarray(
#             J_control_state,
#             float,
#         ).reshape(n_out, 7)

#         # --------------------------------------------------------
#         # Convert instantaneous pose/length sensitivity to the
#         # one-sample control sensitivity used by MPC:
#         #
#         #   dr       = dt * v_world
#         #   dphi_body = dt * omega_body
#         #   dL       = dt * dL_rate
#         # --------------------------------------------------------
#         B_full = dt * J_control_state

#         if B_full.shape != (n_out, 7):
#             raise ValueError(
#                 "Full one-step Jacobian has shape "
#                 f"{B_full.shape}; expected {(n_out, 7)}."
#             )

#         if not np.all(np.isfinite(B_full)):
#             raise FloatingPointError(
#                 "Full one-step Jacobian contains "
#                 "non-finite values."
#             )

#         return B_full

#     def get_last_diag():
#         return dict(last_diag)

#     J_fn.get_last_diag = get_last_diag

#     return J_fn


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
    """
    Build either:
        - MPCControllerTipXY, or
        - InverseJacobianControllerTipXY.

    Both controllers expose the same public interface required by
    run_simulation().
    """
    run_cfg.validate()

    if design_cfg is None:
        design_cfg = ControllerDesignConfig()

    lumen_C = np.asarray(lumen_C, float)
    lumen_R = np.asarray(
        lumen_R,
        float,
    ).reshape(-1)

    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(
            f"lumen_C must have shape (M, >=3), "
            f"got {lumen_C.shape}."
        )

    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError(
            f"lumen_R length {lumen_R.size} does not match "
            f"lumen_C length {lumen_C.shape[0]}."
        )

    p0 = make_pose8_from_start_point(
        start_point=start_point,
        L0=L0,
    )
    print(f"p0 is {p0}")
    p_min, p_max = make_pose_bounds(
        start_point=start_point,
        design_cfg=design_cfg,
    )

    u_max = np.asarray(
        design_cfg.u_max,
        float,
    ).reshape(design_cfg.n_u)

    trust_radius = np.asarray(
        design_cfg.trust_radius,
        float,
    ).reshape(design_cfg.n_u)

    w_u = np.asarray(
        design_cfg.w_u,
        float,
    ).reshape(design_cfg.n_u)

    w_du = np.asarray(
        design_cfg.w_du,
        float,
    ).reshape(design_cfg.n_u)

    # ------------------------------------------------------------
    # Plant model
    # ------------------------------------------------------------
    forward6d_plant = ControllerForwardAdapter(
        copy.deepcopy(plant_model)
    )
    forward6d_plant.start_step()

    def forward3d_plant(p8, *, commit=False):
        y6 = forward6d_plant(
            p8,
            commit=commit,
        )

        return np.asarray(
            y6,
            float,
        ).reshape(-1)[:design_cfg.n_out]


    # Expose the wrapped plant adapter/model.
    forward3d_plant.start_step = (
        forward6d_plant.start_step
    )
    forward3d_plant.reset = (
        forward6d_plant.reset
    )
    forward3d_plant.adapter = forward6d_plant
    forward3d_plant.model = forward6d_plant.model

    # ------------------------------------------------------------
    # Jacobian model
    # ------------------------------------------------------------
    J_fn = make_controller_jacobian_fn(
        jacobian_model=jacobian_model,
        dt=dt,
        n_out=design_cfg.n_out,
        angular_velocity_frame="body",
        jacobian_mode="fast",
        copy_model=True,
    )

    # ------------------------------------------------------------
    # Arguments shared by MPC and inverse Jacobian
    # ------------------------------------------------------------
    common_kwargs = dict(
        Jxy_fn=J_fn,
        forward_tip_fn=forward3d_plant,
        dt=float(dt),
        Np=int(run_cfg.Np),
        n_out=design_cfg.n_out,
        n_u=design_cfg.n_u,
        n_p=design_cfg.n_p,
        w_xy=design_cfg.w_tracking,
        w_u=w_u,
        w_du=w_du,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        model_mode="ltv",
        solver_mode=str(run_cfg.solver_mode),
        N_sqp=int(run_cfg.N_sqp),
        trust_radius=trust_radius,
        trust_radius_min=np.asarray(
            design_cfg.trust_radius_min,
            float,
        ),
        trust_radius_max=np.asarray(
            design_cfg.trust_radius_max,
            float,
        ),
        enable_hard_epm_tip_clearance=(
            design_cfg.enable_hard_epm_tip_clearance
        ),
        epm_tip_hard_min_m=(
            design_cfg.epm_tip_hard_min_m
        ),
        dL_index=design_cfg.dL_index,
        dL_back_max=design_cfg.dL_back_max,
        dL_fwd_max=design_cfg.dL_fwd_max,
        qp_reg=design_cfg.qp_reg,
    )

    # ------------------------------------------------------------
    # Instantiate selected controller
    # ------------------------------------------------------------
    if run_cfg.controller_kind == "mpc":
        controller = MPCControllerTipXY(
            **common_kwargs,
        )

    elif run_cfg.controller_kind == "inverse_jacobian":
        controller = InverseJacobianControllerTipXY(
            **common_kwargs,
            sequence_mode=run_cfg.inverse_sequence_mode,
            kp=run_cfg.inverse_kp,
            desired_step_max=(
                run_cfg.inverse_desired_step_max_m
            ),
            extra_damping=(
                run_cfg.inverse_extra_damping
            ),
        )

    else:
        raise ValueError(
            f"Unknown controller_kind: "
            f"{run_cfg.controller_kind!r}."
        )

    # ------------------------------------------------------------
    # Shared reference configuration
    # ------------------------------------------------------------
    controller.lumen_C = lumen_C
    controller.lumen_R = lumen_R

    controller.use_xy_ref_distance = (
        design_cfg.use_xy_ref_distance
    )
    controller.ref_lookahead_pts = (
        design_cfg.ref_lookahead_pts
    )
    controller.allow_ref_backward = (
        design_cfg.allow_ref_backward
    )
    controller.ref_stride_pts = (
        design_cfg.ref_stride_pts
    )

    controller.ref_stage_weights = np.linspace(
        design_cfg.ref_weight_start,
        design_cfg.ref_weight_end,
        controller.Np,
    )   
    

    controller.dL_guess = design_cfg.dL_guess

    controller.set_initial_params(p0)
    return {
        "controller": controller,
        "mpc": controller,
        "controller_kind": run_cfg.controller_kind,
        "p0": p0,
        "p_min": p_min,
        "p_max": p_max,
        "u_max": u_max,
        "forward6d_plant": forward6d_plant,
        "design_cfg": design_cfg,

        # Contact/Jacobian debugging.
        "J_fn": J_fn,
        "jacobian_model_owned": J_fn.model,
    }