import copy
from dataclasses import dataclass

import numpy as np

from proper_research.simulation_controller.controller import MPCControllerTipXY
from proper_research.simulation.magnetic_beam.controller_adapters import (
    ControllerForwardAdapter,
)
from proper_research.simulation.simulations.scenario import ControllerRunConfig
from .jacobian_utils import analytic_J_robot_xy_yaw_dL_with_diag
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
        1e-2,
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

    u_max: tuple[float, float, float, float, float, float, float] = (
        3.0,
        3.0,
        0.0,
        np.deg2rad(60.0),
        np.deg2rad(60.0),
        np.deg2rad(360.0),
        0.5,
    )

    trust_radius = (
        5,              # vx: ±50 mm/s
        5,              # vy: ±50 mm/s
        1,              # vz
        np.deg2rad(90.0),  # wx
        np.deg2rad(90.0),  # wy
        np.deg2rad(120.0), # wz
        5,              # dL: ±80 mm/s
    )


    L_min: float = 0.01
    L_max: float = 0.05

    dL_index: int = 6
    dL_back_max: float = 0.005
    dL_fwd_max: float = 0.1

    enable_hard_epm_tip_clearance: bool = True
    epm_tip_hard_min_m: float = 0.11

    qp_reg: float = 1e-9

    use_xy_ref_distance: bool = True
    ref_lookahead_pts: int = 2
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

def make_jacobian_fn(
    *,
    jacobian_model,
    dt: float,
    n_out: int,
):
    dt = float(dt)

    last_diag = {}

    def J_fn(p8):
        nonlocal last_diag

        p8 = np.asarray(p8, float).reshape(8)

        forward6d_jac = ControllerForwardAdapter(copy.deepcopy(jacobian_model))
        forward6d_jac.start_step()

        Jred_state, beam_diag = analytic_J_robot_xy_yaw_dL_with_diag(
            p8,
            forward6d_jac,
            n_out=n_out,
        )

        last_diag = dict(beam_diag)

        Jred_state = np.asarray(Jred_state, float)

        if Jred_state.shape != (n_out, 4):
            raise ValueError(
                f"analytic_J_robot_xy_yaw_dL returned {Jred_state.shape}, "
                f"expected {(n_out, 4)}."
            )

        B_reduced = Jred_state @ np.diag([dt, dt, dt, dt])

        B_full = J_full_from_robot_reduced_tip_tangent(
            B_reduced,
            n_out_full=n_out,
        )

        B_full = np.asarray(B_full, float)

        if B_full.shape != (n_out, 7):
            raise ValueError(
                f"J_full_from_robot_reduced_tip_tangent returned {B_full.shape}, "
                f"expected {(n_out, 7)}."
            )

        return B_full

    def get_last_diag():
        return dict(last_diag)

    J_fn.get_last_diag = get_last_diag

    return J_fn


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
    Build one MPC controller for one experiment.

    Parameters
    ----------
    plant_model:
        Forward model used for rollout/application. In your main study this
        should usually be the contact-aware model.

    jacobian_model:
        Forward model used only inside the MPC Jacobian callback. This is the
        model that changes between "contact" and "no_contact".

    run_cfg:
        Experiment run configuration: Np, N_sqp, solver_mode, rollout_steps,
        max_steps.

    design_cfg:
        Controller weights, bounds, trust region and clearance settings.
    """
    if design_cfg is None:
        design_cfg = ControllerDesignConfig()

    lumen_C = np.asarray(lumen_C, float)
    lumen_R = np.asarray(lumen_R, float).reshape(-1)

    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(f"lumen_C must have shape (M, >=3), got {lumen_C.shape}.")

    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError(
            f"lumen_R length {lumen_R.size} does not match lumen_C length "
            f"{lumen_C.shape[0]}."
        )

    p0 = make_pose8_from_start_point(
        start_point=start_point,
        L0=L0,
    )

    p_min, p_max = make_pose_bounds(
        start_point=start_point,
        design_cfg=design_cfg,
    )

    u_max = np.asarray(design_cfg.u_max, float).reshape(design_cfg.n_u)
    trust_radius = np.asarray(design_cfg.trust_radius, float).reshape(design_cfg.n_u)
    w_u = np.asarray(design_cfg.w_u, float).reshape(design_cfg.n_u)
    w_du = np.asarray(design_cfg.w_du, float).reshape(design_cfg.n_u)

    forward6d_plant = ControllerForwardAdapter(copy.deepcopy(plant_model))
    forward6d_plant.start_step()

    def forward3d_plant(p8, *, commit=False):
        y6 = forward6d_plant(p8, commit=commit)
        return np.asarray(y6, float).reshape(-1)[:design_cfg.n_out]
    # This is the Jacobian model used by the MPC.
    J_fn = make_jacobian_fn(
        jacobian_model=jacobian_model,
        dt=dt,
        n_out=design_cfg.n_out,
    )

    mpc = MPCControllerTipXY(
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
        enable_hard_epm_tip_clearance=design_cfg.enable_hard_epm_tip_clearance,
        epm_tip_hard_min_m=design_cfg.epm_tip_hard_min_m,
        dL_index=design_cfg.dL_index,
        dL_back_max=design_cfg.dL_back_max,
        dL_fwd_max=design_cfg.dL_fwd_max,
        qp_reg=design_cfg.qp_reg,
    )

    # Lumen/reference data used by the controller.
    mpc.lumen_C = lumen_C
    mpc.lumen_R = lumen_R

    mpc.use_xy_ref_distance = design_cfg.use_xy_ref_distance
    mpc.ref_lookahead_pts = design_cfg.ref_lookahead_pts
    mpc.allow_ref_backward = design_cfg.allow_ref_backward
    mpc.ref_stride_pts = design_cfg.ref_stride_pts

    mpc.ref_stage_weights = np.linspace(
        design_cfg.ref_weight_start,
        design_cfg.ref_weight_end,
        mpc.Np,
    )

    mpc.dL_guess = design_cfg.dL_guess

    mpc.set_initial_params(p0)

    return {
        "mpc": mpc,
        "p0": p0,
        "p_min": p_min,
        "p_max": p_max,
        "u_max": u_max,
        "forward6d_plant": forward6d_plant,
        "design_cfg": design_cfg,
    }
def make_jacobian_fn(
    *,
    jacobian_model,
    dt: float,
    n_out: int,
):
    """
    Build MPC Jacobian callback.

    The jacobian_model is separate from the plant model so that the experiment
    can compare contact-aware versus no-contact Jacobians.
    """
    dt = float(dt)

    jac_model_working = copy.deepcopy(jacobian_model)
    forward_jac = ControllerForwardAdapter(jac_model_working)

    last_diag = {}

    def J_fn(p8):
        nonlocal last_diag

        p8 = np.asarray(p8, float).reshape(8)

        forward_jac.start_step()

        Jred_state, beam_diag = analytic_J_robot_xy_yaw_dL_with_diag(
            p8,
            forward_jac,
            n_out=n_out,
        )

        last_diag = dict(beam_diag)

        Jred_state = np.asarray(Jred_state, float)

        if Jred_state.shape != (n_out, 4):
            raise ValueError(
                f"analytic_J_robot_xy_yaw_dL_with_diag returned "
                f"{Jred_state.shape}, expected {(n_out, 4)}."
            )

        # Controls are rates, so convert sensitivity wrt pose increments into
        # sensitivity wrt one control sample.
        B_reduced = Jred_state @ np.diag([dt, dt, dt, dt])

        B_full = J_full_from_robot_reduced_tip_tangent(
            B_reduced,
            n_out_full=n_out,
        )

        B_full = np.asarray(B_full, float)

        if B_full.shape != (n_out, 7):
            raise ValueError(
                f"J_full_from_robot_reduced_tip_tangent returned "
                f"{B_full.shape}, expected {(n_out, 7)}."
            )

        return B_full

    def get_last_diag():
        return dict(last_diag)

    J_fn.get_last_diag = get_last_diag

    return J_fn