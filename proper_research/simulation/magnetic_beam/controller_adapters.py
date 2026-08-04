from __future__ import annotations

import copy
import numpy as np

from .kinematics import (
    pose8_quat_to_pose7_rotvec,
    quat_normalize,
    quat_to_R,
    skew,
    tip_tangent_from_centerline,
)


def assemble_tip_robot_jacobian(
    J_tip_actuation_tangent: np.ndarray,
    J_arm_spatial: np.ndarray,
    *,
    insertion_jacobian: float | np.ndarray = 1.0,
    source_offset_world: np.ndarray | None = None,
) -> np.ndarray:
    """
    Chain beam, robot-arm, and insertion-unit Jacobians.

    Args:
        J_tip_actuation_tangent:
            Shape ``(3, 7)`` from
            ``model.jacobian_tip_actuation_tangent(p7)``.  Its input rate is
            ``[v_source_world, omega_source_world, Ldot]``.
        J_arm_spatial:
            Shape ``(6, n_arm)`` with rows ordered
            ``[v_reference_world, omega_world]``.
        insertion_jacobian:
            Scalar or row vector mapping insertion joint rates to ``Ldot``.
            A scalar of one represents a prismatic coordinate measured directly
            in metres of inserted length.
        source_offset_world:
            Optional vector from the arm Jacobian reference point to the magnet
            centre, expressed in world coordinates.  Omit it when
            ``J_arm_spatial`` is already evaluated at the magnet centre.

    Returns:
        Shape ``(3, n_arm + n_insertion)`` for independent controls ordered
        ``[q_arm, q_insertion]``.
    """
    J_beam = np.asarray(J_tip_actuation_tangent, float)
    J_arm = np.asarray(J_arm_spatial, float)

    if J_beam.shape != (3, 7):
        raise ValueError(
            "J_tip_actuation_tangent must have shape (3, 7), "
            f"got {J_beam.shape}."
        )
    if J_arm.ndim != 2 or J_arm.shape[0] != 6:
        raise ValueError(
            "J_arm_spatial must have shape (6, n_arm) with rows "
            f"[v_world; omega_world], got {J_arm.shape}."
        )

    Jv = J_arm[0:3, :]
    Jw = J_arm[3:6, :]

    if source_offset_world is not None:
        offset = np.asarray(source_offset_world, float).reshape(3)
        # v_source = v_reference + omega x offset
        Jv = Jv - skew(offset) @ Jw

    J_tip_arm = J_beam[:, 0:3] @ Jv + J_beam[:, 3:6] @ Jw

    J_insert = np.asarray(insertion_jacobian, float)
    if J_insert.ndim == 0:
        J_insert = J_insert.reshape(1)
    else:
        J_insert = J_insert.reshape(-1)

    if J_insert.size == 0:
        return J_tip_arm
    if not np.all(np.isfinite(J_insert)):
        raise ValueError("insertion_jacobian contains non-finite values.")

    J_tip_insert = J_beam[:, 6:7] @ J_insert.reshape(1, -1)
    return np.hstack([J_tip_arm, J_tip_insert])


class ControllerJacobianAdapter:
    """
    Direct controller interface to the beam model's implicit Jacobian.

    No beam Jacobian is reimplemented here.  The adapter calls
    ``MagneticBeamForwardModel.jacobian_tip_actuation_tangent`` and performs
    only the controller-coordinate conversion:

        omega_world = R_source @ omega_body.

    The same model instance is retained between calls, preserving its forward
    warm start, exact Jacobian cache, and local Hessian reuse.
    """

    def __init__(
        self,
        forward_model,
        *,
        angular_velocity_frame: str = "body",
        jacobian_mode: str = "fast",
    ):
        if not hasattr(forward_model, "jacobian_tip_actuation_tangent"):
            raise TypeError(
                "forward_model must provide "
                "jacobian_tip_actuation_tangent(p7)."
            )
        if angular_velocity_frame not in {"body", "world"}:
            raise ValueError(
                "angular_velocity_frame must be 'body' or 'world'."
            )
        if jacobian_mode not in {"fast", "accurate"}:
            raise ValueError("jacobian_mode must be 'fast' or 'accurate'.")

        self.model = forward_model
        self.angular_velocity_frame = angular_velocity_frame
        self.jacobian_mode = jacobian_mode
        self.last_p7: np.ndarray | None = None
        self.last_J_world: np.ndarray | None = None
        self.last_J_controller: np.ndarray | None = None
        self.last_forward_info: dict | None = None
        self.last_diag: dict = {}

    def __call__(self, p8: np.ndarray) -> np.ndarray:
        p8 = np.asarray(p8, float).reshape(8)
        p7 = pose8_quat_to_pose7_rotvec(p8)

        result = self.model.solve(
            p7,
            commit=True,
            reuse_cache=True,
        )
        if not result.success:
            raise RuntimeError(
                "Cannot linearize a failed beam equilibrium: "
                f"{result.info.get('message', 'unknown solver failure')}."
            )

        J_world = np.asarray(
            self.model.jacobian_tip_actuation_tangent(
                p7,
                solve_if_needed=False,
                mode=self.jacobian_mode,
                reuse_cached=True,
            ),
            float,
        )
        if J_world.shape != (3, 7):
            raise ValueError(
                "jacobian_tip_actuation_tangent must return shape (3, 7), "
                f"got {J_world.shape}."
            )

        J_controller = J_world.copy()

        problem = self.model.build_problem(p7)

        parts = (
            result.info.get("parts", {})
            if isinstance(result.info, dict)
            else {}
        )

        gap_nodes = parts.get("gap_nodes")
        F_nodes = parts.get("F_nodes")

        if gap_nodes is None:
            gap_nodes_array = np.empty(0, dtype=float)
        else:
            gap_nodes_array = np.asarray(
                gap_nodes,
                float,
            ).reshape(-1)

        if F_nodes is None:
            contact_force_norm = np.nan
        else:
            contact_force_norm = float(
                np.linalg.norm(
                    np.asarray(F_nodes, float)
                )
            )

        params = self.model.contact_cfg.params

        smooth_eps = (
            float(params.smooth_eps)
            if params is not None
            else np.nan
        )

        if gap_nodes_array.size:
            gap_min = float(np.min(gap_nodes_array))

            penetrating_nodes = int(
                np.count_nonzero(gap_nodes_array <= 0.0)
            )

            smoothing_band_nodes = int(
                np.count_nonzero(
                    gap_nodes_array <= 3.0 * smooth_eps
                )
            )
        else:
            gap_min = np.nan
            penetrating_nodes = 0
            smoothing_band_nodes = 0


        if self.angular_velocity_frame == "body":
            q_source = quat_normalize(p8[3:7])
            R_source = quat_to_R(q_source)
            J_controller[:, 3:6] = J_world[:, 3:6] @ R_source

        if not np.all(np.isfinite(J_controller)):
            raise FloatingPointError(
                "Controller-coordinate beam Jacobian contains non-finite values."
            )

        diag = {}
        if hasattr(self.model, "get_last_jacobian_diag"):
            diag.update(self.model.get_last_jacobian_diag())
        diag.update(
            {
                "jacobian_source": (
                    "MagneticBeamForwardModel."
                    "jacobian_tip_actuation_tangent"
                ),
                "jacobian_mode": self.jacobian_mode,
                "controller_angular_velocity_frame": (
                    self.angular_velocity_frame
                ),
                "forward_cache_hit": bool(
                    result.info.get("cache_hit", False)
                ),
                "forward_solve_path": result.info.get("solve_path"),
            }
        )
        diag.update(
            {
                "jacobian_contact_forward_enabled": bool(
                    self.model.contact_cfg.enabled
                ),
                "jacobian_contact_requested": bool(
                    self.model.contact_cfg.use_in_jacobian
                ),
                "jacobian_problem_use_contact": bool(
                    problem.use_contact
                ),
                "jacobian_problem_use_contact_in_jacobian": bool(
                    problem.use_contact_in_jacobian
                ),
                "jacobian_lumen_attached": (
                    problem.lumen_query is not None
                ),
                "jacobian_contact_energy": float(
                    parts.get("W_cf", np.nan)
                ),
                "jacobian_gap_min_m": gap_min,
                "jacobian_penetrating_nodes": penetrating_nodes,
                "jacobian_smoothing_band_nodes": smoothing_band_nodes,
                "jacobian_contact_force_norm": contact_force_norm,
            }
        )
        self.last_p7 = p7.copy()
        self.last_J_world = J_world.copy()
        self.last_J_controller = J_controller.copy()
        self.last_forward_info = copy.deepcopy(result.info)
        self.last_diag = diag

        return J_controller.copy()

    def get_last_diag(self) -> dict:
        return dict(self.last_diag)

    def reset(self) -> None:
        self.last_p7 = None
        self.last_J_world = None
        self.last_J_controller = None
        self.last_forward_info = None
        self.last_diag = {}
        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()


def make_controller_jacobian_fn(
    *,
    jacobian_model,
    dt: float,
    n_out: int = 3,
    angular_velocity_frame: str = "body",
    jacobian_mode: str = "fast",
    copy_model: bool = True,
):
    """
    Build MPC's one-sample Jacobian directly from the magnetic-beam model.

    The returned function maps

        [v_world, omega_body_or_world, Ldot]

    to the one-sample tip displacement and therefore returns

        B = dt * d(tip)/d[v, omega, Ldot].
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if int(n_out) != 3:
        raise ValueError(
            "The magnetic-beam model returns a 3D tip Jacobian, so n_out "
            f"must be 3; got {n_out}."
        )

    owned_model = (
        copy.deepcopy(jacobian_model)
        if copy_model
        else jacobian_model
    )
    adapter = ControllerJacobianAdapter(
        owned_model,
        angular_velocity_frame=angular_velocity_frame,
        jacobian_mode=jacobian_mode,
    )

    def J_fn(p8):
        J_state = adapter(p8)
        B = dt * np.asarray(J_state, float).reshape(3, 7)
        if not np.all(np.isfinite(B)):
            raise FloatingPointError(
                "One-sample controller Jacobian contains non-finite values."
            )
        return B

    J_fn.get_last_diag = adapter.get_last_diag
    J_fn.reset = adapter.reset
    J_fn.adapter = adapter
    J_fn.model = adapter.model
    diag = J_fn.get_last_diag()


    return J_fn


class ControllerForwardAdapter:
    """
    Adapter between the controller/MPC pose convention and MagneticBeamForwardModel.

    Controller input:
        p8 = [x, y, z, qw, qx, qy, qz, L]

    Model input:
        p7 = [x, y, z, rx, ry, rz, L]

    Output:
        y = [tip_x, tip_y, tip_z, tangent_x, tangent_y, tangent_z]
    """

    def __init__(self, forward_model):
        self.model = forward_model
        self._baseline_cache = None

        self.last_tip: np.ndarray | None = None
        self.last_p_centerline: np.ndarray | None = None
        self.last_info: dict | None = None

    def start_step(self) -> None:
        """
        Freeze the model state at the start of an MPC/controller step.
        """
        self._baseline_cache = self.model.get_cache_copy()

    def reset(self) -> None:
        """
        Clear adapter and model cache.
        """
        self._baseline_cache = None
        self.last_tip = None
        self.last_p_centerline = None
        self.last_info = None

        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

    def __call__(self, p8: np.ndarray, *, commit: bool = False) -> np.ndarray:
        """
        Evaluate model and return [tip_xyz, tip_tangent].

        commit=False:
            Evaluate from the current step baseline, then restore the model.

        commit=True:
            Evaluate and keep the new solve as the model warm-start/cache.
        """
        p7 = pose8_quat_to_pose7_rotvec(p8)

        if self._baseline_cache is None:
            self.start_step()

        if commit:
            result = self.model.solve(p7, commit=True)
            self._baseline_cache = self.model.get_cache_copy()
        else:
            old_cache = self.model.get_cache_copy()
            try:
                self.model.set_cache(self._baseline_cache)
                result = self.model.solve(p7, commit=True)
            finally:
                self.model.set_cache(old_cache)

        self.last_tip = result.tip.copy()
        self.last_p_centerline = result.p.copy()
        self.last_info = copy.deepcopy(result.info)

        tangent = tip_tangent_from_centerline(result.p)

        return np.hstack([result.tip, tangent])


class TipOnlyAdapter:
    """
    Adapter for controllers that only need tip position.

    Input:
        p8 = [x, y, z, qw, qx, qy, qz, L]

    Output:
        tip_xyz
    """

    def __init__(self, forward_model):
        self.model = forward_model

    def __call__(self, p8: np.ndarray, *, commit: bool = False) -> np.ndarray:
        p7 = pose8_quat_to_pose7_rotvec(p8)

        if commit:
            return self.model.solve(p7, commit=True).tip.copy()

        old_cache = self.model.get_cache_copy()
        try:
            result = self.model.solve(p7, commit=True)
            return result.tip.copy()
        finally:
            self.model.set_cache(old_cache)


class Pose7TipTangentAdapter:
    """
    Adapter for code that already uses p7 directly.

    Input:
        p7 = [x, y, z, rx, ry, rz, L]

    Output:
        y = [tip_xyz, tip_tangent]
    """

    def __init__(self, forward_model):
        self.model = forward_model

        self.last_tip: np.ndarray | None = None
        self.last_p_centerline: np.ndarray | None = None
        self.last_info: dict | None = None

    def __call__(self, p7: np.ndarray, *, commit: bool = False) -> np.ndarray:
        if commit:
            result = self.model.solve(p7, commit=True)
        else:
            old_cache = self.model.get_cache_copy()
            try:
                result = self.model.solve(p7, commit=True)
            finally:
                self.model.set_cache(old_cache)

        self.last_tip = result.tip.copy()
        self.last_p_centerline = result.p.copy()
        self.last_info = copy.deepcopy(result.info)

        tangent = tip_tangent_from_centerline(result.p)

        return np.hstack([result.tip, tangent])
