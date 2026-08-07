from __future__ import annotations

import copy

import numpy as np

from proper_research.simulation_controller.geometry import (
    pose8_quat_to_pose7_rotvec,
    quat_to_R,
    quat_wxyz_normalize,
)


def make_controller_output_jacobian_fn(
    *,
    jacobian_model,
    dt: float,
    n_out: int,
    angular_velocity_frame: str = "body",
    jacobian_mode: str = "fast",
    copy_model: bool = True,
):
    """Build the MPC one-step Jacobian for 3D or 6D beam output.

    Output conventions
    ------------------
    n_out=3:
        [tip_x, tip_y, tip_z]
    n_out=6:
        [tip_x, tip_y, tip_z, tangent_x, tangent_y, tangent_z]

    The model performs one nominal equilibrium solve at the requested pose. Its
    optimized implicit sensitivity then forms both position and tangent rows
    from the same H, Gtheta and linear solve. No perturbed equilibrium solves are
    introduced by this adapter.
    """
    dt = float(dt)
    n_out = int(n_out)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if n_out not in {3, 6}:
        raise ValueError("n_out must be 3 or 6.")
    if angular_velocity_frame not in {"body", "world"}:
        raise ValueError("angular_velocity_frame must be 'body' or 'world'.")
    if jacobian_mode not in {"fast", "accurate"}:
        raise ValueError("jacobian_mode must be 'fast' or 'accurate'.")

    model = copy.deepcopy(jacobian_model) if copy_model else jacobian_model
    last_diag: dict = {}

    def J_fn(p8):
        nonlocal last_diag
        p8 = np.asarray(p8, dtype=float).reshape(8)
        p7 = pose8_quat_to_pose7_rotvec(p8)

        # Exactly one nominal equilibrium solve for this Jacobian-model pose.
        # An exact model-cache hit avoids even that solve when the pose repeats.
        model.solve(p7, commit=True, reuse_cache=True)

        if not hasattr(model, "jacobian_output_actuation_tangent"):
            if n_out == 3:
                J_world = np.asarray(
                    model.jacobian_tip_actuation_tangent(
                        p7,
                        solve_if_needed=False,
                        mode=jacobian_mode,
                    ),
                    dtype=float,
                ).reshape(3, 7)
            else:
                raise AttributeError(
                    "n_out=6 requires model.jacobian_output_actuation_tangent(). "
                    "Install the updated optimized sensitivity and forward model."
                )
        else:
            J_world6 = np.asarray(
                model.jacobian_output_actuation_tangent(
                    p7,
                    solve_if_needed=False,
                    mode=jacobian_mode,
                ),
                dtype=float,
            ).reshape(6, 7)
            J_world = J_world6[:n_out, :]

        J_translation = J_world[:, 0:3]
        J_rotation = J_world[:, 3:6]
        J_length = J_world[:, 6:7]

        if angular_velocity_frame == "body":
            q_src = quat_wxyz_normalize(p8[3:7])
            J_rotation = J_rotation @ quat_to_R(q_src)

        J_control_state = np.concatenate(
            (J_translation, J_rotation, J_length),
            axis=1,
        )
        B = dt * J_control_state
        if B.shape != (n_out, 7):
            raise RuntimeError(
                f"Controller Jacobian has shape {B.shape}; expected {(n_out, 7)}."
            )
        if not np.all(np.isfinite(B)):
            raise FloatingPointError("Controller Jacobian contains non-finite values.")

        if hasattr(model, "get_last_jacobian_diag"):
            last_diag = dict(model.get_last_jacobian_diag())
        else:
            last_diag = dict(getattr(model, "last_jacobian_diag", {}) or {})
        last_diag["controller_output_dimension"] = n_out
        return B

    def get_last_diag():
        return dict(last_diag)

    J_fn.get_last_diag = get_last_diag
    J_fn.model = model
    J_fn.n_out = n_out
    return J_fn
