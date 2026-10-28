"""Joint-space adapter for the magnetic-beam MPC.

This module keeps the controller independent of a particular UR or beam-model
class while enforcing one unambiguous interface:

    state   = [q1, q2, q3, q4, q5, q6, insertion_length]
    control = [qd1, qd2, qd3, qd4, qd5, qd6, insertion_rate]

All positions, tangents, twists, and Jacobians must be expressed in the same
robot-base/world frame.  The robot geometric Jacobian must be evaluated at the
offset magnet centre and ordered [linear; angular].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


Array = np.ndarray


def _state7(value: Any, name: str = "state") -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 7 or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must be finite [q1..q6, insertion] with shape (7,), "
            f"got {result.shape}."
        )
    return result.copy()


def _transform(value: Any, name: str) -> Array:
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 transform.")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{name} has an invalid homogeneous last row.")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1.0e-8):
        raise ValueError(f"{name} rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1.0e-8):
        raise ValueError(f"{name} rotation determinant is not +1.")
    return T.copy()


@dataclass
class JointSpaceBeamMPCAdapter:
    """Compose robot differential kinematics with beam sensitivity.

    Required callbacks
    ------------------
    magnet_transform_fn:
        ``magnet_transform_fn(q6) -> T_R_M``.  The returned transform must
        already contain the complete flange/TCP-to-magnet offset.

    magnet_geometric_jacobian_fn:
        ``magnet_geometric_jacobian_fn(q6) -> J_M`` with shape (6, 6), row
        order [world linear; world angular], evaluated at the magnet centre.

    beam_output_fn:
        ``beam_output_fn(T_R_M, insertion, commit=False) -> y``.  A callback
        without the ``commit`` keyword is also accepted.

    beam_output_jacobian_fn:
        ``beam_output_jacobian_fn(T_R_M, insertion) -> J_B``.  ``J_B`` has
        seven columns ordered [world translation, small world rotation,
        insertion].  Its output rows must match ``beam_output_fn``.

    The continuous chain-rule Jacobian is

        J_yu = J_B @ blockdiag(J_M, 1).

    The MPC controller multiplies this by ``dt`` exactly once.
    """

    magnet_transform_fn: Callable[[Array], Array]
    magnet_geometric_jacobian_fn: Callable[[Array], Array]
    beam_output_fn: Callable[..., Array]
    beam_output_jacobian_fn: Callable[[Array, float], Array]
    output_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    model: Any | None = None
    last_diag: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        indices = tuple(int(i) for i in self.output_indices)
        if not indices:
            raise ValueError("output_indices cannot be empty.")
        if min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("output_indices must be unique and non-negative.")
        self.output_indices = indices

    @property
    def n_out(self) -> int:
        return len(self.output_indices)

    def start_step(self) -> None:
        """Forward an optional model step hook used by continuation caches."""
        hook = getattr(self.beam_output_fn, "start_step", None)
        if callable(hook):
            hook()

    def magnet_transform(self, state: Any) -> Array:
        state = _state7(state)
        return _transform(
            self.magnet_transform_fn(state[:6]), "robot-base magnet transform"
        )

    def magnet_geometric_jacobian(self, state: Any) -> Array:
        state = _state7(state)
        J = np.asarray(
            self.magnet_geometric_jacobian_fn(state[:6]), dtype=float
        ).reshape(6, 6)
        if not np.all(np.isfinite(J)):
            raise FloatingPointError(
                "magnet_geometric_jacobian_fn returned non-finite values."
            )
        return J

    def _beam_output(self, T_R_M: Array, insertion: float, commit: bool) -> Array:
        try:
            output = self.beam_output_fn(T_R_M, insertion, commit=commit)
        except TypeError:
            output = self.beam_output_fn(T_R_M, insertion)
        output = np.asarray(output, dtype=float).reshape(-1)
        required = max(self.output_indices) + 1
        if output.size < required or not np.all(np.isfinite(output)):
            raise FloatingPointError(
                "beam_output_fn returned an invalid output: "
                f"shape={output.shape}, required rows={required}."
            )
        return output

    def forward_output(self, state: Any, commit: bool = False) -> Array:
        """Return the selected nonlinear beam output from ``[q,L]``."""
        state = _state7(state)
        T_R_M = self.magnet_transform(state)
        full = self._beam_output(T_R_M, float(state[6]), bool(commit))
        selected = full[np.asarray(self.output_indices, dtype=int)]
        self.last_diag.update(
            {
                "joint_space_adapter": 1,
                "magnet_position_R_m": T_R_M[:3, 3].copy(),
                "insertion_length_m": float(state[6]),
            }
        )
        return selected

    def continuous_output_jacobian(self, state: Any) -> Array:
        """Return ``d output / d[qd1..qd6,dL]`` before time discretisation."""
        state = _state7(state)
        T_R_M = self.magnet_transform(state)
        J_M = self.magnet_geometric_jacobian(state)
        J_B_full = np.asarray(
            self.beam_output_jacobian_fn(T_R_M, float(state[6])), dtype=float
        )
        if J_B_full.ndim != 2 or J_B_full.shape[1] != 7:
            raise ValueError(
                "beam_output_jacobian_fn must return shape (N, 7), got "
                f"{J_B_full.shape}."
            )
        required = max(self.output_indices) + 1
        if J_B_full.shape[0] < required or not np.all(np.isfinite(J_B_full)):
            raise FloatingPointError(
                "beam_output_jacobian_fn returned invalid output rows."
            )

        G = np.zeros((7, 7), dtype=float)
        G[:6, :6] = J_M
        G[6, 6] = 1.0
        J_B = J_B_full[np.asarray(self.output_indices, dtype=int), :]
        J_yu = J_B @ G

        singular_values = np.linalg.svd(J_yu, compute_uv=False)
        self.last_diag.update(
            {
                "joint_space_adapter": 1,
                "J_beam_actuation": J_B.copy(),
                "J_magnet_joint": J_M.copy(),
                "J_output_joint_insertion": J_yu.copy(),
                "J_output_joint_insertion_singular_values": singular_values,
            }
        )
        return J_yu

    def __call__(self, state: Any) -> Array:
        """Allow the adapter itself to be passed as ``Jxy_fn``."""
        return self.continuous_output_jacobian(state)

    def magnet_position(self, state: Any) -> Array:
        """Callback for joint-space magnet-clearance constraints."""
        return self.magnet_transform(state)[:3, 3].copy()

    def magnet_position_jacobian(self, state: Any) -> Array:
        """Return ``d p_M / d[q1..q6,L]``, shape (3, 7)."""
        J = np.zeros((3, 7), dtype=float)
        J[:, :6] = self.magnet_geometric_jacobian(state)[:3, :]
        return J

    def finite_difference_output_jacobian(
        self,
        state: Any,
        *,
        joint_step_rad: float = 1.0e-6,
        insertion_step_m: float = 1.0e-6,
    ) -> Array:
        """Finite-difference the complete joint/insertion-to-beam mapping."""
        state = _state7(state)
        steps = np.full(7, float(joint_step_rad), dtype=float)
        steps[6] = float(insertion_step_m)
        if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
            raise ValueError("Finite-difference steps must be finite and positive.")

        J_fd = np.empty((self.n_out, 7), dtype=float)
        for coordinate, step in enumerate(steps):
            delta = np.zeros(7, dtype=float)
            delta[coordinate] = step
            plus = self.forward_output(state + delta, commit=False)
            minus = self.forward_output(state - delta, commit=False)
            J_fd[:, coordinate] = (plus - minus) / (2.0 * step)
        return J_fd

    def validate_chain_rule(
        self,
        state: Any,
        *,
        joint_step_rad: float = 1.0e-6,
        insertion_step_m: float = 1.0e-6,
    ) -> dict[str, Any]:
        """Compare the analytical chain against complete finite differences."""
        analytical = self.continuous_output_jacobian(state)
        finite_difference = self.finite_difference_output_jacobian(
            state,
            joint_step_rad=joint_step_rad,
            insertion_step_m=insertion_step_m,
        )
        difference = analytical - finite_difference
        relative = float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(finite_difference), 1.0e-12)
        )
        result = {
            "analytical": analytical,
            "finite_difference": finite_difference,
            "difference": difference,
            "relative_frobenius_error": relative,
            "maximum_absolute_element_error": float(np.max(np.abs(difference))),
        }
        self.last_diag.update(
            {
                "joint_space_chain_relative_error": relative,
                "joint_space_chain_max_abs_error": result[
                    "maximum_absolute_element_error"
                ],
            }
        )
        return result

    def get_last_diag(self) -> dict[str, Any]:
        return dict(self.last_diag)


class _ForwardOutputProxy:
    """Callable proxy preserving model/adapter hooks expected by the controller."""

    def __init__(self, adapter: JointSpaceBeamMPCAdapter):
        self.adapter = adapter
        self.model = adapter.model

    def __call__(self, state: Any, commit: bool = False) -> Array:
        return self.adapter.forward_output(state, commit=commit)

    def start_step(self) -> None:
        self.adapter.start_step()


def controller_callbacks(
    adapter: JointSpaceBeamMPCAdapter,
) -> dict[str, Callable[..., Array]]:
    """Return the four callbacks consumed by ``MPCControllerTipXY``."""
    return {
        "Jxy_fn": adapter,
        "forward_tip_fn": _ForwardOutputProxy(adapter),
        "magnet_position_fn": adapter.magnet_position,
        "magnet_position_jacobian_fn": adapter.magnet_position_jacobian,
    }
