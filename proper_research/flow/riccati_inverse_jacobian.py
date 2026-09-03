"""DARE / Riccati-augmented resolved-rate beam-tip controller, flow-fed.

Builds on ``controllers/inverse_jacobian_controller.py`` (left untouched) and
replaces its hand-tuned proportional gain with a discrete-algebraic-Riccati
feedback law that has integral action.

Formulation
-----------
The flow ring measures the transverse tip error
``y_k = P (p*_k - p_tip(z_k))``, where ``P`` (2x3) projects onto the plane
perpendicular to the centreline tangent.  A commanded transverse tip velocity
``v_k`` moves it as ``y_{k+1} = y_k - dt v_k``.  Augment with the integral of
``y`` and pick ``v`` by a DARE (this is a Riccati-optimal PI law on the tip
error -- a principled replacement for the resolved-rate proportional gain):

    x = [ y ; xi ]        xi_{k+1} = xi_k + dt y_k

    A = [[ I2   , 0  ]]   B = [[ -dt I2 ]]   Q = diag(Q_y, Q_i)   R = R_v
        [[ dt I2, I2 ]]       [[ 0      ]]

    P = A^T P A - A^T P B (R + B^T P B)^-1 B^T P A + Q
    K = (R + B^T P B)^-1 B^T P A                              (2x4, constant)
    v = -K [y ; xi]

The commanded tip velocity is then realised through the *existing* beam
Jacobian, exactly as the resolved-rate baseline does:

    J_t   = P J           (transverse beam Jacobian, 2x7 -- insertion column
                           dropped when ``insertion_locked``)
    w     = J_t^+ v  +  (I - J_t^+ J_t) k_n (z_ref - z)/dt
    u     = u_ref + w                                         then clipped

* The **integral term rejects the plant-model mismatch and the Dean-flow bias**
  -- both biases; a proportional law would leave a steady-state offset.
* ``J_t^+`` is a lightly-damped least-squares inverse; the nullspace projector
  uses the *undamped* ``J_t^+`` so it is an exact projector and never fights the
  tip correction.  The redundant four joint directions are regulated toward the
  reference configuration.
* **Insertion is locked out** of the feedback: the advance axis follows the
  feedforward schedule and is never recruited for centring.
* Clipping (velocity -> acceleration -> state box) is identical to the baseline.
* The controller **never sees the true tip** -- only ``measured_beam_position``,
  filled by the closed-loop runner from the simulated flow ring.

Because ``A`` and ``B`` of the DARE do not contain ``J_t``, the gain ``K`` is a
fixed optimal PI matrix; ``J_t`` enters only through ``J_t^+``.  This is itself
a result: the Riccati gain for tip-error regulation is Jacobian-independent, and
what the beam model buys is the mapping to joints, not the gain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.linalg import solve_discrete_are

try:
    from proper_research.controllers.beam_jacobian_providers import (
        declare_provider, provenance_line,
    )
    from proper_research.controllers.inverse_jacobian_controller import (
        InverseJacobianStep,
    )
except ImportError:  # pragma: no cover
    from controllers.beam_jacobian_providers import (  # type: ignore
        declare_provider, provenance_line,
    )
    from controllers.inverse_jacobian_controller import InverseJacobianStep  # type: ignore

Array = np.ndarray


def transverse_projection(tangent: Array) -> Array:
    """2x3 matrix whose rows span the plane perpendicular to ``tangent``.

    The in-plane basis is anchored to world ``+z`` (not to the tangent) so it is
    a smooth, flip-free function of the tangent for any catheter that never
    points near vertical.
    """
    t = np.asarray(tangent, dtype=float).reshape(3)
    n = np.linalg.norm(t)
    if n < 1e-12:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = t / n
    seed = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, t) * t
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return np.stack([u, v], axis=0)


@dataclass(frozen=True)
class RiccatiConfig:
    """Weights and options for the DARE feedback (dimensionless multipliers)."""

    tip_error_scale_m: float = 1.0e-4                       # 0.1 mm
    tip_velocity_scale_m_s: float = 2.0e-3                  # 2 mm/s transverse authority

    # Tuned for robustness to the plant-vs-model Jacobian mismatch and the
    # transverse frame rotating through a bend.  Two findings drove this:
    #   * a near-deadbeat proportional gain has no margin for the mismatch and
    #     oscillates -- hence the modest tip_tracking / tip_velocity ratio;
    #   * a strong integral term goes lightly-damped-complex against the same
    #     mismatch (a ~2 s limit cycle), so the integral is kept very small --
    #     just enough to trim a slow bias without setting up the oscillation.
    # Result on the real plant: monotone convergence of a 0.67 mm offset to
    # ~30 um, held.
    tip_tracking_weight: float = 3.0e2
    integral_weight: float = 2.0e0
    tip_velocity_weight: float = 8.0e0

    nullspace_gain: float = 0.10                            # pull the nullspace to z_ref
    integral_clamp_m_s: float = 2.0e-3                      # anti-windup on xi
    damping: float = 3.0e-3                                 # Levenberg parameter for J_t^+ v
    feedforward: bool = True
    insertion_locked: bool = True

    def validate(self) -> None:
        if self.tip_error_scale_m <= 0 or self.tip_velocity_weight <= 0:
            raise ValueError("scales and tip_velocity_weight must be positive.")
        if self.damping <= 0:
            raise ValueError("damping is a Levenberg parameter; must be positive.")


@dataclass
class RiccatiDiagnostics:
    spectral_radius: float = float("nan")
    integral_norm_m_s: float = float("nan")
    residual_estimate_m: float = float("nan")
    tip_velocity_norm_m_s: float = float("nan")


class RiccatiInverseJacobianController:
    """Resolved-rate inverse-Jacobian control with a DARE PI feedback gain."""

    name = "riccati_inverse_jacobian"
    description = "DARE-augmented resolved-rate inverse Jacobian, flow-fed"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        sample_period_s: float,
        velocity_limit: Any,
        acceleration_limit: Any,
        state_min: Any,
        state_max: Any,
        config: RiccatiConfig | None = None,
        allow_undeclared_jacobian: bool = True,
    ) -> None:
        self.reference = reference
        self.jacobian_provider, self.jacobian_provenance = declare_provider(
            jacobian_provider, allow_undeclared=allow_undeclared_jacobian
        )
        self.jacobian_is_contact_free = (
            self.jacobian_provenance.get("contact_used_in_jacobian") is False
        )
        self.dt = float(sample_period_s)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("sample_period_s must be finite and positive.")
        self.velocity_limit = np.asarray(velocity_limit, float).reshape(7)
        self.acceleration_limit = np.asarray(acceleration_limit, float).reshape(7)
        self.state_min = np.asarray(state_min, float).reshape(7)
        self.state_max = np.asarray(state_max, float).reshape(7)
        self.cfg = config or RiccatiConfig()
        self.cfg.validate()

        self._ctrl_axes = np.arange(6) if self.cfg.insertion_locked else np.arange(7)
        self._K, self._spectral = self._solve_dare()
        self.diagnostics = RiccatiDiagnostics(spectral_radius=self._spectral)
        # integral of the transverse tip error, accumulated as a WORLD-frame
        # 3-vector so it stays consistent as the centreline tangent (and hence
        # the transverse frame) rotates through a bend
        self.xi_world = np.zeros(3, dtype=float)

    # -- DARE (scaled, Jacobian-independent) ---------------------------
    def _solve_dare(self) -> tuple[Array, float]:
        dt = self.dt
        s_y = self.cfg.tip_error_scale_m
        s_v = self.cfg.tip_velocity_scale_m_s
        # scaled model: ytil = y/s_y, vtil = v/s_v  ->  ytil_{k+1} = ytil - dt (s_v/s_y) vtil
        b = dt * s_v / s_y
        A = np.block([[np.eye(2), np.zeros((2, 2))],
                      [dt * np.eye(2), np.eye(2)]])
        B = np.vstack([-b * np.eye(2), np.zeros((2, 2))])
        Q = np.diag([self.cfg.tip_tracking_weight] * 2 + [self.cfg.integral_weight] * 2)
        R = self.cfg.tip_velocity_weight * np.eye(2)
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)     # 2 x 4 (scaled)
        rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
        return K, rho

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "description": self.description,
            "jacobian": provenance_line(self.jacobian_provenance),
            "spectral_radius": self._spectral,
            "tip_tracking_weight": self.cfg.tip_tracking_weight,
            "integral_weight": self.cfg.integral_weight,
            "tip_velocity_weight": self.cfg.tip_velocity_weight,
            "nullspace_gain": self.cfg.nullspace_gain,
            "insertion_locked": self.cfg.insertion_locked,
        }

    def reset(self) -> None:
        self.xi_world = np.zeros(3, dtype=float)
        self.diagnostics = RiccatiDiagnostics(spectral_radius=self._spectral)

    def _reference_index(self, control_index: int) -> int:
        return int(np.clip(int(control_index) + 1, 0, self.reference.sample_count - 1))

    def solve(
        self,
        *,
        measured_state: Any,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> InverseJacobianStep:
        started = time.perf_counter()
        state = np.asarray(measured_state, float).reshape(7)
        measured_tip = np.asarray(measured_beam_position, float).reshape(3)
        previous = np.asarray(previous_input, float).reshape(7)
        index = self._reference_index(control_index)

        desired = np.asarray(self.reference.desired_position_m, float)[index]
        tangent = np.asarray(self.reference.desired_tangent, float)[index]
        z_ref = np.asarray(self.reference.state, float)[index]
        u_ref = np.asarray(self.reference.input, float)[index]

        Pm = transverse_projection(tangent)                    # 2 x 3
        J = np.asarray(self.jacobian_provider(state), float).reshape(3, 7)
        if not np.all(np.isfinite(J)):
            raise FloatingPointError("The beam Jacobian is not finite.")
        Jt = (Pm @ J)[:, self._ctrl_axes]                      # 2 x nw

        y = Pm @ (desired - measured_tip)                      # transverse error toward centreline

        # accumulate the integral in the world frame, then re-express it in the
        # current transverse frame -- so a rotating frame through the bend does
        # not corrupt the accumulated correction
        self.xi_world = self.xi_world + self.dt * (Pm.T @ y)
        norm = float(np.linalg.norm(self.xi_world))
        if norm > self.cfg.integral_clamp_m_s and norm > 0:
            self.xi_world *= self.cfg.integral_clamp_m_s / norm
        xi = Pm @ self.xi_world                                # 2, current frame
        self.diagnostics.integral_norm_m_s = float(np.linalg.norm(xi))
        self.diagnostics.residual_estimate_m = float(np.linalg.norm(y))

        # scaled DARE feedback -> commanded transverse tip velocity
        s_y, s_v = self.cfg.tip_error_scale_m, self.cfg.tip_velocity_scale_m_s
        x_tilde = np.concatenate([y, xi]) / s_y
        v = s_v * (-self._K @ x_tilde)                         # 2, m/s
        self.diagnostics.tip_velocity_norm_m_s = float(np.linalg.norm(v))

        # inverse-Jacobian mapping to joint velocities
        gram = Jt @ Jt.T + (self.cfg.damping ** 2) * np.eye(2)
        Jt_pinv_damped = Jt.T @ np.linalg.solve(gram, np.eye(2))
        w_task = Jt_pinv_damped @ v                            # nw

        # exact nullspace projector (undamped) toward the reference configuration
        w_null = np.zeros(self._nw)
        if self.cfg.nullspace_gain > 0.0:
            try:
                Jt_pinv = Jt.T @ np.linalg.solve(Jt @ Jt.T, np.eye(2))
                projector = np.eye(self._nw) - Jt_pinv @ Jt
                dz_c = (z_ref - state)[self._ctrl_axes]
                w_null = projector @ (self.cfg.nullspace_gain * dz_c / self.dt)
            except np.linalg.LinAlgError:
                w_null = np.zeros(self._nw)

        command = np.zeros(7)
        command[self._ctrl_axes] = w_task + w_null
        if self.cfg.feedforward:
            command = command + u_ref

        command = self._clip(command, state, previous)
        elapsed = time.perf_counter() - started

        predicted_state = state + self.dt * command
        predicted_tip = measured_tip + J @ (self.dt * command)
        return InverseJacobianStep(
            command=command,
            planned_input=u_ref.copy(),
            predicted_states=predicted_state.reshape(1, 7),
            predicted_inputs=command.reshape(1, 7),
            objective=float(np.linalg.norm(y)),
            status="riccati_dare",
            success=True,
            iterations=1,
            solve_time_s=float(elapsed),
            primal_residual=float(self._spectral),
            dual_residual=float(self.diagnostics.integral_norm_m_s),
            measured_beam_position=measured_tip.copy(),
            instantaneous_output_residual=np.array([y[0], y[1], np.nan]),
            estimated_output_residual=np.full(3, np.nan),
            predicted_beam_positions=predicted_tip.reshape(1, 3),
            predicted_beam_errors=(predicted_tip - desired).reshape(1, 3),
            first_predicted_beam_error_m=float(np.linalg.norm(predicted_tip - desired)),
        )

    @property
    def _nw(self) -> int:
        return int(self._ctrl_axes.size)

    def _clip(self, command: Array, state: Array, previous: Array) -> Array:
        command = np.clip(command, -self.velocity_limit, self.velocity_limit)
        command = np.clip(
            command,
            previous - self.acceleration_limit * self.dt,
            previous + self.acceleration_limit * self.dt,
        )
        command = np.clip(
            command,
            (self.state_min - state) / self.dt,
            (self.state_max - state) / self.dt,
        )
        return np.clip(command, -self.velocity_limit, self.velocity_limit)


def build_riccati_inverse_jacobian_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    mpc_config: Any,
    config: RiccatiConfig | None = None,
) -> RiccatiInverseJacobianController:
    """Build from the same ``ConfigurationMPCConfig`` the MPCs use."""
    return RiccatiInverseJacobianController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        sample_period_s=float(mpc_config.sample_period_s),
        velocity_limit=mpc_config.velocity_limit,
        acceleration_limit=mpc_config.acceleration_limit,
        state_min=mpc_config.state_min,
        state_max=mpc_config.state_max,
        config=config,
    )


# --------------------------------------------------------------------------
# self-test -- no project imports, mock reference + Jacobian
# --------------------------------------------------------------------------
def _self_test() -> bool:
    ok = True
    dt = 0.01
    n = 300

    class _Ref:
        sample_count = n
        time_s = np.arange(n) * dt
        state = np.zeros((n, 7))
        input = np.zeros((n, 7))
        desired_position_m = np.zeros((n, 3))
        desired_tangent = np.tile([1.0, 0.0, 0.0], (n, 1))

    J0 = np.zeros((3, 7))
    J0[1, 0] = 0.02
    J0[2, 1] = 0.02
    J0[0, 6] = 1.0

    ctrl = RiccatiInverseJacobianController(
        reference=_Ref(), jacobian_provider=lambda s: J0, sample_period_s=dt,
        velocity_limit=np.array([0.5] * 6 + [0.2]),
        acceleration_limit=np.array([10.0] * 6 + [1.0]),
        state_min=np.array([-3.0] * 6 + [-1.0]),
        state_max=np.array([3.0] * 6 + [1.0]),
        config=RiccatiConfig(),
    )
    t1 = ctrl._spectral < 1.0
    print(f"  [{'ok  ' if t1 else 'FAIL'}] DARE closed loop stable  rho = {ctrl._spectral:.4f}")
    ok &= t1

    d = np.array([0.0, 1.5e-3, -1.0e-3])
    state, prev = np.zeros(7), np.zeros(7)
    errs = []
    for k in range(n - 1):
        true_tip = J0 @ state + d
        step = ctrl.solve(measured_state=state, measured_beam_position=true_tip,
                          control_index=k, previous_input=prev)
        state = state + dt * step.command
        prev = step.command
        errs.append(float(np.linalg.norm(true_tip[1:])))
    settle = next((i for i, e in enumerate(errs) if e < 0.05 * errs[0]), n)
    t2 = errs[-1] < 0.05 * errs[0] and min(errs) < 5e-5
    print(f"  [{'ok  ' if t2 else 'FAIL'}] flow-only DARE rejects a constant tip bias  "
          f"{1e3*errs[0]:.3f} mm -> {1e6*errs[-1]:.2f} um (5% settle {settle*dt:.2f} s)")
    ok &= t2

    t3 = float(np.max(np.abs(state[2:6]))) < 0.3
    print(f"  [{'ok  ' if t3 else 'FAIL'}] redundant joints stay bounded  "
          f"max |q3..q6| = {np.max(np.abs(state[2:6])):.4f}")
    ok &= t3

    # a rotated transverse frame (tangent not along x) must still converge
    class _Ref2(_Ref):
        desired_tangent = np.tile(np.array([0.7, 0.0, 0.714]) /
                                  np.linalg.norm([0.7, 0.0, 0.714]), (n, 1))
    J1 = np.zeros((3, 7))
    J1[0, 0] = 0.02
    J1[1, 1] = 0.02
    J1[2, 2] = 0.02
    ctrl2 = RiccatiInverseJacobianController(
        reference=_Ref2(), jacobian_provider=lambda s: J1, sample_period_s=dt,
        velocity_limit=np.array([0.5] * 6 + [0.2]),
        acceleration_limit=np.array([10.0] * 6 + [1.0]),
        state_min=np.array([-3.0] * 6 + [-1.0]),
        state_max=np.array([3.0] * 6 + [1.0]))
    state, prev = np.zeros(7), np.zeros(7)
    d2 = np.array([1.0e-3, -0.8e-3, 0.5e-3])
    e0 = e1 = 0.0
    Pm = transverse_projection(_Ref2.desired_tangent[0])
    for k in range(n - 1):
        tt = J1 @ state + d2
        yv = float(np.linalg.norm(Pm @ tt))
        if k == 0:
            e0 = yv
        e1 = yv
        step = ctrl2.solve(measured_state=state, measured_beam_position=tt,
                           control_index=k, previous_input=prev)
        state = state + dt * step.command
        prev = step.command
    t4 = e1 < 0.05 * e0
    print(f"  [{'ok  ' if t4 else 'FAIL'}] converges in a rotated transverse frame  "
          f"{1e3*e0:.3f} -> {1e6*e1:.1f} um")
    ok &= t4

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    ok = _self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "RiccatiInverseJacobianController",
    "RiccatiConfig",
    "RiccatiDiagnostics",
    "build_riccati_inverse_jacobian_controller",
    "transverse_projection",
]
