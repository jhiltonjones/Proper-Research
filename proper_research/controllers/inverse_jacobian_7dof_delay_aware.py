"""Matched-authority 7DOF inverse: normalized-DLS extension of
`inverse_jacobian_2dof_delay_aware.py` giving the insertion channel real
feedback authority, so the inverse-vs-MPC comparison is no longer confounded
by "MPC has 7 feedback actuators, inverse only has 6."

Everything about the delay-aware preview (d_k, z_hat_base, e_pred, the
realization index r=k+3) is copied byte-for-byte from
`TwoDOFDelayAwareController` -- see that module's docstring for the full
derivation/validation of the preview architecture. The ONLY thing that
changes here is the feedback-allocation step: instead of a 6-column
un-normalized DLS pseudoinverse (joints only, `u_L = u_ref,L` fixed), this
controller solves a 7-column DLS pseudoinverse in NORMALIZED actuator
coordinates and lets insertion carry a feedback trim too.

Why normalized coordinates
---------------------------
Joints are in rad/s, insertion is in m/s -- six orders of magnitude apart in
raw units. A raw 7-column DLS pinv `J^+` would implicitly compare those units
directly (nothing in a bare least-squares solve knows insertion's "0.002" is
a large fraction of its own authority while a joint's "0.002" is a small
fraction of its own authority), so which channel absorbs a given task error
would depend on unit choice, not on physical actuator authority. Fixed with

    S_u = diag(0.05, ..., 0.05, 0.005)      (rad/s x6, m/s -- SAME convention
                                              as this project's `input_error_scale`,
                                              e.g. `run_level_stats.py`'s S_U)
    J_u = J @ S_u
    J_u^+ = J_u^T (J_u J_u^T + lambda_u^2 I)^-1
    delta_u = S_u @ (J_u^+ v_task)

Damping rescaling (exact-reduction requirement)
-------------------------------------------------
`lambda_u` is NOT reused numerically equal to the un-normalized `damping`
(5e-2, the existing INV-6 value): because S_u's joint entries are a uniform
scalar (0.05 on every one of the 6 joint columns), zeroing the insertion
column recovers INV-6 exactly (see docstring derivation + the unit test
`test_disabled_insertion_reproduces_inv6`) ONLY if

    lambda_u = damping_existing * su_joint     (= 5e-2 * 0.05 = 2.5e-3)

Derivation: for a uniform column scale c on a submatrix (here c=su_joint on
the 6 joint columns, with the insertion column zeroed), (cJ)^+_{lambda_u} =
(1/c) J^+_{lambda_u/c}. Setting lambda_u/c = damping_existing reproduces
INV-6's Gram matrix exactly (up to a scalar factor that cancels against the
S_u rescale on the way back out), so `lambda_u = c * damping_existing` is the
unique choice making "insertion feedback disabled" bit-identical to INV-6 at
kp=1, kn=0 -- not an arbitrary pick.

Everything else (nominal preview, max_joint_step_rad, kp, kn=0 fixed) is
copied from INV-6. The insertion channel gets an ANALOGOUS per-tick position
step clip, `max_L_step_m = dt * insertion_velocity_limit_m_s`, so its total
command (u_ref,L + delta_u_L) is capped at the same |u_L| <= 2mm/s MPC uses,
via the same clip-the-total-step mechanism INV-6 already uses for joints
(not a separate clip on the feedback trim alone -- see module docstring of
the caller for why that would be unfair).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .inverse_jacobian_2dof_trim import TwoDOFTrimStep

Array = np.ndarray

__all__ = [
    "SevenDOFDelayAwareConfig",
    "SevenDOFDelayAwareController",
    "build_inv_7dof_delay_aware_controller",
]


@dataclass
class SevenDOFDelayAwareConfig:
    kp: float = 1.0
    kn: float = 0.0                    # kept for interface parity; unused (locked 0, matches INV-6)
    damping: float = 5.0e-2            # SAME numeric value as INV-6's damping (unnormalized rad units)
    su_joint: float = 0.05             # rad/s, input_error_scale convention
    su_insertion: float = 5.0e-3       # m/s, input_error_scale convention
    dt: float = 0.1
    max_joint_step_rad: float = 0.010  # == dt * 0.10 rad/s, same as INV-6/MPC joint velocity_limit
    max_L_step_m: float = 2.0e-4       # == dt * 2.0e-3 m/s, same as MPC insertion velocity_limit
    delay_samples: int = 2
    enable_logging: bool = False


@dataclass
class SevenDOFDelayAwareStep(TwoDOFTrimStep):
    pass


class SevenDOFDelayAwareController:
    """Matched-authority 7DOF version of `TwoDOFDelayAwareController`. See
    module docstring for the normalized-DLS derivation and the damping
    rescale that guarantees exact reduction to INV-6 when insertion
    authority is disabled."""

    name = "inv_7dof_delay_aware"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        config: SevenDOFDelayAwareConfig | None = None,
        schedule: Array | None = None,
    ) -> None:
        self.reference = reference
        self.jacobian_provider = jacobian_provider
        self.schedule = None if schedule is None else np.asarray(schedule, dtype=float)
        self.config = config or SevenDOFDelayAwareConfig()
        self.q_nom: Array | None = None    # persistent 7D nominal accumulator (insertion channel only used from it)
        self.q_cmd7: Array | None = None   # persistent 7D commanded-position accumulator (joints + insertion)
        self.log: dict[str, list] = {
            k: []
            for k in (
                "t_idx", "r_idx", "du_task_norm", "du_task_vec", "d_k_norm",
                "e_pred_norm", "e_pred_vec", "z_nom_D", "q_cmd7_mirror",
                "rate_saturated_any", "u_ref_vec", "delta_u_vec",
            )
        }

    def reset(self) -> None:
        self.q_nom = None
        self.q_cmd7 = None

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "kp": self.config.kp,
            "kn": self.config.kn,
            "damping": self.config.damping,
            "su_joint": self.config.su_joint,
            "su_insertion": self.config.su_insertion,
            "delay_samples": self.config.delay_samples,
        }

    def _su(self) -> Array:
        cfg = self.config
        return np.array([cfg.su_joint] * 6 + [cfg.su_insertion], dtype=float)

    def _lambda_u(self) -> float:
        # see module docstring: rescale by su_joint so zeroing the insertion
        # column reproduces INV-6's Gram matrix exactly.
        return self.config.damping * self.config.su_joint

    def solve(
        self,
        *,
        measured_state: Any,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
        insertion_authority: bool = True,
    ) -> SevenDOFDelayAwareStep:
        """`insertion_authority=False` zeros the insertion column of the
        normalized Jacobian (equivalent to su_insertion=0) -- used ONLY by
        the exact-reduction unit test, not by any run script."""
        cfg = self.config
        N = self.reference.sample_count
        idx = int(np.clip(control_index, 0, N - 1))
        idx_r = int(np.clip(idx + 1 + cfg.delay_samples, 0, N - 1))

        state = np.asarray(measured_state, dtype=float).reshape(7)
        u_ref = np.asarray(self.reference.input[idx], dtype=float).reshape(7)

        if self.q_nom is None:
            self.q_nom = state.copy()
            self.q_cmd7 = state.copy()
        self.q_nom = self.q_nom + cfg.dt * u_ref  # insertion channel: undelayed, unchanged from INV-6

        z_ref = np.asarray(self.reference.state, dtype=float)
        desired_all = np.asarray(self.reference.desired_position_m, dtype=float)

        # --- current-tick model residual d_k -- IDENTICAL to INV-6 ---
        if self.schedule is not None:
            jac_k = self.schedule[idx]
        else:
            jac_k = np.asarray(self.jacobian_provider(state), dtype=float).reshape(3, 7)
        p_nominal_k = desired_all[idx]
        measured_tip = np.asarray(measured_beam_position, dtype=float).reshape(3)
        d_k = measured_tip - (p_nominal_k + jac_k @ (state - z_ref[idx]))

        # --- delay-compensated nominal preview -- IDENTICAL to INV-6 ---
        q_nom_D_next6 = z_ref[idx_r, :6].copy()
        z_hat_base = np.concatenate([q_nom_D_next6, [self.q_nom[6]]])
        if self.schedule is not None:
            jac_r = self.schedule[idx_r]
        else:
            jac_r = np.asarray(self.jacobian_provider(z_hat_base), dtype=float).reshape(3, 7)
        p_nominal_r = desired_all[idx_r]
        p_hat_base = p_nominal_r + jac_r @ (z_hat_base - z_ref[idx_r]) + d_k

        p_des_r = p_nominal_r
        e_pred = p_des_r - p_hat_base  # IDENTICAL e_pred to INV-6 -- only the allocation below differs

        # --- 7-channel normalized-DLS feedback allocation ---
        su = self._su()
        if not insertion_authority:
            su = su.copy()
            su[6] = 0.0
        J_u = jac_r * su[None, :]  # (3,7)
        lambda_u = self._lambda_u()
        gram_u = J_u @ J_u.T + (lambda_u ** 2) * np.eye(3)
        J_u_pinv = J_u.T @ np.linalg.solve(gram_u, np.eye(3))  # (7,3)
        du_tilde = J_u_pinv @ (cfg.kp * e_pred)
        du_task = su * du_tilde  # (7,) -- physical-unit position trim, joints[rad] + insertion[m]

        z_star = z_hat_base + du_task
        step_raw = z_star - self.q_cmd7
        max_step = np.array([cfg.max_joint_step_rad] * 6 + [cfg.max_L_step_m])
        step = np.clip(step_raw, -max_step, max_step)
        rate_saturated = bool(np.any(np.abs(step_raw) > max_step))
        self.q_cmd7 = self.q_cmd7 + step
        command = step / cfg.dt

        if cfg.enable_logging:
            self.log["t_idx"].append(idx)
            self.log["r_idx"].append(idx_r)
            self.log["du_task_norm"].append(float(np.linalg.norm(du_task)))
            self.log["du_task_vec"].append(du_task.copy())
            self.log["d_k_norm"].append(float(np.linalg.norm(d_k)))
            self.log["e_pred_norm"].append(float(np.linalg.norm(e_pred)))
            self.log["e_pred_vec"].append(e_pred.copy())
            self.log["z_nom_D"].append(z_hat_base.copy())
            self.log["q_cmd7_mirror"].append(self.q_cmd7.copy())
            self.log["rate_saturated_any"].append(rate_saturated)
            self.log["u_ref_vec"].append(u_ref.copy())
            self.log["delta_u_vec"].append(command - u_ref)

        return SevenDOFDelayAwareStep(
            command=command,
            planned_input=u_ref.copy(),
            info={
                "status": "inv_7dof_delay_aware", "success": True,
                "du_task_norm": float(np.linalg.norm(du_task)),
                "r_idx": idx_r,
            },
        )


def build_inv_7dof_delay_aware_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    kp: float = 1.0,
    kn: float = 0.0,
    damping: float = 5.0e-2,
    su_joint: float = 0.05,
    su_insertion: float = 5.0e-3,
    dt: float = 0.1,
    max_joint_step_rad: float = 0.010,
    max_L_step_m: float = 2.0e-4,
    delay_samples: int = 2,
    enable_logging: bool = False,
    schedule: Array | None = None,
) -> SevenDOFDelayAwareController:
    return SevenDOFDelayAwareController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        schedule=schedule,
        config=SevenDOFDelayAwareConfig(
            kp=kp, kn=kn, damping=damping, su_joint=su_joint, su_insertion=su_insertion,
            dt=dt, max_joint_step_rad=max_joint_step_rad, max_L_step_m=max_L_step_m,
            delay_samples=delay_samples, enable_logging=enable_logging,
        ),
    )
