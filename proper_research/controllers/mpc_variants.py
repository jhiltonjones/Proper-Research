"""The three MPC rungs: LTI, LTV offline Jacobian, LTV SQP online Jacobian.

All three are ``BeamOutputTrackingMPC`` — the same condensed QP, the same
constraints, the same residual estimator.  They differ in exactly one thing:
**where the beam Jacobian comes from.**  That is what makes the ladder
informative; anything else that varied would confound it.

    mpc_lti              one Jacobian, frozen at a single reference sample,
                         held for the entire run
    mpc_ltv_offline      one Jacobian per reference sample, all precomputed
                         before the loop, never revisited
    mpc_ltv_sqp_online   relinearised at the *measured* state every control
                         step, optionally iterated within the step

Reading the rungs
-----------------
**LTI -> LTV offline** prices scheduling the model along the path.  A large gap
means the Jacobian varies materially over the trajectory, so a single
linearisation cannot cover it.

**LTV offline -> SQP online** prices relinearising on the measured state.  The
offline schedule is evaluated at ``z_ref``; if the plant has drifted away from
the reference — which is exactly what a contact-free Jacobian against a contact
plant provokes — then ``J(z_ref)`` is the derivative at the wrong point.  A
large gap here says the plant leaves the neighbourhood where the offline
linearisation is valid.

You can predict this gap before running anything.  The layer-4 analysis
computes the one-step prediction error of the affine model against what the
plant actually did; when that error is comparable to the per-step tip motion,
the offline Jacobian is not a good local model and the online rung should win.

Cost
----
Online relinearisation rebuilds ``Jbar``, the Hessian and the OSQP ``P`` values
every step.  Report worst-case solve time next to accuracy: an accuracy win you
cannot compute inside one sample period is not a win.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from proper_research.controllers.beam_jacobian_providers import declare_provider, provenance_line
except ImportError:  # standalone
    from proper_research.controllers.beam_jacobian_providers import declare_provider, provenance_line  # type: ignore

Array = np.ndarray


def _tag(controller: Any, name: str, description: str, provenance: dict) -> Any:
    """Stamp a rung with what it is and which Jacobian it was built from."""
    controller.variant_name = name
    controller.variant_description = description
    controller.jacobian_provenance = dict(provenance)
    controller.jacobian_is_contact_free = (
        provenance.get("contact_used_in_jacobian") is False
    )
    controller.describe = lambda: {  # type: ignore[method-assign]
        "controller": name,
        "description": description,
        "jacobian": provenance_line(provenance),
        "jacobian_provenance": dict(provenance),
    }
    return controller


def _beam_output_module() -> Any:
    try:
        from proper_research.simulation.simulations import (
            simulate_time_parameterized_beam_output_mpc as module,
        )
    except ModuleNotFoundError:  # standalone review
        import simulate_time_parameterized_beam_output_mpc as module  # type: ignore
    return module


# --------------------------------------------------------------------------
# rung 3: LTV with the offline schedule (your current controller, unchanged)
# --------------------------------------------------------------------------
def build_ltv_offline_mpc(
    *,
    reference: Any,
    config: Any,
    beam_config: Any,
    reference_position_jacobians: Array,
    nominal_reference_positions_m: Any | None = None,
    jacobian_provenance: dict | None = None,
) -> Any:
    """``BeamOutputTrackingMPC`` with the precomputed per-sample schedule.

    This is the existing controller with nothing added.  It is here so all four
    rungs are constructed through one module and no rung can accidentally be
    given different weights or a different reference.

    ``nominal_reference_positions_m`` is worth passing deliberately.  It
    defaults to ``reference.desired_position_m`` — the centreline — but the
    planned configuration does not put the tip exactly on the centreline, so
    the planner's own tracking error is absorbed into ``d_hat`` along with
    genuine plant mismatch and the two stop being separable.  Passing the
    planner's *achieved* tip positions makes ``d_hat`` mean what its name says.
    """
    module = _beam_output_module()
    controller = module.BeamOutputTrackingMPC(
        reference=reference,
        config=config,
        beam_config=beam_config,
        reference_position_jacobians=reference_position_jacobians,
        nominal_reference_positions_m=nominal_reference_positions_m,
    )
    return _tag(
        controller,
        "mpc_ltv_offline",
        "condensed QP, one analytical Jacobian per reference sample, "
        "precomputed offline and never revisited",
        jacobian_provenance or {},
    )


# --------------------------------------------------------------------------
# rung 2: LTI
# --------------------------------------------------------------------------
def build_lti_mpc(
    *,
    reference: Any,
    config: Any,
    beam_config: Any,
    reference_position_jacobians: Array,
    freeze_index: int = 0,
    nominal_reference_positions_m: Any | None = None,
    jacobian_provenance: dict | None = None,
) -> Any:
    """The same QP with one Jacobian held for the whole run.

    Implemented by replacing the schedule with a repeated single matrix rather
    than by changing the controller: the QP structure, the prediction and the
    residual estimator are then provably identical to the LTV rung, and only
    the model content differs.

    ``freeze_index`` chooses which reference sample to linearise at.  0 — the
    start of the trajectory — is the honest default: it is the only point a
    genuinely time-invariant design could know in advance.  Freezing at the
    midpoint flatters the LTI rung and should be labelled if used.
    """
    module = _beam_output_module()
    schedule = np.asarray(reference_position_jacobians, dtype=float)
    if schedule.ndim != 3 or schedule.shape[1:] != (3, 7):
        raise ValueError(
            f"reference_position_jacobians must have shape (samples, 3, 7); "
            f"received {schedule.shape}."
        )
    index = int(np.clip(freeze_index, 0, schedule.shape[0] - 1))
    frozen = np.repeat(schedule[index][None, :, :], schedule.shape[0], axis=0)
    controller = module.BeamOutputTrackingMPC(
        reference=reference,
        config=config,
        beam_config=beam_config,
        reference_position_jacobians=frozen,
        nominal_reference_positions_m=nominal_reference_positions_m,
    )
    controller.freeze_index = index
    return _tag(
        controller,
        "mpc_lti",
        f"condensed QP, one Jacobian frozen at reference sample {index}",
        jacobian_provenance or {},
    )


# --------------------------------------------------------------------------
# rung 4: LTV with online relinearisation (SQP)
# --------------------------------------------------------------------------
def build_sqp_online_mpc(
    *,
    reference: Any,
    config: Any,
    beam_config: Any,
    reference_position_jacobians: Array,
    jacobian_provider: Callable[[Array], Array],
    inner_iterations: int = 1,
    relinearise_horizon: bool = False,
    nominal_reference_positions_m: Any | None = None,
    allow_undeclared_jacobian: bool = True,
) -> Any:
    """The same QP, relinearised at the measured state each control step.

    ``inner_iterations = 1`` is one relinearisation per step — the usual
    real-time SQP: solve once at the current point and apply.  Above 1 the
    solve is repeated, re-evaluating the Jacobian at the first predicted state,
    which is a short SQP rather than a full one.  Each extra iteration costs a
    Jacobian evaluation and a full QP solve, so it shows up directly in the
    worst-case solve time.

    ``relinearise_horizon=False`` writes the measured-state Jacobian into every
    horizon slot: one model for the whole prediction, evaluated where the plant
    actually is.  ``True`` keeps the offline schedule's *shape* and shifts it by
    the difference between the measured-state Jacobian and the schedule's value
    at the current index — a cheap way to keep some of the path variation while
    still correcting for where the plant really is.  Neither costs an extra
    beam solve beyond the one Jacobian call.
    """
    module = _beam_output_module()
    jacobian_at, provenance = declare_provider(
        jacobian_provider, allow_undeclared=allow_undeclared_jacobian
    )
    schedule = np.asarray(reference_position_jacobians, dtype=float).copy()
    if schedule.ndim != 3 or schedule.shape[1:] != (3, 7):
        raise ValueError(
            f"reference_position_jacobians must have shape (samples, 3, 7); "
            f"received {schedule.shape}."
        )
    iterations = max(1, int(inner_iterations))

    class OnlineSQPBeamOutputMPC(module.BeamOutputTrackingMPC):  # type: ignore[misc]
        """``BeamOutputTrackingMPC`` whose Jacobian follows the measurement."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._offline_schedule = schedule.copy()
            self.relinearisation_count = 0
            self.jacobian_time_s = 0.0

        def relinearise(self, state: Array, control_index: int) -> Array:
            import time as _time

            started = _time.perf_counter()
            jacobian = np.asarray(
                jacobian_at(np.asarray(state, dtype=float).reshape(7)),
                dtype=float,
            ).reshape(3, 7)
            self.jacobian_time_s += _time.perf_counter() - started
            self.relinearisation_count += 1
            if relinearise_horizon:
                index = int(
                    np.clip(control_index, 0, self._offline_schedule.shape[0] - 1)
                )
                shift = jacobian - self._offline_schedule[index]
                self.reference_position_jacobians[:] = (
                    self._offline_schedule + shift[None, :, :]
                )
            else:
                self.reference_position_jacobians[:] = jacobian[None, :, :]
            return jacobian

        def solve(
            self,
            *,
            measured_state: Any,
            measured_beam_position: Any,
            control_index: int,
            previous_input: Any,
        ) -> Any:
            linearisation_state = np.asarray(measured_state, dtype=float).reshape(7)
            step = None
            for iteration in range(iterations):
                self.relinearise(linearisation_state, control_index)
                step = super().solve(
                    measured_state=measured_state,
                    measured_beam_position=measured_beam_position,
                    control_index=control_index,
                    previous_input=previous_input,
                )
                if not step.success or iteration + 1 >= iterations:
                    break
                # Re-linearise where the current solution says the plant will
                # be one step from now, not where it is.
                linearisation_state = np.asarray(
                    step.predicted_states, dtype=float
                )[0]
            assert step is not None
            return step

    controller = OnlineSQPBeamOutputMPC(
        reference=reference,
        config=config,
        beam_config=beam_config,
        reference_position_jacobians=schedule,
        nominal_reference_positions_m=nominal_reference_positions_m,
    )
    return _tag(
        controller,
        "mpc_ltv_sqp_online",
        f"condensed QP relinearised at the measured state, {iterations} SQP "
        f"iteration(s) per step, "
        + (
            "horizon shape retained"
            if relinearise_horizon
            else "single model over the horizon"
        ),
        provenance,
    )


# --------------------------------------------------------------------------
# shared construction
# --------------------------------------------------------------------------
def precompute_schedule(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    allow_undeclared_jacobian: bool = True,
) -> Array:
    """Evaluate the Jacobian once per reference sample.

    Deliberately routed through the same provider the online rung uses, so the
    offline schedule and the online relinearisation differ only in *where* they
    are evaluated — never in which model they come from.
    """
    jacobian_at, _ = declare_provider(
        jacobian_provider, allow_undeclared=allow_undeclared_jacobian
    )
    count = int(reference.sample_count)
    schedule = np.empty((count, 3, 7), dtype=float)
    states = np.asarray(reference.state, dtype=float)
    for index in range(count):
        schedule[index] = np.asarray(
            jacobian_at(states[index]), dtype=float
        ).reshape(3, 7)
    if not np.all(np.isfinite(schedule)):
        raise FloatingPointError("The precomputed Jacobian schedule is not finite.")
    return schedule


def build_all_mpc_variants(
    *,
    reference: Any,
    config: Any,
    beam_config: Any,
    jacobian_provider: Callable[[Array], Array],
    schedule: Array | None = None,
    freeze_index: int = 0,
    sqp_inner_iterations: int = 1,
    relinearise_horizon: bool = False,
    nominal_reference_positions_m: Any | None = None,
    allow_undeclared_jacobian: bool = True,
) -> dict[str, Any]:
    """All three MPC rungs from one schedule and one provider.

    Every rung is stamped with the provider's provenance, so each controller
    can be asked directly whether its Jacobian is contact-free.
    """
    _, provenance = declare_provider(
        jacobian_provider, allow_undeclared=allow_undeclared_jacobian
    )
    if schedule is None:
        schedule = precompute_schedule(
            reference=reference,
            jacobian_provider=jacobian_provider,
            allow_undeclared_jacobian=allow_undeclared_jacobian,
        )
    shared = dict(
        reference=reference,
        config=config,
        beam_config=beam_config,
        nominal_reference_positions_m=nominal_reference_positions_m,
    )
    return {
        "mpc_lti": build_lti_mpc(
            reference_position_jacobians=schedule,
            freeze_index=freeze_index,
            jacobian_provenance=provenance,
            **shared,
        ),
        "mpc_ltv_offline": build_ltv_offline_mpc(
            reference_position_jacobians=schedule,
            jacobian_provenance=provenance,
            **shared,
        ),
        "mpc_ltv_sqp_online": build_sqp_online_mpc(
            reference_position_jacobians=schedule,
            jacobian_provider=jacobian_provider,
            inner_iterations=sqp_inner_iterations,
            relinearise_horizon=relinearise_horizon,
            allow_undeclared_jacobian=allow_undeclared_jacobian,
            **shared,
        ),
    }


__all__ = [
    "build_all_mpc_variants",
    "build_lti_mpc",
    "build_ltv_offline_mpc",
    "build_sqp_online_mpc",
    "precompute_schedule",
]
