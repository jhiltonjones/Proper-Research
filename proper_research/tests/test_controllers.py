"""Exercise all four controllers against the real MPC modules, no project models.

The plant is a mildly nonlinear tip map with a contact-like term.  The
contact-free Jacobian omits that term, which is the model mismatch every
controller is asked to reject — the same asymmetry the real experiment sets up
between the contact plant and the no-contact Jacobian model.

What this proves:

* each controller constructs against the real ``BeamOutputTrackingMPC`` and the
  real ``ConfigurationReference`` / ``ConfigurationMPCConfig``;
* the LTI rung really does hold one Jacobian and the SQP rung really does
  relinearise (counted, not assumed);
* every command respects the shared velocity and acceleration limits;
* all four reduce the tip error they start with.
"""

from __future__ import annotations

import math

import numpy as np

import simulate_time_parameterized_beam_output_mpc as beam_module
import simulate_time_parameterized_configuration_mpc as base
from inverse_jacobian_controller import build_inverse_jacobian_controller
from mpc_variants import build_all_mpc_variants, precompute_schedule

SAMPLES, DT, HORIZON = 90, 0.02, 8


def make_reference() -> base.ConfigurationReference:
    t = np.arange(SAMPLES, dtype=float) * DT
    state = np.zeros((SAMPLES, 7))
    state[:, 0] = 0.30 * np.sin(2 * np.pi * 0.30 * t)
    state[:, 1] = 0.20 * np.cos(2 * np.pi * 0.22 * t) - 0.20
    state[:, 6] = 0.004 + 0.0025 * t
    rate = np.vstack([np.zeros((1, 7)), np.diff(state, axis=0) / DT])
    return base.ConfigurationReference(
        time_s=t,
        path_coordinate_m=0.01 * t,
        state=state,
        input=rate,
        acceleration=np.vstack([np.zeros((1, 7)), np.diff(rate, axis=0) / DT]),
        desired_position_m=np.stack(
            [0.30 * state[:, 0], 0.20 * state[:, 1], 5.0 * state[:, 6]], axis=1
        ),
        desired_tangent=np.tile(np.array([1.0, 0.0, 0.0]), (SAMPLES, 1)),
        planned_beam_feasible=np.ones(SAMPLES, dtype=bool),
        planned_position_error_m=np.zeros(SAMPLES),
        planned_tangent_error_rad=np.zeros(SAMPLES),
        sample_period_s=DT,
        source="controller_test",
    )


class Plant:
    """Tip map with a contact-like stiffening term the Jacobian may omit."""

    def __init__(self, contact: bool = True) -> None:
        self.contact = contact

    def tip(self, state) -> np.ndarray:
        q = np.asarray(state, dtype=float).reshape(7)
        base_tip = np.array([0.30 * q[0], 0.20 * q[1], 5.0 * q[6]])
        stiffening = np.array([0.05 * q[0] ** 3, 0.03 * q[1] ** 3, 0.0])
        contact = (
            np.array([0.015 * np.tanh(4.0 * q[0]), 0.006 * np.tanh(4.0 * q[1]), 0.0])
            if self.contact
            else np.zeros(3)
        )
        return base_tip + stiffening + contact

    def jacobian(self, state, *, with_contact: bool) -> np.ndarray:
        q = np.asarray(state, dtype=float).reshape(7)
        J = np.zeros((3, 7))
        J[0, 0] = 0.30 + 0.15 * q[0] ** 2
        J[1, 1] = 0.20 + 0.09 * q[1] ** 2
        J[2, 6] = 5.0
        if with_contact and self.contact:
            J[0, 0] += 0.060 / np.cosh(4.0 * q[0]) ** 2
            J[1, 1] += 0.024 / np.cosh(4.0 * q[1]) ** 2
        return J


def run(controller, reference, plant, config, steps: int = SAMPLES - 1) -> dict:
    state = np.asarray(reference.state[0], dtype=float) + np.array(
        [0.05, -0.04, 0.0, 0.0, 0.0, 0.0, 5.0e-4]
    )
    previous = np.zeros(7)
    velocity_limit = np.asarray(config.velocity_limit, dtype=float)
    acceleration_limit = np.asarray(config.acceleration_limit, dtype=float)
    errors, solve_ms, worst_velocity, worst_acceleration = [], [], 0.0, 0.0
    failures = 0
    for index in range(steps):
        measured = plant.tip(state)
        errors.append(
            float(np.linalg.norm(measured - reference.desired_position_m[
                min(index, reference.sample_count - 1)
            ]))
        )
        step = controller.solve(
            measured_state=state,
            measured_beam_position=measured,
            control_index=index,
            previous_input=previous,
        )
        if not step.success:
            failures += 1
        command = np.asarray(step.command, dtype=float).reshape(7)
        worst_velocity = max(
            worst_velocity, float(np.max(np.abs(command) / velocity_limit))
        )
        worst_acceleration = max(
            worst_acceleration,
            float(np.max(np.abs(command - previous) / (DT * acceleration_limit))),
        )
        solve_ms.append(1e3 * float(step.solve_time_s))
        state = state + DT * command
        previous = command
    final = float(
        np.linalg.norm(plant.tip(state) - reference.desired_position_m[-1])
    )
    return {
        "initial_error_mm": 1e3 * errors[0],
        "rms_error_mm": 1e3 * float(np.sqrt(np.mean(np.square(errors)))),
        "final_error_mm": 1e3 * final,
        "worst_velocity_ratio": worst_velocity,
        "worst_acceleration_ratio": worst_acceleration,
        "mean_solve_ms": float(np.mean(solve_ms)),
        "max_solve_ms": float(np.max(solve_ms)),
        "failures": failures,
    }


def main() -> None:
    reference = make_reference()
    plant = Plant(contact=True)

    class Pack(dict):
        pass

    pack = Pack(
        p_min=np.array([-3.0] * 6 + [0.0]),
        p_max=np.array([3.0] * 6 + [0.1]),
        u_max=np.array([2.0] * 6 + [0.05]),
    )
    config = base.make_default_mpc_config(
        reference=reference,
        controller_pack=pack,
        prediction_horizon=HORIZON,
        joint_acceleration_limit_rad_s2=20.0,
        insertion_acceleration_limit_m_s2=1.0,
        solver_backend="scipy" if base.osqp is None else "auto",
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3,) * 3,
        position_tracking_weight=1.0,
        terminal_weight_multiplier=20.0,
    )

    # The declared model mismatch: the plant has contact, the Jacobian does not.
    def contact_free_jacobian(state):
        return plant.jacobian(state, with_contact=False)

    def contact_jacobian(state):
        return plant.jacobian(state, with_contact=True)

    gaps = [
        float(
            np.linalg.norm(contact_jacobian(s) - contact_free_jacobian(s))
            / max(np.linalg.norm(contact_jacobian(s)), 1e-30)
        )
        for s in np.asarray(reference.state, dtype=float)
    ]
    print(
        f"[mismatch] contact vs contact-free Jacobian: "
        f"relative max {max(gaps):.4f}, rms {np.sqrt(np.mean(np.square(gaps))):.4f}"
    )
    assert max(gaps) > 0.01, "the mismatch is too small for this test to mean anything"

    schedule = precompute_schedule(
        reference=reference, jacobian_provider=contact_free_jacobian
    )
    controllers = build_all_mpc_variants(
        reference=reference,
        config=config,
        beam_config=beam_config,
        jacobian_provider=contact_free_jacobian,
        schedule=schedule,
        sqp_inner_iterations=1,
    )
    controllers["naive_inverse_jacobian"] = build_inverse_jacobian_controller(
        reference=reference,
        jacobian_provider=contact_free_jacobian,
        mpc_config=config,
        position_gain=1.0,
        damping=2.0e-3,
        nullspace_gain=1.0,
    )

    # The LTI rung must really hold one Jacobian.
    lti = controllers["mpc_lti"]
    assert np.allclose(
        lti.reference_position_jacobians,
        lti.reference_position_jacobians[0][None, :, :],
    ), "the LTI rung does not hold a single Jacobian"
    assert not np.allclose(
        controllers["mpc_ltv_offline"].reference_position_jacobians,
        controllers["mpc_ltv_offline"].reference_position_jacobians[0][None, :, :],
    ), "the LTV schedule is constant; this trajectory cannot separate the rungs"

    order = (
        "naive_inverse_jacobian",
        "mpc_lti",
        "mpc_ltv_offline",
        "mpc_ltv_sqp_online",
    )
    print(
        f"\n{'controller':<24}{'rms mm':>9}{'final mm':>10}{'|v|/max':>9}"
        f"{'|a|/max':>9}{'solve ms':>10}{'max ms':>9}"
    )
    results = {}
    for name in order:
        stats = run(controllers[name], reference, plant, config)
        results[name] = stats
        print(
            f"{name:<24}{stats['rms_error_mm']:>9.4f}{stats['final_error_mm']:>10.4f}"
            f"{stats['worst_velocity_ratio']:>9.3f}{stats['worst_acceleration_ratio']:>9.3f}"
            f"{stats['mean_solve_ms']:>10.3f}{stats['max_solve_ms']:>9.2f}"
        )

    # SLSQP satisfies constraints only to its ftol, so the scipy fallback
    # leaves microscopic overshoots that OSQP with polish does not. This is
    # worth knowing beyond the test: the simulator's own applied-command check
    # uses input_constraint_tolerance=1e-8, so running the real loop on the
    # scipy backend can hard-error on "Applied command violates a hard
    # velocity/acceleration bound" for a solution that is numerically fine.
    osqp_backend = base.osqp is not None
    limit_slack = 1e-9 if osqp_backend else 5e-5
    if not osqp_backend:
        print(
            "\n[backend] OSQP is unavailable, so these solves used SciPy SLSQP. "
            f"Limit checks allow {limit_slack:g} of slack for its ftol; the real "
            "simulator's applied-command check is 1e-8 and would reject that.\n"
            "          Install OSQP before running the comparison for real."
        )

    failures, warnings = [], []
    for name, stats in results.items():
        if stats["failures"]:
            target = warnings if not osqp_backend else failures
            target.append(
                f"{name}: {stats['failures']} solve failure(s)"
                + (" (SLSQP line search; expected on the scipy fallback)"
                   if not osqp_backend else "")
            )
        if stats["worst_velocity_ratio"] > 1.0 + limit_slack:
            failures.append(
                f"{name}: exceeded the velocity limit by "
                f"{stats['worst_velocity_ratio'] - 1.0:.2e}"
            )
        if stats["worst_acceleration_ratio"] > 1.0 + limit_slack:
            failures.append(
                f"{name}: exceeded the acceleration limit by "
                f"{stats['worst_acceleration_ratio'] - 1.0:.2e}"
            )
        if stats["final_error_mm"] > stats["initial_error_mm"]:
            failures.append(f"{name}: did not reduce the tip error")
    if warnings:
        print("[warnings]\n  " + "\n  ".join(warnings))

    sqp = controllers["mpc_ltv_sqp_online"]
    print(
        f"\n[sqp] relinearised {sqp.relinearisation_count} times "
        f"({1e3 * sqp.jacobian_time_s:.2f} ms of Jacobian evaluation)"
    )
    if sqp.relinearisation_count < SAMPLES - 1:
        failures.append("the SQP rung did not relinearise once per step")

    if failures:
        raise SystemExit("FAILURES:\n  " + "\n  ".join(failures))
    print(
        "\n[CONTROLLER TEST] PASS - four controllers built against the real MPC "
        "modules, all limits respected, all reduce tip error"
    )


if __name__ == "__main__":
    main()
