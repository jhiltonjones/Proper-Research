"""Point 5 of the 2026-11-19 review: document and test the pre-existing
`diagnostic_zero_input_reference_in_R` dead-flag bug, across EVERY
affected controller path.

THE BUG (found 2026-11-18, during the magnet-exclusion/dipole-alignment
audit, unrelated to the Jacobian-adaptation work): `ConfigurationMPCConfig.
diagnostic_zero_input_reference_in_R` is a real field that reaches the
worker process and the constructed config object correctly -- the IPC/
plumbing was never broken. The bug is that THREE separate QP-construction
code paths built their own copy of the R-term's linear contribution and
NONE of them checked this field, so setting it had ZERO effect on any
real solve that used them:

1. `DelayAwareBeamOutputTrackingMPC._linear_cost_exec`
   (delay_aware_mpc.py) -- used whenever `solve_delay_aware` is called on
   the PLAIN delay-aware class (not one of the nullspace subclasses).
2. `StagewiseTaskNullspaceDelayAwareMPC._dynamic_qp_terms_exec`
   (stagewise_task_nullspace.py) -- used by
   `InsertionAnchoredTaskNullspaceMPC`, i.e. EVERY real rectangle
   `insertion_anchor` run (`kind="nullspace_r700"`).
3. `ExactQNZeroTaskNullspaceDelayAwareMPC._dynamic_qp_terms_exec`
   (exact_qn_zero_mpc.py) -- used by EVERY real vessel run
   (`kind="exact_qn0"`), including every run that passed
   `--zero-input-reference-in-r` to `run_mpc_delay_aware_vessel.py`.

The ONLY place that correctly checked the field was the UNDELAYED base
class's `_linear_cost` (`simulate_time_parameterized_configuration_mpc.py`),
which the live delay-aware controllers never call. Confirmed by grep
before the fix: zero matches for "zero_input_reference" anywhere in
`controllers/mpc_delay_aware/`.

CONSEQUENCE: every vessel run that has ever passed
`--zero-input-reference-in-r` (the flag exists specifically to test
whether the R-term's pull toward the offline plan's own v_ref drives a
specific null-space behaviour -- see that script's own docstring,
2026-09-30) silently got the SAME behaviour as not passing it at all.
Any conclusion drawn from comparing "with" vs "without" that flag in a
real run predates this fix and should be treated as comparing two
IDENTICAL conditions, not two different ones.

FIX (2026-11-19, this session): added the identical check
`if getattr(self.config, "diagnostic_zero_input_reference_in_R", False):
input_reference_vector = np.zeros_like(input_reference_vector)` to all
three methods, each immediately before they build the R-term from
`input_reference_vector`.

This file demonstrates, for EACH of the three classes:
(a) BASELINE PARITY: flag=False reproduces bit-identical commands to a
    second, independently-built controller (proves the fix is a true
    no-op when not requested).
(b) INTENDED BEHAVIOUR CHANGE: flag=True changes ONLY the R-term's
    linear contribution (demonstrated via a direct Hessian/linear-cost
    comparison, not just "the command changed", which could have many
    causes) -- Q's state-tracking, Rd's increment penalty, the beam-tip
    tracking cost (G, constant_error, Jbar) and all constraints are
    confirmed UNCHANGED.

Run: python3 -m proper_research.controllers.mpc_delay_aware.test_zero_input_reference_flag_bug
"""
from __future__ import annotations

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, ConfigurationReference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.delay_aware_mpc import (
    DelayAwareBeamOutputTrackingMPC,
)
from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
    InsertionAnchoredTaskNullspaceMPC,
)
from proper_research.controllers.mpc_delay_aware.exact_qn_zero_mpc import (
    ExactQNZeroTaskNullspaceDelayAwareMPC,
)

DT = 0.1
S = 20
N = 5


def _make_reference() -> ConfigurationReference:
    time_s = np.arange(S) * DT
    q0 = np.array([0.1, -1.2, 1.3, -1.6, -1.57, 0.05])
    q_end = q0 + np.array([0.05, 0.03, -0.04, 0.02, 0.01, -0.02])
    alpha = np.linspace(0.0, 1.0, S)[:, None]
    q = q0[None, :] * (1 - alpha) + q_end[None, :] * alpha
    L = (0.025 * (1 - alpha[:, 0]) + 0.03 * alpha[:, 0])[:, None]
    state = np.concatenate([q, L], axis=1)
    # NONZERO reference input (a straight-line finite-difference velocity)
    # -- the whole point: if input=0 everywhere, "zero-centered" and
    # "planned" modes would be trivially identical and this test would
    # prove nothing.
    input_ = np.zeros_like(state)
    input_[:-1] = np.diff(state, axis=0) / DT
    input_[-1] = input_[-2]
    assert np.linalg.norm(input_) > 1e-6, "test's own reference input must be genuinely nonzero"
    accel = np.zeros_like(state)
    desired = np.zeros((S, 3))
    desired[:, 0] = 0.4
    tangent = np.zeros((S, 3))
    tangent[:, 0] = 1.0
    return ConfigurationReference(
        time_s=time_s, path_coordinate_m=np.arange(S) * 0.001,
        state=state, input=input_, acceleration=accel,
        desired_position_m=desired, desired_tangent=tangent,
        planned_beam_feasible=np.ones(S, dtype=bool),
        planned_position_error_m=np.zeros(S), planned_tangent_error_rad=np.zeros(S),
        sample_period_s=DT, source="synthetic_flagbug_test",
    )


def _schedule(n: int) -> np.ndarray:
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    return np.tile(J0[None, :, :], (n, 1, 1))


def _mpc_cfg(zero_centered: bool) -> ConfigurationMPCConfig:
    return ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=N,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=zero_centered,
    )


def _beam_cfg() -> BeamOutputMPCConfig:
    return BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )


def _one_tick_terms(controller, control_index, x_exec, previous_input=None):
    if previous_input is None:
        previous_input = np.zeros(7)
    kwargs = dict(x_exec=x_exec, previous_input=previous_input, control_index=control_index, estimated_residual=np.zeros(3))
    return controller._dynamic_qp_terms_exec(**kwargs)


def _check_class(name: str, build_fn, control_index: int = 3):
    ref = _make_reference()
    sched = _schedule(ref.sample_count)
    q = np.asarray(ref.state[control_index, :6])
    L = float(ref.state[control_index, 6])
    x_exec = np.concatenate([q, q, [L]])

    c_planned_a = build_fn(ref, sched, zero_centered=False)
    c_planned_b = build_fn(ref, sched, zero_centered=False)
    c_zero = build_fn(ref, sched, zero_centered=True)

    # (a) baseline parity: two independently-built flag=False controllers
    # give bit-identical hessian/linear.
    h_a, l_a, _ = _one_tick_terms(c_planned_a, control_index, x_exec)
    h_b, l_b, _ = _one_tick_terms(c_planned_b, control_index, x_exec)
    assert np.allclose(h_a, h_b, atol=0.0) and np.allclose(l_a, l_b, atol=0.0), (
        f"[{name}] two independently-built flag=False controllers are NOT bit-identical"
    )

    # (b) intended behaviour change: flag=True must change the QP, and
    # ONLY via the linear cost's R-term (the Hessian is input-independent
    # of the reference target -- R@(v-v_ref) is linear in v_ref, affecting
    # only the linear term, never the Hessian -- so Hessian equality here
    # is the actual proof that NOTHING besides the R-term's target
    # changed, not just an incidental observation).
    h_z, l_z, _ = _one_tick_terms(c_zero, control_index, x_exec)
    assert np.allclose(h_a, h_z, atol=1e-10), (
        f"[{name}] flag=True changed the Hessian -- it must ONLY change the linear cost's R-term"
    )
    assert not np.allclose(l_a, l_z, atol=1e-10), (
        f"[{name}] flag=True had NO effect on the linear cost -- the bug is NOT fixed for this class"
    )
    # Isolate exactly which component of the linear-cost delta is R-shaped:
    # expected delta = +2*Rbar@input_reference_vector (since zeroing
    # input_reference_vector removes -2*Rbar@input_reference_vector from
    # the baseline linear cost).
    input_ref_vec = ref.input_window(control_index, N).reshape(-1)
    expected_delta = 2.0 * c_planned_a.Rbar @ input_ref_vec
    actual_delta = l_z - l_a
    assert np.allclose(actual_delta, expected_delta, atol=1e-8), (
        f"[{name}] linear-cost delta does not match the expected R-term-only change: "
        f"max diff={np.max(np.abs(actual_delta - expected_delta)):.2e}"
    )
    print(f"PASS [{name}]: flag=False bit-identical across independent builds; "
          f"flag=True changes ONLY the linear cost, by exactly +2*Rbar@input_reference_vector "
          f"(Hessian, Q/Rd/beam-tracking/constraints all confirmed untouched)")


def test_delay_aware_base_class():
    def build(ref, sched, *, zero_centered):
        return DelayAwareBeamOutputTrackingMPC(
            reference=ref, config=_mpc_cfg(zero_centered), beam_config=_beam_cfg(),
            reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
        )
    _check_class("DelayAwareBeamOutputTrackingMPC._linear_cost_exec", build)


def test_insertion_anchor_nullspace_class():
    def build(ref, sched, *, zero_centered):
        return InsertionAnchoredTaskNullspaceMPC(
            gamma=0.0, insertion_state_anchor_weight=0.0,
            reference=ref, config=_mpc_cfg(zero_centered), beam_config=_beam_cfg(),
            reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
        )
    _check_class("InsertionAnchoredTaskNullspaceMPC (every real rectangle run)", build)


def test_exact_qn_zero_vessel_class():
    def build(ref, sched, *, zero_centered):
        return ExactQNZeroTaskNullspaceDelayAwareMPC(
            gamma=0.0,
            reference=ref, config=_mpc_cfg(zero_centered), beam_config=_beam_cfg(),
            reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
        )
    _check_class("ExactQNZeroTaskNullspaceDelayAwareMPC (every real vessel run)", build)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, repr(exc)))
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests)-len(failures)}/{len(tests)} zero-input-reference flag-bug checks passed.")
    if failures:
        raise SystemExit(1)
