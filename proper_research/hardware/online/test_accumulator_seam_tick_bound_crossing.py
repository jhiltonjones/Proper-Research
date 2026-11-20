"""Hardware-mocked test of the REAL production insertion-dispatch branch
(close_loop_path_follow.accumulator_seam_tick), extracted from main() so
this test calls the SAME code main() calls -- not a reimplementation of
the clip/accumulator arithmetic.

Covers BOTH MPC and INV: both of this project's rectangle_stage_a runners
(run_mpc_delay_aware_insertion_anchor.py and run_inv_7dof_delay_aware.py)
set cfg.accumulator_seam=True and cfg.feedforward_joint_trajectory=False
identically (confirmed by direct grep in test_shared_config_matches_both_
runners below) -- so both run through this one function.

2026-11-20 (actuator-dispatch fix): the envelope is enforced BEFORE
AdvancerSink.submit_rate -- the clipped target is computed first, and the
EFFECTIVE rate implied by it is what's dispatched, never the raw
corr[6]/command[6].

2026-11-20 (final-rate consistency fix): AdvancerSink.submit_rate has its
OWN independent |rate|<=max_rate_m_s clamp and now returns the rate it
actually accepted. accumulator_seam_tick uses THAT accepted rate (not the
pre-advancer effective rate) for insertion_m/ins_cmd bookkeeping, and
returns both the requested and final rates for the caller to log
distinctly. _MockAdvancer below replicates the real class's clamp+return
contract exactly, so a request that exceeds the advancer's own rate cap
is exercised here too, independent of any position-bound crossing.

No hardware: this test imports and calls accumulator_seam_tick directly
(a plain function, no camera/robot/servo construction anywhere here).
"""
import numpy as np
import pytest

from proper_research.hardware.online.close_loop_path_follow import (
    PathFollowConfig,
    accumulator_seam_tick,
)

ENVELOPE_MIN_M = 0.020
ENVELOPE_MAX_M = 0.070


class _MockAdvancer:
    """Replicates the real AdvancerSink.submit_rate contract exactly:
    clamps |rate|<=max_rate_m_s and returns the accepted rate. Records
    every (requested, accepted) pair."""

    def __init__(self, max_rate_m_s=0.01):
        self.max_rate_m_s = max_rate_m_s
        self.calls = []  # (requested_rate, dt)
        self.accepted = []  # accepted rate per call

    def submit_rate(self, insertion_rate_m_s: float, dt: float) -> float:
        self.calls.append((insertion_rate_m_s, dt))
        rate = float(insertion_rate_m_s)
        accepted = max(-self.max_rate_m_s, min(self.max_rate_m_s, rate))
        self.accepted.append(accepted)
        return accepted


def _build_row_insertion_length_field(insertion_m: float) -> float:
    """Verbatim reproduction of close_loop_path_follow.py's own
    path_follow.jsonl row-dict expression (line ~2412):
        "insertion_length_m": round(float(insertion_m), 5)
    """
    return round(float(insertion_m), 5)


def _base_cfg(**overrides):
    kwargs = dict(
        accumulator_seam=True, feedforward_joint_trajectory=False,
        insertion_command_min_m=ENVELOPE_MIN_M, insertion_command_max_m=ENVELOPE_MAX_M,
        joint_velocity_limit_rad_s=0.5, max_joint_step_rad=0.05,
    )
    kwargs.update(overrides)
    return PathFollowConfig(**kwargs)


def test_shared_config_matches_both_runners():
    import proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor as mpc_runner
    import proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware as inv_runner
    import inspect

    mpc_src = inspect.getsource(mpc_runner)
    inv_src = inspect.getsource(inv_runner)
    for src, name in ((mpc_src, "MPC"), (inv_src, "INV")):
        assert "cfg.accumulator_seam = True" in src, f"{name} runner must set accumulator_seam=True"
        assert "cfg.feedforward_joint_trajectory = False" in src, f"{name} runner must set feedforward_joint_trajectory=False"


@pytest.mark.parametrize("controller_label", ["MPC", "INV"])
def test_bound_crossing_through_real_dispatch_function(controller_label):
    """Both MPC and INV dispatch through the identical accumulator_seam_
    tick() function -- this parametrization documents that one test
    covers both. Uses the approved 20-70mm envelope."""
    cfg = _base_cfg()
    advancer = _MockAdvancer(max_rate_m_s=10.0)  # high cap -- isolates the position-bound effect
    dt = 0.1
    servo_err_log = []
    q0 = np.zeros(6)
    ins_cmd, insertion_m, q_cmd = None, 0.040, None

    # Tick 1: a large negative command[6] forces a bound crossing below 20mm.
    command1 = np.array([0.01, -0.01, 0.0, 0.0, 0.0, 0.0, -5.0])
    (ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target,
     requested_rate, final_rate, _) = accumulator_seam_tick(
        cfg, ins_cmd=ins_cmd, insertion_m=insertion_m, q_cmd=q_cmd, q=q0,
        corr=command1, command=command1, dt=dt, servo_err_log=servo_err_log,
        advancer=advancer,
    )

    # 1. The clip is real: pinned at the 20mm floor, not the unclipped -0.46m target.
    assert insertion_m == pytest.approx(0.020)
    assert ins_cmd == insertion_m
    assert requested_rate == pytest.approx(-5.0)
    assert final_rate == pytest.approx((0.020 - 0.040) / dt)
    assert advancer.calls[-1][0] == pytest.approx((0.020 - 0.040) / dt), (
        "the advancer receives the envelope-aware EFFECTIVE rate (derived from the clipped "
        "target), not the raw -5.0 request"
    )

    # 2. The saved row would record the CLIPPED command.
    logged_insertion_length_m = _build_row_insertion_length_field(insertion_m)
    assert logged_insertion_length_m == pytest.approx(0.020)

    # 3. Next controller call: feed tick 1's returned state into tick 2.
    q1 = q_target.copy()
    command2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    (ins_cmd2, insertion_m2, q_cmd2, q_cmd_prev2, q_target2,
     requested_rate2, final_rate2, _) = accumulator_seam_tick(
        cfg, ins_cmd=ins_cmd, insertion_m=insertion_m, q_cmd=q_cmd, q=q1,
        corr=command2, command=command2, dt=dt, servo_err_log=servo_err_log,
        advancer=advancer,
    )
    assert ins_cmd2 == pytest.approx(0.020 + 0.1 * dt), (
        "tick 2 must continue from tick 1's CLIPPED accumulator value, not a stale unclipped one"
    )
    assert insertion_m2 == ins_cmd2
    assert q_cmd_prev2 is not None and np.allclose(q_cmd_prev2, q1)
    assert len(servo_err_log) == 2
    assert final_rate2 == pytest.approx(0.1)


def test_retraction_recovery_at_lower_bound():
    """Retraction/recovery at the 20mm bound: while pinned at the floor
    by an over-retracting command, a subsequent extending command must
    be dispatched and tracked normally -- never stuck at a boundary."""
    cfg = _base_cfg()
    advancer = _MockAdvancer(max_rate_m_s=10.0)
    dt = 0.1
    servo_err_log = []
    q0 = np.zeros(6)

    command1 = np.array([0, 0, 0, 0, 0, 0, -5.0])
    ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target, _, _, _ = accumulator_seam_tick(
        cfg, ins_cmd=None, insertion_m=0.025, q_cmd=None, q=q0,
        corr=command1, command=command1, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert insertion_m == pytest.approx(0.020)
    assert advancer.calls[-1][0] < 0.0

    command2 = np.array([0, 0, 0, 0, 0, 0, -5.0])
    ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target, _, _, _ = accumulator_seam_tick(
        cfg, ins_cmd=ins_cmd, insertion_m=insertion_m, q_cmd=q_cmd, q=q_target,
        corr=command2, command=command2, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert insertion_m == pytest.approx(0.020)
    assert advancer.calls[-1][0] == pytest.approx(0.0), "at the floor, a further-retracting command must be zeroed, not negative"

    command3 = np.array([0, 0, 0, 0, 0, 0, 0.5])
    ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target, _, _, _ = accumulator_seam_tick(
        cfg, ins_cmd=ins_cmd, insertion_m=insertion_m, q_cmd=q_cmd, q=q_target,
        corr=command3, command=command3, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert insertion_m == pytest.approx(0.020 + 0.5 * dt)
    assert advancer.calls[-1][0] == pytest.approx(0.5), "recovery from the floor must dispatch the full extending rate, unclamped"


def test_request_exceeds_advancer_rate_cap_independent_of_position_bounds():
    """Item 2: a request well inside the 20-70mm position envelope but
    whose RATE exceeds the advancer's own max_rate_m_s. The accepted
    rate (post rate-cap) must be what's used for bookkeeping and
    logging, not the requested one."""
    cfg = _base_cfg()
    advancer = _MockAdvancer(max_rate_m_s=0.01)  # tight, realistic cap
    dt = 0.1
    servo_err_log = []
    q0 = np.zeros(6)

    # Mid-envelope start, a huge requested rate (10 m/s) -- the position
    # target (0.045 + 1.0m) is nowhere near a bound, so only the
    # advancer's own rate cap should limit this, not insertion_command_
    # min_m/max_m.
    command1 = np.array([0, 0, 0, 0, 0, 0, 10.0])
    (ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target,
     requested_rate, final_rate, _) = accumulator_seam_tick(
        cfg, ins_cmd=None, insertion_m=0.045, q_cmd=None, q=q0,
        corr=command1, command=command1, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert requested_rate == pytest.approx(10.0)
    assert final_rate == pytest.approx(0.01), "the FINAL rate must be the advancer's own cap, not the 10 m/s request"
    # The advancer is called with the ENVELOPE-clipped effective rate
    # (position target 0.045+1.0m clipped to the 70mm ceiling -> rate
    # (0.070-0.045)/0.1 = 0.25 m/s) -- NOT the raw 10 m/s request, and
    # NOT yet the advancer's own further rate-cap (that clamp happens
    # inside submit_rate itself, confirmed by the returned final_rate).
    assert advancer.calls[-1][0] == pytest.approx((0.070 - 0.045) / dt)
    assert advancer.accepted[-1] == pytest.approx(0.01)
    # bookkeeping consistency: insertion_m must reflect the ACCEPTED rate,
    # not the requested one -- 0.045 + 0.01*0.1 = 0.046, not 0.045+1.0.
    assert insertion_m == pytest.approx(0.045 + 0.01 * dt)
    assert ins_cmd == insertion_m
    logged = _build_row_insertion_length_field(insertion_m)
    assert logged == pytest.approx(0.046)


def test_next_controller_call_sees_rate_cap_consistent_history():
    """Chains two ticks through the rate-cap scenario, confirming the
    NEXT controller call's accumulator input matches what was actually
    accepted (0.01 m/s steps), not the originally requested 10 m/s."""
    cfg = _base_cfg()
    advancer = _MockAdvancer(max_rate_m_s=0.01)
    dt = 0.1
    servo_err_log = []
    q0 = np.zeros(6)

    command1 = np.array([0, 0, 0, 0, 0, 0, 10.0])
    ins_cmd, insertion_m, q_cmd, q_cmd_prev, q_target, _, _, _ = accumulator_seam_tick(
        cfg, ins_cmd=None, insertion_m=0.045, q_cmd=None, q=q0,
        corr=command1, command=command1, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert ins_cmd == pytest.approx(0.046)

    command2 = np.array([0, 0, 0, 0, 0, 0, 10.0])
    ins_cmd2, insertion_m2, q_cmd2, q_cmd_prev2, q_target2, _, final_rate2, _ = accumulator_seam_tick(
        cfg, ins_cmd=ins_cmd, insertion_m=insertion_m, q_cmd=q_cmd, q=q_target,
        corr=command2, command=command2, dt=dt, servo_err_log=servo_err_log, advancer=advancer,
    )
    assert final_rate2 == pytest.approx(0.01)
    assert ins_cmd2 == pytest.approx(0.046 + 0.01 * dt), (
        "tick 2 must continue from tick 1's ACTUALLY-ACCEPTED 0.046m, not a value implying "
        "the originally-requested 10 m/s was ever applied"
    )
