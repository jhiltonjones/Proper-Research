"""2026-11-20: two focused checks on the REAL AdvancerSink class (not a
reimplementation):

1. submit_rate now returns the FINAL accepted rate (after its own
   |rate|<=max_rate_m_s clamp) -- previously returned None, with no way
   for a caller to observe the clamp. Verified directly against the
   production clamp formula.

2. "Does each submitted command expire or remain active beyond its
   assumed integration interval?" -- answered by reading and exercising
   the real class: submit_rate integrates into a PERSISTENT running
   total (_commanded_mm/_residual_mm), dispatched asynchronously by a
   SEPARATE writer thread at its own cadence (dispatch_period_s),
   decoupled from the dt any caller passes to submit_rate. A submitted
   rate's effect does NOT expire after one dt -- it accumulates in
   _residual_mm until a future writer-thread tick dispatches a whole
   quantum (mm_per_step) of it, and any leftover carries forward again.
   This is confirmed here by calling submit_rate several times with the
   writer thread never started (dry_run construction without start())
   and observing _residual_mm keep growing across calls, not reset.

No hardware: dry_run=True, and the writer/reader threads are never
started in test 2 (constructed, start() not called), so this never
opens a serial port.
"""
import pytest

from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig


def test_submit_rate_returns_accepted_rate_after_own_clamp():
    sink = AdvancerSink(AdvancerSinkConfig(dry_run=True, max_rate_m_s=0.01))
    accepted = sink.submit_rate(10.0, 0.1)
    assert accepted == pytest.approx(0.01), "a request far exceeding max_rate_m_s must return the clamped rate"

    accepted = sink.submit_rate(-10.0, 0.1)
    assert accepted == pytest.approx(-0.01)

    accepted = sink.submit_rate(0.005, 0.1)
    assert accepted == pytest.approx(0.005), "a request within the cap must be returned unmodified"

    accepted = sink.submit_rate(float("nan"), 0.1)
    assert accepted == 0.0, "a NaN request must return 0.0, not None or NaN"


def test_submitted_commands_integrate_persistently_not_per_tick():
    """Commands do not expire at the end of their own dt -- they
    accumulate in _residual_mm, a running total the (separately
    clocked) writer thread drains asynchronously. Demonstrated here by
    calling submit_rate repeatedly with the writer thread never started
    and observing the residual keep growing, not reset between calls."""
    sink = AdvancerSink(AdvancerSinkConfig(dry_run=True, max_rate_m_s=1.0))
    dt = 0.1
    # Each call below would move 0.01m/s * 0.1s * 1000 = 1.0mm if applied
    # once; if commands "expired" after one dt with no persistent
    # accumulation, residual would never exceed one call's own 1.0mm.
    for _ in range(5):
        sink.submit_rate(0.01, dt)
    feedback = sink.feedback()
    assert feedback.residual_mm == pytest.approx(5 * 1.0), (
        f"expected the residual to accumulate across all 5 submitted ticks (5.0mm), "
        f"got {feedback.residual_mm} -- if this is ever 1.0mm, commands are being "
        f"treated as expiring after one dt instead of integrating persistently"
    )
    assert feedback.commanded_distance_mm == pytest.approx(5.0)
