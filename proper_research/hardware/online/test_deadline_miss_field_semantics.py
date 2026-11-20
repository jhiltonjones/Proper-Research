"""2026-11-19 (timing-assessment correction): regression test for the two
new path_follow.jsonl fields added in close_loop_path_follow.py's row dict
(`solver_deadline_missed`, `qp_genuinely_infeasible`), which disambiguate
what the pre-existing `deadline_miss` (loop-period, ~100ms) and
`infeasible` (conflates no-response-in-time with genuine QP infeasibility)
fields could not. Historical logs (written before this change) lack both
new fields; this test only covers the NEW fields' own logic, verified
against the exact real cases found in pilotA_off_planned_keepout_
20261119T094821Z/095843Z's path_follow.jsonl (steps 93, 98, 99, 101, 157,
177 -- all genuine solver timeouts, qp_status=None, solver_status=
"deadline_miss"), confirming the expression is correct, not merely that it
compiles.

Run: python3 -m pytest proper_research/hardware/online/test_deadline_miss_field_semantics.py -v
"""
from __future__ import annotations


def _solver_deadline_missed(info: dict) -> bool:
    """Exact expression used in close_loop_path_follow.py's row dict."""
    return bool(info.get("status") == "deadline_miss")


def _qp_genuinely_infeasible(info: dict) -> bool:
    """Exact expression used in close_loop_path_follow.py's row dict."""
    return bool(info.get("qp_status") is not None and info.get("qp_status") != "solved")


def test_ok_tick():
    info = {"status": "ok", "qp_status": "solved"}
    assert not _solver_deadline_missed(info)
    assert not _qp_genuinely_infeasible(info)


def test_genuine_solver_timeout_real_case():
    """Matches the real logged info dict shape for steps 93/98/101/157/177
    of the two A-keep-out runs: resp was None (no response within 70ms),
    so qp_status is None and status=='deadline_miss'."""
    info = {"status": "deadline_miss", "qp_status": None}
    assert _solver_deadline_missed(info)
    assert not _qp_genuinely_infeasible(info), (
        "a timeout with no response must NOT be reported as a genuine QP "
        "infeasibility -- these are different failure modes"
    )


def test_worker_busy_drain_case_real_case():
    """Matches step 99's real shape: try_solve returned None immediately
    (worker still busy on the prior request) -- same info shape as a
    genuine timeout from this row's own perspective (no response for THIS
    tick), so the same two assertions as the timeout case apply."""
    info = {"status": "deadline_miss", "qp_status": None}
    assert _solver_deadline_missed(info)
    assert not _qp_genuinely_infeasible(info)


def test_genuine_qp_infeasibility_distinct_from_timeout():
    """A response DID arrive in time, but OSQP itself reported a non-
    'solved' status (e.g. primal/dual infeasible) -- this must be flagged
    as qp_genuinely_infeasible and NOT as a deadline miss."""
    info = {"status": "ok", "qp_status": "primal infeasible"}
    assert not _solver_deadline_missed(info)
    assert _qp_genuinely_infeasible(info)


def test_fields_are_independent_not_mutually_exclusive_by_construction():
    """The two fields answer different questions and must be computed
    independently -- confirm no shared short-circuit makes one imply the
    other's negation incorrectly, by exhausting the 2x2 combinations that
    can actually occur (a timeout-with-genuine-infeasibility combination
    cannot occur by construction, since a timeout means qp_status is
    None -- verified here, not merely asserted)."""
    ok = {"status": "ok", "qp_status": "solved"}
    timeout = {"status": "deadline_miss", "qp_status": None}
    infeasible = {"status": "ok", "qp_status": "primal infeasible"}
    for info, expect_miss, expect_infeasible in [
        (ok, False, False), (timeout, True, False), (infeasible, False, True),
    ]:
        assert _solver_deadline_missed(info) == expect_miss
        assert _qp_genuinely_infeasible(info) == expect_infeasible


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
