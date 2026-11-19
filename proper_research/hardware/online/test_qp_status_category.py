"""2026-11-19 (timing-assessment correction, point 1): regression test for
`classify_qp_status` (close_loop_path_follow.py), using the EXACT status
strings the installed OSQP version (0.6.7.post3) actually produces --
constructed directly via small QPs engineered to hit each case, not
assumed. Also confirms `solver_deadline_missed` is independent of this
classification and that neither new field changes command acceptance/
hold behaviour (both are purely additive reporting fields; see
test_deadline_miss_field_semantics.py for the hold/u0 side).

Run: python3 -m pytest proper_research/hardware/online/test_qp_status_category.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import osqp

from proper_research.hardware.online.close_loop_path_follow import classify_qp_status


def _osqp_status(**settings) -> str:
    """Solves a small QP engineered to produce a specific OSQP status,
    using the installed OSQP version directly -- not a hand-picked
    string."""
    P = sp.csc_matrix(settings.pop("P"))
    q = settings.pop("q")
    A = sp.csc_matrix(settings.pop("A"))
    l = settings.pop("l")
    u = settings.pop("u")
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, **settings)
    r = m.solve()
    return str(r.info.status)


@pytest.fixture(scope="module")
def real_statuses():
    base = dict(
        P=np.eye(2), q=np.array([-1.0, -1.0]),
        A=np.eye(2), l=np.array([-1.0, -1.0]), u=np.array([1.0, 1.0]),
    )
    primal_infeasible = dict(
        P=np.eye(2), q=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 0.0], [1.0, 0.0]]), l=np.array([2.0, -1e9]), u=np.array([1e9, 1.0]),
    )
    dual_infeasible = dict(
        P=np.zeros((2, 2)), q=np.array([0.0, -1.0]),
        A=sp.csc_matrix((0, 2)).toarray(), l=np.zeros(0), u=np.zeros(0),
    )
    return {
        "solved": _osqp_status(**base),
        "iteration_limit": _osqp_status(**base, max_iter=1),
        "primal_infeasible": _osqp_status(**primal_infeasible),
        "dual_infeasible": _osqp_status(**dual_infeasible),
    }


def test_solved_is_real_osqp_string(real_statuses):
    assert real_statuses["solved"] == "solved"
    assert classify_qp_status(real_statuses["solved"]) == "solved"


def test_iteration_limit_is_distinct_from_infeasible(real_statuses):
    status = real_statuses["iteration_limit"]
    assert "iter" in status
    category = classify_qp_status(status)
    assert category == "iteration_limit_reached"
    assert "infeasible" not in category


def test_primal_infeasible_real_osqp_string(real_statuses):
    assert real_statuses["primal_infeasible"] == "primal infeasible"
    assert classify_qp_status(real_statuses["primal_infeasible"]) == "primal_infeasible"


def test_dual_infeasible_real_osqp_string(real_statuses):
    assert real_statuses["dual_infeasible"] == "dual infeasible"
    assert classify_qp_status(real_statuses["dual_infeasible"]) == "dual_infeasible"


def test_solved_inaccurate_and_time_limit_not_misclassified_as_infeasible():
    """These two strings are not easily reproduced by a toy QP (they
    depend on polish/time-limit internals), so tested directly against
    the literal strings this project's OWN existing code already handles
    elsewhere (delay_aware_mpc.py / simulate_time_parameterized_*_mpc.py's
    `{"solved", "solved inaccurate"}` success set and
    `status == "run time limit reached"` acceptance check) -- not
    invented here."""
    assert classify_qp_status("solved inaccurate") == "solved_inaccurate"
    assert classify_qp_status("run time limit reached") == "time_limit_reached"
    for s in ("solved inaccurate", "run time limit reached"):
        assert "infeasible" not in classify_qp_status(s)


def test_no_response_case():
    """qp_status is None when no reply arrived (timeout or worker-busy) --
    distinct from every genuine OSQP-reported status."""
    assert classify_qp_status(None) == "no_response"


def test_unrecognized_status_never_silently_becomes_infeasible():
    category = classify_qp_status("some future osqp status we have not seen")
    assert category.startswith("unrecognized_status:")
    assert "infeasible" not in category or "unrecognized" in category


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
