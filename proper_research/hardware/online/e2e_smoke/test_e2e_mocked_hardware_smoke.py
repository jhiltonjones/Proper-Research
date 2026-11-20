"""2026-11-20: permanent, hardware-blocked end-to-end regression for
close_loop_path_follow.main() + run_mpc_delay_aware_insertion_anchor.main(),
run as a real subprocess (not an in-process import) so the real worker's
own multiprocessing.Process spawn and the module-level RTDE-construction
poisoning in _mocked_hardware_harness.py behave exactly as in manual
verification. This is what found and exercises the real main-loop
consumer of the delta_q fix (close_loop_path_follow.py's `if np.any(np.abs
(delta_q) > 1e-9): started_moving = True` check, right after
accumulator_seam_tick() returns) -- a prior refactor broke this with an
UnboundLocalError that no isolated unit test of the extracted function
could have caught, because none of them execute the surrounding main()
code that reads delta_q afterward. Each scenario below reaching real
control ticks (not crashing) is the regression check for that fix.

No real hardware is reachable from this process: URRTDERobot and
RobotJointStream are replaced outright, and rtde_control.RTDEControlInterface
/ rtde_receive.RTDEReceiveInterface (still real, importable .so modules --
import alone opens no socket) are replaced with classes whose __init__
raises immediately, so any accidental construction attempt fails fast and
explicitly instead of opening a real connection.
"""
import json
import os
import subprocess
import sys

import pytest

HARNESS = os.path.join(os.path.dirname(__file__), "_mocked_hardware_harness.py")
REPO_ROOT = "/home/jack/Proper-Research"


def _run_scenario(scenario, out_dir):
    result = subprocess.run(
        [sys.executable, "-u", HARNESS, scenario, str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"scenario={scenario!r} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    scenario_dir = os.path.join(out_dir, scenario)
    run_dirs = [
        os.path.join(scenario_dir, d)
        for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d))
    ]
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    return run_dirs[0]


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_normal_scenario_completes_all_ticks_no_dispatch_crash(tmp_path):
    """Exercises the real delta_q main-loop consumer across 8 real ticks
    with a self-consistent (static) synthetic plant -- genuinely feasible
    QP solves throughout, not an artifact of an inconsistent measurement."""
    run_dir = _run_scenario("normal", tmp_path)
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    pf_rows = _load_jsonl(os.path.join(run_dir, "path_follow.jsonl"))
    assert len(pred_rows) == 8
    assert len(pf_rows) == 8
    assert all(r["success"] for r in pred_rows), [r["qp_status"] for r in pred_rows]
    assert all(r["qp_status"] == "solved" for r in pred_rows)
    # request identity: every real tick sends a genuine, increasing request
    assert [r["worker_request_id"] for r in pred_rows] == list(range(1, 9))
    # joint-sample ages are real and non-negative
    for r in pred_rows:
        age = r.get("clearance_sample_age_s")
        assert age is not None and age >= 0.0, r


def test_timeout_scenario_misses_deadline_without_fabricating_request_ids(tmp_path):
    run_dir = _run_scenario("timeout", tmp_path)
    pf_rows = _load_jsonl(os.path.join(run_dir, "path_follow.jsonl"))
    assert len(pf_rows) >= 1
    # a 0.01ms deadline is essentially unmeetable, but real scheduling
    # jitter means an occasional tick can still land in time -- require
    # the large majority to miss rather than every single one, so this
    # stays a meaningful regression check without being timing-flaky.
    miss_count = sum(1 for r in pf_rows if r["solver_deadline_missed"])
    assert miss_count >= max(1, len(pf_rows) - 1), (
        f"expected most ticks to miss a 0.01ms deadline, got {miss_count}/{len(pf_rows)}"
    )
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    # a request WAS sent each tick (genuinely submitted, just not answered
    # in time) -- never fabricated, but also never silently dropped
    assert [r["worker_request_id"] for r in pred_rows] == list(range(1, len(pred_rows) + 1))
    assert all(r["qp_status"] is None for r in pred_rows)


def test_abort_scenario_dispatches_nothing_before_or_after(tmp_path):
    run_dir = _run_scenario("abort", tmp_path)
    pf_rows = _load_jsonl(os.path.join(run_dir, "path_follow.jsonl"))
    assert pf_rows == [], "a pre-solve abort must log zero completed control ticks"
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    assert len(pred_rows) == 1
    row = pred_rows[0]
    assert row["early_abort_reason"] is not None and "magnet_exclusion_violated" in row["early_abort_reason"]
    # no request was ever sent for a pre-solve abort -- not fabricated
    assert row["worker_request_id"] is None
    assert row["predicted_commands"] is None


def test_freeze_jacobian_flag_builds_and_uses_a_constant_schedule(tmp_path):
    """Condition B: --freeze-jacobian (added 2026-11-20) must build a
    schedule file whose every row equals the row at --freeze-jacobian-
    index, and the real worker must accept and solve against it with no
    code change on the worker side."""
    run_dir = _run_scenario("frozen_jacobian", tmp_path)
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    assert len(pred_rows) == 8
    assert all(r["success"] for r in pred_rows), [r["qp_status"] for r in pred_rows]

    frozen_path = os.path.join(
        os.path.dirname(run_dir), "e2e_smoke_frozen_jacobian_frozen_jacobian_schedule.npy"
    )
    assert os.path.exists(frozen_path), f"expected frozen schedule file at {frozen_path}"
    import numpy as np
    schedule = np.load(frozen_path)
    assert schedule.ndim == 3
    for row in schedule:
        assert np.array_equal(row, schedule[0]), "every row of a frozen schedule must be identical"


def test_adaptation_active_runs_real_estimator_across_consecutive_ticks(tmp_path):
    """Condition E: --jacobian-adaptation-mode active must run the real
    online g_hat estimator across real consecutive ticks (not just
    parse without crashing) -- a single isolated try_solve() call can't
    exercise this, since the estimator needs consecutive measured-vs-
    predicted deltas. This mocked plant has almost no excitation, so
    every tick is expected to reject the update (insufficient_
    excitation) rather than accept one -- that is the estimator's own
    designed safety behavior under low excitation, not a bug; the
    regression check here is that the diagnostics are genuinely
    populated and structured correctly, not that an update was
    accepted."""
    run_dir = _run_scenario("adaptation_active", tmp_path)
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    assert len(pred_rows) == 20
    assert all(r["success"] for r in pred_rows), [r["qp_status"] for r in pred_rows]
    for r in pred_rows:
        diag = r.get("jacobian_adaptation_diagnostics")
        assert diag is not None, "active mode must populate diagnostics every successful tick"
        assert diag["mode"] == "active"
        assert "g_hat" in diag and "g_used" in diag and "accepted" in diag


def test_experimental_magnet_exclusion_radius_is_additive_not_destructive(tmp_path):
    """Constraint study (CONSTRAINT_STUDY_HARDWARE_PROPOSAL.md): --experimental-
    magnet-exclusion-radius-mm (added 2026-11-20) must raise the EFFECTIVE
    in-QP radius without ever modifying the real, reviewed floor/margin
    values -- controller_metadata.json must show floor/margin unchanged at
    their passed-in values (230/3mm), the pre-existing "commanded" field
    unchanged at floor+margin (233mm, byte-identical to every prior A/B/D/E
    run), and the NEW effective-radius field reflecting the larger
    experimental value actually used in the QP."""
    run_dir = _run_scenario("constrained", tmp_path)
    pred_rows = _load_jsonl(os.path.join(run_dir, "predicted_beam_positions.jsonl"))
    assert len(pred_rows) == 8
    assert all(r["success"] for r in pred_rows), [r["qp_status"] for r in pred_rows]

    with open(os.path.join(run_dir, "controller_metadata.json")) as f:
        meta = json.load(f)
    assert meta["magnet_beam_base_exclusion_floor_mm"] == 230.0
    assert meta["magnet_exclusion_robust_margin_mm"] == 3.0
    assert meta["magnet_beam_base_exclusion_commanded_radius_mm"] == 233.0
    assert meta["magnet_exclusion_experimental_radius_mm"] == 276.0
    assert meta["magnet_exclusion_effective_radius_mm"] == 276.0
