"""2026-11-19: regression test for the --protective-config-reviewed startup
gate. A prior version of this check lived inside _wrapped_build_offline_
solver(), a module-level function with no `args` in scope -- it raised
NameError at runtime instead of ever refusing anything (a real hardware
run hit this). The check now lives in main(), right after argparse, before
any schedule/worker/robot setup -- this test calls main() directly and
confirms it raises SystemExit with the expected message BEFORE touching
the (nonexistent) schedule-cache path, i.e. no hardware or file I/O is
reached. No hardware.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.rectangle_stage_a import run_mpc_delay_aware_insertion_anchor as mod


def _argv(extra: list[str]) -> list[str]:
    return [
        "--plan-dir", "/nonexistent/plan/dir",
        "--out-dir", "/nonexistent/out/dir",
        "--run-name", "test",
        "--schedule-cache", "/nonexistent/schedule.npy",
        *extra,
    ]


def test_refuses_without_protective_config_reviewed(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog"] + _argv([]))  # no --dry-run, no --protective-config-reviewed
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert "REFUSING to start" in str(exc_info.value)
    assert "protective" in str(exc_info.value).lower()


def test_dry_run_bypasses_the_gate(monkeypatch):
    """--dry-run must still bypass the gate (no real hardware motion
    happens in dry-run mode) -- confirms the check's condition, not just
    that it fires unconditionally. Without the gate raising, execution
    proceeds past it to the plan/schedule verification on a nonexistent
    path -- that's expected to fail too, but with a DIFFERENT exception
    (RuntimeError, from common.py's own unrelated plan-verification gate),
    proving the protective-config gate itself did not fire here."""
    monkeypatch.setattr(sys, "argv", ["prog"] + _argv(["--dry-run"]))
    with pytest.raises(RuntimeError) as exc_info:
        mod.main()
    assert "REFUSING to start" not in str(exc_info.value)


def test_protective_config_reviewed_flag_bypasses_the_gate(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"] + _argv(["--protective-config-reviewed"]))
    with pytest.raises(RuntimeError) as exc_info:
        mod.main()
    assert "REFUSING to start" not in str(exc_info.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
