"""2026-11-19: exercises a bound-crossing insertion request through the
ACTUAL shared dispatch path (close_loop_path_follow.py's accumulator_seam
branch), not a standalone np.clip reproduction. Covers both MPC's and
INV's real runner CLI argument-to-PathFollowConfig wiring, and the
multi-tick consistency of the clipped value across the accumulator,
the next tick's own starting point, and the logged row field -- all
using the SAME source lines read from the file, not reinvented.

No hardware: both runners' main() is driven with --skip-preflight, and
pf.main (the function that would connect camera/robot/servo) is
monkeypatched to raise a sentinel exception immediately on entry, so
execution stops the instant hardware would otherwise be touched. The
worker IS spawned for the MPC runner (process-isolated, no hardware --
same as every other offline test this session uses).
"""
import sys

import numpy as np
import pytest

import proper_research.hardware.online.close_loop_path_follow as pf


class _StopBeforeHardware(Exception):
    pass


def _stub_pf_main():
    raise _StopBeforeHardware()


PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"


def test_mpc_runner_cli_flags_populate_pathfollowconfig(monkeypatch):
    monkeypatch.setattr(pf, "main", _stub_pf_main)
    import proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor as runner

    argv = [
        "run_mpc_delay_aware_insertion_anchor.py",
        "--plan-dir", PLAN_DIR, "--schedule-cache", SCHED,
        "--out-dir", "/tmp", "--skip-preflight", "--protective-config-reviewed",
        "--insertion-command-min-mm", "30.0", "--insertion-command-max-mm", "50.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runner.main()
        pytest.fail("expected _StopBeforeHardware before any hardware connection")
    except _StopBeforeHardware:
        pass
    finally:
        if getattr(runner, "_WORKER", None) is not None:
            runner._WORKER.close()
            runner._WORKER = None

    assert pf.CONFIG.insertion_command_min_m == pytest.approx(0.030)
    assert pf.CONFIG.insertion_command_max_m == pytest.approx(0.050)


def test_inv_runner_cli_flags_populate_pathfollowconfig(monkeypatch):
    monkeypatch.setattr(pf, "main", _stub_pf_main)
    import proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware as runner

    argv = [
        "run_inv_7dof_delay_aware.py",
        "--plan-dir", PLAN_DIR, "--schedule-cache", SCHED,
        "--out-dir", "/tmp", "--skip-preflight",
        "--insertion-command-min-mm", "30.0", "--insertion-command-max-mm", "50.0",
        # 2026-11-20: run_inv_7dof_delay_aware.py gained the same
        # --protective-config-reviewed gate run_mpc_delay_aware_
        # insertion_anchor.py already had (condition C previously had no
        # such gate at all -- a real asymmetry found while preparing
        # condition C's launch command). Required for any non-dry-run
        # start; this test isn't exercising that gate, so satisfy it.
        "--protective-config-reviewed",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runner.main()
        pytest.fail("expected _StopBeforeHardware before any hardware connection")
    except _StopBeforeHardware:
        pass

    assert pf.CONFIG.insertion_command_min_m == pytest.approx(0.030)
    assert pf.CONFIG.insertion_command_max_m == pytest.approx(0.050)


# 2026-11-20: the standalone np.clip reproduction previously here was
# replaced by proper_research.hardware.online.test_accumulator_seam_
# tick_bound_crossing, which calls close_loop_path_follow.
# accumulator_seam_tick() -- the function main() itself was refactored
# to call this same turn -- directly, with a mock AdvancerSink, instead
# of reproducing the clip/accumulator/logging arithmetic. See that
# file for the bound-crossing, multi-tick-continuity, and logged-row
# consistency checks. This file keeps only the CLI-flag-population
# checks above, which that file does not duplicate.
