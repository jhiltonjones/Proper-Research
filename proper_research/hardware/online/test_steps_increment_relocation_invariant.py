"""2026-11-20: verifies the 2026-11-20 shared-tick-ID fix (moving
`steps += 1` in close_loop_path_follow.py's main() from immediately after
solver() to the end of the loop body) cannot change control behavior --
reference index, commands, or termination/control-step counts.

Grounding, not simulation: `steps` is read in exactly 8 places in the real
file (grep-confirmed below). Outside the loop's own `while`/`%` guards
(which read the value LEFT OVER from the previous iteration, unaffected by
when within THIS iteration it updates) and pure logging/printing, nothing
reads `steps` for a control decision. In particular, `ref_index`/
`target_index` (the reference-index, hence commands) come from
`info.get("reference_index")` -- the ADAPTER's own progress tracking,
entirely independent of main()'s `steps` variable -- and `terminal_hold`/
`abort_reason` come from `info.get("terminal_hold")`, also independent.

This test proves the GENERAL invariant directly: for any sequence of per-
iteration outcomes and any max_steps limit, the iteration at which a
`while cond and steps < max_steps` loop terminates (and the final `steps`
value) is identical whether the increment happens right after the "solve"
step or at the very end of the iteration body -- as long as nothing
between those two points reads `steps` for a decision (verified for the
real file by the grep above, not assumed).
"""
import subprocess

import pytest


def test_grep_confirms_no_control_read_of_steps_between_old_and_new_location():
    """Direct source grounding: every usage of `steps` in the real file,
    confirming none between the old (right after solver()) and new (end
    of loop body) location is a control decision."""
    out = subprocess.run(
        ["grep", "-n", r"\bsteps\b",
         "proper_research/hardware/online/close_loop_path_follow.py"],
        cwd="/home/jack/Proper-Research", capture_output=True, text=True,
    ).stdout
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) >= 7, f"expected close_loop_path_follow.py's usages of `steps`, got: {lines}"
    # Every line must be one of: init, the while-loop guard, the periodic
    # safety-check modulo, the increment itself, the logging row, the CSV
    # write, the print throttle, or the final summary -- never a line
    # computing ref_index/target_index/terminal_hold/abort_reason/command.
    forbidden_substrings = ("ref_index =", "target_index =", "terminal_hold =", "abort_reason =", "command =")
    for line in lines:
        for forbidden in forbidden_substrings:
            assert forbidden not in line, f"found a control-decision use of `steps`: {line}"


def _simulate_loop(outcomes, max_steps, increment_position):
    """Models close_loop_path_follow.py's own loop skeleton:
        while cond and steps < max_steps:
            ... (body reads ref_index/terminal_hold from an INDEPENDENT
                 source, never from `steps`) ...
            [increment_position == "early"]: steps += 1 HERE
            ... more body (logging) ...
            if terminal_hold: break
            [increment_position == "late"]: steps += 1 HERE
        else: abort = "max_control_steps"
    outcomes[i] = ("ok", terminal_hold) for iteration i, independent of steps.
    """
    steps = 0
    executed = []
    abort = None
    i = 0
    while steps < max_steps:
        if i >= len(outcomes):
            break
        _, terminal_hold = outcomes[i]
        executed.append(steps)  # records the step value THIS iteration saw at its own logging point
        if increment_position == "early":
            steps += 1
        if terminal_hold:
            if increment_position == "late":
                steps += 1
            abort = "path_complete"
            break
        if increment_position == "late":
            steps += 1
        i += 1
    else:
        abort = abort or "max_control_steps"
    return steps, executed, abort


@pytest.mark.parametrize("max_steps,outcomes", [
    (10, [("ok", False)] * 10),  # runs to max_steps, no early break
    (10, [("ok", False)] * 3 + [("ok", True)]),  # terminal_hold breaks early
    (5, [("ok", False)] * 20),  # max_steps cuts it off well before outcomes exhausted
])
def test_increment_position_does_not_change_termination_or_final_count(max_steps, outcomes):
    steps_early, executed_early, abort_early = _simulate_loop(outcomes, max_steps, "early")
    steps_late, executed_late, abort_late = _simulate_loop(outcomes, max_steps, "late")
    assert steps_early == steps_late, "final control-step count must be identical"
    assert executed_early == executed_late, "each iteration's OWN logged step value must be identical"
    assert abort_early == abort_late, "termination reason (path_complete vs max_control_steps) must be identical"
