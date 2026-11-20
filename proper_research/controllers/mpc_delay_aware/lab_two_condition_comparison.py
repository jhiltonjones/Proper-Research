"""ONE command, TWO conditions, direct comparison: current numerical-
thread configuration (whatever it actually is on the machine this runs
on -- queried live, never assumed) vs. the opt-in, runtime-VERIFIED
one-thread configuration (MPC_WORKER_VERIFIED_SINGLE_THREAD=1).

Held FIXED across both conditions, per the user's own "don't conflate
two factors" instruction: workload (same input log/tick window as every
prior comparison in this investigation), code (same commit state --
hashed below), scheduling priority (whatever this machine's real
niceness is -- queried and reported, never forced to any particular
value here; nice=0 is a SEPARATE, already-deferred question), 70ms
deadline, 100ms cadence, N_REPS=3 (same as every other comparison this
investigation has used).

For EACH condition, reports (queried live, not assumed):
  - parent AND worker nice / sched policy / affinity
  - parent AND worker's REAL, live threadpoolctl thread-pool state
  - per-tick: full adapter turnaround (total_adapter_ms, the actual IPC
    round trip the real 70ms deadline covers), worker-side solve time
    (t_solve_ms, wall-clock), and worker-side CPU time (cpu_solve_ms,
    time.process_time()) as three DISTINCT figures, never collapsed
  - miss count
Then re-confirms the TCP-workspace regression test passes under BOTH
conditions, with real residual numbers for direct comparison (not just
"passed").

This script does NOT change the 70ms deadline, 100ms cadence, or any
plan/tolerance/delay_samples. The one-thread condition is an OPT-IN
diagnostic only (MPC_WORKER_VERIFIED_SINGLE_THREAD=1) -- NOT a
production-default change; this script does not flip any default.
"""
import sys, json, os, hashlib, time
MAIN = "/home/jack/Proper-Research"
sys.path.insert(0, MAIN)

_FORBIDDEN_HW_IDENTIFIERS = ("URRtde", "RobotJointStream", "CameraSource",
                             "rtde_control", "rtde_receive", "RTDEControlInterface", "RTDEReceiveInterface")
with open(__file__, "r") as _f:
    _own_source = _f.read()
for _ident in _FORBIDDEN_HW_IDENTIFIERS:
    assert _own_source.count(_ident) <= 1
print("[hardware guard] confirmed: no robot/camera interface imported or constructed.")


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


RUN_LABEL_BASE = os.environ.get("SUSTAINED_TEST_RUN_LABEL")
if not RUN_LABEL_BASE:
    raise SystemExit(
        "Set SUSTAINED_TEST_RUN_LABEL, e.g.:\n"
        "  SUSTAINED_TEST_RUN_LABEL=threadcompare_labmachine_$(date +%Y%m%dT%H%M%S) python3 ...\n"
        "This single run will produce BOTH conditions, suffixed _current and _verified1thread."
    )

CODE_FILES = [
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/delay_aware_mpc.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/worker_process.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/process_isolated_adapter.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/stagewise_task_nullspace.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/insertion_anchor_mpc.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/sustained_worker_deadline_test.py",
]
INPUT_FILES = [f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_per_tick.json"]

print("\n=== Code hashes ===")
for f in CODE_FILES:
    print(f"  {sha256_of(f)}  {f}")
print("=== Input hashes ===")
for f in INPUT_FILES:
    print(f"  {sha256_of(f)}  {f}")

import proper_research.controllers.mpc_delay_aware.worker_process as wp_module
assert wp_module.__file__ == MAIN + "/proper_research/controllers/mpc_delay_aware/worker_process.py"
assert ".claude/worktrees" not in wp_module.__file__


def run_condition(condition_label, env_overrides):
    run_label = f"{RUN_LABEL_BASE}_{condition_label}"
    out_path = f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/sustained_worker_test_results_{run_label}.json"
    timing_log_path = f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/worker_timing_cpu_wallclock_{run_label}.jsonl"
    for p in (out_path, timing_log_path):
        if os.path.exists(p):
            raise FileExistsError(f"REFUSING TO OVERWRITE: {p} already exists.")

    old_env = {}
    for k, v in env_overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v
    os.environ["SUSTAINED_TEST_RUN_LABEL"] = run_label
    os.environ["MPC_WORKER_TIMING_LOG"] = timing_log_path

    print(f"\n{'='*70}\n=== CONDITION: {condition_label}  (env overrides: {env_overrides}) ===\n{'='*70}")
    try:
        import importlib
        import proper_research.controllers.mpc_delay_aware.sustained_worker_deadline_test as sw
        importlib.reload(sw)  # fresh module state per condition, same code file
        results = sw.main()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return dict(condition=condition_label, out_path=out_path, timing_log_path=timing_log_path, results=results)


if __name__ == "__main__":
    print("\n### CONDITION A: current numerical-thread configuration (whatever it actually is) ###")
    result_a = run_condition("current", {})

    print("\n### CONDITION B: opt-in, runtime-VERIFIED one-thread configuration ###")
    result_b = run_condition("verified1thread", {"MPC_WORKER_VERIFIED_SINGLE_THREAD": "1"})

    print(f"\n\n{'='*70}\n=== TCP-WORKSPACE REGRESSION, re-confirmed under BOTH conditions ===\n{'='*70}")
    import threadpoolctl
    print("-- condition A (current thread config, in-process regression check) --")
    from proper_research.controllers.mpc_delay_aware.tcp_workspace_ax_refresh_self_test import (
        run_self_test, run_combination_regressions,
    )
    run_self_test()
    run_combination_regressions()

    print("\n-- condition B (verified one-thread, applied directly to THIS process for the in-process check) --")
    threadpoolctl.threadpool_limits(1)
    _check = threadpoolctl.threadpool_info()
    assert all(d.get("num_threads") == 1 for d in _check), f"thread limit did not take effect: {_check}"
    print(f"verified: {_check}")
    run_self_test()
    run_combination_regressions()

    print(f"\n\n{'='*70}\n=== SUMMARY ===\n{'='*70}")
    for r in (result_a, result_b):
        print(f"{r['condition']}: output={r['out_path']}")
        print(f"              worker timing log={r['timing_log_path']}")
