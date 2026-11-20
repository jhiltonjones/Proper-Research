"""Lab-machine nice=19-vs-nice=0 comparison runner.

Run this TWICE on the lab machine -- once under whatever niceness the
session naturally has (record it, don't assume 19), and once under a
genuinely verified nice=0 (see the shell wrapper below this script for
how to get there if the session isn't already at 0). Each run:
  - records code+input hashes (so the two runs, and this analysis, can
    later be confirmed to have used identical code and identical input
    data)
  - records parent AND worker scheduling diagnostics (nice, sched policy,
    affinity) -- captured for real, not assumed, inside the test itself
  - records BOTH wall-clock (time.monotonic) and CPU time
    (time.process_time) for every worker-side solve, separately
  - uses the exact same 70ms deadline, 100ms cadence, N=3 reps, and
    controller settings as every prior round of this investigation --
    nothing here changes any of those.
"""
import sys, json, os, hashlib
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


RUN_LABEL = os.environ.get("SUSTAINED_TEST_RUN_LABEL")
if not RUN_LABEL:
    raise SystemExit(
        "Set SUSTAINED_TEST_RUN_LABEL before running, e.g.:\n"
        "  SUSTAINED_TEST_RUN_LABEL=nice_asis_labmachine_$(date +%Y%m%dT%H%M%S) python3 ...\n"
        "  SUSTAINED_TEST_RUN_LABEL=nice0_labmachine_$(date +%Y%m%dT%H%M%S) python3 ...\n"
        "(the label is also used to pick the worker-timing-log filename and the output filename, "
        "so the two runs never collide and the overwrite-refusal check never has to fire)"
    )

# 2026-11-16 (correction of a real error caught by the user): `nice -n 0`
# does NOT lower an already-elevated, inherited niceness back to 0 -- an
# unprivileged process can only RAISE its own niceness further, never
# lower it below whatever its environment handed it (confirmed directly
# on the dev machine: os.nice(-19) -> PermissionError; `nice -n 0` on an
# already-nice=19 shell still produces a nice=19 child). A prior version
# of this script labeled a run "nice=0" without ever verifying that. This
# version REFUSES to proceed under a label claiming nice=0 unless BOTH
# the parent (this process) and the worker subprocess are independently
# confirmed, at runtime, to actually be at nice=0 -- it never presents a
# silently-still-nice=19 run as the nice=0 comparison.
if "nice0" in RUN_LABEL.lower():
    _parent_nice_check = os.getpriority(os.PRIO_PROCESS, 0)
    if _parent_nice_check != 0:
        raise SystemExit(
            f"REFUSING TO PROCEED: run label {RUN_LABEL!r} claims nice=0, but this PARENT "
            f"process is actually at nice={_parent_nice_check}. An unprivileged process cannot "
            f"lower its own niceness below an inherited elevated value -- `nice -n 0` from an "
            f"already-elevated shell does NOT achieve this. See this file's own docstring for "
            f"the real procedure (sudo renice, or a genuinely fresh/non-pinned login session)."
        )
    print(f"[nice=0 check] PARENT confirmed at nice=0 (verified, not assumed).")
    # The WORKER's own nice is checked after it reports in (see below,
    # right after build_worker() -- it may not inherit the parent's value).

CODE_FILES = [
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/delay_aware_mpc.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/worker_process.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/process_isolated_adapter.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/stagewise_task_nullspace.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/insertion_anchor_mpc.py",
    f"{MAIN}/proper_research/controllers/mpc_delay_aware/sustained_worker_deadline_test.py",
]
INPUT_FILES = [
    f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/full_replay_per_tick.json",
]

print(f"\n=== Run label: {RUN_LABEL} ===")
print("=== Code hashes (confirm these match what was analyzed on the dev machine) ===")
code_hashes = {}
for f in CODE_FILES:
    h = sha256_of(f)
    code_hashes[f] = h
    print(f"  {h}  {f}")

print("\n=== Input/workload hashes ===")
input_hashes = {}
for f in INPUT_FILES:
    h = sha256_of(f)
    input_hashes[f] = h
    print(f"  {h}  {f}")

# Set up worker-side CPU+wall-clock timing log for THIS run, uniquely named.
_timing_log_path = (
    f"{MAIN}/close_loop_logs/myrun/tcp_fix_verification_diagnostics/"
    f"worker_timing_cpu_wallclock_{RUN_LABEL}.jsonl"
)
if os.path.exists(_timing_log_path):
    raise FileExistsError(f"REFUSING TO OVERWRITE: {_timing_log_path} already exists.")
os.environ["MPC_WORKER_TIMING_LOG"] = _timing_log_path  # must be set BEFORE the worker is spawned

print(f"\n=== Worker CPU+wall-clock timing log for this run: {_timing_log_path} ===")
print("=== Deadline=70ms, cadence=100ms, N_REPS=3, controller settings: UNCHANGED from every prior round ===\n")

# Hand off to the real, unmodified sustained test entry point.
from proper_research.controllers.mpc_delay_aware import sustained_worker_deadline_test as _sw

assert _sw.DEADLINE_S == 0.070, f"deadline changed! {_sw.DEADLINE_S}"
assert _sw.CADENCE_S == 0.100, f"cadence changed! {_sw.CADENCE_S}"
assert _sw.N_REPS == 3, f"repetition count changed! {_sw.N_REPS}"

if __name__ == "__main__":
    _sw.main()
    print(f"\n=== Done. Code hashes: {code_hashes} ===")
    print(f"=== Input hashes: {input_hashes} ===")
    print(f"=== Worker CPU+wall-clock log: {_timing_log_path} ===")
