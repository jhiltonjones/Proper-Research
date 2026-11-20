# K2 post-TCP-fix / post-thread-fix operator procedure (2026-11-16)

Prep only. No hardware execution and no git commands were run by the
assistant that produced this document — everything below is for the
operator to run themselves, after reviewing.

## 1. Offline regressions to run first, in the lab venv

From `/home/jack/Proper-Research`, with the lab's venv activated:

```bash
cd /home/jack/Proper-Research
source .venv/bin/activate   # or whatever activates the lab's own venv

python3 -m proper_research.controllers.mpc_delay_aware.tcp_workspace_ax_refresh_self_test
python3 -m proper_research.controllers.mpc_delay_aware.single_thread_enforcement_regression_test
python3 -m proper_research.controllers.mpc_delay_aware.sustained_worker_deadline_test
python3 -m proper_research.controllers.mpc_delay_aware.continuous_mocked_execution_test
```

All four must pass before proceeding to hardware. `single_thread_enforcement_regression_test.py`
and `sustained_worker_deadline_test.py` now exercise the FINAL PRODUCTION
code path — no experimental environment variable is needed or used.

## 2. Startup confirmation to look for (both BLAS backends, 1 thread, unconditional)

At worker startup, with no flag of any kind set, the real worker now
prints exactly (confirmed live, this session):

```
[worker] single_threaded=True: VERIFIED num_threads=1 for all backends: [{'user_api': 'blas', 'internal_api': 'openblas', ..., 'num_threads': 1, ... 'filepath': '.../numpy.libs/libopenblas...'}, {'user_api': 'blas', 'internal_api': 'openblas', ..., 'num_threads': 1, ... 'filepath': '.../scipy.libs/libscipy_openblas...'}]
```

Both entries (numpy's own OpenBLAS and scipy's separate bundled OpenBLAS)
must show `'num_threads': 1`. If this line is missing, or either entry
shows anything other than 1, or the worker fails to start at all — do
not proceed to hardware; this confirmation did not previously exist in
the code at all (added this round specifically so there's something
real to check for).

## 3. Plan selection — a decision for you to confirm, not a default I'm choosing

Two real candidates, live hashes recomputed just now (not reused from
memory):

| Plan | Path | SHA-256 (live) | Pros | Cons |
|---|---|---|---|---|
| **Original** | `plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz` | `62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613` | Same plan as both official K2 STOP runs (N=15, N=3) — required for a directly comparable delay-identification result | Has its own, SEPARATE, unfixed acceleration-margin bug near the tail (already root-caused in an earlier round of this investigation, not touched by anything in this round) |
| **Marginfix** | `plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz` | `4f00089bf3592bdc9c1f4bdd74827de221fc3152aeb7ed5665499299db443274` | Fixes the acceleration-margin issue; the one actually used throughout this entire TCP-fix/thread-fix investigation; enables genuine full-path completion | NOT the plan either official STOP run used — a run on this plan cannot be compared directly against them for delay identification |

**My recommendation, for you to confirm or override**: if the purpose of
the next supervised run is to validate the TCP-workspace and
single-threaded-BLAS fixes end-to-end with a real chance of reaching the
plan's actual end, use **marginfix** — it's the one every real result in
this investigation (0/90 misses, 60/60 continuous ticks, etc.) was
actually measured against. If the purpose is instead to produce a new,
directly-comparable delay-identification data point against the two
official STOP results, you must use the **original** plan instead,
accepting that its separate acceleration-margin bug could still interrupt
the run near the tail for a completely unrelated reason. These two goals
cannot both be served by one run on one plan. I am not deciding this for
you — please confirm which one before running.

## 4. Hardware command, expected outputs, unchanged stop conditions

Exact command (substitute the plan you selected in step 3 — shown here
for marginfix; swap the `--plan-dir`/`--schedule-cache` for the original
plan if you chose that instead):

```bash
cd /home/jack/Proper-Research
source .venv/bin/activate
python3 proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy \
  --run-name k2_post_tcpfix_threadfix_$(date +%Y%m%dT%H%M%SZ) \
  --horizon 15 --max-tracking-error-mm 7 --tcp-workspace-robust-margin-mm 15
```

**Expected output files** (under `close_loop_logs/myrun/<run-name>/`):
`path_follow.jsonl`, `summary.json`, `controller_metadata.json` (now
includes the TCP-fix provenance fields from earlier in this
investigation), `startup_stationary_tips.jsonl`, `predicted_beam_positions.jsonl`.
**Also capture the full console output** — it now contains the item-2
confirmation line above, which is not written to any file.

**Stop conditions — explicitly unchanged by anything in this entire
investigation**: 70ms per-tick deadline, `delay_samples=2`, the
consecutive-deadline-miss abort threshold (5, in
`close_loop_path_follow.py`), and the existing hold-on-miss/abort
behavior. Nothing here was touched.

## 5. Three separate outcome categories — do not blend these into one verdict

- **(a) Full-path completion**: did the run reach the plan's own final
  sample, or stop early? If you selected the **original** plan, an early
  stop near the tail may be its OWN, SEPARATE, already-known
  acceleration-margin issue — not anything addressed or caused by this
  investigation. If you selected **marginfix**, an early stop there would
  be a genuinely new, unexpected finding worth investigating fresh.
- **(b) Deadline performance**: miss rate and timing distribution — this
  is what this investigation actually measured and fixed (TCP false-
  infeasibility; BLAS over-threading). A clean run here is real evidence
  the fixes work on hardware, not just offline.
- **(c) Delay identification**: the 5-fold d-vote/G_2 check against
  `delay_samples=2`. This is a completely separate, still-unresolved
  question from everything in this round. Running this procedure, even a
  fully clean run, does **not** itself resolve whether `d=2` is correct —
  that requires its own dedicated analysis (`k2_delay_and_step_analysis.py`),
  and is only even meaningful if you ran the **original** plan (per item 3).

## 6. Commit status — honestly, I cannot check this

I have no git access in this session (confirmed blocked repeatedly,
including just now). I can only confirm the **working-tree file content**,
by direct read, which I just did:
- `delay_aware_mpc.py` line 1253: `self._tcp_ws_min is not None` is
  present in the real `Ax`-refresh gate — the TCP fix is in the live
  working tree.
- `worker_process.py`: `_force_single_threaded_blas()`'s body is the
  `threadpoolctl`-based implementation — the thread fix is in the live
  working tree.

**This confirms only that the fixes are present in the files on disk
right now — it says nothing about whether either has actually been
committed.** Please run this yourself to check:
```bash
git log --oneline -- proper_research/controllers/mpc_delay_aware/delay_aware_mpc.py proper_research/controllers/mpc_delay_aware/worker_process.py
```
