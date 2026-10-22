# How to run the SJ/FJ x MPC/INV-7 2x2 grid on a new shape plan

This is the generalized version of the raised-rectangle Jacobian-staleness
study (`close_loop_logs/myrun/zshift_analysis/`), packaged so it can be run
on any offline plan (rectangle, U-shape, wide triangle, at any z-shift,
including 0mm/unraised) without editing hardcoded paths.

All commands below assume `cd /home/jack/Proper-Research` and a live robot
+ camera connection already up. **Before any live step, check the vessel
offline replan job isn't hogging the CPU and stealing real-time QP deadlines**
(`ps aux --sort=-%cpu | head`; if a `replan_vessel*.py` process shows up,
`kill -STOP <pid>` before you start, `kill -CONT <pid>` when you're done —
this caused corrupted results earlier in this project and cost a full
re-run to catch).

## 0. What SJ and FJ mean

- **SJ** (Scheduled Jacobian): the genuine, per-reference-index Jacobian,
  one real beam solve per sample of the plan (`jacobian_mode="accurate"`).
  This is what the controller *should* use.
- **FJ** (Frozen Jacobian): SJ's own index-0 value, held constant across
  every sample (`np.repeat(sj[0:1], N, axis=0)`). This deliberately breaks
  the controller's model to see how badly (and how) each controller fails
  when its linearization goes stale as the path progresses.

## 1. Build the offline plan (if you don't already have one)

```bash
python3 -m proper_research.planning.plan_shape_path \
  --shape rectangle --size-mm 10 --rect-depth-mm 15 --skip-global \
  --output-root plans/<name>_2026-XX-XX
```
(`--shape triangle --triangle-apex-at-start --triangle-base-mm <mm>` for a
triangle, `--shape rectangle` with `--closed` omitted for a U/C-shape open
path. `--skip-global` trades the slower global-optimizer refinement layer
for speed — the inverse-configuration layer alone was accurate enough for
every shape tested so far.)

**If you want a z-shifted (raised/lowered) workspace**, apply the shift
BEFORE calling `plan_shape_path.main()` by importing and calling
`zraise_patch.apply(<mm>)` first — this patches both halves that must move
together: the beam-base pose the planner solves against, and (later, at
run time) the vision-reconstruction plane the live harness uses. See
`build_sj_fj_schedules.py`'s own `--zraise-mm` flag for the pattern, or
just write a 10-line wrapper script like this project's earlier
`build_ushape_zraise.py` / `build_triangle_wide_zraise.py` (job-scratch,
not committed — recreate from this pattern if needed):

```python
import sys
if __name__ == "__main__":
    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(30.0)   # mm
    sys.argv = ["plan_shape_path.py", "--shape", "rectangle", "--size-mm", "10",
                "--rect-depth-mm", "15", "--skip-global",
                "--output-root", "plans/<name>_zraise30mm_2026-XX-XX"]
    from proper_research.planning.plan_shape_path import main
    main()
```
Offline plan builds are slow (10-20 min for the inverse-path layer,
one nonlinear solve per centreline node) — run in background
(`nohup timeout 1800 python3 ... &`, or the harness will auto-background
anything over 120s) and check `all_nodes_feasible=True` in the tail of the
log when done.

## 2. Open-loop sanity check (fast gate, do this before anything else)

```bash
python3 -m proper_research.hardware.online.zshift_grid_toolkit.run_open_loop \
  --plan-dir plans/<name>/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun --run-name <name>_openloop \
  --zraise-mm 30 --insertion-tol-mm 0.3
```
Compare the reported RMS to the shape's historical baseline (or to a
similar-size shape if this is the first run on this shape). Only proceed
to closed-loop controllers if it's in the same ballpark — a large
regression here usually means a real bug (frame mismatch, wrong z-shift),
not a controller problem, and every closed-loop run downstream would be
built on it.

## 3. Build the SJ and FJ schedules

```bash
python3 -m proper_research.hardware.online.zshift_grid_toolkit.build_sj_fj_schedules \
  --plan-dir plans/<name>/time_parameterized_configuration_path \
  --sj-out /tmp/<name>_schedule_SJ.npy \
  --fj-out /tmp/<name>_schedule_FJ.npy \
  --zraise-mm 30
```
Takes ~15-20 minutes for a ~200-sample plan (one real beam solve per
sample, offline, no robot needed — safe to run in background while doing
other work). `--zraise-mm` must exactly match what the plan itself was
built with, or the schedule and the plan will disagree about where the
beam base is.

## 4. Run the closed-loop 2x2 grid, live

```bash
kill -STOP <vessel_replan_pid>   # if running -- see the warning at the top

./proper_research/hardware/online/zshift_grid_toolkit/run_2x2_grid.sh \
  plans/<name>/time_parameterized_configuration_path \
  /tmp/<name>_schedule_SJ.npy /tmp/<name>_schedule_FJ.npy \
  close_loop_logs/myrun <name> <n_reps> 30 0.3

kill -CONT <vessel_replan_pid>
```
Arguments: `plan_dir sj_schedule fj_schedule out_dir run_name_prefix n_reps
zraise_mm [insertion_tol_mm=0.3]`. This resets insertion before **every**
rep (insertion drifts across runs) and runs all 4 cells
(`mpc_sj`, `mpc_fj`, `inv7_sj`, `inv7_fj`) x `n_reps` each, in that order.
It does **not** stop on a failed/aborted rep — an FJ arm diverging is an
expected, informative outcome, not a bug — every rep's outcome (path_complete
or the exact stop_reason) is printed in the final summary, and each rep's
full log is kept at `/tmp/grid_<prefix>_<cell>_rep<i>.log`.

Start with `n_reps=1` per cell (or even just the FJ arms first) to sanity-
check before committing to a full multi-rep run — that's what caught the
original rectangle-FJ divergence cheaply rather than after a wasted 5-rep
batch, and it's what the diagnostic protocol in
`close_loop_logs/myrun/zshift_analysis/fj_full_report.py`'s docstring is
built to interrogate if it happens again.

Run directories are named
`<out_dir>/<prefix>_<controller>_<JACOBIAN>_rep<i>_<timestamp>`, e.g.
`close_loop_logs/myrun/ushape_zraise30mm_mpc_FJ_rep1_20260924T160000Z` —
the report script below discovers them by this naming convention, so don't
rename them.

## 5. Plot and review the results

```bash
python3 -m proper_research.hardware.online.zshift_grid_toolkit.plot_grid_report \
  --plan-dir plans/<name>/time_parameterized_configuration_path \
  --sj-schedule /tmp/<name>_schedule_SJ.npy \
  --fj-schedule /tmp/<name>_schedule_FJ.npy \
  --run-dir close_loop_logs/myrun --run-name-prefix <name> \
  --shape-label "<Human-readable shape name>" --zraise-mm 30 \
  --report-dir close_loop_logs/myrun/<name>_analysis \
  --relinearize
```
Produces 4 figures + 3 tables in `--report-dir`:
- **Fig 1**: tracking error vs path progress, every rep individually
  (never averaged), aborts marked with an X — the primary result.
- **Fig 2**: Jacobian staleness `r_J(k)` + weak-axis principal angle,
  aligned above tracking error, to check staleness precedes (not follows)
  divergence. `--relinearize` also relinearizes the real beam model at a
  few representative live-measured states (~10-20s extra, needs the beam
  model but no robot) — omit it for a quick text-only pass using only the
  offline schedule curve.
- **Fig 3**: predicted-vs-actual motion direction (`c_k`, cosine
  similarity) and the "confidently wrong" tick fraction
  (`dV_pred<0 & dV_real>0`) for a representative failing FJ rep vs its SJ
  counterpart.
- **Fig 4**: normalized commanded-correction magnitude, joint-increment
  constraint activity, and effective feedback gain `K_eff(k)` for MPC-FJ
  vs INV-FJ — the "why does MPC diverge while INV-7 just degrades" figure.
- **Table A**: completion rate / aborts / RMS-on-completed (flagged `*` if
  partial, since aborted runs truncate and understate the real RMS) /
  common-path RMS / max error / joint-saturation fraction, per arm.
- **Table A2**: common-path RMS broken into 0-33/33-66/66-100% progress
  regimes, to show whether an FJ/SJ gap is uniform or concentrated late.
- **Table B**: per-replicate MPC-FJ breakdown (n is usually small — don't
  hide individual outcomes behind an aggregate).
- **Table C**: mechanism table (solver success/deadline-misses/delay-
  consistency, to rule out a software/execution explanation; `r_J`,
  weak-axis angle, tail `c_k`, wrong-descent %, first saturation tick).

The representative "failing" rep picked for Figs 2-4 is the first MPC-FJ
rep that didn't reach `path_complete`, if any; otherwise its rep1.

## Reference: the original raised-rectangle run

The first full run of this study (before this toolkit existed, hand-built
with per-shape hardcoded scripts) is at
`close_loop_logs/myrun/zshift_analysis/` — `fj_full_report.py` there is the
rectangle-specific, hardcoded ancestor of `plot_grid_report.py` and is a
useful worked example of the exact numbers/figures to expect. Headline
result on the raised (+30mm) rectangle: MPC-FJ completed 1/4 reps (3
safety-aborted), INV-FJ completed 4/4 (degraded ~2.6x RMS but never
diverged) — same stale Jacobian, very different failure mode by
controller architecture.
