# MPC parameter studies: what they answer and how to run them

This directory (`proper_research/studies/mpc_ablations/`) consolidates every
offline study behind this project's MPC controller design — the nullspace
formulation, the R/gamma/disturbance/delay parameter sweeps, and the closed
manipulability-ablation investigation — into one place you can run yourself.
Nothing here touches hardware.

Read `proper_research/controllers/mpc_delay_aware/CONTROLLER_JUSTIFICATION.md`
first if you haven't — it's the narrative document explaining *why* every
frozen parameter was chosen, section by section, with a summary table (§13)
mapping each parameter to its evidence. This directory is the "run it
yourself and see the plots" companion to that document; `SUPPLEMENTARY_EVIDENCE.md`
in the same folder as the justification doc is its numeric appendix.

## The one thing to understand before running anything: replay methodology

Every script here uses **single-tick counterfactual replay**: at each tick of
a REAL logged closed-loop run, it re-solves one or more controller variants
against the ACTUAL recorded measured state / previous command / previous
input, and compares their decisions. Because each tick is re-solved
independently against the real recorded state (not each variant's own
rollout), trajectories never diverge or compound — this isolates each
formulation's *one-step decision* cleanly. It is an open-loop prediction
comparison, not a new closed-loop simulation; CONTROLLER_JUSTIFICATION.md
§14 is explicit about that distinction and why it still matters.

The default replay source is a real, complete (274-tick) closed-loop
rectangle run:
`close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z`
Every script accepts `--log-dir` to replay against a different real run
instead (any `close_loop_logs/myrun/...` directory with a `path_follow.jsonl`
and `controller_metadata.json`, same `plan_dir`).

## Setup (once)

The first script you run will build and cache a genuine Jacobian schedule
for the rectangle plan (one nonlinear beam solve per reference sample, ~90–
150s); every later script reuses the same cache
(`/tmp/rectangle_study_schedule.npy` by default — override with
`--schedule-cache`). Run study 1 first so the wait happens once, up front:

```bash
cd /home/jack/Proper-Research
python -m proper_research.studies.mpc_ablations.study01_nullspace_qn_ablation
```

## The studies

All scripts save a PNG plot under `outputs/<study_name>/` (created
automatically) and print a numeric summary to the terminal.

### study01 — the nullspace (Q_N) ablation

**Question:** does the reference-nullspace state-tracking cost `Q_N` do
anything beyond what the full velocity reference `u_ref` (which already
includes its own redundant/nullspace motion) provides on its own?

**Conditions:** A = `StagewiseTaskNullspaceDelayAwareMPC(gamma=0)`, the
frozen live controller (`Q_N` on). B = `qn_ablation.QZeroMPC`, identical
except the entire state cost is zeroed.

**What to look at:** task tracking (`E_chosen` at +3/+5/+10) barely moves
between A and B — but the nullspace-projected predicted deviation `e_N`
grows substantially under B, especially at longer horizon depths. This is
exactly the mechanism relevant to the vessel-navigation live drift
investigated this session (see `paper/model_necessity_study.tex` and this
session's own vessel Q_N=0 findings): a velocity-tracking cost alone does
not anchor accumulated configuration drift.

```bash
python -m proper_research.studies.mpc_ablations.study01_nullspace_qn_ablation
```

### study02 — the gamma sweep

**Question:** what happens as the state cost moves from isotropic
(`gamma=1`, "M0") to exact task-nullspace-only (`gamma=0`, frozen)?

`StagewiseTaskNullspaceDelayAwareMPC` already takes `gamma` as a constructor
argument — this sweep is that one class at several gamma values, no separate
controller needed. `gamma=1` reproduces the isotropic Q exactly (regression
guard: CONTROLLER_JUSTIFICATION.md §11).

**What to look at:** task tracking (`E_chosen`) vs gamma, and actuator
saturation (fraction of ticks near a rate limit) vs gamma — the original
sweep's goal was a Pareto improvement in tracking vs saturation, not minimum
tracking error alone.

```bash
python -m proper_research.studies.mpc_ablations.study02_gamma_sweep \
    --gammas 0.0,0.10,0.25,1.0
```

### study03 — the R (input-tracking weight) sweep

**Question:** at `gamma=0`, how does the input-tracking weight `R = mult *
R0` trade off tracking accuracy against actuator saturation, and where does
`R700` (the frozen value) sit?

**What to look at:** `E_chosen` vs R (log scale), and saturation split by
*channel* (joints vs insertion) — the original finding was that "constraint
activity" was almost entirely an insertion-channel phenomenon (joints
`q2`–`q6` saturated in 0.000 of ticks in every condition tested), and that
`R700` and `R1000` give essentially identical joint-channel activity (R700
isn't unnecessarily conservative).

```bash
python -m proper_research.studies.mpc_ablations.study03_r_sweep \
    --r-multipliers 1,10,100,300,500,700,1000
```

### study04 — disturbance persistence (beta_d) sensitivity

**Question:** how much does `beta_d` (how much of the estimated disturbance
persists across the whole prediction horizon) actually change commands and
predictions? `beta_d=1` is frozen; this is a sensitivity check, not a
superiority claim (CONTROLLER_JUSTIFICATION.md §3 is explicit that the
`beta_d=0.82` alternative was never shown superior, only that the choice is
non-trivial).

**What to look at:** command RMS shift (%) and predicted-beam RMS shift (mm),
both relative to `beta_d=1.0`, across a small sweep.

```bash
python -m proper_research.studies.mpc_ablations.study04_disturbance_beta_sweep \
    --beta-ds 0.5,0.82,1.0
```

### study05 — delay model (d) sweep

**Question:** how much does the joint command-to-effect delay model
(`delay_samples`, aka `d`) matter for prediction accuracy?

**Important caveat:** the original evidence (CONTROLLER_JUSTIFICATION.md §2)
was a proper 5-fold leave-one-out cross-validation across several logs
(`d=2` beating `d=0` by 6–11x). This script replays `d in {0,1,2}` against
ONE real log and compares `E_chosen` — good for seeing the qualitative
effect and as a template for your own multi-log LOO study, **not** a
reproduction of the original statistical claim. If you want the full LOO
result, run this script once per available real log (see `close_loop_logs/`)
and aggregate the per-log `E_chosen` numbers yourself.

```bash
python -m proper_research.studies.mpc_ablations.study05_delay_model_sweep \
    --delay-samples 0,1,2
```

### manipulability_ablation/ — the closed authority-shaping investigation

A complete, self-contained, already-finished research branch (closed
2026-09-22 as a validated **negative result** — the live controller was
never changed). Consolidated here unmodified except for two things: the
hardcoded `sys.path.insert(0, ".../jobs/3710eca5/tmp")` lines were removed
(the sibling module `authority_objective.py` is already importable when you
run these scripts from inside `manipulability_ablation/scripts/`), and every
hardcoded pickle path was redirected from that same scratch directory to the
local `manipulability_ablation/data/` folder (already populated — you don't
need to re-run the expensive stages to read the results).

**Read `manipulability_ablation/README.md` first** — it's the full narrative:
what was tried (a projected log-volume "authority" secondary objective,
proposed as a drop-in replacement for `Q_N`), the Phase-0 decision gate, the
conflict diagnostic against real live logs, the weight-calibration pilot, the
decisive A/B/C replay (spec: 175 ticks × 6 modes = 1050 solves), the
root-cause directional-derivative decomposition, and the gradient-staleness
follow-up that ruled out one hypothesis for the failure mode. One-line
summary: the authority metric had real exploitable nullspace gradients, but
its linear surrogate mispredicted the sign of the true authority change in
40–43% of ticks — a curvature/step-size failure, not staleness — so the
existing `Q_N` reference-nullspace anchor was retained.

**To re-read the existing results** (fast, uses the saved `data/*.pkl`):
```bash
cd proper_research/studies/mpc_ablations/manipulability_ablation/scripts
python authority_abc_analysis.py
```

**To rerun the full pipeline from scratch** (slow — Phase 0 alone is
~15min; run in this order, each stage's output feeds the next):
```bash
cd proper_research/studies/mpc_ablations/manipulability_ablation/scripts
python authority_unit_tests_math.py         # regression tests C, D (pure math, fast)
python authority_phase0.py                  # ~15min: caches phi/sigma/g/P_N/g_N per reference index -> ../data/authority_phase0_schedule.pkl
python authority_regression_tests.py        # regression tests A, B, H, I, J, K, L -- run before trusting anything below
python authority_conflict_diagnostic.py     # Q_N vs authority-ascent direction, on real live logs -> ../data/authority_conflict_diagnostic.pkl
python authority_weight_pilot.py            # lambda calibration from solution sensitivity -> ../data/authority_weight_pilot.pkl
python authority_abc_replay.py              # THE decisive result: 1050 solves -> ../data/authority_abc_replay.pkl
python authority_abc_analysis.py            # summarizes the replay -> ../data/authority_abc_summary.pkl
python authority_directional_decomp.py      # root-cause: f_manip . d decomposition at 2 representative ticks
python authority_gradient_staleness.py      # tests (and rejects) the staleness hypothesis -> ../data/authority_gradient_staleness.pkl
```
These must be run with the shell's working directory set to
`manipulability_ablation/scripts/` (the pickle paths are relative to that,
matching how the original investigation was structured).

## The controller class map (what's importable from where)

All of the above import controller *classes* from
`proper_research/controllers/mpc_delay_aware/` rather than duplicating
formulations. For your own experiments:

| class | file | what it is |
|---|---|---|
| `StagewiseTaskNullspaceDelayAwareMPC` | `stagewise_task_nullspace.py` | **the frozen, live-validated controller.** `gamma` is a constructor arg — this is what studies 2–5 sweep. |
| `build_stagewise_projectors` | `stagewise_task_nullspace.py` | builds the exact orthogonal `P_R`/`P_N` projectors from the numerical-rank SVD of the scaled task Jacobian at each horizon stage. |
| `ExactQNZeroTaskNullspaceDelayAwareMPC` | `exact_qn_zero_mpc.py` | forces `P_N=P_R=0` *exactly* (not `gamma=0`, which zeros the row-space weight but still computes/uses `P_N`) — built this session for the vessel study. |
| `QZeroMPC` | `qn_ablation.py` | study01's condition B (entire state cost `Q=0`). |
| `JointOnlyFeedbackMPC`, `AuthorityAwareTaskCostMPC` | `insertion_ablations.py` | the two insertion-vs-joint-authority diagnostics (§9) — not wired into a study script here yet; same replay pattern applies if you want one. |
| `TaskNullspaceDelayAwareMPC` | `task_nullspace_mpc.py` | the original (non-stagewise) gamma formulation — superseded by the stagewise version above, kept for reference. |
| `AuthorityShapedDelayAwareMPC` | `authority_nullspace_mpc.py` | the manipulability-ablation controller (modes A/B/C) — see that package above. |
| target-consistent "MT" family | `target_consistent*.py` (4 files) | the rejected target-shift formulation (§6.1) — no study script here; the verdict was clear enough (worse constraint saturation for no clean gain) that it wasn't worth reconstructing a sweep for. |

## Honest gaps

- The *original* historical gamma-sweep / R-sweep / MT-rejection driver
  scripts (that produced the exact numbers narrated in
  CONTROLLER_JUSTIFICATION.md §6.1–6.3) were interactive/scratch work from
  an earlier session and were never committed as standalone files. Studies
  02–03 here are a clean **reconstruction** using the same preserved
  controller classes and the same replay methodology — they'll reproduce
  the qualitative shape of the original findings, but the exact numbers
  will differ from the document (different replay log, different tick
  sample, and the original sweeps used more conditions / more ticks).
- Study05's delay-model comparison is a simplified single-log version of a
  claim that was originally established via 5-fold cross-validation — see
  its own docstring/section above.
- `insertion_ablations.py`'s two classes don't have a dedicated study script
  yet (they're straightforward to add with the same `replay_common.py`
  pattern — ask if you want it).
