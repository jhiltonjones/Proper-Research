"""Reusable offline plotting/scoring script for a rectangle_stage_a pilot
run's two log files (predicted_beam_positions.jsonl + path_follow.jsonl),
joined on the shared "step" tick ID -- aligned by construction between
the two files since the 2026-11-20 fix in close_loop_path_follow.py (see
that file's own comment at the `steps += 1` relocation). No hardware;
pure offline replay of already-saved files.

Usage:
    python3 -m proper_research.analysis.pilot_logging_check.plot_pilot_run \\
        <run_dir containing predicted_beam_positions.jsonl + path_follow.jsonl> \\
        [output_dir, default: same as run_dir]

2026-11-20 (formatting + bug-fix pass, after the first real baseline-A
pilot run): two real defects found running this against real 235-tick
hardware data (not the 8-tick synthetic completion-check trial this
script was originally written against):
  1. Field-name bug: score_forecasts() read "tip_meas_m" from path_follow
     .jsonl, but the real field is "tip_mm" (millimetres, not metres).
     ".get() or [nan]*3" silently masked the KeyError-equivalent miss,
     so EVERY measured tip came back NaN and the forecast-scoring panel
     reported 0 matched samples against 235 successful forecasts --
     not a real data problem, a field-name mismatch. Fixed: reads
     tip_mm and converts to metres.
  2. Layout: the two forecast panels (B: fan plot, C: lead-time curve)
     were drawn into an entirely SEPARATE second figure/file, leaving
     two subplot axes in the main 3x2 figure permanently empty, and
     every panel title was hardcoded for the specific 8-tick synthetic
     demo run ("this synthetic trial", "8-tick completion check"),
     which is actively misleading on a real run. Fixed: all six panels
     (A-F) now live in ONE figure; captions report the real tick/
     success/exclusion counts for whatever run is passed in, with no
     hardcoded run-specific numbers; title-vs-axis overlap fixed via
     constrained_layout plus shorter per-panel titles (the fuller
     caveat text that used to sit in the title now goes in each
     panel's own footnote-sized subtitle, positioned below the axes
     where it cannot collide with the row above).

Field dictionary (the keys this script reads; see each producer's own
inline comments for full provenance):
  predicted_beam_positions.jsonl:
    step                      -- shared tick ID (see module docstring above)
    worker_request_id         -- worker's request seq if a response arrived in time; None otherwise
    success                   -- bool, whether this tick's solve completed in time
    predicted_beam_positions_m -- full-horizon forecast (N x 3), robot frame, metres
    predicted_states          -- full-horizon delayed physical-state model (N x 7)
    predicted_commands        -- full-horizon commanded-state model (N x 7)
    predicted_inputs          -- full-horizon planned inputs (N x 7)
    jacobian_scale_applied    -- g_used this tick (float) or None if unavailable
    magnet_exclusion_gap_mm   -- measured magnet-to-lumen clearance, mm
  path_follow.jsonl:
    step                              -- shared tick ID (same join key)
    t_s                               -- elapsed time, seconds, CLOCK_MONOTONIC-relative
    ref_index                         -- reference/progress index this tick used
    tip_mm                            -- measured tip position, robot frame, MILLIMETRES (not tip_meas_m)
    error_mm                          -- [dx,dy,dz] tip error, mm
    insertion_length_m                -- accepted (post-both-limiters) insertion command, metres
    requested_insertion_rate_m_s      -- pre-limiter insertion rate request
    final_insertion_rate_m_s          -- post-envelope-clip AND post-advancer-rate-cap rate
    solver_deadline_missed            -- bool
    adapter_abort_reason              -- str or None
    external_measured_clearance_gap_m -- measured clearance (same shared-check field as INV uses)
    clearance_sample_age_s            -- age (s) of the joint sample the clearance check used; None if unavailable
    jacobian_adaptation_diagnostics   -- dict (mode, g_hat, g_used, accepted, reason, ...) or None
    worker_request_id / worker_still_pending_request_id -- request identity, see process_isolated_adapter.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/jack/.claude/jobs/043c0b8b/tmp/completion_check_run"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else RUN_DIR
PRED_PATH = f"{RUN_DIR}/predicted_beam_positions.jsonl"
PF_PATH = f"{RUN_DIR}/path_follow.jsonl"
OUT_PNG = f"{OUT_DIR}/pilot_run_panels.png"

pred_rows = [json.loads(l) for l in open(PRED_PATH)]
pf_rows = [json.loads(l) for l in open(PF_PATH)]
pred_by_step = {r["step"]: r for r in pred_rows}
pf_by_step = {r["step"]: r for r in pf_rows}
steps = sorted(set(pred_by_step) & set(pf_by_step))
assert steps == sorted(pred_by_step) == sorted(pf_by_step), "step IDs must be identical sets across both files -- shared tick ID join failed"

n_ticks = len(steps)
n_success = sum(1 for s in steps if pred_by_step[s].get("success"))
t_all = [pf_by_step[s]["t_s"] for s in steps]
dt = float(np.median(np.diff(t_all))) if len(t_all) > 1 else 0.1

fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
fig.suptitle(
    f"{RUN_DIR.rstrip('/').rsplit('/', 1)[-1]}  --  {n_ticks} ticks, {n_success} successful solves, dt~{dt:.3f}s",
    fontsize=10,
)

t = t_all

# Panel A: tip error over time.
err_mm = [pf_by_step[s]["error_mm"] for s in steps]
axes[0, 0].plot(t, err_mm, "o-", ms=3)
axes[0, 0].set_title("A: tracking error over time", fontsize=10)
axes[0, 0].set_xlabel("t (s)"); axes[0, 0].set_ylabel("tip error (mm)")
axes[0, 0].text(
    0.0, -0.22, "components [dx,dy,dz], planar vs camera-Z not separated here",
    transform=axes[0, 0].transAxes, fontsize=6.5, va="top",
)

# Panel D: g_hat/g_used over time with accept/reject.
g_used = []
for s in steps:
    diag = pred_by_step[s].get("jacobian_scale_applied")
    g_used.append(diag if diag is not None else np.nan)
axes[0, 1].plot(t, g_used, "s-", ms=3, label="g_used (jacobian_scale_applied)")
axes[0, 1].axhline(-0.3, color="r", ls="--", lw=0.8, label="g_min/g_max bounds")
axes[0, 1].axhline(0.5, color="r", ls="--", lw=0.8)
axes[0, 1].set_title("D: adaptation gain over time", fontsize=10)
axes[0, 1].set_xlabel("t (s)"); axes[0, 1].set_ylabel("g")
axes[0, 1].legend(fontsize=6.5, loc="best")
axes[0, 1].text(
    0.0, -0.22, "None plotted as a gap -- unavailable on a miss/abort tick",
    transform=axes[0, 1].transAxes, fontsize=6.5, va="top",
)

# Panel E: insertion reference/accepted command, requested vs final rate.
ins_m = [pf_by_step[s]["insertion_length_m"] for s in steps]
req_rate = [pf_by_step[s]["requested_insertion_rate_m_s"] for s in steps]
final_rate = [pf_by_step[s]["final_insertion_rate_m_s"] for s in steps]
ax = axes[1, 0]
ax.plot(t, [v * 1e3 for v in ins_m], "o-", ms=3, color="tab:blue", label="accepted insertion (mm)")
ax.axhline(20, color="grey", ls=":", lw=0.8)
ax.axhline(70, color="grey", ls=":", lw=0.8)
ax.set_ylabel("insertion (mm)"); ax.set_xlabel("t (s)")
ax2 = ax.twinx()
ax2.plot(t, req_rate, "x--", ms=3, color="tab:orange", label="requested rate (m/s)", alpha=0.6)
ax2.plot(t, final_rate, "+-", ms=3, color="tab:green", label="final/accepted rate (m/s)")
ax2.set_ylabel("insertion rate (m/s)")
ax.set_title("E: insertion command + requested vs final rate", fontsize=10)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="best")
ax.text(
    0.0, -0.22, "dotted lines: 20/70mm commanded envelope",
    transform=ax.transAxes, fontsize=6.5, va="top",
)

# Panel F: clearance, deadline/abort events, aligned on the shared step ID.
miss = [1 if pf_by_step[s]["solver_deadline_missed"] else 0 for s in steps]
abort = [1 if pf_by_step[s]["adapter_abort_reason"] else 0 for s in steps]
ax = axes[1, 1]
ax.step(t, miss, where="mid", label="deadline miss (0/1)")
ax.step(t, abort, where="mid", label="abort (0/1)")
ax.set_title("F: deadline misses / abort events", fontsize=10)
ax.set_xlabel("t (s)"); ax.set_ylabel("event")
ax.set_ylim(-0.1, 1.3)
ax.legend(fontsize=6.5, loc="best")

# Panels B/C: forecast fan + lead-time error curve -- SCORING LOGIC.
def score_forecasts(pred_rows, pf_rows, dt, delay_samples):
    """For each successful forecast at tick k, derive each horizon
    stage j's target time as t_k + j*dt (the stage-to-time convention:
    stage j's PHYSICAL/delayed prediction targets the measurement that
    will exist at that future time -- see predicted_states' own Ep/Sp
    docstring in delay_aware_mpc.py for the delay-aware indexing this
    assumes). Matches against the actual measured tip at the NEAREST
    later tick within max_gap_s; excludes stages with no match within
    that gap, reporting the exclusion count.
    """
    max_gap_s = 0.5 * dt
    pf_by_step = {r["step"]: r for r in pf_rows}
    t_by_step = {r["step"]: r["t_s"] for r in pf_rows}
    tip_by_step = {}
    for r in pf_rows:
        tip_mm = pf_by_step[r["step"]].get("tip_mm")
        tip_by_step[r["step"]] = (
            np.asarray(tip_mm, dtype=float) / 1e3 if tip_mm is not None else np.array([np.nan] * 3)
        )
    results = []  # (lead_stage_j, abs_error_m) per matched sample
    excluded = 0
    for r in pred_rows:
        if not r.get("success") or r.get("predicted_beam_positions_m") is None:
            continue
        k = r["step"]
        t_k = t_by_step.get(k)
        if t_k is None:
            continue
        stages = np.asarray(r["predicted_beam_positions_m"], dtype=float)
        for j in range(stages.shape[0]):
            target_t = t_k + j * dt
            candidates = [(abs(t_by_step[s] - target_t), s) for s in t_by_step if t_by_step[s] >= t_k]
            if not candidates:
                excluded += 1
                continue
            gap, best_s = min(candidates)
            if gap > max_gap_s:
                excluded += 1
                continue
            meas = tip_by_step[best_s]
            if np.any(np.isnan(meas)):
                excluded += 1
                continue
            err = float(np.linalg.norm(stages[j] - meas))
            results.append((j, err))
    return results, excluded


results, excluded = score_forecasts(pred_rows, pf_rows, dt=dt, delay_samples=2)
by_lead = {}
for j, err in results:
    by_lead.setdefault(j, []).append(err)
leads = sorted(by_lead)
means = [np.mean(by_lead[j]) * 1e3 for j in leads]
counts = [len(by_lead[j]) for j in leads]

# Panel B: a sample of raw forecast "fans" -- a handful of successful
# ticks' full-horizon predicted tip path overlaid on the measured tip
# path near that tick, so a qualitative look is available even when the
# lead-time curve in C is sparse.
ax = axes[2, 0]
tip_meas_xy = np.array([
    (np.asarray(pf_by_step[s]["tip_mm"], dtype=float)[:2] if pf_by_step[s].get("tip_mm") else [np.nan, np.nan])
    for s in steps
])
ax.plot(tip_meas_xy[:, 0], tip_meas_xy[:, 1], "-", color="k", lw=1.0, alpha=0.6, label="measured tip (xy)")
sample_ticks = [r for r in pred_rows if r.get("success") and r.get("predicted_beam_positions_m") is not None]
fan_sample = sample_ticks[:: max(1, len(sample_ticks) // 8)][:8] if sample_ticks else []
for r in fan_sample:
    stages = np.asarray(r["predicted_beam_positions_m"], dtype=float) * 1e3
    ax.plot(stages[:, 0], stages[:, 1], "--", lw=0.8, alpha=0.7)
ax.set_title("B: forecast fans vs measured tip path (xy, mm)", fontsize=10)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.legend(fontsize=6.5, loc="best")
ax.text(
    0.0, -0.18, f"{len(fan_sample)} of {len(sample_ticks)} successful forecasts shown (dashed)",
    transform=ax.transAxes, fontsize=6.5, va="top",
)

# Panel C: forecast error by lead time.
ax = axes[2, 1]
ax.plot(leads, means, "o-")
annot_stride = max(1, len(leads) // 5)  # thin labels so they don't overlap on a long horizon
for idx, (j, m, c) in enumerate(zip(leads, means, counts)):
    if idx % annot_stride == 0 or idx == len(leads) - 1:
        ax.annotate(f"n={c}", (j, m), fontsize=6, textcoords="offset points", xytext=(0, 6))
ax.set_xlabel("lead stage j (target time = t_k + j*dt)")
ax.set_ylabel("mean |forecast - measured| (mm)")
ax.set_title("C: forecast error by lead time", fontsize=10)
ax.text(
    0.0, -0.22,
    f"N={len(results)} matched / {excluded} excluded (no measurement within {max(0.5 * dt, 1e-6):.3f}s). "
    f"Reoptimization between ticks means later-executed inputs can differ from what an earlier "
    f"forecast assumed -- a discrepancy here is not attributable to beam-model error alone.",
    transform=ax.transAxes, fontsize=6, va="top", wrap=True,
)

fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
print(f"Forecast scoring: {len(results)} matched samples across {len(leads)} lead stages, {excluded} excluded")
