"""Section 8: error vs normalized path progress (ref_index/214), one
line per run, all 21 included runs, condition-colored, retaining the
aborted B run's truncation point explicitly (it simply stops early --
not padded, not interpolated past its last real tick)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jack/Proper-Research"
LOGS = os.path.join(ROOT, "close_loop_logs")
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")
REFERENCE_SAMPLES = 214

manifest = json.load(open(os.path.join(OUT_DIR, "manifest.json")))
colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red", "E": "tab:purple"}

fig, ax = plt.subplots(figsize=(10, 6))
labeled = set()
for m in manifest:
    if not m["include"]:
        continue
    cond = m["condition"]
    pf = [json.loads(l) for l in open(os.path.join(LOGS, m["run_dir"], "path_follow.jsonl"))]
    progress = [min(1.0, (r["ref_index"] + 1) / REFERENCE_SAMPLES) for r in pf]
    err = [r["error_norm_mm"] for r in pf]
    label = cond if cond not in labeled else None
    labeled.add(cond)
    aborted = m["stop_reason"] != "path_complete"
    ax.plot(progress, err, color=colors[cond], alpha=0.5 if not aborted else 0.95,
            lw=2.2 if aborted else 1.0, label=label,
            linestyle="--" if aborted else "-")
    if aborted:
        ax.scatter([progress[-1]], [err[-1]], color=colors[cond], marker="X", s=90, zorder=5)
        ax.annotate("abort\n(tracking_error_exceeded)", (progress[-1], err[-1]),
                     fontsize=7, xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("normalized path progress (ref_index+1)/214")
ax.set_ylabel("tracking error |error_norm_mm| (mm)")
ax.set_title("Tracking error vs normalized path progress, all 21 included runs\n(dashed+X = the one aborted run; stops at its real last tick, not padded)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot5_error_vs_progress.png"), dpi=110)
print("Saved plot5_error_vs_progress.png")
