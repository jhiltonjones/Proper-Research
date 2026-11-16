"""Reproducible offline analysis of a K2 path_follow.jsonl log: 5-fold
delay identification (reusing delay_identification_5fold.py's exact
fold/RMSE/G_d logic) plus per-joint step-realization ratios.

2026-11-15 (second correction): the step-realization formula below is
implemented EXACTLY as originally documented in HOWTO_CAMPAIGN2.md
(section E3/"step-realization check", lines ~368-382 as of this
writing) -- same-tick pairing (realized[k] vs q_target_delta_rad[k],
same index k, no shift), denominator clamped with
np.maximum(np.abs(commanded), 1e-12) (not a tunable epsilon mask), over
ALL rows (n = min(len(q_meas)-1, len(q_delta)), no terminal-hold
exclusion). This is the project's own prior documented methodology,
confirmed by direct citation, not reverse-engineered by searching for
whatever convention best matches old reported numbers -- an earlier
pass of this script did exactly that (a grid search over tick-shift/
epsilon/row-inclusion to minimize error against historical medians) and
was explicitly corrected: that method is wrong and has been removed.
Where this exact, documented formula's output does not match an
earlier informally-reported number, THIS output is authoritative and
the earlier number is explicitly non-reproducible -- see the module's
own run output for which, if any, of the historical numbers that
applies to.

Usage
-----
    python3 k2_delay_and_step_analysis.py --log-dir <run_dir> [--delays 0,1,2,3]
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

N_FOLDS = 5
STEP_DENOMINATOR_FLOOR = 1e-12  # exact value from HOWTO_CAMPAIGN2.md's own documented formula;
                                 # NOT a tunable parameter -- this is the specified clamp, not a mask.
LOG_FILES_TO_HASH = [
    "path_follow.jsonl", "summary.json", "controller_metadata.json",
    "startup_stationary_tips.jsonl", "predicted_beam_positions.jsonl",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_rows(log_dir: Path):
    path = log_dir / "path_follow.jsonl"
    return [json.loads(l) for l in open(path)]


def load_arrays(rows):
    q_meas = np.array([r["q_meas_rad"] for r in rows])
    q_delta = np.array([r["q_target_delta_rad"] for r in rows])
    return q_meas, q_delta


def realized_step(q_meas):
    return np.diff(q_meas, axis=0)  # realized_step[k] = q_meas[k+1] - q_meas[k]


# ---------------------------------------------------------------------------
# 5-fold delay identification -- same fold/RMSE/G_d definitions as
# delay_identification_5fold.py (kept identical deliberately, so this
# script's numbers are directly comparable to that one's "official" output).
# ---------------------------------------------------------------------------
def rmse_at_delay(realized, commanded_delta, d, idx):
    valid = idx[idx - d >= 0]
    if len(valid) == 0:
        return np.nan, 0
    pred = commanded_delta[valid - d]
    actual = realized[valid]
    return float(np.sqrt(np.mean((actual - pred) ** 2))), len(valid)


def run_five_fold(q_meas, q_delta, delays):
    realized = realized_step(q_meas)
    T = len(realized)
    bounds = [int(round(i * T / N_FOLDS)) for i in range(N_FOLDS + 1)]
    fold_idx = [np.arange(bounds[i], bounds[i + 1]) for i in range(N_FOLDS)]

    excitation_rms = []
    g_by_fold = {d: [] for d in delays if d != 0}
    best_d_per_fold = []
    for f in range(N_FOLDS):
        held_out = fold_idx[f]
        train = np.concatenate([fold_idx[i] for i in range(N_FOLDS) if i != f])
        excitation_rms.append(float(np.sqrt(np.mean(q_delta[held_out] ** 2))))
        heldout_rmse = {}
        for d in delays:
            _, _ = rmse_at_delay(realized, q_delta, d, train)
            ho_rmse, n_ho = rmse_at_delay(realized, q_delta, d, held_out)
            heldout_rmse[d] = ho_rmse
        base = heldout_rmse.get(0)
        for d in delays:
            if d == 0 or base is None or base == 0:
                continue
            g = 1.0 - (heldout_rmse[d] / base) ** 2
            g_by_fold[d].append(g)
        best_d = min(delays, key=lambda d: heldout_rmse[d] if not np.isnan(heldout_rmse[d]) else np.inf)
        best_d_per_fold.append(best_d)

    return {
        "best_d_per_fold": best_d_per_fold,
        "vote_counts": dict(Counter(best_d_per_fold)),
        "g_by_fold": g_by_fold,
        "excitation_rms_per_fold": excitation_rms,
    }


# ---------------------------------------------------------------------------
# Step-realization ratio -- EXACT formula from HOWTO_CAMPAIGN2.md, no
# tunable parameters, no row exclusion, no alternate pairing option.
# ---------------------------------------------------------------------------
def step_realization(q_meas, q_delta):
    n = min(len(q_meas) - 1, len(q_delta))
    realized = np.diff(q_meas, axis=0)[:n]
    commanded = q_delta[:n]
    ratio = np.abs(realized) / np.maximum(np.abs(commanded), STEP_DENOMINATOR_FLOOR)
    return {
        "medians": np.median(ratio, axis=0).tolist(),
        "means": np.mean(ratio, axis=0).tolist(),
        "n_ticks": int(n),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--delays", default="0,1,2,3")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    delays = [int(d) for d in args.delays.split(",")]

    print("=" * 70)
    print(f"INPUT PINNING -- SHA-256 of every file read, from {log_dir}")
    print("=" * 70)
    for fname in LOG_FILES_TO_HASH:
        fpath = log_dir / fname
        if fpath.exists():
            print(f"  {fname}: {sha256_of(fpath)}")
        else:
            print(f"  {fname}: MISSING")

    rows = load_rows(log_dir)
    q_meas, q_delta = load_arrays(rows)
    print(f"\nT (ticks loaded) = {len(rows)}  (ALL rows, no terminal-hold exclusion)")

    print("\n" + "=" * 70)
    print("5-FOLD DELAY IDENTIFICATION (official definitions, unchanged)")
    print("=" * 70)
    ff = run_five_fold(q_meas, q_delta, delays)
    print("best d per fold:", ff["best_d_per_fold"])
    print("vote counts:", ff["vote_counts"])
    if 2 in ff["g_by_fold"]:
        print("G_2 per fold:", [round(g, 4) for g in ff["g_by_fold"][2]])
    print("commanded-delta RMS excitation per held-out fold:",
          [round(x, 6) for x in ff["excitation_rms_per_fold"]])
    g2_ok = 2 in ff["g_by_fold"] and all(g > 0.9 for g in ff["g_by_fold"][2])
    d2_majority = ff["vote_counts"].get(2, 0) >= 4
    print("RESULT:", "d=2 CONFIRMED" if (g2_ok and d2_majority) else
          "STOP -- d=2 NOT confirmed by this 5-fold check")

    print("\n" + "=" * 70)
    print("STEP-REALIZATION (exact HOWTO_CAMPAIGN2.md formula: same-tick pairing, "
          f"denominator floor {STEP_DENOMINATOR_FLOOR:.0e}, all rows)")
    print("=" * 70)
    sr = step_realization(q_meas, q_delta)
    print(f"n_ticks used: {sr['n_ticks']}")
    print("medians:", [round(x, 3) for x in sr["medians"]])
    print("means:  ", [round(x, 3) for x in sr["means"]])
    print("joints passing >=0.95 (existing criterion, NOT altered here):",
          [j + 1 for j, m in enumerate(sr["medians"]) if m >= 0.95])


if __name__ == "__main__":
    main()
