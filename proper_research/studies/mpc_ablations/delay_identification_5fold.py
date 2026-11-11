"""Five-fold leave-one-out joint-command delay identification, runnable on
a fresh K2 (or any) path_follow.jsonl log. Designed 2026-11-11 because the
original fitting script and its .pkl inputs no longer exist in this
checkout (see HOWTO_CAMPAIGN2.md for that finding) -- this is a new,
complete, from-scratch procedure, not the simplified single-log sweep.

Inputs
------
One run's path_follow.jsonl, with at least ~150 ticks (5 folds x >=30
ticks/fold is the practical minimum for a stable RMSE estimate; the K2
N=15 timing-check run, Section E3, is a natural source -- it already needs
collecting for the period/step-realization checks).

Fields used (confirmed real, from close_loop_path_follow.py's own row
dict, not assumed): `q_target_delta_rad` (this tick's own commanded joint
delta, 6-vector) and `q_meas_rad` (measured joints, 6-vector). The
REALIZED joint step from tick k to k+1 is q_meas_rad[k+1] - q_meas_rad[k];
under the "measure current state, then compute and apply this tick's
command" loop order, that realized step is the one attributable to tick
k's own q_target_delta_rad[k] (same index, not k+1 -- see HOWTO for the
pairing justification).

What a "fold" is here
----------------------
The tick sequence (length T) is split into 5 contiguous, equal-length
blocks (fold boundaries at T//5, 2T//5, ... -- contiguous blocks, not a
random shuffle, because delay dynamics are a TIME SERIES: shuffling ticks
would let a held-out tick's neighbours leak into training and bias d
toward whatever fits noise, not the true actuation lag). For each of the
5 folds, that fold is held out; the other 4 folds' ticks are pooled as
the "train" set.

What's being fit/compared
--------------------------
For each candidate delay d in {0, 1, 2, 3} and each fold: predict the
REALIZED step at tick k as the COMMANDED delta from d ticks earlier,
q_target_delta_rad[k-d], and compute RMSE between that prediction and the
actual realized step, separately on the train ticks and the held-out
ticks. (d=0 is the no-delay baseline, matching the original G formula's
m0.) There is nothing to numerically "fit" beyond choosing d -- d is a
single integer, so "fitting per fold" here means: look at the TRAIN RMSE
for d in {0,1,2,3} and note which is smallest (this should already point
at the same d the whole log suggests); the HELD-OUT RMSE at each d is the
generalization check the 5-fold structure actually buys you.

G_d = 1 - (RMSE_heldout_d / RMSE_heldout_0)^2, per fold -- same formula as
SUPPLEMENTARY_EVIDENCE.md S10, so results are comparable to the historical
(now-unreproducible) number if the author wants that comparison, while
not depending on anything from that original run.

Stop rule
---------
- CONFIRMED (d=2): d=2 gives the lowest held-out RMSE on at least 4 of
  the 5 folds, and G_2 > 0.9 on every fold (mirroring the historical
  0.973-0.992 range as the bar for "explains almost all the delay-induced
  residual").
- STOP, investigate before proceeding: the best d disagrees across folds
  (e.g. 2 of 5 folds prefer d=1 or d=3), or G_2 < 0.9 on any fold, or the
  train-set best d doesn't match d=2 at all. Per the HOWTO: a result other
  than d=2 is a stop condition for K2/Day-1/the core campaign.

Usage
-----
    python3 delay_identification_5fold.py --log path_follow.jsonl
"""
import argparse
import json
import numpy as np


def load_log(path):
    rows = [json.loads(l) for l in open(path)]
    q_meas = np.array([r["q_meas_rad"] for r in rows if "q_meas_rad" in r])
    q_delta = np.array([r["q_target_delta_rad"] for r in rows if "q_target_delta_rad" in r])
    n = min(len(q_meas), len(q_delta))
    return q_meas[:n], q_delta[:n]


def realized_step(q_meas):
    return np.diff(q_meas, axis=0)  # realized_step[k] = q_meas[k+1] - q_meas[k]


def rmse_at_delay(realized, commanded_delta, d, idx):
    # predict realized[k] (k in idx) from commanded_delta[k-d]; skip ticks
    # where k-d < 0 (no history yet).
    valid = idx[idx - d >= 0]
    if len(valid) == 0:
        return np.nan, 0
    pred = commanded_delta[valid - d]
    actual = realized[valid]
    return float(np.sqrt(np.mean((actual - pred) ** 2))), len(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to a path_follow.jsonl")
    ap.add_argument("--delays", default="0,1,2,3")
    args = ap.parse_args()

    q_meas, q_delta = load_log(args.log)
    realized = realized_step(q_meas)  # length T-1
    T = len(realized)
    delays = [int(d) for d in args.delays.split(",")]

    n_folds = 5
    bounds = [int(round(i * T / n_folds)) for i in range(n_folds + 1)]
    fold_idx = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_folds)]

    print(f"T={T} ticks, fold sizes: {[len(f) for f in fold_idx]}")
    if T < 150:
        print(f"WARNING: only {T} ticks -- below the ~150-tick practical minimum "
              f"for 5 folds of >=30 ticks each; treat results cautiously.")

    g_by_fold = {d: [] for d in delays if d != 0}
    best_d_per_fold = []
    for f in range(n_folds):
        held_out = fold_idx[f]
        train = np.concatenate([fold_idx[i] for i in range(n_folds) if i != f])
        print(f"\n-- fold {f}: held-out n={len(held_out)}, train n={len(train)} --")
        heldout_rmse = {}
        for d in delays:
            train_rmse, n_tr = rmse_at_delay(realized, q_delta, d, train)
            ho_rmse, n_ho = rmse_at_delay(realized, q_delta, d, held_out)
            heldout_rmse[d] = ho_rmse
            print(f"  d={d}: train RMSE={train_rmse:.6f} rad (n={n_tr}), "
                  f"held-out RMSE={ho_rmse:.6f} rad (n={n_ho})")
        base = heldout_rmse.get(0)
        for d in delays:
            if d == 0 or base is None or base == 0:
                continue
            g = 1.0 - (heldout_rmse[d] / base) ** 2
            g_by_fold[d].append(g)
            print(f"  G_{d} = {g:.4f}")
        best_d = min(delays, key=lambda d: heldout_rmse[d] if not np.isnan(heldout_rmse[d]) else np.inf)
        best_d_per_fold.append(best_d)
        print(f"  best d on this held-out fold: {best_d}")

    print("\n=== SUMMARY ===")
    print("best d per fold:", best_d_per_fold)
    from collections import Counter
    counts = Counter(best_d_per_fold)
    print("vote counts:", dict(counts))
    if 2 in g_by_fold:
        print(f"G_2 per fold: {[round(g, 4) for g in g_by_fold[2]]}")
        g2_ok = all(g > 0.9 for g in g_by_fold[2])
    else:
        g2_ok = False
    d2_majority = counts.get(2, 0) >= 4
    if d2_majority and g2_ok:
        print("\nRESULT: d=2 CONFIRMED (>=4/5 folds prefer d=2, G_2 > 0.9 on every fold).")
    else:
        print("\nRESULT: STOP -- d=2 NOT confirmed by this 5-fold check "
              "(either fold disagreement or G_2 <= 0.9 on some fold). "
              "Do not proceed to Day-1/core campaign on this log's basis "
              "without investigating why.")


if __name__ == "__main__":
    main()
