"""Script 3 -- plan: snug-bouncing-giraffe.md. NEW runs only.

Step 1: VERIFY (not assume) that the production
ContactJacobianHybridModel's per-tick computation is memoryless, by
replaying every tick's logged (q_meas_rad, insertion_length_m) through
the model in two different orders -- forward (the original sequential
order) and reversed -- and diffing (s_q, delta_L, r_q, weight) tick-
by-tick. project_s_q is a pure global search over q_meas with no
s_prev argument, so if memoryless these must be bit-identical
regardless of order; this is checked explicitly rather than assumed
from reading the code alone.

Step 2: classify every tick of every NEW rep into one of 4 modes
(pure map / blend / schedule fallback / map invalid), using the
model's own r_a/r_b and smoothstep weight, and report the % of ticks
in each mode per rep plus one combined plot (error, r_q, weight, mode
vs s) per rep. Any region where NEW fell back to J1 is flagged
explicitly, since that region cannot be attributed to the map.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")
import run_registry as reg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proper_research.controllers.contact_jacobian_hybrid import ContactJacobianHybridModel, HybridMapConfig

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

DT = reg.DT
VLIM6 = 0.10
INS_RATE_LIM = 2.0e-3
D7 = np.array([VLIM6] * 6 + [INS_RATE_LIM], dtype=float) * DT


def build_model():
    return ContactJacobianHybridModel(HybridMapConfig(map_dir=reg.HYBRID_MAP_DIR), D=D7)


def per_tick_diagnostics(model, q_meas, L_meas):
    s_q = model.project_s_q(q_meas)
    delta_L = float(L_meas) - model.L_plan(s_q)
    r_q = model.r_q(q_meas, s_q)
    w = model.smoothstep_weight(r_q, model.r_a, model.r_b)
    map_valid_here = False
    if w > 0.0:
        _, _, map_valid_here, _ = model.lookup_j3(s_q, delta_L)
    return dict(s_q=s_q, delta_L=delta_L, r_q=r_q, weight=w, map_valid_here=map_valid_here)


def classify_mode(diag):
    if diag["weight"] <= 0.0:
        return "schedule_fallback_distance"
    if not diag["map_valid_here"]:
        return "map_invalid_despite_w_gt_0"
    if diag["weight"] >= 0.999:
        return "pure_map"
    return "blend"


def verify_memorylessness(model, q_list, L_list):
    n = len(q_list)
    forward = [per_tick_diagnostics(model, q_list[k], L_list[k]) for k in range(n)]
    reversed_order = [None] * n
    for k in reversed(range(n)):
        reversed_order[k] = per_tick_diagnostics(model, q_list[k], L_list[k])
    max_abs_diff_s_q = max(abs(forward[k]["s_q"] - reversed_order[k]["s_q"]) for k in range(n))
    max_abs_diff_r_q = max(abs(forward[k]["r_q"] - reversed_order[k]["r_q"]) for k in range(n))
    max_abs_diff_w = max(abs(forward[k]["weight"] - reversed_order[k]["weight"]) for k in range(n))
    identical = max_abs_diff_s_q < 1e-12 and max_abs_diff_r_q < 1e-12 and max_abs_diff_w < 1e-12
    return identical, dict(max_abs_diff_s_q=max_abs_diff_s_q, max_abs_diff_r_q=max_abs_diff_r_q,
                             max_abs_diff_w=max_abs_diff_w)


def main():
    model = build_model()
    print(f"Loaded production hybrid model: r_a={model.r_a:.4f} r_b={model.r_b:.4f} "
          f"n_tiles={len(model.tiles)}")

    all_results = {}
    for rep, run_dir_name in reg.RUNS["NEW"]:
        run_path = reg.run_dir("NEW", rep)
        rows = reg.load_path_follow(run_path)
        q_list = [np.asarray(r["q_meas_rad"], dtype=float) for r in rows]
        L_list = [float(r["insertion_length_m"]) for r in rows]
        t_list = [float(r["t_s"]) for r in rows]
        err_mm = [float(r["error_norm_mm"]) for r in rows]

        identical, diff_report = verify_memorylessness(model, q_list, L_list)
        print(f"\n--- NEW rep{rep}: memorylessness check (forward vs reversed replay order) ---")
        print(f"  identical={identical}  {diff_report}")

        diags = [per_tick_diagnostics(model, q_list[k], L_list[k]) for k in range(len(rows))]
        modes = [classify_mode(d) for d in diags]
        mode_counts = {m: modes.count(m) for m in set(modes)}
        mode_frac = {m: c / len(modes) for m, c in mode_counts.items()}

        print(f"  mode fractions: {json.dumps(mode_frac, indent=2)}")

        s_q = np.array([d["s_q"] for d in diags])
        r_q = np.array([d["r_q"] for d in diags])
        w = np.array([d["weight"] for d in diags])

        fallback_s_ranges = s_q[np.array([m != "pure_map" and m != "blend" for m in modes])]
        if len(fallback_s_ranges) > 0:
            print(f"  FLAG: schedule-fallback/map-invalid ticks occur at s_q in "
                  f"[{fallback_s_ranges.min()*1000:.2f},{fallback_s_ranges.max()*1000:.2f}]mm -- "
                  "any improvement there cannot be attributed to the J3 map.")

        all_results[f"NEW_rep{rep}"] = dict(
            memoryless=identical, memoryless_diff=diff_report,
            mode_fractions=mode_frac, mode_counts=mode_counts,
            s_q_mm=(s_q * 1000).tolist(), r_q=r_q.tolist(), weight=w.tolist(),
            modes=modes,
        )

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(s_q * 1000, err_mm, '.', ms=2)
        axes[0].set_ylabel("tip error (mm)")
        axes[1].plot(s_q * 1000, r_q, '.', ms=2, color='tab:orange')
        axes[1].axhline(model.r_a, color='gray', ls='--', lw=0.8)
        axes[1].axhline(model.r_b, color='gray', ls='--', lw=0.8)
        axes[1].set_ylabel("r_q (dimensionless)")
        axes[2].plot(s_q * 1000, w, '.', ms=2, color='tab:green')
        axes[2].set_ylabel("blend weight w")
        mode_to_int = {"pure_map": 0, "blend": 1, "schedule_fallback_distance": 2, "map_invalid_despite_w_gt_0": 3}
        axes[3].plot(s_q * 1000, [mode_to_int[m] for m in modes], '.', ms=3, color='tab:red')
        axes[3].set_yticks([0, 1, 2, 3])
        axes[3].set_yticklabels(["pure_map", "blend", "sched_fb", "map_inval"])
        axes[3].set_ylabel("mode")
        axes[3].set_xlabel("s_q (mm)")
        fig.suptitle(f"NEW rep{rep} hybrid operation breakdown")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"phase3_hybrid_breakdown_rep{rep}.png"), dpi=120)
        plt.close(fig)

    with open(os.path.join(OUT_DIR, "phase3_hybrid_operation_breakdown.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {OUT_DIR}/phase3_hybrid_operation_breakdown.json and 3 breakdown plots")


if __name__ == "__main__":
    main()
