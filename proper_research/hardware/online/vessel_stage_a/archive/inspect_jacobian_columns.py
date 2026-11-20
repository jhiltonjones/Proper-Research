"""Finite-difference-validate the analytical Jacobian, column by column, at a
handful of representative nodes along an already-solved path -- specifically
checking whether the INSERTION column (index 6, d output/dL) is a source of
bugs/error vs the six joint columns, for both the contact-aware and
no-contact models.

Targets: the rank-deficient (effective rank < 3) nodes found by
compute_contact_vs_nocontact_jacobians.py, their immediate neighbours (to see
whether a column breaks down gradually or discontinuously), plus a spread of
"normal" nodes for baseline comparison.

Usage: python inspect_jacobian_columns.py <plan_output_root> [<jac_csv>]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE

plan_root = Path(sys.argv[1])
jac_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else plan_root / "contact_vs_nocontact_jacobian.csv"

path_csv = plan_root / "vessel_lumen" / "offline_inverse_configuration" / "inverse_configuration_path.csv"
df = pd.read_csv(path_csv)
states = df[["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy()
s_mm_all = df["s_m"].to_numpy() * 1e3
N = states.shape[0]

jac_df = pd.read_csv(jac_csv)
bad_idx = jac_df.index[jac_df["rank_contact"] < 3].tolist()
print(f"rank-deficient node indices (contact model): {bad_idx}")

targets = set()
for i in bad_idx:
    for j in range(max(0, i - 2), min(N, i + 3)):
        targets.add(j)
# evenly spread "normal" baseline nodes
targets.update(int(round(x)) for x in np.linspace(0, N - 1, 12))
targets = sorted(targets)
print(f"inspecting {len(targets)} nodes: {targets}")

BEAM_BASE_COLS = ["q1", "q2", "q3", "q4", "q5", "q6", "L"]
OUT_ROWS = ["tip_x", "tip_y", "tip_z", "tan_x", "tan_y", "tan_z"]

t0 = time.perf_counter()
exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
    build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11,
        jacobian_mode="accurate", plant_contact=True,
    )
)
print(f"[build] model bundle built in {time.perf_counter()-t0:.1f}s")

adapters = {
    "contact": build_diagnostic_adapter(beam_model=bundle.models["contact"], controller_pack=controller_pack, jacobian_mode="accurate"),
    "no_contact": build_diagnostic_adapter(beam_model=bundle.models["no_contact"], controller_pack=controller_pack, jacobian_mode="accurate"),
}

records = []
t0 = time.perf_counter()
for idx in targets:
    state = states[idx]
    s_mm = s_mm_all[idx]
    for model_name, adapter in adapters.items():
        chain = adapter.validate_chain_rule(state, joint_step_rad=1.5e-2, insertion_step_m=5.0e-3)
        J_an = np.asarray(chain["analytical"], dtype=float)
        J_fd = np.asarray(chain["finite_difference"], dtype=float)
        col_an_norm = np.linalg.norm(J_an, axis=0)
        col_fd_norm = np.linalg.norm(J_fd, axis=0)
        col_diff_norm = np.linalg.norm(J_an - J_fd, axis=0)
        col_rel_err = col_diff_norm / np.maximum(col_fd_norm, 1e-9)
        for c in range(7):
            records.append({
                "node_index": idx, "s_mm": s_mm, "model": model_name, "column": BEAM_BASE_COLS[c],
                "analytical_norm": col_an_norm[c], "finite_diff_norm": col_fd_norm[c],
                "abs_diff_norm": col_diff_norm[c], "relative_error": col_rel_err[c],
            })
        # flag anything badly wrong on the insertion column specifically
        if col_rel_err[6] > 0.5 or col_fd_norm[6] < 1e-9:
            print(f"\n*** node={idx} s={s_mm:.3f}mm model={model_name}: INSERTION COLUMN FLAG ***")
            print(f"  analytical dOut/dL = {np.round(J_an[:, 6], 6)}")
            print(f"  finite_diff dOut/dL = {np.round(J_fd[:, 6], 6)}")
            print(f"  relative_error = {col_rel_err[6]:.4f}")
    if (targets.index(idx) + 1) % 5 == 0:
        print(f"  ...{targets.index(idx)+1}/{len(targets)} nodes done, elapsed={time.perf_counter()-t0:.1f}s", flush=True)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s")

out_df = pd.DataFrame.from_records(records)
out_csv = plan_root / "jacobian_column_validation.csv"
out_df.to_csv(out_csv, index=False)
print(f"saved {out_csv}")

# Summary: per-column relative-error statistics, split by model -- this is
# the direct answer to "is the insertion column specifically susceptible".
print("\n=== per-column relative error vs finite difference (median / max) ===")
for model_name in ("contact", "no_contact"):
    sub = out_df[out_df["model"] == model_name]
    print(f"\n{model_name}:")
    for col in BEAM_BASE_COLS:
        s = sub[sub["column"] == col]["relative_error"]
        flag = "insertion" if col == "L" else ""
        print(f"  {col:>3s} {flag:>10s}: median={s.median():.4f}  max={s.max():.4f}  "
              f"n_above_0.3={(s > 0.3).sum()}/{len(s)}")
