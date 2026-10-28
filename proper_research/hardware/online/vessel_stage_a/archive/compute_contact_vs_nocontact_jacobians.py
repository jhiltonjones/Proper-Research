"""Compute the CONTACT and NO-CONTACT model Jacobians at every state along an
already-solved (contact-aware) offline inverse-configuration path.

This does not re-solve anything -- it takes the joint/insertion states
[q1..q6, L] the contact-aware planner actually found and asks: "what would
each model (contact-aware vs contact-blind) say the local output Jacobian
is, AT THESE SAME CONFIGURATIONS?" This isolates the Jacobian's sensitivity
to the contact physics alone, holding the path fixed -- a direct, apples-
to-apples comparison, unlike comparing two independently-solved paths
(which also differ in WHERE the solver went).

Usage: python compute_contact_vs_nocontact_jacobians.py <plan_output_root>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE, beam_base_pose6

plan_root = Path(sys.argv[1])
csv_path = plan_root / "vessel_lumen" / "offline_inverse_configuration" / "inverse_configuration_path.csv"
df = pd.read_csv(csv_path)
feasible = df["feasible"].astype(bool).to_numpy()
if not np.all(feasible):
    print(f"WARNING: {int((~feasible).sum())}/{len(df)} nodes are not feasible; "
          "including them anyway (diagnostic only).")

s_mm = df["s_m"].to_numpy() * 1e3
states = df[["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy()
N = states.shape[0]
print(f"{N} nodes loaded from {csv_path}")

# Build both models against the SAME lumen the contact run actually used.
# initial_poses only affects p0/IK-seed bookkeeping, not the two models'
# physics -- build_model_bundle's robot_dh/T_F_M and contact-wall geometry
# are what matter here, and both come from CURRENT_LUMEN_FILE regardless.
BEAM_BASE_PIVOT = beam_base_pose6()
t0 = time.perf_counter()
exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
    build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11,
        jacobian_mode="accurate", plant_contact=True,
    )
)
print(f"[build] model bundle built in {time.perf_counter()-t0:.1f}s")

model_c = bundle.models["contact"]
model_nc = bundle.models["no_contact"]
adapter_c = build_diagnostic_adapter(beam_model=model_c, controller_pack=controller_pack, jacobian_mode="accurate")
adapter_nc = build_diagnostic_adapter(beam_model=model_nc, controller_pack=controller_pack, jacobian_mode="accurate")


def _svd_diag(J6x7: np.ndarray) -> tuple[float, int, np.ndarray]:
    J_pos = J6x7[:3]  # position-only 3x7 block, matching this project's existing cond_eff(J) convention
    U, S, _ = np.linalg.svd(J_pos)
    rank = int(np.sum(S > 1e-6 * max(S[0], 1e-30)))
    cond = float(S[0] / max(S[-1], 1e-12)) if S.size else float("nan")
    return cond, rank, S


cond_c = np.full(N, np.nan)
cond_nc = np.full(N, np.nan)
rank_c = np.zeros(N, dtype=int)
rank_nc = np.zeros(N, dtype=int)
fro_diff = np.full(N, np.nan)
angle_deg = np.full(N, np.nan)  # angle between the two models' dominant (largest-sigma) position singular vectors

t0 = time.perf_counter()
for i in range(N):
    state = states[i]
    Jc = np.asarray(adapter_c.continuous_output_jacobian(state), dtype=float).reshape(6, 7)
    Jnc = np.asarray(adapter_nc.continuous_output_jacobian(state), dtype=float).reshape(6, 7)
    cond_c[i], rank_c[i], Sc = _svd_diag(Jc)
    cond_nc[i], rank_nc[i], Snc = _svd_diag(Jnc)
    fro_diff[i] = float(np.linalg.norm(Jc[:3] - Jnc[:3]))
    Uc, _, _ = np.linalg.svd(Jc[:3])
    Unc, _, _ = np.linalg.svd(Jnc[:3])
    cos_angle = float(np.clip(abs(np.dot(Uc[:, 0], Unc[:, 0])), -1.0, 1.0))
    angle_deg[i] = float(np.degrees(np.arccos(cos_angle)))
    if (i + 1) % 50 == 0 or i == N - 1:
        print(f"  [{i+1}/{N}] s={s_mm[i]:.2f}mm  cond_c={cond_c[i]:.3e} "
              f"cond_nc={cond_nc[i]:.3e}  fro_diff={fro_diff[i]:.4f}  "
              f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)

print(f"\nTotal Jacobian evaluation time: {time.perf_counter()-t0:.1f}s for {N} nodes x 2 models")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.semilogy(s_mm, cond_c, "-", color="tab:red", lw=2, label="contact-aware model")
ax.semilogy(s_mm, cond_nc, "-", color="tab:blue", lw=2, label="no-contact model")
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("effective condition number (position 3x7 block)")
ax.set_title("Jacobian conditioning along the CONTACT solution's own path")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

ax = axes[0, 1]
ax.plot(s_mm, rank_c, "-", color="tab:red", lw=2, label="contact-aware model", drawstyle="steps-post")
ax.plot(s_mm, rank_nc, "--", color="tab:blue", lw=2, label="no-contact model", drawstyle="steps-post")
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("effective rank (position 3x7 block)")
ax.set_title("Jacobian rank")
ax.set_yticks(range(0, 4))
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(s_mm, fro_diff, "-", color="tab:purple", lw=2)
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("||J_contact - J_no_contact||_F  (position block)")
ax.set_title("Raw Jacobian difference between the two models")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(s_mm, angle_deg, "-", color="tab:green", lw=2)
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("angle [deg]")
ax.set_title("Angle between dominant singular directions (contact vs no-contact)")
ax.grid(alpha=0.3)

contact_active = df["contact_active"].astype(bool).to_numpy() if "contact_active" in df.columns else None
if contact_active is not None and contact_active.any():
    for ax in axes.ravel():
        ax.fill_between(s_mm, *ax.get_ylim(), where=contact_active, color="gray", alpha=0.12,
                         label="contact_active" if ax is axes[0, 0] else None, zorder=0)

fig.suptitle(f"Contact vs no-contact Jacobian, evaluated AT the contact solution's own path: {plan_root.name}",
             fontsize=12)
fig.tight_layout()
out_png = plan_root / "contact_vs_nocontact_jacobian.png"
fig.savefig(out_png, dpi=150)

print(f"\ncondition number: contact [{np.nanmin(cond_c):.3e}, {np.nanmax(cond_c):.3e}], "
      f"no-contact [{np.nanmin(cond_nc):.3e}, {np.nanmax(cond_nc):.3e}]")
print(f"rank: contact [{rank_c.min()}, {rank_c.max()}], no-contact [{rank_nc.min()}, {rank_nc.max()}]")
print(f"Frobenius diff: [{np.nanmin(fro_diff):.4f}, {np.nanmax(fro_diff):.4f}]")
print(f"dominant-direction angle: [{np.nanmin(angle_deg):.2f}, {np.nanmax(angle_deg):.2f}] deg")
print(f"saved {out_png}")

out_csv = plan_root / "contact_vs_nocontact_jacobian.csv"
pd.DataFrame({
    "s_mm": s_mm, "cond_contact": cond_c, "cond_no_contact": cond_nc,
    "rank_contact": rank_c, "rank_no_contact": rank_nc,
    "frobenius_diff": fro_diff, "dominant_direction_angle_deg": angle_deg,
    "contact_active": contact_active if contact_active is not None else np.zeros(N, dtype=bool),
}).to_csv(out_csv, index=False)
print(f"saved {out_csv}")
