"""Regenerate the final transient_57mm_deep_diagnostic.png using the
per-tick peak-refine contact-state data (transient_peak_refine.csv) in place
of the coarse 8-sample-per-rep grid, now that the true tracking-error peak
location has been pinned down (s~=60.2-60.5mm, not s~=57mm -- see
transient_57mm_part2.py's 1D coarse pass, which first revealed the peak sits
just past its own window, and transient_peak_refine.py, which resolved it at
full tick resolution). No new model solves -- purely a plotting pass over
already-computed tables.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"

df_abc = pd.read_csv(f"{OUT}/tables/transient_57mm_diagnostic_summary.csv")
df_1d_coarse = pd.read_csv(f"{OUT}/tables/transient_57mm_1D_contact_state.csv")
df_peak = pd.read_csv(f"{OUT}/tables/transient_peak_refine.csv")

colors = {2: "#1b7f3b", 3: "#2f8fd1", 4: "#b3331d"}
fig, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)

ax = axes[0]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_track_mm"], color=colors[rep], lw=1.3, label=f"rep{rep}")
for rep, d in df_peak.groupby("rep"):
    pk = d[d["is_peak"]]
    ax.scatter(pk["s_mm"], pk["e_track_mm"], color=colors[rep], marker="*", s=140, zorder=5,
               edgecolor="k", linewidth=0.5)
ax.set_ylabel("$e_{track}$ (mm)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("s=53-61mm deep diagnostic, scheduled MPC-$J_C$@210mm, all 3 reps\n"
             "(stars = true per-rep peak, located at s~60.2-60.5mm, not s~57mm)", fontsize=10)

ax = axes[1]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_x_mm"], color=colors[rep], lw=1.0, ls="-", label=f"rep{rep} $e_x$" if rep == 2 else None)
    ax.plot(d["s_mm"], d["e_y_mm"], color=colors[rep], lw=1.0, ls="--", alpha=0.7)
    ax.plot(d["s_mm"], d["e_z_mm"], color=colors[rep], lw=1.0, ls=":", alpha=0.7)
ax.set_ylabel("1A: $e_x,e_y,e_z$ (mm)\n(solid/dash/dot; $e_z$=0 always)", fontsize=8)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["state_age_s"] * 1e3, color=colors[rep], lw=1.0, ls="-", label=f"rep{rep} age" if rep == 2 else None)
ax2 = ax.twinx()
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax2.plot(d["s_mm"], d["dt_actual_s"] * 1e3, color=colors[rep], lw=0.8, ls=":", alpha=0.6)
ax.set_ylabel("1A: meas. age (ms, solid)", fontsize=8)
ax2.set_ylabel("tick dt (ms, dotted)", fontsize=7)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[3]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.semilogy(d["s_mm"], d["kappa_ref_1_per_m"].clip(lower=1e-12), color=colors[rep], lw=1.0)
ax.set_ylabel("1B: ref curvature $\\kappa$\n(1/m, log scale)", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[4]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_chi_norm"], color=colors[rep], lw=1.2, marker="o", ms=3)
ax.set_ylabel("1C: $\\|\\Delta\\chi_{cmd}-\\Delta\\chi_{actual}\\|$", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[5]
for rep, d in df_1d_coarse.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["gap_min_m"] * 1e3, color=colors[rep], lw=0.8, ls="--", alpha=0.4, marker="s", ms=3)
for rep, d in df_peak.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["gap_min_m"] * 1e3, color=colors[rep], lw=1.6, marker="o", ms=4,
            label=f"rep{rep} (fine)" if rep == 2 else None)
    pk = d[d["is_peak"]]
    ax.scatter(pk["s_mm"], pk["gap_min_m"] * 1e3, color=colors[rep], marker="*", s=140, zorder=5, edgecolor="k")
ax.axhline(0.05, color="gray", lw=0.6, ls=":")
ax.text(53.2, 0.051, "pen_switch=0.05mm", fontsize=6, color="gray")
ax.set_ylabel("1D: min contact gap\n(mm; dashed=coarse 8pt/rep,\nsolid=fine per-tick)", fontsize=7)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[6]
for rep, d in df_peak.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["contact_force_norm"], color=colors[rep], lw=1.6, marker="o", ms=4)
    pk = d[d["is_peak"]]
    ax.scatter(pk["s_mm"], pk["contact_force_norm"], color=colors[rep], marker="*", s=140, zorder=5, edgecolor="k")
ax.set_ylabel("1D: contact force norm\n(fine per-tick, a.u.)", fontsize=8)
ax.set_xlabel("path progress s (mm)")
ax.grid(alpha=0.3)

fig.tight_layout()
f = f"{OUT}/figures/transient_57mm_deep_diagnostic.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"saved {f}")
