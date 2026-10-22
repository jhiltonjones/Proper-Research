"""Validation sweep: old (tikhonov) vs new (truncated_svd) implicit
sensitivity, both checked against INDEPENDENT finite differencing of the
nonlinear forward model (direct re-solve to equilibrium at each perturbed
pose, central difference -- no implicit/analytical sensitivity machinery at
all) -- the same methodology jnc_contact_source_jacobian.py already uses for
the four established Stage-2 states.

States swept:
  - the 4 Stage-2 states (phi=35deg, r=225mm, L=40/50mm, psi=+-30deg)
  - the previously-catastrophic Stage-3 node 150 (phi30_L30_left1mm_newwall,
    0.5mm-tolerance contact run) plus its immediate neighbours

For each state, records the global relative Frobenius error of J_tikhonov
and J_truncated_svd against J_FD, plus the contact Hessian's spectral gap
ratio and effective rank -- then plots error vs spectral-gap-ratio for both
inversion modes, to show whether truncated_svd stays accurate across both
regimes while tikhonov degrades specifically where the gap is large.

Usage: python validate_rank_aware_hessian_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import (
    BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix,
)
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context

EPS_TRANS = 0.002
EPS_ROT = np.radians(1.0)
EPS_L = 5.0e-4


def fd_output_jacobian(model, xyz0, rotvec0, L0):
    """Independent brute-force central difference directly on model.solve --
    no implicit/analytical sensitivity machinery involved at all."""
    from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import solve_pose

    def out_of(xyz, rotvec, L):
        o = solve_pose(model, xyz, rotvec, L)
        return np.concatenate([np.asarray(o["tip"]), np.asarray(o["beam_tangent_at_tip"])])

    J = np.zeros((6, 7))
    for i in range(7):
        if i < 3:
            eps = EPS_TRANS
            d = np.zeros(3); d[i] = eps
            op = out_of(xyz0 + d, rotvec0, L0)
            om = out_of(xyz0 - d, rotvec0, L0)
        elif i < 6:
            eps = EPS_ROT
            ax = i - 3
            d = np.zeros(3); d[ax] = eps
            Rp = Rot.from_rotvec(d).as_matrix() @ Rot.from_rotvec(rotvec0).as_matrix()
            Rm = Rot.from_rotvec(-d).as_matrix() @ Rot.from_rotvec(rotvec0).as_matrix()
            op = out_of(xyz0, Rot.from_matrix(Rp).as_rotvec(), L0)
            om = out_of(xyz0, Rot.from_matrix(Rm).as_rotvec(), L0)
        else:
            eps = EPS_L
            op = out_of(xyz0, rotvec0, L0 + eps)
            om = out_of(xyz0, rotvec0, L0 - eps)
        J[:, i] = (op - om) / (2 * eps)
    return J


def analytic_jacobian(model, xyz0, rotvec0, L0, mode, hessian_inversion):
    p7 = np.concatenate([xyz0, rotvec0, [L0]])
    model.solve(p7, commit=True, reuse_cache=True)
    J = model.jacobian_output_actuation_tangent(
        p7, solve_if_needed=False, mode=mode, hessian_inversion=hessian_inversion,
    )
    info = dict(model.last_sens_info)
    return np.asarray(J, dtype=float).reshape(6, 7), info


def relerr(J_a, J_fd):
    return float(np.linalg.norm(J_a - J_fd) / max(np.linalg.norm(J_fd), 1e-12))


def main():
    records = []

    # --- 4 Stage-2 states ---
    beam_base_xyz = BEAM_BASE_XYZ_M.copy()
    beam_base_xyz6 = beam_base_pose6()
    PHI_DEG, RADIUS_MM = 35.0, 225.0
    for L_MM in (40.0, 50.0):
        for PSI_DEG in (30.0, -30.0):
            L_M = L_MM / 1000.0
            phi = np.radians(PHI_DEG)
            xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
            R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
            R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
            R0 = R_aligned @ R_psi
            rotvec0 = Rot.from_matrix(R0).as_rotvec()

            bundle = build_model_bundle(CURRENT_LUMEN_FILE, beam_base_xyz6, REFERENCE_MAGNET_POSE6, L_M)
            model = bundle.models["contact"]

            J_fd = fd_output_jacobian(model, xyz0, rotvec0, L_M)
            J_tik, info_tik = analytic_jacobian(model, xyz0, rotvec0, L_M, "accurate", "tikhonov")
            J_tsvd, info_tsvd = analytic_jacobian(model, xyz0, rotvec0, L_M, "accurate", "truncated_svd")

            label = f"Stage2 psi={PSI_DEG:+.0f} L={L_MM:.0f}mm"
            records.append({
                "label": label, "group": "stage2",
                "relerr_tikhonov": relerr(J_tik, J_fd), "relerr_truncated_svd": relerr(J_tsvd, J_fd),
                "effective_rank": info_tsvd.get("effective_rank"), "n_total": info_tsvd.get("n_total"),
                "gap_ratio": info_tsvd.get("gap_ratio"), "smallest_retained_eigval": info_tsvd.get("smallest_retained_eigval"),
                "largest_eigval": info_tsvd.get("largest_eigval"),
            })
            print(f"{label}: relerr tikhonov={records[-1]['relerr_tikhonov']:.4f}  "
                  f"truncated_svd={records[-1]['relerr_truncated_svd']:.4f}  "
                  f"rank={records[-1]['effective_rank']}/{records[-1]['n_total']}  "
                  f"gap_ratio={records[-1]['gap_ratio']}", flush=True)

    # --- Stage-3 node 150 + neighbours ---
    plan_root = Path("plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06")
    df = pd.read_csv(plan_root / "vessel_lumen" / "offline_inverse_configuration" / "inverse_configuration_path.csv")
    exp_cfg, bundle3, controller_pack, out_root, centreline, lumen_R, provenance = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
    )
    model3 = bundle3.models["contact"]
    adapter3 = build_diagnostic_adapter(beam_model=model3, controller_pack=controller_pack, jacobian_mode="accurate")
    import proper_research.simulation.simulations.controller_factory_joint_space as cfjs

    for idx in (148, 149, 150, 151, 152):
        state = df.loc[idx, ["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy(dtype=float)
        T_R_M = adapter3.magnet_transform(state)
        xyz0 = T_R_M[:3, 3].copy()
        rotvec0 = Rot.from_matrix(T_R_M[:3, :3]).as_rotvec()
        L0 = float(state[6])

        J_fd = fd_output_jacobian(model3, xyz0, rotvec0, L0)
        J_tik, info_tik = analytic_jacobian(model3, xyz0, rotvec0, L0, "accurate", "tikhonov")
        J_tsvd, info_tsvd = analytic_jacobian(model3, xyz0, rotvec0, L0, "accurate", "truncated_svd")

        label = f"Stage3 node{idx} s={df.loc[idx,'s_m']*1e3:.2f}mm"
        records.append({
            "label": label, "group": "stage3",
            "relerr_tikhonov": relerr(J_tik, J_fd), "relerr_truncated_svd": relerr(J_tsvd, J_fd),
            "effective_rank": info_tsvd.get("effective_rank"), "n_total": info_tsvd.get("n_total"),
            "gap_ratio": info_tsvd.get("gap_ratio"), "smallest_retained_eigval": info_tsvd.get("smallest_retained_eigval"),
            "largest_eigval": info_tsvd.get("largest_eigval"),
        })
        print(f"{label}: relerr tikhonov={records[-1]['relerr_tikhonov']:.4f}  "
              f"truncated_svd={records[-1]['relerr_truncated_svd']:.4f}  "
              f"rank={records[-1]['effective_rank']}/{records[-1]['n_total']}  "
              f"gap_ratio={records[-1]['gap_ratio']}", flush=True)

    out_df = pd.DataFrame.from_records(records)
    out_csv = Path("plans/stage3_design/rank_aware_hessian_validation_sweep.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    # --- Plot: error vs gap ratio, both inversion modes ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for group, marker, color in (("stage2", "o", "tab:green"), ("stage3", "s", "tab:red")):
        sub = out_df[out_df["group"] == group]
        ax.scatter(sub["gap_ratio"].clip(lower=1), sub["relerr_tikhonov"] * 100, marker=marker, color=color,
                   label=f"{group} tikhonov (old)", s=70, facecolors="none", edgecolors=color)
        ax.scatter(sub["gap_ratio"].clip(lower=1), sub["relerr_truncated_svd"] * 100, marker=marker, color=color,
                   label=f"{group} truncated_svd (new)", s=70)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hessian spectral gap ratio (sigma_i/sigma_i+1)")
    ax.set_ylabel("relative error vs direct FD [%]")
    ax.set_title("Error vs spectral gap: hollow=old, filled=new")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    x = np.arange(len(out_df))
    width = 0.35
    ax.bar(x - width/2, out_df["relerr_tikhonov"] * 100, width, label="tikhonov (old)", color="tab:orange")
    ax.bar(x + width/2, out_df["relerr_truncated_svd"] * 100, width, label="truncated_svd (new)", color="tab:blue")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(out_df["label"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("relative error vs direct FD [%]")
    ax.set_title("Per-state error, old vs new")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Rank-aware Hessian inversion validation: old vs new vs independent forward-model FD", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = Path("plans/stage3_design/rank_aware_hessian_validation_sweep.png")
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
