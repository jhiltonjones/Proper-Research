"""Stage-3-style bug check on the two Jacobian schedules just built for the
live closed-loop MPC run (/tmp/vessel_c_schedule.npy, /tmp/vessel_nc_schedule.npy,
built 2026-10-07 by run_mpc_delay_aware_vessel.build_or_load_schedule).

Two layers, same priorities as the original Stage-3 investigation:

1. CHEAP, no new solves: scan each full schedule's singular-value spectrum,
   condition number, and sample-to-sample Frobenius jump vs s, looking for
   the kind of isolated single-tick numerical-artifact spike this project
   has hit before (the "fast" mode's spurious sigma1~79 vs ~1.0-1.1
   neighbours, paper/model_necessity_study.tex Sec. mn-exp3) -- this build
   used "accurate" mode throughout, so such a spike would be a genuine new
   finding, not an expected failure mode.

2. EXPENSIVE, independent ground truth: at five representative locations
   per plan (pre-contact, contact entry, s~37.5mm, s~60mm, near end -- same
   REGIONS as compare_cnc_corrected_sweep.py), re-derive the position
   Jacobian two ways and compare all three against each other:
     a. independent FD ladder (diagnose_position_jacobian_convergence.
        fd_ladder_position_only, now accepting an explicit warm-start seed)
     b. the analytical "accurate" Jacobian (tikhonov AND truncated_svd),
        freshly recomputed
     c. the ACTUAL schedule[idx] entry loaded from the saved .npy -- byte-
        for-byte what the live MPC run will read
   All three computed from the SAME branch-consistent state: every sample
   is reached by sequential continuation from s=0 (compare_cnc_corrected_
   sweep.walk_continuation), not an isolated solve -- the exact class of
   bug ("isolated re-solve lands on the wrong equilibrium branch") this
   project's Stage-3 investigation found and fixed its methodology around.

Usage: python validate_mpc_schedule_jacobians.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs

from proper_research.hardware.online.vessel_stage_a.archive.diagnose_position_jacobian_convergence import (
    fd_ladder_position_only, relerr,
)
from proper_research.hardware.online.vessel_stage_a.archive.compare_cnc_corrected_sweep import walk_continuation
from proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel import (
    _recalibrated_make_initial_poses,
)

OUT_DIR = "plans/stage3_design/openloop_hw_analysis"
REGIONS = [
    ("pre-contact", 2.0), ("contact entry", 17.5), ("s~37.5mm", 37.5),
    ("s~60mm", 60.0), ("near end", 73.0),
]
PLANS = {
    "CONTACT": dict(
        plan_root="plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06",
        schedule_path="/tmp/vessel_c_schedule.npy", model_key="contact",
    ),
    "NO-CONTACT": dict(
        plan_root="plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06",
        schedule_path="/tmp/vessel_nc_schedule.npy", model_key="no_contact",
    ),
}


def scan_schedule(name: str, schedule: np.ndarray, path_s_mm: np.ndarray) -> pd.DataFrame:
    """Cheap, no-solve pass: SVD/condition/continuity of the saved schedule."""
    n = schedule.shape[0]
    rows = []
    prev = None
    for i in range(n):
        J = schedule[i]
        sv = np.linalg.svd(J, compute_uv=False)
        jump = float(np.linalg.norm(J - prev)) if prev is not None else float("nan")
        rows.append({
            "plan": name, "idx": i, "s_mm": float(path_s_mm[i]),
            "sigma1": float(sv[0]), "sigma2": float(sv[1]), "sigma3": float(sv[2]),
            "cond": float(sv[0] / max(sv[2], 1e-12)), "frob_jump_vs_prev": jump,
        })
        prev = J
    df = pd.DataFrame(rows)

    # flag isolated spikes: a sample whose sigma1 is a large multiple of
    # BOTH neighbours' sigma1 (an isolated numerical artifact, not a smooth
    # physical trend -- the exact "fast"-mode pattern found previously).
    sigma1 = df["sigma1"].to_numpy()
    spikes = []
    for i in range(1, n - 1):
        left, right = sigma1[i - 1], sigma1[i + 1]
        if sigma1[i] > 5.0 * max(left, right, 1e-9):
            spikes.append(i)
    print(f"[{name}] schedule scan: n={n} sigma1 range=[{sigma1.min():.4f},{sigma1.max():.4f}] "
          f"cond range=[{df['cond'].min():.2f},{df['cond'].max():.2f}] "
          f"isolated sigma1 spikes: {spikes if spikes else 'NONE'}")
    return df


def jacobian_at_seed(model, adapter, chi, u0_seed, hessian_inversion):
    T_R_M = adapter.magnet_transform(chi)
    p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
    problem = model.build_problem(p7)
    result = solve_quasistatic_insertion_optimized(
        problem, u0_flat=u0_seed, options=model.beam, result_detail=model.result_detail,
    )
    model._commit_result(p7, result, problem.L_model)
    model._invalidate_jacobian_values()
    J6 = model.jacobian_output_actuation_tangent(
        p7, solve_if_needed=False, mode="accurate", hessian_inversion=hessian_inversion,
    )
    info = dict(model.last_sens_info or {})
    return np.asarray(J6, dtype=float).reshape(6, 7)[:3], info


def main():
    initial_poses = _recalibrated_make_initial_poses()
    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate",
        plant_contact=True, initial_poses=initial_poses,
    )

    scan_frames = []
    deep_rows = []

    for plan_name, cfg in PLANS.items():
        model = bundle.models[cfg["model_key"]]
        adapter = build_diagnostic_adapter(beam_model=model, controller_pack=controller_pack, jacobian_mode="accurate")

        npz = np.load(f"{cfg['plan_root']}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
        state_ref = npz["state_reference"]
        path_s_mm = npz["path_s_m"] * 1e3
        schedule = np.load(cfg["schedule_path"])
        assert schedule.shape[0] == state_ref.shape[0], (
            f"{plan_name}: schedule has {schedule.shape[0]} samples but state_reference has "
            f"{state_ref.shape[0]} -- schedule does not match this plan's reference, refusing"
        )

        scan_frames.append(scan_schedule(plan_name, schedule, path_s_mm))

        target_k = {label: int(np.argmin(np.abs(path_s_mm - s_target))) for label, s_target in REGIONS}
        n_needed = max(target_k.values()) + 1
        print(f"\n[{plan_name}] walking {n_needed} samples via sequential continuation "
              f"(branch-consistent warm-start source)...", flush=True)
        _, u_stars = walk_continuation(model, adapter, state_ref, n_needed)

        for label, idx in target_k.items():
            chi = state_ref[idx]
            u0_seed = u_stars[idx - 1] if idx > 0 else None
            T_R_M = adapter.magnet_transform(chi)
            xyz0 = T_R_M[:3, 3].copy()
            rotvec0 = Rot.from_matrix(T_R_M[:3, :3]).as_rotvec()
            L0 = float(chi[6])
            tag = f"{plan_name} {label} (idx={idx}, s={path_s_mm[idx]:.2f}mm)"
            print(f"\n{'='*70}\n{tag}\n{'='*70}")

            J_fd, J_fd_smallest, per_col, fro, model = fd_ladder_position_only(
                model, xyz0, rotvec0, L0, tag, u0_seed=u0_seed,
            )

            J_tik, _ = jacobian_at_seed(model, adapter, chi, u0_seed, "tikhonov")
            J_tsvd, info_tsvd = jacobian_at_seed(model, adapter, chi, u0_seed, "truncated_svd")
            J_sched = schedule[idx]

            e_sched_vs_fd = relerr(J_sched, J_fd)
            e_tik_vs_fd = relerr(J_tik, J_fd)
            e_tsvd_vs_fd = relerr(J_tsvd, J_fd)
            e_sched_vs_tik = relerr(J_sched, J_tik)

            print(f"FD reference (largest-h, branch-consistent warm-start):")
            print(np.round(J_fd, 5))
            print(f"saved schedule[{idx}]:")
            print(np.round(J_sched, 5))
            print(f"relerr: schedule_vs_FD={e_sched_vs_fd:.4f}  tikhonov_vs_FD={e_tik_vs_fd:.4f}  "
                  f"truncated_svd_vs_FD={e_tsvd_vs_fd:.4f}  schedule_vs_fresh_tikhonov={e_sched_vs_tik:.4f}")
            print(f"truncated_svd rank info: effective_rank={info_tsvd.get('effective_rank')}/"
                  f"{info_tsvd.get('n_total')} gap_ratio={info_tsvd.get('gap_ratio')}")

            deep_rows.append({
                "plan": plan_name, "region": label, "idx": idx, "s_mm": float(path_s_mm[idx]),
                "relerr_schedule_vs_FD": e_sched_vs_fd, "relerr_tikhonov_vs_FD": e_tik_vs_fd,
                "relerr_truncated_svd_vs_FD": e_tsvd_vs_fd, "relerr_schedule_vs_fresh_tikhonov": e_sched_vs_tik,
                "effective_rank": info_tsvd.get("effective_rank"), "n_total": info_tsvd.get("n_total"),
            })

    scan_df = pd.concat(scan_frames, ignore_index=True)
    scan_df.to_csv(f"{OUT_DIR}/mpc_schedule_scan.csv", index=False)
    deep_df = pd.DataFrame.from_records(deep_rows)
    deep_df.to_csv(f"{OUT_DIR}/mpc_schedule_fd_validation.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(deep_df[["plan", "region", "s_mm", "relerr_schedule_vs_FD", "relerr_tikhonov_vs_FD",
                    "relerr_truncated_svd_vs_FD", "relerr_schedule_vs_fresh_tikhonov",
                    "effective_rank", "n_total"]].to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    for i, plan_name in enumerate(PLANS):
        sub = scan_df[scan_df["plan"] == plan_name]
        axes[0].plot(sub["s_mm"].to_numpy(), sub["sigma1"].to_numpy(),
                     label=f"{plan_name} sigma1", lw=1.2)
    axes[0].set_ylabel("sigma1 (largest singular value)")
    axes[0].set_xlabel("path progress s (mm)")
    axes[0].set_title("Schedule scan: singular-value spectrum (no isolated spikes = healthy)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for i, plan_name in enumerate(PLANS):
        sub = scan_df[scan_df["plan"] == plan_name]
        axes[1].plot(sub["s_mm"].to_numpy(), sub["cond"].to_numpy(),
                     label=f"{plan_name} cond", lw=1.2)
    axes[1].set_ylabel("condition number sigma1/sigma3")
    axes[1].set_xlabel("path progress s (mm)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_mpc_schedule_scan.png", dpi=160)
    plt.close(fig)
    print(f"\nsaved {OUT_DIR}/mpc_schedule_scan.csv, mpc_schedule_fd_validation.csv, fig6_mpc_schedule_scan.png")


if __name__ == "__main__":
    main()
