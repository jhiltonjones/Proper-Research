#!/usr/bin/env python3
"""Study 1 -- the nullspace (Q_N) ablation: does the reference-nullspace
state-tracking cost do anything beyond what the full velocity reference
(u_ref, including its own redundant/nullspace component) already provides?

Reproduces CONTROLLER_JUSTIFICATION.md sec 6.4 / qn_ablation.py's own
docstring, but as a clean, runnable, plot-producing script against a REAL
logged closed-loop run instead of a one-off interactive session.

Condition A: StagewiseTaskNullspaceDelayAwareMPC, gamma=0 -- the FROZEN,
             live-validated controller. Q_N = P_N (exact task-nullspace
             state cost) is ON.
Condition B: qn_ablation.QZeroMPC -- identical in every other respect, but
             the ENTIRE state cost is zeroed (Q=0 exactly). u_ref stays
             full (unchanged) in both conditions.

At every tick of the real log, both controllers are re-solved against the
SAME actual recorded (z_meas, q_cmd, q_cmd_prev, previous_input) --
"single-tick counterfactual replay", so trajectories never diverge/compound.
For each we compute:
  - E_chosen at horizon steps +3/+5/+10 (predicted-vs-desired beam position
    error within the QP's own horizon -- the open-loop task-tracking proxy)
  - e_N(k) = ||P_N,j @ S_z^-1 (z_j|k - z_ref,k+j)||, the nullspace-projected
    predicted configuration deviation, at the same three horizon depths --
    THIS is the quantity that actually distinguishes the two conditions.

Usage
-----
    python -m proper_research.studies.mpc_ablations.01_nullspace_qn_ablation \\
        --log-dir close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proper_research.controllers.mpc_delay_aware.qn_ablation import QZeroMPC
from proper_research.controllers.mpc_delay_aware.stagewise_task_nullspace import (
    StagewiseTaskNullspaceDelayAwareMPC, build_stagewise_projectors,
)
from proper_research.controllers.mpc_delay_aware.target_consistent import (
    build_beam_plane_projection,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import replay_common as rc

HORIZON_CHECKPOINTS = (3, 5, 10)  # +3/+5/+10, matching E_chosen/e_N convention throughout this project


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=rc.DEFAULT_LOG_DIR)
    p.add_argument("--schedule-cache", default=rc.DEFAULT_SCHEDULE_CACHE)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--input-tracking-weight", type=float, default=7.0, help="R700 by default")
    args = p.parse_args()

    print(f"[qn_ablation] real log: {args.log_dir}")
    rows, meta = rc.load_real_log(args.log_dir)
    plan_dir = meta.get("plan_dir", "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path")
    print(f"[qn_ablation] plan_dir: {plan_dir}  ({len(rows)} ticks in the log)")

    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    dt = float(reference.sample_period_s)
    schedule = rc.build_schedule(plan_dir, args.schedule_cache)

    mpc_config = rc.build_mpc_config(dt=dt, horizon=args.horizon, input_tracking_weight=args.input_tracking_weight)
    beam_config = rc.build_beam_config()

    ctrl_A = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    ctrl_B = QZeroMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )

    C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
    S_z = np.asarray(mpc_config.state_error_scale, dtype=float)

    ticks = rc.pick_tick_indices(len(rows))
    print(f"[qn_ablation] replaying {len(ticks)} ticks (of {len(rows)}) under both conditions...")

    results = {"A": {h: [] for h in HORIZON_CHECKPOINTS}, "B": {h: [] for h in HORIZON_CHECKPOINTS}}
    e_n_results = {"A": {h: [] for h in HORIZON_CHECKPOINTS}, "B": {h: [] for h in HORIZON_CHECKPOINTS}}
    rms_joint_cmd = {"A": [], "B": []}

    for i in ticks:
        inputs = rc.reconstruct_solve_inputs(rows, i)
        state_ref_window = reference.state_window(inputs["control_index"], args.horizon)
        indices = np.clip(inputs["control_index"] + 1 + np.arange(args.horizon), 0, reference.sample_count - 1)
        J_window = np.asarray(schedule, dtype=float)[indices]
        P_R, P_N, ranks = build_stagewise_projectors(J_schedule=J_window, C=C, state_error_scale=S_z)

        for label, ctrl in (("A", ctrl_A), ("B", ctrl_B)):
            step = ctrl.solve_delay_aware(**inputs)
            rms_joint_cmd[label].append(float(np.sqrt(np.mean(np.asarray(step.command[:6]) ** 2))))
            for h in HORIZON_CHECKPOINTS:
                stage = h - 1  # +3 -> predicted_beam_errors[2], etc.
                if stage < step.predicted_beam_errors.shape[0]:
                    results[label][h].append(float(np.linalg.norm(step.predicted_beam_errors[stage])) * 1e3)
                    z_dev = (step.predicted_states[stage] - state_ref_window[stage]) / S_z
                    e_n_results[label][h].append(float(np.linalg.norm(P_N[stage] @ z_dev)))

    print("\n[qn_ablation] E_chosen (mm), mean over replayed ticks:")
    for h in HORIZON_CHECKPOINTS:
        a, b = np.mean(results["A"][h]), np.mean(results["B"][h])
        print(f"  +{h:>2}: A(Q_N on)={a:.3f}  B(Q_N off)={b:.3f}  delta={b-a:+.3f}")

    print("\n[qn_ablation] e_N (nullspace-projected predicted deviation), mean over replayed ticks:")
    for h in HORIZON_CHECKPOINTS:
        a, b = np.mean(e_n_results["A"][h]), np.mean(e_n_results["B"][h])
        ratio = b / max(a, 1e-9)
        print(f"  +{h:>2}: A(Q_N on)={a:.4f}  B(Q_N off)={b:.4f}  ratio={ratio:.2f}x")

    print(f"\n[qn_ablation] RMS joint command (rad/s): A={np.mean(rms_joint_cmd['A']):.4f}  "
          f"B={np.mean(rms_joint_cmd['B']):.4f}")

    out = rc.ensure_outdir("study01_nullspace_qn_ablation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(HORIZON_CHECKPOINTS))
    width = 0.35
    axes[0].bar(x - width / 2, [np.mean(results["A"][h]) for h in HORIZON_CHECKPOINTS], width, label="A: Q_N on (frozen)")
    axes[0].bar(x + width / 2, [np.mean(results["B"][h]) for h in HORIZON_CHECKPOINTS], width, label="B: Q_N off")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"+{h}" for h in HORIZON_CHECKPOINTS])
    axes[0].set_ylabel("E_chosen (mm)"); axes[0].set_title("Task tracking: barely changes"); axes[0].legend()

    axes[1].bar(x - width / 2, [np.mean(e_n_results["A"][h]) for h in HORIZON_CHECKPOINTS], width, label="A: Q_N on (frozen)")
    axes[1].bar(x + width / 2, [np.mean(e_n_results["B"][h]) for h in HORIZON_CHECKPOINTS], width, label="B: Q_N off")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"+{h}" for h in HORIZON_CHECKPOINTS])
    axes[1].set_ylabel("e_N (normalized nullspace deviation)")
    axes[1].set_title("Predicted posture drift: grows without Q_N"); axes[1].legend()
    fig.suptitle(f"Q_N ablation, replayed on real log ({args.log_dir.split('/')[-1]})")
    fig.tight_layout()
    fig.savefig(out / "qn_ablation.png", dpi=150)
    print(f"\n[qn_ablation] plot -> {out / 'qn_ablation.png'}")


if __name__ == "__main__":
    main()
