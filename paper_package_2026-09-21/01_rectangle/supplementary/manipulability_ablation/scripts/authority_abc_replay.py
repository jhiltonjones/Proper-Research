"""Strict A/B/C replay (spec sections 14-16), single-tick counterfactual
(same style as insertion_limit_sweep.py earlier this session): at each real
historical tick from 5 real MPC-SJ rectangle live runs, re-solve the QP
under three secondary-objective regimes using the ACTUAL recorded
(x_exec, previous_input, measured_beam_position) -- isolates each mode's
one-step decision, does not let trajectories diverge/compound.

  A = reference_nullspace (gamma=0)   -- validated baseline, bit-identical
  B = none                             -- essential per reviewer: isolates
                                          "removing the anchor helped" from
                                          "manipulability helped"
  C = manipulability, lambda in {1,3,10,30}  -- weak/medium/strong/aggressive
                                          per the weight pilot just run

Per spec section 16 guardrail instrumentation, computed on EVERY solve:
  ||P_N * dz~_1||        (secondary-direction departure at first predicted stage)
  phi_linear_predicted   (g_N^T dz~_1)
  phi_true(z_predicted)  (real logdet authority at the predicted first stage)
  dist_state, dist_vel   (distance to nearest bound)
  E_chosen at +3/+5/+10  (mm, task tracking of the PREDICTION, not closed-loop)
  control effort          (||u_joints||, |u_insertion|)
  config departure        (||z1_pred - z_ref||/S_z)

phi_true requires the expensive state-dependent beam Jacobian (~0.22s/call)
so it is evaluated on a subsample of ticks (every 8th) across all 5 runs;
all other (cheap, QP-only) metrics are computed on every subsampled tick too
for consistency -- there is no cheap-metric benefit to a finer subsample
since the QP solve itself dominates cost far less than phi_true.
"""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/3710eca5/tmp")
import json
import math
import pickle
import time
from dataclasses import replace as dc_replace
import numpy as np

from proper_research.planning.planning_context import build_planning_context
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.controllers.mpc_delay_aware.target_consistent import build_beam_plane_projection
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import BeamOutputMPCConfig
from proper_research.controllers.mpc_delay_aware.authority_nullspace_mpc import AuthorityShapedDelayAwareMPC
from authority_objective import build_Jbar, phi_authority, nullspace_projector

PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCHEDULE_PATH = "/tmp/rectangle_genuine_ltv_schedule.npy"
N, DT = 15, 0.1
N_REF = 265
S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
S_P = np.array([0.5e-3, 0.5e-3])
D_U = np.array([0.10] * 6 + [2.0e-3])
S_U = np.array([0.05] * 6 + [5.0e-3])
C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))

LAMBDAS = [1.0, 3.0, 10.0, 30.0]
MODES = ["A_reference_nullspace", "B_none"] + [f"C_manip_lam{lam:g}" for lam in LAMBDAS]

MPC_SJ_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112405Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112843Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114154Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114458Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114842Z",
]
SUBSAMPLE_STRIDE = 8

reference_raw = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
schedule = np.load(SCHEDULE_PATH)
z_ref_all = np.asarray(reference_raw.state, dtype=float)

with open("/home/jack/.claude/jobs/3710eca5/tmp/authority_phase0_schedule.pkl", "rb") as f:
    phase0 = pickle.load(f)
authority_schedule = {"g_N": phase0["g_N"]}

meta = json.load(open(f"/home/jack/Proper-Research/{MPC_SJ_RUNS[0]}/controller_metadata.json"))
R_fit = np.asarray(meta["planner_to_live_R_fit"], dtype=float)
t_fit = np.asarray(meta["planner_to_live_t_fit_m"], dtype=float)

print("[setup] building state-dependent Jacobian provider (for phi_true only)...")
_, bundle, controller_pack, _ = build_planning_context()
jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
print("[setup] done.\n")


def phi_true(z):
    J = C @ jac_provider(z)
    Jbar = build_Jbar(J, S_P, D_U, DT)
    p, _ = phi_authority(Jbar)
    return p


RD_SCALE = tuple([DT * 0.40] * 6 + [DT * 10.0 * 2.0e-3])
mpc_config = ConfigurationMPCConfig(
    sample_period_s=DT, prediction_horizon=N,
    state_min=tuple([-2 * math.pi] * 6 + [-0.05]), state_max=tuple([2 * math.pi] * 6 + [0.20]),
    velocity_limit=tuple([0.10] * 6 + [2.0e-3]), acceleration_limit=tuple([0.40] * 6 + [0.02]),
    input_increment_scale=RD_SCALE, input_tracking_weight=1.0e-2 * 700.0,
)
beam_config = BeamOutputMPCConfig(
    position_error_scale_m=(5.0e-4,) * 3, position_tracking_weight=1.0,
    use_dare_terminal_cost=False, directional_damping=0.0,
)
STATE_MIN = np.array(mpc_config.state_min); STATE_MAX = np.array(mpc_config.state_max)
VLIM = np.array(mpc_config.velocity_limit)


def apply_frame_transform(controller):
    des_planner = np.asarray(controller.reference.desired_position_m, dtype=float)
    des_R = des_planner @ R_fit.T + t_fit
    kwargs = {"desired_position_m": des_R}
    tan = getattr(controller.reference, "desired_tangent", None)
    if tan is not None:
        kwargs["desired_tangent"] = np.asarray(tan, dtype=float) @ R_fit.T
    controller.reference = dc_replace(controller.reference, **kwargs)
    controller.nominal_reference_positions_m = des_R.copy()
    return des_R


def make_controller(mode):
    common = dict(
        reference=load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False),
        config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    if mode == "A_reference_nullspace":
        ctrl = AuthorityShapedDelayAwareMPC(secondary_objective="reference_nullspace", gamma=0.0, **common)
    elif mode == "B_none":
        ctrl = AuthorityShapedDelayAwareMPC(secondary_objective="none", **common)
    else:
        lam = float(mode.split("lam")[1])
        ctrl = AuthorityShapedDelayAwareMPC(
            secondary_objective="manipulability", manipulability_weight=lam,
            authority_schedule=authority_schedule, **common,
        )
    des_R = apply_frame_transform(ctrl)
    return ctrl, des_R


print("[setup] instantiating one controller per mode (reused across all ticks)...")
controllers = {}
des_positions = {}
for mode in MODES:
    ctrl, des_R = make_controller(mode)
    controllers[mode] = ctrl
    des_positions[mode] = des_R  # identical across modes (same reference+frame fit), kept per-mode for clarity
print(f"[setup] {len(controllers)} controllers ready: {MODES}\n")

results = []
t_start = time.time()
n_phi_evals = 0
for run_idx, run_dir in enumerate(MPC_SJ_RUNS):
    rows = [json.loads(l) for l in open(f"/home/jack/Proper-Research/{run_dir}/path_follow.jsonl")]
    pred_rows = [json.loads(l) for l in open(f"/home/jack/Proper-Research/{run_dir}/predicted_beam_positions.jsonl")]
    n_ticks = min(len(rows), len(pred_rows))
    tick_indices = list(range(0, n_ticks, SUBSAMPLE_STRIDE))
    print(f"[run {run_idx+1}/{len(MPC_SJ_RUNS)}] {run_dir.split('/')[-1]}: {n_ticks} ticks, "
          f"{len(tick_indices)} subsampled (stride={SUBSAMPLE_STRIDE})")

    for k in tick_indices:
        r, pr = rows[k], pred_rows[k]
        q_meas = np.asarray(r["q_meas_rad"], dtype=float)
        insertion_cmd = float(pr["insertion_cmd_m"])
        z_meas = np.concatenate([q_meas, [insertion_cmd]])
        tip_m = np.asarray(r["tip_mm"], dtype=float) * 1e-3
        ref_index = int(r["ref_index"])
        q_cmd_k = np.asarray(pr["q_cmd_k"], dtype=float)
        q_cmd_km1 = np.asarray(pr["q_cmd_km1"], dtype=float)
        u_prev = np.asarray(pr["u_prev"], dtype=float)
        kwargs = dict(z_meas=z_meas, q_cmd=q_cmd_k, q_cmd_prev=q_cmd_km1, insertion_m=insertion_cmd,
                      measured_beam_position=tip_m, control_index=ref_index, previous_input=u_prev)

        idx_lin1 = min(ref_index + 1, N_REF - 1)
        z_lin1 = z_ref_all[idx_lin1]
        phi_lin1 = phase0["phi"][idx_lin1]
        g_N_lin1 = phase0["g_N"][idx_lin1]
        J_task_lin1 = C @ schedule[idx_lin1]
        _, P_N1, _ = nullspace_projector(J_task_lin1, S_Z)

        do_phi = (k // SUBSAMPLE_STRIDE) % 1 == 0  # every subsampled tick (already sparse)

        tick_result = dict(run=run_idx, tick=k, ref_index=ref_index,
                            phi0=phase0["phi"][ref_index], norm_gN0=float(np.linalg.norm(phase0["g_N"][ref_index])))
        per_mode = {}
        for mode in MODES:
            ctrl = controllers[mode]
            step = ctrl.solve_delay_aware(**kwargs)
            u0 = step.command
            z1_pred = step.predicted_states[0]
            dz_tilde1 = (z1_pred - z_lin1) / S_Z
            PN_dz1 = float(np.linalg.norm(P_N1 @ dz_tilde1))
            dphi_lin = float(g_N_lin1 @ dz_tilde1)

            phi_true1 = None
            dphi_true = None
            r_phi = None
            if do_phi:
                phi_true1 = phi_true(z1_pred)
                dphi_true = phi_true1 - phi_lin1
                r_phi = dphi_true / dphi_lin if abs(dphi_lin) > 1e-6 else float("nan")
                n_phi_evals += 1

            e_chosen = {}
            for h in (3, 5, 10):
                stage = min(h - 1, len(step.predicted_beam_positions) - 1)
                idx_h = min(ref_index + h, N_REF - 1)
                p_hat = step.predicted_beam_positions[stage]
                des_h = des_positions[mode][idx_h]
                e_chosen[h] = float(np.linalg.norm(p_hat - des_h)) * 1e3

            dist_state = float(np.min(np.minimum(z1_pred - STATE_MIN, STATE_MAX - z1_pred)))
            dist_vel = float(np.min(VLIM - np.abs(u0)))
            config_departure = float(np.linalg.norm((z1_pred - z_lin1) / S_Z))

            per_mode[mode] = dict(
                u0=u0.copy(), effort_joints=float(np.linalg.norm(u0[:6]) / np.linalg.norm(D_U[:6])),
                effort_insertion=float(abs(u0[6]) / D_U[6]),
                PN_dz1=PN_dz1, dphi_lin=dphi_lin, phi_true1=phi_true1, dphi_true=dphi_true, r_phi=r_phi,
                e_chosen=e_chosen, dist_state=dist_state, dist_vel=dist_vel, config_departure=config_departure,
            )
        tick_result["per_mode"] = per_mode
        results.append(tick_result)

    elapsed = time.time() - t_start
    print(f"  ... done run {run_idx+1}, cumulative elapsed={elapsed:.0f}s, phi_true evals so far={n_phi_evals}")

with open("/home/jack/.claude/jobs/3710eca5/tmp/authority_abc_replay.pkl", "wb") as f:
    pickle.dump(dict(results=results, modes=MODES, lambdas=LAMBDAS), f)

print(f"\nTOTAL: {len(results)} ticks x {len(MODES)} modes = {len(results)*len(MODES)} solves, "
      f"{n_phi_evals} phi_true evals, elapsed={time.time()-t_start:.0f}s")
print("saved authority_abc_replay.pkl")
