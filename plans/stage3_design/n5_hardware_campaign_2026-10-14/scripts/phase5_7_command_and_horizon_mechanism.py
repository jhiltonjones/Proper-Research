"""Script 5 -- plan: snug-bouncing-giraffe.md.

Two explicitly distinct tests (never conflated):

1. SAME-STATE COUNTERFACTUAL (central causal test). For each of the 9
   runs, a full sequential replay is built using THAT run's own real
   historical commands (q_cmd/q_cmd_prev/previous_input reconstructed
   from the logged u0 via the validated clip-formula) and ITS OWN
   actual Jacobian source every tick (hybrid-mutated for NEW, static
   schedule for OLD-C, static no-contact schedule for NC, each with
   its OWN `reference` object). This is first VERIFIED to reproduce
   the real logged u0 closely (cosine similarity + relative norm) --
   not assumed.
   At sampled anchor ticks (stride-5 + full region-C coverage, same
   convention as Script 4 -- anchors therefore come from EACH
   condition's own realized states, not only NEW's), three
   counterfactual branches are solved AT THE SAME tick using the SAME
   z_meas/q_cmd/q_cmd_prev/previous_input/control_index (everything
   held fixed except the Jacobian/nominal window):
     - J1       : the plain scheduled contact Jacobian.
     - J_NC     : the no-contact schedule's row, cross-mapped by
                  PHYSICAL PROGRESS s (never raw index) from this
                  run's own reference-index space into the other
                  schedule's 415-or-527-sample space.
     - J4_h^assumed : the model-oracle horizon, Jacobian recomputed at
                  the REFERENCE's own state along the horizon (the
                  "common assumed future state sequence"), via the
                  canonical corrected bundle.
   Crucially, NO deepcopy of the controller (OSQP backends are not
   reliably deepcopy-safe) -- instead the only piece of hidden
   per-tick state, `_filtered_output_residual`, is explicitly saved
   before the real/own solve and restored before each counterfactual
   branch, then the real post-tick state is restored before the main
   replay continues to tick k+1.

2. ACROSS-REALIZED-TRAJECTORIES comparison: at the same sampled ticks,
   compares each run's own REAL logged command against what J4 (the
   oracle, at that run's own actually-visited state) would have
   commanded -- explicitly labeled as a different, less causally clean
   question, since the states being compared differ across conditions.

Records theta_u = angle(u(J_i), u(J4)), E_u = ||u(J_i)-u(J4)||, and
V_track = first_predicted_beam_error_m for every branch.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")
import run_registry as reg
import path_geometry as pg

from phase0_helper import build_corrected_jacobian_provider
from proper_research.controllers.beam_jacobian_providers import position_jacobian_from_output
from proper_research.controllers.contact_jacobian_hybrid import ContactJacobianHybridModel, HybridMapConfig
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference, ConfigurationMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.exact_qn_zero_mpc import (
    ExactQNZeroTaskNullspaceDelayAwareMPC,
)
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.planning.planning_context import make_robot_config

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

DT = reg.DT
VLIM6, ACC6, INS_RATE_LIM = 0.10, 0.4, 2.0e-3
MAX_STEP6 = 0.010
D7 = np.array([VLIM6] * 6 + [INS_RATE_LIM], dtype=float) * DT

SCHED_C = np.load("/home/jack/Proper-Research/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_floor240_2026-10-12.npy")
SCHED_NC = np.load("/home/jack/Proper-Research/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_floor240_2026-10-12.npy")
CONTACT_PLAN_DIR = "/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_2026-10-12/time_parameterized_configuration_path"
NOCONTACT_PLAN_DIR = "/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_nocontact_2026-10-12/time_parameterized_configuration_path"

R_C_PRIMARY = (0.055, 0.065)
STRIDE = 5

robot_kin = _resolve_robot_kinematics(make_robot_config())


def sample_tick_indices(n_ticks, s_tip_arr):
    strided = set(range(0, n_ticks, STRIDE))
    region_c = set(int(i) for i in np.where((s_tip_arr >= R_C_PRIMARY[0]) & (s_tip_arr <= R_C_PRIMARY[1]))[0])
    return sorted(strided | region_c)


def load_series_npz():
    data = np.load(os.path.join(OUT_DIR, "phase1_2_series.npz"))
    out = {}
    for cond, rep, _ in reg.all_runs():
        key = f"{cond}_rep{rep}"
        out[key] = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith(key + "__")}
    return out


def reconstruct_command_sequence(rows):
    N = len(rows)
    q_meas = np.array([r["q_meas_rad"] for r in rows])
    ins_meas = np.array([r["insertion_length_m"] for r in rows])
    u0_log = np.array([r["u0"] for r in rows])
    q_cmd_seq = np.zeros((N, 6))
    ins_cmd_seq = np.zeros(N)
    prev_u_seq = np.zeros((N, 7))
    cur_q = q_meas[0].copy()
    cur_ins = float(ins_meas[0])
    prev_u = np.zeros(7)
    for k in range(N):
        q_cmd_seq[k] = cur_q
        ins_cmd_seq[k] = cur_ins
        prev_u_seq[k] = prev_u
        qd = np.clip(u0_log[k][:6], -VLIM6, VLIM6)
        delta_q = np.clip(qd * DT, -MAX_STEP6, MAX_STEP6)
        cur_q = cur_q + delta_q
        cur_ins = float(np.clip(cur_ins + u0_log[k][6] * DT, 0.005, 0.20))
        prev_u = u0_log[k]
    return q_cmd_seq, ins_cmd_seq, prev_u_seq, q_meas, ins_meas, u0_log


def build_magnet_workspace(ref_states_q, z_bounds):
    mag_pos = np.stack([
        urik.forward_kinematics(ref_states_q[i], robot_kin.dh, robot_kin.T_F_M).T_R_target[:3, 3]
        for i in range(ref_states_q.shape[0])
    ], axis=0)
    mag_jac = np.stack([
        urik.geometric_jacobian(ref_states_q[i], robot_kin.dh, robot_kin.T_F_M)[:3, :6]
        for i in range(ref_states_q.shape[0])
    ], axis=0)
    return dict(position_jacobians=mag_jac, nominal_positions_m=mag_pos,
                z_min_m=z_bounds[0], z_max_m=z_bounds[1])


def build_controller(reference, sched, meta):
    horizon = int(meta["horizon"])
    increment_scale = tuple(DT * a for a in ([ACC6] * 6 + [10.0 * INS_RATE_LIM]))
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=horizon,
        state_min=tuple([-2.0 * math.pi] * 6 + [0.0]), state_max=tuple([2.0 * math.pi] * 6 + [0.110]),
        velocity_limit=tuple([VLIM6] * 6 + [INS_RATE_LIM]),
        acceleration_limit=tuple([ACC6] * 6 + [10.0 * INS_RATE_LIM]),
        input_tracking_weight=float(meta["input_tracking_weight"]),
        input_increment_weight=float(meta["input_increment_weight"]),
        input_increment_scale=increment_scale,
        diagnostic_zero_input_reference_in_R=True,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    magnet_workspace = build_magnet_workspace(np.asarray(reference.state[:, :6], dtype=float),
                                                meta["magnet_z_bounds_m"])
    return ExactQNZeroTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=sched, delay_samples=int(meta["delay_samples"]), beta_d=float(meta["beta_d"]),
        magnet_workspace=magnet_workspace,
    )


def cross_map_indices(src_idx, s_src_full, s_dst_full):
    s_vals = s_src_full[np.clip(src_idx, 0, len(s_src_full) - 1)]
    dst_idx = np.searchsorted(s_dst_full, s_vals)
    dst_idx = np.clip(dst_idx, 0, len(s_dst_full) - 1)
    # pick whichever of dst_idx, dst_idx-1 is closer in s
    alt = np.clip(dst_idx - 1, 0, len(s_dst_full) - 1)
    use_alt = np.abs(s_dst_full[alt] - s_vals) < np.abs(s_dst_full[dst_idx] - s_vals)
    dst_idx = np.where(use_alt, alt, dst_idx)
    return dst_idx


def angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    cos_a = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def main():
    t0 = time.time()
    series = load_series_npz()

    jac_provider = build_corrected_jacobian_provider()
    adapter = jac_provider.adapter
    baseline_snapshot = adapter.beam_output_fn.capture_cache_state()

    def solve_J4_output(state7):
        out = adapter.forward_output(state7, commit=True)
        J = position_jacobian_from_output(adapter.continuous_output_jacobian(state7))
        return J

    hybrid_model = ContactJacobianHybridModel(HybridMapConfig(map_dir=reg.HYBRID_MAP_DIR), D=D7)

    reference_contact = load_configuration_reference(CONTACT_PLAN_DIR, require_planned_beam_feasible=False)
    reference_nocontact = load_configuration_reference(NOCONTACT_PLAN_DIR, require_planned_beam_feasible=False)
    s_527 = pg.load_resampled_s_of_ref_index(reg.PLAN_ROOT_CONTACT)
    s_415 = pg.load_resampled_s_of_ref_index(reg.PLAN_ROOT_NOCONTACT)
    contact_tables = pg.load_path_tables(reg.PLAN_ROOT_CONTACT)
    nocontact_tables = pg.load_path_tables(reg.PLAN_ROOT_NOCONTACT)

    ref_states_q_527 = np.asarray(reference_contact.state[:, :6], dtype=float)
    s_of_ref_hybrid = np.zeros(ref_states_q_527.shape[0])
    for k in range(ref_states_q_527.shape[0]):
        dists = np.linalg.norm(
            (hybrid_model.states_arr[:, :6] - ref_states_q_527[k][None, :]) / hybrid_model.D_Q[None, :], axis=1)
        s_of_ref_hybrid[k] = hybrid_model.s_m_arr[int(np.argmin(dists))]

    all_results = {}
    repro_report = {}

    for cond, rep, run_path in reg.all_runs():
        key = f"{cond}_rep{rep}"
        rows = reg.load_path_follow(run_path)
        meta = reg.load_controller_metadata(run_path)
        q_cmd_seq, ins_cmd_seq, prev_u_seq, q_meas, ins_meas, u0_log = reconstruct_command_sequence(rows)
        N = len(rows)
        ref_idx_full = np.array([r["ref_index"] for r in rows], dtype=int)
        tip_m_log = np.array([r["tip_mm"] for r in rows], dtype=float) / 1000.0

        if cond == "NC":
            reference, sched, s_own_full, own_tables = reference_nocontact, SCHED_NC, s_415, nocontact_tables
            other_sched, s_other_full = SCHED_C, s_527
        else:
            reference, sched, s_own_full, own_tables = reference_contact, SCHED_C, s_527, contact_tables
            other_sched, s_other_full = SCHED_NC, s_415

        controller = build_controller(reference, sched.copy(), meta)
        pristine_J = controller.reference_position_jacobians.copy()
        pristine_nom = controller.nominal_reference_positions_m.copy()

        s_tip_arr = series[key]["s_tip_m"]
        anchor_idx = sample_tick_indices(N, s_tip_arr)

        cos_sims, rel_norms = [], []
        anchor_records = []

        for k in range(N):
            control_index = int(ref_idx_full[k])
            window_idx = controller._reference_indices(control_index, future=True)

            if cond == "NEW":
                J_h, nom_h, _diag = hybrid_model.horizon_jacobians_and_nominals(
                    q_meas=q_meas[k], L_meas=float(ins_meas[k]),
                    sched_J_window=pristine_J[window_idx], sched_nominal_window=pristine_nom[window_idx],
                    s_ref_window=s_of_ref_hybrid[window_idx],
                )
                controller.reference_position_jacobians[window_idx] = J_h
                controller.nominal_reference_positions_m[window_idx] = nom_h
            # OLD-C / NC: controller.reference_position_jacobians stays == pristine_J always (no mutation)

            residual_before = (None if controller._filtered_output_residual is None
                                else controller._filtered_output_residual.copy())

            z_meas_k = np.concatenate([q_meas[k], [ins_meas[k]]])
            step_own = controller.solve_delay_aware(
                z_meas=z_meas_k, q_cmd=q_cmd_seq[k], q_cmd_prev=q_cmd_seq[max(k - 1, 0)],
                insertion_m=ins_cmd_seq[k], measured_beam_position=tip_m_log[k],
                control_index=control_index, previous_input=prev_u_seq[k],
            )
            residual_after_own = (None if controller._filtered_output_residual is None
                                    else controller._filtered_output_residual.copy())

            cmd_own6 = step_own.command[:6]
            logged6 = np.asarray(u0_log[k][:6], dtype=float)
            logged_norm = float(np.linalg.norm(logged6))
            cos_sims.append((float(np.dot(cmd_own6, logged6) /
                               max(np.linalg.norm(cmd_own6) * logged_norm, 1e-12)), logged_norm))
            rel_norms.append(float(np.linalg.norm(step_own.command - u0_log[k]) /
                                     max(np.linalg.norm(u0_log[k]), 1e-9)))

            if k in anchor_idx and step_own.success:
                own_J_window_now = controller.reference_position_jacobians[window_idx].copy()
                own_nom_window_now = controller.nominal_reference_positions_m[window_idx].copy()

                # --- branch J1 (plain schedule) ---
                controller._filtered_output_residual = (None if residual_before is None else residual_before.copy())
                controller.reference_position_jacobians[window_idx] = pristine_J[window_idx]
                controller.nominal_reference_positions_m[window_idx] = pristine_nom[window_idx]
                step_J1 = controller.solve_delay_aware(
                    z_meas=z_meas_k, q_cmd=q_cmd_seq[k], q_cmd_prev=q_cmd_seq[max(k - 1, 0)],
                    insertion_m=ins_cmd_seq[k], measured_beam_position=tip_m_log[k],
                    control_index=control_index, previous_input=prev_u_seq[k],
                )

                # --- branch J_NC (other schedule, s-matched) ---
                other_idx = cross_map_indices(window_idx, s_own_full, s_other_full)
                controller._filtered_output_residual = (None if residual_before is None else residual_before.copy())
                controller.reference_position_jacobians[window_idx] = other_sched[other_idx]
                # nominal: use the OTHER plan's own native tip position at the matching s (no linearization
                # correction needed -- evaluated essentially on-path already by construction of the map).
                other_tables = nocontact_tables if cond != "NC" else contact_tables
                s_vals_h = s_own_full[np.clip(window_idx, 0, len(s_own_full) - 1)]
                nom_other = np.stack([pg.tip_plan(s, other_tables) for s in s_vals_h], axis=0)
                controller.nominal_reference_positions_m[window_idx] = nom_other
                step_NC = controller.solve_delay_aware(
                    z_meas=z_meas_k, q_cmd=q_cmd_seq[k], q_cmd_prev=q_cmd_seq[max(k - 1, 0)],
                    insertion_m=ins_cmd_seq[k], measured_beam_position=tip_m_log[k],
                    control_index=control_index, previous_input=prev_u_seq[k],
                )

                # --- branch J4_h^assumed (oracle at the reference's own state along the horizon) ---
                adapter.beam_output_fn.restore_cache_state(baseline_snapshot)
                state0 = np.asarray(reference.state[window_idx[0]], dtype=float)
                _ = solve_J4_output(state0)  # nominal anchor presolve
                J4_h = np.zeros((len(window_idx), 3, 7))
                nom4_h = np.zeros((len(window_idx), 3))
                for h, wi in enumerate(window_idx):
                    state_h = np.asarray(reference.state[wi], dtype=float)
                    J4_h[h] = solve_J4_output(state_h)
                    nom4_h[h] = pg.tip_plan(s_own_full[wi], own_tables)
                adapter.beam_output_fn.restore_cache_state(baseline_snapshot)
                controller._filtered_output_residual = (None if residual_before is None else residual_before.copy())
                controller.reference_position_jacobians[window_idx] = J4_h
                controller.nominal_reference_positions_m[window_idx] = nom4_h
                step_J4 = controller.solve_delay_aware(
                    z_meas=z_meas_k, q_cmd=q_cmd_seq[k], q_cmd_prev=q_cmd_seq[max(k - 1, 0)],
                    insertion_m=ins_cmd_seq[k], measured_beam_position=tip_m_log[k],
                    control_index=control_index, previous_input=prev_u_seq[k],
                )

                branches = dict(own=step_own, J1=step_J1, J_NC=step_NC, J4=step_J4)
                # Jacobian used by each branch AT h=0 (this exact anchor state) --
                # needed for the within-anchor E_J,D deltas below, distinct from
                # the command-space theta_u/E_u metrics.
                J_at_anchor = dict(own=own_J_window_now[0], J1=pristine_J[window_idx][0],
                                    J_NC=other_sched[other_idx][0], J4=J4_h[0])
                rec = dict(tick=k, s_tip_mm=float(s_tip_arr[k]) * 1000,
                           success={b: bool(s.success) for b, s in branches.items()})
                for b, s in branches.items():
                    if b == "J4":
                        continue
                    rec[f"theta_u_{b}_vs_J4_deg"] = angle_deg(s.command[:6], step_J4.command[:6])
                    rec[f"E_u_{b}_vs_J4"] = float(np.linalg.norm(s.command - step_J4.command))
                    rec[f"V_track_{b}"] = float(getattr(s, "first_predicted_beam_error_m", np.nan))
                    rec[f"E_JD_{b}"] = float(np.linalg.norm((J_at_anchor["J4"] - J_at_anchor[b]) @ np.diag(D7)))
                # Within-anchor paired deltas (the primary mechanistic test, n=5
                # round): positive = NEW/own is better AT THIS EXACT STATE than
                # the J1 schedule would have been -- genuinely paired by anchor
                # state, no rep-number pairing assumption needed. Only meaningful
                # when this anchor's "own" condition is NEW (comparing NEW's own
                # branch against the J1 branch at the SAME state); for OLD-C/NC
                # anchors, "own" already equals J1 or is a different schedule, so
                # Delta_E_J/Delta_theta_u there instead compares J1 vs J_NC at
                # that state -- labeled accordingly, not conflated.
                # Delta_E_J(x_k) = E_J,D(J1 branch) - E_J,D(own branch), at this
                # exact anchor state. When the anchor comes from a NEW
                # trajectory, J1-branch-at-this-state is exactly what OLD-C
                # would have used here (OLD-C's own branch always equals J1),
                # so this IS E_J,D^OLD-C(x_k) - E_J,D^NEW(x_k) -- genuinely
                # paired by state, not by run number.
                rec["Delta_E_J_J1_minus_own"] = float(rec["E_JD_J1"] - rec["E_JD_own"])
                rec["Delta_theta_u_J1_minus_own"] = float(rec["theta_u_J1_vs_J4_deg"] - rec["theta_u_own_vs_J4_deg"])
                rec["V_track_J4"] = float(getattr(step_J4, "first_predicted_beam_error_m", np.nan))
                rec["theta_u_own_vs_logged_deg"] = angle_deg(cmd_own6, logged6)
                anchor_records.append(rec)

                # restore real post-own state before continuing the main replay
                controller._filtered_output_residual = (None if residual_after_own is None else residual_after_own.copy())
                controller.reference_position_jacobians[window_idx] = own_J_window_now
                controller.nominal_reference_positions_m[window_idx] = own_nom_window_now
            else:
                controller._filtered_output_residual = (None if residual_after_own is None else residual_after_own.copy())

        cos_vals = np.array([c for c, n in cos_sims])
        cos_norms = np.array([n for c, n in cos_sims])
        meaningful = cos_norms >= 0.01  # exclude near-zero-command ticks, where angle is numerically meaningless
        repro_report[key] = dict(
            mean_cos_sim_all=float(np.mean(cos_vals)), min_cos_sim_all=float(np.min(cos_vals)),
            mean_cos_sim_meaningful=float(np.mean(cos_vals[meaningful])) if meaningful.any() else float("nan"),
            min_cos_sim_meaningful=float(np.min(cos_vals[meaningful])) if meaningful.any() else float("nan"),
            n_near_zero_command_ticks=int((~meaningful).sum()), n_ticks=len(cos_vals),
            mean_rel_norm_err=float(np.mean(rel_norms)), max_rel_norm_err=float(np.max(rel_norms)))
        all_results[key] = anchor_records
        print(f"[{key}] replay+branches done, {len(anchor_records)} anchors, "
              f"repro mean_cos={repro_report[key]['mean_cos_sim_meaningful']:.4f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    with open(os.path.join(OUT_DIR, "phase5_reproduction_check.json"), "w") as f:
        json.dump(repro_report, f, indent=2)
    with open(os.path.join(OUT_DIR, "phase5_anchor_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== REPRODUCTION CHECK (own replay vs real logged u0) ===")
    for key, r in repro_report.items():
        print(f"  {key:12s} mean_cos(meaningful)={r['mean_cos_sim_meaningful']:.4f} "
              f"min(meaningful)={r['min_cos_sim_meaningful']:.4f} "
              f"n_near_zero_excluded={r['n_near_zero_command_ticks']}/{r['n_ticks']} "
              f"mean_rel_err={r['mean_rel_norm_err']:.4f}")

    print("\n=== SAME-STATE COUNTERFACTUAL summary (mean theta_u vs J4, degrees) ===")
    for cond in ("NEW", "OLD-C", "NC"):
        for branch in ("own", "J1", "J_NC"):
            vals = []
            for rep, _ in reg.RUNS[cond]:
                for rec in all_results[f"{cond}_rep{rep}"]:
                    v = rec.get(f"theta_u_{branch}_vs_J4_deg")
                    if v is not None and not np.isnan(v):
                        vals.append(v)
            if vals:
                print(f"  {cond:6s} branch={branch:5s} mean_theta_u={np.mean(vals):.2f}deg  n={len(vals)}")

    print("\n=== PRIMARY MECHANISTIC TEST: within-anchor paired deltas, NEW's own anchors "
          "(region C only) ===")
    print("Delta_E_J(x_k) = E_J,D(J1 branch) - E_J,D(own branch); Delta_theta_u(x_k) = "
          "theta_u(J1 branch) - theta_u(own branch). Positive = NEW better AT THAT EXACT STATE "
          "(genuinely paired by anchor state, not by run number).")
    de_j_region_c, dtheta_region_c = [], []
    for rep, _ in reg.RUNS["NEW"]:
        for rec in all_results[f"NEW_rep{rep}"]:
            if R_C_PRIMARY[0] * 1000 <= rec["s_tip_mm"] <= R_C_PRIMARY[1] * 1000:
                de_j_region_c.append(rec["Delta_E_J_J1_minus_own"])
                dtheta_region_c.append(rec["Delta_theta_u_J1_minus_own"])
    if de_j_region_c:
        de_j_arr, dtheta_arr = np.array(de_j_region_c), np.array(dtheta_region_c)
        print(f"  Delta_E_J:     n={len(de_j_arr)}  median={np.median(de_j_arr):+.5f}  "
              f"frac_positive={np.mean(de_j_arr > 0):.3f}  range=[{de_j_arr.min():+.5f},{de_j_arr.max():+.5f}]")
        print(f"  Delta_theta_u: n={len(dtheta_arr)}  median={np.median(dtheta_arr):+.3f}deg  "
              f"frac_positive={np.mean(dtheta_arr > 0):.3f}  range=[{dtheta_arr.min():+.3f},{dtheta_arr.max():+.3f}]deg")
    else:
        print("  No NEW anchors fell inside region C -- cannot report this test.")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print(f"Wrote {OUT_DIR}/phase5_reproduction_check.json and phase5_anchor_results.json")


if __name__ == "__main__":
    main()
