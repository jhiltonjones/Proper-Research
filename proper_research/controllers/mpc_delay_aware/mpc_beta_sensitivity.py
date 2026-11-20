"""Disturbance-persistence (beta_d) sensitivity study: beta_d=1.0 vs
beta_d=0.82, replayed against a real 262-tick live MPC-LTV run at every
tick, both under the real hard constraints and under FREE-BOTH (velocity
and acceleration limits both relaxed to 1e3) to isolate the unconstrained
optimum from constraint saturation.

This is the exact script behind CONTROLLER_JUSTIFICATION.md Sec. 3 and
SUPPLEMENTARY_EVIDENCE.md Sec. S11 (rho_desire, RMS||u0-u_ref||, S_beta,
Delta_p_beta). It was originally a job-scratch script and is committed here
verbatim for provenance/reproducibility; PLAN_DIR and SCHEDULE_PATH below
point at the specific bimaterial-rectangle plan and genuine-LTV schedule
used for that run and are not guaranteed to still exist -- regenerate them
via plan_shape_path.py / mpc_variants.precompute_schedule if re-running.
MPC_RUN_DIR is a committed log under close_loop_logs/ and should still be
present.

rho_desire (Sec. S11.1, "FREE-BOTH unconstrained-optimum aggressiveness"):
per tick k, rho_desire_k = max_i |u_free_k,i - u_prev_k,i| / delta_u_max_i,
where u_free_k is the first-stage command from the SAME QP solved with
velocity/acceleration limits relaxed (FREE-BOTH), and delta_u_max_i is the
real hard per-tick increment limit for channel i. P(rho_desire>1) is the
fraction of ticks where the true unconstrained optimum already exceeds the
real limit -- i.e. how often the constraint is genuinely binding rather
than just formally present.
"""
import json, math, time, pickle
from dataclasses import replace
import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.delay_aware_mpc import DelayAwareBeamOutputTrackingMPC

PLAN_DIR = "/home/jack/.claude/jobs/3710eca5/tmp/plans_bimaterial/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path"
SCHEDULE_PATH = "/home/jack/.claude/jobs/3710eca5/tmp/mpc_genuine_ltv_schedule.npy"
MPC_RUN_DIR = "close_loop_logs/rectangle_mpc_ltv_accumC_2026-09-17/mpc_ltv_accumC_rep2_genuineLTV_20260917T181546Z"

N, DT = 15, 0.1
VLIM_JOINTS, INSERTION_RATE_LIMIT, ACCEL_JOINTS = 0.10, 2.0e-3, 0.40
DELTA_U_MAX = np.array([ACCEL_JOINTS*DT]*6 + [10.0*INSERTION_RATE_LIMIT*DT])
VLIM = np.array([VLIM_JOINTS]*6 + [INSERTION_RATE_LIMIT])
TOL = 1e-6

reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
schedule = np.load(SCHEDULE_PATH)
reference_input = np.asarray(reference.input, dtype=float)

vlim = tuple([VLIM_JOINTS]*6 + [INSERTION_RATE_LIMIT])
alim = tuple([ACCEL_JOINTS]*6 + [10.0*INSERTION_RATE_LIMIT])
mpc_config = ConfigurationMPCConfig(
    sample_period_s=DT, prediction_horizon=N,
    state_min=tuple([-2*math.pi]*6+[-0.05]), state_max=tuple([2*math.pi]*6+[0.20]),
    velocity_limit=vlim, acceleration_limit=alim,
)
beam_config = BeamOutputMPCConfig(
    position_error_scale_m=(5.0e-4,5.0e-4,5.0e-4), position_tracking_weight=1.0,
    use_dare_terminal_cost=False, directional_damping=0.0, wall_avoidance_gain=0.0,
)
vlim_free = tuple([1.0e3]*7)
alim_free = tuple([1.0e3]*7)
mpc_config_free = replace(mpc_config, velocity_limit=vlim_free, acceleration_limit=alim_free)

print("[beta sensitivity] building controllers (beta_d in {1.0, 0.82}, constrained + FREE-BOTH)...")
c_b1  = DelayAwareBeamOutputTrackingMPC(reference=reference, config=mpc_config, beam_config=beam_config,
                                         reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0)
c_b82 = DelayAwareBeamOutputTrackingMPC(reference=reference, config=mpc_config, beam_config=beam_config,
                                         reference_position_jacobians=schedule, delay_samples=2, beta_d=0.82)
f_b1  = DelayAwareBeamOutputTrackingMPC(reference=reference, config=mpc_config_free, beam_config=beam_config,
                                         reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0)
f_b82 = DelayAwareBeamOutputTrackingMPC(reference=reference, config=mpc_config_free, beam_config=beam_config,
                                         reference_position_jacobians=schedule, delay_samples=2, beta_d=0.82)
print("[beta sensitivity] done")

rows = [json.loads(l) for l in open(f"{MPC_RUN_DIR}/path_follow.jsonl")]
K = len(rows)
q_meas = [np.asarray(r["q_meas_rad"], dtype=float) for r in rows]
tip_m = [np.asarray(r["tip_mm"], dtype=float)*1e-3 for r in rows]
ref_index = [int(r["ref_index"]) for r in rows]
u0_logged = [np.asarray(r["u0"], dtype=float) for r in rows]
prev_input_seq = [np.zeros(7)] + u0_logged[:-1]
insertion_at_solve = [0.025] + [float(r["insertion_length_m"]) for r in rows[:-1]]
q_target_rad = [np.asarray(r["q_target_rad"], dtype=float) for r in rows]
q_cmd_seq = q_target_rad
q_cmd_prev_seq = [q_target_rad[0]] + q_target_rad[:-1]

t0 = time.time()
results = {"b1": [], "b82": []}
free_results = {"b1": [], "b82": []}
for k in range(K):
    z_meas = np.concatenate([q_meas[k], [insertion_at_solve[k]]])
    kwargs = dict(z_meas=z_meas, q_cmd=q_cmd_seq[k], q_cmd_prev=q_cmd_prev_seq[k],
                  insertion_m=insertion_at_solve[k], measured_beam_position=tip_m[k],
                  control_index=ref_index[k], previous_input=prev_input_seq[k])
    s_b1 = c_b1.solve_delay_aware(**kwargs)
    s_b82 = c_b82.solve_delay_aware(**kwargs)
    results["b1"].append((s_b1, prev_input_seq[k], ref_index[k]))
    results["b82"].append((s_b82, prev_input_seq[k], ref_index[k]))

    f_s_b1 = f_b1.solve_delay_aware(**kwargs)
    f_s_b82 = f_b82.solve_delay_aware(**kwargs)
    free_results["b1"].append((f_s_b1, prev_input_seq[k], ref_index[k]))
    free_results["b82"].append((f_s_b82, prev_input_seq[k], ref_index[k]))

print(f"[beta sensitivity] replayed {K} ticks x 4 in {time.time()-t0:.1f}s")
with open("/home/jack/.claude/jobs/3710eca5/tmp/mpc_beta_sensitivity_results.pkl", "wb") as f:
    pickle.dump({"results": results, "free_results": free_results, "DELTA_U_MAX": DELTA_U_MAX}, f)
print("saved.")

# ================= metrics =================
def rho_desire(free_list):
    return np.array([np.max(np.abs(step.command - u_prev)/DELTA_U_MAX) for step, u_prev, idx in free_list])

print("\n=== 1. FREE-BOTH rho_desire ===")
for cond in ("b1", "b82"):
    rho = rho_desire(free_results[cond])
    print(f"beta={cond}: median={np.median(rho):.3f} p95={np.percentile(rho,95):.3f} "
          f"max={np.max(rho):.3f} P(rho>1)={np.mean(rho>1):.3f}")

def stagewise(cond_list):
    du_active = np.zeros((len(cond_list), N)); u_active = np.zeros((len(cond_list), N))
    for i, (step, u_prev, idx) in enumerate(cond_list):
        pu = step.predicted_inputs; prev = u_prev
        for j in range(N):
            du = pu[j] - prev
            du_active[i, j] = np.any(np.abs(du) >= DELTA_U_MAX - TOL)
            u_active[i, j] = np.any(np.abs(pu[j]) >= VLIM - TOL)
            prev = pu[j]
    return du_active.mean(axis=0), u_active.mean(axis=0)

print("\n=== 2. Constrained: stagewise P(du_j active), P(u_j active) ===")
for cond in ("b1", "b82"):
    p_du, p_u = stagewise(results[cond])
    print(f"beta={cond} P(du_j active): " + " ".join(f"{v:.2f}" for v in p_du))
    print(f"beta={cond} P(u_j  active): " + " ".join(f"{v:.2f}" for v in p_u))

def rms_u0_uref(cond_list):
    diffs = np.array([step.command - reference_input[idx] for step, u_prev, idx in cond_list])
    return np.sqrt(np.mean(np.sum(diffs**2, axis=1))), diffs

print("\n=== 3. RMS||u0-u_ref|| ===")
rms_vals = {}
for cond in ("b1", "b82"):
    rms, diffs = rms_u0_uref(results[cond])
    rms_vals[cond] = rms
    print(f"beta={cond}: RMS||u0-u_ref||={rms*1e3:.3f}mrad")

print("\n=== 4. Direct controller sensitivity S_beta ===")
u0_b1  = np.array([step.command for step, u_prev, idx in results["b1"]])
u0_b82 = np.array([step.command for step, u_prev, idx in results["b82"]])
du_beta = u0_b82 - u0_b1
delta_u_beta = np.sqrt(np.mean(np.sum(du_beta**2, axis=1)))
rms_u0_b1 = np.sqrt(np.mean(np.sum(u0_b1**2, axis=1)))
S_beta = delta_u_beta / rms_u0_b1
print(f"Delta_u_beta = RMS||u0^0.82 - u0^1.0|| = {delta_u_beta*1e3:.4f}mrad")
print(f"RMS||u0^1.0|| = {rms_u0_b1*1e3:.4f}mrad")
print(f"S_beta = {S_beta:.4f}  ({S_beta*100:.2f}%)")

per_actuator_delta = np.sqrt(np.mean(du_beta**2, axis=0))
labels = [f"q{i+1}" for i in range(6)] + ["L"]
print("per-actuator RMS(u0^0.82 - u0^1.0), mrad/mm:")
for lab, v in zip(labels, per_actuator_delta):
    unit = "mrad" if lab != "L" else "mm"
    scale = 1e3
    print(f"  {lab}: {v*scale:.4f}{unit}")

print("\n=== 5. Prediction sensitivity Delta_p_beta ===")
p_b1  = np.array([step.predicted_beam_positions for step, u_prev, idx in results["b1"]])   # (K,N,3)
p_b82 = np.array([step.predicted_beam_positions for step, u_prev, idx in results["b82"]])
dp = p_b82 - p_b1
delta_p_beta = np.sqrt(np.mean(np.sum(dp.reshape(K, -1)**2, axis=1)))
print(f"Delta_p_beta = RMS||p_stack^0.82 - p_stack^1.0|| (full N=15 stack) = {delta_p_beta*1e3:.4f}mm")
dp0 = p_b82[:,0,:] - p_b1[:,0,:]
delta_p0_beta = np.sqrt(np.mean(np.sum(dp0**2, axis=1)))
print(f"Delta_p_beta (stage j=1 only) = {delta_p0_beta*1e3:.4f}mm")
