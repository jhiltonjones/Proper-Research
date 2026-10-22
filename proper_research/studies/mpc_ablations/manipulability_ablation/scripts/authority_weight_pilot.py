"""Representative-snapshot weight-sensitivity pilot (cheap, ~10 snapshots x
7 lambda values), per the reviewer's explicit request: choose lambda_manip
from SOLUTION sensitivity (du0, nullspace departure, true phi gain vs
predicted, E_chosen), not from a global ||f_manip||/||f_rest|| norm ratio
(which conflates task-relevant and nullspace-relevant directions).

Snapshots chosen from the cached Phase-0 schedule + conflict diagnostic:
  idx=127: max phi (2.11)          idx=126: max ||g_N|| (0.265)
  idx=264: min phi (0.17)          idx=95, 220, 240: anchor-vs-authority
                                     opposition regions (cos_theta<-0.7)
  idx=30,70,160,190: ordinary mid-path points

Each snapshot is evaluated ON-REFERENCE (x_exec = z_ref at that index,
measured_beam_position = desired_position at that index, so d_k=0) -- this
isolates the secondary-objective effect from confounding real tracking
disturbance, exactly the clean controlled comparison a weight-calibration
pilot needs.
"""
import sys
import math
import pickle
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
S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
S_P = np.array([0.5e-3, 0.5e-3])
D_U = np.array([0.10] * 6 + [2.0e-3])
S_U = np.array([0.05] * 6 + [5.0e-3])  # input_error_scale convention, used for ||du0||_{S_u^-1}
C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
LAMBDAS = [0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
SNAPSHOT_INDICES = [30, 70, 95, 126, 127, 160, 190, 220, 240, 264]

reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
schedule = np.load(SCHEDULE_PATH)
z_ref_all = np.asarray(reference.state, dtype=float)
u_ref_all = np.asarray(reference.input, dtype=float)
des_all = np.asarray(reference.desired_position_m, dtype=float)
N_ref = z_ref_all.shape[0]

with open("../data/authority_phase0_schedule.pkl", "rb") as f:
    phase0 = pickle.load(f)
authority_schedule = {"g_N": phase0["g_N"]}

print("[setup] building state-dependent Jacobian provider...")
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


def make_controller(lam):
    return AuthorityShapedDelayAwareMPC(
        secondary_objective="manipulability", manipulability_weight=lam,
        authority_schedule=authority_schedule,
        reference=load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False),
        config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )


results = []
for idx in SNAPSHOT_INDICES:
    z0 = z_ref_all[idx]
    u_ref_idx = u_ref_all[idx]
    q_cmd_k = z0[:6]
    q_cmd_km1 = z_ref_all[max(idx - 1, 0)][:6]
    prev_u = u_ref_all[max(idx - 1, 0)]
    tip = des_all[idx]  # on-reference: measured == desired, d_k = 0

    kwargs = dict(z_meas=z0, q_cmd=q_cmd_k, q_cmd_prev=q_cmd_km1, insertion_m=float(z0[6]),
                  measured_beam_position=tip, control_index=idx, previous_input=prev_u)

    idx_lin1 = min(idx + 1, N_ref - 1)  # first predicted stage's reference index
    z_lin1 = z_ref_all[idx_lin1]
    phi_lin1 = phase0["phi"][idx_lin1]
    g_N_lin1 = phase0["g_N"][idx_lin1]

    idx3 = min(idx + 3, N_ref - 1)
    des3 = des_all[idx3]

    per_lambda = {}
    u0_base = None
    for lam in LAMBDAS:
        ctrl = make_controller(lam)
        step = ctrl.solve_delay_aware(**kwargs)
        if lam == 0.0:
            u0_base = step.command.copy()
        du0 = step.command - u0_base
        du0_norm = float(np.linalg.norm(du0 / S_U))

        z1_pred = step.predicted_states[0]
        dz_tilde1 = (z1_pred - z_lin1) / S_Z
        _, P_N1, _ = nullspace_projector(C @ schedule[idx_lin1], S_Z)
        PN_dz = float(np.linalg.norm(P_N1 @ dz_tilde1))

        dphi_lin = float(g_N_lin1 @ dz_tilde1)
        phi_true1 = phi_true(z1_pred)
        dphi_true = phi_true1 - phi_lin1
        r_phi = dphi_true / dphi_lin if abs(dphi_lin) > 1e-6 else float("nan")

        p_hat3 = step.predicted_beam_positions[2]
        Ec3 = float(np.linalg.norm(p_hat3 - des3)) * 1e3

        dist_state = float(np.min(np.minimum(z1_pred - STATE_MIN, STATE_MAX - z1_pred)))
        dist_vel = float(np.min(VLIM - np.abs(step.command)))

        per_lambda[lam] = dict(
            du0_norm=du0_norm, PN_dz=PN_dz, dphi_true=dphi_true, dphi_lin=dphi_lin,
            r_phi=r_phi, Ec3=Ec3, dist_state=dist_state, dist_vel=dist_vel,
        )
    results.append(dict(idx=idx, phi0=phase0["phi"][idx], norm_gN=np.linalg.norm(phase0["g_N"][idx]),
                         per_lambda=per_lambda))
    print(f"idx={idx:4d} (phi={phase0['phi'][idx]:.3f} ||g_N||={np.linalg.norm(phase0['g_N'][idx]):.3f}):")
    for lam in LAMBDAS:
        r = per_lambda[lam]
        print(f"   lambda={lam:7.1f}: ||du0||_Su-1={r['du0_norm']:7.4f}  ||P_N dz~||={r['PN_dz']:7.4f}  "
              f"dphi_true={r['dphi_true']:+.4f}  dphi_lin={r['dphi_lin']:+.4f}  r_phi={r['r_phi']:+.2f}  "
              f"Ec3={r['Ec3']:.4f}mm  dist_state={r['dist_state']:.4f}  dist_vel={r['dist_vel']:.4f}")

with open("../data/authority_weight_pilot.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nsaved authority_weight_pilot.pkl")
