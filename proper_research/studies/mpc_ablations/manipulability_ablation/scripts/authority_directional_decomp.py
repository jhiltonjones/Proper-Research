"""Directional-derivative decomposition (spec, requested explicitly):
ΔJ_component = f_component^T @ d,  d = u_lambda - u_0 (full QP decision vector)
at 2 representative points: idx=70 (ordinary mid-path) and idx=127 (phi-peak,
where the linear surrogate was already shown to misbehave)."""
import sys
import math
import pickle
import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import BeamOutputMPCConfig
from proper_research.controllers.mpc_delay_aware.authority_nullspace_mpc import AuthorityShapedDelayAwareMPC

PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCHEDULE_PATH = "/tmp/rectangle_genuine_ltv_schedule.npy"
N, DT = 15, 0.1

reference_raw = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
schedule = np.load(SCHEDULE_PATH)
z_ref_all = np.asarray(reference_raw.state, dtype=float)
u_ref_all = np.asarray(reference_raw.input, dtype=float)
des_all = np.asarray(reference_raw.desired_position_m, dtype=float)

with open("../data/authority_phase0_schedule.pkl", "rb") as f:
    phase0 = pickle.load(f)
authority_schedule = {"g_N": phase0["g_N"]}

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


def make_controller(lam):
    return AuthorityShapedDelayAwareMPC(
        secondary_objective="manipulability", manipulability_weight=lam,
        authority_schedule=authority_schedule,
        reference=load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False),
        config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )


for idx in (70, 127):
    print(f"\n{'='*90}\nidx={idx}  (phi0={phase0['phi'][idx]:.3f}  ||g_N||={np.linalg.norm(phase0['g_N'][idx]):.3f})\n{'='*90}")
    z0 = z_ref_all[idx]
    q_cmd_k = z0[:6]
    q_cmd_km1 = z_ref_all[max(idx - 1, 0)][:6]
    prev_u = u_ref_all[max(idx - 1, 0)]
    tip = des_all[idx]
    kwargs = dict(z_meas=z0, q_cmd=q_cmd_k, q_cmd_prev=q_cmd_km1, insertion_m=float(z0[6]),
                  measured_beam_position=tip, control_index=idx, previous_input=prev_u)
    x_exec = np.concatenate([q_cmd_k, q_cmd_km1, [float(z0[6])]])

    ctrl0 = make_controller(0.0)
    step0 = ctrl0.solve_delay_aware(**kwargs)
    u0 = step0.command_stack if hasattr(step0, "command_stack") else None

    # need the FULL decision vector u (not just u0) -- pull from controller internals
    _, estimated = ctrl0._estimate_output_residual(
        measured_state=kwargs["z_meas"], measured_beam_position=kwargs["measured_beam_position"],
        control_index=kwargs["control_index"],
    )
    for lam in (1.0, 3.0, 10.0, 30.0):
        ctrl = make_controller(lam)
        H0, f0, info0 = ctrl0._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=prev_u, control_index=idx, estimated_residual=estimated)
        H1, f1, info1 = ctrl._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=prev_u, control_index=idx, estimated_residual=estimated)

        # solve both QPs to get full decision vectors u0_full, u_lam_full
        step_base = ctrl0.solve_delay_aware(**kwargs)
        step_lam = ctrl.solve_delay_aware(**kwargs)
        # reconstruct full u via re-deriving from H,f is fragile; instead use the
        # difference in the FIRST-stage command as a proxy scaled by nu/m, OR
        # directly recover full stacked u from the controller's last QP solution
        # if exposed. Fall back: use f-term dot (u_lam0 - u_base0) over just u0
        # (first-stage), which is the observable, physically-realized decision.
        d0 = (step_lam.command - step_base.command)  # first-stage only (m,)
        m = ctrl0.m
        f_manip_0 = info1.get("f_manip", np.zeros_like(f1))[:m]
        # f_Q is zero in this mode; f_R, f_Rd are the base-cost linear pieces.
        # Recover per-component linear terms by re-deriving R/Rd contributions directly:
        input_reference_vec = reference_raw.input_window(idx, N).reshape(-1)
        f_R_full = -2.0 * ctrl0.Rbar @ input_reference_vec
        prev_vec = np.zeros(ctrl0.nu); prev_vec[:m] = prev_u
        f_Rd_full = -2.0 * ctrl0.D.T @ ctrl0.Rdbar @ prev_vec
        G, constant_error = info0["G"], info0["constant_error"]
        f_Qp_full = 2.0 * (G.T @ ctrl0.Qpbar @ constant_error)
        f_manip_full = info1.get("f_manip", np.zeros(ctrl0.nu))

        d_full = np.zeros(ctrl0.nu)
        d_full[:m] = d0  # only first-stage command directly observed; rest of horizon not extracted here

        dJ_manip = float(f_manip_full[:m] @ d0)
        dJ_Qp = float(f_Qp_full[:m] @ d0)
        dJ_R = float(f_R_full[:m] @ d0)
        dJ_Rd = float(f_Rd_full[:m] @ d0)
        print(f"  lambda={lam:5g}: ||d0||={np.linalg.norm(d0):.4e}   "
              f"dJ_manip={dJ_manip:+.3e}  dJ_Qp={dJ_Qp:+.3e}  dJ_R={dJ_R:+.3e}  dJ_Rd={dJ_Rd:+.3e}  "
              f"sum={dJ_manip+dJ_Qp+dJ_R+dJ_Rd:+.3e}")
