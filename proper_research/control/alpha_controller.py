import numpy as np 
from proper_research.models.magnetic_model import B_and_beta_from_phi_and_alpha
from proper_research.models.beam_model import constant, root_theta
from proper_research.parameters import AlphaControllerParams
from proper_research.control.core import PIDState, pid_step, damped_inverse_scalar

def tip_angle_from_theta(phi, alpha, R, m0, mu0, mag, A_cs, L, E, I):
    B ,beta = B_and_beta_from_phi_and_alpha(phi, alpha, R, m0, mu0)
    lam = constant(B, mag, A_cs, L, E, I)
    beta_min = 1e-4
    if abs(beta)<beta_min:
        return 0.0
    phi_eff = abs(beta)
    theta_abs = root_theta(lam,phi_eff)
    theta_L = theta_abs* np.sign(beta)
    return theta_L, B

def build_danger_zones(phi_values, alphas, all_dtheta_dalpha,
                           eps_grad=1e-3,
                           buffer_deg=5.0):
    buffer = np.deg2rad(buffer_deg)
    danger_intervals_by_phi = {}
    for i, phi in enumerate(phi_values):
        grad = np.array(all_dtheta_dalpha[i])
        alpha_curve = alphas

        sign_grad = np.sign(grad)
        sign_grad[np.abs(grad)< eps_grad] = 0

        sign_change_indices = []

        for j in range(1, len(sign_grad)):
            if sign_grad[j-1] ==0 or sign_grad[j] == 0:
                continue
            if sign_grad[j]*sign_grad[j-1] <0:
                sign_change_indices.append(j)
        snap_alphas = alpha_curve[sign_change_indices]
        intervals = []
        for a_snap in snap_alphas:
            a_min = a_snap - buffer
            a_max = a_snap + buffer
            intervals.append((a_min, a_max))
        danger_intervals_by_phi[phi] = intervals
    return danger_intervals_by_phi
def alpha_in_intervals(alpha, intervals):
    for a_min, a_max in intervals:
        if a_min <= alpha <= a_max:
            return True
    return False
def dtheta_dalpha(phi, alpha, R, m0, mu0, mag, A_cs, L, E, I, dalpha=1e-4):
    theta_plus,_ = tip_angle_from_theta(phi, alpha+dalpha, R, m0, mu0, mag, A_cs, L, E, I)
    theta_minus,_ = tip_angle_from_theta(phi, alpha-dalpha, R, m0, mu0, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus)/ (2*dalpha)
def alpha_controller_measured(
    theta_des,
    phi_fixed,
    alpha_init,
    R, m0, mu0,
    mag, A_cs, L, E, I,
    danger_intervals_by_phi,
    phi_values,
    measure_theta_fn,
    move_fn,
    params: AlphaControllerParams,
    use_online = False,
):

    phi_arr = np.array(phi_values)
    idx_phi = np.argmin(np.abs(phi_arr - phi_fixed))
    phi_key = phi_arr[idx_phi]
    danger_intervals = danger_intervals_by_phi.get(phi_key, [])
    max_step = np.deg2rad(15.0)
    min_step_jac = np.deg2rad(1)
    alpha = alpha_init
    alpha_prev, theta_prev, J_meas = None, None, None
    pid_state = PIDState()

    for k in range(params.max_iter):
        move_fn(alpha)

        theta_meas = measure_theta_fn()
        if use_online ==True and alpha_prev is not None:
            dalpha = alpha - alpha_prev
            dtheta = theta_meas - theta_prev
            if abs(dalpha) > min_step_jac:
                J_new = dtheta/dalpha
                if J_meas is None:
                    J_meas = J_new
                else:
                    beta = 0.2
                    J_meas = (1-beta)*J_meas + beta*J_new
                print(f"[online J] dalpha={np.rad2deg(dalpha):.2f} deg, "
                f"dtheta={np.rad2deg(dtheta):.2f} deg, "
                f"J_new={J_new:.4f}, J_meas={J_meas:.4f}")
        alpha_prev, theta_prev = alpha, theta_meas
            

        if abs(theta_meas) < params.theta_zero_thresh:
            print(f"[alpha controller] |theta_meas| ~ 0 (|θ|={np.rad2deg(theta_meas):.2f} deg). "
                  "Stopping to avoid flip.")
            return alpha, theta_meas, None, "danger_zone"


        theta_meas_eff = -theta_meas      
        e = theta_des - theta_meas_eff

        print(f"Theta desired is:  {np.rad2deg(theta_des):.2f} deg")
        print(f"Theta measured is: {np.rad2deg(theta_meas):.2f} deg")
        print(f"Error is:          {np.rad2deg(e):.2f} deg")

        theta_dot_cmd, pid_state = pid_step(
            e, pid_state, params.dt, params.Kp, params.Ki, params.Kd
        )

        J_alpha = dtheta_dalpha(phi_fixed, alpha, R, m0, mu0,
                                mag, A_cs, L, E, I)
        if J_meas is not None:
            J_used = (1-params.weight)*J_alpha + params.weight * J_meas
            print(f"Online update with J model = {J_alpha} and J_meas = {J_meas}")
        else:
            J_used = J_alpha
        if abs(J_used) > params.grad_runtime_thresh:
            print(f"[alpha controller] |dθ/dα| too large ({J_used:.2e}). "
                  "Stopping near snap region.")
            return alpha, theta_meas, None, "grad_too_large"


        dalpha_rate = damped_inverse_scalar(J_used, theta_dot_cmd, params.damping)

        
        step = np.clip(dalpha_rate * params.dt, -max_step, max_step)

        alpha_new = np.clip(alpha - step, params.alpha_min, params.alpha_max)

        if alpha_in_intervals(alpha_new, danger_intervals):
            print("[alpha controller] Proposed alpha enters danger zone (snap region). "
                  "Not proceeding further.")
            return alpha, theta_meas, None, "danger_zone"

        print(f"Alpha new is: {np.rad2deg(alpha_new):.2f} deg, step: {np.rad2deg(step):.2f} deg")

        if abs(e) <= np.deg2rad(params.tol_deg):
            print(f"[alpha controller] Converged in {k} steps "
                  f"(measured θ = {np.rad2deg(theta_meas):.2f} deg)")
            _, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0,
                                             mag, A_cs, L, E, I)
            return alpha, theta_meas, B_curr, "converged"

        alpha = alpha_new
    _, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0,
                                    mag, A_cs, L, E, I)
    print("[alpha controller] Max iterations reached without full convergence")
    return alpha, theta_meas, B_curr, "max_iter"

