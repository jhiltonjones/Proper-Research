import numpy as np
from magnetic_magnitude import magnetic_moment, B_and_beta_from_phi_and_alpha
from modelling_phi_angle import constant, root_theta
import matplotlib.pyplot as plt
import pickle

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
def alpha_controller(theta_des,
                     phi_fixed,
                     alpha_init,
                     R, m0, mu0,
                     mag, A_cs, L, E, I,
                     danger_intervals_by_phi,
                     phi_values,
                     alpha_min = -np.pi,
                     alpha_max = np.pi,
                     dt = 0.05,
                     Kp = 5.0, Ki = 0.0, Kd = 0.5,
                     damping = 1e-3,
                     max_iter = 300,
                     grad_runtime_thresh = 20.0,
                     theta_zero_thresh = np.deg2rad(3.0)):
    phi_arr = np.array(phi_values)
    idx_phi = np.argmin(np.abs(phi_arr-phi_fixed))
    phi_key = phi_arr[idx_phi]
    danger_intervals = danger_intervals_by_phi.get(phi_key, [])
    alpha = alpha = alpha_init
    e_int = 0
    e_prev = 0

    for k in range(max_iter):
        theta_curr, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0, 
                                                  mag, A_cs, L, E, I)
        if abs(theta_curr) < theta_zero_thresh:
            print(f"[alpha controller] |theta| ~ 0 (|θ|={np.rad2deg(theta_curr):.2f} deg). "
                  "Stopping to avoid flip.")
            return alpha, theta_curr, B_curr, "danger_zone"
        e = theta_des - theta_curr
        e_int += e*dt
        e_dot = (e-e_prev) / dt
        theta_dot_cmd = Kp * e + Ki * e_int + Kd * e_dot
        J_alpha = dtheta_dalpha(phi_fixed, alpha, R, m0, mu0,
                                mag, A_cs, L, E, I)
        if abs(J_alpha) > grad_runtime_thresh:
            print(f"[alpha controller] |dθ/dα| too large ({J_alpha:.2e}). "
            "Stopping near snap region.")
            return alpha, theta_curr, B_curr, "grad_too_large"
        JJt = J_alpha**2
        gain = theta_dot_cmd / (JJt+damping**2)
        dalpha = J_alpha*gain
        alpha_new = alpha + dalpha *dt
        alpha_new = min(max(alpha_new, alpha_min), alpha_max)   
        if alpha_in_intervals(alpha_new, danger_intervals):
            print("[alpha controller] Proposed alpha enters danger zone (snap region). "
                  "Not proceeding further.")
            return alpha, theta_curr, B_curr, "danger_zone"

        alpha = alpha_new

        theta_new, B_curr = tip_angle_from_theta(phi_fixed, alpha, R, m0, mu0,
                                                 mag, A_cs, L, E, I)
        e_new = theta_des - theta_new

        if abs(e_new) <= np.deg2rad(0.5):
            print(f"[alpha controller] Converged in {k} steps")
            return alpha, theta_new, B_curr, "converged"

        e_prev = e_new

    print("[alpha controller] Max iterations reached without full convergence")
    return alpha, theta_new, B_curr, "max_iter"



if __name__ == '__main__':

    mu_0 = 4e-7*np.pi
    B_r = 1.2
    r_epm = 0.03
    len_epm = 0.09
    length_beam = 0.04 
    r = 0.0015
    A_cs = np.pi * r**2
    len_beam = 0.06
    E = 3.6e6
    I = np.pi * r**4/4
    mag = 128e3
    mag_epm = magnetic_moment(B_r, mu_0, r_epm, len_epm)
    R = 0.135

    phi_values = np.deg2rad(np.linspace(0.1,90,5))
    alphas = np.deg2rad(np.linspace(-180, 180, 91))
    allBmag = []
    all_bend = []
    all_dtheta_dalpha = []
    for phi in phi_values:
        bend_values = []
        B_values  = []
        for alpha in alphas:
            theta_L, B = tip_angle_from_theta(phi, alpha, R, mag_epm, mu_0, mag, A_cs, length_beam, E, I)
            bend_values.append(theta_L)
            B_values.append(B)
        all_bend.append(bend_values)
        allBmag.append(B_values)
    for bending in all_bend:
        bending = np.array(bending)
        dtheta_dalpha = np.gradient(bending, alpha)
        all_dtheta_dalpha.append(dtheta_dalpha)
    danger_intervals_by_phi = build_danger_zones(phi_values, alphas, all_dtheta_dalpha, eps_grad=1e-3, buffer_deg=5)
    data_to_save = {
        "phi_values": phi_values,
        "alphas": alphas,
        "danger_intervals_by_phi": danger_intervals_by_phi,

    }
    with open("alpha_danger_map.pkl", "wb") as f: 
        pickle.dump(data_to_save,f)
    print("Saved alpha_danger_map to file.")
    
    
    
    plt.figure()
    for phi_a, b_curve in zip(phi_values, allBmag):
        plt.plot(np.rad2deg(alphas), b_curve, label=r"$\phi_{\text{arc}}$")
        plt.xlabel("Alpha (deg)")
        plt.ylabel("|B| (T)")
        plt.title("Total magnetic field magnitude vs alpha")
        plt.grid(True)
        plt.legend()


    plt.figure()
    for phi_a, bend_curve in zip(phi_values, all_bend):
        plt.plot(np.rad2deg(alphas), np.rad2deg(bend_curve), label= f"phi_arc = {np.rad2deg(phi_a):.2f}")
    plt.xlabel("Alpha (deg)")
    plt.ylabel("Bending angle (deg)")
    plt.title("Beam bending vs alpha for different phi_arc")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for phi_a, dcurve in zip(phi_values, all_dtheta_dalpha):
        plt.plot(np.rad2deg(alphas), dcurve, label = f"phi is : {np.rad2deg(phi_a):.2f}")
    plt.xlabel("Alpha Values")
    plt.ylabel("Gradient of theta with respect to alpha")
    plt.title("Beam bending gradient for alpha values")

    plt.grid(True)
    plt.legend()

    plt.show()
