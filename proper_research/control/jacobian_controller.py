import numpy as np 
from proper_research.models.beam_model import(theta_angle_solved, dtheta_dB, dtheta_dL, dtheta_dphi)
from proper_research.models.magnetic_model import magnetic_field

def jacobian_controller(theta_des,B_init, phi_init, L_init, 
                        mag, A_cs, E, I, L_min = 0.04, L_max=0.06, B_min=0.025, B_max = 0.025,
                        phi_min = np.deg2rad(-90), phi_max = np.deg2rad(90), dt=0.05,
                        Kp = 5.0, Ki = 0.0, Kd = 0.5, 
                        max_iter = 50, damping = 1e-3):


    B = B_init
    phi = phi_init
    L = L_init

    e_init = 0
    e_prev = 0.0
    for k in range(max_iter):
        # print(B, phi, mag, A_cs, L, E, I)
        act_theta = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - act_theta
        e_dot = (e-e_prev)/dt
        e_init += e*dt

        theta_do_cmd = Kp*e + Ki*e_init + Kd * e_dot

        J_B = dtheta_dB(B, phi, mag, A_cs, L, E, I)
        J_L = dtheta_dL(B, phi, mag, A_cs, L, E, I)
        J_phi = dtheta_dphi(B, phi, mag, A_cs, L, E, I)

        if (B <= B_min and theta_do_cmd* J_B < 0):
            J_B = 0.0
        if (B>= B_max and theta_do_cmd*J_B > 0):
            J_B = 0.0

        if (L <= L_min and theta_do_cmd* J_L < 0):
            J_L = 0.0

        if (L>= L_max and theta_do_cmd*J_L > 0):
            J_L = 0.0
        
        if (phi <= phi_min and theta_do_cmd* J_phi < 0):
            J_phi = 0.0
        if (phi>= phi_max and theta_do_cmd*J_phi > 0):
            J_phi = 0.0
        
        JJt = J_B**2 + J_L**2 + J_phi**2
        gain = theta_do_cmd/ (JJt+damping**2)
        dB = J_B * gain
        dL = J_L * gain
        dphi = J_phi * gain 

        B += dB*dt
        L += dL*dt
        phi += dphi*dt

        B = min(max(B, B_min), B_max)
        L = min(max(L, L_min), L_max)
        phi = min(max(phi, phi_min), phi_max)

        theta_update = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - theta_update
        if abs(e) <= np.deg2rad(.5):
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return B, phi, L, np.rad2deg(theta_update)
def dB_dx(x, mu_0, mu, mu_hat, h=1e-4):
    p_plus = np.array([x+h, 0, 0])
    p_minus = np.array([x-h, 0, 0])

    Bp = magnetic_field(mu_0, mu, p_plus, mu_hat)
    Bm = magnetic_field(mu_0, mu, p_minus, mu_hat)
    return (Bp[0] - Bm[0])/ (2*h)
def solve_mag_pose(field_des, x_init, mu_0, mu, mu_hat, dt = 0.05, Kp = 1, Ki = 0.0, Kd = 0.01, damping = 1e-3, max_iter = 500):
    e_init = 0.0
    e_prev = 0.0
    x = x_init

    for k in range(max_iter):
        p_vec = np.array([x, 0, 0])
        # print(f"parameters are x: {x}, mu: {mu}, mu_hat: {mu_hat}")
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        # print(f"Current field is {field_curr[0]}, desired field is {field_des}")
        e = field_des - field_curr[0]
        e_init +=e*dt
        e_dot = (e-e_prev)*dt
        theta_dot_cmd = Kp*e + Ki*e_init + Kd*e_dot
        J_x = dB_dx(x, mu_0, mu, mu_hat)
        JJt = J_x**2
        gain = theta_dot_cmd / (JJt + damping**2)
        dx = J_x *gain
        x += dx * dt
        p_vec = np.array([x, 0, 0])
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        e = field_des - field_curr[0]
        if abs(e) <= 1e-5:
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return x, field_curr