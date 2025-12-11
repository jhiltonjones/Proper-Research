import numpy as np 
from proper_research.models.beam_model import(theta_angle_solved, dtheta_dB, dtheta_dL, dtheta_dphi)


def jacobian_controller(theta_des,B_init, phi_init, L_init, 
                        mag, A_cs, E, I, L_min = 0.04, L_max=0.06, B_min=0.025, B_max = 0.025,
                        phi_min = np.deg2rad(-90), phi_max = np.deg2rad(90), dt=0.05,
                        Kp = 5.0, Ki = 0.0, Kd = 0.5, 
                        max_iter = 300, damping = 1e-3):


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