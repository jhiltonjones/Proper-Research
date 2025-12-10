import numpy as np
from magnetic_magnitude import magnetic_field
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
def constant(B, mag, A_cs, L, E, I):
    return (mag*B*A_cs*L**2)/(E*I)

def integral_cos(phi, theta_L, eps=1e-6):
    def integrand(theta):
        w = np.cos(phi - theta_L) - np.cos(phi - theta)
        if w <= 0:
            return 0.0
        return 1/np.sqrt(w)
    upper = theta_L-eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    xi_val = 0.5*val**2
    return xi_val

def root_theta(rhs_eq, phi, tol = 1e-6):
        eps = 1e-4
        theta_min = eps
        theta_max = max(phi - eps, theta_min)
        def f(theta_l):
            return integral_cos(phi, theta_l) - rhs_eq
        sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
        return sol.root
def tip_angle_from_B_phi_L(B, phi, mag, A_cs, L, E, I):
    # print(B, phi, mag, A_cs, L, E, I)
    lam = constant(B, mag, A_cs, L, E, I)
    theta = root_theta(lam, phi)
    return theta  

def intergal_x(phi, theta_L, constant=0, constant_use=True, eps = 1e-4):
    def integrand(theta):
        w = np.cos(phi-theta_L)-np.cos(phi- theta)
        return np.cos(theta)/ np.sqrt(w)
    upper = theta_L-eps
    x_pos,_ = quad(integrand, 0, upper, limit = 200)
    if constant_use ==True:
        return constant * x_pos
    else:
        return x_pos

def integral_y(phi, theta_L, constant=0, constant_use=True, eps=1e-4):
    def integrand(theta):
        w = np.cos(phi-theta_L) - np.cos(phi - theta)
        return np.sin(theta)/ np.sqrt(w)
    upper = theta_L-eps
    y_pos,_ = quad(integrand, 0, upper, limit=200)
    if constant_use == True:
        return constant * y_pos
    else:
        return y_pos
    
def find_theta_L(x_m, phi, constant, eps = 1e-4):
    def f(theta_l):
        return intergal_x(phi, theta_l, constant) - x_m
    bracket = (eps, phi-eps)
    theta_sol = root_scalar(f, bracket=bracket)
    return theta_sol.root

def find_angle_and_length(phi, delta_x, delta_y, mag, B, A_cs, E, I, eps = 1e-4, eps_bracket = 1e-3):
    def f(theta_l):
        X = intergal_x(phi, theta_l, constant_use=False)
        Y = integral_y(phi, theta_l, constant_use=False)
        return Y*delta_x - X*delta_y
    bracket = (eps_bracket, phi-eps)
    theta_l_s = root_scalar(f, bracket=bracket)
    theta_l_sol = theta_l_s.root
    xi_integral = integral_cos(phi, theta_l_sol)
    length = np.sqrt(((E*I)*xi_integral)/(mag*B*A_cs))
    return theta_l_sol, length

if __name__ == '__main__':
    length = 0.04
    E = 4.5e6
    radius = 0.0015
    A_cs = np.pi * radius**2
    I = np.pi * radius**4 / 4
    phi = np.deg2rad(75)
    B = 0.02
    mag = 128e3
    rhs_eq = constant(B, mag, A_cs, length, E, I)
    print(rhs_eq)
    theta_L = root_theta(rhs_eq, phi)
    print(np.rad2deg(theta_L))
    angle = tip_angle_from_B_phi_L(B, phi, mag, A_cs, length, E, I)
    print(np.rad2deg(angle))
    constant_carti = np.sqrt((E*I)/(2*mag*B*A_cs))
    x_pos = intergal_x(phi, theta_L, constant_carti)
    y_pos = integral_y(phi, theta_L, constant_carti)
    print(f"X coordinate is: {x_pos*1000}, Y coordinate is: {y_pos*1000}")

    theta_from_x = find_theta_L(0.029, phi, constant_carti)
    print(f"theta position from x is: {np.rad2deg(theta_from_x)}")
 
    theta_angle, length_from_carti = find_angle_and_length(phi,  0.034, 0.001, mag, B, A_cs, E, I)
    print(f"The recovered angle is {np.rad2deg(theta_angle)} with length {length_from_carti}")
    lengths = np.linspace(0.04, 0.06, 4)
    phis = np.linspace(np.deg2rad(1), np.pi/2, 30)
    graph_inputs = []

    for L in lengths:
        angles = []
        for phi in phis:
            rhs_eq = constant(B, mag, A_cs, L, E, I)
            theta_L = root_theta(rhs_eq, phi)
            angles.append(theta_L)
        graph_inputs.append(angles)

    plt.figure()
    for L, graph in zip(lengths, graph_inputs):
        plt.plot(np.rad2deg(phis), np.rad2deg(graph), label=f"L={L:.3f} m")
    plt.title("Bending Angle vs Phi Angle for different Beam Lengths")
    plt.xlabel("Phi")
    plt.ylabel("Bending of beam")
    plt.grid()
    plt.legend()
    plt.show()

    
