import numpy as np
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

def theta_angle_solved(B, phi, mag, A_cs, L, E, I):
    rhs_eq = constant(B, mag, A_cs, L, E, I)
    theta_L = root_theta(rhs_eq, phi)
    return theta_L

def dtheta_dB(B, phi, mag, A_cs, L, E, I, dB=1e-4):
    theta_plus = theta_angle_solved(B+dB, phi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B-dB, phi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dB)

def dtheta_dphi(B, phi, mag, A_cs, L, E, I, dphi=1e-4):
    theta_plus = theta_angle_solved(B, phi+dphi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B, phi-dphi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dphi)

def dtheta_dL(B, phi, mag, A_cs, L, E, I, dL=1e-4):
    theta_plus = theta_angle_solved(B, phi, mag, A_cs, L+dL, E, I)
    theta_minus = theta_angle_solved(B, phi, mag, A_cs, L-dL, E, I)
    return (theta_plus - theta_minus) / (2*dL)