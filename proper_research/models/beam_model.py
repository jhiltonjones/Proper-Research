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

# def root_theta(rhs_eq, phi, tol = 1e-6):
#         eps = 1e-4
#         theta_min = eps
#         theta_max = max(phi - eps, theta_min)
#         def f(theta_l):
#             return integral_cos(phi, theta_l) - rhs_eq
#         sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
#         return sol.root


def root_theta(rhs_eq, phi, tol=1e-6, debug=False, do_scan=True, n_scan=200):
    eps = 1e-4

    # Basic domain guard
    if not np.isfinite(phi) or not np.isfinite(rhs_eq):
        raise ValueError(f"Non-finite inputs: phi={phi}, rhs_eq={rhs_eq}")

    theta_min = eps
    theta_max = phi - eps

    # If phi too small, we cannot bracket in (0, phi)
    if theta_max <= theta_min:
        msg = f"phi too small for bracket: phi={phi:.6g}, theta_min={theta_min:.6g}, theta_max={theta_max:.6g}"
        if debug:
            print("[root_theta] " + msg)
        raise ValueError(msg)

    def f(theta_l):
        val = integral_cos(phi, theta_l) - rhs_eq
        return float(val)

    fa = f(theta_min)
    fb = f(theta_max)

    if debug:
        print("[root_theta] rhs_eq=", rhs_eq)
        print(f"[root_theta] phi={phi:.8f}")
        print(f"[root_theta] bracket=[{theta_min:.8f}, {theta_max:.8f}]")
        print(f"[root_theta] f(theta_min)={fa:.8e}, f(theta_max)={fb:.8e}")

    # NaN/Inf check
    if (not np.isfinite(fa)) or (not np.isfinite(fb)):
        raise ValueError(f"Non-finite f at bracket endpoints: fa={fa}, fb={fb}, phi={phi}, rhs_eq={rhs_eq}")

    # If we already have sign change, solve normally
    if fa * fb < 0:
        sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
        return float(sol.root)

    # Optional: scan for a sign-changing sub-bracket
    if do_scan:
        thetas = np.linspace(theta_min, theta_max, n_scan)
        vals = np.array([f(t) for t in thetas], dtype=float)

        # Find any sign change
        for i in range(len(thetas) - 1):
            v0, v1 = vals[i], vals[i+1]
            if not np.isfinite(v0) or not np.isfinite(v1):
                continue
            if v0 == 0.0:
                if debug:
                    print(f"[root_theta] exact root at theta={thetas[i]:.8f}")
                return float(thetas[i])
            if v0 * v1 < 0:
                a, b = float(thetas[i]), float(thetas[i+1])
                if debug:
                    print(f"[root_theta] scanned bracket found: [{a:.8f}, {b:.8f}] with f(a)={v0:.3e}, f(b)={v1:.3e}")
                sol = root_scalar(f, bracket=[a, b], xtol=tol)
                return float(sol.root)

        if debug:
            print("[root_theta] scan failed: no sign change found in bracket grid.")
            print(f"[root_theta] f range: min={np.nanmin(vals):.3e}, max={np.nanmax(vals):.3e}")

    # If still no bracket: raise with useful info
    raise ValueError(
        "No sign change in root bracket for theta. "
        f"phi={phi}, rhs_eq={rhs_eq}, f(theta_min)={fa}, f(theta_max)={fb}"
    )

def tip_angle_from_B_phi_L(B, phi, mag, A_cs, L, E, I):
    print(B, phi, mag, A_cs, L, E, I)
    lam = constant(B, mag, A_cs, L, E, I)
    theta = root_theta(lam, phi)
    return theta  

def integral_x(phi, theta_L, constant=0, constant_use=True, eps = 1e-4):
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
        return integral_x(phi, theta_l, constant) - x_m
    bracket = (eps, phi-eps)
    theta_sol = root_scalar(f, bracket=bracket)
    return theta_sol.root

def find_angle_and_length(phi, delta_x, delta_y, mag, B, A_cs, E, I, eps = 1e-4, eps_bracket = 1e-3):
    def f(theta_l):
        X = integral_x(phi, theta_l, constant_use=False)
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
if __name__ == '__main__':
    # theta = np.deg2rad(26.56)
    E = 2e6
    radius = 0.0008
    A_cs = np.pi * radius**2
    I = np.pi * radius**4 / 4
    B = 0.01
    mag = 128e3
    L = 0.05
    phi = np.deg2rad(50)
    # theta_L = 30
    # constant_carti = np.sqrt((E*I)/(2*mag*B*A_cs))
    # x_pos = integral_x(phi=np.deg2rad(phi), theta_L=np.deg2rad(theta_L), constant=constant_carti)
    # print(f"x_pos is : {x_pos}")
    # y_pos = integral_y(phi=np.deg2rad(phi), theta_L=np.deg2rad(theta_L), constant=constant_carti)
    # print(f"y_pos is : {y_pos}")
    # angle = np.arctan2(y_pos, x_pos)
    # print(np.rad2deg(angle))
    angles = []
    phis = np.linspace(0.1,80,16)
    for p in phis:
        phi = np.deg2rad(p)
        act_theta = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        theta_angle_deg = np.rad2deg(act_theta)
        angles.append(theta_angle_deg) 

    plt.figure(figsize=(8, 5))
    plt.plot(phis, angles, marker='o', linestyle='-', color='b', label='Theta vs Phi')

    plt.title('Angle vs Phis')
    plt.xlabel('Phi (degrees or radians)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    plt.show()