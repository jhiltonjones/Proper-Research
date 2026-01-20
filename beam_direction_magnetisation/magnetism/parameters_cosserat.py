import numpy as np
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_moment

rho = 0.15
alpha_deg =20
alpha_overhead = np.deg2rad(0)
alpha = np.deg2rad(alpha_deg)
theta_y=np.deg2rad(-40)


MU0_OVER_4PI = 1e-7
mag = 128e3
r = 0.0015
E = 3e6
A_cs = np.pi * r**2
I = np.pi * r**4 / 4
L = 0.06
nu = 0.49
G = E / (2*(1+nu))
J = 0.5*np.pi*r**4
EI = E*I
GJ = G*J
mu_line = mag * A_cs  
ell_m = 0.025
s_m = 0.025
s_k = 0.025
mu_0 = 4e-7*np.pi
B_r = 1.1
r_epm = 0.03
p_epm = 0.09
mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)
m_ext_full = mag_epm * np.array([1.0, 0.0, 0.0])
rho_mat = 1200          # kg/m^3 
g = 9.8
w = rho_mat * A_cs * g # N/m
f_g = np.array([0.0, 0.0, -w])[:, None] 