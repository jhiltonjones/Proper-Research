from beam_direction_magnetisation.cosserat_w_minimal_energy import (make_lumen_centerline_turning, CosseratForwardModel, 
                                                                    dipole_from_pose, u0_from_bvp, solve_quasistatic_insertion, 
                                                                    lumen_violation_profile)
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d                                                                    
from proper_research.parameters import default_magnet_params, default_beam_params
import numpy as np
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
import numpy as np
import matplotlib.pyplot as plt
from proper_research.control.mpc_boundary import resample_polyline

beam_params = default_beam_params()
mag_params = default_magnet_params()
L_min_energy = 0.118
mag_len = beam_params.length_of_mag
m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
pivot_point = np.array([
0.67, -0.719, -0.093, -3.087, 0.341, 0.067
], float)


# start_point = np.array([
# 0.6681328220229531, -0.7055298925316631, 0.1517853768068757, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
# ], float)
start_point = np.array([ 0.680, -0.739,  0.092, -2.937, 1.041 ,
    0.078 ])
# start_point[2] -=0.25
wire_len = L_min_energy - mag_len
T_ur_pivot = ur_pose6_to_T(pivot_point)   
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

T_ur_mag = ur_pose6_to_T(start_point)      
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

q = q0_ur
R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
t0 = R0 @ np.array([-1.0, 0.0, 0.0])   

s_straight = 0.05

lumen_C = make_lumen_centerline_turning(
    p_start=p0_ur,
    t0=t0,
    length=0.08 + s_straight,     
    n_pts=130,               
    bend_axis=np.array([0.0, 0.0, 1.0]),
    bend_angle=np.deg2rad(40.0),
    bend_start=0.01 + s_straight,    
    bend_end=0.08 + s_straight       
)

lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
lumen_R = np.full(len(lumen_C), 0.004)
lumen_path = lumen_C

model = CosseratForwardModel(
    p0=p0_ur,
    q0=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    m_local_fun=make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0),
    m_moment=0.0,
    wire_len=wire_len,
)
lumen_R = np.full(len(lumen_C), 0.004)  

m_src = dipole_from_pose(q_src_ur, m_body)

u0 = None  
hist = solve_quasistatic_insertion(
    p0=p0_ur, q0=q0_ur,
    L0=0.03, Lf=L_min_energy, dL=0.001,
    wire_len_fun=lambda L: L - mag_len,
    Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
    r_src=r_src_ur, m_src=m_src,
    m_local_fun=model.m_local_fun, m_moment=0.0,
    lumen_C=lumen_C, lumen_R=lumen_R,
    N=30, maxiter=50, use_lumen = True,
    u_init=u0
)

pE = hist[-1]["p"]
info = hist[-1]["info"]
viol = lumen_violation_profile(pE, lumen_C, lumen_R)
print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
print("mean positive violation [m] =", np.maximum(viol,0).mean())
print(info["parts"])
print("energy-min tip:", pE[:, -1])
s_cmp = info["s"]             
p_energy = pE

q = q0_ur
R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
t0 = R0 @ np.array([-1.0, 0.0, 0.0])   
p_straight = p0_ur.reshape(3,1) + t0.reshape(3,1) * s_cmp.reshape(1,-1)

plot_energy_only_3d(
    p_energy,
    lumen_C=lumen_C, lumen_R=lumen_R,
    p0=p0_ur, p_straight=None,
    title="Energy-min + Lumen"
)
