from beam_direction_magnetisation.cosserat_w_minimal_energy import (make_lumen_centerline_turning, CosseratForwardModel, 
                                                                    dipole_from_pose, u0_from_bvp, solve_quasistatic_insertion, 
                                                                    lumen_violation_profile)
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d,make_lumen_centerline_double_turn                                                                    
from proper_research.parameters import default_magnet_params, default_beam_params
import numpy as np
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
import numpy as np
import matplotlib.pyplot as plt
from proper_research.control.mpc_boundary import resample_polyline
from scipy.spatial.transform import Rotation as Rot
import numpy as np
def unit(v, eps=1e-12):
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    return v / (n + eps)

def bend_angles_from_centerline(pE):
    """
    pE: (3, Nnodes) centerline points from solve_quasistatic_insertion
    Returns:
      bend_total_deg: angle between base tangent and tip tangent
      bend_tip_local_deg: angle between last two segment tangents (local tip bend)
    """
    P = np.asarray(pE, float)
    assert P.shape[0] == 3 and P.shape[1] >= 3

    # segment tangents
    t_base = unit(P[:, 1] - P[:, 0])
    t_tip  = unit(P[:, -1] - P[:, -2])

    # total bend
    c0 = float(np.clip(np.dot(t_base, t_tip), -1.0, 1.0))
    bend_total_deg = float(np.degrees(np.arccos(c0)))

    # local tip bend (optional)
    t_tip_prev = unit(P[:, -2] - P[:, -3])
    c1 = float(np.clip(np.dot(t_tip_prev, t_tip), -1.0, 1.0))
    bend_tip_local_deg = float(np.degrees(np.arccos(c1)))

    return bend_total_deg, bend_tip_local_deg
def quat_wxyz_to_xyzw(qwxyz):
    qw, qx, qy, qz = qwxyz
    return np.array([qx, qy, qz, qw], float)

def quat_xyzw_to_wxyz(qxyzw):
    qx, qy, qz, qw = qxyzw
    return np.array([qw, qx, qy, qz], float)

def rotate_quat_about_local_z_wxyz(qwxyz, angle_rad):
    """
    Rotate the *pose* about its LOCAL +Z axis by angle_rad.
    This is a right-multiply: R_new = R_old * Rz(angle).
    """
    R0 = Rot.from_quat(quat_wxyz_to_xyzw(qwxyz))
    Rz = Rot.from_rotvec([0.0, 0.0, angle_rad])
    Rn = R0 * Rz
    return quat_xyzw_to_wxyz(Rn.as_quat())

def rotate_quat_about_world_z_wxyz(qwxyz, angle_rad):
    """
    Rotate the *pose* about WORLD +Z axis by angle_rad.
    This is a left-multiply: R_new = Rz(angle) * R_old.
    """
    R0 = Rot.from_quat(quat_wxyz_to_xyzw(qwxyz))
    Rz = Rot.from_rotvec([0.0, 0.0, angle_rad])
    Rn = Rz * R0
    return quat_xyzw_to_wxyz(Rn.as_quat())

def dipole_world_dir_from_pose(qwxyz, m_body):
    """
    Returns the dipole direction in world coordinates (3,).
    Helpful sanity check: if this doesn't change with your sweep,
    your rotation isn't being applied the way you think.
    """
    R0 = Rot.from_quat(quat_wxyz_to_xyzw(qwxyz)).as_matrix()
    return R0 @ np.asarray(m_body, float).reshape(3,)
beam_params = default_beam_params()
mag_params = default_magnet_params()
L_min_energy = 0.059
mag_len = beam_params.length_of_mag
m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
pivot_point = np.array([
0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)


# start_point = np.array([
# 0.6681328220229531, -0.7055298925316631, 0.1517853768068757, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
# ], float)
start_point = np.array([
0.696,-0.639,-0.090 ,-3.047,-0.546,+0.049
], float)


t = start_point[:3]
rvec = start_point[3:]

# R0 = Rot.from_rotvec(rvec)

# R_spin_localZ = Rot.from_rotvec([0, 0, np.pi])   # 180° about local Z
# R_new = R0 * R_spin_localZ                     # right-multiply = local rotation

# start_point = np.hstack([t, R_new.as_rotvec()])
# start_point[2] -=0.25
wire_len = L_min_energy - mag_len
T_ur_pivot = ur_pose6_to_T(pivot_point)   
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

T_ur_mag = ur_pose6_to_T(start_point)      
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

q = q0_ur
R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
t0 = R0 @ np.array([-1.0, 0.0, 0.0])   

s_straight = 0.01

# lumen_C = make_lumen_centerline_turning(
#     p_start=p0_ur,
#     t0=t0,
#     length=0.06 + s_straight,     
#     n_pts=130,                      
#     bend_axis=np.array([0.0, 0.0, 1.0]),
#     bend_angle=np.deg2rad(-85.0),
#     bend_start=0.01 + s_straight,    
#     bend_end=0.06 + s_straight       
# )
lumen_C = make_lumen_centerline_double_turn(
    p0_ur, t0,
    length=0.08, n_pts=60,
    bend_axis=np.array([0., 0., 1.]),
    bend1_angle=np.deg2rad(90.0),
    bend1_start=0.01, bend1_end=0.03,
    bend2_angle=np.deg2rad(-90.0),
    bend2_start=0.03, bend2_end=0.05,  # <= length (0.08)
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
    L0=0.01, Lf=L_min_energy, dL=0.001,
    wire_len_fun=lambda L: L - mag_len,
    Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
    r_src=r_src_ur, m_src=m_src,
    m_local_fun=model.m_local_fun, m_moment=0.0,
    lumen_C=lumen_C, lumen_R=lumen_R,
    N=35, maxiter=70, use_lumen = False,
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

# magnet pose as pose8-ish (x,y,z,qw,qx,qy,qz,Ldummy)
p_mag = np.hstack([r_src_ur, q_src_ur, 0.0])  # L not used by plot; any float is fine

plot_energy_only_3d(
    p_energy,
    lumen_C=lumen_C, lumen_R=lumen_R,
    p0=p0_ur, p_straight=None,
    p_mag=p_mag,               # <-- ADD THIS
    mag_axis="x",              # matches your m_body = [mag_epm,0,0] convention
    mag_arrow_len=0.02,
    title="Energy-min + Lumen", show=True
)
# import matplotlib.pyplot as plt
# import numpy as np

# # Choose which axis to rotate about:
# #   local Z (usually what people mean by "spin the tool about its own axis")
# #   world Z (spin about global vertical)
# use_local_z = True

# rot_fn = rotate_quat_about_local_z_wxyz if use_local_z else rotate_quat_about_world_z_wxyz


# angles_deg = np.linspace(-90, 90, 37)
# tips = []
# dip_dirs = []
# viol_max = []

# bend_total_deg_list = []
# bend_tip_local_deg_list = []

# u_init = None

# for a_deg in angles_deg:
#     a_rad = np.deg2rad(a_deg)

#     q_src_rot = rot_fn(q_src_ur, a_rad)
#     dip_dirs.append(dipole_world_dir_from_pose(q_src_rot, m_body))

#     m_src_rot = dipole_from_pose(q_src_rot, m_body)

#     hist_a = solve_quasistatic_insertion(
#         p0=p0_ur, q0=q0_ur,
#         L0=0.01, Lf=L_min_energy, dL=0.001,
#         wire_len_fun=lambda L: L - mag_len,
#         Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
#         r_src=r_src_ur, m_src=m_src_rot,
#         m_local_fun=model.m_local_fun, m_moment=0.0,
#         lumen_C=lumen_C, lumen_R=lumen_R,
#         N=35, maxiter=70,
#         use_lumen=False,
#         u_init=u_init
#     )

#     pE_a = hist_a[-1]["p"]           # (3, Nnodes)
#     tip_a = pE_a[:, -1].copy()
#     tips.append(tip_a)

#     # bending angles
#     bend_total_deg, bend_tip_local_deg = bend_angles_from_centerline(pE_a)
#     bend_total_deg_list.append(bend_total_deg)
#     bend_tip_local_deg_list.append(bend_tip_local_deg)

#     # lumen violation diagnostic
#     viol_a = lumen_violation_profile(pE_a, lumen_C, lumen_R)
#     viol_max.append(float(np.max(viol_a)))

# tips = np.asarray(tips)
# dip_dirs = np.asarray(dip_dirs)
# viol_max = np.asarray(viol_max)
# bend_total_deg_list = np.asarray(bend_total_deg_list)
# bend_tip_local_deg_list = np.asarray(bend_tip_local_deg_list)

# plt.figure()
# plt.plot(angles_deg, tips[:, 1], marker="o")
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("tip y (m)")
# plt.title("Tip y vs Z rotation")
# plt.grid(True)

# plt.figure()
# plt.plot(angles_deg, bend_total_deg_list, marker="s")
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("bend angle (deg)")
# plt.title("Beam end-to-end bend angle vs Z rotation")
# plt.grid(True)

# plt.show()
# # Plot tip components vs angle
# plt.figure()
# plt.plot(angles_deg, tips[:, 0])
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("tip x (m)")
# plt.title("Tip x vs Z rotation")
# plt.grid(True)

# plt.figure()
# plt.plot(angles_deg, tips[:, 1])
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("tip y (m)")
# plt.title("Tip y vs Z rotation")
# plt.grid(True)

# plt.figure()
# plt.plot(angles_deg, tips[:, 2])
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("tip z (m)")
# plt.title("Tip z vs Z rotation")
# plt.grid(True)

# # Plot dipole direction vs angle (sanity check that rotation is actually changing the field direction)
# plt.figure()
# plt.plot(angles_deg, dip_dirs[:, 0], label="m_world_x")
# plt.plot(angles_deg, dip_dirs[:, 1], label="m_world_y")
# plt.plot(angles_deg, dip_dirs[:, 2], label="m_world_z")
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("dipole direction (arb)")
# plt.title("World dipole direction vs Z rotation (sanity check)")
# plt.grid(True)
# plt.legend()

# # Plot XY scatter colored by angle
# plt.figure()
# sc = plt.scatter(tips[:, 0], tips[:, 1], c=angles_deg, s=25)
# plt.xlabel("tip x (m)")
# plt.ylabel("tip y (m)")
# plt.title("Tip XY vs Z rotation (color=deg)")
# plt.grid(True)
# plt.colorbar(sc, label="deg")

# # Lumen violation vs angle
# plt.figure()
# plt.plot(angles_deg, viol_max)
# plt.xlabel("Z rotation (deg)")
# plt.ylabel("max lumen violation (m)")
# plt.title("Max lumen violation vs Z rotation")
# plt.grid(True)

# plt.show()

# # Quick numeric summary (especially for your “why is y barely changing?” question)
# k0 = int(np.argmin(np.abs(angles_deg - 0.0)))
# dy = tips[:, 1] - tips[k0, 1]
# print("Baseline tip @ 0deg:", tips[k0])
# print("dy range (mm):", (1e3 * (dy.max() - dy.min())))
# print("max |dy| (mm):", 1e3 * np.max(np.abs(dy)))
# k_best = int(np.argmax(np.abs(dy)))
# best_deg = float(angles_deg[k_best])
# print("Best angle for |dy|:", best_deg, "deg, tip:", tips[k_best], "dy (mm):", 1e3 * dy[k_best])

# q_best = rot_fn(q_src_ur, np.deg2rad(best_deg))
# m_src_best = dipole_from_pose(q_best, m_body)

# hist_best = solve_quasistatic_insertion(
#     p0=p0_ur, q0=q0_ur,
#     L0=0.01, Lf=L_min_energy, dL=0.001,
#     wire_len_fun=lambda L: L - mag_len,
#     Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
#     r_src=r_src_ur, m_src=m_src_best,
#     m_local_fun=model.m_local_fun, m_moment=0.0,
#     lumen_C=lumen_C, lumen_R=lumen_R,
#     N=35, maxiter=70, use_lumen=False,
#     u_init=None
# )

# pE_best = hist_best[-1]["p"]
# p_mag_best = np.hstack([r_src_ur, q_best, 0.0])

# plot_energy_only_3d(
#     pE_best,
#     lumen_C=lumen_C, lumen_R=lumen_R,
#     p0=p0_ur, p_straight=None,
#     p_mag=p_mag_best,
#     mag_axis="x",
#     mag_arrow_len=0.02,
#     title=f"Energy-min + Lumen (Z rot = {best_deg:.1f} deg)",
#     show=True
# )