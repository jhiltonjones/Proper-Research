from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    make_lumen_centerline_turning,
    CosseratForwardModel,
    dipole_from_pose,
    u0_from_bvp,
    solve_quasistatic_insertion,
    lumen_violation_profile,
)
from beam_direction_magnetisation.post_processing.post_processing import (
    plot_energy_only_3d,
    make_lumen_centerline_double_turn,
)
from proper_research.control.no_path_plan_mpc import DeterministicForward6D
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
from proper_research.control.mpc_boundary import resample_polyline
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, effective_lengths
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt
import copy

beam_params = default_beam_params()
mag_params = default_magnet_params()

L_min_energy = 0.05
mag_len = beam_params.length_of_mag
m_body = np.array([mag_params.mag_epm, 0.0, 0.0])

pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1,
    -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)

start_point = np.array([
    0.7281328220229531, -0.7555298925316631, -0.09,
    -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)

# rotate start pose 180 deg about local Z
t = start_point[:3]
rvec = start_point[3:]
R0 = Rot.from_rotvec(rvec)
R_spin_localZ = Rot.from_rotvec([0, 0, np.deg2rad(-35)])
R_new = R0 * R_spin_localZ
start_point = np.hstack([t, R_new.as_rotvec()])

# pivot pose
T_ur_pivot = ur_pose6_to_T(pivot_point)
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

# source magnet pose
T_ur_mag = ur_pose6_to_T(start_point)
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

# initial lumen tangent
Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

s_straight = 0.03
lumen_C = make_lumen_centerline_turning(
    p_start=p0_ur,
    t0=t0,
    length=0.06 + s_straight,
    n_pts=130,
    bend_axis=np.array([0.0, 0.0, 1.0]),
    bend_angle=np.deg2rad(90.0),
    bend_start=0.0 + s_straight,
    bend_end=0.03 + s_straight,
)
lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
lumen_R = np.full(len(lumen_C), 0.004)

forward_model = EnergyMinForwardWithLumen(
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    u_star=np.zeros(3),
    m_body=m_body,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    N_nodes=35,
    maxiter=70,
    L0_init=0.01,
    dL_internal=0.002,
    L_tip_full=0.04,
    L_tip_min=0.01,
    use_lumen_jac=True
)
forward_model_wrong = EnergyMinForwardWithLumen(
    p0_ur=p0_ur,
    q0_ur=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    u_star=np.zeros(3),
    m_body=m_body,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    N_nodes=35,
    maxiter=70,
    L0_init=0.01,
    dL_internal=0.002,
    L_tip_full=0.04,
    L_tip_min=0.01,
    use_lumen_jac=False
)
fwd6 = DeterministicForward6D(forward_model)
fwd6_wrong = DeterministicForward6D(forward_model_wrong)
L_ins = L_min_energy
r_src = start_point[:3]
rvec_src = start_point[3:6]

q_xyzw = Rot.from_rotvec(rvec_src).as_quat()
q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
p8 = np.hstack([r_src, q_wxyz, L_ins])

fwd6.start_step()
y = fwd6(p8, commit=True)

tip_pos = y[:3]
tip_tangent = y[3:6]
centreline = fwd6.last_p_centerline

print("tip position:", tip_pos)
print("tip tangent:", tip_tangent)
print("centreline shape:", None if centreline is None else centreline.shape)
print("tip position:", tip_pos)
print("tip tangent:", tip_tangent)
print("centreline shape:", None if centreline is None else centreline.shape)

p_mag = np.hstack([r_src_ur, q_src_ur, 0.0])

plot_energy_only_3d(
    centreline,
    lumen_C=lumen_C,
    lumen_R=lumen_R,
    p0=p0_ur,
    tip=tip_pos,
    tip_from_centerline=centreline[:, -1],
    p_mag=p_mag,
    mag_axis="x",
    mag_arrow_len=0.02,
    show=True,
)