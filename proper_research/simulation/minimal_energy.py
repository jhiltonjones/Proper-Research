from beam_direction_magnetisation.cosserat_w_minimal_energy import (make_lumen_centerline_turning, CosseratForwardModel, 
                                                                    dipole_from_pose, u0_from_bvp, solve_quasistatic_insertion, 
                                                                    _set_axes_equal_about_data, lumen_violation_profile,
                                                                    plot_lumen_rings)
from proper_research.parameters import default_magnet_params, default_beam_params
import numpy as np
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T, T_to_p_quat_wxyz
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_only_3d(p_energy, lumen_C=None, lumen_R=None, p0=None, p_straight=None,
                        title="Energy-min centerline", show_rings=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Energy-min centerline
    ax.plot(p_energy[0], p_energy[1], p_energy[2], "--", label="Energy-min (3D)")

    # Optional straight baseline + base
    if p_straight is not None:
        ax.plot(p_straight[0], p_straight[1], p_straight[2], ":", label="Straight baseline")
    if p0 is not None:
        ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="o", label="Base")

    # Vessel visualization
    if (lumen_C is not None) and (lumen_R is not None):
        C = np.asarray(lumen_C, float)
        R = np.asarray(lumen_R, float)
        ax.plot(C[:, 0], C[:, 1], C[:, 2], label="Lumen centerline")

        if show_rings:
            # You already have plot_lumen_rings defined below in your file,
            # OR import it from your module and delete the local copy.
            plot_lumen_rings(ax, C, R, n_theta=28, alpha=0.2, linewidth=0.6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    # Nice equal aspect (optional) using all plotted points
    pts = [p_energy.T]
    if p_straight is not None: pts.append(p_straight.T)
    if p0 is not None: pts.append(np.asarray(p0, float).reshape(1,3))
    if lumen_C is not None: pts.append(np.asarray(lumen_C, float))
    _set_axes_equal_about_data(ax, np.vstack(pts))

    plt.show()
beam_params = default_beam_params()
mag_params = default_magnet_params()
L_min_energy = 0.1
mag_len = beam_params.length_of_mag
m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
pivot_point = np.array([
0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)


start_point = np.array([
0.6681328220229531, -0.7055298925316631, 0.1517853768068757, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)
start_point[2] -=0.25
wire_len = L_min_energy - mag_len
T_ur_pivot = ur_pose6_to_T(pivot_point)   
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

T_ur_mag = ur_pose6_to_T(start_point)      
r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

# base tangent direction (same convention you already use)
q = q0_ur
R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
global INSERTION_DIR_WORLD
INSERTION_DIR_WORLD = t0 / (np.linalg.norm(t0) + 1e-12)
# lumen centerline starts at pivot base and bends
lumen_C = make_lumen_centerline_turning(
    p_start=p0_ur,
    t0=t0,
    length=0.08,              # make it longer than rod so distance queries behave well
    n_pts=80,
    bend_axis=np.array([0.0, 0.0, 1.0]),  # bend in x-y plane
    bend_angle=np.deg2rad(120.0),
    bend_start=0.01,
    bend_end=0.08
)
model = CosseratForwardModel(
    p0=p0_ur,
    q0=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    m_local_fun=make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0),
    m_moment=0.0,
    wire_len=wire_len,
)
lumen_R = np.full(len(lumen_C), 0.004)  # 4 mm radius

m_src = dipole_from_pose(q_src_ur, m_body)

u0 = None  # or np.zeros(3*(N-1)) but N changes; easiest is None
hist = solve_quasistatic_insertion(
    p0=p0_ur, q0=q0_ur,
    L0=L_min_energy, Lf=L_min_energy, dL=0.001,
    wire_len_fun=lambda L: L - mag_len,
    Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
    r_src=r_src_ur, m_src=m_src,
    m_local_fun=model.m_local_fun, m_moment=0.0,
    lumen_C=lumen_C, lumen_R=lumen_R,
    N=15, maxiter=15, use_lumen = True,
    u_init=u0
)
# take final
pE = hist[-1]["p"]
info = hist[-1]["info"]
viol = lumen_violation_profile(pE, lumen_C, lumen_R)
print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
print("mean positive violation [m] =", np.maximum(viol,0).mean())
print(info["parts"])
print("energy-min tip:", pE[:, -1])
# choose a common comparison grid
s_cmp = info["s"]                      # energy-min grid
p_energy = pE

# optional straight baseline (same convention as your earlier straight tip)
# build straight line from base tangent
q = q0_ur
R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
p_straight = p0_ur.reshape(3,1) + t0.reshape(3,1) * s_cmp.reshape(1,-1)

plot_energy_only_3d(
    p_energy,
    lumen_C=lumen_C, lumen_R=lumen_R,
    p0=p0_ur, p_straight=p_straight,
    title="Energy-min + Lumen"
)
