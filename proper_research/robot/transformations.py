import numpy as np
pivot_point = np.array([
0.7681328220229531, -0.7112731669220016, 0.16,  np.pi, 0.001,0.001
], float)


start_point = np.array([
0.6058941096487977, -0.7112731669220016, 0.16,  np.pi, 0.001,0.001
], float)
# start_point = np.array([
# 0.6158941096487977, -0.7112731669220016, 0.16,  np.pi, 0.001,0.001
# ], float)
def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])

def rotx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def transGen(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def rotvec_to_R(r):
    r = np.asarray(r)
    theta = np.linalg.norm(r)
    if theta < 1e-9:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array([
        [0,    -kz,   ky],
        [kz,    0,   -kx],
        [-ky,  kx,    0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def R_to_rotvec(R):
    R = np.asarray(R)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    k = np.array([rx, ry, rz])
    return theta * k



def rotate_around_point_transform(axis, pivot_pos, theta):

    if axis in ('x', 1):
        R = rotx(theta)
    elif axis in ('y', 2):
        R = roty(theta)
    else:
        R = rotz(theta)

    c = np.asarray(pivot_pos).reshape(3) 
    I = np.eye(3)
    t = (I - R) @ c                 
    return transGen(R, t)


def quat_normalize(q):
    n = np.linalg.norm(q, axis=0)
    n = np.maximum(n, 1e-18)
    return q / n
def quat_mul(q, p):
    """
    Hamilton product for arrays.
    q, p: (4,N) each
    returns (4,N)
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.vstack([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])
def quat_to_R(q):
    qw, qx, qy, qz = q
    # Rotation matrix components
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    R = np.stack([
        np.stack([R11, R12, R13], axis=-1),
        np.stack([R21, R22, R23], axis=-1),
        np.stack([R31, R32, R33], axis=-1)
    ], axis=-2)  # (N,3,3)
    return R
def quat_to_R_single(q):
    q = q / max(np.linalg.norm(q), 1e-18)
    qw, qx, qy, qz = q
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)
    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)
    return np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])

import numpy as np

# --- your rotvec_to_R and R_to_rotvec here ---

def pose_to_T(pose6):
    x, y, z, rx, ry, rz = map(float, pose6)
    R = rotvec_to_R([rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def T_to_pose(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    x, y, z = T[:3, 3]
    rv = R_to_rotvec(T[:3, :3])
    return np.array([x, y, z, rv[0], rv[1], rv[2]], dtype=float)

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# If your input frame has x/y swapped relative to intended update frame:
def preprocess_update_pose(p):
    p = np.array(p, dtype=float).copy()
    p[0], p[1] = p[1], p[0]  # swap x/y
    return p


T_R_G = np.eye(4)
T_R_G[:3,:3] = rotvec_to_R(pivot_point[3:])   # include orientation!
T_R_G[:3, 3] = pivot_point[:3]

R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]

def tcp_from_global(pose6_in_G):
    dx, dy, dz, rx, ry, rz = map(float, pose6_in_G)

    T = np.eye(4)

    # Rotation: pivot orientation + relative rotation in global
    T[:3, :3] = R0 @ rotvec_to_R([rx, ry, rz])

    # Translation: UR-base translation from pivot
    T[:3, 3] = t0 + np.array([dx, dy, dz], float)

    return T_to_pose(T)


tcp_start = tcp_from_global([-0.15, 0.0, 0.25, 0,0,0])
tcp_start2 = tcp_from_global([-0.17, 0.2, 0.25, 0,0,0])
# print("tcp_start:", tcp_start)
# print("tcp_start:", tcp_start2)
# print("diff to start_point:", tcp_start - tcp_start2)
import numpy as np

def quat_normalize_wxyz(q):
    q = np.asarray(q, float).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_wxyz_to_R_single(q_wxyz):
    q = quat_normalize_wxyz(q_wxyz)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], float)

def R_to_rotvec_ur(R):
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ], float) / (2.0*np.sin(theta))
    return axis * theta

def rot_err_angle(Ra, Rb):
    R = Ra.T @ Rb
    tr = np.trace(R)
    cosang = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    return float(np.arccos(cosang))

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], float)
def R_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion [w,x,y,z]."""
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # find the largest diagonal element and proceed
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    q = np.array([w,x,y,z], float)
    return q / (np.linalg.norm(q) + 1e-12)
def quat_from_R_wxyz(R):
    # You already have R_to_quat_wxyz; use that if you prefer.
    return R_to_quat_wxyz(R)

def test_orbit_about_global_origin():
    # pick a starting point in GLOBAL (relative to pivot origin)
    p0 = np.array([-0.15, 0.0, 0.25], float)   # your start offset, in UR-base translation convention

    for deg in [-60, -30, 0, 30, 60]:
        th = np.deg2rad(deg)

        # rotate position around global Z about origin
        p_th = Rz(th) @ p0

        # set orientation as yaw about global Z (optional but good to test)
        R_th = Rz(th)
        q_th = quat_from_R_wxyz(R_th)          # quaternion [w,x,y,z]
        rv_th = R_to_rotvec_ur(quat_wxyz_to_R_single(q_th))

        # command TCP using your mapping
        tcp_pose6 = tcp_from_global([p_th[0], p_th[1], p_th[2], rv_th[0], rv_th[1], rv_th[2]])

        # verify translation exactly equals t0 + p_th
        T_tcp = pose_to_T(tcp_pose6)
        t_got = T_tcp[:3, 3]
        t_exp = t0 + p_th
        p_err = np.linalg.norm(t_got - t_exp)

        # verify rotation equals R0 @ R_th (matrix compare)
        R_got = T_tcp[:3, :3]
        R_exp = R0 @ R_th
        ang_err = rot_err_angle(R_exp, R_got)
        com_pose = get_point(0,deg)
        # print(f"deg={deg:>4}  p_th={p_th}  tcp_xyz={repr(tcp_pose6)}, tcp_from_com={com_pose} , DIFF = {tcp_pose6 - com_pose}"
        #       f"p_err={p_err:.3e}  rot_err_deg={np.rad2deg(ang_err):.3e}")
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = [0.0, 0.0, -0.25]
T_M_TCP = inv_T(T_TCP_M)

def tcp_from_magnet_global(magnet_pose6_in_G):
    """
    Input: desired MAGNET pose in global frame G (UR pose6: [x,y,z,rx,ry,rz])
           where x,y,z are in UR-base translation axes relative to pivot origin,
           and rotation is relative (composed onto pivot orientation).
    Output: required TCP pose in UR base frame (UR pose6) for moveL().
    """
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Build desired magnet transform in UR base:
    # Rotation: pivot orientation + relative magnet rotation
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])

    # Translation: UR-base translation from pivot
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3, :3] = R_M
    T_R_M[:3, 3]  = p_M

    # Convert desired magnet pose -> TCP command
    T_R_TCP = T_R_M @ T_M_TCP

    return T_to_pose(T_R_TCP)
R_start = rotvec_to_R(start_point[3:])
p_tcp_start = start_point[:3]
p_mag_world_at_start = pivot_point[:3]  # magnet at catheter base

p_TCP_M = R_start.T @ (p_mag_world_at_start - p_tcp_start)  # expressed in TCP frame
# print("p_TCP_M (in TCP frame):", p_TCP_M)
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = p_TCP_M
T_M_TCP = inv_T(T_TCP_M)
R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]

def tcp_from_magnet_global(magnet_pose6_in_G):
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Magnet desired pose in UR base:
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3,:3] = R_M
    T_R_M[:3, 3] = p_M

    # TCP command that places magnet there:
    T_R_TCP = T_R_M @ T_M_TCP
    return T_to_pose(T_R_TCP)
def pose6_in_G_from_tcp_pose6_in_R(tcp_pose6_R, pivot_point, T_TCP_M):
    """
    Given TCP pose in UR base (R), compute magnet pose in GLOBAL (G),
    where G is the catheter base frame aligned with pivot orientation/position.
    """
    # ^R T_G is pivot pose (full pose)
    T_R_G = pose_to_T(pivot_point)

    # ^R T_TCP
    T_R_TCP = pose_to_T(tcp_pose6_R)

    # ^R T_M
    T_R_M = T_R_TCP @ T_TCP_M

    # ^G T_M = (^R T_G)^-1 * ^R T_M
    T_G_M = inv_T(T_R_G) @ T_R_M
    return T_to_pose(T_G_M)


def tcp_from_magnet_global(magnet_pose6_in_G, pivot_point, T_TCP_M):
    """
    Magnet pose is expressed in G (catheter base frame).
    Returns TCP pose in UR base R.
    """
    T_R_G = pose_to_T(pivot_point)
    T_G_M = pose_to_T(magnet_pose6_in_G)

    T_R_M = T_R_G @ T_G_M
    T_R_TCP = T_R_M @ inv_T(T_TCP_M)
    return T_to_pose(T_R_TCP)
# 1) Use your calibrated T_TCP_M (built from p_TCP_M)
# T_TCP_M[:3,3] = p_TCP_M

# tcp_com = get_point(0, 30)  # your "known-good" TCP
# mag_G   = pose6_in_G_from_tcp_pose6_in_R(tcp_com, pivot_point, T_TCP_M)
# tcp_rt  = tcp_from_magnet_global(mag_G, pivot_point, T_TCP_M)

# print("tcp_com:", tcp_com)
# print("mag_G:", mag_G)
# print("tcp_rt:", tcp_rt)
# print("tcp diff:", tcp_rt - tcp_com)

import numpy as np

def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])

def rotx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def transGen(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def rotvec_to_R(r):
    r = np.asarray(r)
    theta = np.linalg.norm(r)
    if theta < 1e-9:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array([
        [0,    -kz,   ky],
        [kz,    0,   -kx],
        [-ky,  kx,    0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def R_to_rotvec(R):
    R = np.asarray(R)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    k = np.array([rx, ry, rz])
    return theta * k



def rotate_around_point_transform(axis, pivot_pos, theta):

    if axis in ('x', 1):
        R = rotx(theta)
    elif axis in ('y', 2):
        R = roty(theta)
    else:
        R = rotz(theta)

    c = np.asarray(pivot_pos).reshape(3) 
    I = np.eye(3)
    t = (I - R) @ c                 
    return transGen(R, t)

def get_point(theta_angle_x, theta_angle_z, 
              start_point = start_point, 
              pivot_point = pivot_point):
    # start_point = np.array([0.6832139195419068, -0.5209069210505941, 0.42302409097655347, 2.443031645419655, -1.901367452027098, -0.01920900651044155])

    # pivot_point = np.array([0.823733332875323, -0.5209069210505941, 0.20740971416415823, -2.086667151308778, 2.3466032555651344, 0.04533170327422023])


    ee_pos0 = start_point[:3]
    ee_rvec0 = start_point[3:]
    R_b_e0 = rotvec_to_R(ee_rvec0)
    H_b_e0 = transGen(R_b_e0, ee_pos0)


    mag_offset = np.array([0, 0, .2])
    H_e_m = transGen(np.eye(3), mag_offset)   


    H_b_m0 = H_b_e0 @ H_e_m


    pivot_pos = pivot_point[:3]

    theta_z = np.deg2rad(theta_angle_z)
    H_rot_z = rotate_around_point_transform('z', pivot_pos, theta_z)
    theta_x = np.deg2rad(theta_angle_x)
    H_rot_x = rotate_around_point_transform('x', pivot_pos, theta_x)

    H_rot = H_rot_x @ H_rot_z
    H_b_m1 = H_rot @ H_b_m0

    H_m_e = np.linalg.inv(H_e_m)
    H_b_e1 = H_b_m1 @ H_m_e

    new_pos = H_b_e1[:3, 3]
    new_R   = H_b_e1[:3, :3]
    new_rvec = R_to_rotvec(new_R)

    new_pose_for_robot = np.hstack([new_pos, new_rvec])
    # print("p_m0:", H_b_m0[:3, 3])
    # print("p_m1:", H_b_m1[:3, 3])
    # print("p_e1:", H_b_e1[:3, 3])
    # print("mag z change:", H_b_m1[2, 3] - H_b_m0[2, 3])
    # print("tcp z change:", H_b_e1[2, 3] - H_b_e0[2, 3])
    x_axis = new_R[:, 0]
    y_axis = new_R[:, 1]
    z_axis = new_R[:, 2]

    # print("TCP x-axis:", x_axis)
    # print("TCP y-axis:", y_axis)
    # print("TCP z-axis:", z_axis)
    # print("New EE pose to send to robot:")
    # print(repr(new_pose_for_robot))
    return new_pose_for_robot
def quat_normalize(q):
    n = np.linalg.norm(q, axis=0)
    n = np.maximum(n, 1e-18)
    return q / n
def quat_mul(q, p):
    """
    Hamilton product for arrays.
    q, p: (4,N) each
    returns (4,N)
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.vstack([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])
def quat_to_R(q):
    qw, qx, qy, qz = q
    # Rotation matrix components
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    R = np.stack([
        np.stack([R11, R12, R13], axis=-1),
        np.stack([R21, R22, R23], axis=-1),
        np.stack([R31, R32, R33], axis=-1)
    ], axis=-2)  # (N,3,3)
    return R
def quat_to_R_single(q):
    q = q / max(np.linalg.norm(q), 1e-18)
    qw, qx, qy, qz = q
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)
    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)
    return np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])

import numpy as np

# --- your rotvec_to_R and R_to_rotvec here ---

def pose_to_T(pose6):
    x, y, z, rx, ry, rz = map(float, pose6)
    R = rotvec_to_R([rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def T_to_pose(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    x, y, z = T[:3, 3]
    rv = R_to_rotvec(T[:3, :3])
    return np.array([x, y, z, rv[0], rv[1], rv[2]], dtype=float)

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# If your input frame has x/y swapped relative to intended update frame:
def preprocess_update_pose(p):
    p = np.array(p, dtype=float).copy()
    p[0], p[1] = p[1], p[0]  # swap x/y
    return p




T_R_G = np.eye(4)
T_R_G[:3,:3] = rotvec_to_R(pivot_point[3:])   # include orientation!
T_R_G[:3, 3] = pivot_point[:3]

R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]

def tcp_from_global(pose6_in_G):
    dx, dy, dz, rx, ry, rz = map(float, pose6_in_G)

    T = np.eye(4)

    # Rotation: pivot orientation + relative rotation in global
    T[:3, :3] = R0 @ rotvec_to_R([rx, ry, rz])

    # Translation: UR-base translation from pivot
    T[:3, 3] = t0 + np.array([dx, dy, dz], float)

    return T_to_pose(T)


# tcp_start = tcp_from_global([-0.15, 0.0, 0.25, 0,0,0])
# tcp_start2 = tcp_from_global([-0.17, 0.2, 0.25, 0,0,0])
# print("tcp_start:", tcp_start)
# print("tcp_start:", tcp_start2)
# print("diff to start_point:", tcp_start - tcp_start2)
import numpy as np

def quat_normalize_wxyz(q):
    q = np.asarray(q, float).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_wxyz_to_R_single(q_wxyz):
    q = quat_normalize_wxyz(q_wxyz)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], float)

def R_to_rotvec_ur(R):
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ], float) / (2.0*np.sin(theta))
    return axis * theta

def rot_err_angle(Ra, Rb):
    R = Ra.T @ Rb
    tr = np.trace(R)
    cosang = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    return float(np.arccos(cosang))

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], float)
def R_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion [w,x,y,z]."""
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # find the largest diagonal element and proceed
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    q = np.array([w,x,y,z], float)
    return q / (np.linalg.norm(q) + 1e-12)
def quat_from_R_wxyz(R):
    # You already have R_to_quat_wxyz; use that if you prefer.
    return R_to_quat_wxyz(R)

def test_orbit_about_global_origin():
    # pick a starting point in GLOBAL (relative to pivot origin)
    p0 = np.array([-0.15, 0.0, 0.25], float)   # your start offset, in UR-base translation convention

    for deg in [-60, -30, 0, 30, 60]:
        th = np.deg2rad(deg)

        # rotate position around global Z about origin
        p_th = Rz(th) @ p0

        # set orientation as yaw about global Z (optional but good to test)
        R_th = Rz(th)
        q_th = quat_from_R_wxyz(R_th)          # quaternion [w,x,y,z]
        rv_th = R_to_rotvec_ur(quat_wxyz_to_R_single(q_th))

        # command TCP using your mapping
        tcp_pose6 = tcp_from_global([p_th[0], p_th[1], p_th[2], rv_th[0], rv_th[1], rv_th[2]])

        # verify translation exactly equals t0 + p_th
        T_tcp = pose_to_T(tcp_pose6)
        t_got = T_tcp[:3, 3]
        t_exp = t0 + p_th
        p_err = np.linalg.norm(t_got - t_exp)

        # verify rotation equals R0 @ R_th (matrix compare)
        R_got = T_tcp[:3, :3]
        R_exp = R0 @ R_th
        ang_err = rot_err_angle(R_exp, R_got)
        com_pose = get_point(0,deg)
        # print(f"deg={deg:>4}  p_th={p_th}  tcp_xyz={repr(tcp_pose6)}, tcp_from_com={com_pose} , DIFF = {tcp_pose6 - com_pose}"
        #       f"p_err={p_err:.3e}  rot_err_deg={np.rad2deg(ang_err):.3e}")
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = [0.0, 0.0, -0.25]
T_M_TCP = inv_T(T_TCP_M)

def tcp_from_magnet_global(magnet_pose6_in_G):
    """
    Input: desired MAGNET pose in global frame G (UR pose6: [x,y,z,rx,ry,rz])
           where x,y,z are in UR-base translation axes relative to pivot origin,
           and rotation is relative (composed onto pivot orientation).
    Output: required TCP pose in UR base frame (UR pose6) for moveL().
    """
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Build desired magnet transform in UR base:
    # Rotation: pivot orientation + relative magnet rotation
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])

    # Translation: UR-base translation from pivot
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3, :3] = R_M
    T_R_M[:3, 3]  = p_M

    # Convert desired magnet pose -> TCP command
    T_R_TCP = T_R_M @ T_M_TCP

    return T_to_pose(T_R_TCP)
R_start = rotvec_to_R(start_point[3:])
p_tcp_start = start_point[:3]
p_mag_world_at_start = pivot_point[:3]  # magnet at catheter base

p_TCP_M = R_start.T @ (p_mag_world_at_start - p_tcp_start)  # expressed in TCP frame
# print("p_TCP_M (in TCP frame):", p_TCP_M)
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = p_TCP_M
T_M_TCP = inv_T(T_TCP_M)
R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]

def tcp_from_magnet_global(magnet_pose6_in_G):
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Magnet desired pose in UR base:
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3,:3] = R_M
    T_R_M[:3, 3] = p_M

    # TCP command that places magnet there:
    T_R_TCP = T_R_M @ T_M_TCP
    return T_to_pose(T_R_TCP)
def pose6_in_G_from_tcp_pose6_in_R(tcp_pose6_R, pivot_point, T_TCP_M):
    """
    Given TCP pose in UR base (R), compute magnet pose in GLOBAL (G),
    where G is the catheter base frame aligned with pivot orientation/position.
    """
    # ^R T_G is pivot pose (full pose)
    T_R_G = pose_to_T(pivot_point)

    # ^R T_TCP
    T_R_TCP = pose_to_T(tcp_pose6_R)

    # ^R T_M
    T_R_M = T_R_TCP @ T_TCP_M

    # ^G T_M = (^R T_G)^-1 * ^R T_M
    T_G_M = inv_T(T_R_G) @ T_R_M
    return T_to_pose(T_G_M)
import numpy as np

def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])

def rotx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def transGen(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def rotvec_to_R(r):
    r = np.asarray(r)
    theta = np.linalg.norm(r)
    if theta < 1e-9:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array([
        [0,    -kz,   ky],
        [kz,    0,   -kx],
        [-ky,  kx,    0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def R_to_rotvec(R):
    R = np.asarray(R)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    k = np.array([rx, ry, rz])
    return theta * k



def rotate_around_point_transform(axis, pivot_pos, theta):

    if axis in ('x', 1):
        R = rotx(theta)
    elif axis in ('y', 2):
        R = roty(theta)
    else:
        R = rotz(theta)

    c = np.asarray(pivot_pos).reshape(3) 
    I = np.eye(3)
    t = (I - R) @ c                 
    return transGen(R, t)

# def get_point(theta_angle_x, theta_angle_z, 
#               start_point = np.array([0.633601454734283, -0.5654185162875351, 0.4570517315269843, -3.116988654350607, 0.19059356279735162, 0.028215660130034903]), 
#               pivot_point = np.array([0.7836091530378535, -0.5654053885267907, 0.20700816061967686, -3.116988654350607, 0.19059356279735162, 0.028215660130034903])):
#     # start_point = np.array([0.6832139195419068, -0.5209069210505941, 0.42302409097655347, 2.443031645419655, -1.901367452027098, -0.01920900651044155])

#     # pivot_point = np.array([0.823733332875323, -0.5209069210505941, 0.20740971416415823, -2.086667151308778, 2.3466032555651344, 0.04533170327422023])


#     ee_pos0 = start_point[:3]
#     ee_rvec0 = start_point[3:]
#     R_b_e0 = rotvec_to_R(ee_rvec0)
#     H_b_e0 = transGen(R_b_e0, ee_pos0)


#     mag_offset = np.array([0, 0, .2])
#     H_e_m = transGen(np.eye(3), mag_offset)   


#     H_b_m0 = H_b_e0 @ H_e_m


#     pivot_pos = pivot_point[:3]

#     theta_z = np.deg2rad(theta_angle_z)
#     H_rot_z = rotate_around_point_transform('z', pivot_pos, theta_z)
#     theta_x = np.deg2rad(theta_angle_x)
#     H_rot_x = rotate_around_point_transform('x', pivot_pos, theta_x)

#     H_rot = H_rot_x @ H_rot_z
#     H_b_m1 = H_rot @ H_b_m0

#     H_m_e = np.linalg.inv(H_e_m)
#     H_b_e1 = H_b_m1 @ H_m_e

#     new_pos = H_b_e1[:3, 3]
#     new_R   = H_b_e1[:3, :3]
#     new_rvec = R_to_rotvec(new_R)

#     new_pose_for_robot = np.hstack([new_pos, new_rvec])

#     # print("New EE pose to send to robot:")
#     # print(repr(new_pose_for_robot))
#     return new_pose_for_robot
def quat_normalize(q):
    n = np.linalg.norm(q, axis=0)
    n = np.maximum(n, 1e-18)
    return q / n
def quat_mul(q, p):
    """
    Hamilton product for arrays.
    q, p: (4,N) each
    returns (4,N)
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.vstack([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])
def quat_to_R(q):
    qw, qx, qy, qz = q
    # Rotation matrix components
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    R = np.stack([
        np.stack([R11, R12, R13], axis=-1),
        np.stack([R21, R22, R23], axis=-1),
        np.stack([R31, R32, R33], axis=-1)
    ], axis=-2)  # (N,3,3)
    return R
def quat_to_R_single(q):
    q = q / max(np.linalg.norm(q), 1e-18)
    qw, qx, qy, qz = q
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)
    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)
    return np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])

import numpy as np

# --- your rotvec_to_R and R_to_rotvec here ---

def pose_to_T(pose6):
    x, y, z, rx, ry, rz = map(float, pose6)
    R = rotvec_to_R([rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def T_to_pose(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    x, y, z = T[:3, 3]
    rv = R_to_rotvec(T[:3, :3])
    return np.array([x, y, z, rv[0], rv[1], rv[2]], dtype=float)

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# If your input frame has x/y swapped relative to intended update frame:
def preprocess_update_pose(p):
    p = np.array(p, dtype=float).copy()
    p[0], p[1] = p[1], p[0]  # swap x/y
    return p



T_R_G = np.eye(4)
T_R_G[:3,:3] = rotvec_to_R(pivot_point[3:])   # include orientation!
T_R_G[:3, 3] = pivot_point[:3]

R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]

def tcp_from_global(pose6_in_G):
    dx, dy, dz, rx, ry, rz = map(float, pose6_in_G)

    T = np.eye(4)

    # Rotation: pivot orientation + relative rotation in global
    T[:3, :3] = R0 @ rotvec_to_R([rx, ry, rz])

    # Translation: UR-base translation from pivot
    T[:3, 3] = t0 + np.array([dx, dy, dz], float)

    return T_to_pose(T)


# tcp_start = tcp_from_global([-0.15, 0.0, 0.25, 0,0,0])
# tcp_start2 = tcp_from_global([-0.17, 0.2, 0.25, 0,0,0])
# print("tcp_start:", tcp_start)
# print("tcp_start:", tcp_start2)
# print("diff to start_point:", tcp_start - tcp_start2)
import numpy as np

def quat_normalize_wxyz(q):
    q = np.asarray(q, float).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_wxyz_to_R_single(q_wxyz):
    q = quat_normalize_wxyz(q_wxyz)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], float)

def R_to_rotvec_ur(R):
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ], float) / (2.0*np.sin(theta))
    return axis * theta

def rot_err_angle(Ra, Rb):
    R = Ra.T @ Rb
    tr = np.trace(R)
    cosang = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    return float(np.arccos(cosang))

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], float)
def R_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion [w,x,y,z]."""
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # find the largest diagonal element and proceed
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    q = np.array([w,x,y,z], float)
    return q / (np.linalg.norm(q) + 1e-12)
def quat_from_R_wxyz(R):
    # You already have R_to_quat_wxyz; use that if you prefer.
    return R_to_quat_wxyz(R)

def test_orbit_about_global_origin():
    # pick a starting point in GLOBAL (relative to pivot origin)
    p0 = np.array([-0.15, 0.0, 0.25], float)   # your start offset, in UR-base translation convention

    for deg in [-60, -30, 0, 30, 60]:
        th = np.deg2rad(deg)

        # rotate position around global Z about origin
        p_th = Rz(th) @ p0

        # set orientation as yaw about global Z (optional but good to test)
        R_th = Rz(th)
        q_th = quat_from_R_wxyz(R_th)          # quaternion [w,x,y,z]
        rv_th = R_to_rotvec_ur(quat_wxyz_to_R_single(q_th))

        # command TCP using your mapping
        tcp_pose6 = tcp_from_global([p_th[0], p_th[1], p_th[2], rv_th[0], rv_th[1], rv_th[2]])

        # verify translation exactly equals t0 + p_th
        T_tcp = pose_to_T(tcp_pose6)
        t_got = T_tcp[:3, 3]
        t_exp = t0 + p_th
        p_err = np.linalg.norm(t_got - t_exp)

        # verify rotation equals R0 @ R_th (matrix compare)
        R_got = T_tcp[:3, :3]
        R_exp = R0 @ R_th
        ang_err = rot_err_angle(R_exp, R_got)
        com_pose = get_point(0,deg)
        # print(f"deg={deg:>4}  p_th={p_th}  tcp_xyz={repr(tcp_pose6)}, tcp_from_com={com_pose} , DIFF = {tcp_pose6 - com_pose}"
        #       f"p_err={p_err:.3e}  rot_err_deg={np.rad2deg(ang_err):.3e}")
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = [0.0, 0.0, -0.25]
T_M_TCP = inv_T(T_TCP_M)

def tcp_from_magnet_global(magnet_pose6_in_G):
    """
    Input: desired MAGNET pose in global frame G (UR pose6: [x,y,z,rx,ry,rz])
           where x,y,z are in UR-base translation axes relative to pivot origin,
           and rotation is relative (composed onto pivot orientation).
    Output: required TCP pose in UR base frame (UR pose6) for moveL().
    """
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Build desired magnet transform in UR base:
    # Rotation: pivot orientation + relative magnet rotation
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])

    # Translation: UR-base translation from pivot
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3, :3] = R_M
    T_R_M[:3, 3]  = p_M

    # Convert desired magnet pose -> TCP command
    T_R_TCP = T_R_M @ T_M_TCP

    return T_to_pose(T_R_TCP)
R_start = rotvec_to_R(start_point[3:])
p_tcp_start = start_point[:3]
p_mag_world_at_start = pivot_point[:3]  # magnet at catheter base

p_TCP_M = R_start.T @ (p_mag_world_at_start - p_tcp_start)  # expressed in TCP frame
# print("p_TCP_M (in TCP frame):", p_TCP_M)
T_TCP_M = np.eye(4)
T_TCP_M[:3, 3] = p_TCP_M
T_M_TCP = inv_T(T_TCP_M)
R0 = rotvec_to_R(pivot_point[3:])
t0 = pivot_point[:3]
def quat_wxyz_to_R_single(q_wxyz):
    q = np.asarray(q_wxyz, float).reshape(4,)
    q = q / (np.linalg.norm(q) + 1e-12)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], float)

def R_to_rotvec_ur(R):
    R = np.asarray(R, float).reshape(3,3)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ], float) / (2.0*np.sin(theta))
    return axis * theta

def tcp_from_magnet_global(magnet_pose6_in_G):
    mx, my, mz, mrx, mry, mrz = map(float, magnet_pose6_in_G)

    # Magnet desired pose in UR base:
    R_M = R0 @ rotvec_to_R([mrx, mry, mrz])
    p_M = t0 + np.array([mx, my, mz], float)

    T_R_M = np.eye(4)
    T_R_M[:3,:3] = R_M
    T_R_M[:3, 3] = p_M

    # TCP command that places magnet there:
    T_R_TCP = T_R_M @ T_M_TCP
    return T_to_pose(T_R_TCP)
def pose6_in_G_from_tcp_pose6_in_R(tcp_pose6_R, pivot_point, T_TCP_M):
    """
    Given TCP pose in UR base (R), compute magnet pose in GLOBAL (G),
    where G is the catheter base frame aligned with pivot orientation/position.
    """
    # ^R T_G is pivot pose (full pose)
    T_R_G = pose_to_T(pivot_point)

    # ^R T_TCP
    T_R_TCP = pose_to_T(tcp_pose6_R)

    # ^R T_M
    T_R_M = T_R_TCP @ T_TCP_M

    # ^G T_M = (^R T_G)^-1 * ^R T_M
    T_G_M = inv_T(T_R_G) @ T_R_M
    return T_to_pose(T_G_M)
def tcp_from_magnet_global(magnet_pose6_in_G, pivot_point, T_TCP_M):
    T_R_G   = pose_to_T(pivot_point)            # ^R T_G
    T_G_M   = pose_to_T(magnet_pose6_in_G)      # ^G T_M
    T_R_M   = T_R_G @ T_G_M                     # ^R T_M
    T_R_TCP = T_R_M @ inv_T(T_TCP_M)            # ^R T_TCP
    return T_to_pose(T_R_TCP)
def magnet_pose6_in_G_from_sim(r_src, q_wxyz):
    R = quat_wxyz_to_R_single(q_wxyz)
    rv = R_to_rotvec_ur(R)
    r_src = np.asarray(r_src, float).reshape(3,)
    return np.array([r_src[0], r_src[1], r_src[2], rv[0], rv[1], rv[2]], float)

def tcp_cmd_from_sim_magnet(r_src_G, q_wxyz_G, pivot_point, T_TCP_M):
    mag_pose6_G = magnet_pose6_in_G_from_sim(r_src_G, q_wxyz_G)
    tcp_pose6_R = tcp_from_magnet_global(mag_pose6_G, pivot_point, T_TCP_M)
    return tcp_pose6_R
# r_src = np.array([0.08660254, 0.05, 0.0])
# q_src = np.array([0.96592583, 0.0, 0.0, 0.25881905])

# tcp_cmd = tcp_cmd_from_sim_magnet(r_src, q_src, pivot_point, T_TCP_M)
# print("TCP command:", tcp_cmd)
# tcp_com = get_point(0, 30)
# print("tcp diff:", tcp_cmd - tcp_com)
if __name__ == "__main__":
    new_pose = get_point(0,60)
    print(f"New pose is: {repr(new_pose)}")