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

def get_point(theta_angle_x, theta_angle_z, start_point = np.array([0.8232165993178376, -0.5209241826328589, 0.4533174290989068, -2.0741634614536983, -2.345875896286576, 0.07273609243115334]), pivot_point = np.array([0.963733332875323, -0.5209069210505941, 0.20740971416415823, -2.086667151308778, 2.3466032555651344, 0.04533170327422023])):
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

    # print("New EE pose to send to robot:")
    # print(repr(new_pose_for_robot))
    return new_pose_for_robot
