import numpy as np
from beam_direction_magnetisation.magnetism.parameters_cosserat import mag_epm
from scipy.spatial.transform import Rotation as Rot
def R_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])
def R_z(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def epm_pose_overhead_spin_z(r_tip, rho, phi_z, m0_dir=np.array([-1.0, 0.0, 0.0])):
    r_src = r_tip + np.array([0, 0.0, rho])  
    Rm = R_z(phi_z)                             
    m_src = mag_epm * (Rm @ m0_dir)              
    return r_src, m_src

def epm_pose_front(r_tip, rho, theta_z, theta_y, m0_dir=np.array([1.0,0.0,0.0])):
    v0 = np.array([rho, 0.0, 0.0])      
    Rm = R_y(theta_y) @ R_z(theta_z)       

    r_src = r_tip + Rm @ v0            
    m_src = mag_epm * (Rm @ m0_dir) *1   
    return r_src, m_src
def ur_pose6_to_T(pose6):
    """
    UR RTDE TCP pose6: [x, y, z, rx, ry, rz]
    where [rx,ry,rz] is rotation vector (axis-angle), radians.
    Returns 4x4 transform.
    """
    pose6 = np.asarray(pose6, float).ravel()
    if pose6.size != 6:
        raise ValueError("Expected UR pose6 = [x,y,z,rx,ry,rz]")

    p = pose6[:3]
    rvec = pose6[3:6]
    Rm = Rot.from_rotvec(rvec).as_matrix()

    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T