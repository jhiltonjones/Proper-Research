import numpy as np
from scipy.spatial.transform import Rotation as Rot

def quat_normalize(q, eps=1e-12):
    n = np.linalg.norm(q, axis=0, keepdims=True)
    n = np.maximum(n, eps)
    return q / n


def quat_to_rot(q):

    qw, qx, qy, qz = q
    N = qw.size
    R = np.empty((N, 3, 3))

    R[:,0,0] = 1 - 2*(qy*qy + qz*qz)
    R[:,0,1] = 2*(qx*qy - qw*qz)
    R[:,0,2] = 2*(qx*qz + qw*qy)

    R[:,1,0] = 2*(qx*qy + qw*qz)
    R[:,1,1] = 1 - 2*(qx*qx + qz*qz)
    R[:,1,2] = 2*(qy*qz - qw*qx)

    R[:,2,0] = 2*(qx*qz - qw*qy)
    R[:,2,1] = 2*(qy*qz + qw*qx)
    R[:,2,2] = 1 - 2*(qx*qx + qy*qy)

    return R

def quat_derivative_body(q, u):

    qw, qx, qy, qz = q
    ux, uy, uz = u

    dq = np.zeros_like(q)
    dq[0] = -0.5*(qx*ux + qy*uy + qz*uz)
    dq[1] =  0.5*(qw*ux + qy*uz - qz*uy)
    dq[2] =  0.5*(qw*uy + qz*ux - qx*uz)
    dq[3] =  0.5*(qw*uz + qx*uy - qy*ux)
    return dq

def quat_to_R(q):
    qn = quat_normalize(q.reshape(4, 1))
    return quat_to_rot(qn)[0]  # now correct: (1,3,3)[0] -> (3,3)
def T_to_p_quat_wxyz(T):
    """
    Convert 4x4 transform -> position (3,) and quaternion [w,x,y,z].
    """
    p = T[:3, 3].copy()
    Rm = T[:3, :3]
    q_xyzw = Rot.from_matrix(Rm).as_quat()  # [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
    q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
    return p, q_wxyz
def quat_wxyz_normalize(qwxyz):
    q = np.asarray(qwxyz, float).copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n
def quat_wxyz_mul(q1, q2):
    # (w,x,y,z) ⊗ (w,x,y,z)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], float)
def rotvec_to_quat_wxyz(rvec):
    r = Rot.from_rotvec(np.asarray(rvec, float))
    q_xyzw = r.as_quat()  # [x,y,z,w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)

def quat_wxyz_to_rotvec(qwxyz):
    qw, qx, qy, qz = quat_wxyz_normalize(qwxyz)
    r = Rot.from_quat([qx, qy, qz, qw])  # xyzw
    return r.as_rotvec()

def small_rot_quat_wxyz(dphi):
    # dphi is a small rotation vector in radians (axis*angle)
    dphi = np.asarray(dphi, float).ravel()
    a = np.linalg.norm(dphi)
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    axis = dphi / a
    half = 0.5 * a
    return np.array([np.cos(half), *(np.sin(half) * axis)], float)
def unit(v, eps=1e-12):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)


