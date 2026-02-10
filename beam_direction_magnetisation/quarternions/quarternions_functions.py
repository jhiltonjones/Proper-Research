import numpy as np

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