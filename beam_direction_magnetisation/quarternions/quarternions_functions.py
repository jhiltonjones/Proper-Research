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