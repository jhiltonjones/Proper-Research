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
def quat_mul(q1, q2):
    # Hamilton product, q = q1 ⊗ q2, with q = [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_exp_body(u, ds):
    """Quaternion exponential for body strain u over step ds."""
    u = np.asarray(u, float).reshape(3,)
    th = np.linalg.norm(u) * ds
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    axis = u / (np.linalg.norm(u) + 1e-12)
    h = 0.5 * th
    return np.array([np.cos(h), *(np.sin(h) * axis)], float)
def quat_to_R(q):
    qn = quat_normalize(q.reshape(4, 1))
    return quat_to_rot(qn)[0]  # now correct: (1,3,3)[0] -> (3,3)
def quat_wxyz_normalize(q):
    """
    Normalize quaternion in wxyz convention.

    q = [w, x, y, z]
    """
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))

    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)

    q = q / n

    # Optional sign convention: keep scalar part non-negative.
    if q[0] < 0.0:
        q = -q

    return q


def quat_wxyz_multiply(q1, q2):
    """
    Quaternion product q = q1 ⊗ q2 in wxyz convention.
    """
    q1 = quat_wxyz_normalize(q1)
    q2 = quat_wxyz_normalize(q2)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        float,
    )


def quat_wxyz_to_R(q):
    """
    Convert quaternion in wxyz convention to rotation matrix.

    q = [w, x, y, z]
    """
    w, x, y, z = quat_wxyz_normalize(q)

    return np.array(
        [
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
        ],
        float,
    )


def quat_wxyz_from_rotvec(rotvec):
    """
    Convert rotation vector to quaternion in wxyz convention.

    rotvec direction = rotation axis
    rotvec norm      = rotation angle in radians
    """
    rotvec = np.asarray(rotvec, float).reshape(3)
    theta = float(np.linalg.norm(rotvec))

    if theta < 1e-12:
        return quat_wxyz_normalize(
            np.array(
                [
                    1.0,
                    0.5 * rotvec[0],
                    0.5 * rotvec[1],
                    0.5 * rotvec[2],
                ],
                float,
            )
        )

    axis = rotvec / theta
    half = 0.5 * theta

    return np.array(
        [
            np.cos(half),
            *(np.sin(half) * axis),
        ],
        float,
    )


def integrate_pose8_body(p, u, dt):
    """
    Integrate an 8D pose using body/world velocity controls.

    Pose:
        p[0:3] = magnet position in world
        p[3:7] = quaternion wxyz
        p[7]   = insertion length

    Control:
        u[0:3] = translational velocity of magnet centre in world frame
        u[3:6] = angular velocity in body frame, rad/s
        u[6]   = insertion velocity/rate

    Returns:
        p_next, shape (8,)

    Notes:
        - Translational velocity is assumed world-frame.
        - Angular velocity is assumed body-frame.
        - Quaternion update uses:
              q_next = q_current ⊗ delta_q_body
    """
    p = np.asarray(p, float).reshape(8).copy()
    u = np.asarray(u, float).reshape(-1)

    if u.size < 7:
        raise ValueError(f"u must have at least 7 entries, got {u.size}.")

    dt = float(dt)

    p_next = p.copy()

    # Position update.
    p_next[0:3] = p[0:3] + dt * u[0:3]

    # Orientation update.
    q = quat_wxyz_normalize(p[3:7])
    omega_body = u[3:6]
    dq = quat_wxyz_from_rotvec(dt * omega_body)

    q_next = quat_wxyz_multiply(q, dq)
    p_next[3:7] = quat_wxyz_normalize(q_next)

    # Insertion update.
    p_next[7] = p[7] + dt * float(u[6])

    return p_next


def build_Pm_world(dt, Np, m=7):
    """
    Build linear map from stacked controls U to stacked magnet centre positions.

    Assumes:
        u_k[0:3] = world-frame translational velocity

    Pose position dynamics:
        r_{k+1} = r_k + dt * u_k[0:3]

    For stacked U = [u0, u1, ..., u_{Np-1}], this returns Pm such that:

        r_stack = r0_stack + Pm @ U

    where:

        r_stack = [r1, r2, ..., r_Np]

    Shape:
        Pm: (3*Np, Np*m)
    """
    dt = float(dt)
    Np = int(Np)
    m = int(m)

    if Np <= 0:
        raise ValueError("Np must be positive.")

    if m < 3:
        raise ValueError("m must be at least 3 because u[0:3] is translation.")

    Pm = np.zeros((3 * Np, Np * m), float)

    for k in range(Np):
        row = slice(3 * k, 3 * (k + 1))

        for j in range(k + 1):
            col = slice(j * m, j * m + 3)
            Pm[row, col] = dt * np.eye(3)

    return Pm


def unit(v, eps=1e-12):
    """
    Return unit vector.
    """
    v = np.asarray(v, float).reshape(-1)
    n = float(np.linalg.norm(v))

    if n < eps:
        return np.zeros_like(v)

    return v / n
def pose7_rotvec_to_pose8_quat(p7):
    p7 = np.asarray(p7, float).ravel()
    t = p7[0:3]
    rvec = p7[3:6]
    L = p7[6]
    q = rotvec_to_quat_wxyz(rvec)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3], L], float)

def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)
def rotvec_to_quat_wxyz(rvec):
    r = Rot.from_rotvec(np.asarray(rvec, float))
    q_xyzw = r.as_quat()  # [x,y,z,w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
def quat_wxyz_to_rotvec(qwxyz):
    qw, qx, qy, qz = quat_wxyz_normalize(qwxyz)
    r = Rot.from_quat([qx, qy, qz, qw])  # xyzw
    return r.as_rotvec()