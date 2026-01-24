import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def Rx(roll):
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def Ry(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c,0,s],[0,1,0],[-s,0,c]], float)

def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def R_from_rpy_zyx(roll, pitch, yaw):
    # ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll)
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

def rot_to_quat_wxyz(R):
    # your existing implementation is fine; keep ONE copy only
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], float)
    q = q / (np.linalg.norm(q) + 1e-12)

    # optional but recommended: enforce a unique sign to reduce flips
    if q[0] < 0:
        q = -q
    return q

def unpack_pose_euler_L(p):
    p = np.asarray(p, float).ravel()
    assert p.size == 7
    r_src = p[0:3]
    roll, pitch, yaw = p[3:6]
    L = float(p[6])

    # IMPORTANT: wrap angles here so Euler->quat is consistent step-to-step
    roll  = wrap_pi(roll)
    pitch = np.clip(pitch, -np.pi/2, np.pi/2)  # if that's your chosen constraint
    yaw   = wrap_pi(yaw)

    R = R_from_rpy_zyx(roll, pitch, yaw)
    q_src = rot_to_quat_wxyz(R)
    return r_src, q_src, L
def quat_to_rot_wxyz(q):
    """Quaternion q in [w,x,y,z]"""
    q = np.asarray(q, float).ravel()
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)
def rot_to_euler_zyx(R):
    """
    Returns roll, pitch, yaw such that:
      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    R = np.asarray(R, float).reshape(3,3)

    # pitch
    pitch = np.arcsin(np.clip(-R[2,0], -1.0, 1.0))

    c = np.cos(pitch)
    if abs(c) < 1e-9:
        # gimbal lock: choose roll=0
        roll = 0.0
        yaw  = np.arctan2(-R[0,1], R[1,1])
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw  = np.arctan2(R[1,0], R[0,0])

    return roll, pitch, yaw