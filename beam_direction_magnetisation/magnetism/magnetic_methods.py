import numpy as np
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_normalize, quat_to_rot, quat_to_R

MU0_over_4pi = 1e-7
def magnetic_force_analytical(r_pts, m_pts, r_src, m_src, r_min = 1e-6):
    R = r_pts - r_src[None, :]
    R2 = np.sum(R*R, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, r_min**2))

    if m_pts.ndim == 1:
        m = np.repeat(m_pts[None, :], r_pts.shape[0], axis=0)
    else:
        m = m_pts

    m_dot_R  = np.sum(m * R, axis=1)    
    m0_dot_R = R @ m_src               
    m_dot_m0 = m @ m_src               

    invR5 = 1.0 / (Rnorm**5)
    invR2 = 1.0 / (Rnorm**2)

    term = (
        m_dot_R[:, None]  * m_src[None, :] +
        m0_dot_R[:, None] * m +
        m_dot_m0[:, None] * R -
        5.0*(m_dot_R*m0_dot_R*invR2)[:, None] * R
    )
    F = 3.0 * MU0_over_4pi * invR5[:, None] * term
    return F

def dipole_field_from_source(r_pts, r_src, m_src, r_min=1e-6):

    R = r_pts - r_src[None, :]
    R2 = np.sum(R*R, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, r_min**2))
    Rhat = R / Rnorm[:, None]
    mdot = Rhat @ m_src  
    invR3 = 1.0 / (Rnorm**3)
    B = MU0_over_4pi * invR3[:, None] * (3.0 * mdot[:, None] * Rhat - m_src[None, :])
    return B    

def magnetic_moment(B_r, mu_0, r, p):
    return (B_r / mu_0) * (np.pi * r**2 * p)

def magnetic_wrench_density_cosserat(p, q, m_ext, r_src, m_local, r_min=1e-6):
    qn = quat_normalize(q)
    R = quat_to_rot(qn) 
    m_pts = np.einsum('nij,j->ni', R, m_local) 
    r_pts = p.T 
    B = dipole_field_from_source(r_pts, r_src, m_ext, r_min=r_min)      
    f = magnetic_force_analytical(r_pts, m_pts, r_src, m_ext, r_min=r_min) 
    f=f*-1 
    tau = np.cross(m_pts, B)                                      
    return f.T, tau.T, B.T
def magnetic_wrench_density_cosserat_profile(p, q, s, m_ext, r_src, m_local_fun, m_front_or_overhead, r_min=1e-6):
    qn = quat_normalize(q)
    R = quat_to_rot(qn)                 

    m_loc = m_local_fun(s, m_front_or_overhead)                 # (3,N) body frame
    m_pts = np.einsum('nij,jn->in', R, m_loc)   # (3,N) world frame
    # print("m_loc[:, 0]  =", m_loc[:, 0])
    # print("m_loc[:, -1] =", m_loc[:, -1])
    # print("nonzero m nodes =", np.where(np.linalg.norm(m_loc, axis=0) > 1e-12)[0])
    r_pts = p.T                            # (N,3)
    B = dipole_field_from_source(r_pts, r_src, m_ext, r_min=r_min).T     # (3,N)

    f = magnetic_force_analytical(r_pts, m_pts.T, r_src, m_ext, r_min=r_min).T  # (3,N)
    # f=0*f
    tau = np.cross(m_pts.T, B.T).T         # (3,N)
    # tau = tau*0
    return f, tau, B
def magnetic_wrench_density_cosserat_profile_segments(
    p, q, s, m_ext, r_src, m_local_fun, m_front_or_overhead, r_min=1e-6
):
    """
    Segment-based magnetic quantities evaluated at segment midpoints.

    Parameters
    ----------
    p : ndarray, shape (3, N)
        Node positions.
    q : ndarray, shape (N, 4) or compatible with quat_to_rot
        Node quaternions.
    s : ndarray, shape (N,)
        Node arclength coordinates.
    m_ext : ndarray, shape (3,)
        External/source dipole in world frame.
    r_src : ndarray, shape (3,)
        External/source position in world frame.
    m_local_fun : callable
        Returns local magnetization profile in body frame.
    m_front_or_overhead : any
        Passed through to m_local_fun.
    r_min : float
        Minimum radius for dipole singularity protection.

    Returns
    -------
    f_mid : ndarray, shape (3, N-1)
        Magnetic force density at segment midpoints, world frame.
    tau_mid : ndarray, shape (3, N-1)
        Magnetic torque density at segment midpoints, world frame.
    B_mid : ndarray, shape (3, N-1)
        Magnetic field at segment midpoints, world frame.
    m_mid_world : ndarray, shape (3, N-1)
        Magnetization at segment midpoints, world frame.
    s_mid : ndarray, shape (N-1,)
        Segment midpoint arclengths.
    """
    s = np.asarray(s, float).ravel()
    qn = quat_normalize(q)
    R_nodes = quat_to_rot(qn)   # (N,3,3)

    # segment midpoints in arclength and position
    s_mid = 0.5 * (s[:-1] + s[1:])
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])

    # q is assumed shape (4, N)
    q0 = qn[:, :-1].copy()   # (4, N-1)
    q1 = qn[:, 1:].copy()    # (4, N-1)

    # enforce same hemisphere per segment
    flip = np.sum(q0 * q1, axis=0) < 0.0
    q1[:, flip] *= -1.0

    q_mid = q0 + q1
    q_mid /= (np.linalg.norm(q_mid, axis=0, keepdims=True) + 1e-12)

    R_mid = quat_to_rot(q_mid)   # if quat_to_rot supports (4, N-1)

    # local magnetization in body frame at segment midpoints
    m_loc_mid = np.asarray(m_local_fun(s_mid, m_front_or_overhead), float)  # (3,N-1)

    # rotate to world frame
    m_mid_world = np.einsum('nij,jn->in', R_mid, m_loc_mid)                 # (3,N-1)

    # field at segment midpoints
    r_mid_pts = p_mid.T                                                      # (N-1,3)
    B_mid = dipole_field_from_source(r_mid_pts, r_src, m_ext, r_min=r_min).T  # (3,N-1)

    # force density at segment midpoints
    f_mid = magnetic_force_analytical(
        r_mid_pts, m_mid_world.T, r_src, m_ext, r_min=r_min
    ).T                                                                      # (3,N-1)

    # torque density at segment midpoints
    tau_mid = np.cross(m_mid_world.T, B_mid.T).T                             # (3,N-1)

    return f_mid, tau_mid, B_mid, m_mid_world, s_mid
def magnetic_wrench_density_cosserat_point(
    *,
    s,
    p,
    q,
    m_ext,
    r_src,
    m_local_fun,
    m_front_or_overhead,
    r_min=1e-6,
):
    """
    Pointwise magnetic quantities at one arclength location.

    Parameters
    ----------
    s : float
        Arclength coordinate.
    p : ndarray, shape (3,)
        Position in world frame.
    q : ndarray, shape (4,)
        Quaternion at this location.
    m_ext : ndarray, shape (3,)
        External/source dipole in world frame.
    r_src : ndarray, shape (3,)
        External/source position in world frame.
    m_local_fun : callable
        Returns local magnetization profile in body frame.
    m_front_or_overhead : any
        Passed through to m_local_fun.
    r_min : float
        Minimum radius for dipole singularity protection.

    Returns
    -------
    f_ext : ndarray, shape (3,)
        Magnetic force density in world frame.
    tau_ext : ndarray, shape (3,)
        Magnetic torque density in world frame.
    B : ndarray, shape (3,)
        Magnetic field in world frame.
    m_world : ndarray, shape (3,)
        Local magnetization in world frame.
    """
    s = float(s)
    p = np.asarray(p, float).reshape(3,)
    q = quat_normalize(np.asarray(q, float).reshape(4,))
    m_ext = np.asarray(m_ext, float).reshape(3,)
    r_src = np.asarray(r_src, float).reshape(3,)

    # rotation at this point
    R = quat_to_R(q)   # must return (3,3) for one quaternion

    # body-frame magnetization profile at this point
    m_loc = np.asarray(
        m_local_fun(np.array([s], dtype=float), m_front_or_overhead),
        float
    ).reshape(3,)

    # rotate to world frame
    m_world = R @ m_loc

    # magnetic field at this point
    B = dipole_field_from_source(
        p[None, :], r_src, m_ext, r_min=r_min
    )[0]

    # force density at this point
    f_ext = magnetic_force_analytical(
        p[None, :], m_world[None, :], r_src, m_ext, r_min=r_min
    )[0]

    # torque density at this point
    tau_ext = np.cross(m_world, B)

    return f_ext, tau_ext, B, m_world