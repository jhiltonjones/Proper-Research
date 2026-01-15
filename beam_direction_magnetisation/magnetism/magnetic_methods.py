import numpy as np
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_normalize, quat_to_rot

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

    r_pts = p.T                            # (N,3)
    B = dipole_field_from_source(r_pts, r_src, m_ext, r_min=r_min).T     # (3,N)

    f = magnetic_force_analytical(r_pts, m_pts.T, r_src, m_ext, r_min=r_min).T  # (3,N)
    # f=0*f
    tau = np.cross(m_pts.T, B.T).T         # (3,N)

    return f, tau, B