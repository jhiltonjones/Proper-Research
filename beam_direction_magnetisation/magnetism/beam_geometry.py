import numpy as np 
from beam_direction_magnetisation.magnetism.parameters_cosserat import mu_line, alpha, EI, GJ, ell_m,s_m,s_k

mu_tip = mu_line
# m_tip_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
# def m_local_profile(s, m_tip_front_or_overhead):
#     mask = ((s >= s_m) & (s <= s_m + ell_m)).astype(float)
#     return m_tip_front_or_overhead[:, None] * mask[None, :]
def smooth_top_hat(s, s0, s1, eps):
    return 0.5*(np.tanh((s - s0)/eps) - np.tanh((s - s1)/eps))

def m_local_profile(s, m_vec, eps=1e-3):
    w = smooth_top_hat(s, s_m, s_m + ell_m, eps)   # shape (N,)
    return m_vec[:, None] * w[None, :]

EI1, GJ1 = EI, GJ        
EI2, GJ2 = EI, GJ  

def Kbt_inv_profile(s):

    mask = (s >= s_k)
    EI_s = np.where(mask, EI2, EI1)
    GJ_s = np.where(mask, GJ2, GJ1)

    Kinv = np.zeros((3,3,s.size))
    Kinv[0,0,:] = 1.0 / GJ_s
    Kinv[1,1,:] = 1.0 / EI_s
    Kinv[2,2,:] = 1.0 / EI_s
    return Kinv