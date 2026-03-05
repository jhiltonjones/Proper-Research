import numpy as np 
from proper_research.parameters import default_beam_params
beam_params =default_beam_params()
mu_tip = beam_params.mag * beam_params.A_cs
EI  = beam_params.E * beam_params.I
GJ = beam_params.G * beam_params.J
# m_tip_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
# def m_local_profile(s, m_tip_front_or_overhead):
#     mask = ((s >= s_m) & (s <= s_m + ell_m)).astype(float)
#     return m_tip_front_or_overhead[:, None] * mask[None, :]
def smooth_top_hat(s, s0, s1, eps):
    return 0.5*(np.tanh((s - s0)/eps) - np.tanh((s - s1)/eps))


EI1, GJ1 = EI, GJ        
EI2, GJ2 = EI, GJ  


def Kbt_inv_profile(s, s_k, bend_soft=2, tors_soft=2):
    mask = (s >= s_k)
    EI_s = np.where(mask, EI2, EI1)
    GJ_s = np.where(mask, GJ2, GJ1)

    Kinv = np.zeros((3,3,s.size))
    Kinv[0,0,:] = tors_soft / GJ_s
    Kinv[1,1,:] = bend_soft / EI_s
    Kinv[2,2,:] = bend_soft / EI_s
    return Kinv



def m_local_wire_plus_magnetised_tip(
    s,
    len_wire,
    *,
    len_tip,               # <-- NEW (meters)
    alpha_end=0.0,
    mode="axial",
    eps=1e-3,
):
    s = np.atleast_1d(s)
    N = s.size

    # gate magnetisation into tip region only
    w = smooth_top_hat(s, len_wire, len_wire + len_tip, eps)  # (N,)

    if mode == "axial":
        alpha_s = np.zeros_like(s)
    elif mode == "constant":
        alpha_s = alpha_end * np.ones_like(s)
    elif mode == "ramp":
        xi = (s - len_wire) / (len_tip + 1e-12)
        xi = np.clip(xi, 0.0, 1.0)
        alpha_s = xi * alpha_end
    else:
        raise ValueError(f"Unknown mode='{mode}'")

    m = np.zeros((3, N))
    m[0, :] = mu_tip * np.cos(alpha_s)
    m[1, :] = mu_tip * np.sin(alpha_s)
    m[2, :] = 0.0

    return m * w[None, :]
def make_m_local_fun_wire_tip(len_wire, *, len_tip, alpha_end=0.0, mode="axial", eps=1e-3):
    def _m_local(s, _unused=None):
        return m_local_wire_plus_magnetised_tip(
            s, len_wire, len_tip=len_tip, alpha_end=alpha_end, mode=mode, eps=eps
        )
    return _m_local