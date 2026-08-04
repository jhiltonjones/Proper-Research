import numpy as np 
from proper_research.parameters import default_beam_params
beam_params =default_beam_params()
mu_tip = beam_params.mag * beam_params.A_cs
# EI  = beam_params.E * beam_params.I
# GJ = beam_params.G * beam_params.J
# m_tip_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
# def m_local_profile(s, m_tip_front_or_overhead):
#     mask = ((s >= s_m) & (s <= s_m + ell_m)).astype(float)
#     return m_tip_front_or_overhead[:, None] * mask[None, :]
def smooth_top_hat(s, s0, s1, eps):
    return 0.5*(np.tanh((s - s0)/eps) - np.tanh((s - s1)/eps))
def rod_section_stiffness(r, E, nu):
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    EA = E * A
    EI = E * I
    GJ = G * J

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": EA,
        "EI": EI,
        "GJ": GJ,
    }

# EI_wire, GJ_wire = 3.29e-6, 2.21e-6
# EI_wire, GJ_wire = 0.19 * EI, 0.19* GJ
# EI_tip,  GJ_tip  = 1 * EI, 1* GJ
# print(f"EI wire is : {EI_wire}, and GJ_wire : {GJ_wire}")
# print(f"EI tip is : {EI_tip}, and GJ tip : {GJ_tip}")
rod = rod_section_stiffness(
    r=beam_params.r,
    E=beam_params.E,
    nu=0.49,
)

EA_wire = rod["EA"]
EI_wire = rod["EI"]
GJ_wire = rod["GJ"]

EA_tip = rod["EA"]
EI_tip = rod["EI"]
GJ_tip = rod["GJ"]
def Kbt_inv_profile(s, len_wire, bend_soft=2, tors_soft=2):
    mask_tip = (s >= len_wire)

    EI_s = np.where(mask_tip, EI_tip, EI_wire)
    GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)
    # print(f"EI wire is : {EI_wire}, and GJ_wire : {GJ_wire}")
    # print(f"EI tip is : {EI_tip}, and GJ tip : {GJ_tip}")
    Kinv = np.zeros((3, 3, s.size))
    Kinv[0, 0, :] = tors_soft / GJ_s
    Kinv[1, 1, :] = bend_soft / EI_s
    Kinv[2, 2, :] = bend_soft / EI_s
    return Kinv


def m_local_wire_plus_magnetised_tip(
    s,
    len_wire,
    *,
    len_tip,
    alpha_end=0.0,
    mode="axial",
    eps=1e-3,
):
    s = np.atleast_1d(s)
    N = s.size
    w = smooth_top_hat(s, len_wire, len_wire + len_tip, eps)

    m = np.zeros((3, N))

    if mode == "axial":
        m[0, :] = -mu_tip
    elif mode == "constant":
        alpha_s = alpha_end * np.ones_like(s)
        m[0, :] = -mu_tip * np.cos(alpha_s)
        m[1, :] = -mu_tip * np.sin(alpha_s)
    elif mode == "ramp":
        xi = (s - len_wire) / (len_tip + 1e-12)
        xi = np.clip(xi, 0.0, 1.0)
        alpha_s = xi * alpha_end
        m[0, :] = -mu_tip * np.cos(alpha_s)
        m[1, :] = -mu_tip * np.sin(alpha_s)
    else:
        raise ValueError(f"Unknown mode='{mode}'")

    return m * w[None, :]

def make_m_local_fun_wire_tip(len_wire, *, len_tip, alpha_end=0.0, mode="axial", eps=1e-3):
    def _m_local(s, _unused=None):
        return m_local_wire_plus_magnetised_tip(
            s, len_wire, len_tip=len_tip, alpha_end=alpha_end, mode=mode, eps=eps
        )
    return _m_local
def make_uniform_axial_m_local_fun(moment_per_length):
    moment_per_length = float(moment_per_length)

    def m_local_fun(s_mid, unused_parameter):
        s_mid = np.asarray(s_mid, float).reshape(-1)

        m_local = np.zeros((3, s_mid.size))

        # The undeformed beam tangent in this model is local -x.
        m_local[0, :] = -moment_per_length

        return m_local

    return m_local_fun