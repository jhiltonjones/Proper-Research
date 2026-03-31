
from dataclasses import dataclass
import numpy as np
from proper_research.models.magnetic_model import magnetic_moment


@dataclass
class BeamParams:
    mag: float       # magnetisation [A/m]
    r: float         # radius [m]
    E: float         # Young's modulus [Pa]
    A_cs: float      # cross-sectional area [m^2]
    I: float         # second moment of area [m^4]
    J: float
    G: float
    L_init: float    # initial L [m]
    B_init: float    # initial B [T]
    phi_init: float  # initial phi [rad]
    length_of_mag: float
    f_g: float
@dataclass
class MagnetParams:
    mu_0: float      # vacuum permeability
    B_r: float       # remanence [T]
    r_epm: float     # EPM radius [m]
    p_epm: float     # EPM length [m]
    m_hat: tuple     # direction of dipole
    x_init: float    # initial distance [m]
    mag_epm: float


@dataclass
class VisualPIDParams:
    Kp: float
    Ki: float
    Kd: float
    dt: float
    tol_deg: float
    max_iters: int
    phi_min: float
    phi_max: float

@dataclass
class AlphaControllerParams:
    Kp: float
    Ki: float
    Kd: float
    dt: float
    weight: float
    alpha_init: float
    damping: float
    max_iter: int
    grad_runtime_thresh: float
    theta_zero_thresh: float
    tol_deg: float
    alpha_min: float
    alpha_max: float



def default_beam_params():
    mag = 100e3
    r = 0.0018
    E = 2e8
    A_cs = np.pi * r**2
    I = np.pi * r**4 / 4
    phi_intit = np.deg2rad(10)
    nu = 0.49
    J = 0.5*np.pi*r**4
    G = E / (2*(1+nu))
    length_of_mag = 0.04
    w = 1200 * A_cs * 9.8
    f_g = np.array([0.0, 0.0, -w])
    return BeamParams(
        mag=mag,
        r=r,
        E=E,
        A_cs=A_cs,
        I=I,
        L_init=0.05,
        B_init=0.025,
        phi_init=phi_intit,
        G = G,
        J=J,
        length_of_mag = length_of_mag,
        f_g = f_g
    )

def default_magnet_params():
    mu_0 = 4e-7 * np.pi
    B_r = 1.05
    r_epm = 0.03
    p_epm = 0.09
    m_hat = (1.0, 0.0, 0.0)
    x_init = 0.09
    mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)
    return MagnetParams(
        mu_0=mu_0,
        B_r=B_r,
        r_epm=r_epm,
        p_epm=p_epm,
        m_hat=m_hat,
        x_init=x_init,
        mag_epm = mag_epm,
    )

def default_visual_pid_params():
    return VisualPIDParams(
        Kp=0.3,
        Ki=0.0,
        Kd=0.05,
        dt=0.2,
        tol_deg=0.5,
        max_iters=20,
        phi_min=np.deg2rad(-90),
        phi_max=np.deg2rad(90),
    )

def default_alpha_controller_params():
    return AlphaControllerParams(
        Kp=5.0,
        Ki=0.0,
        Kd=0.5,
        dt=0.05,
        weight = 0.2,
        alpha_init = 0.0,
        damping=1e-3,
        max_iter=10,
        grad_runtime_thresh=20.0,
        theta_zero_thresh=np.deg2rad(3.0),
        tol_deg=0.5,
        alpha_min=np.deg2rad(-180),
        alpha_max=np.deg2rad(180),
    )

ROBOT_IP = "192.168.56.101"
