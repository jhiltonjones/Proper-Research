from dataclasses import dataclass
import numpy as np


@dataclass
class MPCConfig:
    dt: float = 0.05
    Np: int = 10
    n_out: int = 3
    n_pose: int = 8
    n_u: int = 7

    model_mode: str = "ltv"
    solver_mode: str = "sqp_full"
    N_sqp: int = 3

    sqp_tol_u: float = 1e-4
    sqp_tol_rel_u: float = 1e-3

    dL_index: int = 6
    dL_back_max: float = 0.002
    dL_fwd_max: float = np.inf

    enable_hard_epm_tip_clearance: bool = True
    epm_tip_hard_min_m: float = 0.011

    trust_radius: object = None
    use_affine_matching: bool = True

    qp_reg: float = 1e-9