from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from .contact import ContactParams


@dataclass(frozen=True)
class BaseFrameConfig:
    p0_ur: np.ndarray
    q0_ur: np.ndarray
    u_star: np.ndarray

    def __post_init__(self):
        p0_ur = np.asarray(self.p0_ur, float).reshape(3)
        q0_ur = np.asarray(self.q0_ur, float).reshape(4)
        u_star = np.asarray(self.u_star, float).reshape(3)

        q0_ur = q0_ur / (np.linalg.norm(q0_ur) + 1e-12)

        if not np.all(np.isfinite(p0_ur)):
            raise ValueError(f"p0_ur contains non-finite values: {p0_ur}.")
        if not np.all(np.isfinite(q0_ur)):
            raise ValueError(f"q0_ur contains non-finite values: {q0_ur}.")
        if not np.all(np.isfinite(u_star)):
            raise ValueError(f"u_star contains non-finite values: {u_star}.")

        object.__setattr__(self, "p0_ur", p0_ur)
        object.__setattr__(self, "q0_ur", q0_ur)
        object.__setattr__(self, "u_star", u_star)


@dataclass(frozen=True)
class BeamSolveProblem:
    p0: np.ndarray
    q0: np.ndarray

    L_ins: float
    L_model: float
    wire_len: float
    tip_len: float

    Kinv_fun: Callable
    u_star: np.ndarray

    r_src: np.ndarray
    q_src: np.ndarray
    m_src: np.ndarray
    m_body: np.ndarray
    m_local_fun: Callable
    m_moment: float

    lumen_query: Any | None
    contact: ContactParams | None
    use_contact: bool
    use_contact_in_jacobian: bool

    N_nodes: int
    L_tip_full: float
    L_tip_min: float

    def __post_init__(self):
        p0 = np.asarray(self.p0, float).reshape(3)
        q0 = np.asarray(self.q0, float).reshape(4)
        u_star = np.asarray(self.u_star, float).reshape(3)

        r_src = np.asarray(self.r_src, float).reshape(3)
        q_src = np.asarray(self.q_src, float).reshape(4)
        m_src = np.asarray(self.m_src, float).reshape(3)
        m_body = np.asarray(self.m_body, float).reshape(3)

        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q_src = q_src / (np.linalg.norm(q_src) + 1e-12)

        for name, arr in {
            "p0": p0,
            "q0": q0,
            "u_star": u_star,
            "r_src": r_src,
            "q_src": q_src,
            "m_src": m_src,
            "m_body": m_body,
        }.items():
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values: {arr}.")

        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "q0", q0)
        object.__setattr__(self, "u_star", u_star)
        object.__setattr__(self, "r_src", r_src)
        object.__setattr__(self, "q_src", q_src)
        object.__setattr__(self, "m_src", m_src)
        object.__setattr__(self, "m_body", m_body)

        if self.L_ins < 0:
            raise ValueError(f"L_ins must be non-negative, got {self.L_ins}.")
        if self.L_model <= 0:
            raise ValueError(f"L_model must be positive, got {self.L_model}.")
        if self.wire_len < 0:
            raise ValueError(f"wire_len must be non-negative, got {self.wire_len}.")
        if self.tip_len < 0:
            raise ValueError(f"tip_len must be non-negative, got {self.tip_len}.")
        if self.N_nodes < 2:
            raise ValueError(f"N_nodes must be >= 2, got {self.N_nodes}.")
        if self.L_tip_full <= 0:
            raise ValueError(f"L_tip_full must be positive, got {self.L_tip_full}.")
        if self.L_tip_min <= 0:
            raise ValueError(f"L_tip_min must be positive, got {self.L_tip_min}.")

        if self.use_contact and self.lumen_query is None:
            raise ValueError("use_contact=True requires lumen_query.")
        if self.use_contact and self.contact is None:
            raise ValueError("use_contact=True requires contact parameters.")

        if self.use_contact_in_jacobian and not self.use_contact:
            raise ValueError(
                "use_contact_in_jacobian=True requires use_contact=True."
            )

        if self.contact is not None:
            self.contact.validate()