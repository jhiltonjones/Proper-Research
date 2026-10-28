from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import copy
import numpy as np


@dataclass
class ForwardCache:
    """
    Mutable cache owned by MagneticBeamForwardModel.

    This stores the latest committed forward solve so that:
      - the next forward solve can warm-start from u_flat_opt
      - the Jacobian can use the same nominal solution
      - plotting/controller code can inspect the latest centerline and info
    """
    p7_last: np.ndarray | None = None
    L_model: float | None = None
    u_flat_opt: np.ndarray | None = None
    tip: np.ndarray | None = None
    centerline: np.ndarray | None = None
    q: np.ndarray | None = None
    u_seg: np.ndarray | None = None
    info: dict[str, Any] | None = None
    hist: list[dict[str, Any]] | None = None

    def is_empty(self) -> bool:
        return self.u_flat_opt is None

    def copy(self) -> "ForwardCache":
        return copy.deepcopy(self)


@dataclass
class SolveResult:
    """
    Result of one solved beam equilibrium.

    p:
        Centerline positions, usually shape (3, N).

    q:
        Orientation quaternions along the beam.

    u_seg:
        Segment strain, shape (N - 1, 3).

    tip:
        Tip position, shape (3,).

    u_flat_opt:
        Optimised flattened strain, shape (3 * (N - 1),).

    info:
        Diagnostics and optimiser output.

    hist:
        Optional continuation history. Usually populated by solve_quasistatic_insertion.
    """
    p: np.ndarray
    q: np.ndarray | None
    u_seg: np.ndarray | None
    tip: np.ndarray
    u_flat_opt: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    hist: list[dict[str, Any]] | None = None

    def __post_init__(self):
        self.p = np.asarray(self.p, float)
        self.tip = np.asarray(self.tip, float).reshape(3)
        self.u_flat_opt = np.asarray(self.u_flat_opt, float).reshape(-1)

        if self.q is not None:
            self.q = np.asarray(self.q, float)

        if self.u_seg is not None:
            self.u_seg = np.asarray(self.u_seg, float)

    @property
    def success(self) -> bool:
        return bool(self.info.get("success", False))

    @property
    def W(self) -> float:
        return float(self.info.get("W", np.nan))

    @property
    def parts(self) -> dict[str, Any]:
        return self.info.get("parts", {})


def cache_from_result(
    *,
    p7: np.ndarray,
    result: SolveResult,
    L_model: float,
) -> ForwardCache:
    """
    Build a ForwardCache from a committed SolveResult.
    """
    return ForwardCache(
        p7_last=np.asarray(p7, float).reshape(7).copy(),
        L_model=float(L_model),
        u_flat_opt=result.u_flat_opt.copy(),
        tip=result.tip.copy(),
        centerline=result.p.copy(),
        q=None if result.q is None else result.q.copy(),
        u_seg=None if result.u_seg is None else result.u_seg.copy(),
        info=copy.deepcopy(result.info),
        hist=result.hist,
    )