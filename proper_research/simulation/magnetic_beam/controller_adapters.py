from __future__ import annotations

import copy
import numpy as np

from .kinematics import pose8_quat_to_pose7_rotvec, tip_tangent_from_centerline


class ControllerForwardAdapter:
    """
    Adapter between the controller/MPC pose convention and MagneticBeamForwardModel.

    Controller input:
        p8 = [x, y, z, qw, qx, qy, qz, L]

    Model input:
        p7 = [x, y, z, rx, ry, rz, L]

    Output:
        y = [tip_x, tip_y, tip_z, tangent_x, tangent_y, tangent_z]
    """

    def __init__(self, forward_model):
        self.model = forward_model
        self._baseline_cache = None

        self.last_tip: np.ndarray | None = None
        self.last_p_centerline: np.ndarray | None = None
        self.last_info: dict | None = None

    def start_step(self) -> None:
        """
        Freeze the model state at the start of an MPC/controller step.
        """
        self._baseline_cache = self.model.get_cache_copy()

    def reset(self) -> None:
        """
        Clear adapter and model cache.
        """
        self._baseline_cache = None
        self.last_tip = None
        self.last_p_centerline = None
        self.last_info = None

        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

    def __call__(self, p8: np.ndarray, *, commit: bool = False) -> np.ndarray:
        """
        Evaluate model and return [tip_xyz, tip_tangent].

        commit=False:
            Evaluate from the current step baseline, then restore the model.

        commit=True:
            Evaluate and keep the new solve as the model warm-start/cache.
        """
        p7 = pose8_quat_to_pose7_rotvec(p8)

        if self._baseline_cache is None:
            self.start_step()

        if commit:
            result = self.model.solve(p7, commit=True)
            self._baseline_cache = self.model.get_cache_copy()
        else:
            old_cache = self.model.get_cache_copy()
            try:
                self.model.set_cache(self._baseline_cache)
                result = self.model.solve(p7, commit=True)
            finally:
                self.model.set_cache(old_cache)

        self.last_tip = result.tip.copy()
        self.last_p_centerline = result.p.copy()
        self.last_info = copy.deepcopy(result.info)

        tangent = tip_tangent_from_centerline(result.p)

        return np.hstack([result.tip, tangent])


class TipOnlyAdapter:
    """
    Adapter for controllers that only need tip position.

    Input:
        p8 = [x, y, z, qw, qx, qy, qz, L]

    Output:
        tip_xyz
    """

    def __init__(self, forward_model):
        self.model = forward_model

    def __call__(self, p8: np.ndarray, *, commit: bool = False) -> np.ndarray:
        p7 = pose8_quat_to_pose7_rotvec(p8)

        if commit:
            return self.model.solve(p7, commit=True).tip.copy()

        old_cache = self.model.get_cache_copy()
        try:
            result = self.model.solve(p7, commit=True)
            return result.tip.copy()
        finally:
            self.model.set_cache(old_cache)


class Pose7TipTangentAdapter:
    """
    Adapter for code that already uses p7 directly.

    Input:
        p7 = [x, y, z, rx, ry, rz, L]

    Output:
        y = [tip_xyz, tip_tangent]
    """

    def __init__(self, forward_model):
        self.model = forward_model

        self.last_tip: np.ndarray | None = None
        self.last_p_centerline: np.ndarray | None = None
        self.last_info: dict | None = None

    def __call__(self, p7: np.ndarray, *, commit: bool = False) -> np.ndarray:
        if commit:
            result = self.model.solve(p7, commit=True)
        else:
            old_cache = self.model.get_cache_copy()
            try:
                result = self.model.solve(p7, commit=True)
            finally:
                self.model.set_cache(old_cache)

        self.last_tip = result.tip.copy()
        self.last_p_centerline = result.p.copy()
        self.last_info = copy.deepcopy(result.info)

        tangent = tip_tangent_from_centerline(result.p)

        return np.hstack([result.tip, tangent])