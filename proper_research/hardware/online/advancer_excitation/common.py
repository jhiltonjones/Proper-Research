"""Shared constants and small helpers for the advancer-excitation test.

Kept deliberately independent of ``rectangle_stage_a`` (different purpose --
the robot is not commanded at all here) except where reusing a helper avoids
duplicating something that must stay correct in one place only
(``load_plan_initial_state`` -- reading a plan's own start state, not a
hand-copied constant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mirrors close_loop_path_follow.PathFollowConfig.insertion_rate_limit_m_s --
# the real experiment's advancer rate bound (2mm/s). Kept as a distinct
# constant (not imported) because this package intentionally never imports
# close_loop_path_follow -- the robot/harness are out of scope here.
INSERTION_RATE_LIMIT_M_S = 2.0e-3

ROBOT_IP = "192.168.56.101"


@dataclass
class LengthPlan:
    L_min_m: float
    L_max_m: float
    representative_rate_m_s: float  # mean |insertion rate| actually used on this plan

    @property
    def delta_L_m(self) -> float:
        return self.L_max_m - self.L_min_m

    def four_test_lengths_m(self) -> tuple[float, float, float, float]:
        d = self.delta_L_m
        return (
            self.L_min_m,
            self.L_min_m + d / 3.0,
            self.L_min_m + 2.0 * d / 3.0,
            self.L_max_m,
        )


def length_plan_from_plan_dir(plan_dir: str) -> LengthPlan:
    """L_min/L_max/representative insertion rate straight from a plan's own
    state/input arrays -- never hand-copy a shape's insertion range."""
    from pathlib import Path

    npz_path = Path(plan_dir) / (Path(plan_dir).name + ".npz")
    if not npz_path.exists():
        candidates = list(Path(plan_dir).glob("*.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not find a unique plan .npz under {plan_dir!r} "
                f"(looked for {npz_path.name}, found {[c.name for c in candidates]})."
            )
        npz_path = candidates[0]
    data = np.load(npz_path)
    state = data["state_reference"] if "state_reference" in data else data["state"]
    L = np.asarray(state[:, 6], dtype=float)
    rate = 0.0
    if "input_reference" in data:
        rate = float(np.mean(np.abs(np.asarray(data["input_reference"], dtype=float)[:, 6])))
    if rate <= 0.0:
        rate = INSERTION_RATE_LIMIT_M_S * 0.5  # fallback: half the hardware rate limit
    return LengthPlan(L_min_m=float(L.min()), L_max_m=float(L.max()), representative_rate_m_s=rate)


PIVOT_XYZ = np.array([0.525575, -0.670028, -0.016567])   # StateStreamConfig.T_robot_beam_pose6[:3]
AXIAL_DIR_R = np.array([-1.0, 0.0, 0.0])                 # StateStreamConfig.beam_axial_axis_R


@dataclass
class BendPose:
    label: str
    q6: np.ndarray
    b_mm: float          # off-axis (lateral) displacement of the plan's own desired tip position
    L_at_index_mm: float
    sample_index: int


def select_bend_poses(plan_dir: str, percentiles=(10, 50, 90), labels=("low", "medium", "high")) -> list[BendPose]:
    """Pick real, feasible plan waypoints spanning low/medium/high beam
    bending -- never invented poses. Bending proxy: the off-axis (lateral)
    component of the plan's own `desired_position_m` relative to the pivot,
    projected orthogonal to the beam's axial direction. For a straight,
    unbent beam this is ~0 by construction (the beam extends straight along
    the axial axis); any steering-induced curvature shows up as a nonzero
    off-axis component -- so this is a genuine, pre-computed bending signal
    already in the plan, not a fresh measurement."""
    from pathlib import Path

    npz_path = Path(plan_dir) / (Path(plan_dir).name + ".npz")
    if not npz_path.exists():
        candidates = list(Path(plan_dir).glob("*.npz"))
        npz_path = candidates[0]
    data = np.load(npz_path)
    desired = np.asarray(data["desired_position_m"], dtype=float)
    feasible = np.asarray(data["beam_sample_feasible"], dtype=bool)
    state = np.asarray(data["state"], dtype=float)

    rel = desired - PIVOT_XYZ[None, :]
    off_axis = rel - np.outer(rel @ AXIAL_DIR_R, AXIAL_DIR_R)
    b = np.linalg.norm(off_axis, axis=1)
    b_masked = np.where(feasible, b, np.nan)

    poses = []
    for label, pct in zip(labels, percentiles):
        target = np.nanpercentile(b_masked, pct)
        idx = int(np.nanargmin(np.abs(b_masked - target)))
        poses.append(BendPose(
            label=label, q6=state[idx, :6].copy(), b_mm=float(b_masked[idx]) * 1e3,
            L_at_index_mm=float(state[idx, 6]) * 1e3, sample_index=idx,
        ))
    return poses


def damped_pinv_joint_jacobian(J_joints_3x6: np.ndarray, damping: float = 5.0e-2) -> np.ndarray:
    """The exact DLS pseudoinverse ``TwoDOFTrimController`` uses (3x6 joint
    block only, gram = J J^T + damping^2 I). Kept identical here so the
    causality replay's ``J^+`` matches the live controller's, not a
    reasonable-looking approximation of it."""
    Jq = np.asarray(J_joints_3x6, dtype=float).reshape(3, 6)
    gram = Jq @ Jq.T + (damping ** 2) * np.eye(3)
    return Jq.T @ np.linalg.solve(gram, np.eye(3))
