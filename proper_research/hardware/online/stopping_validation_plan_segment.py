"""2026-11-19: loads a contiguous segment of the already-reviewed, already-
verified time-parameterized plan for use as the stopping-validation trial's
motion -- NOT invented waypoints or a direct start-to-end interpolation.

Reads `time_parameterized_configuration_path.npz` (the same artifact the
real controller/open-loop runners consume) and exposes the verified column
semantics:
  state[:, 0:6]      -- joint angles, rad
  state[:, 6]         -- insertion length, m
  state_rate[:, 0:6]  -- joint angular rates, rad/s (plan's own derivative)
  state_rate[:, 6]    -- insertion rate, m/s
  time_s              -- sample timestamps, s (sample_period_s apart)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BEAM_BASE_XYZ_M = np.array([0.670575, -0.719727, -0.039627])  # same constant used throughout this review


@dataclass
class PlanSegment:
    plan_dir: str
    i0: int
    i1: int  # inclusive
    time_s: np.ndarray          # (n,)
    joints_rad: np.ndarray      # (n, 6)
    insertion_m: np.ndarray     # (n,)
    joint_rate_rad_s: np.ndarray  # (n, 6)
    insertion_rate_m_s: np.ndarray  # (n,)
    magnet_xyz_m: np.ndarray    # (n, 3)
    clearance_m: np.ndarray     # (n,) distance to BEAM_BASE_XYZ_M

    @property
    def n(self) -> int:
        return len(self.time_s)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def predicted_magnet_speeds_mm_s(self) -> dict:
        if self.n < 2:
            return {"cartesian_mean": None, "cartesian_max": None, "inward_mean": None, "inward_max": None, "inward_min": None}
        dt = np.diff(self.time_s)
        vel = np.diff(self.magnet_xyz_m, axis=0) / dt[:, None]
        speed = np.linalg.norm(vel, axis=1)
        radial_dir = (BEAM_BASE_XYZ_M[None, :] - self.magnet_xyz_m[:-1])
        radial_dir = radial_dir / np.linalg.norm(radial_dir, axis=1, keepdims=True)
        inward = np.sum(vel * radial_dir, axis=1)
        return {
            "cartesian_mean_mm_s": float(speed.mean() * 1e3), "cartesian_max_mm_s": float(speed.max() * 1e3),
            "inward_mean_mm_s": float(inward.mean() * 1e3), "inward_max_mm_s": float(inward.max() * 1e3),
            "inward_min_mm_s": float(inward.min() * 1e3),
        }

    def summary_table(self) -> dict:
        speeds = self.predicted_magnet_speeds_mm_s()
        return {
            "plan_dir": self.plan_dir,
            "sample_indices": [self.i0, self.i1],
            "n_samples": self.n,
            "duration_s": round(self.duration_s, 3),
            "start_joints_rad": [round(float(x), 6) for x in self.joints_rad[0]],
            "end_joints_rad": [round(float(x), 6) for x in self.joints_rad[-1]],
            "insertion_start_mm": round(float(self.insertion_m[0]) * 1e3, 3),
            "insertion_end_mm": round(float(self.insertion_m[-1]) * 1e3, 3),
            "insertion_range_mm": [round(float(self.insertion_m.min()) * 1e3, 3), round(float(self.insertion_m.max()) * 1e3, 3)],
            "magnet_clearance_range_mm": [round(float(self.clearance_m.min()) * 1e3, 3), round(float(self.clearance_m.max()) * 1e3, 3)],
            **{k: (round(v, 3) if v is not None else None) for k, v in speeds.items()},
        }


def load_plan_npz(plan_dir: str) -> dict:
    npz_path = Path(plan_dir) / "time_parameterized_configuration_path.npz"
    if not npz_path.exists():
        npz_path = Path(plan_dir) / "time_parameterized_configuration_path" / "time_parameterized_configuration_path.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"no time_parameterized_configuration_path.npz found under {plan_dir}")
    d = np.load(npz_path)
    required = {"time_s", "state", "state_rate"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"plan npz missing required fields: {missing}")
    if d["state"].shape[1] != 7:
        raise ValueError(f"expected state to have 7 columns (6 joints + insertion), got {d['state'].shape[1]}")
    return {k: d[k] for k in d.files}


def extract_segment(plan_dir: str, i0: int, i1: int, fk_fn) -> PlanSegment:
    """`fk_fn(q6) -> xyz_m`: the project's own verified magnet forward
    kinematics (same function used throughout this review), injected rather
    than imported here so this module stays independently testable."""
    d = load_plan_npz(plan_dir)
    n_total = d["state"].shape[0]
    if not (0 <= i0 < i1 < n_total):
        raise ValueError(f"segment [{i0},{i1}] out of range for plan with {n_total} samples")
    time_s = d["time_s"][i0:i1 + 1]
    state = d["state"][i0:i1 + 1]
    rate = d["state_rate"][i0:i1 + 1]
    joints = state[:, :6]
    insertion = state[:, 6]
    magnet_xyz = np.array([fk_fn(q) for q in joints])
    clearance = np.linalg.norm(magnet_xyz - BEAM_BASE_XYZ_M[None, :], axis=1)
    return PlanSegment(
        plan_dir=plan_dir, i0=i0, i1=i1, time_s=time_s, joints_rad=joints, insertion_m=insertion,
        joint_rate_rad_s=rate[:, :6], insertion_rate_m_s=rate[:, 6],
        magnet_xyz_m=magnet_xyz, clearance_m=clearance,
    )


def find_representative_segment(plan_dir: str, fk_fn, window_samples: int = 30) -> tuple[int, int]:
    """Picks the contiguous window of `window_samples` with the highest mean
    joint speed (a representative, non-stationary slice), from the plan's
    OWN state_rate -- never an invented interpolation. Caller still presents
    this for operator review; this function only proposes it."""
    d = load_plan_npz(plan_dir)
    rate = d["state_rate"]
    joint_speed = np.linalg.norm(rate[:, :6], axis=1)
    n = len(joint_speed)
    if window_samples >= n:
        return 0, n - 1
    scores = [joint_speed[i:i + window_samples].mean() for i in range(n - window_samples)]
    i0 = int(np.argmax(scores))
    return i0, i0 + window_samples - 1
