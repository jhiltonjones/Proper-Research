"""Rigid z-shift patch, generalized from the raised (+30mm) rectangle study.

Shifts BOTH halves of the offline-plan/live-harness pairing by the same
amount in robot-frame R.z:
  1. the beam-base pivot pose and start point used by the offline planner
     (`make_initial_poses`, patched in both
     `initial_conditions` and `planning_context` since each imports its own
     reference to the function), and
  2. the vision-reconstruction plane the live harness uses to interpret
     camera detections (`StateStreamConfig.T_robot_beam_pose6`).

Patching only #1 (as the earlier vessel-planning-raised-base bug did)
leaves the live vision reconstruction reading the OLD, unraised plane while
the plan and controller expect the NEW one -- a silent, self-consistent-
looking but wrong disturbance estimate. Both halves must be patched before
any planning or live harness code runs.

Usage (import and call once, before touching planning_context / plan_shape_path
/ close_loop_path_follow / any of the run_*.py launchers):

    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(30.0)   # or 0.0 / omit for an unraised (original-height) plan
"""
import dataclasses

import numpy as np

_applied_mm = None


def apply(zraise_mm: float) -> None:
    global _applied_mm
    zraise_mm = float(zraise_mm)
    if zraise_mm == 0.0:
        return
    if _applied_mm is not None:
        if _applied_mm == zraise_mm:
            return  # idempotent re-apply at the same height
        raise RuntimeError(
            f"zraise_patch already applied at {_applied_mm}mm in this process; "
            f"cannot reapply at {zraise_mm}mm."
        )

    z_raise_m = zraise_mm / 1000.0

    import proper_research.simulation.simulations.initial_conditions as _initial_conditions_mod
    import proper_research.planning.planning_context as _planning_context_mod
    import proper_research.hardware.online.close_loop_path_follow as pf
    from proper_research.hardware.online.state_stream import StateStreamConfig as _RealStateStreamConfig

    _orig_make_initial_poses = _initial_conditions_mod.make_initial_poses

    def _raised_make_initial_poses():
        p, s, L, dt = _orig_make_initial_poses()
        p = np.array(p, dtype=float).copy()
        s = np.array(s, dtype=float).copy()
        p[2] += z_raise_m
        s[2] += z_raise_m
        return p, s, L, dt

    _initial_conditions_mod.make_initial_poses = _raised_make_initial_poses
    _planning_context_mod.make_initial_poses = _raised_make_initial_poses

    def _raised_state_stream_config(**kwargs):
        cfg = _RealStateStreamConfig(**kwargs)
        if "T_robot_beam_pose6" not in kwargs:
            pose6 = list(cfg.T_robot_beam_pose6)
            pose6[2] += z_raise_m
            cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
        return cfg

    pf.StateStreamConfig = _raised_state_stream_config

    _applied_mm = zraise_mm
    print(f"[zraise_patch] applied +{zraise_mm}mm to make_initial_poses "
          f"and StateStreamConfig.T_robot_beam_pose6")


def applied_mm():
    return _applied_mm
