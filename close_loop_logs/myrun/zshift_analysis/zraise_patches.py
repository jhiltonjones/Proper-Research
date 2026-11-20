"""Shared raised-workspace patches for live closed-loop runs on the
+30mm-raised rectangle plan (plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24).

Mirrors vessel_stage_a's own two raised-base fixes exactly:
1. make_initial_poses() (beam base pivot + start point) -- needed so any
   fresh schedule/Jacobian build (build_planning_context) sees the SAME
   beam base the plan was actually built against, not the old unraised one.
2. StateStreamConfig's T_robot_beam_pose6 default -- needed because vision
   reconstruction projects the 2D camera image onto a plane at this fixed
   z, INDEPENDENT of make_initial_poses(); without this the open-loop run's
   own error metric wouldn't show it (z is effectively projected out for
   this in-plane shape) but the closed-loop MPC/INV-7 controllers use the
   raw unprojected z in their disturbance estimator / task cost, and would
   see an uncorrected ~30mm phantom z-residual -- exactly the vessel
   incident from 2026-09-23. Caught here 2026-09-24 before any closed-loop
   run, not after.

Import this module and call apply() before importing/calling any
run_*_delay_aware*.py main().
"""
import dataclasses

import numpy as np

Z_RAISE_M = 0.030


def apply():
    import proper_research.simulation.simulations.initial_conditions as _initial_conditions_mod
    import proper_research.planning.planning_context as _planning_context_mod
    import proper_research.hardware.online.close_loop_path_follow as pf
    from proper_research.hardware.online.state_stream import StateStreamConfig as _RealStateStreamConfig

    _ORIG_MAKE_INITIAL_POSES = _initial_conditions_mod.make_initial_poses

    def _raised_make_initial_poses():
        p, s, L, dt = _ORIG_MAKE_INITIAL_POSES()
        p = np.array(p, dtype=float).copy()
        s = np.array(s, dtype=float).copy()
        p[2] += Z_RAISE_M
        s[2] += Z_RAISE_M
        return p, s, L, dt

    _initial_conditions_mod.make_initial_poses = _raised_make_initial_poses
    _planning_context_mod.make_initial_poses = _raised_make_initial_poses

    def _RaisedStateStreamConfig(**kwargs):
        cfg = _RealStateStreamConfig(**kwargs)
        if "T_robot_beam_pose6" not in kwargs:
            pose6 = list(cfg.T_robot_beam_pose6)
            pose6[2] += Z_RAISE_M
            cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
        return cfg

    pf.StateStreamConfig = _RaisedStateStreamConfig
    print(f"[zraise_patches] applied: make_initial_poses +{Z_RAISE_M*1e3:.0f}mm, "
          f"StateStreamConfig.T_robot_beam_pose6 +{Z_RAISE_M*1e3:.0f}mm")
