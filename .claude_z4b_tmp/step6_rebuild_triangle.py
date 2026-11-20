"""Z4b step 6: executed rebuild of the triangle-35 reference plan from the
current (post Z4b steps 1,2,3,5) code, into a NEW campaign plan directory
(the original triangle_base35mm_skipglobal_ad2_2026-11-10 is left untouched).
Verbatim recipe from the original AD2 build (ad2_triangle35_bypass.py),
repointed at this worktree and a new output root.
"""
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

WORKTREE = "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis"
sys.path.insert(0, WORKTREE)

import numpy as np

from proper_research.planning.planning_context import (
    build_planning_context,
    make_inverse_config,
)
from proper_research.planning.offline_inverse_configuration_head_exclusion import (
    InverseConfigurationPlannerConfig,
    solve_from_controller_pack,
)
from proper_research.planning.time_parameterized_configuration_path import (
    TimeParameterizationConfig,
    time_parameterize_saved_inverse_path,
)
from proper_research.planning.plan_shape_path import _calibrated_initial_poses

SRC_NPZ = Path(WORKTREE) / "plans/triangle_base35mm_skipglobal_magdist_2026-11-04/shape_triangle/shape_centreline.npz"
OUT_ROOT = Path(WORKTREE) / "plans/triangle_base35mm_skipglobal_z4b_2026-11-11"

d = np.load(SRC_NPZ)
centreline = np.asarray(d["lumen_C"], dtype=float)
lumen_R = np.asarray(d["lumen_R"], dtype=float)
tip0 = np.asarray(d["tip0"], dtype=float)
u_axis = np.asarray(d["u_axis"], dtype=float)
v_axis = np.asarray(d["v_axis"], dtype=float)
print(f"[bypass] loaded centreline: {centreline.shape[0]} pts from {SRC_NPZ}")

initial_poses_override = _calibrated_initial_poses()
_, bundle, controller_pack, out_root = build_planning_context(
    initial_poses=initial_poses_override
)
out_root = OUT_ROOT
out_root.mkdir(parents=True, exist_ok=True)

perimeter = float(np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))
print(f"[shape] triangle (loaded) -> {centreline.shape[0]} points, perimeter {1e3*perimeter:.2f} mm")

shape_dir = out_root / "shape_triangle"
shape_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    shape_dir / "shape_centreline.npz",
    lumen_C=centreline, lumen_R=lumen_R, tip0=tip0, u_axis=u_axis, v_axis=v_axis,
)

planner_config = make_inverse_config()
shared = {
    f.name: getattr(planner_config, f.name)
    for f in fields(InverseConfigurationPlannerConfig)
    if hasattr(planner_config, f.name)
}
shared["tangent_tolerance_rad"] = math.radians(min(180.0, 179.95))
shared["position_tolerance_m"] = 1.0e-3

magnet_R = np.asarray(
    controller_pack["plant_diagnostic_joint_adapter"].magnet_transform(
        np.asarray(controller_pack["p0"])
    ),
    dtype=float,
)[:3, 3]
gaps = np.linalg.norm(centreline - magnet_R[None, :], axis=1)
exclusion_radius_m = float(gaps.min())
shared["source_magnet_lumen_exclusion_radius_m"] = exclusion_radius_m
print(f"[layer1] magnet exclusion: magnet@{np.round(magnet_R,3).tolist()}  min gap {1e3*exclusion_radius_m:.1f} mm")

if "magnet_distance_weight" in shared:
    shared["magnet_distance_weight"] = 1.0
if "magnet_distance_max_m" in shared:
    shared["magnet_distance_max_m"] = 0.5
if "magnet_distance_scale_m" in shared:
    shared["magnet_distance_scale_m"] = 0.01
if "tcp_box_weight" in shared:
    shared["tcp_box_weight"] = 1.0
if "tcp_box_margin_m" in shared:
    shared["tcp_box_margin_m"] = 0.027
if "tcp_box_scale_m" in shared:
    shared["tcp_box_scale_m"] = 0.01

planner_config = InverseConfigurationPlannerConfig(**shared)
print(f"[layer1] tangent tol=180deg, position tol={1e3*planner_config.position_tolerance_m:.2f}mm, "
      f"magnet_distance_weight={planner_config.magnet_distance_weight}, "
      f"tcp_box_weight={planner_config.tcp_box_weight}, "
      f"tcp_box_margin_m={planner_config.tcp_box_margin_m}")

inverse_dir = shape_dir / "offline_inverse_configuration_60"
print(f"\n[layer1] inverse planning -> {inverse_dir}", flush=True)
t0 = time.perf_counter()
inverse_result = solve_from_controller_pack(
    controller_pack=controller_pack,
    lumen_C=centreline,
    config=planner_config,
    output_dir=inverse_dir,
)
print(f"[layer1] done in {time.perf_counter()-t0:.1f}s  all_nodes_feasible={inverse_result.all_nodes_feasible}", flush=True)

if not inverse_result.all_nodes_feasible:
    raise SystemExit("[layer1] did not reach a fully node-feasible path.")

tp_config = TimeParameterizationConfig(
    sample_period_s=0.1,
    state_velocity_limit=tuple([0.10]*6 + [0.002]),
    state_acceleration_limit=tuple([0.5]*6 + [0.02]),
    maximum_path_speed_m_s=3.0e-3,
    constraint_tolerance=1.0e-8,
    velocity_safety_factor=0.8,
    acceleration_safety_factor=0.8,
    require_nonlinear_beam_feasible=False,
    require_saved_global_feasible=False,
    require_saved_dense_feasible=False,
)
time_param_dir = out_root / "time_parameterized_configuration_path"
print(f"\n[layer3] time-parameterising the INVERSE path -> {time_param_dir}", flush=True)
t1 = time.perf_counter()
time_parameterize_saved_inverse_path(
    inverse_output_dir=inverse_dir,
    controller_pack=controller_pack,
    config=tp_config,
    output_dir=time_param_dir,
    lumen_C=centreline,
)
tp_dur = time.perf_counter() - t1
print(f"[layer3] done in {tp_dur:.1f}s", flush=True)

beam_base = [0.670575, -0.719727, -0.039627]
summary = {
    "shape": "triangle",
    "size_mm": 14.0,
    "closed": True,
    "centreline_points": int(centreline.shape[0]),
    "perimeter_mm": 1e3 * perimeter,
    "time_param_samples": None,
    "time_param_duration_s": tp_dur,
    "time_param_dir": str(time_param_dir),
    "note": "Z4b step 6 (2026-11-11): executed rebuild from the current (post-Z4b) code, verbatim AD2 recipe, new campaign directory.",
    "beam_base_xyz_m": beam_base,
    "rig_config": {"beam_base_xyz_m": beam_base, "tcp_to_magnet_pose6": [0.0, 0.0, 0.43, 0.0, 0.0, 0.0]},
    "tcp_box_activation_margin_m": 0.027,
    "tcp_box_gate_margin_m": 0.025,
}
try:
    tpp = np.load(time_param_dir / "time_parameterized_configuration_path.npz")
    summary["time_param_samples"] = int(tpp["state"].shape[0])
except Exception as exc:
    print(f"[warn] could not read time_param_samples: {exc}")

(shape_dir / "plan_shape_summary.json").write_text(json.dumps(summary, indent=2))
print(f"[bypass] wrote {shape_dir / 'plan_shape_summary.json'}")
print("[bypass] DONE", flush=True)
