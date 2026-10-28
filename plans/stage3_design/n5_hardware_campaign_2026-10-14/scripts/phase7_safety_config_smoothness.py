"""Script 7 -- plan: snug-bouncing-giraffe.md.

Safety margins, command smoothness, and configuration/insertion
comparison, for all 9 runs.

IMPORTANT SCOPE CORRECTION (found while building this script, not
assumed): the WIDE Cartesian workspace box (cfg.workspace_xyz_min_m/
max_m in run_mpc_delay_aware_vessel.py) is checked in
close_loop_path_follow.py against `reader.latest_pose(...)[:3]` -- the
RAW RTDE ROBOT-BASE-FRAME TCP POSE -- NOT against
`urik.forward_kinematics(...).T_R_target[:3,3]` (frame "R", relative
to the beam base), which is a DIFFERENT frame. A first attempt at this
script wrongly assumed these were the same point and got nonsensical
~-200mm "violations" on every tick of every run; checked directly
against close_loop_path_follow.py's actual check (reader.latest_pose),
not guessed. The raw RTDE TCP pose is NOT logged anywhere in
path_follow.jsonl, and the extra robot-base-to-beam-base calibration
transform needed to reconstruct it post-hoc is not reliably available
here, so this script does NOT attempt to report a numeric margin for
that wide box. Its feasibility is instead evidenced directly: every
one of the 9 runs completed with stop_reason=path_complete (confirmed
in Script 1's manifest) -- a tcp_out_of_workspace violation would have
aborted the run immediately with that exact stop_reason, so the box
was never violated, even though no numeric margin trace is available.

What IS computed correctly here, in the magnet's own frame "R"
(T_R_target[:3,3] from forward_kinematics with the SAME dh/T_F_M used
by `_configure_magnet_workspace`'s own position_jacobians/nominal_
positions_m -- confirmed by reading that function, not assumed): the
TIGHT two-sided z constraint from controller_metadata.json's
magnet_z_bounds_m (the actual in-QP hard constraint
`_magnet_workspace_bounds` enforces), the magnet-exclusion-radius
margin, and an insertion-offset margin (measured insertion vs. the
native path's own nominal insertion at the measured physical progress
s_tip).

Preregistered "materially" rule for the safety hypothesis (written
before these results are inspected): safety is judged on ABSOLUTE
FEASIBILITY FIRST -- h_j > 0 for every hard constraint face actually
measurable here, every run (plus path_complete as the direct feasibility
evidence for the wide box). Beyond that, margins/roughness are compared
DESCRIPTIVELY ONLY (observed minimums reported side by side) -- no
post-hoc pass/fail tolerance is invented.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")
import run_registry as reg
import path_geometry as pg

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.planning.planning_context import make_robot_config

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

robot_kin = _resolve_robot_kinematics(make_robot_config())


def fk_magnet_pos(q):
    return urik.forward_kinematics(np.asarray(q, dtype=float), robot_kin.dh, robot_kin.T_F_M).T_R_target[:3, 3]


def per_tick_margins(condition, rep, run_path):
    rows = reg.load_path_follow(run_path)
    meta = reg.load_controller_metadata(run_path)
    z_min_tight, z_max_tight = meta["magnet_z_bounds_m"]
    excl_point = np.asarray(meta["magnet_exclusion_point_R"], dtype=float)
    excl_radius = float(meta["magnet_exclusion_radius_m"])
    tables = pg.load_path_tables(reg.plan_root_for(condition))

    n = len(rows)
    faces = dict(
        z_tight_min=np.zeros(n), z_tight_max=np.zeros(n),
        exclusion_gap=np.zeros(n), insertion_margin=np.zeros(n),
    )
    u0_all = np.array([r["u0"] for r in rows], dtype=float)

    for k, r in enumerate(rows):
        q_meas = np.asarray(r["q_meas_rad"], dtype=float)
        p = fk_magnet_pos(q_meas)
        faces["z_tight_min"][k] = p[2] - z_min_tight
        faces["z_tight_max"][k] = z_max_tight - p[2]
        faces["exclusion_gap"][k] = float(np.linalg.norm(p - excl_point)) - excl_radius

        L_meas = float(r["insertion_length_m"])
        s_tip = pg.project_s_cartesian(np.asarray(r["tip_mm"], dtype=float) / 1000.0, tables)
        L_nom = pg.L_plan(s_tip, tables)
        faces["insertion_margin"][k] = float(meta["insertion_offset_abort_m"]) - abs(L_meas - L_nom)

    face_names = list(faces.keys())
    face_matrix = np.stack([faces[f] for f in face_names], axis=1)  # (n, n_faces)
    min_face_idx = np.argmin(face_matrix, axis=1)
    h_min_per_tick = np.min(face_matrix, axis=1)
    closest_face_overall = face_names[int(np.argmin(np.min(face_matrix, axis=0)))]

    cmd_increments = np.linalg.norm(np.diff(u0_all[:, :6], axis=0), axis=1)

    return dict(
        condition=condition, rep=rep,
        faces={f: dict(min=float(faces[f].min()), p01=float(np.percentile(faces[f], 1))) for f in face_names},
        min_margin_overall_m=float(h_min_per_tick.min()),
        closest_face_overall=closest_face_overall,
        n_ticks_any_face_violated=int(np.sum(h_min_per_tick <= 0)),
        cmd_increment_mean=float(np.mean(cmd_increments)), cmd_increment_p90=float(np.percentile(cmd_increments, 90)),
        cmd_increment_p99=float(np.percentile(cmd_increments, 99)), cmd_increment_max=float(np.max(cmd_increments)),
    )


def main():
    results = {}
    for cond, rep, run_path in reg.all_runs():
        results[f"{cond}_rep{rep}"] = per_tick_margins(cond, rep, run_path)

    print("=" * 100)
    print("ABSOLUTE FEASIBILITY CHECK")
    print("=" * 100)
    print("Wide workspace box (cfg.workspace_xyz_min_m/max_m): not numerically reconstructable from "
          "path_follow.jsonl (checked against the raw RTDE robot-base TCP pose, not logged here) -- "
          "feasibility evidenced directly by all 9 runs completing with stop_reason=path_complete "
          "(Script 1 manifest); a violation would have aborted the run with stop_reason="
          "tcp_out_of_workspace instead.")
    print()
    print("Remaining hard faces (h_j > 0 required, every run, every tick):")
    any_violation = False
    for key, r in results.items():
        viol = r["n_ticks_any_face_violated"] > 0
        any_violation = any_violation or viol
        print(f"  {key:12s} min_margin={r['min_margin_overall_m']*1000:+.3f}mm "
              f"closest_face={r['closest_face_overall']:14s} "
              f"n_ticks_violated={r['n_ticks_any_face_violated']}  {'VIOLATION' if viol else 'feasible'}")
    print(f"\nOVERALL: {'AT LEAST ONE VIOLATION FOUND' if any_violation else 'ALL 9 RUNS FEASIBLE ON EVERY HARD FACE, EVERY TICK'}")

    print()
    print("=" * 100)
    print("DESCRIPTIVE MARGIN COMPARISON (observed minimum per face, by condition -- no pass/fail tolerance)")
    print("=" * 100)
    face_names = list(next(iter(results.values()))["faces"].keys())
    for face in face_names:
        print(f"--- face: {face} ---")
        for cond in ("NEW", "OLD-C", "NC"):
            vals = [results[f"{cond}_rep{r}"]["faces"][face]["min"] for r, _ in reg.RUNS[cond]]
            print(f"  {cond:6s} min-of-mins={min(vals)*1000:+.3f}mm  per-rep=[{', '.join(f'{v*1000:+.3f}' for v in vals)}]mm")

    print()
    print("=" * 100)
    print("COMMAND SMOOTHNESS (||u_k - u_{k-1}|| over joint velocities, rad/s)")
    print("=" * 100)
    for cond in ("NEW", "OLD-C", "NC"):
        for rep, _ in reg.RUNS[cond]:
            r = results[f"{cond}_rep{rep}"]
            print(f"  {cond:6s} rep{rep}: mean={r['cmd_increment_mean']:.4f} p90={r['cmd_increment_p90']:.4f} "
                  f"p99={r['cmd_increment_p99']:.4f} max={r['cmd_increment_max']:.4f}")

    with open(os.path.join(OUT_DIR, "phase7_safety_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR}/phase7_safety_results.json")


if __name__ == "__main__":
    main()
