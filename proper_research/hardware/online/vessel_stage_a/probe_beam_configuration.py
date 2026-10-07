#!/usr/bin/env python3
"""Probe the beam-plus-magnet model at a single hand-specified configuration,
independent of any planner/controller -- for debugging exactly what the
model predicts (full centerline shape, tip position/tangent, dipole
direction) given a chosen source-magnet position or joint configuration,
an insertion length, and contact-model on/off.

This is the tool that found the 2026-10-03 stale-lumen-Z bug: it lets you
sweep one input (insertion depth, or magnet Z) while holding everything
else fixed, and see exactly how the model's output responds -- a smooth,
physically sane response is healthy; a frozen/discontinuous one usually
means something (e.g. a wall-contact reference) isn't using the input
you think it's using.

Usage
-----
Single configuration, from a joint vector, contact-aware model::

    python -m proper_research.hardware.online.vessel_stage_a.probe_beam_configuration \\
        --lumen-file vessel_lumen_robot_frame_left1p5mm_zcorrected.json \\
        --joints -0.6261474 -1.7615390 -1.8847219 -1.0656255 1.5728815 -1.0911615 \\
        --insertion-mm 25 \\
        --contact \\
        --out debug_outputs/probe_contact_L25.png

Single configuration, from an explicit magnet pose6 (xyz + rotvec), no-contact model::

    python -m proper_research.hardware.online.vessel_stage_a.probe_beam_configuration \\
        --lumen-file vessel_lumen_robot_frame_left1p5mm_zcorrected.json \\
        --magnet-pose6 0.496 -0.573 -0.0397 -2.674 1.648 -0.0008 \\
        --insertion-mm 25 \\
        --no-contact

Sweep insertion depth at a fixed pose (reproduces the shallow-insertion
diagnostic from this session)::

    python -m ...probe_beam_configuration --joints ... --contact \\
        --insertion-mm 25 --insertion-sweep-mm 20,25,30,33,36,40,50,60

Sweep magnet Z at fixed XY/orientation/insertion (reproduces the stale-
wall-Z bug-hunt)::

    python -m ...probe_beam_configuration --joints ... --contact \\
        --insertion-mm 25 --magnet-z-perturb-mm -20,-10,-5,0,5,10,20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import SOURCE_DIPOLE_BODY_AXIS as _SOURCE_DIPOLE_BODY_AXIS_TUPLE

# Single canonical source: rig_calibration.SOURCE_DIPOLE_BODY_AXIS (see
# that module's docstring for the full recalibration history).
SOURCE_DIPOLE_BODY_AXIS = np.asarray(_SOURCE_DIPOLE_BODY_AXIS_TUPLE, dtype=float)


def _dipole_world_direction(magnet_rotvec: np.ndarray) -> np.ndarray:
    R = Rot.from_rotvec(magnet_rotvec).as_matrix()
    d = R @ SOURCE_DIPOLE_BODY_AXIS
    return d / np.linalg.norm(d)


def _centerline_of(result) -> np.ndarray:
    c = np.asarray(result.p, dtype=float)
    if c.shape[0] == 3 and c.shape[1] != 3:
        c = c.T
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True,
                    help="vessel geometry -- wall collision geometry (if --contact) AND "
                         "the reference path drawn/compared against in the plot/printed "
                         "errors either way. MUST already be Z-corrected to match "
                         "build_vessel_plan.BEAM_BASE_PIVOT_Z (see "
                         "vessel_lumen_robot_frame_*_zcorrected.json) -- a stale-Z lumen "
                         "will reproduce the 2026-10-03 bug this tool is built to catch.")
    pose_group = p.add_mutually_exclusive_group(required=True)
    pose_group.add_argument("--joints", type=float, nargs=6, metavar="Q",
                             help="6 joint angles [rad] -- magnet pose derived via FK + "
                                  "the robot's own calibrated flange-to-magnet transform.")
    pose_group.add_argument("--magnet-pose6", type=float, nargs=6, metavar="V",
                             help="magnet pose directly: x y z [m] + rotvec_x rotvec_y "
                                  "rotvec_z [rad], robot frame R. Bypasses robot "
                                  "kinematics entirely -- use this to test a magnet pose "
                                  "that doesn't correspond to any real joint solution yet.")
    p.add_argument("--insertion-mm", type=float, required=True)
    contact_group = p.add_mutually_exclusive_group(required=True)
    contact_group.add_argument("--contact", action="store_true")
    contact_group.add_argument("--no-contact", action="store_true")
    p.add_argument("--insertion-sweep-mm", type=str, default=None,
                    help="comma-separated extra insertion lengths [mm] to also solve and "
                         "overlay, holding magnet pose fixed -- e.g. 20,25,30,36,50")
    p.add_argument("--magnet-z-perturb-mm", type=str, default=None,
                    help="comma-separated magnet world-Z offsets [mm] to sweep (added to "
                         "the chosen magnet Z), holding insertion fixed -- e.g. "
                         "-20,-10,0,10,20. Prints tip response only (no extra plot curves); "
                         "a frozen/non-monotonic response usually means a stale geometry "
                         "reference is dominating the solve, not genuine physics.")
    p.add_argument("--out", type=Path, default=None,
                    help="plot PNG path. Default: debug_outputs/probe_<contact|nocontact>_L<mm>.png")
    args = p.parse_args()

    from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
        BEAM_BASE_PIVOT_Z, BEAM_BASE_PIVOT_XY_ROT,
    )

    beam_base_xyz6 = np.array([
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z,
        3.14159265, 0.0, 0.0,
    ])
    beam_base_xyz = beam_base_xyz6[:3]

    if args.joints is not None:
        from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
        from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin
        q6 = np.asarray(args.joints, dtype=float)
        fk = urik.forward_kinematics(q6, _robot_kin.dh, _robot_kin.T_F_M)
        magnet_xyz = np.asarray(fk.T_R_target)[:3, 3]
        magnet_rotvec = Rot.from_matrix(np.asarray(fk.T_R_target)[:3, :3]).as_rotvec()
        magnet_pose6 = np.concatenate([magnet_xyz, magnet_rotvec])
        print(f"[probe] joints -> FK -> magnet_pose6 = {magnet_pose6.tolist()}")
    else:
        magnet_pose6 = np.asarray(args.magnet_pose6, dtype=float)
        print(f"[probe] magnet_pose6 (given directly) = {magnet_pose6.tolist()}")

    # make_initial_poses is only consulted for the beam-base pivot and a
    # placeholder start_point/L0 -- build_vessel_planning_context re-solves
    # its own IK seed internally, but we override state0 explicitly below
    # with our own magnet_pose6/insertion regardless, so the placeholder
    # start_point here is never actually used for the probe itself.
    initial_poses = (beam_base_xyz6.copy(), magnet_pose6.copy(), args.insertion_mm / 1000.0, 0.01)

    from proper_research.planning.vessel_context import build_vessel_planning_context
    exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=args.lumen_file, insertion_max_m=0.08, initial_poses=initial_poses,
        )
    )
    contact = bool(args.contact)
    model = bundle.models["contact" if contact else "no_contact"]

    L0_m = args.insertion_mm / 1000.0
    p7 = np.concatenate([magnet_pose6, [L0_m]])
    result = model.solve(p7, commit=False, reuse_cache=False)
    centerline = _centerline_of(result)
    tip = np.asarray(result.tip, dtype=float)

    dipole_world = _dipole_world_direction(magnet_pose6[3:6])
    closest_idx = int(np.argmin(np.linalg.norm(lumen_C - tip, axis=1)))
    tangent_local = lumen_C[min(closest_idx + 1, len(lumen_C) - 1)] - lumen_C[closest_idx]
    tangent_local = tangent_local / np.linalg.norm(tangent_local)
    # beam tangent: local centerline direction at the tip end
    beam_tangent = centerline[-1] - centerline[-2]
    beam_tangent = beam_tangent / np.linalg.norm(beam_tangent)
    pos_err_mm = float(np.linalg.norm(tip - lumen_C[closest_idx]) * 1000)
    ang_err_deg = float(np.degrees(np.arccos(np.clip(beam_tangent @ tangent_local, -1, 1))))

    print(f"[probe] condition = {'contact' if contact else 'no-contact'}, insertion = {args.insertion_mm}mm")
    print(f"[probe] magnet-to-beam-base distance = {np.linalg.norm(magnet_pose6[:3]-beam_base_xyz)*1000:.2f}mm")
    print(f"[probe] dipole world direction = {dipole_world.tolist()}  "
          f"(angle off beam-axial -X = {np.degrees(np.arccos(np.clip(-dipole_world[0],-1,1))):.1f}deg)")
    print(f"[probe] tip = {tip.tolist()}")
    print(f"[probe] nearest lumen point index = {closest_idx} (of {len(lumen_C)}), "
          f"position error = {pos_err_mm:.2f}mm, local tangent angle error = {ang_err_deg:.2f}deg")

    if args.magnet_z_perturb_mm:
        print()
        print("[probe] magnet Z-perturbation sweep:")
        for dz_mm in (float(x) for x in args.magnet_z_perturb_mm.split(",")):
            p7_pert = p7.copy()
            p7_pert[2] += dz_mm / 1000.0
            res = model.solve(p7_pert, commit=False, reuse_cache=False)
            print(f"  dz={dz_mm:+7.2f}mm -> tip={np.round(res.tip, 4).tolist()}  "
                  f"tip_z={res.tip[2]*1000:7.2f}mm")

    sweep_shapes = {}
    if args.insertion_sweep_mm:
        for L_mm in (float(x) for x in args.insertion_sweep_mm.split(",")):
            p7_sw = np.concatenate([magnet_pose6, [L_mm / 1000.0]])
            res = model.solve(p7_sw, commit=False, reuse_cache=False)
            sweep_shapes[L_mm] = _centerline_of(res)
            print(f"[probe] insertion sweep L={L_mm:.1f}mm -> tip_z={res.tip[2]*1000:.2f}mm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(centerline[:, 0]*1000, centerline[:, 1]*1000, "o-", color="tab:red",
                 label=f"beam L={args.insertion_mm:.0f}mm", markersize=4, zorder=5)
    axes[1].plot(centerline[:, 0]*1000, centerline[:, 2]*1000, "o-", color="tab:red",
                 label=f"beam L={args.insertion_mm:.0f}mm", markersize=4, zorder=5)
    cmap = plt.get_cmap("viridis")
    for i, (L_mm, c) in enumerate(sorted(sweep_shapes.items())):
        color = cmap(i / max(1, len(sweep_shapes) - 1))
        axes[0].plot(c[:, 0]*1000, c[:, 1]*1000, "-", color=color, alpha=0.7, label=f"L={L_mm:.0f}mm")
        axes[1].plot(c[:, 0]*1000, c[:, 2]*1000, "-", color=color, alpha=0.7, label=f"L={L_mm:.0f}mm")

    axes[0].plot(lumen_C[:, 0]*1000, lumen_C[:, 1]*1000, "k--", label="vessel centreline", linewidth=1.5)
    axes[1].plot(lumen_C[:, 0]*1000, lumen_C[:, 2]*1000, "k--", label="vessel centreline", linewidth=1.5)
    axes[0].plot(*(beam_base_xyz[:2]*1000), "k^", markersize=12, label="beam base", zorder=6)
    axes[1].plot(beam_base_xyz[0]*1000, beam_base_xyz[2]*1000, "k^", markersize=12, label="beam base", zorder=6)
    axes[0].plot(magnet_pose6[0]*1000, magnet_pose6[1]*1000, "gs", markersize=14, label="source magnet", zorder=6)
    axes[1].plot(magnet_pose6[0]*1000, magnet_pose6[2]*1000, "gs", markersize=14, label="source magnet", zorder=6)

    arrow_len_mm = 100
    p_plus = magnet_pose6[:3] + dipole_world * arrow_len_mm/1000/2
    p_minus = magnet_pose6[:3] - dipole_world * arrow_len_mm/1000/2
    axes[0].annotate("", xy=(p_plus[0]*1000, p_plus[1]*1000), xytext=(p_minus[0]*1000, p_minus[1]*1000),
                      arrowprops=dict(arrowstyle="-|>", color="green", lw=4, mutation_scale=25), zorder=7)
    axes[1].annotate("", xy=(p_plus[0]*1000, p_plus[2]*1000), xytext=(p_minus[0]*1000, p_minus[2]*1000),
                      arrowprops=dict(arrowstyle="-|>", color="green", lw=4, mutation_scale=25), zorder=7)

    axes[0].set_xlabel("R.x (mm)"); axes[0].set_ylabel("R.y (mm)"); axes[0].set_title("XY plane (top-down)")
    axes[1].set_xlabel("R.x (mm)"); axes[1].set_ylabel("R.z (mm)"); axes[1].set_title("XZ plane (side)")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(
        f"{'contact' if contact else 'no-contact'} model, L={args.insertion_mm:.0f}mm  |  "
        f"pos_err={pos_err_mm:.1f}mm  tangent_err={ang_err_deg:.1f}deg  |  "
        f"dipole z-component={dipole_world[2]:.3f}",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = args.out or Path("debug_outputs") / (
        f"probe_{'contact' if contact else 'nocontact'}_L{args.insertion_mm:.0f}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"[probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
