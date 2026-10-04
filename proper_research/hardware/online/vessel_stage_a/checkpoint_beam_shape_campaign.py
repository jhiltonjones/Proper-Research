#!/usr/bin/env python3
"""Static-checkpoint predicted-vs-measured FULL beam-shape campaign for the
2026-09-27/28 vessel plan (215mm magnet start, 210.43mm fixed magnet-to-
beam-base safety floor, 65mm insertion ceiling -- see
plans/vessel_lumen_zraise30mm_start215_excl210_ins65mm_2026-09-27).

At each of several checkpoints (a target arm pose + a target insertion
length), this script:
  1. safely moves the arm to the checkpoint's joints (magnet-exclusion
     path-safety checked before any motion, with an automatic retreat-
     waypoint retry if the direct straight-line path would violate the
     210.43mm floor mid-transit -- the tight floor makes this a routine
     occurrence, not a fault);
  2. closes the loop on vision-measured insertion length to physically
     drive the advancer to the checkpoint's target insertion (the advancer
     has no encoder -- nothing moves unless commanded);
  3. captures a live frame and reconstructs the FULL beam centreline (not
     just the tip) via the same marker-detection/skeleton-routing pipeline
     bounds_beam.py uses, mapped into robot frame R through the SAME
     PlanarPixelCalibration + T_R_B chain the live tip tracker
     (NewFrameTipMapper) uses -- this is the piece that needed fixing
     2026-09-28: reconstruct_beam_within_vessel's OWN marker pipeline
     disagreed with the validated live tracker (different ROI file, and a
     hardcoded base_px debug leftover), so marker detection is done here
     directly instead, with the SAME ROI/thresholds/pivot_hint as the
     live tracker;
  4. computes the model-PREDICTED full beam centreline at the exact
     measured (q, insertion) state, from BOTH the contact-aware model and
     the free-space (no-contact) model, so the two can be compared without
     any confound from a different live trajectory.

Results (raw measured + both predicted centrelines, per checkpoint) are
saved to a JSON file. Use plot_checkpoint_beam_shape_campaign.py to turn
that into the residual-vs-arclength / contact-vs-no-contact comparison
figures.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign \\
        --out-dir checkpoint_campaign_logs/myrun --live

Edit CHECKPOINTS below to add/change checkpoint poses+insertions.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch

zraise_patch.apply(30.0)

import proper_research.simulation.simulations.initial_conditions as initial_conditions_mod
import proper_research.planning.planning_context as planning_context_mod

# ---------------------------------------------------------------------------
# This vessel campaign's fixed geometry (2026-09-27/28) -- update these if
# you build a new vessel plan with a different start/floor/insertion.
# ---------------------------------------------------------------------------
PIVOT = np.array([0.670575, -0.719727, 0.013433, 3.14159265, 0.0, 0.0])
START_PT = [0.31133949, -0.65191377, 0.01328917, -3.07774931, 0.57528249, 0.04521943]
BEAM_BASE = np.array([[0.670575, -0.719727, 0.013433]])
EXCLUSION_RADIUS_M = 0.210430
INSERTION_MAX_M = 0.065
LUMEN_FILE = "vessel_lumen_robot_frame_raised3cm_2026-09-27.json"
ROBOT_IP = "192.168.56.101"
ADVANCER_PORT = "/dev/ttyACM0"


def _make_initial_poses():
    return PIVOT.copy(), np.array(START_PT), 0.025, 0.01


initial_conditions_mod.make_initial_poses = _make_initial_poses
planning_context_mod.make_initial_poses = _make_initial_poses

from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.planning.planning_context import make_design_config, make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import (
    build_controller,
    _resolve_robot_kinematics,
)
from proper_research.hardware.online.state_stream import StateStreamConfig, NewFrameTipMapper
from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
from proper_research.hardware.online.messages import now_monotonic
from proper_research.vision import bounds_beam as bb
from proper_research.hardware.ur_rtde_robot import URRTDERobot
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a import common

# ---------------------------------------------------------------------------
# Checkpoints: (label, arm joints q [rad], target insertion length [mm]).
# The q's below are real states pulled from a completed live open-loop run
# of this plan (close_loop_logs/myrun/vessel_start215_excl210_ins65mm_
# openloop_v2_20260928T094802Z/path_follow.jsonl) at 5 points spanning its
# insertion range, plus one same-insertion/different-pose control built by
# IK-solving a pose 20mm further from the beam base than the "mid" checkpoint
# (same orientation) -- see build_alt_pose_waypoint() below if you need to
# regenerate that one for a different plan.
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    {"label": "low_ins_27mm", "target_L_mm": 27.0,
     "q": np.array([-0.802955, -1.976075, -1.658898, -1.061503, 1.528828, -1.992319])},
    {"label": "lowmid_ins_33mm", "target_L_mm": 33.0,
     "q": np.array([-0.785408, -1.960485, -1.680459, -1.085323, 1.54502, -1.987218])},
    {"label": "mid_ins_40mm", "target_L_mm": 40.0,
     "q": np.array([-0.697775, -1.933221, -1.728423, -1.242191, 1.603471, -1.916181])},
    {"label": "midhigh_ins_47mm", "target_L_mm": 47.0,
     "q": np.array([-0.652266, -1.952528, -1.70435, -1.277968, 1.560683, -1.871665])},
    {"label": "high_ins_54mm", "target_L_mm": 54.0,
     "q": np.array([-0.65019, -1.955203, -1.697366, -1.308336, 1.567468, -1.843085])},
]


_robot_kin = _resolve_robot_kinematics(make_robot_config())


def magnet_transform_fn(q6: np.ndarray) -> np.ndarray:
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def build_alt_pose_waypoint(reference_q: np.ndarray, extra_distance_m: float = 0.020) -> np.ndarray:
    """IK-solve a pose `extra_distance_m` further from the beam base than
    `reference_q`, same orientation -- a same-insertion/different-pose
    control point (insertion is a separate, arm-independent DOF, so this
    changes the magnet's configuration without touching insertion)."""
    T_ref = urik.forward_kinematics(reference_q, _robot_kin.dh, _robot_kin.T_F_M)
    m_ref = np.asarray(T_ref.T_R_target[:3, 3])
    d_ref = np.linalg.norm(m_ref - BEAM_BASE[0])
    direction = (m_ref - BEAM_BASE[0]) / d_ref
    alt_xyz = BEAM_BASE[0] + direction * (d_ref + extra_distance_m)
    rotvec_ref = Rot.from_matrix(T_ref.T_R_target[:3, :3]).as_rotvec()
    T_target = urik.pose6_to_T(np.r_[alt_xyz, rotvec_ref])
    ik = urik.inverse_kinematics_dls(
        T_R_target=T_target, q_seed_rad=reference_q, dh=_robot_kin.dh,
        T_F_target=_robot_kin.T_F_M, cfg=_robot_kin.ik_cfg,
    )
    if not ik.converged:
        raise RuntimeError("alt-pose waypoint IK did not converge")
    return ik.q_rad


def safe_move(q_target: np.ndarray, label: str) -> None:
    """Move to q_target, routing through a retreat waypoint (further from
    the beam base) if the direct straight-line path would violate the
    210.43mm floor -- routine given how tight that floor is, not a fault."""
    common.check_robot_safe(robot_ip=ROBOT_IP)
    try:
        common.reset_to_plan_initial_safe(
            q_target, robot_ip=ROBOT_IP, magnet_transform_fn=magnet_transform_fn,
            magnet_exclusion_lumen_C_m=BEAM_BASE, magnet_exclusion_radius_m=EXCLUSION_RADIUS_M,
        )
        return
    except RuntimeError as exc:
        print(f"  [{label}] direct path unsafe ({exc}); routing via a retreat waypoint")

    robot = URRTDERobot(ROBOT_IP, frequency=125.0)
    robot.connect()
    q_now = np.array(robot.get_joints())
    robot.close()
    m_now = magnet_transform_fn(q_now)
    d_now = np.linalg.norm(m_now - BEAM_BASE[0])
    direction = (m_now - BEAM_BASE[0]) / d_now
    retreat_xyz = BEAM_BASE[0] + direction * (d_now + 0.030)
    T_now = urik.forward_kinematics(q_now, _robot_kin.dh, _robot_kin.T_F_M)
    rotvec_now = Rot.from_matrix(T_now.T_R_target[:3, :3]).as_rotvec()
    T_retreat = urik.pose6_to_T(np.r_[retreat_xyz, rotvec_now])
    ik = urik.inverse_kinematics_dls(
        T_R_target=T_retreat, q_seed_rad=q_now, dh=_robot_kin.dh,
        T_F_target=_robot_kin.T_F_M, cfg=_robot_kin.ik_cfg,
    )
    if not ik.converged:
        raise RuntimeError(f"retreat-waypoint IK failed for {label}")
    common.reset_to_plan_initial_safe(
        ik.q_rad, robot_ip=ROBOT_IP, magnet_transform_fn=magnet_transform_fn,
        magnet_exclusion_lumen_C_m=BEAM_BASE, magnet_exclusion_radius_m=EXCLUSION_RADIUS_M,
    )
    common.reset_to_plan_initial_safe(
        q_target, robot_ip=ROBOT_IP, magnet_transform_fn=magnet_transform_fn,
        magnet_exclusion_lumen_C_m=BEAM_BASE, magnet_exclusion_radius_m=EXCLUSION_RADIUS_M,
    )


def _build_raised_camera_mapper(out_dir: Path):
    base_cfg = StateStreamConfig(exposure=29.0, marker_min_count=2)
    pose6 = list(base_cfg.T_robot_beam_pose6)
    pose6[2] += 0.03
    scfg = dataclasses.replace(base_cfg, T_robot_beam_pose6=tuple(pose6))
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(cam_index=0, exposure=29.0, image_filename=str(out_dir / "_capture.png"),
                     roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
                     pivot_hint=tuple(scfg.pivot_hint_px)),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: 0.03, frame_processor=mapper,
    )
    return camera, mapper, scfg


def measure_current_insertion_mm(camera: CameraSource, scfg: StateStreamConfig, n: int = 10, timeout_s: float = 15.0) -> float:
    pivot_xyz = np.asarray(scfg.T_robot_beam_pose6[:3], dtype=float)
    samples = []
    deadline = now_monotonic() + timeout_s
    while len(samples) < n and now_monotonic() < deadline:
        est, age = camera.latest(0.5)
        if est is not None:
            samples.append(float(np.linalg.norm(np.asarray(est.tip_position_m) - pivot_xyz)) * 1000.0)
        time.sleep(0.05)
    if not samples:
        raise RuntimeError("no live tip detections while measuring insertion")
    return float(np.median(samples))


def reset_insertion_to_target(target_mm: float, *, live: bool, out_dir: Path,
                               rate_mm_s: float = 1.5, tolerance_mm: float = 0.4, max_iters: int = 6) -> float:
    """Closed-loop drive the advancer to target_mm, using the RAISED pivot
    (the generic advancer_excitation/reset_insertion.py uses an unraised
    hardcoded pivot and is NOT correct for this raised setup)."""
    camera, mapper, scfg = _build_raised_camera_mapper(out_dir)
    camera.start()
    advancer = AdvancerSink(AdvancerSinkConfig(port=ADVANCER_PORT, dry_run=not live))
    advancer.start()
    try:
        time.sleep(1.0)
        current = measure_current_insertion_mm(camera, scfg)
        print(f"  [insertion] start: L={current:.2f}mm target={target_mm:.2f}mm")
        for it in range(max_iters):
            delta = target_mm - current
            if abs(delta) <= tolerance_mm:
                print(f"  [insertion] within tolerance after {it} correction(s)")
                break
            duration_s = abs(delta) / rate_mm_s
            rate_m_s = np.sign(delta) * rate_mm_s * 1e-3
            t_last = now_monotonic()
            deadline = t_last + duration_s
            while now_monotonic() < deadline:
                now = now_monotonic()
                advancer.submit_rate(rate_m_s, now - t_last)
                t_last = now
                time.sleep(0.02)
            advancer.submit_rate(0.0, 0.0)
            deadline_settle = now_monotonic() + 5.0
            while now_monotonic() < deadline_settle:
                fb = advancer.feedback()
                if abs(fb.residual_mm) < advancer.config.min_command_mm and fb.commands_in_flight == 0:
                    break
                time.sleep(0.05)
            time.sleep(1.0)
            current = measure_current_insertion_mm(camera, scfg)
            print(f"  [insertion] iter {it + 1}: L={current:.2f}mm (target {target_mm:.2f}mm)")
        else:
            print(f"  [insertion] WARNING: did not converge (final L={current:.2f}mm)")
        return current
    finally:
        advancer.submit_rate(0.0, 0.0)
        time.sleep(0.2)
        advancer.stop()
        camera.stop()


def measure_beam_shape(out_dir: Path):
    """Capture a frame and return (beam_R (N,3), L_chord_m). Marker
    detection uses the SAME ROI/thresholds/pivot_hint as the validated live
    tip tracker -- NOT bounds_beam.reconstruct_beam_within_vessel's own
    pipeline, which uses a different ROI file and disagreed badly when
    checked directly (2026-09-28)."""
    camera, mapper, scfg = _build_raised_camera_mapper(out_dir)
    camera.start()
    try:
        time.sleep(1.5)
        est, age = camera.latest(1.0)
        if est is None:
            raise RuntimeError("no live tip estimate")
        raw_frame = camera._latest_frame
        if raw_frame is None:
            raise RuntimeError("no raw frame")
    finally:
        camera.stop()

    candidates = bb.detect_4_red_markers_in_roi(
        raw_frame, roi_box=None, roi_polygon=mapper._roi_polygon,
        min_area=scfg.marker_min_area_px2, max_area=scfg.marker_max_area_px2,
        sat_min=scfg.marker_sat_min, val_min=scfg.marker_val_min,
        hue1_high=scfg.marker_red_hue1_high, hue2_low=scfg.marker_red_hue2_low,
        show_debug=False, min_markers=scfg.marker_min_count, max_markers=scfg.marker_max_count,
    )
    base_c, mag_c, tangent_c, tip_c = bb.order_beam_marker_candidates(candidates, pivot_hint=tuple(scfg.pivot_hint_px))
    markers = {
        "base_px": tuple(base_c["point"]),
        "mag_start_px": tuple(mag_c["point"]) if mag_c is not None else None,
        "tangent_start_px": tuple(tangent_c["point"]) if tangent_c is not None else tuple(tip_c["point"]),
        "tip_px": tuple(tip_c["point"]),
    }
    beam_points_marker_px, ordered_pts = bb.beam_polyline_from_markers(markers, n_samples_per_segment=40)
    ordered_pts = np.asarray(ordered_pts, float)
    _length_px, _skeleton, beam_path_px = bb.compute_marker_anchored_black_beam_length_px(
        image_bgr=raw_frame, ordered_pts=ordered_pts, threshold=100, tube_radius_px=25, bridge_radius_px=8,
    )
    beam_px = np.asarray(beam_path_px, float)
    if beam_px.ndim != 2 or beam_px.shape[0] < 2:
        beam_px = np.asarray(beam_points_marker_px, float)

    base_px_arr = np.asarray(markers["base_px"], dtype=float).reshape(1, 2)
    beam_B = mapper.calibration.pixels_to_beam(beam_px)
    offset_B = mapper.calibration.pixels_to_beam(base_px_arr)
    beam_R = mapper.T_R_B.apply_points(beam_B - offset_B)

    pivot_xyz_raised = np.asarray(scfg.T_robot_beam_pose6[:3], dtype=float)
    L_chord = float(np.linalg.norm(beam_R[-1] - pivot_xyz_raised))
    return beam_R, L_chord


def build_adapters():
    """Build BOTH the contact-aware and free-space (no-contact) diagnostic
    adapters, sharing the same lumen/robot config."""
    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = build_vessel_planning_context(
        lumen_file=LUMEN_FILE, insertion_max_m=INSERTION_MAX_M,
    )
    adapter_contact = controller_pack["plant_diagnostic_joint_adapter"]

    _, start_point0, L0_0, dt0 = _make_initial_poses()
    plant_model_nc = bundle.models["no_contact"]
    controller_pack_nc = build_controller(
        start_point=start_point0, L0=L0_0, dt=dt0,
        plant_model=plant_model_nc, jacobian_model=plant_model_nc,
        lumen_C=centreline, lumen_R=lumen_R,
        run_cfg=exp_cfg.controller, design_cfg=make_design_config(),
        robot_cfg=dataclasses.replace(make_robot_config(), insertion_max_m=INSERTION_MAX_M),
    )
    adapter_nocontact = controller_pack_nc["plant_diagnostic_joint_adapter"]
    return adapter_contact, adapter_nocontact, centreline, lumen_R


def predict_centreline(adapter, q6: np.ndarray, L_m: float) -> np.ndarray:
    forward_adapter = adapter.beam_output_fn.forward_adapter
    state = np.concatenate([q6, [L_m]])
    forward_adapter.start_step()
    adapter.forward_output(state, commit=False)
    return forward_adapter.last_p_centerline.T  # (N, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--live", action="store_true", help="actually drive the advancer (default: dry-run)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_contact, adapter_nocontact, centreline, lumen_R = build_adapters()
    print("[campaign] built contact + no-contact adapters")

    checkpoints = list(CHECKPOINTS)
    checkpoints.append({
        "label": "mid_ins_40mm_ALT_POSE", "target_L_mm": 40.0,
        "q": build_alt_pose_waypoint(CHECKPOINTS[2]["q"]),
    })

    results = []
    for i, cp in enumerate(checkpoints):
        label = cp["label"]
        print(f"\n[{i + 1}/{len(checkpoints)}] {label} (target L={cp['target_L_mm']}mm) ...")

        safe_move(cp["q"], label)
        time.sleep(0.5)
        reset_insertion_to_target(cp["target_L_mm"], live=args.live, out_dir=out_dir)

        robot = URRTDERobot(ROBOT_IP, frequency=125.0)
        robot.connect()
        q_meas = np.array(robot.get_joints())
        robot.close()

        beam_R_measured, L_chord = measure_beam_shape(out_dir)
        print(f"  q_meas={q_meas.tolist()}  L_chord={L_chord * 1000:.2f}mm  n_points={beam_R_measured.shape[0]}")

        pred_contact = predict_centreline(adapter_contact, q_meas, L_chord)
        pred_nocontact = predict_centreline(adapter_nocontact, q_meas, L_chord)

        results.append({
            "label": label,
            "target_L_mm": cp["target_L_mm"],
            "q_meas": q_meas.tolist(),
            "L_chord_m": L_chord,
            "beam_R_measured": beam_R_measured.tolist(),
            "pred_contact": pred_contact.tolist(),
            "pred_nocontact": pred_nocontact.tolist(),
        })

    out_path = out_dir / "checkpoint_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[campaign] saved -> {out_path}")


if __name__ == "__main__":
    main()
