#!/usr/bin/env python3
"""Reusable source-magnet + beam calibration / Jacobian sweep.

Moves the source magnet through translation (an xy-plane arc around the beam
base at several radii, plus a Z sweep) and dipole rotation (about world Z and
world Y), plus a COMBINED arc + dipole-rotation sweep, recording the camera
tip, the online model's predicted tip, the full analytic beam Jacobian, the
magnet pose and the dipole direction at every pose.  A single long-lived robot
connection drives the whole sweep (move_j via IK, reconnect-on-failure) --
this is the pattern that has been reliable for these sweeps; the per-process
robot connections that `compare_controllers_live.py` / `open_loop_playback.py`
use are a different (and more fragile) regime for that reason.

Hard constraints (checked per pose, never violated):
  * magnet centre stays >= --min-radius-mm from the beam base
  * magnet z stays within +/- --z-limit-mm of the beam base z

Analysis (also runnable standalone via --analyze-only) fits, per sweep block,
`camera_bend = slope * model_bend + bias` and reports the residual; the
multi-radius arc block additionally disambiguates two explanations for any
residual scaling error:
  * a beam-material error (fit a single moment/stiffness SCALE, distance
    assumed correct), vs
  * a magnet-distance-calibration error (fit a single RADIAL OFFSET applied
    to every commanded magnet position, moment/stiffness assumed correct --
    this is what a wrong T_tcp_magnet_pose6 would look like: "the source
    magnet is closer/farther than we think").
Whichever fits the camera data with lower residual is the more likely
explanation; report both so a human can judge.

    # full sweep (translation + Z + dipole + combined), insertion 30 mm
    python -m proper_research.hardware.online.beam_magnet_calibration_sweep

    # just the analysis, from a previous run's data file
    python -m proper_research.hardware.online.beam_magnet_calibration_sweep \
        --analyze-only calibration_2026-09-10/calib_sweep_<stamp>/data.json

    # skip a block, use a different insertion / radius floor
    python -m proper_research.hardware.online.beam_magnet_calibration_sweep \
        --insertion-mm 34 --min-radius-mm 220 --skip-combined
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[3]
REF_ORI = np.array([-3.07806404, 0.57586908, 0.04569585])


@contextlib.contextmanager
def _quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


# ==========================================================================
# pose generation (all in the beam frame B: B.x axial, B.y in-plane, B.z blind)
# ==========================================================================
@dataclass
class Pose:
    magnet_world: np.ndarray   # (3,) magnet centre in R
    ori_rotvec: np.ndarray     # (3,) TCP orientation in R
    kind: str
    label: str


def _polar(beam_base: np.ndarray, radius_m: float, azimuth_deg: float, z_off_m: float = 0.0):
    p = np.deg2rad(azimuth_deg)
    return beam_base + radius_m * np.array([-np.cos(p), np.sin(p), 0.0]) + np.array([0, 0, z_off_m])


def build_poses(args, beam_base: np.ndarray) -> list[Pose]:
    poses: list[Pose] = []
    min_r = args.min_radius_mm * 1e-3

    if not args.skip_arc:
        for radius_mm in args.arc_radii_mm:
            r = radius_mm * 1e-3
            if r < min_r - 1e-6:
                print(f"[skip] arc radius {radius_mm}mm < floor {args.min_radius_mm}mm")
                continue
            for az in args.arc_azimuths_deg:
                poses.append(Pose(_polar(beam_base, r, az), REF_ORI, "arc",
                                  f"arc_r{int(radius_mm)}_az{int(az):+d}"))

    if not args.skip_z:
        r = max(args.z_sweep_radius_mm, args.min_radius_mm) * 1e-3
        for dz_mm in args.z_offsets_mm:
            if abs(dz_mm) > args.z_limit_mm + 1e-6:
                continue
            poses.append(Pose(_polar(beam_base, r, 0.0, dz_mm * 1e-3), REF_ORI, "z",
                              f"z_r{int(args.z_sweep_radius_mm)}_dz{int(dz_mm):+d}"))

    if not args.skip_dipole:
        for radius_mm in args.dipole_radii_mm:
            r = radius_mm * 1e-3
            if r < min_r - 1e-6:
                print(f"[skip] dipole radius {radius_mm}mm < floor {args.min_radius_mm}mm")
                continue
            for az_pos in args.dipole_positions_deg:
                m = _polar(beam_base, r, az_pos)
                for yaw in args.dipole_yaw_deg:
                    o = (Rot.from_rotvec([0, 0, np.deg2rad(yaw)]) * Rot.from_rotvec(REF_ORI)).as_rotvec()
                    poses.append(Pose(m, o, "dipZ",
                                      f"dipZ_r{int(radius_mm)}_pos{int(az_pos):+d}_yaw{int(yaw):+d}"))
                for pitch in args.dipole_pitch_deg:
                    o = (Rot.from_rotvec([0, np.deg2rad(pitch), 0]) * Rot.from_rotvec(REF_ORI)).as_rotvec()
                    poses.append(Pose(m, o, "dipY",
                                      f"dipY_r{int(radius_mm)}_pos{int(az_pos):+d}_pitch{int(pitch):+d}"))

    if not args.skip_combined:
        r = max(args.combined_radius_mm, args.min_radius_mm) * 1e-3
        for az in args.combined_azimuths_deg:
            m = _polar(beam_base, r, az)
            for yaw in args.combined_yaw_deg:
                o = (Rot.from_rotvec([0, 0, np.deg2rad(yaw)]) * Rot.from_rotvec(REF_ORI)).as_rotvec()
                poses.append(Pose(m, o, "combined", f"comb_r{int(args.combined_radius_mm)}_az{int(az):+d}_yaw{int(yaw):+d}"))

    return poses


# ==========================================================================
# hardware + model setup
# ==========================================================================
def build_model():
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware import robotics_frame_measurement_validation as rfmv
    import proper_research.hardware.beam_hardware_experiment_v2 as ev2

    with _quiet():
        b = ev2._base_module()
        b.configure_bounds_beam_paths(ev2.CONFIG)
        mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
        lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
        mbr = rfmv.model_base_rotation_from_beam(
            np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2]
        )
        adapter, _ = rfmv.build_forward_model_in_shared_frame(
            experiment_cfg=ev2.CONFIG, T_R_B=mapper.T_R_B, lumen=lumen,
            model_base_rotation_R=mbr,
            dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body),
        )
    return mapper, adapter


def model_at(adapter, T_R_B, tcp6, insertion_m):
    from proper_research.hardware import robotics_frame_measurement_validation as rfmv

    _, _, T_R_M = rfmv.source_transform_from_tcp(np.asarray(tcp6, dtype=float), rfmv.CONFIG)
    p8 = rfmv.transform_to_p8(T_R_M, float(insertion_m))
    with _quiet():
        p7, out = adapter.commit_nominal(p8)
        J = np.asarray(adapter.raw_model.jacobian_tip_actuation_tangent(p7), dtype=float).reshape(3, 7)
    m_body = np.asarray(rfmv.CONFIG.source_dipole_unit_in_magnet_body, dtype=float)
    dip_R = T_R_M.rotation @ (m_body / np.linalg.norm(m_body))
    return (
        T_R_B.inverse().apply_points(np.asarray(out, dtype=float).reshape(-1)[:3]),
        T_R_B.inverse().apply_points(np.asarray(T_R_M.translation, dtype=float)),
        T_R_B.rotation.T @ dip_R,
        J,
    )


def run_sweep(args) -> Path:
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.simulation.simulations.initial_conditions import (
        TCP_TO_MAGNET_POSE6, make_initial_poses,
    )
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
    from proper_research.hardware.ur_rtde_robot import URRTDERobot
    from proper_research.hardware.online.state_stream import StateStreamConfig

    pivot_point, _, L0_default, _ = make_initial_poses()
    beam_base = np.asarray(pivot_point[:3], dtype=float)
    ins_m = (args.insertion_mm * 1e-3) if args.insertion_mm else L0_default
    dh = urik.corrected_dh_from_config(urik.CONFIG)
    tcp_frame_off = np.array(TCP_TO_MAGNET_POSE6[:3])

    def tcp_for_magnet(m, ori):
        Rm = Rot.from_rotvec(ori).as_matrix()
        return np.concatenate([np.asarray(m) - Rm @ tcp_frame_off, ori])

    def ik(tcp6, seed):
        r = urik.inverse_kinematics_dls(
            T_R_target=urik.pose6_to_T(tcp6), q_seed_rad=tuple(seed), dh=dh,
            T_F_target=None, cfg=urik.CONFIG,
        )
        return np.array(r.q_rad), r.converged, r.final_position_error_m

    mapper, adapter = build_model()
    T_R_B = mapper.T_R_B

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or (REPO / "calibration_2026-09-10" / f"calib_sweep_{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.json"
    (out_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items()}, indent=2))

    scfg = StateStreamConfig(exposure=29.0)
    camera = CameraSource(
        CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/calib_sweep.png",
                    roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
                    pivot_hint=tuple(scfg.pivot_hint_px)),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: ins_m, frame_processor=mapper,
    )
    camera.start()

    def cam_tipB(n=25, tmo=8.0):
        tips = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < tmo and len(tips) < n:
            est, _ = camera.latest(0.5)
            if est is not None:
                tips.append(np.asarray(est.tip_position_m))
            time.sleep(0.1)
        if not tips:
            return None, 0
        return T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0)), len(tips)

    adv = None
    if not args.skip_beam_length_set:
        adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))
        adv.start()
        time.sleep(2.5)
        print(f"[beam] setting length -> {1e3 * ins_m:.0f} mm")
        for it in range(10):
            tb, nf = cam_tipB()
            if tb is None:
                print("[beam] no camera; skipping length set")
                break
            L = float(np.linalg.norm(tb[:2]))
            err_mm = 1e3 * (ins_m - L)
            print(f"[beam]  iter {it}: {1e3 * L:.1f} mm (target {1e3 * ins_m:.0f}, err {err_mm:+.1f})")
            if abs(err_mm) < 0.7:
                break
            adv.submit_distance(float(np.clip(err_mm, -4.0, 4.0)))
            t0 = time.monotonic()
            while time.monotonic() - t0 < 20:
                fb = adv.feedback()
                if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
                    break
                time.sleep(0.1)
            time.sleep(0.4)
        adv.stop()

    poses = build_poses(args, beam_base)
    print(f"[sweep] {len(poses)} poses queued, insertion {1e3 * ins_m:.1f} mm, "
          f"min radius {args.min_radius_mm} mm, z limit +/-{args.z_limit_mm} mm")

    data: list[dict] = []
    r = URRTDERobot(args.robot_ip, frequency=125.0)

    def ready():
        try:
            return r.get_safety_mode() in (1, 2) and not r.is_protective_stopped() and r.get_robot_mode() == 7
        except Exception:
            return False

    try:
        q_ref = np.array(r.get_joints())
        p_ref = np.array(r.get_tcp_pose())
        seed = q_ref.copy()
        tbc, nf = cam_tipB()
        tbm, mB, dB, J = model_at(adapter, T_R_B, p_ref, ins_m)
        print(f"REF cam {np.round(1e3 * tbc, 2) if tbc is not None else None} "
              f"model {np.round(1e3 * tbm, 2)} magB {np.round(1e3 * mB, 0)}")
        data.append({"label": "ref", "kind": "ref", "tcp": list(p_ref), "ins": ins_m,
                     "cam_tipB": None if tbc is None else list(tbc), "cam_frames": nf,
                     "mod_tipB": list(tbm), "magnetB": list(mB), "dipoleB": list(dB), "J": J.tolist()})

        for pose in poses:
            d = np.linalg.norm(pose.magnet_world - beam_base)
            if d < args.min_radius_mm * 1e-3 - 1e-6:
                print(f"  {pose.label}: r={1e3 * d:.1f}mm < floor {args.min_radius_mm}mm, SKIP")
                continue
            if abs(pose.magnet_world[2] - beam_base[2]) > args.z_limit_mm * 1e-3 + 1e-6:
                print(f"  {pose.label}: z out of +/-{args.z_limit_mm}mm, SKIP")
                continue

            tcp = tcp_for_magnet(pose.magnet_world, pose.ori_rotvec)
            qt, conv, perr = ik(tcp, seed)
            if not conv or perr > 2e-3 or np.max(np.abs(qt - seed)) > 1.0:
                print(f"  {pose.label}: IK reject (conv={conv} err={1e3 * perr:.1f}mm "
                      f"dq={np.max(np.abs(qt - seed)):.2f})")
                continue
            for attempt in (1, 2):
                if not ready():
                    try:
                        r.reconnect()
                    except Exception:
                        pass
                    time.sleep(0.5)
                try:
                    r.move_j(list(qt), speed=0.3, acceleration=0.3)
                    break
                except Exception as exc:
                    print(f"  {pose.label}: move attempt{attempt} {exc!r}")
                    if attempt == 2:
                        break
                    try:
                        r.reconnect()
                    except Exception:
                        pass
                    time.sleep(0.6)
            time.sleep(0.4)
            try:
                qc = np.array(r.get_joints())
                pc = np.array(r.get_tcp_pose())
            except Exception:
                print("  read fail, aborting sweep")
                break
            if np.max(np.abs(qc - qt)) > 0.03:
                print(f"  {pose.label}: not reached, skip")
                continue
            seed = qc.copy()

            tbc, nf = cam_tipB()
            tbm, mB, dB, J = model_at(adapter, T_R_B, pc, ins_m)
            cv = None if tbc is None else 1e3 * tbc[1]
            print(f"  {pose.label:26s} magB[{1e3 * mB[0]:.0f},{1e3 * mB[1]:.0f},{1e3 * mB[2]:.0f}] "
                  f"dipB[{dB[0]:.2f},{dB[1]:.2f},{dB[2]:.2f}] cam_v={cv} mod_v={1e3 * tbm[1]:.2f} ({nf}fr)")
            data.append({"label": pose.label, "kind": pose.kind, "tcp": list(pc), "ins": ins_m,
                         "cam_tipB": None if tbc is None else list(tbc), "cam_frames": nf,
                         "mod_tipB": list(tbm), "magnetB": list(mB), "dipoleB": list(dB), "J": J.tolist()})
            data_path.write_text(json.dumps(data))

        if ready():
            try:
                r.move_j(list(q_ref), speed=0.3, acceleration=0.3)
                print("returned to reference")
            except Exception as exc:
                print("return-to-ref failed:", exc)
    finally:
        camera.stop()
        r.close()

    data_path.write_text(json.dumps(data, indent=1))
    print(f"\n[sweep] saved {len(data)} poses -> {data_path}")
    return out_dir


# ==========================================================================
# analysis
# ==========================================================================
def _slope_bias(y, x):
    A = np.vstack([x, np.ones_like(x)]).T
    s, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(s), float(c)


def _subgroup_and_x(d: dict) -> tuple[str, float]:
    """Split a systematic 'kind_fixed..._swept' label into (fixed_subgroup, x).

    Blocks like dipZ/dipY/combined hold a magnet POSITION fixed while sweeping
    dipole rotation; lumping different positions into one regression against
    rotation angle confounds the position-driven offset with the rotation
    slope and can hide a real sensitivity gap (verified 2026-09-11: a lumped
    dipZ fit read slopeR~0.93, the correct per-position fit reads ~0.30).  So
    group by everything in the label except the innermost swept token.
    """
    kind = d["kind"]
    parts = d["label"].split("_")
    dp = np.array(d["dipoleB"], dtype=float)
    if kind == "arc":
        return parts[1], float(np.array(d["magnetB"])[1] * 1e3)          # by radius; x = magnet B.y
    if kind == "z":
        return "z", float(np.array(d["magnetB"])[2] * 1e3)               # x = magnet B.z
    if kind in ("dipZ", "dipY", "combined"):
        # dipZ/dipY labels are "kind_r<radius>_pos<az>_yaw|pitch<v>"; combined is
        # "comb_r<radius>_az<az>_yaw<v>" -- both have the fixed part in parts[1:3].
        subgroup = f"{parts[1]}_{parts[2]}"
        if kind == "dipY":
            return subgroup, float(np.degrees(np.arctan2(-dp[2], dp[0])))    # x = dipole pitch about B.y
        return subgroup, float(np.degrees(np.arctan2(dp[1], dp[0])))         # x = dipole yaw about B.z
    return "all", float(np.array(d["magnetB"])[1] * 1e3)


def analyse(out_dir: Path) -> None:
    data = json.loads((out_dir / "data.json").read_text())
    rows = [d for d in data if d.get("cam_tipB") is not None]
    if not rows:
        print("no rows with camera data")
        return

    kinds: dict[str, list[dict]] = {}
    for d in rows:
        kinds.setdefault(d["kind"], []).append(d)

    print(f"\n{'block':22s} {'n':>3s}  {'cam slope':>10s} {'mod slope':>10s} {'slopeR':>7s}  "
          f"{'cam bias':>9s} {'mod bias':>9s}  {'RMS':>6s}   [mm, mm/mm or mm/deg]")
    for kind, sub in kinds.items():
        subgroups: dict[str, list[dict]] = {}
        for d in sub:
            g, _ = _subgroup_and_x(d)
            subgroups.setdefault(g, []).append(d)
        slope_ratios, rmss = [], []
        for g, sg in subgroups.items():
            if len(sg) < 2:
                continue
            cam_v = np.array([d["cam_tipB"][1] for d in sg]) * 1e3
            mod_v = np.array([d["mod_tipB"][1] for d in sg]) * 1e3
            x = np.array([_subgroup_and_x(d)[1] for d in sg])
            sc, cc = _slope_bias(cam_v, x)
            sm, cm = _slope_bias(mod_v, x)
            rms = float(np.sqrt(np.mean((mod_v - cam_v) ** 2)))
            ratio = sm / sc if abs(sc) > 1e-6 else float("nan")
            slope_ratios.append(ratio); rmss.append(rms)
            print(f"{kind + ':' + g:22s} {len(sg):>3d}  {sc:10.4f} {sm:10.4f} {ratio:7.3f}  "
                  f"{cc:9.3f} {cm:9.3f}  {rms:6.2f}")
        if len(subgroups) > 1:
            print(f"{kind + ' (mean over ' + str(len(subgroups)) + ' subgroups)':22s} "
                  f"{'':>3s}  {'':>10s} {'':>10s} {np.nanmean(slope_ratios):7.3f}  "
                  f"{'':>9s} {'':>9s}  {np.nanmean(rmss):6.2f}")
        if kind == "dipZ":
            by_radius: dict[str, list[float]] = {}
            for g, sg in subgroups.items():
                if len(sg) < 2:
                    continue
                r_token = g.split("_")[0]  # "r<radius>"
                cam_v = np.array([d["cam_tipB"][1] for d in sg]) * 1e3
                mod_v = np.array([d["mod_tipB"][1] for d in sg]) * 1e3
                x = np.array([_subgroup_and_x(d)[1] for d in sg])
                sc, _ = _slope_bias(cam_v, x)
                sm, _ = _slope_bias(mod_v, x)
                by_radius.setdefault(r_token, []).append(sm / sc if abs(sc) > 1e-6 else float("nan"))
            if len(by_radius) >= 2:
                print("\n=== dipole-ROTATION sensitivity vs distance "
                      "(is the gap a near-field artifact, or constant/material?) ===")
                items = sorted(by_radius.items(), key=lambda kv: int(kv[0][1:]))
                for r_token, ratios in items:
                    print(f"  {r_token:>6s} : mean slopeR = {np.nanmean(ratios):.3f}  (n={len(ratios)})")
                rs = np.array([int(k[1:]) for k, _ in items], dtype=float)
                means = np.array([np.nanmean(v) for _, v in items])
                ds, dc = _slope_bias(means, rs)
                print(f"  slopeR vs radius: {ds * 1e3:+.4f} per metre  "
                      f"(compare the arc block's translation gap, which DID shrink "
                      f"with radius -- if this is ~flat, the rotation gap is NOT a "
                      f"near-field/point-dipole artifact and points to the beam's own "
                      f"magnetisation model instead)")

    # ---- distance-law disambiguation on the multi-radius arc block ----
    arc = [d for d in rows if d["kind"] == "arc"]
    radii = sorted({round(1e3 * np.linalg.norm(np.array(d["magnetB"])[:2]), -1) for d in arc})
    if len(arc) >= 6 and len(radii) >= 2:
        print("\n=== distance-law check (arc block, multiple radii) ===")
        cam_v = np.array([d["cam_tipB"][1] for d in arc]) * 1e3
        mod_v = np.array([d["mod_tipB"][1] for d in arc]) * 1e3
        rms0 = float(np.sqrt(np.mean((mod_v - cam_v) ** 2)))
        # hypothesis A: multiplicative moment/stiffness scale
        best_scale, best_rms_a = 1.0, rms0
        for s in np.linspace(0.5, 2.0, 61):
            r = np.sqrt(np.mean((s * mod_v - cam_v) ** 2))
            if r < best_rms_a:
                best_rms_a, best_scale = r, s
        # hypothesis B: the model recomputed with the magnet moved radially by
        # a constant offset (mimics a T_tcp_magnet_pose6 distance error)
        try:
            from proper_research.hardware import robotics_frame_measurement_validation as rfmv
            from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
            with _quiet():
                mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
            T_R_B = mapper.T_R_B
            _, adapter2 = build_model()
            best_off, best_rms_b = 0.0, rms0
            for off_mm in np.linspace(-40, 40, 41):
                preds = []
                for d in arc:
                    magB = np.array(d["magnetB"])
                    dirB = magB / (np.linalg.norm(magB) + 1e-12)
                    magB_shift = magB + dirB * (off_mm * 1e-3)
                    magR_shift = T_R_B.apply_points(magB_shift)
                    tcp = np.array(d["tcp"])
                    # rebuild the TCP so the magnet sits at magR_shift (translation only)
                    from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6
                    ori = tcp[3:]
                    Rm = Rot.from_rotvec(ori).as_matrix()
                    tcp_shift = np.concatenate([magR_shift - Rm @ np.array(TCP_TO_MAGNET_POSE6[:3]), ori])
                    with _quiet():
                        tbm, _, _, _ = model_at(adapter2, T_R_B, tcp_shift, d["ins"])
                    preds.append(1e3 * tbm[1])
                preds = np.array(preds)
                rr = float(np.sqrt(np.mean((preds - cam_v) ** 2)))
                if rr < best_rms_b:
                    best_rms_b, best_off = rr, off_mm
            print(f"  baseline (no correction)      : RMS {rms0:.2f} mm")
            print(f"  A) moment/stiffness scale={best_scale:.2f}: RMS {best_rms_a:.2f} mm")
            print(f"  B) magnet radial offset={best_off:+.1f} mm : RMS {best_rms_b:.2f} mm  "
                  f"(negative = the magnet behaves as if CLOSER than the FK/T_tcp_magnet_pose6 says)")
            if best_rms_a < best_rms_b - 0.05:
                print("  -> better explained by a BEAM-MATERIAL scale than a magnet-distance offset.")
            elif best_rms_b < best_rms_a - 0.05:
                print("  -> better explained by a MAGNET-DISTANCE calibration offset than a beam-material scale.")
            else:
                print("  -> inconclusive; residuals are comparable.")
        except Exception as exc:  # noqa: BLE001
            print(f"  (distance-offset hypothesis skipped: {exc!r})")

    (out_dir / "summary.json").write_text(json.dumps({
        "n_rows": len(rows),
        "kinds": {k: len(v) for k, v in kinds.items()},
    }, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"arc": "tab:blue", "z": "tab:green", "dipZ": "tab:orange",
             "dipY": "tab:brown", "combined": "tab:red", "ref": "k"}
    for kind, sub in kinds.items():
        mag_y = np.array([d["magnetB"][1] for d in sub]) * 1e3
        cam_v = np.array([d["cam_tipB"][1] for d in sub]) * 1e3
        mod_v = np.array([d["mod_tipB"][1] for d in sub]) * 1e3
        c = colors.get(kind, "tab:purple")
        ax[0].scatter(mag_y, cam_v, color=c, marker="o", s=20, label=f"{kind} cam")
        ax[0].scatter(mag_y, mod_v, color=c, marker="x", s=25, label=f"{kind} model")
    ax[0].set_xlabel("magnet B.y [mm]"); ax[0].set_ylabel("tip B.y [mm]")
    ax[0].set_title("camera (o) vs model (x)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=6, ncol=2)
    z = np.array([d["magnetB"][2] for d in rows]) * 1e3
    cam_z = np.array([d["cam_tipB"][2] for d in rows]) * 1e3
    mod_z = np.array([d["mod_tipB"][2] for d in rows]) * 1e3
    ax[1].scatter(z, cam_z, color="tab:blue", marker="o", s=20, label="cam out-of-plane")
    ax[1].scatter(z, mod_z, color="tab:red", marker="x", s=25, label="model out-of-plane")
    ax[1].set_xlabel("magnet B.z [mm]"); ax[1].set_ylabel("tip B.z [mm]")
    ax[1].set_title("out-of-plane (blind axis, model only)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "calib_sweep.png", dpi=110)
    print(f"\nwrote {out_dir}/calib_sweep.png  {out_dir}/summary.json")


# ==========================================================================
def _arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--robot-ip", default="192.168.56.101")
    p.add_argument("--insertion-mm", type=float, default=None,
                   help="fixed beam length for the whole sweep [mm]; default = "
                        "initial_conditions.make_initial_poses()'s L0 (30 mm)")
    p.add_argument("--min-radius-mm", type=float, default=211.0,
                   help="hard floor: magnet must stay >= this far from the beam base")
    p.add_argument("--z-limit-mm", type=float, default=70.0,
                   help="hard limit: magnet z within +/- this of the beam base z")
    p.add_argument("--skip-beam-length-set", action="store_true")

    p.add_argument("--skip-arc", action="store_true")
    p.add_argument("--arc-radii-mm", type=float, nargs="+", default=[211.0, 250.0, 290.0])
    p.add_argument("--arc-azimuths-deg", type=float, nargs="+", default=[0, 15, 30, -15, -30])

    p.add_argument("--skip-z", action="store_true")
    p.add_argument("--z-sweep-radius-mm", type=float, default=250.0)
    p.add_argument("--z-offsets-mm", type=float, nargs="+", default=[0, 30, 60, -30, -60])

    p.add_argument("--skip-dipole", action="store_true")
    p.add_argument("--dipole-radii-mm", type=float, nargs="+", default=[250.0],
                   help="one or more radii for the dipole-rotation block -- pass "
                        "several (e.g. 211 250 290) to test whether the rotation "
                        "sensitivity gap shrinks with distance the way translation's "
                        "does (point-dipole near-field artifact) or stays constant "
                        "(beam-magnetisation-model gap).")
    p.add_argument("--dipole-positions-deg", type=float, nargs="+", default=[0, 20])
    p.add_argument("--dipole-yaw-deg", type=float, nargs="+", default=[0, 10, 20, -10, -20])
    p.add_argument("--dipole-pitch-deg", type=float, nargs="+", default=[15, -15])

    p.add_argument("--skip-combined", action="store_true")
    p.add_argument("--combined-radius-mm", type=float, default=225.0)
    p.add_argument("--combined-azimuths-deg", type=float, nargs="+", default=[-25, 0, 25])
    p.add_argument("--combined-yaw-deg", type=float, nargs="+", default=[-15, 0, 15])

    p.add_argument("--out-dir", default=None)
    p.add_argument("--analyze-only", default=None)
    return p.parse_args()


def main() -> None:
    args = _arguments()
    if args.analyze_only:
        analyse(Path(args.analyze_only))
        return
    out_dir = run_sweep(args)
    analyse(out_dir)


if __name__ == "__main__":
    main()
