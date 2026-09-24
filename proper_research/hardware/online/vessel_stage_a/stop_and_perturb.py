#!/usr/bin/env python3
"""Stop-and-perturb model-validation campaign for the vessel study
(2026-09-24), per the user's protocol: BEFORE trusting a relinearized
contact-model MPC, validate the contact/no-contact beam models against
real hardware perturbations at fixed operating points -- not just
whether they reproduce the open-loop path (which only excites roughly
one combination of the 7 configuration directions).

At each of a small set of reference nodes (early/mid/late along the
vessel plan), holds the robot+insertion at the reference state z0,
measures the settled tip, then perturbs along 4 physically-interpretable
directions (min-joint-cost magnet-toward-lumen, orthogonal lateral,
null-space-of-translation orientation, and insertion), symmetric +/-,
returning to z0 and re-measuring between every perturbation (repeatability/
hysteresis check). A final mixed (held-out) perturbation combines two
directions to test generalization beyond the individual coordinate axes
used to build the empirical Jacobian.

This script only COLLECTS data -- position, transient tip trajectory, and
the contact/no_contact model's own prediction + analytic Jacobian at
every hold point. All of E_0/g_i/theta_i/A_i/linearity/t_90/return-
repeatability are computed OFFLINE afterward from the logged JSON, not
live -- matching the user's explicit preference to keep expensive
computation out of the control-adjacent hot path (see this session's
process-isolated-worker precedent) and to avoid conflating data
collection with analysis.

Every robot motion (to a node, to a perturbed pose, back to a node, back
to the plan's own start at the end) is routed through the SAME magnet-
exclusion-radius / magnet-z-bounds path-safety check validated live this
session (see common.py's `_validate_reset_path_safe`), generalized here
to arbitrary (q_from, q_to) pairs, not just the reset-to-start case.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.stop_and_perturb \\
        --plan-dir uprgrade_configuration/.../time_parameterized_configuration_path \\
        --lumen-file vessel_lumen_robot_frame_raised3cm_trimmed.json \\
        --insertion-max-mm 55 \\
        --magnet-exclusion-reference-joints vessel_magnet_exclusion_reference_joints_2026-09-22.json \\
        --out-dir stop_perturb_logs/myrun \\
        --dry-run          # print the full planned sequence + safety checks, touch nothing

Drop --dry-run to actually move the robot, after reviewing the dry-run
output.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.planning_context import make_robot_config
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)

from . import common

# --- raised-base patch, applied BEFORE any build_vessel_planning_context
# call -- forgetting this was caught live 2026-09-24 (see
# step_contact_model_validation.py's own note): the model's beam-base/
# lumen geometry must match the physically raised rig, or predictions get
# a spurious ~30mm offset unrelated to real model accuracy.
import proper_research.simulation.simulations.initial_conditions as _initial_conditions_mod
import proper_research.planning.planning_context as _planning_context_mod

_ORIG_MAKE_INITIAL_POSES = _initial_conditions_mod.make_initial_poses


def _raised_make_initial_poses():
    p, s, L, dt = _ORIG_MAKE_INITIAL_POSES()
    p = np.array(p, dtype=float).copy()
    s = np.array(s, dtype=float).copy()
    p[2] += common.Z_RAISE_M
    s[2] += common.Z_RAISE_M
    return p, s, L, dt


_robot_kin = _resolve_robot_kinematics(make_robot_config())


def _magnet_transform_fn(q6: np.ndarray) -> np.ndarray:
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def _geom_jac(q6: np.ndarray) -> np.ndarray:
    return urik.geometric_jacobian(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)


def _min_norm_dir(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    G = A @ A.T
    return A.T @ np.linalg.solve(G, u)


def _pure_rotation_dir(A_lin: np.ndarray, A_ang: np.ndarray, u_ang: np.ndarray) -> np.ndarray:
    n = A_lin.shape[1]
    P_null = np.eye(n) - A_lin.T @ np.linalg.solve(A_lin @ A_lin.T, A_lin)
    B = A_ang @ P_null
    w = B.T @ np.linalg.solve(B @ B.T + 1e-12 * np.eye(3), u_ang)
    return P_null @ w


# ---------------------------------------------------------------------
# node/direction design (mirrors the offline-verified design pass)
# ---------------------------------------------------------------------
NODES = {"early": 20, "mid": 90, "late": 140}
LIN_STEP_M = 0.008
TOWARD_STEP_M = 0.008    # restored to the original 8mm: with the 230mm floor
                         # (vs. the previous 309.2mm empirical-vessel one)
                         # there's ~84mm of margin at these nodes, plenty of
                         # room for a full-size perturbation.
ANG_STEP_RAD = 0.05
INS_STEP_M = 0.005

MAGNET_RISE_LIMIT_M = 0.04
MAGNET_FLOOR_MARGIN_M = 0.04
# 2026-09-24: dropped from the vessel-specific empirical exclusion radius
# (314.2mm true, 309.2mm w/ 5mm tolerance -- left ~0mm margin for any
# "closer" perturbation) to the more fundamental, already hardware-
# validated mechanical/geometric floor from the original calibration
# campaign: magnet centre >= 230mm from the beam base/pivot (magnet is
# ~150mm long, near edge 155mm from base -- see
# magnet-workspace-constraints memory). This is a genuine safety floor,
# not merely a tighter/looser choice -- it's the number this project has
# actually validated as mechanically safe.
MAGNET_EXCLUSION_RADIUS_M = 0.23

N_PATH_SAMPLES = 100
JOINT_MOVE_SPEED = 0.15
JOINT_MOVE_ACCEL = 0.3
SETTLE_S = 2.0
TRANSIENT_DURATION_S = 4.0
TRANSIENT_PERIOD_S = 0.1


def build_node_directions(q0, L0, ref_idx, desired_tangent, lumen_C):
    m0 = _magnet_transform_fn(q0)
    dists = np.linalg.norm(lumen_C - m0[None, :], axis=1)
    j_near = int(np.argmin(dists))
    gap0_m = float(dists[j_near])
    r_hat = (m0 - lumen_C[j_near])
    r_hat = r_hat / max(np.linalg.norm(r_hat), 1e-12)
    tangent = np.asarray(desired_tangent, dtype=float)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)

    u_toward = -r_hat
    u_lateral = np.cross(r_hat, tangent)
    u_lateral = u_lateral / max(np.linalg.norm(u_lateral), 1e-12)
    orient_axis = u_lateral.copy()

    J6 = _geom_jac(q0)
    A_lin, A_ang = J6[:3, :], J6[3:, :]

    v_toward = _min_norm_dir(A_lin, u_toward)
    v_toward /= max(np.linalg.norm(v_toward), 1e-12)
    v_lateral = _min_norm_dir(A_lin, u_lateral)
    v_lateral /= max(np.linalg.norm(v_lateral), 1e-12)
    v_orient = _pure_rotation_dir(A_lin, A_ang, orient_axis)
    v_orient /= max(np.linalg.norm(v_orient), 1e-12)

    return dict(
        m0=m0, gap0_m=gap0_m, r_hat=r_hat, tangent=tangent,
        directions={
            "toward_lumen": dict(v=v_toward, step=TOWARD_STEP_M, kind="joint"),
            "lateral": dict(v=v_lateral, step=LIN_STEP_M, kind="joint"),
            "orientation": dict(v=v_orient, step=ANG_STEP_RAD, kind="joint"),
            "insertion": dict(v=None, step=INS_STEP_M, kind="insertion"),
        },
        mixed=dict(
            v=(v_toward + v_lateral), step=1.0, kind="joint_mixed",
            components=[("toward_lumen", v_toward, TOWARD_STEP_M), ("lateral", v_lateral, LIN_STEP_M)],
        ),
    )


def path_clears(q_from, q_to, *, lumen_C, excl_radius_m, z_bounds, n=N_PATH_SAMPLES):
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    worst_gap = math.inf
    worst_z_viol = None
    for t in np.linspace(0.0, 1.0, n):
        q = q_from + t * (q_to - q_from)
        xyz = _magnet_transform_fn(q)
        gap = float(np.min(np.linalg.norm(lumen_C - xyz[None, :], axis=1)))
        worst_gap = min(worst_gap, gap)
        if xyz[2] < z_bounds[0] or xyz[2] > z_bounds[1]:
            worst_z_viol = float(xyz[2])
    clear = worst_gap >= excl_radius_m and worst_z_viol is None
    return clear, worst_gap, worst_z_viol


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--insertion-max-mm", type=float, required=True)
    p.add_argument("--magnet-exclusion-reference-joints", required=True)
    p.add_argument("--magnet-exclusion-radius-mm", type=float, default=MAGNET_EXCLUSION_RADIUS_M * 1e3,
                    help="live safety floor: minimum magnet-to-lumen distance, mm. Defaults to "
                         "the validated mechanical floor (230mm, magnet centre >= 23cm from the "
                         "beam base -- see magnet-workspace-constraints memory), not the tighter "
                         "vessel-specific empirical radius used by the live MPC runner.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nodes", default="early,mid,late")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                    help="print the full planned move sequence + safety checks, touch no hardware")
    args = p.parse_args()

    _initial_conditions_mod.make_initial_poses = _raised_make_initial_poses
    _planning_context_mod.make_initial_poses = _raised_make_initial_poses

    npz_path = glob.glob(args.plan_dir + "/*.npz")[0]
    plan = np.load(npz_path)
    state_reference = plan["state_reference"]
    desired_tangent = plan["desired_tangent"]

    exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance = build_vessel_planning_context(
        lumen_file=args.lumen_file, insertion_max_m=args.insertion_max_mm * 1e-3,
    )
    adapters = {
        True: build_diagnostic_adapter(beam_model=bundle.models["contact"], controller_pack=controller_pack, jacobian_mode="accurate"),
        False: build_diagnostic_adapter(beam_model=bundle.models["no_contact"], controller_pack=controller_pack, jacobian_mode="accurate"),
    }

    with open(args.magnet_exclusion_reference_joints) as f:
        ref = json.load(f)
    joints_rad = np.asarray(ref["joints_rad"], dtype=float)
    magnet_ref_xyz = _magnet_transform_fn(joints_rad)
    exclusion_radius_true_m = float(np.min(np.linalg.norm(
        np.asarray(lumen_C, dtype=float).reshape(-1, 3) - magnet_ref_xyz[None, :], axis=1
    )))
    exclusion_live_m = args.magnet_exclusion_radius_mm * 1e-3
    lumen_C_arr = np.asarray(lumen_C, dtype=float).reshape(-1, 3)

    q0_plan, L0_plan = common.load_plan_initial_state(args.plan_dir)
    magnet_z_start = float(_magnet_transform_fn(q0_plan)[2])
    z_bounds = (magnet_z_start - MAGNET_FLOOR_MARGIN_M, magnet_z_start + MAGNET_RISE_LIMIT_M)

    print(f"[stop-perturb] exclusion live threshold={exclusion_live_m*1e3:.1f}mm "
          f"(fixed mechanical floor; empirical vessel-reference radius for context: "
          f"{exclusion_radius_true_m*1e3:.1f}mm)")
    print(f"[stop-perturb] magnet-z bounds=[{z_bounds[0]*1e3:.1f},{z_bounds[1]*1e3:.1f}]mm")

    node_names = [n.strip() for n in args.nodes.split(",") if n.strip()]
    node_plan = {}
    for name in node_names:
        ref_idx = NODES[name]
        z0 = state_reference[ref_idx]
        q0, L0 = z0[:6].copy(), float(z0[6])
        info = build_node_directions(q0, L0, ref_idx, desired_tangent[ref_idx], lumen_C_arr)
        node_plan[name] = dict(ref_idx=ref_idx, q0=q0, L0=L0, **info)

    # --- build + safety-check the full move sequence before touching anything ---
    # Transitions BETWEEN nodes must NOT be a single direct move_j: caught
    # live 2026-09-24 in this script's own dry-run -- the straight-line
    # joint-space path from one node's reference state to another's dips
    # below the exclusion threshold (305.1mm) even though both endpoints
    # individually sit near 311-314mm, since the nodes can be far apart
    # along the vessel path. Instead, transit via the plan's OWN
    # intermediate reference waypoints (dense stride), which are safe by
    # construction -- this is exactly the path the validated open-loop run
    # already executed.
    TRANSIT_STRIDE = 5

    def transit_waypoints(from_ref_idx, to_ref_idx):
        if from_ref_idx == to_ref_idx:
            return []
        step = TRANSIT_STRIDE if to_ref_idx > from_ref_idx else -TRANSIT_STRIDE
        idxs = list(range(from_ref_idx, to_ref_idx, step))
        idxs.append(to_ref_idx)
        return idxs[1:]  # skip the starting point itself

    sequence = []  # list of dicts: {kind, node, dir, sign, q_target or L_target, label}
    _prev_ref_idx = 0
    for name in node_names:
        np_ = node_plan[name]
        for wp_idx in transit_waypoints(_prev_ref_idx, np_["ref_idx"]):
            sequence.append(dict(kind="transit", node=name, label=f"transit_to_{name}_ref{wp_idx}",
                                  q=state_reference[wp_idx, :6].copy(), L=float(state_reference[wp_idx, 6])))
        _prev_ref_idx = np_["ref_idx"]
        sequence.append(dict(kind="hold", node=name, label=f"{name}_z0_baseline", q=np_["q0"], L=np_["L0"]))
        for dname, d in np_["directions"].items():
            if d["kind"] == "insertion":
                sequence.append(dict(kind="hold", node=name, label=f"{name}_{dname}_plus", q=np_["q0"], L=np_["L0"] + d["step"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_return1", q=np_["q0"], L=np_["L0"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_{dname}_minus", q=np_["q0"], L=np_["L0"] - d["step"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_return2", q=np_["q0"], L=np_["L0"]))
            else:
                dq = d["v"] * d["step"]
                sequence.append(dict(kind="hold", node=name, label=f"{name}_{dname}_plus", q=np_["q0"] + dq, L=np_["L0"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_return1", q=np_["q0"], L=np_["L0"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_{dname}_minus", q=np_["q0"] - dq, L=np_["L0"]))
                sequence.append(dict(kind="hold", node=name, label=f"{name}_return2", q=np_["q0"], L=np_["L0"]))
        mix = np_["mixed"]
        dq_mix = sum(v * step for (_, v, step) in mix["components"])
        sequence.append(dict(kind="hold", node=name, label=f"{name}_mixed_heldout", q=np_["q0"] + dq_mix, L=np_["L0"]))
        sequence.append(dict(kind="hold", node=name, label=f"{name}_return_final", q=np_["q0"], L=np_["L0"]))

    # final transit back to the plan's own start (via waypoints, same reason
    # as inter-node transitions -- a single direct move_j back to ref_idx=0
    # from a late node is not guaranteed safe either).
    for wp_idx in transit_waypoints(_prev_ref_idx, 0):
        sequence.append(dict(kind="transit", node="return_to_start", label=f"transit_to_start_ref{wp_idx}",
                              q=state_reference[wp_idx, :6].copy(), L=float(state_reference[wp_idx, 6])))
    sequence.append(dict(kind="hold", node="return_to_start", label="plan_start_final", q=q0_plan, L=L0_plan))

    print(f"\n[stop-perturb] {len(sequence)} hold-points planned across {len(node_names)} node(s)")
    print("[stop-perturb] verifying every consecutive leg's joint-space path is safe...")
    prev_q = q0_plan
    all_safe = True
    for i, step in enumerate(sequence):
        clear, worst_gap, worst_z = path_clears(
            prev_q, step["q"], lumen_C=lumen_C_arr, excl_radius_m=exclusion_live_m, z_bounds=z_bounds,
        )
        status = "OK" if clear else "UNSAFE"
        if not clear:
            all_safe = False
        print(f"  [{i:3d}] {step['label']:28s} worst_gap={worst_gap*1e3:6.1f}mm "
              f"worst_z_violation={'-' if worst_z is None else f'{worst_z*1e3:.1f}mm'}  {status}")
        step["_leg_clear"] = clear
        step["_leg_worst_gap_m"] = worst_gap
        prev_q = step["q"]

    if not all_safe:
        raise RuntimeError(
            "refusing to run: at least one planned leg is unsafe -- see UNSAFE rows above. "
            "No hardware motion attempted."
        )
    print("[stop-perturb] all legs verified safe.\n")

    if args.dry_run:
        print("[stop-perturb] --dry-run: stopping here, no hardware touched.")
        return

    # --- live execution ---
    from proper_research.hardware.ur_rtde_robot import URRTDERobot
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig

    if not args.skip_preflight:
        common.check_camera_healthy(expected_insertion_m=L0_plan, insertion_tol_mm=3.0)
        common.reset_to_plan_initial_safe(
            q0_plan, magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=lumen_C_arr, magnet_exclusion_radius_m=exclusion_live_m,
            magnet_z_bounds_m=z_bounds,
        )
        common.check_robot_safe()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_path = out_dir / f"stop_perturb_{stamp}.json"
    (out_dir / f"stop_perturb_{stamp}_nodes.json").write_text(json.dumps(
        {name: dict(ref_idx=v["ref_idx"], q0=v["q0"].tolist(), L0=v["L0"], gap0_mm=v["gap0_m"] * 1e3)
         for name, v in node_plan.items()}, indent=2))

    scfg = common._raised_stream_stream_config()
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/stop_perturb_camera.png",
                     roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
                     pivot_hint=tuple(scfg.pivot_hint_px)),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: L0_plan, frame_processor=mapper,
    )
    camera.start()
    adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))
    adv.start()
    time.sleep(2.5)

    robot = URRTDERobot(common.ROBOT_IP, frequency=125.0)
    robot.connect()

    def ready():
        try:
            return robot.get_safety_mode() in (1, 2) and not robot.is_protective_stopped()
        except Exception:
            return False

    def robust(fn, *fn_args, attempts=3, **fn_kwargs):
        """Call an RTDE method, reconnecting and retrying on any exception --
        the RTDE link has been observed to drop asynchronously ("End of
        file") mid-campaign (caught live 2026-09-24, after the `early`
        node completed cleanly), unrelated to robot safety state. A single
        unhandled drop previously crashed the whole script; this makes a
        transient link drop recoverable instead of fatal."""
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                last_exc = e
                print(f"    [robust] attempt {attempt}/{attempts} failed: {e!r} -- reconnecting...")
                try:
                    robot.reconnect()
                except Exception as reconnect_exc:
                    print(f"    [robust] reconnect failed: {reconnect_exc!r}")
                time.sleep(1.0)
        raise last_exc

    def set_insertion(target_L_m, tol_mm=0.7, max_iters=25):
        for _ in range(max_iters):
            tips = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 4.0 and len(tips) < 15:
                est, _age = camera.latest(0.5)
                if est is not None:
                    tips.append(np.asarray(est.tip_position_m))
                time.sleep(0.1)
            if not tips:
                print("  [insertion] no camera reading; aborting insertion set")
                return False
            pivot = np.asarray(scfg.T_robot_beam_pose6[:3])
            L = float(np.linalg.norm(np.mean(tips, axis=0) - pivot))
            err_mm = 1e3 * (target_L_m - L)
            if abs(err_mm) < tol_mm:
                return True
            adv.submit_distance(float(np.clip(err_mm, -4.0, 4.0)))
            t1 = time.monotonic()
            while time.monotonic() - t1 < 20:
                fb = adv.feedback()
                if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
                    break
                time.sleep(0.1)
            time.sleep(0.4)
        print("  [insertion] did not converge within max_iters")
        return False

    def measure_transient():
        samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < TRANSIENT_DURATION_S:
            est, _age = camera.latest(0.5)
            if est is not None:
                samples.append((time.monotonic() - t0, np.asarray(est.tip_position_m).tolist()))
            time.sleep(TRANSIENT_PERIOD_S)
        return samples

    def predict_models(q6, L_m):
        z7 = np.concatenate([np.asarray(q6, dtype=float), [float(L_m)]])
        out = {}
        for contact in (True, False):
            try:
                f = adapters[contact].forward_output(z7, commit=False)[:3]
                J = adapters[contact].continuous_output_jacobian(z7)[:3, :]
                out["contact" if contact else "no_contact"] = dict(f_pred_m=f.tolist(), J=J.tolist())
            except Exception as e:
                out["contact" if contact else "no_contact"] = dict(error=str(e))
        return out

    records = []
    prev_q_live = np.array(robust(robot.get_joints))
    try:
        for i, step in enumerate(sequence):
            print(f"[{i+1}/{len(sequence)}] {step['label']} ...")
            q_target = step["q"]
            L_target = step["L"]

            if not ready():
                robot.reconnect()
                time.sleep(0.5)
            moved_ok = True
            try:
                if np.max(np.abs(q_target - prev_q_live)) > 1e-9:
                    robust(robot.move_j, list(q_target), speed=JOINT_MOVE_SPEED, acceleration=JOINT_MOVE_ACCEL)
            except Exception as e:
                print(f"  move failed after retries: {e!r}")
                moved_ok = False
            if moved_ok:
                q_meas = np.array(robust(robot.get_joints))
                if np.max(np.abs(q_meas - q_target)) > 0.02:
                    print(f"  WARNING: not at target (max joint offset "
                          f"{np.max(np.abs(q_meas-q_target)):.4f} rad)")
                prev_q_live = q_meas.copy()

                if step["kind"] == "transit":
                    # transit waypoints only need to be passed through
                    # quickly and safely -- no settle/transient/insertion-
                    # closed-loop-set/model-prediction needed, they aren't
                    # measurement points. Saves ~6s/waypoint over ~57
                    # waypoints (~6 minutes) with no loss of data quality.
                    ins_ok, transient, preds = True, [], {}
                else:
                    ins_ok = set_insertion(L_target)
                    time.sleep(SETTLE_S)
                    transient = measure_transient()
                    preds = predict_models(q_meas, L_target)
            else:
                q_meas, ins_ok, transient, preds = None, False, [], {}

            rec = dict(
                index=i, label=step["label"], node=step["node"],
                q_target=q_target.tolist(), L_target=L_target,
                q_meas=(None if q_meas is None else q_meas.tolist()),
                move_ok=moved_ok, insertion_ok=ins_ok,
                transient=transient, predictions=preds,
                t_wall=datetime.now(timezone.utc).isoformat(),
            )
            records.append(rec)
            data_path.write_text(json.dumps(records))
    finally:
        adv.stop()
        camera.stop()
        robot.close()

    data_path.write_text(json.dumps(records, indent=1))
    print(f"\n[stop-perturb] saved {len(records)} records -> {data_path}")


if __name__ == "__main__":
    main()
