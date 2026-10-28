"""2026-09-10 sweep C: INSERTION-LENGTH sweep, 0.018 - 0.036 m.

Magnet parked at a fixed off-axis position (moderate in-plane bend).
Advancer retracts the beam 33 -> 18 mm, then advances 18 -> 36 mm, then
back to ~33 mm.  Camera length + bend and model prediction (model
insertion set to the camera-measured length) recorded at each stop.

Constraint: total inserted length stays within [0.018, 0.036] m.
"""
import numpy as np, time, json, io, contextlib
from scipy.spatial.transform import Rotation as Rot
np.set_printoptions(precision=5, suppress=True)

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

BEAM_BASE = np.array([0.525575, -0.670028, -0.016567])
REF_ORI = np.array([-3.07806404, 0.57586908, 0.04569585])
MIN_R = 0.278
LEN_LO, LEN_HI = 0.018, 0.036
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/sweep_insertion_data.json"

# magnet parked off-axis: r = 0.30, phi = 18 deg  (moderate bend, not saturated)
PHI = np.deg2rad(18.0)
M_PARK = BEAM_BASE + 0.30 * np.array([-np.cos(PHI), np.sin(PHI), 0.0])

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
DH = urik.corrected_dh_from_config(urik.CONFIG)
from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6
TCP_FRAME_OFF = np.array(TCP_TO_MAGNET_POSE6[:3])

def tcp_for_magnet(m, ori):
    Rm = Rot.from_rotvec(ori).as_matrix()
    return np.concatenate([np.asarray(m) - Rm @ TCP_FRAME_OFF, ori])

def ik(tcp6, seed):
    r = urik.inverse_kinematics_dls(T_R_target=urik.pose6_to_T(tcp6), q_seed_rad=tuple(seed),
                                    dh=DH, T_F_target=None, cfg=urik.CONFIG)
    return np.array(r.q_rad), r.converged, r.final_position_error_m

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
    adapter, _ = rfmv.build_forward_model_in_shared_frame(
        experiment_cfg=ev2.CONFIG, T_R_B=mapper.T_R_B, lumen=lumen,
        model_base_rotation_R=mbr, dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
T_R_B = mapper.T_R_B

def model_at(tcp6, ins):
    _, _, T_R_M = rfmv.source_transform_from_tcp(np.asarray(tcp6), rfmv.CONFIG)
    p8 = rfmv.transform_to_p8(T_R_M, float(ins))
    with quiet():
        p7, out = adapter.commit_nominal(p8)
        J = np.asarray(adapter.raw_model.jacobian_tip_actuation_tangent(p7)).reshape(3, 7)
    tipB = T_R_B.inverse().apply_points(np.asarray(out).reshape(-1)[:3])
    magB = T_R_B.inverse().apply_points(T_R_M.translation)
    return tipB, magB, J

INS_GUESS = {"v": 0.0335}
from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
scfg = StateStreamConfig(exposure=29.0)
cam = CameraSource(CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/sweepINS.png",
    roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
    pivot_hint=tuple(scfg.pivot_hint_px)), pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
    robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
    insertion_length_getter=lambda: INS_GUESS["v"], frame_processor=mapper)
cam.start()

def cam_tipB(n=30, tmo=9.0):
    tips = []; t0 = time.monotonic()
    while time.monotonic() - t0 < tmo and len(tips) < n:
        est, _ = cam.latest(0.5)
        if est is not None:
            tips.append(np.asarray(est.tip_position_m))
        time.sleep(0.1)
    if not tips:
        return None, 0
    return T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0)), len(tips)

from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))

def advance_mm(dist_mm, timeout=25.0):
    """Blocking: submit a signed travel and wait for the writer to drain."""
    adv.submit_distance(float(dist_mm))
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        fb = adv.feedback()
        if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
            break
        time.sleep(0.1)
    return adv.feedback()

DATA = []
from proper_research.hardware.ur_rtde_robot import URRTDERobot
r = URRTDERobot("192.168.56.101", frequency=125.0)

def ready():
    try:
        return r.get_safety_mode() in (1, 2) and not r.is_protective_stopped() and r.get_robot_mode() == 7
    except Exception:
        return False

def record(tag, phase):
    tbc, nf = cam_tipB()
    if tbc is None:
        print(f"  {tag}: NO CAMERA"); return None
    cam_len_m = float(np.hypot(tbc[0], tbc[1])) / 1.0  # tip is in metres already? -> B frame metres
    cam_len_m = float(np.linalg.norm(tbc[:2]))
    INS_GUESS["v"] = cam_len_m
    pc = np.array(r.get_tcp_pose())
    tbm, mB, J = model_at(pc, cam_len_m)
    ic = np.degrees(np.arctan2(tbc[1], tbc[0])); oc = np.degrees(np.arctan2(tbc[2], np.hypot(tbc[0], tbc[1])))
    im = np.degrees(np.arctan2(tbm[1], tbm[0])); om = np.degrees(np.arctan2(tbm[2], np.hypot(tbm[0], tbm[1])))
    fb = adv.feedback()
    print(f"  {tag:14s} [{phase}] cam_len {1e3*cam_len_m:5.1f}mm cam_in {ic:+6.2f} cam_out {oc:+5.2f} | "
          f"mod_len {1e3*np.linalg.norm(tbm[:2]):5.1f} mod_in {im:+6.2f} mod_out {om:+5.2f} | adv_disp {fb.dispatched_distance_mm:+.2f} ({nf}fr)")
    d = {"tag": tag, "phase": phase, "tcp": list(pc), "cam_tipB": list(tbc), "cam_len_m": cam_len_m,
         "cam_frames": nf, "mod_tipB": list(tbm), "magnetB": list(mB), "J": J.tolist(),
         "adv_dispatched_mm": fb.dispatched_distance_mm, "adv_steps": fb.steps_dispatched_total}
    DATA.append(d)
    json.dump(DATA, open(OUT, "w"))
    return cam_len_m

try:
    adv.start(); time.sleep(2.5)
    q_ref = np.array(r.get_joints()); seed = q_ref.copy()

    # park the magnet off-axis
    tcp = tcp_for_magnet(M_PARK, REF_ORI)
    qt, conv, perr = ik(tcp, seed)
    assert conv and perr < 2e-3 and np.max(np.abs(qt - seed)) < 0.9, (conv, perr, np.max(np.abs(qt - seed)))
    if not ready():
        r.reconnect(); time.sleep(0.5)
    r.move_j(list(qt), speed=0.3, acceleration=0.3); time.sleep(0.5)
    seed = np.array(r.get_joints())

    L0 = record("park", "start")
    if L0 is None or not (0.012 <= L0 <= 0.045):
        raise SystemExit(f"unexpected start length {L0}")

    # ---- retract toward 18 mm in ~3 mm steps ----
    cur = L0
    while cur * 1e3 > 19.0:
        step = -min(3.0, (cur - LEN_LO) * 1e3 + 0.5)
        if cur * 1e3 + step < LEN_LO * 1e3 - 0.3:
            step = LEN_LO * 1e3 - cur * 1e3
        advance_mm(step); time.sleep(0.4)
        nl = record(f"ret_{int(round(cur*1e3+step))}", "retract")
        if nl is None:
            break
        if nl >= cur - 0.0005:
            print("  camera length not decreasing -> stop retract"); break
        cur = nl

    # ---- advance toward 36 mm in ~3 mm steps ----
    while cur * 1e3 < 35.0:
        step = min(3.0, LEN_HI * 1e3 - cur * 1e3 + 0.3)
        advance_mm(step); time.sleep(0.4)
        nl = record(f"adv_{int(round(cur*1e3+step))}", "advance")
        if nl is None:
            break
        if nl <= cur + 0.0005:
            print("  camera length not increasing -> stop advance"); break
        cur = nl

    # ---- back to ~33.5 mm ----
    back = (0.0335 - cur) * 1e3
    if abs(back) > 0.4:
        advance_mm(back); time.sleep(0.4)
        record("return_335", "return")

    if ready():
        r.move_j(list(q_ref), speed=0.3, acceleration=0.3); print("returned to ref joints")
finally:
    try: adv.stop()
    except Exception: pass
    cam.stop(); r.close()
json.dump(DATA, open(OUT, "w"), indent=1)
print(f"\nsaved {len(DATA)} stops -> {OUT}")
