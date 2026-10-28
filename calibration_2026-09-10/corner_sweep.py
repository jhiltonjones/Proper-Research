"""2026-09-10 CORNER sweep: at the triangle-corner insertion (38 mm), move the
source magnet in B.y (world Y) and B.x (world -X, axial distance) and record
where the REAL beam tip lands vs where the online model predicts.  Used to
reverse-engineer the offline-plan beam parameters at the large-deflection
(corner) operating point.

The plan's own corner nodes:
  +base corner : tip (u,v) = (37.6, +3.7) mm,  planner magnet_B = [281, +27, +6]
  -base corner : tip (u,v) = (37.5, -3.7) mm,  planner magnet_B = [282, -30, -7]
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
MIN_R = 0.238
Z_LO, Z_HI = BEAM_BASE[2] - 0.07, BEAM_BASE[2] + 0.07
INS = 0.038
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/corner_sweep_data.json"

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
DH = urik.corrected_dh_from_config(urik.CONFIG)
from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6
TCP_FRAME_OFF = np.array(TCP_TO_MAGNET_POSE6[:3])

def tcp_for_magnet(m, ori=REF_ORI):
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
BX, BY, BZ = (np.asarray(T_R_B.rotation[:, 0]), np.asarray(T_R_B.rotation[:, 1]), np.asarray(T_R_B.rotation[:, 2]))

def magnet_world(bx, by, bz=0.0):
    return BEAM_BASE + bx * BX + by * BY + bz * BZ

def to_B(p_world):
    d = np.asarray(p_world) - BEAM_BASE
    return np.array([d @ BX, d @ BY, d @ BZ])

def model_at(tcp6, ins):
    _, _, T_R_M = rfmv.source_transform_from_tcp(np.asarray(tcp6), rfmv.CONFIG)
    p8 = rfmv.transform_to_p8(T_R_M, float(ins))
    with quiet():
        p7, out = adapter.commit_nominal(p8)
        J = np.asarray(adapter.raw_model.jacobian_tip_actuation_tangent(p7)).reshape(3, 7)
    return (T_R_B.inverse().apply_points(np.asarray(out).reshape(-1)[:3]),
            T_R_B.inverse().apply_points(T_R_M.translation), J)

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
scfg = StateStreamConfig(exposure=29.0)
cam = CameraSource(CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/corner.png",
    roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
    pivot_hint=tuple(scfg.pivot_hint_px)), pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
    robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
    insertion_length_getter=lambda: INS, frame_processor=mapper)
cam.start()

def cam_tipB(n=25, tmo=8.0):
    tips = []; t0 = time.monotonic()
    while time.monotonic() - t0 < tmo and len(tips) < n:
        est, _ = cam.latest(0.5)
        if est is not None:
            tips.append(np.asarray(est.tip_position_m))
        time.sleep(0.1)
    if not tips:
        return None, 0
    return T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0)), len(tips)

# --- set the beam to INS mm first ---
from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))
adv.start(); time.sleep(2.5)
for it in range(10):
    tb, nf = cam_tipB()
    if tb is None:
        print("no camera"); break
    L = float(np.linalg.norm(tb[:2])); err = 1e3 * (INS - L)
    print(f"[beam] {1e3*L:.1f} mm (target {1e3*INS:.0f}, err {err:+.1f})")
    if abs(err) < 0.7:
        break
    adv.submit_distance(float(np.clip(err, -4, 4)))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20:
        fb = adv.feedback()
        if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
            break
        time.sleep(0.1)
    time.sleep(0.4)
adv.stop()

# --- poses (magnet B-coords -> world) ---
poses = []
for by in (0, 12, 24, 36, 48, 60, -12, -24, -36, -48, -60):
    poses.append((magnet_world(0.280, by * 1e-3, 0.0), "by", f"by{by:+d}_bx280"))
for bx in (0.280, 0.265, 0.250):
    poses.append((magnet_world(bx, 0.030, 0.0), "bx", f"by+30_bx{int(bx*1e3)}"))
    poses.append((magnet_world(bx, -0.030, 0.0), "bx", f"by-30_bx{int(bx*1e3)}"))
# fine sweep near the +corner and -corner
for by in (28, 32, 40, 44):
    poses.append((magnet_world(0.280, by * 1e-3, 0.0), "byf", f"by{by:+d}_bx280"))
for by in (-28, -32, -40, -44):
    poses.append((magnet_world(0.280, by * 1e-3, 0.0), "byf", f"by{by:+d}_bx280"))

DATA = []
from proper_research.hardware.ur_rtde_robot import URRTDERobot
r = URRTDERobot("192.168.56.101", frequency=125.0)

def ready():
    try:
        return r.get_safety_mode() in (1, 2) and not r.is_protective_stopped() and r.get_robot_mode() == 7
    except Exception:
        return False

try:
    q_ref = np.array(r.get_joints()); p_ref = np.array(r.get_tcp_pose())
    seed = q_ref.copy()
    tbc, nf = cam_tipB(); tbm, mB, J = model_at(p_ref, INS)
    print(f"REF cam {np.round(1e3*tbc,2)} model {np.round(1e3*tbm,2)} magB {np.round(1e3*mB,0)} ins {INS} ({nf}fr)")
    DATA.append({"label": "ref", "kind": "ref", "tcp": list(p_ref), "ins": INS,
                 "cam_tipB": list(tbc), "cam_frames": nf, "mod_tipB": list(tbm), "magnetB": list(mB), "J": J.tolist()})
    for m, kind, lab in poses:
        d = np.linalg.norm(m - BEAM_BASE)
        if d < MIN_R:
            print(f"  {lab}: r {d:.3f} < {MIN_R}, skip"); continue
        if not (Z_LO - 1e-4 <= m[2] <= Z_HI + 1e-4):
            print(f"  {lab}: z {m[2]:.3f} out, skip"); continue
        tcp = tcp_for_magnet(m)
        qt, conv, perr = ik(tcp, seed)
        if not conv or perr > 2e-3 or np.max(np.abs(qt - seed)) > 1.0:
            print(f"  {lab}: IK reject (conv={conv} err={perr*1e3:.1f} dq={np.max(np.abs(qt-seed)):.2f})"); continue
        for att in (1, 2):
            if not ready():
                try: r.reconnect()
                except Exception: pass
                time.sleep(0.5)
            try:
                r.move_j(list(qt), speed=0.3, acceleration=0.3); break
            except Exception as e:
                print(f"  {lab}: move att{att} {repr(e)[:45]}")
                if att == 2: break
                try: r.reconnect()
                except Exception: pass
                time.sleep(0.6)
        time.sleep(0.4)
        try:
            qc = np.array(r.get_joints()); pc = np.array(r.get_tcp_pose())
        except Exception:
            print("  read fail, abort"); break
        if np.max(np.abs(qc - qt)) > 0.03:
            print(f"  {lab}: not reached"); continue
        seed = qc.copy()
        tbc, nf = cam_tipB(); tbm, mB, J = model_at(pc, INS)
        cv = None if tbc is None else 1e3 * tbc[1]
        mv = 1e3 * tbm[1]
        print(f"  {lab:16s} magB[{1e3*mB[0]:.0f},{1e3*mB[1]:.0f},{1e3*mB[2]:.0f}] "
              f"cam_v {cv if cv is None else round(cv,2)}mm  mod_v {mv:.2f}mm  "
              f"cam_u {None if tbc is None else round(1e3*tbc[0],1)} mod_u {1e3*tbm[0]:.1f} ({nf}fr)")
        DATA.append({"label": lab, "kind": kind, "tcp": list(pc), "ins": INS,
                     "cam_tipB": None if tbc is None else list(tbc), "cam_frames": nf,
                     "mod_tipB": list(tbm), "magnetB": list(mB), "J": J.tolist()})
        json.dump(DATA, open(OUT, "w"))
    if ready():
        try:
            r.move_j(list(q_ref), speed=0.3, acceleration=0.3); print("returned to ref")
        except Exception as e:
            print("return fail", e)
finally:
    cam.stop(); r.close()
json.dump(DATA, open(OUT, "w"), indent=1)
print(f"\nsaved {len(DATA)} poses -> {OUT}")
