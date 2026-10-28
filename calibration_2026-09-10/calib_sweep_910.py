"""2026-09-10 beam+magnet calibration sweep (N52 100x100 magnet, new pose).
Y arc + limited Z + radius, move_j via IK, camera + model + Jacobian per pose.
Magnet stays >= 28 cm from the beam base and within [-4, +7] cm of its start z.
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
Z_LO, Z_HI = BEAM_BASE[2] - 0.04, BEAM_BASE[2] + 0.07
INS = 0.045

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
DH = urik.corrected_dh_from_config(urik.CONFIG)

# flange->magnet in flange frame (from initial_conditions)
from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6
TCP_FRAME_OFF = np.array(TCP_TO_MAGNET_POSE6[:3])

def tcp_for_magnet(m, ori):
    R = Rot.from_rotvec(ori).as_matrix()
    return np.concatenate([np.asarray(m) - R @ TCP_FRAME_OFF, ori])

def ik(tcp6, seed):
    r = urik.inverse_kinematics_dls(T_R_target=urik.pose6_to_T(tcp6), q_seed_rad=tuple(seed),
                                    dh=DH, T_F_target=None, cfg=urik.CONFIG)
    return np.array(r.q_rad), r.converged, r.final_position_error_m

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=27.0, beam_axial_axis_R=(0, 0, 1.0), beam_plane_normal_axis_R=(-1.0, 0, 0)))
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
    return (T_R_B.inverse().apply_points(np.asarray(out).reshape(-1)[:3]),
            T_R_B.inverse().apply_points(T_R_M.translation), J)

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
scfg = StateStreamConfig(exposure=27.0, beam_axial_axis_R=(0, 0, 1.0), beam_plane_normal_axis_R=(-1.0, 0, 0))
cam = CameraSource(CameraConfig(cam_index=0, exposure=27.0, image_filename="/dev/shm/calib910.png",
    roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
    pivot_hint=tuple(scfg.pivot_hint_px)), pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
    robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
    insertion_length_getter=lambda: INS, frame_processor=mapper)
cam.start()

def cam_tipB(n=20, tmo=7.0):
    tips = []; t0 = time.monotonic()
    while time.monotonic() - t0 < tmo and len(tips) < n:
        est, _ = cam.latest(0.5)
        if est is not None:
            tips.append(np.asarray(est.tip_position_m))
        time.sleep(0.1)
    if not tips:
        return None, 0
    return T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0)), len(tips)

# --- poses (all in R; magnet target, orientation = reference) ---
poses = []
# 1. Y arc around the beam base at r = 0.28, z = base
for phi in (0, 8, 16, 24, -8, -16, -24, -32, 32):
    p = np.deg2rad(phi)
    m = BEAM_BASE + 0.28 * np.array([-np.cos(p), np.sin(p), 0.0])
    poses.append((m, "y", f"y_phi{phi}"))
# 2. Radius sweep straight back (-X), y=base, z=base
for r in (0.30, 0.34, 0.40):
    poses.append((BEAM_BASE + np.array([-r, 0.0, 0.0]), "r", f"back_r{int(r * 100)}"))
# 3. Z sweep at r=0.30, y=base
for dz in (0.03, 0.06, -0.03):
    m = BEAM_BASE + np.array([-np.sqrt(max(0.30**2 - dz**2, 0.28**2)), 0.0, dz])
    poses.append((m, "z", f"z_dz{int(dz * 100):+d}"))
# 4. combined Y at larger radius
for phi in (18, -18):
    p = np.deg2rad(phi)
    m = BEAM_BASE + 0.34 * np.array([-np.cos(p), np.sin(p), 0.0])
    poses.append((m, "y", f"y34_phi{phi}"))

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
    print(f"REF cam {np.round(1e3 * tbc, 2)} model {np.round(1e3 * tbm, 2)} magB {np.round(1e3 * mB, 0)} ({nf}fr)")
    DATA.append({"label": "ref", "kind": "ref", "tcp": list(p_ref), "q": list(q_ref), "ins": INS,
                 "cam_tipB": list(tbc), "cam_frames": nf, "mod_tipB": list(tbm), "magnetB": list(mB), "J": J.tolist()})
    for m, kind, lab in poses:
        d = np.linalg.norm(m - BEAM_BASE)
        if d < MIN_R:
            print(f"  {lab}: r {d:.3f} < min, skip"); continue
        if not (Z_LO - 1e-4 <= m[2] <= Z_HI + 1e-4):
            print(f"  {lab}: z {m[2]:.3f} out of [{Z_LO:.3f},{Z_HI:.3f}], skip"); continue
        tcp = tcp_for_magnet(m, REF_ORI)
        qt, conv, perr = ik(tcp, seed)
        if not conv or perr > 2e-3 or np.max(np.abs(qt - seed)) > 0.8:
            print(f"  {lab}: IK reject (conv={conv} err={perr * 1e3:.1f}mm dq={np.max(np.abs(qt - seed)):.2f})"); continue
        for att in (1, 2):
            if not ready():
                try: r.reconnect()
                except Exception: pass
                time.sleep(0.5)
            try:
                r.move_j(list(qt), speed=0.4, acceleration=0.4); break
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
        _, _, T_R_M = rfmv.source_transform_from_tcp(pc, rfmv.CONFIG)
        if np.linalg.norm(np.asarray(T_R_M.translation) - BEAM_BASE) < MIN_R:
            print(f"  {lab}: magnet too close, skip"); continue
        tbc, nf = cam_tipB(); tbm, mB, J = model_at(pc, INS)
        ic = None if tbc is None else np.degrees(np.arctan2(tbc[1], tbc[0]))
        im = np.degrees(np.arctan2(tbm[1], tbm[0])); om = np.degrees(np.arctan2(tbm[2], np.hypot(tbm[0], tbm[1])))
        print(f"  {lab:12s} magB[{1e3 * mB[0]:.0f},{1e3 * mB[1]:.0f},{1e3 * mB[2]:.0f}] cam_in {ic if ic is None else round(ic, 2)} mod_in {im:.2f} mod_out {om:.2f} ({nf}fr)")
        DATA.append({"label": lab, "kind": kind, "tcp": list(pc), "q": list(qc), "ins": INS,
                     "cam_tipB": None if tbc is None else list(tbc), "cam_frames": nf,
                     "mod_tipB": list(tbm), "magnetB": list(mB), "J": J.tolist()})
        json.dump(DATA, open("/home/jack/.claude/jobs/3710eca5/tmp/calib910_data.json", "w"))
    if ready():
        try:
            r.move_j(list(q_ref), speed=0.4, acceleration=0.4); print("returned to ref")
        except Exception as e:
            print("return fail", e)
finally:
    cam.stop(); r.close()
json.dump(DATA, open("/home/jack/.claude/jobs/3710eca5/tmp/calib910_data.json", "w"), indent=1)
print(f"\nsaved {len(DATA)} poses")
