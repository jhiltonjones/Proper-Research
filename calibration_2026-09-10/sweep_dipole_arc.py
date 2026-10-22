"""2026-09-10 sweep B: xy-plane ARC around the beam base + DIPOLE ROTATION.

- Part A: magnet on an arc around the beam base in the world x-y plane,
  radius >= 28 cm, z = base z (well inside +-7 cm).  Orientation fixed at
  the reference => the dipole stays along world -X while the magnet swings.
- Part B: magnet held at the reference (coaxial) position; the TCP is
  re-oriented so the dipole rotates about world Z (in-plane observable)
  and world Y (camera-blind, model only).
- Part C: same dipole rotation, but with the magnet parked off-axis so
  there is a baseline bend to perturb.

move_j via IK, reconnect-on-failure.  Camera + model tip + full Jacobian
recorded per pose.  Fixed insertion INS.
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
R_REF = Rot.from_rotvec(REF_ORI)
MIN_R = 0.278
Z_LO, Z_HI = BEAM_BASE[2] - 0.07, BEAM_BASE[2] + 0.07
INS = 0.034
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/sweep_dipole_arc_data.json"

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
    # dipole direction in R
    m_body = np.asarray(rfmv.CONFIG.source_dipole_unit_in_magnet_body, float)
    dip_R = T_R_M.rotation @ (m_body / np.linalg.norm(m_body))
    dipB = T_R_B.rotation.T @ dip_R
    return tipB, magB, dipB, J

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
scfg = StateStreamConfig(exposure=29.0)
cam = CameraSource(CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/sweepDA.png",
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

# ---------------- pose list ----------------
poses = []  # (magnet_xyz_R, ori_rotvec, kind, label)

# Part A: xy-plane arc around the beam base
for r in (0.285, 0.32, 0.38):
    for phi in (0, 10, 20, 30, 40, -10, -20, -30, -40):
        p = np.deg2rad(phi)
        m = BEAM_BASE + r * np.array([-np.cos(p), np.sin(p), 0.0])
        poses.append((m, REF_ORI, "arc", f"arc_r{int(r*100)}_phi{phi:+d}"))

M_COAX = BEAM_BASE + np.array([-0.30, 0.0, 0.0])
M_OFF = BEAM_BASE + 0.30 * np.array([-np.cos(np.deg2rad(22)), np.sin(np.deg2rad(22)), 0.0])

# Part B: dipole rotation about world Z (in-plane) and world Y (blind), magnet coaxial
for az in (0, 6, 12, 18, 24, -6, -12, -18, -24):
    o = (Rot.from_rotvec([0, 0, np.deg2rad(az)]) * R_REF).as_rotvec()
    poses.append((M_COAX, o, "dipZ", f"dipZ_coax_{az:+d}"))
for ay in (10, 20, -10, -20):
    o = (Rot.from_rotvec([0, np.deg2rad(ay), 0]) * R_REF).as_rotvec()
    poses.append((M_COAX, o, "dipY", f"dipY_coax_{ay:+d}"))

# Part C: dipole rotation about world Z with a baseline off-axis bend
for az in (0, 10, 20, -10, -20):
    o = (Rot.from_rotvec([0, 0, np.deg2rad(az)]) * R_REF).as_rotvec()
    poses.append((M_OFF, o, "dipZoff", f"dipZ_off_{az:+d}"))

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
    tbc, nf = cam_tipB(); tbm, mB, dB, J = model_at(p_ref, INS)
    print(f"REF cam {np.round(1e3*tbc,2)} model {np.round(1e3*tbm,2)} magB {np.round(1e3*mB,0)} dipB {np.round(dB,3)} ({nf}fr)")
    DATA.append({"label": "ref", "kind": "ref", "tcp": list(p_ref), "q": list(q_ref), "ins": INS,
                 "cam_tipB": list(tbc), "cam_frames": nf, "mod_tipB": list(tbm),
                 "magnetB": list(mB), "dipoleB": list(dB), "J": J.tolist()})
    for m, ori, kind, lab in poses:
        d = np.linalg.norm(m - BEAM_BASE)
        if d < MIN_R:
            print(f"  {lab}: r {d:.3f} < min, skip"); continue
        if not (Z_LO - 1e-4 <= m[2] <= Z_HI + 1e-4):
            print(f"  {lab}: z out, skip"); continue
        tcp = tcp_for_magnet(m, ori)
        qt, conv, perr = ik(tcp, seed)
        if not conv or perr > 2e-3 or np.max(np.abs(qt - seed)) > 0.9:
            print(f"  {lab}: IK reject (conv={conv} err={perr*1e3:.1f}mm dq={np.max(np.abs(qt-seed)):.2f})"); continue
        for att in (1, 2):
            if not ready():
                try: r.reconnect()
                except Exception: pass
                time.sleep(0.5)
            try:
                r.move_j(list(qt), speed=0.35, acceleration=0.35); break
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
        tbc, nf = cam_tipB(); tbm, mB, dB, J = model_at(pc, INS)
        ic = None if tbc is None else np.degrees(np.arctan2(tbc[1], tbc[0]))
        oc = None if tbc is None else np.degrees(np.arctan2(tbc[2], np.hypot(tbc[0], tbc[1])))
        im = np.degrees(np.arctan2(tbm[1], tbm[0])); om = np.degrees(np.arctan2(tbm[2], np.hypot(tbm[0], tbm[1])))
        print(f"  {lab:18s} magB[{1e3*mB[0]:.0f},{1e3*mB[1]:.0f},{1e3*mB[2]:.0f}] dipB[{dB[0]:.2f},{dB[1]:.2f},{dB[2]:.2f}] "
              f"cam_in {None if ic is None else round(ic,2)} cam_out {None if oc is None else round(oc,2)} "
              f"mod_in {im:.2f} mod_out {om:.2f} ({nf}fr)")
        DATA.append({"label": lab, "kind": kind, "tcp": list(pc), "q": list(qc), "ins": INS,
                     "cam_tipB": None if tbc is None else list(tbc), "cam_frames": nf,
                     "mod_tipB": list(tbm), "magnetB": list(mB), "dipoleB": list(dB), "J": J.tolist()})
        json.dump(DATA, open(OUT, "w"))
    if ready():
        try:
            r.move_j(list(q_ref), speed=0.35, acceleration=0.35); print("returned to ref")
        except Exception as e:
            print("return fail", e)
finally:
    cam.stop(); r.close()
json.dump(DATA, open(OUT, "w"), indent=1)
print(f"\nsaved {len(DATA)} poses -> {OUT}")
