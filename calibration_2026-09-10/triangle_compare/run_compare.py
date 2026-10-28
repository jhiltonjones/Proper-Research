"""Live: run the triangle plan under the inverse-Jacobian controller and the
MPC-LTI controller, back to back, and save both tip_trajectory.csv paths.

Between runs: move the robot back to the reference joints and reset the beam
to 30 mm with the advancer (camera-measured).
"""
import sys, time, json, io, contextlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot

PLAN_DIR = "/home/jack/Proper-Research/plans/triangle_8mm_2026-09-10/time_parameterized_configuration_path"
TARGET_LEN_M = 0.030
REF_JOINTS = np.array([-0.85634357, -1.94584002, -1.76176286, -1.03319450, 1.56234264, -2.05640871])

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


def move_to_ref():
    from proper_research.hardware.ur_rtde_robot import URRTDERobot
    r = URRTDERobot("192.168.56.101", frequency=125.0)
    try:
        for att in (1, 2, 3):
            try:
                if r.get_safety_mode() not in (1, 2) or r.is_protective_stopped():
                    r.reconnect(); time.sleep(0.5)
                r.move_j(list(REF_JOINTS), speed=0.35, acceleration=0.35)
                q = np.array(r.get_joints())
                if np.max(np.abs(q - REF_JOINTS)) < 0.02:
                    print(f"[reset] at reference joints (dq={np.max(np.abs(q-REF_JOINTS)):.4f})")
                    return True
            except Exception as e:
                print(f"[reset] move att{att}: {e!r}")
                try: r.reconnect()
                except Exception: pass
                time.sleep(0.6)
        return False
    finally:
        r.close()


def set_beam_length(target_m, tol_mm=0.6, max_iter=8):
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware import robotics_frame_measurement_validation as rfmv
    from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig

    scfg = StateStreamConfig(exposure=29.0)
    with quiet():
        mapper = NewFrameTipMapper(scfg)
    T_R_B = mapper.T_R_B
    cam = CameraSource(CameraConfig(cam_index=0, exposure=29.0, image_filename="/dev/shm/cmp_len.png",
        roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
        pivot_hint=tuple(scfg.pivot_hint_px)), pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: target_m, frame_processor=mapper)
    cam.start()
    adv = AdvancerSink(AdvancerSinkConfig(port="/dev/ttyACM0", dry_run=False, delay_us=40))
    adv.start(); time.sleep(2.5)

    def meas(n=25, tmo=8.0):
        tips = []; t0 = time.monotonic()
        while time.monotonic() - t0 < tmo and len(tips) < n:
            est, _ = cam.latest(0.5)
            if est is not None:
                tips.append(np.asarray(est.tip_position_m))
            time.sleep(0.1)
        if not tips:
            return None
        t = T_R_B.inverse().apply_points(np.mean(np.vstack(tips), axis=0))
        return float(np.linalg.norm(t[:2]))

    try:
        for it in range(max_iter):
            L = meas()
            if L is None:
                print("[beam] no camera length"); return None
            err_mm = 1e3 * (target_m - L)
            print(f"[beam] iter {it}: length {1e3*L:.1f} mm  (target {1e3*target_m:.0f}, err {err_mm:+.1f} mm)")
            if abs(err_mm) <= tol_mm:
                return L
            step = float(np.clip(err_mm, -4.0, 4.0))
            adv.submit_distance(step)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 20:
                fb = adv.feedback()
                if abs(fb.residual_mm) < 0.166 and fb.commands_in_flight == 0:
                    break
                time.sleep(0.1)
            time.sleep(0.5)
        return meas()
    finally:
        try: adv.stop()
        except Exception: pass
        cam.stop()


def run_controller(kind, run_name):
    import proper_research.hardware.online.close_loop_path_follow as pf
    pf.CONFIG.dry_run = False
    pf.CONFIG.reference_source = "plan_dir"
    pf.CONFIG.plan_dir = PLAN_DIR
    pf.CONFIG.control_hz = 10.0
    pf.CONFIG.controller_kind = kind
    pf.CONFIG.control_insertion = True
    pf.CONFIG.advancer_dry_run = False
    pf.CONFIG.initial_insertion_m = TARGET_LEN_M
    pf.CONFIG.max_control_steps = 600
    pf.CONFIG.run_name = run_name
    pf.CONFIG.exposure = 29.0
    if kind in ("mpc_lti", "mpc_ltv_offline"):
        pf.CONFIG.mpc_prediction_horizon = 12
        pf.CONFIG.mpc_freeze_index = 0
    pf.main()
    # newest output dir with this run_name
    import glob, os
    dirs = sorted(glob.glob(f"close_loop_logs/{run_name}_*"), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def main():
    results = {}
    for kind, name in (("naive_inverse_jacobian", "cmp_invjac"), ("mpc_lti", "cmp_mpclti")):
        print(f"\n{'='*70}\n  {kind}\n{'='*70}")
        if not move_to_ref():
            print("[abort] could not reach reference joints"); break
        L = set_beam_length(TARGET_LEN_M)
        if L is None or abs(L - TARGET_LEN_M) > 1.5e-3:
            print(f"[abort] beam length not set (got {L})"); break
        print(f"[ok] beam at {1e3*L:.1f} mm; starting {kind}")
        time.sleep(1.0)
        out = run_controller(kind, name)
        print(f"[done] {kind} -> {out}")
        results[kind] = out
        json.dump(results, open("/home/jack/.claude/jobs/3710eca5/tmp/compare_dirs.json", "w"), indent=1)
    print("\nreset to reference...")
    move_to_ref()
    print("RESULTS:", json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
