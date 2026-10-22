import csv
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from robot_class import URRtde
from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit
# ----------------------------
# Quaternion helpers (your code)
# ----------------------------
def quat_wxyz_normalize(qwxyz):
    q = np.asarray(qwxyz, float).copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    return q / n

def quat_wxyz_to_rotvec(qwxyz):
    qw, qx, qy, qz = quat_wxyz_normalize(qwxyz)
    r = Rot.from_quat([qx, qy, qz, qw])  # scipy uses xyzw
    return r.as_rotvec()

def pose8row_to_pose6_ur(row):
    x = float(row["p_x"])
    y = float(row["p_y"])
    z = float(row["p_z"])
    q = np.array([
        float(row["p_qw"]), float(row["p_qx"]),
        float(row["p_qy"]), float(row["p_qz"])
    ], float)
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([x, y, z, rvec[0], rvec[1], rvec[2]], float)

# ----------------------------
# Safety checks
# ----------------------------
def within_workspace(pose6, xyz_min=None, xyz_max=None):
    xyz = np.asarray(pose6[:3], float)
    if xyz_min is not None and np.any(xyz < np.asarray(xyz_min, float)):
        return False
    if xyz_max is not None and np.any(xyz > np.asarray(xyz_max, float)):
        return False
    return True

def max_step_ok(prev_pose6, next_pose6, max_trans_m=np.inf, max_rot_rad=np.inf):
    if prev_pose6 is None:
        return True
    dp = np.linalg.norm(np.asarray(next_pose6[:3]) - np.asarray(prev_pose6[:3]))
    dr = np.linalg.norm(np.asarray(next_pose6[3:6]) - np.asarray(prev_pose6[3:6]))
    return (dp <= max_trans_m) and (dr <= max_rot_rad)

# ----------------------------
# Advancer helper
# ----------------------------
def dl_row_to_mm(row, mpc_dt_s):
    """
    CSV stores u0_dL (likely m/s). Convert per-step command to mm.
    ΔL_mm = u0_dL * dt * 1000
    """
    u0_dL = float(row["u0_dL"])
    return 1000.0 * u0_dL * float(mpc_dt_s)

# ----------------------------
# Main replay
# ----------------------------
def replay_csv(
    csv_path: str,
    robot_ip: str,
    dry_run: bool = True,

    # row filtering
    every_n: int = 1,
    k_start: int | None = None,
    k_end: int | None = None,

    # timing
    sleep_s: float = 0.05,
    mpc_dt_s: float = 0.01,     # <-- set to the dt used when log was recorded

    # robot pose offsets
    z_offset: float = 0.25,     # meters added before sending to UR

    # workspace bounds
    xyz_min=(0.20, -1.50, -0.30),
    xyz_max=(1.20, +1.50, +1.50),

    # max jump bounds
    max_trans_m=np.inf,
    max_rot_rad=np.inf,

    # UR moveL params
    use_moveL_params: bool = False,
    v: float = 0.10,
    a: float = 0.30,

    # Advancer integration
    use_advancer: bool = False,
    advancer_port: str = "/dev/ttyACM0",
    advancer_baud: int = 115200,
    advancer_delay_us: int = 20,
    advancer_min_cmd_mm: float = 0.166,   # one step worth (matches your AdvancerUnit)
):
    # Import here so dry-run users without serial can still parse CSV
    AdvancerUnit = None
    adv = None
    if use_advancer and not dry_run:
        try:
            # Change this import to wherever your class lives
            from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit  # e.g. proper_research.hardware.advancer import AdvancerUnit
        except Exception as e:
            raise RuntimeError(f"Failed to import AdvancerUnit: {e}")

    # Read CSV rows
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def row_k(r): return int(float(r["k"]))
    rows = sorted(rows, key=row_k)

    if k_start is not None:
        rows = [r for r in rows if row_k(r) >= int(k_start)]
    if k_end is not None:
        rows = [r for r in rows if row_k(r) <= int(k_end)]
    if every_n > 1:
        rows = rows[::every_n]

    if len(rows) == 0:
        raise RuntimeError("No rows selected from CSV. Check k_start/k_end/every_n.")

    # Pre-convert robot poses + dL mm commands
    replay_items = []
    for r in rows:
        k = row_k(r)
        pose6 = pose8row_to_pose6_ur(r)
        dL_mm = dl_row_to_mm(r, mpc_dt_s)
        replay_items.append((k, pose6, dL_mm, r))

    # Connect robot + advancer
    robo = URRtde(robot_ip)
    if use_advancer and not dry_run:
        adv = AdvancerUnit(port=advancer_port, baudrate=advancer_baud)

    try:
        robo.get_pose()

        prev = None
        dl_residual_mm = 0.0   # accumulate sub-step dL so tiny commands aren't lost

        for (k, pose6, dL_mm, row) in replay_items:
            # UR pose with z offset
            pose6_send = pose6.copy()
            pose6_send[2] += float(z_offset)

            # Safety checks on actual sent UR pose
            if not within_workspace(pose6_send, xyz_min=xyz_min, xyz_max=xyz_max):
                raise RuntimeError(f"Pose out of workspace at k={k}: {pose6_send}")

            if not max_step_ok(prev, pose6_send, max_trans_m=max_trans_m, max_rot_rad=max_rot_rad):
                raise RuntimeError(
                    f"Pose jump too large at k={k}.\n"
                    f"  prev={prev}\n"
                    f"  next={pose6_send}\n"
                    f"  (tune max_trans_m/max_rot_rad or downsample more)"
                )

            # Accumulate dL and quantize to step-able distance
            dl_residual_mm += float(dL_mm)

            adv_cmd_mm = 0.0
            if abs(dl_residual_mm) >= float(advancer_min_cmd_mm):
                # send only integer multiples of one-step-equivalent distance
                n_quanta = int(np.trunc(dl_residual_mm / float(advancer_min_cmd_mm)))
                adv_cmd_mm = n_quanta * float(advancer_min_cmd_mm)
                dl_residual_mm -= adv_cmd_mm

            print(
                f"[REPLAY] k={k:06d} "
                f"UR_send={pose6_send.tolist()} "
                f"dL_row={dL_mm:+.4f} mm "
                f"adv_cmd={adv_cmd_mm:+.4f} mm "
                f"residual={dl_residual_mm:+.4f} mm"
            )

            # Alternate: advancer first, then robot
            if not dry_run:
                # 1) Advancer command first
                if use_advancer and (adv is not None) and (abs(adv_cmd_mm) > 0.0):
                    if adv_cmd_mm > 0:
                        adv.forward(abs(adv_cmd_mm), delay_us=advancer_delay_us)
                    else:
                        adv.backward(abs(adv_cmd_mm), delay_us=advancer_delay_us)

                # 2) UR command
                if use_moveL_params:
                    robo.moveL(pose6_send.tolist(), v=v, a=a)
                else:
                    robo.moveL(pose6_send.tolist())

            prev = pose6_send.copy()
            time.sleep(float(sleep_s))

        print("Finished replay.")

    finally:
        # Shut down both devices safely
        try:
            if adv is not None:
                adv.shutdown()
        except Exception as e:
            print(f"[WARN] Advancer shutdown error: {e}")

        try:
            robo.shutdown()
        except Exception:
            pass

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    ROBOT_IP = "192.168.56.101"
    CSV_PATH = "/home/jack/Proper-Research/mpc_run_90_beam_tan/log.csv"

    replay_csv(
        csv_path=CSV_PATH,
        robot_ip=ROBOT_IP,
        dry_run=False,             # True first!
        every_n=1,
        k_start=None,
        k_end=None,
        sleep_s=0.05,
        mpc_dt_s=0.01,             # IMPORTANT: match your MPC logging dt
        z_offset=0.25,

        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),

        max_trans_m=0.5,
        max_rot_rad=0.5,

        use_moveL_params=False,
        v=0.10,
        a=0.30,

        use_advancer=True,         # enable advancer
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166, # one step worth
    )