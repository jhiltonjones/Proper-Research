import csv
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from robot_class import URRtde

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
    """
    row must have keys:
      p_x,p_y,p_z,p_qw,p_qx,p_qy,p_qz

    returns UR pose6: [x,y,z, rx,ry,rz]
    """
    x = float(row["p_x"])
    y = float(row["p_y"])
    z = float(row["p_z"])
    q = np.array([float(row["p_qw"]), float(row["p_qx"]), float(row["p_qy"]), float(row["p_qz"])], float)
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([x, y, z, rvec[0], rvec[1], rvec[2]], float)

# ----------------------------
# Safety checks (tune these!)
# ----------------------------
def within_workspace(pose6, xyz_min=None, xyz_max=None):
    xyz = np.asarray(pose6[:3], float)
    if xyz_min is not None and np.any(xyz < np.asarray(xyz_min, float)):
        return False
    if xyz_max is not None and np.any(xyz > np.asarray(xyz_max, float)):
        return False
    return True

def max_step_ok(prev_pose6, next_pose6, max_trans_m=0.02, max_rot_rad=0.35):
    """
    Reject big jumps:
      max_trans_m: max cartesian jump per command
      max_rot_rad : max rotation-vector delta norm per command
    """
    if prev_pose6 is None:
        return True
    dp = np.linalg.norm(np.asarray(next_pose6[:3]) - np.asarray(prev_pose6[:3]))
    dr = np.linalg.norm(np.asarray(next_pose6[3:6]) - np.asarray(prev_pose6[3:6]))
    return (dp <= max_trans_m) and (dr <= max_rot_rad)

# ----------------------------
# Main replay
# ----------------------------
def replay_csv(
    csv_path: str,
    robot_ip: str,
    dry_run: bool = True,
    every_n: int = 1,
    k_start: int | None = None,
    k_end: int | None = None,
    sleep_s: float = 0.05,
    # workspace bounds (edit to match your setup)
    xyz_min=(0.20, -1.50, -0.30),
    xyz_max=(1.20, +1.50, +1.50),
    # max jump bounds (edit)
    max_trans_m=0.01,
    max_rot_rad=0.25,
    # optional: moveL tuning (only used if your URRtde.moveL supports these args)
    use_moveL_params: bool = False,
    v: float = 0.10,   # m/s
    a: float = 0.30,   # m/s^2
):
    # Read CSV rows
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Filter and downsample
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

    # Convert to pose6 list
    pose6_list = []
    for r in rows:
        pose6 = pose8row_to_pose6_ur(r)
        pose6_list.append((row_k(r), pose6))

    # Connect robot
    robo = URRtde(robot_ip)
    try:
        robo.get_pose()

        prev = None
        for (k, pose6) in pose6_list:
            # Safety checks
            if not within_workspace(pose6, xyz_min=xyz_min, xyz_max=xyz_max):
                raise RuntimeError(f"Pose out of workspace at k={k}: {pose6}")

            if not max_step_ok(prev, pose6, max_trans_m=max_trans_m, max_rot_rad=max_rot_rad):
                raise RuntimeError(
                    f"Pose jump too large at k={k}.\n"
                    f"  prev={prev}\n"
                    f"  next={pose6}\n"
                    f"  (tune max_trans_m/max_rot_rad or downsample more)"
                )

            # Print and/or send
            print(f"[REPLAY] k={k:06d} pose6={pose6.tolist()}")

            if not dry_run:
                if use_moveL_params:
                    # only if your URRtde.moveL signature supports it
                    robo.moveL(pose6.tolist(), v=v, a=a)
                else:
                    robo.moveL(pose6.tolist())

            prev = pose6.copy()
            time.sleep(float(sleep_s))

        print("Finished replay.")
    finally:
        try:
            robo.shutdown()
        except Exception:
            pass

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    ROBOT_IP = "192.168.56.101"
    CSV_PATH = "mpc_run_027/log.csv"   # <-- change to your actual path

    replay_csv(
        csv_path=CSV_PATH,
        robot_ip=ROBOT_IP,
        dry_run=True,       # set False to actually move
        every_n=1,
        k_start=None,
        k_end=None,
        sleep_s=0.05,
        # tighten these to your cell:
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.25,
        use_moveL_params=False,
    )