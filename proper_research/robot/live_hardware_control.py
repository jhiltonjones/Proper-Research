from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_to_rotvec

from robot_class import URRtde
def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

def p8_to_ur_pose6_and_L(p8):
    p7 = pose8_quat_to_pose7_rotvec(p8)
    return p7[:6].copy(), float(p7[6])

def u0_to_advancer_mm(u0, dt):
    return float(u0[6]) * float(dt) * 1000.0

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
class LiveHardwareController:
    def __init__(
        self,
        robot_ip,
        dry_run=True,
        use_advancer=True,
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=0.0,
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    ):
        self.robot_ip = robot_ip
        self.dry_run = bool(dry_run)
        self.use_advancer = bool(use_advancer)
        self.advancer_delay_us = int(advancer_delay_us)
        self.advancer_min_cmd_mm = float(advancer_min_cmd_mm)
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.max_trans_m = float(max_trans_m)
        self.max_rot_rad = float(max_rot_rad)
        self.z_offset = float(z_offset)
        self.use_moveL_params = bool(use_moveL_params)
        self.v = float(v)
        self.a = float(a)

        self.adv = None
        self.prev_pose6 = None
        self.dl_residual_mm = 0.0

        if self.use_advancer and not self.dry_run:
            self.adv = AdvancerUnit(port=advancer_port, baudrate=advancer_baud)
    def get_robot_pose_once(self):
        robo = URRtde(self.robot_ip)
        try:
            return robo.get_pose()
        finally:
            robo.shutdown()
    def send_step(self, p_now, u0, dt):
        ur_pose6_next, _ = p8_to_ur_pose6_and_L(p_now)
        ur_pose6_send = ur_pose6_next.copy()
        ur_pose6_send[2] += self.z_offset

        if not within_workspace(ur_pose6_send, self.xyz_min, self.xyz_max):
            raise RuntimeError(f"UR pose out of workspace: {ur_pose6_send}")

        dL_mm = u0_to_advancer_mm(u0, dt)
        self.dl_residual_mm += dL_mm

        adv_cmd_mm = 0.0
        if abs(self.dl_residual_mm) >= self.advancer_min_cmd_mm:
            n_quanta = int(np.trunc(self.dl_residual_mm / self.advancer_min_cmd_mm))
            adv_cmd_mm = n_quanta * self.advancer_min_cmd_mm
            self.dl_residual_mm -= adv_cmd_mm

        print(
            "[LIVE CMD]",
            "UR_send =", ur_pose6_send.tolist(),
            f"dL_step = {dL_mm:+.4f} mm",
            f"adv_cmd = {adv_cmd_mm:+.4f} mm",
            f"residual = {self.dl_residual_mm:+.4f} mm",
        )

        if not self.dry_run:
            if self.use_advancer and self.adv is not None and abs(adv_cmd_mm) > 0.0:
                if adv_cmd_mm > 0:
                    self.adv.forward(abs(adv_cmd_mm), delay_us=self.advancer_delay_us)
                else:
                    self.adv.backward(abs(adv_cmd_mm), delay_us=self.advancer_delay_us)

            robo = URRtde(self.robot_ip)
            try:
                if self.use_moveL_params:
                    robo.moveL(ur_pose6_send.tolist(), speed=self.v, accel=self.a)
                else:
                    robo.moveL(ur_pose6_send.tolist())
            finally:
                robo.shutdown()

        self.prev_pose6 = ur_pose6_send.copy()

    def shutdown(self):
        if self.adv is not None:
            self.adv.shutdown()