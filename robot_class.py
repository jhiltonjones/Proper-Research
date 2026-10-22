import time

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


class URRtde:
    def __init__(
        self,
        robot_ip,
        external_control=False,
        external_control_port=50002,
        frequency=500.0,
    ):
        self.robot_ip = robot_ip
        self.external_control = external_control
        self.external_control_port = int(external_control_port)
        self.frequency = float(frequency)

        self.rtde_c = None
        self.rtde_r = None

        self.connect()

    def connect(self):
        print(
            f"Connecting to robot at {self.robot_ip} "
            f"@ {self.frequency} Hz"
        )

        if self.external_control:
            flags = (
                RTDEControl.FLAG_USE_EXT_UR_CAP
                | RTDEControl.FLAG_VERBOSE
            )

            print("Using External Control URCap.")
            print(
                f"PC should be listening on port "
                f"{self.external_control_port}."
            )
            print(
                "Start Python first, then press Play on the "
                "External Control program on the teach pendant."
            )

            self.rtde_c = RTDEControl(
                self.robot_ip,
                self.frequency,
                flags,
                self.external_control_port,
            )

        else:
            print("Using normal Remote Control mode.")

            self.rtde_c = RTDEControl(
                self.robot_ip,
                self.frequency,
            )

        self.rtde_r = RTDEReceive(
            self.robot_ip,
            self.frequency,
        )

        print("Connected")

    def is_connected(self):
        c_ok = (
            self.rtde_c is not None
            and self.rtde_c.isConnected()
        )
        r_ok = (
            self.rtde_r is not None
            and self.rtde_r.isConnected()
        )
        return c_ok and r_ok

    def ensure_connected(self, retries=3, delay=0.5):
        if self.is_connected():
            return True

        print("RTDE connection lost.")

        #
        # IMPORTANT:
        # External-Control mode should not blindly call
        # reuploadScript(), because the control program is supplied
        # through the External Control URCap.
        #
        if self.external_control:
            print(
                "External-Control connection was lost. "
                "A fresh External Control session may require "
                "restarting Python and pressing Play on the pendant."
            )
            return False

        print("Attempting Remote-mode recovery...")

        for attempt in range(1, retries + 1):
            try:
                if (
                    self.rtde_c is not None
                    and not self.rtde_c.isConnected()
                ):
                    ok = self.rtde_c.reconnect()
                    print(
                        f"Control reconnect attempt "
                        f"{attempt}: {ok}"
                    )

                if (
                    self.rtde_r is not None
                    and not self.rtde_r.isConnected()
                ):
                    ok = self.rtde_r.reconnect()
                    print(
                        f"Receive reconnect attempt "
                        f"{attempt}: {ok}"
                    )

                if (
                    self.rtde_c is not None
                    and self.rtde_c.isConnected()
                ):
                    try:
                        self.rtde_c.reuploadScript()
                    except Exception as exc:
                        print(
                            f"reuploadScript warning: {exc}"
                        )

                if self.is_connected():
                    print("RTDE recovered")
                    return True

            except Exception as exc:
                print(
                    f"Reconnect attempt {attempt} "
                    f"failed: {exc}"
                )

            time.sleep(delay)

        print("Remote reconnect failed.")
        return False

    def disconnect(self):
        try:
            if self.rtde_c is not None:
                self.rtde_c.disconnect()
        except Exception as exc:
            print(f"Control disconnect warning: {exc}")

        try:
            if self.rtde_r is not None:
                self.rtde_r.disconnect()
        except Exception as exc:
            print(f"Receive disconnect warning: {exc}")

    def shutdown(self):
        try:
            if self.rtde_c is not None:
                self.rtde_c.stopScript()
        except Exception as exc:
            print(f"stopScript warning: {exc}")

        self.disconnect()
        print("Robot session closed")

    def get_pose(self):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")
        return self.rtde_r.getActualTCPPose()

    def get_joints(self):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")
        return self.rtde_r.getActualQ()

    def get_jacobian(self, q=None):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")

        if q is None:
            return self.rtde_c.getJacobian()

        return self.rtde_c.getJacobian(list(q))

    def speed_j(self, qd, acceleration=0.5, time=0.0):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")

        return self.rtde_c.speedJ(
            list(qd),
            float(acceleration),
            float(time),
        )

    def speed_l(self, xd, acceleration=0.25, time=0.0):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")

        return self.rtde_c.speedL(
            list(xd),
            float(acceleration),
            float(time),
        )

    def speed_stop(self, acceleration=2.0):
        try:
            if not self.ensure_connected():
                return False

            return self.rtde_c.speedStop(
                float(acceleration)
            )

        except Exception as exc:
            print(f"speed_stop warning: {exc}")
            return False

    def set_watchdog(self, min_frequency_hz=10.0):
        try:
            if not self.ensure_connected():
                return False

            return self.rtde_c.setWatchdog(
                float(min_frequency_hz)
            )

        except Exception as exc:
            print(f"set_watchdog warning: {exc}")
            return False

    def moveL(self, pose, speed=0.25, accel=0.5):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")

        return self.rtde_c.moveL(
            pose,
            speed,
            accel,
        )

    def moveJ(
        self,
        q,
        speed=0.7,
        accel=0.7,
        blend_radius=0.0,
    ):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")

        return self.rtde_c.moveJ(
            q,
            speed,
            accel,
            blend_radius,
        )

    def robot_mode(self):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")
        return self.rtde_r.getRobotMode()

    def safety_mode(self):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")
        return self.rtde_r.getSafetyMode()

    def protective_stopped(self):
        if not self.ensure_connected():
            raise RuntimeError("Robot is not connected")
        return self.rtde_r.isProtectiveStopped()

    def go_home(
        self,
        target_tcp=None,
    ):
        if target_tcp is None:
            target_tcp = [
                0.528583395185381,
                -0.41938710566998094,
                0.4502478513259566,
                -1.9773976132360036,
                2.420306911735209,
                -0.016056316733630456,
            ]

        return self.moveL(
            target_tcp,
            speed=0.1,
            accel=0.2,
        )

    def go_home_joint(
        self,
        target_joint=None,
    ):
        if target_joint is None:
            target_joint = [
                -0.4124181906329554,
                -1.581008096734518,
                -1.974020004272461,
                -1.1402791303447266,
                1.5827715396881104,
                -0.21076661745180303,
            ]

        return self.moveJ(target_joint)