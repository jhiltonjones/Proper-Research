import rtde_control
import rtde_receive


class MinimalURRobot:
    def __init__(self, robot_ip, frequency=125.0):
        self.control = None
        self.receive = None

        try:
            # Connects and uploads the ur_rtde control script.
            self.control = rtde_control.RTDEControlInterface(
                robot_ip, frequency
            )

            # Separate connection for reading robot information.
            self.receive = rtde_receive.RTDEReceiveInterface(
                robot_ip, frequency
            )
        except Exception:
            self.close()
            raise

    def is_connected(self):
        return (
            self.control is not None
            and self.receive is not None
            and self.control.isConnected()
            and self.receive.isConnected()
        )

    def check_ready(self):
        """Check the basic robot state before movement."""
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        robot_mode = self.receive.getRobotMode()
        safety_mode = self.receive.getSafetyMode()

        if robot_mode != 7:
            raise RuntimeError(
                f"Robot mode is {robot_mode}, expected 7 (RUNNING). "
                "Power the arm and release the brakes."
            )

        if safety_mode not in (1, 2):
            raise RuntimeError(
                f"Safety mode is {safety_mode}; expected "
                "1 (NORMAL) or 2 (REDUCED)."
            )

        if self.receive.isProtectiveStopped():
            raise RuntimeError("Robot is protective-stopped")

    @staticmethod
    def _six_values(values, name):
        values = [float(value) for value in values]

        if len(values) != 6:
            raise ValueError(
                f"{name} must contain exactly six values"
            )

        return values

    def get_joints(self):
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        return list(self.receive.getActualQ())

    def get_tcp_pose(self):
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        return list(self.receive.getActualTCPPose())

    def move_j(self, joints, speed=0.10, acceleration=0.20):
        """
        Joint-space movement.

        joints: six joint angles in radians
        speed: joint speed in rad/s
        acceleration: joint acceleration in rad/s^2
        """
        target = self._six_values(joints, "joints")
        self.check_ready()

        return self.control.moveJ(
            target,
            speed,
            acceleration,
        )

    def move_l(self, pose, speed=0.05, acceleration=0.10):
        """
        Straight-line TCP movement.

        pose: [x, y, z, rx, ry, rz]
        x, y, z: metres
        rx, ry, rz: rotation-vector radians
        speed: m/s
        acceleration: m/s^2
        """
        target = self._six_values(pose, "pose")
        self.check_ready()

        return self.control.moveL(
            target,
            speed,
            acceleration,
        )

    def close(self):
        if self.control is not None:
            try:
                if self.control.isConnected():
                    self.control.stopScript()
            except Exception:
                pass

            try:
                self.control.disconnect()
            except Exception:
                pass

            self.control = None

        if self.receive is not None:
            try:
                self.receive.disconnect()
            except Exception:
                pass

            self.receive = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    ROBOT_IP = "192.168.10.10"

    with MinimalURRobot(ROBOT_IP) as robot:
        print("Connected:", robot.is_connected())
        print("Joints [rad]:", robot.get_joints())
        print("TCP pose [m, rad]:", robot.get_tcp_pose())

