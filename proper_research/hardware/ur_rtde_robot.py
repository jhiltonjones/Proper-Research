""".

Install with:
    python -m pip install ur_rtde

"""

from __future__ import annotations

import math
import threading
import time
from typing import Iterable, List, Optional, Sequence

import rtde_control
import rtde_receive


class URConnectionError(RuntimeError):
    """The application cannot communicate with both RTDE interfaces."""


class URNotReadyError(RuntimeError):
    """The robot is connected but is not ready to accept a motion command."""


class URRTDERobot:
    """Control and monitor one Universal Robots controller using ur_rtde.

    The default frequency of 125 Hz is suitable for normal moveJ/moveL calls
    and monitoring on an ordinary Linux kernel. This class deliberately does
    not retry a motion automatically after a connection failure.
    """

    ROBOT_MODE_RUNNING = 7
    SAFETY_MODE_NORMAL = 1
    SAFETY_MODE_REDUCED = 2

    def __init__(self, ip: str, frequency: float = 125.0) -> None:
        if not ip or not isinstance(ip, str):
            raise ValueError("ip must be the robot's IPv4 address or hostname")
        if not math.isfinite(frequency) or frequency <= 0.0 or frequency > 500.0:
            raise ValueError("frequency must be greater than 0 and at most 500 Hz")

        self.ip = ip
        self.frequency = float(frequency)
        self._control: Optional[rtde_control.RTDEControlInterface] = None
        self._receive: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._lock = threading.RLock()
        self.connect()

    @staticmethod
    def _vector6(values: Iterable[float], name: str):
        result = [float(value) for value in values]
        if len(result) != 6:
            raise ValueError(f"{name} must contain exactly six values")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} contains a non-finite value")
        return result

    @staticmethod
    def _positive(value: float, name: str):
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be a finite number greater than zero")
        return result

    def connect(self):
        """Connect both interfaces and upload the default ur_rtde script."""
        with self._lock:
            if self.is_connected():
                return

            self._disconnect_unlocked(stop_script=False)
            try:
                self._control = rtde_control.RTDEControlInterface(
                    self.ip, self.frequency
                )
                self._receive = rtde_receive.RTDEReceiveInterface(
                    self.ip, self.frequency
                )
                if not self.is_connected():
                    raise URConnectionError("one or both RTDE interfaces failed")
            except Exception as exc:
                self._disconnect_unlocked(stop_script=True)
                raise URConnectionError(
                    f"Could not connect to UR controller at {self.ip}"
                ) from exc

    def reconnect(self, attempts: int = 3, delay: float = 1.0):
        """Explicitly rebuild the session without resuming any previous move."""
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        if delay < 0.0:
            raise ValueError("delay cannot be negative")

        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                self.close()
                if attempt:
                    time.sleep(delay)
                self.connect()
                return
            except Exception as exc:
                last_error = exc

        raise URConnectionError(
            f"Could not reconnect to {self.ip} after {attempts} attempts"
        ) from last_error

    def is_connected(self):
        control_ok = self._control is not None and self._control.isConnected()
        receive_ok = self._receive is not None and self._receive.isConnected()
        return bool(control_ok and receive_ok)

    def _require_connected_unlocked(self):
        if not self.is_connected():
            raise URConnectionError(
                "RTDE is disconnected. Call reconnect() and verify robot state "
                "before sending another motion command."
            )

    def _require_motion_ready_unlocked(self):
        self._require_connected_unlocked()
        assert self._receive is not None

        robot_mode = self._receive.getRobotMode()
        safety_mode = self._receive.getSafetyMode()

        if robot_mode != self.ROBOT_MODE_RUNNING:
            raise URNotReadyError(
                f"Robot mode is {robot_mode}, expected RUNNING "
                f"({self.ROBOT_MODE_RUNNING}). Power the arm and release brakes."
            )
        if safety_mode not in (
            self.SAFETY_MODE_NORMAL,
            self.SAFETY_MODE_REDUCED,
        ):
            raise URNotReadyError(
                f"Safety mode is {safety_mode}; motion is allowed only in "
                "NORMAL or REDUCED mode by this class."
            )
        if self._receive.isProtectiveStopped():
            raise URNotReadyError("Robot is protective-stopped")

    def get_tcp_pose(self):
        """Return [x, y, z, rx, ry, rz]: metres and rotation-vector radians."""
        with self._lock:
            self._require_connected_unlocked()
            assert self._receive is not None
            return list(self._receive.getActualTCPPose())

    def get_joints(self):
        """Return the six actual joint angles in radians."""
        with self._lock:
            self._require_connected_unlocked()
            assert self._receive is not None
            return list(self._receive.getActualQ())

    def get_robot_mode(self):
        with self._lock:
            self._require_connected_unlocked()
            assert self._receive is not None
            return int(self._receive.getRobotMode())

    def get_safety_mode(self):
        with self._lock:
            self._require_connected_unlocked()
            assert self._receive is not None
            return int(self._receive.getSafetyMode())

    def is_protective_stopped(self):
        with self._lock:
            self._require_connected_unlocked()
            assert self._receive is not None
            return bool(self._receive.isProtectiveStopped())

    def move_l(
        self,
        pose: Sequence[float],
        speed: float = 0.10,
        acceleration: float = 0.20,
        asynchronous: bool = False,
    ):
        """Move the active TCP linearly to a base-frame Cartesian pose."""
        target = self._vector6(pose, "pose")
        speed = self._positive(speed, "speed")
        acceleration = self._positive(acceleration, "acceleration")

        with self._lock:
            self._require_motion_ready_unlocked()
            assert self._control is not None
            return bool(
                self._control.moveL(
                    target, speed, acceleration, bool(asynchronous)
                )
            )

    def move_j(
        self,
        joints: Sequence[float],
        speed: float = 0.25,
        acceleration: float = 0.30,
        asynchronous: bool = False,
    ):
        """Move to six joint angles in radians.

        """
        target = self._vector6(joints, "joints")
        speed = self._positive(speed, "speed")
        acceleration = self._positive(acceleration, "acceleration")

        with self._lock:
            self._require_motion_ready_unlocked()
            assert self._control is not None
            return bool(
                self._control.moveJ(
                    target, speed, acceleration, bool(asynchronous)
                )
            )

    def go_home_joint(
        self,
        home_joints: Sequence[float],
        speed: float = 0.25,
        acceleration: float = 0.30,
    ):
        """Move to an application-specific, previously validated joint target."""
        return self.move_j(home_joints, speed, acceleration, asynchronous=False)

    def _disconnect_unlocked(self, stop_script: bool):
        control, receive = self._control, self._receive
        self._control = None
        self._receive = None

        if control is not None:
            if stop_script:
                try:
                    if control.isConnected():
                        control.stopScript()
                except Exception:
                    pass
            try:
                control.disconnect()
            except Exception:
                pass

        if receive is not None:
            try:
                receive.disconnect()
            except Exception:
                pass

    def close(self):
        """Stop this class's uploaded control script and disconnect."""
        with self._lock:
            self._disconnect_unlocked(stop_script=True)

    def __enter__(self) -> "URRTDERobot":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    # Replace this with the robot's IP address.
    ROBOT_IP = "192.168.56.101"

    with URRTDERobot(ROBOT_IP) as robot:
        print("Connected:", robot.is_connected())
        print("Robot mode:", robot.get_robot_mode())
        print("Safety mode:", robot.get_safety_mode())
        print("Joints [rad]:", robot.get_joints())
        print("TCP pose [m, rad]:", robot.get_tcp_pose())
