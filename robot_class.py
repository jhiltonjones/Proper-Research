import time
# import rtde_control
# import rtde_receive


class URRtde:
    def __init__(self, ip: str, frequency: float = 125.0):
        self.ip = ip
        self.frequency = frequency
        self.rtde_c = None
        self.rtde_r = None
        self.connect()

    def connect(self):
        print(f"Connecting to robot at {self.ip} @ {self.frequency} Hz")
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip, self.frequency)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip, self.frequency)
        print("Connected")

    def is_connected(self):
        c_ok = self.rtde_c is not None and self.rtde_c.isConnected()
        r_ok = self.rtde_r is not None and self.rtde_r.isConnected()
        return c_ok and r_ok

    def ensure_connected(self, retries: int = 3, delay: float = 0.5):
        if self.is_connected():
            return True

        print("RTDE connection lost. Attempting recovery...")

        for attempt in range(1, retries + 1):
            try:
                if self.rtde_c is not None and not self.rtde_c.isConnected():
                    ok = self.rtde_c.reconnect()
                    print(f"Control reconnect attempt {attempt}: {ok}")

                if self.rtde_r is not None and not self.rtde_r.isConnected():
                    ok = self.rtde_r.reconnect()
                    print(f"Receive reconnect attempt {attempt}: {ok}")

                if self.rtde_c is not None and self.rtde_c.isConnected():
                    try:
                        self.rtde_c.reuploadScript()
                    except Exception as e:
                        print(f"reuploadScript warning: {e}")

                if self.is_connected():
                    print("RTDE recovered")
                    return True
            except Exception as e:
                print(f"Reconnect attempt {attempt} failed: {e}")

            time.sleep(delay)

        print("Reconnect() failed; rebuilding interfaces from scratch...")
        self.disconnect()
        time.sleep(delay)
        self.connect()
        return self.is_connected()

    def disconnect(self):
        try:
            if self.rtde_c is not None:
                self.rtde_c.disconnect()
        except Exception as e:
            print(f"Control disconnect warning: {e}")

        try:
            if self.rtde_r is not None:
                self.rtde_r.disconnect()
        except Exception as e:
            print(f"Receive disconnect warning: {e}")

    def shutdown(self):
        # Only call this when your application is ending.
        try:
            if self.rtde_c is not None:
                self.rtde_c.stopScript()
        except Exception as e:
            print(f"stopScript warning: {e}")

        self.disconnect()
        print("Robot session closed")

    def get_pose(self):
        self.ensure_connected()
        pose = self.rtde_r.getActualTCPPose()
        return pose

    def get_joints(self):
        self.ensure_connected()
        q = self.rtde_r.getActualQ()
        return q

    def moveL(self, pose, speed=0.25, accel=0.5):
        self.ensure_connected()
        return self.rtde_c.moveL(pose, speed, accel)

    def moveJ(self, q, speed=0.7, accel=0.7, blend_radius=0.0):
        self.ensure_connected()
        return self.rtde_c.moveJ(q, speed, accel, blend_radius)

    def robot_mode(self):
        self.ensure_connected()
        return self.rtde_r.getRobotMode()

    def safety_mode(self):
        self.ensure_connected()
        return self.rtde_r.getSafetyMode()

    def protective_stopped(self):
        self.ensure_connected()
        return self.rtde_r.isProtectiveStopped()

    def go_home(self, target_tcp= [0.528583395185381, -0.41938710566998094, 0.4502478513259566, -1.9773976132360036, 2.420306911735209, -0.016056316733630456]):
        self.moveL(target_tcp, speed=0.1, accel=0.2)
    def go_home_joint(self, target_joint = [-0.4124181906329554, -1.581008096734518, -1.974020004272461, -1.1402791303447266, 1.5827715396881104, -0.21076661745180303]):
        self.moveJ(target_joint)