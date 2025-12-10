import rtde_control
import rtde_receive 


class URRtde:

    def __init__(self, ip: str):
        self.ip = ip
        print(f"Connecting to Robo at the ip: {ip}")
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        print("Connected with Robo")

    def shutdown(self):
        try:
            self.rtde_c.stopScript()
        except Exception as e:
            print(f"Shutdown warning {e}")
        try:
            self.rtde_c.disconnect()
        except Exception as e:
            print(f"Disconnect warning for the connect module {e}")
        try: 
            self.rtde_r.disconnect()
        except Exception as e:
            print(f"Disconnect warning for the receive module {e}")
        print("Robo is resting")
    
    def get_joints(self):
        q = self.rtde_r.getActualQ()
        print(f"State Joints: {q}")
        return q
    
    def get_pose(self):
        pose = self.rtde_r.getActualTCPPose()
        print(f"TCP pose: {pose}")
        return pose
    
    def moveJ(self, q, speed = 0.7, accel = 0.7, blend_radius = 0.0):
        self.rtde_c.moveJ(q, speed, accel, blend_radius)
    
    def moveL(self, pose, speed=0.25, accel = 0.5):
        self.rtde_c.moveL(pose, speed, accel)

    def go_home(self, target_tcp= [0.528583395185381, -0.41938710566998094, 0.4502478513259566, -1.9773976132360036, 2.420306911735209, -0.016056316733630456]):
        self.moveL(target_tcp, speed=0.1, accel=0.2)
    def go_home_joint(self, target_joint = [-0.4124181906329554, -1.581008096734518, -1.974020004272461, -1.1402791303447266, 1.5827715396881104, -0.21076661745180303]):
        self.moveJ(target_joint)