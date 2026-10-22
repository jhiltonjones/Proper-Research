from robot_class import URRtde
from proper_research.robot.transformations import get_point
import numpy as np
start_point = np.array([
    0.5072749291492671, -0.6894409342863428, 0.2634155953600154+0.05, 1.6796703000712034, -2.639274597381263, -0.023560877572101912
], float)
pivot_point = start_point.copy()
pivot_point[0] +=0.17

ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [0.42801887872839117, -0.7373104561642874, 0.35782223424787696, -3.078136979153704, 0.5757200626927219, 0.04565482188889512]
joints = [-0.8294642607318323, -1.9672757587828578, -1.730027437210083, -1.04366888225589, 1.5631502866744995, -2.029626194630758]
REF_JOINTS = np.array(
    [-0.85634357, -1.94584002, -1.76176286, -1.03319450, 1.56234264, -2.05640871]
)
start_point = np.array(
    [
        0.245575,
        -0.670028,
        0.32,
        -3.07793295,
        0.57537270,
        0.04503358,
    ],
    dtype=float,
)
Sleeping_joints = [5.127925760461949e-06, -1.570810934106344, 1.889864076787262e-05, -1.5707822610205149, -6.500874654591371e-06, -1.699129213506012e-05]
if __name__ == "__main__":
    robo = URRtde(
        ROBOT_IP,
        external_control=False,
    )
    try:
        curr_ppose_add = robo.get_pose()
        print(curr_ppose_add)
        # curr_ppose_add[0] -= 0.5
        # robo.moveL(curr_ppose_add)
        # TCP_TARGET[0]-=.1
        robo.moveL(TCP_TARGET)
    
        # robo.go_home_joint()
        # joints = robo.get_joints()
        # joints[5] +=np.deg2rad(180)
        # robo.moveJ(joints)
        print(f"Joints are {joints}")
        # print(f"New pose is {start_point}")
        # new_pos = get_point(0,0, start_point=TCP_TARGET, pivot_point=pivot_point)
        # robo.moveL(TCP_TARGET)
        # print(f"Get point gives {new_pos}")

        # print(f"Pose is {curr_ppose_add}")
        print("Finished")
    finally:
        print("done")
        robo.shutdown()
