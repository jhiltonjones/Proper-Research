from robot_class import URRtde
from proper_research.robot.transformations import get_point
import numpy as np
start_point = np.array([
    0.5072749291492671, -0.6894409342863428, 0.2634155953600154+0.05, 1.6796703000712034, -2.639274597381263, -0.023560877572101912
], float)
pivot_point = start_point.copy()
pivot_point[0] +=0.17

ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [ 0.18195034690054912+0.4, -0.6878669442307535, 0.31278800256366485+0.05, 2.6086861829895063, 1.6393724345761258, -0.024847687830113752]
pivot_point =  [ 0.84813282,
        -0.68127317,
        -0.10000000+0.1,
        float(np.pi),
        1.0e-3,
        1.0e-3,]

Sleeping_joints = [5.127925760461949e-06, -1.570810934106344, 1.889864076787262e-05, -1.5707822610205149, -6.500874654591371e-06, -1.699129213506012e-05]
if __name__ == "__main__":
    robo = URRtde(ROBOT_IP)
    try:

        # robo.go_home_joint()
        joints = robo.get_joints()
        print(f"Joints are {joints}")
        print(f"New pose is {start_point}")
        new_pos = get_point(0,0, start_point=TCP_TARGET, pivot_point=pivot_point)
        robo.moveL(TCP_TARGET)

        # print(f"Get point gives {new_pos}")
        pose = robo.get_pose()
        print(f"Pose is {pose}")
        print("Finished")
    finally:
        print("done")
        robo.shutdown()

