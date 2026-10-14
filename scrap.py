from robot_class import URRtde
from proper_research.robot.transformations import get_point
import numpy as np
start_point = np.array([
    0.5072749291492671, -0.6894409342863428, 0.2634155953600154+0.05, 1.6796703000712034, -2.639274597381263, -0.023560877572101912
], float)
pivot_point = start_point.copy()
pivot_point[0] +=0.17
tcp_pos = [0.3149930273459505, -0.7199239834356613, 0.3903562217858709, -3.0702151519205936, 0.665727583892539, 5.022306444258972e-05]
tcp_pos2 = [0.364314, -0.584458, 0.390373, -1.949126, 2.463654, -0.000492]
ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [0.32985741784198486, -0.603852381621729, 0.37707877387966343, -2.6412838195695794, 1.589562361413843, 0.07744879771517037]
REF_JOINTS = np.array(
    [-0.6536853949176233, -1.8291098080077113, -1.782264232635498, -1.1741101902774354, 1.6159265041351318, -1.1000617186175745]
)
ref_joints2 = [
    -0.6261470953570765,
    -1.7615391216673792,
    -1.8847217559814453,
    -1.0656255048564454,
    1.5728814601898193,
    -1.091161076222555
  ]
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
        robo.go_home_joint()
        robo.moveL(tcp_pos)

        # TCP_TARGET[0]-=.1
        # robo.moveL(TCP_TARGET)
    

        # joints[5] +=np.deg2rad(180)
        # robo.moveJ(ref_joints2)
        joints = robo.get_joints()

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
