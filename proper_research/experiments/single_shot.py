from proper_research.parameters import ROBOT_IP
from proper_research.robot.ur_rtde import URRtde
robo = URRtde(ROBOT_IP)

try:
    robo.get_joints()
    robo.get_pose()
    robo.go_home_joint()

finally:
    robo.shutdown()