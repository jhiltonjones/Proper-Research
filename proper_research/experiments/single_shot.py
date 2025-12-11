from proper_research.parameters import ROBOT_IP
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
robo = URRtde(ROBOT_IP)

try:
    robo.get_joints()
    robo.get_pose()
    pose = get_point(0,0)
    robo.moveL(pose)
    # robo.go_home_joint()

finally:
    robo.shutdown()