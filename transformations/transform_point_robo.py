from robot_class_trans import URRtde
from transformations import get_point

ROBOT_IP = "192.168.56.101"
base_pos =  [0.7985173296917242, -0.538880495640068, 0.4502643054124873, -1.9773907947518439, 2.4202883191517617, -0.016116851148178644]

robo = URRtde(ROBOT_IP)
try:
    current_joint = robo.get_joints()
    current_pose = robo.get_pose()
    robo.moveL(base_pos)
    input("Press Enter")
    robo.go_home_joint()
finally:
    robo.shutdown()