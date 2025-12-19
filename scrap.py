from robot_class import URRtde
from proper_research.robot.transformations import get_point
ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [0.7985173296917242, -0.538880495640068, 0.4502643054124873, -1.9773907947518439, 2.4202883191517617, -0.016116851148178644]

if __name__ =='__main__':
    robo = URRtde(ROBOT_IP)
    try:
        # current_joint = get_point(0,0)
        robo.get_pose()
        robo.get_joints()

        # current_joint[5]+=0.1
        # robo.moveJ(current_joint)
        # new_pose = get_point(0,0)
        # robo.moveL(new_pose)
        robo.go_home_joint()
        print("Finished")
    finally:
        robo.shutdown()
