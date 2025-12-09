from robot_class import URRtde
ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [0.7985173296917242, -0.538880495640068, 0.4502643054124873, -1.9773907947518439, 2.4202883191517617, -0.016116851148178644]

if __name__ =='__main__':
    robo = URRtde(ROBOT_IP)
    try:
        robo.get_joints()
        robo.get_pose()
        robo.moveL(TCP_TARGET, speed = 0.1, accel=0.2)
        print("finished 1")
        robo.go_home()
        print("Finished")
    finally:
        robo.shutdown()
