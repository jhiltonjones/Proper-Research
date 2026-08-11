from ur_rtde_robot import URRTDERobot

ROBOT_IP = "192.168.56.101"

HOME_JOINTS = [
    -0.4124181906329554,
    -1.581008096734518,
    -1.974020004272461,
    -1.1402791303447266,
    1.5827715396881104,
    -0.21076661745180303,
]

with URRTDERobot(ROBOT_IP, frequency=125.0) as robot:
    print("Joints:", robot.get_joints())
    print("TCP pose:", robot.get_tcp_pose())

    robot.move_j(
        HOME_JOINTS,
        speed=0.50,
        acceleration=0.20,
    )