from robot_class_trans import URRtde
import numpy as np
from camera import detect_red_points_and_angle, new_capture
ROBOT_IP = "192.168.56.101"
start_point = [0.6932184569585337, -0.5367080771619072, 0.4230298822195191, 1.7350107136377495, -2.5619839775632967, 0.003689909797916628]

robo = URRtde(ROBOT_IP)
try:
    current_pose = robo.get_pose()
    base_pose = get_point(0,0)
    robo.moveL(base_pose)
    values = np.linspace(-90,90,15)
    # for i in values:
    #     new_pose = get_point(0,i)
    #     robo.moveL(new_pose)
    #     img_file = new_capture(filename="focused_image.jpg")
    #     pt1, pt2, angle, roi_box = detect_red_points_and_angle(
    #     img_file,
    #     show=False,
    #     use_roi=True
    # )
    #     print(angle)
    robo.moveL(base_pose)
finally:
    robo.shutdown()