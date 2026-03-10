from robot_class import URRtde
from proper_research.robot.transformations import get_point
import numpy as np
# from beam_direction_magnetisation.cosserat_6d_pose import epm_pose_orbit_and_spin_consistent

theta_orbit_z = np.deg2rad(0.0)   # orbit around tip about world z
beta_spin     = np.deg2rad(0.0)   # spin around magnet's own z
theta_orbit_y = 0.0                

test = np.array([ 0.676, -0.768,  0.45700816,  2.75324056, -1.49480981,
       -0.03850979])

rho = 0.13
p0 = np.array([0.0, 0.0, 0.0])
r_tip_nominal = p0 + np.array([0.05, 0.0, 0.0])
ROBOT_IP = "192.168.56.101"
TCP_TARGET =  [0.708, -0.868 , -0.090, -2.925,  -1.0,
        0.039]

Sleeping_joints = [5.127925760461949e-06, -1.570810934106344, 1.889864076787262e-05, -1.5707822610205149, -6.500874654591371e-06, -1.699129213506012e-05]
if __name__ == "__main__":
    robo = URRtde(ROBOT_IP)
    try:
        
        # robo.get_pose()

        # robo.go_home_joint()
        # # posy[2]+=0.2
        new_pose = get_point(0,0)
        robo.moveL(new_pose)
        # new_pose = get_point(0,60) 
        # robo.moveL(new_pose)
        
        # robo.get_pose()
        # posy[5]-=np.pi/2
        # robo.moveJ(posy)
        # robo.moveL(test)
        # robo.moveJ(Sleeping_joints)
        # print("New pose is", repr(new_pose))

        # TCP_TARGET[2] +=.25
        # robo.moveL(TCP_TARGET)
        # robo.get_pose()

        print("Finished")
    finally:
        print("done")
        robo.shutdown()
