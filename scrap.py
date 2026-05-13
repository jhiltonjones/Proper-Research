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
TCP_TARGET =  [0.8099431012797347, -0.39283417540219145, 0.3278606423024446, 0.6905457055161595, 3.043748590217591, -0.03916440883605399]

Sleeping_joints = [5.127925760461949e-06, -1.570810934106344, 1.889864076787262e-05, -1.5707822610205149, -6.500874654591371e-06, -1.699129213506012e-05]
if __name__ == "__main__":
    robo = URRtde(ROBOT_IP)
    try:
        
        # robo.get_pose()

        robo.go_home_joint()
        # # posy[2]+=0.2
        new_pose = get_point(0,0)
        robo.moveL(new_pose)
        # q0 = np.array([-0.6708453337298792, -2.5134712658324183, -1.0908514261245728, -1.107888327246048, 1.5738656520843506, 4.485052585601807])
        # robo.moveJ(q0)
        # q0[5]-=np.deg2rad(90)
        # robo.moveJ(q0)
        # robo_joints = robo.get_joints()
        # robo_joints[5] -=np.deg2rad(5)
        # robo.moveJ(robo_joints)
        # # print(robo_joints)
        # robo_joints[5]+= np.deg2rad(10)
        # robo.moveJ(robo_joints)
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
        pose = robo.get_pose()
        # pose[0] = 0.2
        # robo.moveL(pose)
        print(pose)

        print("Finished")
    finally:
        print("done")
        robo.shutdown()

[0.6403439968599637, -0.6373844449362759, 0.16999284859452013, 3.140523214730508, 0.0009964522447244207, 0.0009903154277649615]