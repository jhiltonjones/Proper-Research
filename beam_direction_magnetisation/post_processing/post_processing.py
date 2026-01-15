import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D  
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_to_rot
def compare_magnet_plots(angles_deg, tip_y_front, tip_y_over, tip_z_front, tip_z_over):
    plt.figure()
    plt.plot(angles_deg, 1e3*np.array(tip_y_front), marker='o', label="Front magnetised at 30 deg")
    plt.plot(angles_deg, 1e3*np.array(tip_y_over),  marker='o', label="Magnetised at 0 deg and placed overhead")
    plt.xlabel("Actuation angle [deg]")
    plt.ylabel("Tip y deflection [mm]")
    plt.title("Tip y deflection vs actuation angle")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(1e3*np.array(tip_y_front), 1e3*np.array(tip_z_front), marker='o', label="30 deg")
    plt.plot(1e3*np.array(tip_y_over),  1e3*np.array(tip_z_over),  marker='o', label="Magnetised at 0 deg and placed overhead")
    plt.xlabel("Tip y [mm]")
    plt.ylabel("Tip z [mm]")
    plt.title("Tip trajectory in y–z plane")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(angles_deg, 1e3*np.array(tip_z_front), marker='o', label="Front magnetised at 30 deg")
    plt.plot(angles_deg, 1e3*np.array(tip_z_over),  marker='o', label="Magnetised at 0 deg and placed overhead")
    plt.xlabel("Actuation angle [deg]")
    plt.ylabel("Tip z deflection [mm]")
    plt.title("Tip z deflection vs actuation angle")
    plt.grid(True)
    plt.legend()
    plt.show()

def axis3d(x,y,z,r_src_f, m_src_f, s_out, p_f, q_f, ang_plot, f_toward):


    fig3d = plt.figure(figsize=(8, 6))
    ax = fig3d.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=2, label="rod centerline")
    ax.scatter([x[0]], [y[0]], [z[0]], s=40, label="base")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], s=40, label="tip")
    pts = np.column_stack([x, y, z])          
    pts = np.vstack([pts, r_src_f[None, :]]) 

    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)

    x_mid = 0.5*(xmin + xmax)
    y_mid = 0.5*(ymin + ymax)
    z_mid = 0.5*(zmin + zmax)

    max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
    half = 0.5*max_range

    margin = 0.15 * max_range
    half = half + margin

    ax.set_xlim(x_mid-half, x_mid+half)
    ax.set_ylim(y_mid-half, y_mid+half)
    ax.set_zlim(z_mid-half, z_mid+half)
    ax.set_box_aspect((1, 1, 1))

    ax.scatter([r_src_f[0]], [r_src_f[1]], [r_src_f[2]], s=60, marker='^', label="EPM")

    mhat = m_src_f / np.linalg.norm(m_src_f)
    arrow_len = 0.02  
    ax.quiver(r_src_f[0], r_src_f[1], r_src_f[2],
            mhat[0], mhat[1], mhat[2],
            length=arrow_len, normalize=True)
    R = quat_to_rot(q_f)
    step = 30             
    dlen = 0.005        

    for i in range(0, len(s_out), step):
        Ri = R[i, :, :]         
        pi = p_f[:, i]

        d1 = Ri @ np.array([1.0, 0.0, 0.0])
        d2 = Ri @ np.array([0.0, 1.0, 0.0])
        d3 = Ri @ np.array([0.0, 0.0, 1.0])

        ax.quiver(pi[0], pi[1], pi[2], d1[0], d1[1], d1[2], length=dlen, normalize=True)
        ax.quiver(pi[0], pi[1], pi[2], d2[0], d2[1], d2[2], length=dlen, normalize=True)
        ax.quiver(pi[0], pi[1], pi[2], d3[0], d3[1], d3[2], length=dlen, normalize=True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Front magnet case: rod shape at angle = {ang_plot:.1f} deg")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(s_out, f_toward)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("s [m]")
    plt.ylabel("f · u_hat [N/m]")
    plt.title("Local magnetic force component toward magnet")
    plt.grid(True)
    plt.show()
