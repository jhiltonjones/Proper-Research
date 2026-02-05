import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D  
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_to_rot
from beam_direction_magnetisation.quarternions.shared_rotations import Rx, Ry, Rz
from scipy.spatial.transform import Rotation as Rot
import cv2 
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

def axis3d(x,y,z,r_src_f, m_src_f, s_out, p_f, q_f, ang_plot, f_toward, vessel_centerline, R_vessel):


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
    add_tube(ax, vessel_centerline, R_vessel, alpha=0.12)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # plt.figure()
    # plt.plot(s_out, f_toward)
    # plt.axhline(0.0, linewidth=1)
    # plt.xlabel("s [m]")
    # plt.ylabel("f · u_hat [N/m]")
    # plt.title("Local magnetic force component toward magnet")
    # plt.grid(True)
    # plt.show()


def add_tube(ax, centerline, R, n_circle=24, n_samples=80, alpha=0.15):
    # Sample points along the polyline
    pts = []
    seglens = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    total = seglens.sum()

    # allocate samples per segment
    for i in range(len(centerline) - 1):
        a, b = centerline[i], centerline[i+1]
        Ni = max(2, int(n_samples * seglens[i] / total))
        t = np.linspace(0, 1, Ni)
        pts.append(a[None,:] * (1-t)[:,None] + b[None,:] * t[:,None])
    pts = np.vstack(pts)  # (Ns,3)

    # Build a tube surface by sweeping a circle
    theta = np.linspace(0, 2*np.pi, n_circle, endpoint=True)
    X = np.zeros((len(theta), len(pts)))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)

    # For planar x-y centerline, a simple fixed circle basis works (normal is z)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])

    for j, c in enumerate(pts):
        circle = (R*np.cos(theta)[:,None]*ex[None,:] +
                  R*np.sin(theta)[:,None]*ey[None,:])
        ring = c[None,:] + circle
        X[:, j] = ring[:, 0]
        Y[:, j] = ring[:, 1]
        Z[:, j] = ring[:, 2]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=alpha)
    ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], '--', linewidth=2, label="vessel centerline")
    """
Drop-in plotting utility for your MPC loop.

What it plots (3D):
- Beam centerline from Cosserat model (base -> tip), using the SAME forward call you use
- Tip position (from your MPC's x_now / x_post)
- Magnet position r_src (from your MPC's p_now / p_post, first 3 entries)
- Dipole direction (magnet local +x axis rotated by roll/pitch/yaw in p), as an arrow
- Optional dashed line tip -> magnet (distance check)

How to use:
1) Paste this block below your definitions of Rx/Ry/Rz (or keep these provided ones)
2) In your MPC loop, after mpc.step(...), call:
     plot_mpc_state_3d(model, m_body, p_post, x_post, show=True)
   or store frames by setting show=False and saving figures.

Assumptions:
- Your model.forward(...) returns a dict that includes either:
    out["p_s"] or out["p"] or out["p_nodes"] or out["r"] as (N,3) centerline points.
  If none exist, we fall back to a straight line base->tip.
- Beam base pose is model.p0 and model.q0 (as you construct it); base point is model.p0.
"""

def R_from_rpy(roll, pitch, yaw):
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v if n < eps else (v / n)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# ---------- beam centerline extraction ----------
def extract_centerline_from_forward_output(out):
    """
    Try common keys for centerline/backbone points.
    Returns None if not found.
    """
    for key in ["p_s", "p", "p_nodes", "r", "centerline", "backbone"]:
        if key in out:
            arr = np.asarray(out[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 2:
                return arr
    return None


# ---------- plotting ----------
def plot_mpc_state_3d(model, m_body, p_pose7, x_tip, L, mag_len,
                      title=None,
                      dipole_scale=0.05,
                      show_tip_to_mag=True,
                      show=True,
                      ax=None):
    """
    model: your CosseratForwardModel instance
    m_body: dipole vector in body frame passed to model.forward
    p_pose7: (7,) [x,y,z, roll,pitch,yaw, L] => magnet pose + beam length used in model.forward
    x_tip: (3,) tip position (your MPC x_now / x_post)
    """

    p_pose7 = np.asarray(p_pose7, dtype=float).ravel()
    x_tip   = np.asarray(x_tip,   dtype=float).ravel()
    assert p_pose7.size == 7
    assert x_tip.size == 3

    r_mag = p_pose7[0:3]
    L = float(p_pose7[6])
    rvec = p_pose7[3:6]
    Rsrc = Rot.from_rotvec(rvec).as_matrix()


    # robust rot->quat not needed for plotting, but model.forward expects q_src quaternion.
    # We'll implement a minimal rot->quat here:
    def rot_to_quat(R):
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S
        q = np.array([qw, qx, qy, qz], dtype=float)
        return q / (np.linalg.norm(q) + 1e-12)

    q_src = rot_to_quat(Rsrc)

    wire_len = L - mag_len
    out = model.forward(L=L, r_src=r_mag, q_src=q_src, m_body=m_body, wire_len=wire_len)

    # Tip from model output (for sanity) if present; otherwise use x_tip provided
    x_tip_model = None
    if isinstance(out, dict) and "p_tip" in out:
        x_tip_model = np.asarray(out["p_tip"], dtype=float).ravel()
        if x_tip_model.size == 3:
            pass
        else:
            x_tip_model = None

    # Centerline
    centerline = extract_centerline_from_forward_output(out) if isinstance(out, dict) else None

    # If no centerline provided by model.forward, fall back to a straight segment base->tip
    # Base point: try model.p0 else origin
    r_base = getattr(model, "p0", np.array([0.0, 0.0, 0.0], dtype=float))
    r_base = np.asarray(r_base, dtype=float).ravel()
    if r_base.size != 3:
        r_base = np.array([0.0, 0.0, 0.0], dtype=float)

    if centerline is None:
        centerline = np.vstack([r_base, x_tip])

    # Dipole direction in world frame: local +x rotated by Rsrc
    dip_hat = unit(Rsrc @ np.array([1.0, 0.0, 0.0], dtype=float))

    # Create axes if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Plot beam centerline
    ax.plot(centerline[:,0], centerline[:,1], centerline[:,2],
            linewidth=3, label="Beam centerline")

    # Plot base and tip
    ax.scatter([centerline[0,0]], [centerline[0,1]], [centerline[0,2]], s=60, label="Beam base")
    ax.scatter([x_tip[0]], [x_tip[1]], [x_tip[2]], s=70, label="Tip (from MPC)")

    if x_tip_model is not None:
        ax.scatter([x_tip_model[0]], [x_tip_model[1]], [x_tip_model[2]], s=40, label="Tip (from model)")

    # Plot magnet position
    ax.scatter([r_mag[0]], [r_mag[1]], [r_mag[2]], s=90, label="Magnet position")

    # Dipole arrow
    ax.quiver(r_mag[0], r_mag[1], r_mag[2],
              dip_hat[0], dip_hat[1], dip_hat[2],
              length=dipole_scale, normalize=True,
              label="Dipole direction (+x_local)")

    # Tip -> magnet dashed line and distance
    if show_tip_to_mag:
        ax.plot([x_tip[0], r_mag[0]],
                [x_tip[1], r_mag[1]],
                [x_tip[2], r_mag[2]],
                linestyle="--", linewidth=2, label=f"Tip→Mag ({np.linalg.norm(r_mag-x_tip):.3f} m)")

    # Labels
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title is None:
        title = f"MPC state: r_src={r_mag}, rpy(deg)={[np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]}"
    ax.set_title(title)

    # Limits from plotted points
    pts = np.vstack([centerline, r_mag.reshape(1,3), x_tip.reshape(1,3)])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    pad = 0.05
    ax.set_xlim(mins[0]-pad, maxs[0]+pad)
    ax.set_ylim(mins[1]-pad, maxs[1]+pad)
    ax.set_zlim(mins[2]-pad, maxs[2]+pad)
    set_axes_equal(ax)

    ax.legend()
    if show:
        plt.show()

    return fig, ax

def plot_target_and_final_tip_on_image(image_filename, target_px, tip_px, err_mm=None):
    img_bgr = cv2.imread(image_filename)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_filename}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)

    # target
    plt.scatter([target_px[0]], [target_px[1]], s=90, marker="x", color="yellow", label="Target", zorder=3)

    # tip
    plt.scatter([tip_px[0]], [tip_px[1]], s=70, marker="o", color="red", label="Final tip", zorder=3)

    # line + annotation
    plt.plot([target_px[0], tip_px[0]], [target_px[1], tip_px[1]],
             linestyle="--", linewidth=1.5, color="white", zorder=2)

    if err_mm is not None:
        plt.text(tip_px[0] + 3, tip_px[1] + 3, f"{err_mm:.1f} mm",
                 color="cyan", fontsize=10, ha="left", va="bottom", zorder=4)

    plt.title("Target vs final measured tip")
    plt.axis("off")
    plt.legend()
    plt.show()
