import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D  
# from proper_research.control.mpc_sig_bound import pose7_rotvec_to_pose8_quat
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
    # plt.legend()
    plt.show()
    plt.figure()
    plt.plot(1e3*np.array(tip_y_front), 1e3*np.array(tip_z_front), marker='o', label="30 deg")
    plt.plot(1e3*np.array(tip_y_over),  1e3*np.array(tip_z_over),  marker='o', label="Magnetised at 0 deg and placed overhead")
    plt.xlabel("Tip y [mm]")
    plt.ylabel("Tip z [mm]")
    plt.title("Tip trajectory in y–z plane")
    plt.grid(True)
    plt.axis('equal')
    # plt.legend()
    plt.show()
    plt.figure()
    plt.plot(angles_deg, 1e3*np.array(tip_z_front), marker='o', label="Front magnetised at 30 deg")
    plt.plot(angles_deg, 1e3*np.array(tip_z_over),  marker='o', label="Magnetised at 0 deg and placed overhead")
    plt.xlabel("Actuation angle [deg]")
    plt.ylabel("Tip z deflection [mm]")
    plt.title("Tip z deflection vs actuation angle")
    plt.grid(True)
    # plt.legend()
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
    # ax.legend()
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

    # ax.legend()
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
    # plt.legend()
    plt.show()
def _ensure_centerline_shape(C):
    """
    Accepts C in either shape (3,N) or (N,3) and returns (N,3).
    """
    C = np.asarray(C, float)
    if C.ndim != 2:
        return None
    if C.shape[0] == 3 and C.shape[1] > 3:
        return C.T
    if C.shape[1] == 3:
        return C
    return None

def quat_wxyz_to_R(qwxyz):
    """
    q = [qw, qx, qy, qz] -> 3x3 rotation matrix
    """
    qw, qx, qy, qz = map(float, qwxyz)
    # normalize to be safe
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

    # standard quaternion -> R
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ], float)
    return R
def plot_energy_only_3d(p_energy, lumen_C=None, lumen_R=None, p0=None, p_straight=None,lumen_others=None,
                        title="Energy-min centerline", show_rings=True, plot_mag = False,
                        targets=None, tip=None, tip_from_centerline=None,
                        p_mag=None, mag_axis="x", mag_arrow_len=0.02,
                        show_tangent_segment=True, show=False, fixed_limits=None, zoom_out=1.25):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --- energy-min centerline (beam) ---
    C_energy = _ensure_centerline_shape(p_energy)
    if C_energy is None:
        raise ValueError("p_energy must be (3,N) or (N,3)")
    ax.plot(C_energy[:, 0], C_energy[:, 1], C_energy[:, 2], "--", )

    if p_straight is not None:
        C_st = _ensure_centerline_shape(p_straight)
        if C_st is not None:
            ax.plot(C_st[:, 0], C_st[:, 1], C_st[:, 2], ":", )

    if p0 is not None:
        ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="o", )

    # --- lumen (active) ---
    if (lumen_C is not None) and (lumen_R is not None):
        C = np.asarray(lumen_C, float)
        R = np.asarray(lumen_R, float)
        ax.plot(C[:, 0], C[:, 1], C[:, 2], )
        if show_rings:
            plot_lumen_rings(ax, C, R, n_theta=28, alpha=0.2, linewidth=0.6)

    # --- lumen (other branches) ---
    if lumen_others is not None:
        for item in lumen_others:
            if len(item) == 2:
                Cb, Rb = item
                lbl = "Lumen branch"
            else:
                Cb, Rb, lbl = item
            Cb = np.asarray(Cb, float)
            Rb = np.asarray(Rb, float)
            ax.plot(Cb[:, 0], Cb[:, 1], Cb[:, 2], label=lbl)
            if show_rings:
                plot_lumen_rings(ax, Cb, Rb, n_theta=28, alpha=0.12, linewidth=0.5)
    # --- tip marker (your existing) ---
    if tip is not None:
        tip = np.asarray(tip, float).reshape(3,)
        ax.scatter([tip[0]], [tip[1]], [tip[2]], marker="x", s=60, )

    # --- tangent point + tangent segment (NEW) ---
    if show_tangent_segment and C_energy.shape[0] >= 2:
        p_prev = C_energy[-2, :]
        p_end  = C_energy[-1, :]

        # "tangent point" (end)
        ax.scatter([p_end[0]], [p_end[1]], [p_end[2]], marker="o", s=50, label="Tangent point")

        # segment used to compute tangent
        ax.plot([p_prev[0], p_end[0]],
                [p_prev[1], p_end[1]],
                [p_prev[2], p_end[2]],
                linewidth=3.0, label="Tangent segment")

    # --- centerline end (your existing; keep if you want both) ---
    if tip_from_centerline is not None:
        tip_from_centerline = np.asarray(tip_from_centerline, float).reshape(3,)
        ax.scatter([tip_from_centerline[0]], [tip_from_centerline[1]], [tip_from_centerline[2]],
                   marker="o", s=40, label="Centerline end")

    # --- MPC targets ---
    if targets is not None:
        T = np.asarray(targets, float).reshape(-1, 3)
        ax.scatter(T[:, 0], T[:, 1], T[:, 2], marker="^", s=30, label="MPC targets")
        ax.plot(T[:, 0], T[:, 1], T[:, 2], "-", linewidth=1.0, label="Target horizon")
        if T.shape[0] > 0:
            ax.text(T[0,0], T[0,1], T[0,2] + 0.0002, "0")
        if T.shape[0] > 1:
            ax.text(T[1,0], T[1,1], T[1,2] + 0.0002, "1")

    # --- external magnet pose  ---
    if p_mag is not None and plot_mag==True:
        p_mag = np.asarray(p_mag, float).ravel()

        # supports pose8=[x,y,z,qw,qx,qy,qz,L] or pose7=[x,y,z,rx,ry,rz,L]
        # If pose7 rotvec, you can convert to quat first if you already have a helper.
        # if p_mag.size >= 7 and p_mag.size != 8:
        #     # assume [x,y,z,rx,ry,rz,L] -> convert using your existing helper if available
        #     # comment this out if you don't want the conversion
        #     p8 = pose7_rotvec_to_pose8_quat(p_mag)  # you already have this in your code
        #     p_mag = p8

        if p_mag.size >= 7:
            pm = p_mag[:3]
            q = p_mag[3:7]  # qw,qx,qy,qz
            Rm = quat_wxyz_to_R(q)

            # choose axis in magnet frame
            if mag_axis == "x":
                a_local = np.array([1.0, 0.0, 0.0])
            elif mag_axis == "y":
                a_local = np.array([0.0, 1.0, 0.0])
            else:
                a_local = np.array([0.0, 0.0, 1.0])

            a_world = Rm @ a_local
            p2 = pm + mag_arrow_len * a_world

            ax.scatter([pm[0]], [pm[1]], [pm[2]], marker="s", s=60, label="Magnet position")
            ax.plot([pm[0], p2[0]], [pm[1], p2[1]], [pm[2], p2[2]], linewidth=3.0,
                    label=f"Magnet {mag_axis}-axis")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    # ax.legend()

    pts = [C_energy]
    if p_straight is not None:
        C_st = _ensure_centerline_shape(p_straight)
        if C_st is not None: pts.append(C_st)
    if p0 is not None: pts.append(np.asarray(p0, float).reshape(1,3))
    if lumen_C is not None: pts.append(np.asarray(lumen_C, float))
    if targets is not None: pts.append(np.asarray(targets, float))
    if lumen_others is not None:
        for item in lumen_others:
            pts.append(np.asarray(item[0], float))
    if tip is not None: pts.append(np.asarray(tip, float).reshape(1,3))
    if p_mag is not None and np.asarray(p_mag).size >= 3 and plot_mag==True:
        pts.append(np.asarray(p_mag[:3], float).reshape(1,3))

    P = np.vstack(pts)

    # # ---- FIX LIMITS ONCE ----
    if fixed_limits is None:
        pmin = P.min(axis=0)
        pmax = P.max(axis=0)
        c = 0.5*(pmin + pmax)
        half = 0.5*np.max(pmax - pmin)

        # Zoom out (1.0 = tight; 1.25 = 25% more; 2.0 = 2x bigger)
        half *= float(zoom_out)

        pmin = c - half
        pmax = c + half
        fixed_limits = (pmin, pmax)

    # # Always apply the same limits
    pmin, pmax = fixed_limits
    ax.set_xlim(pmin[0], pmax[0])
    ax.set_ylim(pmin[1], pmax[1])
    ax.set_zlim(pmin[2], pmax[2])
    ax.set_box_aspect((1, 1, 1))

    if show:
        plt.show()

    return fig, ax, fixed_limits
def plot_error_vs_s(s, p_bvp, p_energy):
    err = np.linalg.norm(p_bvp - p_energy, axis=0)  # (N,)
    plt.figure()
    plt.plot(s, err)
    plt.xlabel("s (m)")
    plt.ylabel("||p_bvp(s) - p_energy(s)|| (m)")
    plt.title("Centerline deviation vs arclength")
    plt.grid(True)
    plt.show()

def make_lumen_centerline_turning(
    p_start,
    t0,
    *,
    length=0.12,
    n_pts=60,
    bend_axis=np.array([0.0, 0.0, 1.0]),  # rotate t about this axis
    bend_angle=np.deg2rad(25.0),          # total bend (radians)
    bend_start=0.02,                      # where bending begins along arc (m)
    bend_end=0.09,                        # where bending ends along arc (m)
):
    """
    Returns C: (n_pts,3) polyline points.
    Tangent starts as t0, then smoothly rotates by bend_angle around bend_axis.
    """
    p_start = np.asarray(p_start, float).reshape(3,)
    t0 = np.asarray(t0, float).reshape(3,)
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)

    a = np.asarray(bend_axis, float).reshape(3,)
    a = a / (np.linalg.norm(a) + 1e-12)

    s = np.linspace(0.0, float(length), int(n_pts))
    ds = np.diff(s)

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x*x*(3 - 2*x)

    # bend progress 0->1 between bend_start and bend_end
    xi = (s - bend_start) / (bend_end - bend_start + 1e-12)
    g = smoothstep(xi)  # (n_pts,)
    theta = bend_angle * g

    def rodrigues(v, axis, ang):
        # rotate vector v about axis by ang (scalar)
        v = np.asarray(v, float)
        axis = np.asarray(axis, float)
        return (v*np.cos(ang)
                + np.cross(axis, v)*np.sin(ang)
                + axis*np.dot(axis, v)*(1 - np.cos(ang)))

    # integrate centerline
    C = np.zeros((s.size, 3), float)
    C[0] = p_start
    for i in range(s.size - 1):
        ti = rodrigues(t0, a, theta[i])
        ti = ti / (np.linalg.norm(ti) + 1e-12)
        C[i+1] = C[i] + ds[i] * ti

    return C
def plot_lumen_rings(ax, C, R, n_theta=24, alpha=0.15, linewidth=0.5):
    """
    Draw translucent rings (circles) along a polyline centerline C with radii R.
    Uses a local normal basis per segment.
    """
    C = np.asarray(C, float)
    R = np.asarray(R, float)

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=True)

    for i in range(len(C)-1):
        a = C[i]
        b = C[i+1]
        t = b - a
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-12:
            continue
        t = t / t_norm

        # pick a reference vector not parallel to t
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(t, ref)
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = np.cross(t, n1)

        # radius at segment endpoints
        r0 = R[i]
        r1 = R[i+1]

        # draw ring at start and end of the segment
        for center, rr in [(a, r0), (b, r1)]:
            ring = center[:, None] + rr*(n1[:,None]*np.cos(thetas)[None,:] +
                                         n2[:,None]*np.sin(thetas)[None,:])
            ax.plot(ring[0], ring[1], ring[2], alpha=alpha, linewidth=linewidth)

import numpy as np

def make_lumen_centerline_double_turn(
    p_start,
    t0,
    *,
    length=0.12,
    n_pts=60,
    bend_axis=np.array([0.0, 0.0, 1.0]),
    # first turn
    bend1_angle=np.deg2rad(25.0),
    bend1_start=0.02,
    bend1_end=0.06,
    # second turn
    bend2_angle=np.deg2rad(-25.0),   # negative = opposite direction
    bend2_start=0.06,
    bend2_end=0.10,
):
    """
    Returns C: (n_pts,3) polyline points.
    Tangent starts as t0, then does two smooth turns by rotating about bend_axis.
    """
    p_start = np.asarray(p_start, float).reshape(3,)
    t0 = np.asarray(t0, float).reshape(3,)
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)

    a = np.asarray(bend_axis, float).reshape(3,)
    a = a / (np.linalg.norm(a) + 1e-12)

    s = np.linspace(0.0, float(length), int(n_pts))
    ds = np.diff(s)

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x*x*(3 - 2*x)

    def window_theta(s, s0, s1, angle):
        # progress 0->1 inside [s0,s1], 0 before, 1 after
        xi = (s - s0) / (s1 - s0 + 1e-12)
        g = smoothstep(xi)
        return angle * g

    # total bend angle vs arc length
    theta = (
        window_theta(s, bend1_start, bend1_end, bend1_angle) +
        window_theta(s, bend2_start, bend2_end, bend2_angle)
    )

    def rodrigues(v, axis, ang):
        v = np.asarray(v, float)
        axis = np.asarray(axis, float)
        return (v*np.cos(ang)
                + np.cross(axis, v)*np.sin(ang)
                + axis*np.dot(axis, v)*(1 - np.cos(ang)))

    C = np.zeros((s.size, 3), float)
    C[0] = p_start
    for i in range(s.size - 1):
        ti = rodrigues(t0, a, theta[i])
        ti = ti / (np.linalg.norm(ti) + 1e-12)
        C[i+1] = C[i] + ds[i] * ti

    return C
def _set_axes_equal_about_data(ax, X):
    """
    X: (M,3) array of all points you want in view.
    Sets equal aspect and recenters.
    """
    X = np.asarray(X, float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ctr  = 0.5 * (mins + maxs)
    ranges = maxs - mins
    r = 0.5 * np.max(ranges)
    if r < 1e-9:
        r = 1.0  # fallback

    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)

    # Matplotlib >=3.3 supports set_box_aspect
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
def plot_centerlines_no_lumen_3d(
    p_bvp,
    p_energy,
    lumen_C=None,
    lumen_R=None,
    p0=None,
    p_straight=None,
    r_src=None,
    q_src=None,
    contact=None,
    show_beam_tube=True,
    title="Centerline comparison",
    save_path=None,
    show=True,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------------------------------
    # Beam centreline comparison
    # -------------------------------------------------
    if p_bvp is not None:
        ax.plot(
            p_bvp[0],
            p_bvp[1],
            p_bvp[2],
            "-",
            linewidth=2.0,
            label="Cosserat BVP",
        )

    if p_energy is not None:
        ax.plot(
            p_energy[0],
            p_energy[1],
            p_energy[2],
            "--",
            linewidth=2.0,
            label="Energy minimisation",
        )

    if p_straight is not None:
        ax.plot(
            p_straight[0],
            p_straight[1],
            p_straight[2],
            ":",
            linewidth=1.5,
            label="Straight baseline",
        )

    if p0 is not None:
        ax.scatter(
            [p0[0]],
            [p0[1]],
            [p0[2]],
            marker="o",
            label="Base",
        )

    # -------------------------------------------------
    # External magnet
    # -------------------------------------------------
    if r_src is not None:
        r_src = np.asarray(r_src, float).reshape(3,)

        ax.scatter(
            [r_src[0]],
            [r_src[1]],
            [r_src[2]],
            marker="*",
            s=120,
            label="External magnet",
        )

        ax.text(
            r_src[0],
            r_src[1],
            r_src[2],
            f"  Magnet\n  ({r_src[0]:.4f}, {r_src[1]:.4f}, {r_src[2]:.4f})",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_title(title)

    # -------------------------------------------------
    # Axis scaling
    # -------------------------------------------------
    pts = []

    if p_energy is not None:
        pts.append(np.asarray(p_energy, float).T)


    if p_bvp is not None:
        pts.append(np.asarray(p_bvp, float).T)

    if p_straight is not None:
        pts.append(np.asarray(p_straight, float).T)

    if p0 is not None:
        pts.append(np.asarray(p0, float).reshape(1, 3))

    if lumen_C is not None:
        pts.append(np.asarray(lumen_C, float))

    if pts:
        all_pts = np.vstack(pts)
        _set_axes_equal_about_data(ax, all_pts)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def plot_centerlines_with_lumen_3d(
    p_bvp,
    p_energy,
    lumen_C=None,
    lumen_R=None,
    p0=None,
    p_straight=None,
    r_src=None,
    q_src=None,
    contact=None,
    show_beam_tube=True,
    title="Centerline + Lumen",
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------------------------------
    # Beam centreline + physical beam diameter
    # -------------------------------------------------
    # ax.plot(
    #     p_energy[0],
    #     p_energy[1],
    #     p_energy[2],
    #     "--",
    #     linewidth=1.5,
    #     label="Beam centreline",
    # )

    if show_beam_tube and contact is not None:
        plot_tube_along_centerline(
            ax,
            p_energy,
            radius=float(contact.r_beam),
            n_theta=28,
            alpha=0.45,
            label=f"Beam body, radius={1e3 * contact.r_beam:.1f} mm",
        )
    if show_beam_tube and contact is not None:
        plot_tube_along_centerline(
            ax,
            p_bvp,
            radius=float(contact.r_beam),
            n_theta=28,
            alpha=0.45,
            label=f"Beam body, radius={1e3 * contact.r_beam:.1f} mm",
        )
    # if p_straight is not None:
    #     ax.plot(
    #         p_straight[0],
    #         p_straight[1],
    #         p_straight[2],
    #         ":",
    #         linewidth=1.5,
    #         label="Straight baseline",
        # )

    if p0 is not None:
        ax.scatter(
            [p0[0]],
            [p0[1]],
            [p0[2]],
            marker="o",
            label="Base",
        )
    # if p_bvp is not None:
    #     ax.plot(
    #         p_bvp[0],
    #         p_bvp[1],
    #         p_bvp[2],
    #         "-",
    #         linewidth=2.0,
    #         label="Non Contact Aware",
    #     )

    # ax.plot(
    #     p_energy[0],
    #     p_energy[1],
    #     p_energy[2],
    #     "--",
    #     linewidth=2.0,
    #     label="Energy minimisation",
    # )
    # -------------------------------------------------
    # Lumen wall
    # -------------------------------------------------
    if (lumen_C is not None) and (lumen_R is not None):
        C = np.asarray(lumen_C, float)
        R_lum = np.asarray(lumen_R, float)

        # ax.plot(
        #     C[:, 0],
        #     C[:, 1],
        #     C[:, 2],
        #     label="Lumen centerline",
        # )

        plot_lumen_rings(
            ax,
            C,
            R_lum,
            n_theta=28,
            alpha=1,
        )

        # Tip and closest lumen point
        p_tip = p_energy[:, -1]
        d, i, t, q = point_to_polyline_distance(p_tip, C)

        # ax.scatter(
        #     [p_tip[0]],
        #     [p_tip[1]],
        #     [p_tip[2]],
        #     marker="^",
        #     label="Energy tip",
        # )

        # ax.scatter(
        #     [q[0]],
        #     [q[1]],
        #     [q[2]],
        #     marker="x",
        #     label="Closest lumen point",
        # )

        # ax.plot(
        #     [p_tip[0], q[0]],
        #     [p_tip[1], q[1]],
        #     [p_tip[2], q[2]],
        #     linewidth=1.0,
        # )

    # # -------------------------------------------------
    # # External magnet
    # # -------------------------------------------------
    if r_src is not None:
        r_src = np.asarray(r_src, float).reshape(3,)

        ax.scatter(
            [r_src[0]],
            [r_src[1]],
            [r_src[2]],
            marker="*",
            s=120,
            label="External magnet",
        )

        ax.text(
            r_src[0],
            r_src[1],
            r_src[2],
            f"  Magnet\n  ({r_src[0]:.4f}, {r_src[1]:.4f}, {r_src[2]:.4f})",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # ax.legend()
    ax.set_title(title)

    # -------------------------------------------------
    # Axis scaling
    # -------------------------------------------------
    pts = []
    pts.append(p_energy.T)

    if show_beam_tube and contact is not None:
        # Include approximate beam tube extent in view limits
        r = float(contact.r_beam)
        pts.append(p_energy.T + np.array([[r, 0.0, 0.0]]))
        pts.append(p_energy.T - np.array([[r, 0.0, 0.0]]))
        pts.append(p_energy.T + np.array([[0.0, r, 0.0]]))
        pts.append(p_energy.T - np.array([[0.0, r, 0.0]]))
        pts.append(p_energy.T + np.array([[0.0, 0.0, r]]))
        pts.append(p_energy.T - np.array([[0.0, 0.0, r]]))

    if p_straight is not None:
        pts.append(p_straight.T)

    if p0 is not None:
        pts.append(np.asarray(p0, float).reshape(1, 3))

    if lumen_C is not None:
        pts.append(np.asarray(lumen_C, float))

    # Optional: do not include magnet in view if it makes local geometry unreadable.
    if r_src is not None:
        pts.append(np.asarray(r_src, float).reshape(1, 3))

    all_pts = np.vstack(pts)
    _set_axes_equal_about_data(ax, all_pts)

    plt.show()

def point_to_polyline_distance(p, C):
    """
    p: (3,)
    C: (M,3) polyline points
    Returns: d_min, seg_idx, t_on_segment, q_closest
    """
    d_min = np.inf
    best = (None, None, None)
    for i in range(len(C) - 1):
        q, t = closest_point_on_segment(p, C[i], C[i+1])
        d = np.linalg.norm(p - q)
        if d < d_min:
            d_min = d
            best = (i, t, q)
    i, t, q = best
    return d_min, i, t, q
def closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    q = a + t * ab
    return q, t
def plot_tube_along_centerline(ax, P, radius, *, n_theta=24, alpha=0.45, label=None):
    """
    Plot a tube of given radius around a 3 x N centreline P.
    """
    P = np.asarray(P, float)
    assert P.shape[0] == 3
    N = P.shape[1]

    # Tangents
    T = np.zeros_like(P)
    T[:, 1:-1] = P[:, 2:] - P[:, :-2]
    T[:, 0] = P[:, 1] - P[:, 0]
    T[:, -1] = P[:, -1] - P[:, -2]
    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-12

    # Build normal/binormal frames
    Nvec = np.zeros_like(P)
    Bvec = np.zeros_like(P)

    ref = np.array([0.0, 0.0, 1.0])

    for i in range(N):
        t = T[:, i]

        # Avoid degeneracy if tangent almost parallel to ref
        if abs(np.dot(t, ref)) > 0.9:
            ref_i = np.array([0.0, 1.0, 0.0])
        else:
            ref_i = ref

        n = np.cross(t, ref_i)
        n /= np.linalg.norm(n) + 1e-12

        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-12

        Nvec[:, i] = n
        Bvec[:, i] = b

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    X = np.zeros((N, n_theta))
    Y = np.zeros((N, n_theta))
    Z = np.zeros((N, n_theta))

    for i in range(N):
        ring = (
            P[:, i, None]
            + radius * np.cos(theta)[None, :] * Nvec[:, i, None]
            + radius * np.sin(theta)[None, :] * Bvec[:, i, None]
        )
        X[i, :] = ring[0]
        Y[i, :] = ring[1]
        Z[i, :] = ring[2]

    ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=0, shade=False)

    if label is not None:
        # Dummy handle for legend
        ax.plot([], [], [], linewidth=6, alpha=alpha, label=label)
def maybe_plot_summary(k_hist, pred1_hist, svd_S_hist, svd_cond_hist):
    K = np.asarray(k_hist)
    pred1 = np.asarray(pred1_hist)
    Smat = np.asarray(svd_S_hist)
    cond = np.asarray(svd_cond_hist)

    plt.figure()
    plt.plot(K, 1e3 * pred1)
    plt.xlabel("k")
    plt.ylabel("pred1_err (mm)")
    plt.title("One-step prediction error vs step")
    plt.grid(True)
    # plt.show()

    plt.figure()
    if Smat.ndim == 2 and Smat.shape[0] == K.size:
        for i in range(Smat.shape[1]):
            plt.plot(K, Smat[:, i], label=f"σ{i+1}")
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("singular values of B (log)")
    plt.title("Jacobian singular values vs step")
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(K, cond)
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("cond(B) (log)")
    plt.title("Jacobian conditioning vs step")
    plt.grid(True)
    # plt.show()

    fig, ax1 = plt.subplots()
    l1, = ax1.plot(K, 1e3 * pred1, label="pred1_err (mm)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("pred1_err (mm)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    l2, = ax2.plot(K, cond, label="cond(B)")
    ax2.set_yscale("log")
    ax2.set_ylabel("cond(B) (log)")

    ax1.legend(handles=[l1, l2], loc="best")
    plt.title("pred1_err vs Jacobian conditioning")
    # plt.show()