import numpy as np
from scipy.spatial.transform import Rotation as Rot

from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    dipole_from_pose, solve_quasistatic_insertion
)
from beam_direction_magnetisation.magnetism.beam_geometry import make_m_local_fun_wire_tip

# You already have: unpack_pose_ur_rotvec_L(p)
# and wire_len_from_L(L, L_mag)

class EnergyMinForwardWithLumen:
    """
    MPC-facing forward model:
      p = [r_src(3), rotvec(3), L]
      returns tip xyz
    Uses lumen boundary (tip/rod handled by your solver) and caches warm-starts.
    """

    def __init__(self, *,
                 p0_ur, q0_ur,
                 Kinv_fun, u_star,
                 mag_len, m_body,
                 lumen_C, lumen_R,
                 N_nodes=15, maxiter=15,
                 L0_init=0.01, dL_internal=0.002, use_lumen_jac = True):
        self.p0_ur = np.asarray(p0_ur, float)
        self.q0_ur = np.asarray(q0_ur, float)

        self.Kinv_fun = Kinv_fun
        self.u_star = np.asarray(u_star, float).reshape(3,)

        self.mag_len = float(mag_len)
        self.m_body = np.asarray(m_body, float).reshape(3,)

        self.lumen_C = np.asarray(lumen_C, float)
        self.lumen_R = np.asarray(lumen_R, float)

        self.N_nodes = int(N_nodes)
        self.maxiter = int(maxiter)

        # internal continuation controls
        self.L0_init = float(L0_init)
        self.dL_internal = float(dL_internal)
        self.use_lumen_jac = use_lumen_jac
        # cache
        self._last = dict(
            p=None,
            L=None,
            u_init=None,
            tip=None,
            p_centerline=None,   # NEW
            info=None,           # NEW
            hist=None            # NEW (optional but handy)
        )

        # public last-result mirrors (nice for plotting/debug)
        self.last_p_centerline = None
        self.last_tip = None
        self.last_info = None
        self.last_hist = None

    def _quat_from_rotvec_ur(self, rvec):
        # UR rotvec is axis-angle; scipy Rot.from_rotvec uses same
        Rm = Rot.from_rotvec(rvec).as_matrix()
        q_xyzw = Rot.from_matrix(Rm).as_quat()  # [x,y,z,w]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
        q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
        return q_wxyz

    def __call__(self, p):
        p = np.asarray(p, float).reshape(-1)
        if p.size != 7:
            raise ValueError("Expected p = [x,y,z, rx,ry,rz, L] (7,)")

        r_src = p[0:3]
        rvec  = p[3:6]
        L     = float(p[6])

        # guard
        wire_len = L - self.mag_len
        if wire_len <= 1e-6:
            # infeasible: return large value to discourage
            return np.array([1e2, 1e2, 1e2], float)

        # magnet pose
        q_src = self._quat_from_rotvec_ur(rvec)

        # dipole vector in world
        m_src = dipole_from_pose(q_src, self.m_body)

        # magnetisation profile changes with wire_len
        m_local_fun = make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0)

        # warm start selection
        u_init = None
        L_start = self.L0_init

        if self._last["L"] is not None:
            # if L is close to last solved L, start there
            if abs(L - self._last["L"]) < 0.01 and self._last["u_init"] is not None:
                u_init = self._last["u_init"]
                L_start = min(self._last["L"], L)  # continue forward or backward conservatively

        # run continuation (internal dL controls speed/robustness)
        hist = solve_quasistatic_insertion(
            p0=self.p0_ur, q0=self.q0_ur,
            L0=L_start, Lf=L, dL=self.dL_internal,
            wire_len_fun=lambda LL: LL - self.mag_len,
            Kinv_fun=self.Kinv_fun, u_star=self.u_star,
            r_src=r_src, m_src=m_src,
            m_local_fun=m_local_fun, m_moment=0.0,
            lumen_C=self.lumen_C, lumen_R=self.lumen_R,
            N=self.N_nodes, maxiter=self.maxiter,
            use_lumen=self.use_lumen_jac,
            u_init=u_init
        )

        if not hist or ("p" not in hist[-1]) or (hist[-1]["p"] is None):
            return np.array([1e4, 1e4, 1e4], float)

        pE = hist[-1]["p"]          # (3,N)
        tip = pE[:, -1].copy()
        info = hist[-1].get("info", {})

        # ---- store for plotting/debug ----
        self.last_p_centerline = pE.copy()
        self.last_tip = tip.copy()
        self.last_info = info
        self.last_hist = hist  # optional (can be big)

        self._last["p_centerline"] = pE.copy()
        self._last["info"] = info
        self._last["hist"] = hist
        
        # your solve_quasistatic_insertion stores warm-start in info["u_flat_opt"] (per your earlier code)
        # warm-start extraction: try common keys
        u_next = None
        for key in ("u_flat_opt", "u_opt_flat", "u_ctrl_opt", "u_opt", "u_init_next"):
            if key in info and info[key] is not None:
                u_next = info[key]
                break
        self._last.update(p=p.copy(), L=L, u_init=u_next, tip=tip.copy())
        return tip