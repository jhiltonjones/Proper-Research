import numpy as np
from scipy.spatial.transform import Rotation as Rot

from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    dipole_from_pose, solve_quasistatic_insertion
)
from beam_direction_magnetisation.magnetism.beam_geometry import make_m_local_fun_wire_tip

# You already have: unpack_pose_ur_rotvec_L(p)
# and wire_len_from_L(L, L_mag)
def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
    """
    L_ins      : commanded insertion (what MPC tracks)
    L_tip_full : physical magnetic tip length (4 cm)
    L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

    Returns (L_model, wire_len, tip_len)
    """
    L_ins = float(L_ins)

    # Magnetised tip inside grows with insertion until full tip is inside
    tip_len = min(L_ins, L_tip_full)

    # Wire is everything beyond the physical tip length
    wire_len = max(L_ins - L_tip_full, 0.0)

    # Total model length is the inserted length, but don't go below minimum model length
    L_model = max(L_ins, L_tip_min)

    # If we are below L_tip_min, we still model a minimum rod,
    # but magnetisation should NOT exceed what's actually inserted:
    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len
class EnergyMinForwardWithLumen:
    def __init__(self, *,
                 p0_ur, q0_ur,
                 Kinv_fun, u_star,
                 m_body,
                 lumen_C, lumen_R,
                 N_nodes=15, maxiter=15,
                 L0_init=0.009, dL_internal=0.002,
                 use_lumen_jac=True,
                 L_tip_full=0.04,
                 L_tip_min=0.01):
        self.p0_ur = np.asarray(p0_ur, float)
        self.q0_ur = np.asarray(q0_ur, float)
        self.L_tip_full  = float(L_tip_full)
        self.L_tip_min = float(L_tip_min)
        self.Kinv_fun = Kinv_fun
        self.u_star = np.asarray(u_star, float).reshape(3,)

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
        L_ins = float(p[6])

        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=self.L_tip_full,
            L_tip_min=self.L_tip_min,   # rename your L_tip_model to L_tip_min for clarity
        )

        # magnet pose -> world dipole
        q_src = self._quat_from_rotvec_ur(rvec)
        m_src = dipole_from_pose(q_src, self.m_body)
        # magnetisation profile: magnetised region is [wire_len, wire_len + tip_len]
        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3
        )

        # warm start
        u_init = None
        L_start = max(self.L_tip_min, min(self.L0_init, L_model))
        if (self._last["L"] is not None) and (self._last["u_init"] is not None):
            if abs(L_model - self._last["L"]) < 0.01:
                u_init = self._last["u_init"]
                L_start = min(self._last["L"], L_model)

        def tip_len_fun(LL):
            # magnet grows with LL until full physical tip inside
            return min(float(LL), self.L_tip_full)

        def wire_len_fun(LL):
            # wire exists only after full tip is inside
            return max(float(LL) - self.L_tip_full, 0.0)
        
        hist = solve_quasistatic_insertion(
            p0=self.p0_ur, q0=self.q0_ur,
            L0=L_start, Lf=L_model, dL=self.dL_internal,
            wire_len_fun=wire_len_fun,
            tip_len_fun=tip_len_fun,
            Kinv_fun=self.Kinv_fun, u_star=self.u_star,
            r_src=r_src, m_src=m_src,
            m_local_fun=None, m_moment=0.0,   # ignored because we rebuild it each step
            lumen_C=self.lumen_C, lumen_R=self.lumen_R,
            N=self.N_nodes, maxiter=self.maxiter,
            use_lumen=self.use_lumen_jac,
            u_init=u_init,
            debug=False
        )
        if (not hist) or (hist[-1].get("p", None) is None):
            return np.array([1e4, 1e4, 1e4], float)

        pE = hist[-1]["p"]          # (3,N)
        tip = pE[:, -1].copy()
        info = hist[-1].get("info", {})

        # store for plotting/debug
        self.last_p_centerline = pE.copy()
        self.last_tip = tip.copy()
        self.last_info = info
        self.last_hist = hist

        # warm-start extraction (optional but recommended)
        u_next = None
        for key in ("u_flat_opt", "u_opt_flat", "u_ctrl_opt", "u_opt", "u_init_next"):
            if key in info and info[key] is not None:
                u_next = info[key]
                break
        self._last.update(p=p.copy(), L=L_model, u_init=u_next, tip=tip.copy())
        # print("[DBG-fwd] r_src:", r_src, "L_ins:", L_ins, "tip:", tip)

        return tip