import numpy as np
from scipy.spatial.transform import Rotation as Rot
import copy
from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    dipole_from_pose, solve_quasistatic_insertion
)
from beam_direction_magnetisation.magnetism.beam_geometry import make_m_local_fun_wire_tip
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_to_rotvec, unit
from proper_research.parameters import default_beam_params
beam_params = default_beam_params()

mu_tip = beam_params.mag * beam_params.A_cs

def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

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
class  EnergyMinForwardWithLumen:
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

        # print("\n[FWD] p =", p)
        # print("[FWD] r_src =", r_src)
        # print("[FWD] rvec =", rvec)
        # print("[FWD] L_ins =", L_ins)

        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=self.L_tip_full,
            L_tip_min=self.L_tip_min,
        )
        # print("[FWD] lengths:",
        #     "L_model =", L_model,
        #     "wire_len =", wire_len,
        #     "tip_len =", tip_len)

        # magnet pose -> world dipole
        q_src = self._quat_from_rotvec_ur(rvec)
        m_src = dipole_from_pose(q_src, self.m_body)
        # print("[FWD] q_src =", q_src)
        # print("[FWD] m_src =", m_src)
        # magnetisation profile: magnetised region is [wire_len, wire_len + tip_len]
        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3
        )
        s_dbg = np.linspace(0.0, L_model, 5)
        m_dbg = m_local_fun(s_dbg, None)
        # print("[M_LOCAL DEBUG]")
        # print("s_dbg =", s_dbg)
        # print("m_local magnitudes =", np.linalg.norm(m_dbg, axis=0))
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
            print("[FWD] lumen_C shape =", self.lumen_C.shape)
        # print("[FWD] lumen_R shape =", self.lumen_R.shape)
        # print("[FWD] N_nodes =", self.N_nodes, "maxiter =", self.maxiter)
        # print("[FWD] use_lumen_jac =", self.use_lumen_jac)
        # print("\n[FWD PARAM DEBUG]")
        # print("L_ins =", L_ins)
        # print("L_model =", L_model)
        # print("wire_len =", wire_len)
        # print("tip_len =", tip_len)

        # print("\n--- MAGNETICS ---")
        # print("m_body (source body dipole) =", self.m_body)
        # print("m_src  (world source dipole) =", m_src)
        # print("mu_tip depends on beam_params.mag =", mu_tip)

        # print("\n--- FUNCTIONS ---")
        # print("Kinv_fun =", self.Kinv_fun)
        # print("m_local_fun object =", m_local_fun)

        # check if m_local_fun is actually valid
        if m_local_fun is None:
            print("m_local_fun STATUS: NONE (NOT USED)")
        else:
            # print("m_local_fun STATUS: ACTIVE")

            # quick probe of magnetisation along the beam
            s_dbg = np.linspace(0.0, L_model, 5)
            m_dbg = m_local_fun(s_dbg, None)
            # print("m_local magnitudes along s =", np.linalg.norm(m_dbg, axis=0))

        # print("m_moment passed to solver =", 0.0)

        # stiffness check
        s_test = np.array([0.0, wire_len * 0.5, wire_len + 0.5 * tip_len], dtype=float)
        Ktest = self.Kinv_fun(s_test, wire_len)

        # print("\n--- KINV DEBUG ---")
        # for i, ss in enumerate(s_test):
        #     print(
        #         f"s={ss:.4f} "
        #         f"Kinv_tors={Ktest[0,0,i]:.6e} "
        #         f"Kinv_b1={Ktest[1,1,i]:.6e} "
        #         f"Kinv_b2={Ktest[2,2,i]:.6e}"
        #     )
        hist = solve_quasistatic_insertion(
            p0=self.p0_ur, q0=self.q0_ur,
            L0=L_start, Lf=L_model, dL=self.dL_internal,
            wire_len_fun=wire_len_fun,
            tip_len_fun=tip_len_fun,
            Kinv_fun=self.Kinv_fun, u_star=self.u_star,
            r_src=r_src, m_src=m_src,
            m_local_fun=m_local_fun, m_moment=0.0,
            lumen_C=self.lumen_C, lumen_R=self.lumen_R,
            N=self.N_nodes, maxiter=self.maxiter,
            use_lumen=self.use_lumen_jac,
            u_init=u_init,
            debug=True
        )
        if (not hist) or (hist[-1].get("p", None) is None):
            print("[FWD] solve failed")
            return np.array([1e4, 1e4, 1e4], float)

        pE = hist[-1]["p"]
        tip = pE[:, -1].copy()
        info = hist[-1].get("info", {})
        parts = info.get("parts", {})
        # print("\n[COSSERAT ENERGY DEBUG]")
        # print(f"W total = {info.get('W', np.nan):.6e}")
        # print(f"W_el    = {parts.get('W_el', np.nan):.6e}")
        # print(f"W_b     = {parts.get('W_b', np.nan):.6e}")
        # print(f"W_t     = {parts.get('W_t', np.nan):.6e}")
        # print(f"W_m     = {parts.get('W_m', np.nan):.6e}")
        # print(f"W_g     = {parts.get('W_g', np.nan):.6e}")
        # print(f"W_cf    = {parts.get('W_cf', np.nan):.6e}")

        # if "Bnorm" in parts:
        #     print(f"max |B| = {np.max(parts['Bnorm']):.6e}")
        # if "mnorm" in parts:
        #     print(f"max |m| = {np.max(parts['mnorm']):.6e}")
        # if "tau_norm" in parts:
        #     print(f"max |m x B| = {np.max(parts['tau_norm']):.6e}")
        # if "angle_deg" in parts:
        #     print(f"mean angle(m,B) [deg] = {np.mean(parts['angle_deg']):.6f}")
        # print("[FWD] tip =", tip)
        # print("[FWD] info keys =", list(info.keys()))

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
        # self._last.update(p=p.copy(), L=L_model, u_init=u_next, tip=tip.copy())
        self._last.update(
            p=p.copy(),
            L=L_model,
            u_init=u_next,
            tip=tip.copy(),
            p_centerline=pE.copy(),
            info=info,
            hist=hist,
        )
        # print("[DBG-fwd] r_src:", r_src, "L_ins:", L_ins, "tip:", tip)

        return tip
class DeterministicForward6D:
    """
    Wraps EnergyMinForwardWithLumen to guarantee:
      - within a step: all evals start from identical internal state
      - optional commit at end of step updates the baseline warm-start
    Returns y = [tip_xyz(3), tip_tangent(3)].
    """

    def __init__(self, forward_model):
        self.fm = forward_model
        self._base = None  # snapshot used within current step
        self.last_p_centerline = None
        self.last_tip = None

    def _snapshot(self):
        fm = self.fm
        return dict(
            _last=copy.deepcopy(getattr(fm, "_last", None)),
            last_p_centerline=None if fm.last_p_centerline is None else fm.last_p_centerline.copy(),
            last_tip=None if fm.last_tip is None else fm.last_tip.copy(),
            last_info=copy.deepcopy(getattr(fm, "last_info", None)),
            last_hist=getattr(fm, "last_hist", None),  # might be big; shallow is fine unless you mutate it
        )

    def _restore(self, snap):
        fm = self.fm
        fm._last = copy.deepcopy(snap["_last"])
        fm.last_p_centerline = None if snap["last_p_centerline"] is None else snap["last_p_centerline"].copy()
        fm.last_tip = None if snap["last_tip"] is None else snap["last_tip"].copy()
        fm.last_info = copy.deepcopy(snap["last_info"])
        fm.last_hist = snap["last_hist"]

    def start_step(self):
        """Freeze the forward model warm-start state for this MPC step."""
        self._base = self._snapshot()

    def _eval_pose8_once(self, p8):
        """Evaluate forward model ONCE and build 6D y."""
        p7 = pose8_quat_to_pose7_rotvec(p8)
        # print("p8.L", p8[7], "p7.L", p7[6])
        # print("[DBG] p7.L =", float(np.asarray(p7).ravel()[6]))
        tip = self.fm(p7)  # should set fm.last_tip + fm.last_p_centerline
        x_tip = np.asarray(tip if tip is not None else self.fm.last_tip, float).reshape(3,)

        C = self.fm.last_p_centerline
        self.last_p_centerline = None if C is None else np.asarray(C, float).copy()
        self.last_tip = x_tip.copy()

        # tangent from end of centerline
        if self.last_p_centerline is None:
            t_tip = np.array([1.0, 0.0, 0.0], float)
        else:
            Cc = self.last_p_centerline
            if Cc.shape[0] == 3:
                p_end, p_prev = Cc[:, -1], Cc[:, -2]
            else:
                p_end, p_prev = Cc[-1, :], Cc[-2, :]
            t_tip = unit(p_end - p_prev)
            if np.linalg.norm(t_tip) < 1e-12:
                t_tip = np.array([1.0, 0.0, 0.0], float)

        return np.hstack([x_tip, t_tip])

    def __call__(self, p8, *, commit=False):
        """
        If commit=False: PURE evaluation (restores base before+after).
        If commit=True : updates base to post-eval (use for plant update once per step).
        """
        if self._base is None:
            # if user forgot, define a baseline anyway
            self._base = self._snapshot()

        # Always start from the step baseline
        self._restore(self._base)
        y = self._eval_pose8_once(p8)

        if commit:
            # adopt the new solver state as baseline for next calls/next step
            self._base = self._snapshot()
        else:
            # restore baseline so subsequent calls are identical
            self._restore(self._base)

        # also mirror attributes for MPC consumption
        self.last_p_centerline = None if self.last_p_centerline is None else self.last_p_centerline.copy()
        return y
import copy
import numpy as np

import copy
import numpy as np

class WarmForwardP8TipTangent:
    def __init__(self, fwd_model):
        self.fwd = fwd_model
        self.last_info = None
        self.last_p_centerline = None
        self.last_tip = None

    def reset_cache(self):
        self.last_info = None
        self.last_p_centerline = None
        self.last_tip = None

        if hasattr(self.fwd, "_last"):
            self.fwd._last = dict(
                p=None,
                L=None,
                u_init=None,
                tip=None,
                p_centerline=None,
                info=None,
                hist=None,
            )
        for name in ["last_info", "last_p_centerline", "last_tip", "last_hist"]:
            if hasattr(self.fwd, name):
                setattr(self.fwd, name, None)

    def clone(self):
        return WarmForwardP8TipTangent(copy.deepcopy(self.fwd))

    def _snapshot(self):
        snap = {
            "wrapper_last_info": copy.deepcopy(self.last_info),
            "wrapper_last_p_centerline": copy.deepcopy(self.last_p_centerline),
            "wrapper_last_tip": copy.deepcopy(self.last_tip),
        }
        for name in ["last_info", "last_p_centerline", "last_tip", "last_hist", "_last"]:
            if hasattr(self.fwd, name):
                snap[name] = copy.deepcopy(getattr(self.fwd, name))
        return snap

    def _restore(self, snap):
        self.last_info = copy.deepcopy(snap.get("wrapper_last_info", None))
        self.last_p_centerline = copy.deepcopy(snap.get("wrapper_last_p_centerline", None))
        self.last_tip = copy.deepcopy(snap.get("wrapper_last_tip", None))

        for k, v in snap.items():
            if k.startswith("wrapper_"):
                continue
            setattr(self.fwd, k, copy.deepcopy(v))

    def __call__(self, p8, commit=True):
        p7 = pose8_quat_to_pose7_rotvec(p8)

        if commit:
            tip = np.asarray(self.fwd(p7), float).reshape(3,)
            pcl = getattr(self.fwd, "last_p_centerline", None)
            info_eval = copy.deepcopy(getattr(self.fwd, "last_info", None))

            self.last_info = info_eval
            self.last_tip = tip.copy()
            self.last_p_centerline = None if pcl is None else copy.deepcopy(np.asarray(pcl, float))

        else:
            snap = self._snapshot()
            tip = None
            pcl = None
            info_eval = None
            try:
                tip = np.asarray(self.fwd(p7), float).reshape(3,)
                pcl = getattr(self.fwd, "last_p_centerline", None)
                info_eval = copy.deepcopy(getattr(self.fwd, "last_info", None))
            finally:
                self._restore(snap)

            # expose evaluated result on wrapper only; inner model remains restored
            self.last_tip = None if tip is None else tip.copy()
            self.last_info = info_eval
            self.last_p_centerline = None if pcl is None else copy.deepcopy(np.asarray(pcl, float))

        if self.last_p_centerline is None:
            raise RuntimeError("Tip tangent unavailable")

        pcl = np.asarray(self.last_p_centerline, float)
        if pcl.ndim != 2:
            raise RuntimeError(f"Unexpected centerline shape: {pcl.shape}")

        if pcl.shape[0] == 3 and pcl.shape[1] >= 2:
            pnm1 = pcl[:, -2]
            pn = pcl[:, -1]
        elif pcl.shape[1] == 3 and pcl.shape[0] >= 2:
            pnm1 = pcl[-2, :]
            pn = pcl[-1, :]
        else:
            raise RuntimeError(f"Unexpected centerline shape: {pcl.shape}")

        t_tip = pn - pnm1
        nt = np.linalg.norm(t_tip)
        if nt < 1e-12:
            raise RuntimeError("Degenerate tip tangent")
        t_tip = t_tip / nt

        return np.hstack([tip, t_tip])
class EnergyMinForwardWithAnalyticJac(EnergyMinForwardWithLumen):
    def jacobian_tip_pose7(self, p7, *, eps_theta=1e-6, eps_hess=1e-6):
        """
        Returns analytic/semi-analytic J_tip wrt:
            p7 = [r_src_x, r_src_y, r_src_z, rx, ry, rz, L_ins]

        Shape:
            J_tip: (3, 7)
        """
        p7 = np.asarray(p7, float).reshape(7,)

        # First make sure forward cache corresponds to this p7
        tip = self(p7)

        r_src = p7[0:3]
        rvec = p7[3:6]
        L_ins = float(p7[6])

        q_src = self._quat_from_rotvec_ur(rvec)

        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=self.L_tip_full,
            L_tip_min=self.L_tip_min,
        )

        info = self.last_info
        if info is None or "u_flat_opt" not in info:
            raise RuntimeError("No cached energy-min solution. Call forward first.")

        u_ref = np.asarray(info["u_flat_opt"], float).copy()

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

        theta0 = np.hstack([r_src, np.zeros(3), L_model])

        energy_grad_fun = make_energy_grad_fun_for_pose(
            p0=self.p0_ur,
            q0=self.q0_ur,
            q_src0=q_src,
            m_body=self.m_body,
            Kinv_fun=self.Kinv_fun,
            u_star=self.u_star,
            m_moment=0.0,
            N=self.N_nodes,
            rotation_convention="world",
            use_magnetic=True,
            use_contact=self.use_lumen_jac,
            lumen_query=LumenQuery(self.lumen_C, self.lumen_R) if self.use_lumen_jac else None,
        )

        def theta_builder(theta):
            return {"theta0": theta0}

        m_src = dipole_from_pose(q_src, self.m_body)

        J_tip, sens_info = energy_min_tip_jacobian_implicit(
            u_opt=u_ref,
            p0=self.p0_ur,
            q0=self.q0_ur,
            L=L_model,
            wire_len=wire_len,
            Kinv_fun=self.Kinv_fun,
            u_star=self.u_star,
            r_src=r_src,
            m_src=m_src,
            m_local_fun=m_local_fun,
            m_moment=0.0,
            N=self.N_nodes,
            theta_builder=theta_builder,
            energy_grad_fun=energy_grad_fun,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
        )

        self.last_J_tip_pose7 = J_tip.copy()
        self.last_sens_info = sens_info

        return J_tip

    def jacobian_tip_pose8(self, p8, *, eps_theta=1e-6, eps_hess=1e-6):
        """
        p8 = [x, y, z, qw, qx, qy, qz, L]

        Returns J_tip wrt p8 by converting through rotvec.

        This uses finite difference only for the quaternion -> rotvec chain,
        not for the energy-min solve.
        """
        p8 = np.asarray(p8, float).reshape(8,)
        p7 = pose8_quat_to_pose7_rotvec(p8)

        J_tip_p7 = self.jacobian_tip_pose7(
            p7,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
        )

        # chain rule: p7 = f(p8)
        A = np.zeros((7, 8), dtype=float)

        # translation
        A[0:3, 0:3] = np.eye(3)

        # length
        A[6, 7] = 1.0

        # quaternion -> rotvec numerical chain only
        eps_q = 1e-7
        for k in range(4):
            p8p = p8.copy()
            p8m = p8.copy()

            p8p[3 + k] += eps_q
            p8m[3 + k] -= eps_q

            # renormalise quaternion
            p8p[3:7] /= np.linalg.norm(p8p[3:7]) + 1e-12
            p8m[3:7] /= np.linalg.norm(p8m[3:7]) + 1e-12

            rv_p = pose8_quat_to_pose7_rotvec(p8p)[3:6]
            rv_m = pose8_quat_to_pose7_rotvec(p8m)[3:6]

            A[3:6, 3 + k] = (rv_p - rv_m) / (2.0 * eps_q)

        return J_tip_p7 @ A