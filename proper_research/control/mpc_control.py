import numpy as np 
import osqp
from scipy.linalg import solve_discrete_are, block_diag
import scipy.sparse as sp
from proper_research.models.beam_model import theta_angle_solved, dtheta_dB, dtheta_dL, dtheta_dphi
def dare_stabilising_K(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

def trust_radius(jac_fn, psi_rad, *,
                 h_rad=np.deg2rad(0.5),
                 eps_theta_rad=np.deg2rad(1.0),
                 Jmin=1e-6, Lmin=1e-6,
                 dpsi_cap=np.deg2rad(50.0)):
    J0 = float(jac_fn(psi_rad))
    Jp = float(jac_fn(psi_rad + h_rad))
    Jm = float(jac_fn(psi_rad - h_rad))
    Jprime = (Jp - Jm) / (2.0*h_rad)
    dpsi_lin  = eps_theta_rad / max(abs(J0),    Jmin)
    dpsi_quad = (2.0*eps_theta_rad / max(abs(Jprime), Lmin))**0.5
    dpsi = min(dpsi_lin, dpsi_quad, dpsi_cap)
    return dpsi

def seq_mat_tv(Phi_list, B_list):
    N = len(Phi_list)
    n,m = B_list[0].shape   
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))
    P = np.eye(n)

    for i in range(N):
        P = Phi_list[i] @ P
        Mx[i*n:(i+1)*n, :] = P
    
    Mc[0:n, 0:m] = B_list[0]

    for i in range(1,N):
        Mc[i*n:(i+1)*n, 0:i*m] = Phi_list[i] @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B_list[i]

    return Mx, Mc
    
def seq_mat_lti(A, B, N):

    n, m = B.shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    A_pow = np.eye(n)

    A_pow = A @ A_pow          
    Mx[0:n, :] = A_pow
    Mc[0:n, 0:m] = B
    for i in range(1, N):
        A_pow = A @ A_pow  
        Mx[i*n:(i+1)*n, :] = A_pow
        Mc[i*n:(i+1)*n, 0:i*m] = A @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B

    return Mx, Mc


def solve_qp_osqp(H,f,A, l, u, U_warm = None):
    P = sp.csc_matrix(0.5 * (H+H.T))
    q = f.astype(float)
    A = sp.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u)
    if U_warm is not None:
        prob.warm_start(x=U_warm)
    res = prob.solve()
    status = res.info.status
    if status not in ("solved", "solved inaccurate"):
        return None, None, status
    return res.x, res.y, status

def J_params_fn(B, phi, L, mag, A_cs, E, I):
    J_B   = dtheta_dB(B, phi, mag, A_cs, L, E, I)
    J_phi = dtheta_dphi(B, phi, mag, A_cs, L, E, I)
    J_L   = dtheta_dL(B, phi, mag, A_cs, L, E, I)
    return np.array([J_B, J_phi, J_L], dtype=float)

class mpc_controller_LTI_multi:

    def __init__(self, *, J_fn, dt=0.05, Np=10,
                 w_th=10.0, w_u=(5e-2, 5e-2, 5e-2),
                 theta_band_deg=10.0,
                 theta_max_deg=90,
                 u_max=(0.01, np.deg2rad(30), 0.01),  
                 p_min=(0.008, np.deg2rad(-80), 0.04), 
                 p_max=(0.020, np.deg2rad(80),  0.06)):

        self.J_fn = J_fn
        self.dt = float(dt)
        self.Np = int(Np)

        self.Q = np.array([[float(w_th)]])
        w_u = np.asarray(w_u, dtype=float).ravel()
        self.R = np.diag(w_u)              

        self.band_deg = float(theta_band_deg)
        self.theta_max = np.deg2rad(theta_max_deg)

        self.u_max = np.asarray(u_max, dtype=float).ravel() 
        self.p_min = np.asarray(p_min, dtype=float).ravel()   
        self.p_max = np.asarray(p_max, dtype=float).ravel()   

        self.A = np.array([[1.0]])       
        self.Qf = None

        self.p = None

        self._rebuild_S()

    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

    def set_dt(self, new_dt):
        self.dt = float(new_dt)
        self._rebuild_S()

    def set_initial_params(self, B0, phi0, L0):
        self.p = np.array([float(B0), float(phi0), float(L0)], dtype=float)

    def _build_lti_model(self):
        if self.p is None:
            raise ValueError("Call set_initial_params(B0, phi0, L0) before step().")

        B0, phi0, L0 = self.p

        J = np.asarray(self.J_fn(B0, phi0, L0), dtype=float).ravel()  # (3,)
        if J.size != 3:
            raise ValueError(f"Expected Jacobian length 3, got {J.size}.")

        Bmat = (self.dt * J.reshape(1, -1))  # (1,3)

        _, P = dare_stabilising_K(self.A, Bmat, self.Q, self.R)
        self.Qf = P

        Mx, Mc = seq_mat_lti(self.A, Bmat, self.Np)
        return J, Bmat, Mx, Mc

    def step(self, ref_seq_rad, theta_meas_rad):
        J, Bmat, Mx, Mc = self._build_lti_model()

        n = 1
        m = Bmat.shape[1]
        Np = self.Np

        Qtil = np.zeros((Np*n, Np*n))
        if Np > 1:
            Qtil[:(Np-1)*n,:(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
        Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf

        Rtil = np.kron(np.eye(Np), self.R)

        xk = np.array([[float(theta_meas_rad)]])
        xref_seq = np.asarray(ref_seq_rad, dtype=float).reshape(Np, 1)
        xref_stack = xref_seq.reshape(Np*n, 1)

        X0_stack = (Mx @ xk).reshape(Np*n, 1)

        H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil)
        f = 2.0 * (Mc.T @ Qtil @ (X0_stack - xref_stack))

        A_list, l_list, u_list = [], [], []

        if np.all(np.isfinite(self.u_max)):
            A_u = np.eye(Np * m)
            u_max_stack = np.tile(self.u_max, Np) 
            A_list.append(A_u)
            l_list.append(-u_max_stack)
            u_list.append(+u_max_stack)


        if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
            A_p = np.kron(self.S_np, np.eye(m)) 
            p0_stack = np.tile(self.p, Np)      

            l_p = np.tile(self.p_min, Np) - p0_stack
            u_p = np.tile(self.p_max, Np) - p0_stack

            A_list.append(A_p)
            l_list.append(l_p)
            u_list.append(u_p)

        if np.isfinite(self.band_deg):
            band = np.deg2rad(self.band_deg)

            band_vec = band * np.ones((Np * n, 1))   

            rhs_p = (band_vec + xref_stack - X0_stack).reshape(-1)
            rhs_n = (band_vec - xref_stack + X0_stack).reshape(-1)

            A_list.append(Mc)
            l_list.append(-np.inf * np.ones(Np * n))
            u_list.append(rhs_p)

            A_list.append(-Mc)
            l_list.append(-np.inf * np.ones(Np * n))
            u_list.append(rhs_n)


        if A_list:
            A_osqp = np.vstack(A_list)
            l_osqp = np.concatenate(l_list)
            u_osqp = np.concatenate(u_list)
        else:
            A_osqp = np.zeros((0, Np * m))
            l_osqp = np.zeros(0)
            u_osqp = np.zeros(0)

        U_opt, y, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=None)
        infeas = (status not in ("solved", "solved inaccurate")) or (U_opt is None)

        if infeas:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
            X_pred = np.full((Np, 1), np.nan)
        else:
            U_seq = np.asarray(U_opt, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]
            X_pred = (X0_stack + Mc @ U_opt.reshape(-1, 1)).reshape(Np, 1)

        self.p = self.p + u0 * self.dt

        self.p = np.minimum(np.maximum(self.p, self.p_min), self.p_max)

        info = dict(
            status=status,
            infeasible=int(infeas),
            u0=u0.copy(),
            p_now=self.p.copy(),
            J=J.copy(),
            Bmat=Bmat.copy(),
            xref_seq_rad=xref_seq.ravel().copy(),
            theta_pred_rad=X_pred.ravel().copy(),
            U_seq=U_seq.copy(),
        )
        return self.p.copy(), info


def make_J_fn(mag, A_cs, E, I):
    def J_fn(B, phi, L):
        return J_params_fn(B, phi, L, mag, A_cs, E, I)
    return J_fn

def fmt_vec(v, scale=1.0, fmt="{:+.3f}"):
    v = np.asarray(v).ravel() * scale
    return "[" + ", ".join(fmt.format(x) for x in v) + "]"

def rollout_debug(mpc, *, mag, A_cs, E, I,
                  theta_ref_traj_rad,
                  T=30,
                  print_horizon=5):

    Np = mpc.Np

    print("k | theta_meas(deg)  theta_ref(deg) |   B(mT)   phi(deg)    L(m) | "
          "u0=[Bdot, phidot, Ldot] | J=[dθ/dB, dθ/dφ, dθ/dL]")
    print("-"*130)

    for k in range(T):
        B, phi, L = mpc.p
        theta_meas = theta_angle_solved(B, phi, mag, A_cs, L, E, I)

        if np.isscalar(theta_ref_traj_rad):
            ref_now = float(theta_ref_traj_rad)
        else:
            ref_now = float(theta_ref_traj_rad[k])

        ref_seq = ref_now * np.ones(Np)

        p_cmd, info = mpc.step(ref_seq, theta_meas_rad=theta_meas)

        u0 = info["u0"]
        J = info.get("J", np.full(3, np.nan))
        theta_pred = info["theta_pred_rad"]
        U_seq = info["U_seq"]

        print(
            f"{k:2d} | {np.rad2deg(theta_meas):9.3f}       {np.rad2deg(ref_now):9.3f} | "
            f"{1e3*p_cmd[0]:7.3f}  {np.rad2deg(p_cmd[1]):8.3f}  {p_cmd[2]:7.4f} | "
            f"{fmt_vec(u0, scale=1.0, fmt='{:+.6f}')} | "
            f"{fmt_vec(J, scale=1.0, fmt='{:+.3e}')}"
        )

        th_preview = np.rad2deg(theta_pred[:print_horizon])
        u_preview = U_seq[:print_horizon, :]

        print("    pred theta(deg) first:", np.array2string(th_preview, precision=3, separator=", "))
        print("    planned u (first rows):")
        for i in range(min(print_horizon, U_seq.shape[0])):
            print(f"      i={i:2d}: {fmt_vec(u_preview[i], fmt='{:+.6f}')}")
        print()

def main():
    dt = .5
    Np = 4

    mag = 128e3
    r = 0.0015
    E = 5.4e6
    A_cs = np.pi * r**2
    I = np.pi * r**4/4

    B_init = 0.01
    phi_init = np.deg2rad(25)
    L_init = 0.04

    mpc = mpc_controller_LTI_multi(
        J_fn=make_J_fn(mag, A_cs, E, I),
        dt=dt, Np=Np,
        w_th=50.0,
        w_u=(1e-4, 1e-6, 1e-4),
        u_max=(0.005, np.deg2rad(5), 0.002),
        p_min=(0.008, np.deg2rad(-90), 0.03),
        p_max=(0.025, np.deg2rad(90),  0.06),
        theta_band_deg=20.0,
    )
    mpc.set_initial_params(B_init, phi_init, L_init)

    theta_ref = np.deg2rad(20)

    rollout_debug(
        mpc,
        mag=mag, A_cs=A_cs, E=E, I=I,
        theta_ref_traj_rad=theta_ref,
        T=4,
        print_horizon=4
    )

if __name__ == "__main__":
    main()

