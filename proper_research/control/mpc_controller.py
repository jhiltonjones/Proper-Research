import numpy as np
class mpc_controller_LTI_multi:
    def __init__(self, *, J_fn, 
                dt = 0.05, Np=10,
                w_th = 0.1, 
                w_u = (1e-4, 1e-6, 1e-4), 
                theta_max_band = 90, 
                u_max = (0.01, np.deg2rad(30), 0.01), 
                p_min = (0.008, np.deg2rad(-80), 0.02), 
                p_max = (0.2, np.deg2rad(80), 0.06)):
        self.J_fn = J_fn
        self.dt = dt
        self.Np = Np
        self.Q = np.array([[w_th]])
        w_u = np.asarray(w_u).ravel()
        self.R = np.diag(w_u)
        self.theta_max = np.deg2rad(theta_max_band)
        self.u_max = np.asarray(u_max).ravel()
        self.p_min = np.asarray(p_min).ravel()
        self.p_max = np.asarray(p_max).ravel()
        self.A = np.array([[1]])
        self.Qf =None
        self.p = None
        self._rebuild_S()
    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np)))*self.dt
    def set_dt(self, new_dt):
        self.dt = new_dt
        self._rebuild_S()
    def set_initial_params(self, B0, phi0, L0):
        self.p = np.array([B0, phi0, L0])
    def _build_lti_model(self):
        B0, phi0, L0 = self.p
        J = np.asarray(self.J_fn(B0, phi0, L0)).ravel()
        Bmat = self.dt * J.reshape(1,-1)
        _, P = dare_stabilising_K(self.A, Bmat, self.Q, self.R)



