import numpy as np

# Adjust these imports to match your project layout.
# Example:
# from your_geometry_file import integrate_pose8_body, quat_wxyz_normalize
from proper_research.simulation_controller.geometry import integrate_pose8_body, quat_wxyz_normalize


class RolloutMixin:
    def _clamp_p(self, p):
        """
        Clamp pose/state parameters to valid bounds.

        Expected pose layout:
            p[0:3]   = magnet position
            p[3:7]   = quaternion, wxyz
            p[7]     = insertion length
        """
        p = np.asarray(p, float).reshape(-1).copy()

        if p.size != self.np:
            raise ValueError(f"p has length {p.size}, expected {self.np}.")

        if self.np < 8:
            raise ValueError("_clamp_p assumes pose dimension np >= 8.")

        p_min = np.asarray(self.p_min, float).reshape(self.np)
        p_max = np.asarray(self.p_max, float).reshape(self.np)

        # Clamp position.
        p[0:3] = np.minimum(np.maximum(p[0:3], p_min[0:3]), p_max[0:3])

        # Clamp insertion length.
        p[7] = float(np.minimum(np.maximum(p[7], p_min[7]), p_max[7]))

        # Normalise quaternion.
        p[3:7] = quat_wxyz_normalize(p[3:7])

        return p

    def _p_nodes_from_U(self, p0, U_seq):
        """
        Return pose nodes [p0, p1, ..., p_Np].

        p_nodes[k] is the pre-control pose for u_k.
        p_nodes[k+1] is the pose after applying u_k.
        """
        p0 = np.asarray(p0, float).reshape(self.np)
        U_seq = np.asarray(U_seq, float).reshape(self.Np, self.m)

        p_running = p0.copy()
        p_nodes = [p_running.copy()]

        for k in range(self.Np):
            u = U_seq[k].reshape(self.m)

            p_running = integrate_pose8_body(p_running, u, self.dt)
            p_running = self._clamp_p(p_running)

            p_nodes.append(p_running.copy())

        return np.asarray(p_nodes, float)

    def _p_seq_from_U(self, p0, U_seq):
        """
        Return post-control poses [p1, ..., p_Np].
        """
        return self._p_nodes_from_U(p0, U_seq)[1:]

    def _make_initial_U_guess(self):
        """
        Initial nominal sequence for SQP/LTV.

        By default, seed only insertion dL so the LTV nominal rollout is not
        completely stationary.
        """
        U = np.zeros((self.Np, self.m), float)

        dL_idx = int(getattr(self, "dL_index", 6))

        if not 0 <= dL_idx < self.m:
            raise ValueError(f"dL_index={dL_idx} outside control dimension m={self.m}.")

        dL_seed = float(getattr(self, "dL_guess", 0.0))

        if dL_seed == 0.0:
            umax = np.asarray(getattr(self, "u_max", np.full(self.m, np.inf)), float)

            if umax.size == self.m and np.isfinite(umax[dL_idx]):
                dL_seed = 0.25 * float(umax[dL_idx])
            else:
                dL_seed = 1e-3

        umax = np.asarray(getattr(self, "u_max", np.full(self.m, np.inf)), float)

        if umax.size == self.m and np.isfinite(umax[dL_idx]):
            dL_seed = float(np.clip(dL_seed, -umax[dL_idx], umax[dL_idx]))

        U[:, dL_idx] = dL_seed

        return U

    def _eval_forward_tip(self, p):
        """
        Evaluate forward_tip_fn robustly.

        Some wrappers accept commit=False; others may not.
        """
        try:
            return np.asarray(
                self.forward_tip_fn(p, commit=False),
                float,
            ).reshape(self.n)
        except TypeError:
            return np.asarray(
                self.forward_tip_fn(p),
                float,
            ).reshape(self.n)

    def _apply_control_sequence(self, U_seq, n_apply):
        """
        Apply the first n_apply controls from U_seq.

        This is the multi-step MPC implementation.
        """
        U_seq = np.asarray(U_seq, float).reshape(self.Np, self.m)
        n_apply = int(np.clip(n_apply, 1, self.Np))

        if self.p is None:
            raise ValueError("Cannot apply controls before self.p is initialised.")

        p_hist = []
        x_hist = []
        u_hist = []

        p = self.p.copy()

        for k in range(n_apply):
            u = U_seq[k].copy()

            p = integrate_pose8_body(p, u, self.dt)
            p = self._clamp_p(p)

            x = self._eval_forward_tip(p)

            p_hist.append(p.copy())
            x_hist.append(x.copy())
            u_hist.append(u.copy())

        self.p = p_hist[-1].copy()
        self.x = x_hist[-1].copy()

        return {
            "U_applied": np.asarray(u_hist, float),
            "p_rollout": np.asarray(p_hist, float),
            "x_rollout": np.asarray(x_hist, float),
        }

    def _shift_warm_start(self, U_seq, n_apply):
        """
        Shift the optimal sequence after applying n_apply controls.

        Example:
            U_seq = [u0, u1, u2, u3]
            n_apply = 2

        New warm start:
            [u2, u3, u3, u3]

        This is usually smoother than appending a fresh initial guess.
        """
        U_seq = np.asarray(U_seq, float).reshape(self.Np, self.m)
        n_apply = int(np.clip(n_apply, 1, self.Np))

        if n_apply < self.Np:
            tail = U_seq[n_apply:, :]

            # Repeat the last optimised control to fill the horizon.
            fill_count = n_apply
            fill = np.tile(U_seq[-1:, :], (fill_count, 1))

            U_next = np.vstack([tail, fill])
        else:
            # If the entire sequence was applied, use the last control as the next guess.
            U_next = np.tile(U_seq[-1:, :], (self.Np, 1))

        self.U_warm = U_next.reshape(-1)