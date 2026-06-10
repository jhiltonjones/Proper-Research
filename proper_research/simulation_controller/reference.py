import numpy as np


class ReferenceMixin:
    def _select_reference_indices(self, x_now):
        """
        Select centreline reference indices for the MPC horizon.

        The first reference point is chosen as:

            closest centreline point to current tip
            + optional lookahead

        Then future reference points are spaced by ref_stride_pts.

        Required attributes:
            self.lumen_C: (M, >=3) centreline array
            self.Np: horizon length
            self.n: output dimension

        Optional attributes:
            self.use_xy_ref_distance: bool
            self.ref_lookahead_pts: int
            self.allow_ref_backward: bool
            self.ref_stride_pts: int
            self.i_ref_last: int
        """
        if not hasattr(self, "lumen_C"):
            raise AttributeError(
                "Controller must have self.lumen_C before calling step()."
            )

        Cc = np.asarray(self.lumen_C, float)

        if Cc.ndim != 2 or Cc.shape[1] < 3:
            raise ValueError(
                f"lumen_C must have shape (M, >=3), got {Cc.shape}."
            )

        M = Cc.shape[0]

        if M == 0:
            raise ValueError("lumen_C is empty.")

        x_now = np.asarray(x_now, float).reshape(-1)

        if x_now.size < 3:
            raise ValueError(
                f"x_now must have at least 3 entries, got {x_now.size}."
            )

        tip = x_now[:3]

        use_xy = bool(getattr(self, "use_xy_ref_distance", True))

        if use_xy:
            d_all = np.linalg.norm(
                Cc[:, :2] - tip[:2].reshape(1, 2),
                axis=1,
            )
        else:
            d_all = np.linalg.norm(
                Cc[:, :3] - tip[:3].reshape(1, 3),
                axis=1,
            )

        i_closest = int(np.argmin(d_all))

        lookahead = int(getattr(self, "ref_lookahead_pts", 0))
        stride = int(getattr(self, "ref_stride_pts", 1))

        if stride <= 0:
            raise ValueError("ref_stride_pts must be positive.")

        i_ref = int(np.clip(i_closest + lookahead, 0, M - 1))

        allow_backward = bool(getattr(self, "allow_ref_backward", False))
        i_prev = int(getattr(self, "i_ref_last", 0))

        if not allow_backward:
            i_ref = max(i_ref, i_prev)

        i_ref = int(np.clip(i_ref, 0, M - 1))

        idx_ref = np.clip(
            i_ref + stride * np.arange(self.Np),
            0,
            M - 1,
        ).astype(int)

        self.i_ref_last = int(i_ref)
        self.idx_ref_last = idx_ref.copy()
        self.i_closest_last = int(i_closest)
        self.dist_to_ref_last = float(d_all[i_ref])
        self.dist_to_closest_last = float(d_all[i_closest])

        return idx_ref