import numpy as np


class ReferenceMixin:
    def _select_reference_indices(self, x_now):
        if not hasattr(self, "lumen_C"):
            raise AttributeError(
                "Controller must have self.lumen_C."
            )

        C = np.asarray(self.lumen_C, float)[:, :3]
        x_now = np.asarray(x_now, float).reshape(-1)
        tip = x_now[:3]

        use_xy = bool(
            getattr(self, "use_xy_ref_distance", True)
        )

        if use_xy:
            d_all = np.linalg.norm(
                C[:, :2] - tip[:2],
                axis=1,
            )
        else:
            d_all = np.linalg.norm(
                C - tip,
                axis=1,
            )

        i_closest = int(np.argmin(d_all))

        # Physical centreline arc length.
        ds = np.linalg.norm(
            np.diff(C, axis=0),
            axis=1,
        )
        s = np.concatenate(([0.0], np.cumsum(ds)))

        lookahead_m = float(
            getattr(self, "ref_lookahead_m", 2.0e-4)
        )
        stride_m = float(
            getattr(self, "ref_stride_m", 2.0e-4)
        )

        if stride_m <= 0.0:
            raise ValueError(
                "ref_stride_m must be positive."
            )

        start_s = min(
            s[-1],
            s[i_closest] + lookahead_m,
        )

        # Initial point at or just beyond requested arc length.
        i_ref = int(
            np.searchsorted(s, start_s, side="left")
        )
        i_ref = int(np.clip(i_ref, 0, len(C) - 1))

        allow_backward = bool(
            getattr(self, "allow_ref_backward", False)
        )
        i_prev = int(
            getattr(self, "i_ref_last", 0)
        )

        if not allow_backward:
            i_ref = max(i_ref, i_prev)

        # Desired physical arc-length locations.
        target_s = (
            s[i_ref]
            + stride_m * np.arange(self.Np)
        )
        target_s = np.clip(target_s, 0.0, s[-1])

        idx_ref = np.searchsorted(
            s,
            target_s,
            side="left",
        )
        idx_ref = np.clip(
            idx_ref,
            0,
            len(C) - 1,
        ).astype(int)

        self.i_ref_last = int(i_ref)
        self.idx_ref_last = idx_ref.copy()
        self.i_closest_last = int(i_closest)
        self.dist_to_ref_last = float(d_all[i_ref])
        self.dist_to_closest_last = float(
            d_all[i_closest]
        )

        self.ref_target_s_last = target_s.copy()
        self.ref_actual_s_last = s[idx_ref].copy()
        self.ref_horizon_span_last = float(
            s[idx_ref[-1]] - s[idx_ref[0]]
        )

        return idx_ref