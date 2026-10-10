from __future__ import annotations

import hashlib

import numpy as np


class ReferenceMixin:
    """Reference generation shared by the debug and optimized controllers.

    ``reference_mode='point'`` preserves the legacy discrete-centreline
    reference. ``reference_mode='contouring'`` uses a continuous local path
    projection and produces centreline positions/tangents for the QP builder.

    The reference is geometric only. It never asks either the contact or the
    no-contact Jacobian which centreline point is reachable. This keeps the
    Jacobian comparison scientifically neutral.
    """

    # ------------------------------------------------------------------
    # Public state reset
    # ------------------------------------------------------------------
    def reset_path_reference_state(self) -> None:
        self.path_progress_s = 0.0
        self.path_progress_initialized = False
        self.active_path_reference = None
        self._active_path_reference_x0_key = None

        self.path_projection_s_last = np.nan
        self.path_projection_distance_last = np.nan
        self.path_projection_segment_last = -1
        self.path_projection_alpha_last = np.nan

        self.ref_s_last = None
        self.ref_points_last = None
        self.ref_tangents_last = None
        self.ref_segment_indices_last = None
        self.idx_ref_last = None

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------
    def _reference_mode(self) -> str:
        mode = str(getattr(self, "reference_mode", "point")).strip().lower()
        aliases = {
            "waypoint": "point",
            "waypoints": "point",
            "path": "contouring",
            "path_following": "contouring",
            "path-following": "contouring",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"point", "contouring"}:
            raise ValueError(
                "reference_mode must be 'point' or 'contouring'; "
                f"got {mode!r}."
            )
        return mode

    def _select_reference_indices(self, x_now):
        """Return legacy-compatible indices and freeze one reference per solve."""
        mode = self._reference_mode()
        if mode == "point":
            self.active_path_reference = None
            self._active_path_reference_x0_key = None
            return self._select_point_reference_indices(x_now)

        path_ref = self._build_path_reference(x_now)
        self.active_path_reference = path_ref
        self._active_path_reference_x0_key = tuple(
            np.asarray(x_now, dtype=float).reshape(-1).tolist()
        )

        # Existing controller/QP signatures still carry idx_ref. In contouring
        # mode the QP uses active_path_reference; these indices are retained for
        # logging and backwards-compatible debug code only.
        idx_ref = np.clip(
            np.asarray(path_ref["segment_indices"], dtype=int) + 1,
            0,
            len(self.lumen_C) - 1,
        )
        self.idx_ref_last = idx_ref.copy()
        return idx_ref

    # ------------------------------------------------------------------
    # Legacy point reference
    # ------------------------------------------------------------------
    def _select_point_reference_indices(self, x_now):
        if not hasattr(self, "lumen_C"):
            raise AttributeError("Controller must have self.lumen_C.")

        C = np.asarray(self.lumen_C, dtype=float)[:, :3]
        x_now = np.asarray(x_now, dtype=float).reshape(-1)
        tip = x_now[:3]

        use_xy = bool(getattr(self, "use_xy_ref_distance", False))
        if use_xy:
            d_all = np.linalg.norm(C[:, :2] - tip[:2], axis=1)
        else:
            d_all = np.linalg.norm(C - tip, axis=1)

        i_closest = int(np.argmin(d_all))
        s = self._arc_length_nodes(C)

        # Metric names are canonical. Old *_pts settings remain supported only
        # as an explicit compatibility fallback.
        if hasattr(self, "ref_lookahead_m"):
            lookahead_m = float(self.ref_lookahead_m)
        else:
            lookahead_pts = int(getattr(self, "ref_lookahead_pts", 1))
            median_ds = float(np.median(np.diff(s)))
            lookahead_m = lookahead_pts * median_ds

        if hasattr(self, "ref_stride_m"):
            stride_m = float(self.ref_stride_m)
        else:
            stride_pts = int(getattr(self, "ref_stride_pts", 1))
            median_ds = float(np.median(np.diff(s)))
            stride_m = stride_pts * median_ds

        if lookahead_m < 0.0:
            raise ValueError("ref_lookahead_m must be non-negative.")
        if stride_m <= 0.0:
            raise ValueError("ref_stride_m must be positive.")

        start_s = min(s[-1], s[i_closest] + lookahead_m)
        i_ref = int(np.searchsorted(s, start_s, side="left"))
        i_ref = int(np.clip(i_ref, 0, len(C) - 1))

        allow_backward = bool(getattr(self, "allow_ref_backward", False))
        i_prev = int(getattr(self, "i_ref_last", 0))
        if not allow_backward:
            i_ref = max(i_ref, i_prev)

        target_s = s[i_ref] + stride_m * np.arange(int(self.Np))
        target_s = np.clip(target_s, 0.0, s[-1])
        idx_ref = np.searchsorted(s, target_s, side="left")
        idx_ref = np.clip(idx_ref, 0, len(C) - 1).astype(int)

        self.i_ref_last = int(i_ref)
        self.idx_ref_last = idx_ref.copy()
        self.i_closest_last = int(i_closest)
        self.dist_to_ref_last = float(d_all[i_ref])
        self.dist_to_closest_last = float(d_all[i_closest])
        self.ref_target_s_last = target_s.copy()
        self.ref_actual_s_last = s[idx_ref].copy()
        self.ref_horizon_span_last = float(s[idx_ref[-1]] - s[idx_ref[0]])
        return idx_ref

    # ------------------------------------------------------------------
    # Cached continuous path geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _arc_length_nodes(C: np.ndarray) -> np.ndarray:
        ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
        if np.any(~np.isfinite(ds)) or np.any(ds <= 1.0e-12):
            raise ValueError(
                "Centreline must be finite and contain no duplicate consecutive points."
            )
        return np.concatenate(([0.0], np.cumsum(ds)))

    @staticmethod
    def _path_signature(C: np.ndarray) -> tuple:
        Cc = np.ascontiguousarray(C, dtype=np.float64)
        digest = hashlib.blake2b(Cc.view(np.uint8), digest_size=16).hexdigest()
        return Cc.shape, digest

    def _prepare_path_geometry(self) -> dict:
        if not hasattr(self, "lumen_C"):
            raise AttributeError("Controller must have self.lumen_C.")

        C = np.asarray(self.lumen_C, dtype=float)
        if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] < 3:
            raise ValueError(
                "lumen_C must have shape (M, >=3) with M >= 2; "
                f"got {C.shape}."
            )
        C = np.ascontiguousarray(C[:, :3])
        if not np.all(np.isfinite(C)):
            raise FloatingPointError("lumen_C contains non-finite values.")

        signature = self._path_signature(C)
        cached = getattr(self, "_path_geometry_cache", None)
        if isinstance(cached, dict) and cached.get("signature") == signature:
            return cached

        segment_vectors = np.diff(C, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        if np.any(segment_lengths <= 1.0e-12):
            raise ValueError("Centreline contains duplicate consecutive points.")

        segment_tangents = segment_vectors / segment_lengths[:, None]
        s_nodes = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        # Smooth node tangents reduce reference-direction jumps at polyline
        # vertices while leaving the centreline positions unchanged.
        node_tangents = np.empty_like(C)
        node_tangents[0] = segment_tangents[0]
        node_tangents[-1] = segment_tangents[-1]
        if C.shape[0] > 2:
            weighted = (
                segment_lengths[:-1, None] * segment_tangents[:-1]
                + segment_lengths[1:, None] * segment_tangents[1:]
            )
            norms = np.linalg.norm(weighted, axis=1, keepdims=True)
            node_tangents[1:-1] = weighted / np.maximum(norms, 1.0e-12)

        cached = {
            "signature": signature,
            "C": C,
            "s_nodes": s_nodes,
            "segment_vectors": segment_vectors,
            "segment_lengths": segment_lengths,
            "segment_tangents": segment_tangents,
            "node_tangents": node_tangents,
        }
        self._path_geometry_cache = cached
        return cached

    @staticmethod
    def _interpolate_path(geometry: dict, s_query: np.ndarray):
        C = geometry["C"]
        s_nodes = geometry["s_nodes"]
        node_tangents = geometry["node_tangents"]

        s_query = np.asarray(s_query, dtype=float).reshape(-1)
        s_query = np.clip(s_query, s_nodes[0], s_nodes[-1])
        segment_index = np.searchsorted(s_nodes, s_query, side="right") - 1
        segment_index = np.clip(segment_index, 0, len(s_nodes) - 2)

        s0 = s_nodes[segment_index]
        s1 = s_nodes[segment_index + 1]
        alpha = (s_query - s0) / np.maximum(s1 - s0, 1.0e-12)

        points = (
            (1.0 - alpha)[:, None] * C[segment_index]
            + alpha[:, None] * C[segment_index + 1]
        )
        tangents = (
            (1.0 - alpha)[:, None] * node_tangents[segment_index]
            + alpha[:, None] * node_tangents[segment_index + 1]
        )
        tangents /= np.maximum(
            np.linalg.norm(tangents, axis=1, keepdims=True),
            1.0e-12,
        )
        return points, tangents, segment_index, alpha

    @staticmethod
    def _project_tip_to_segments(
        *,
        tip: np.ndarray,
        geometry: dict,
        candidate_indices: np.ndarray,
    ) -> dict:
        C = geometry["C"]
        s_nodes = geometry["s_nodes"]
        candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if candidate_indices.size == 0:
            raise ValueError("At least one candidate path segment is required.")

        A = C[candidate_indices]
        AB = C[candidate_indices + 1] - A
        denominator = np.einsum("ij,ij->i", AB, AB)
        alpha = np.einsum("ij,ij->i", tip[None, :] - A, AB)
        alpha /= np.maximum(denominator, 1.0e-12)
        alpha = np.clip(alpha, 0.0, 1.0)

        projected = A + alpha[:, None] * AB
        distance_squared = np.einsum(
            "ij,ij->i", projected - tip[None, :], projected - tip[None, :]
        )
        best_local = int(np.argmin(distance_squared))
        segment = int(candidate_indices[best_local])
        alpha_best = float(alpha[best_local])
        s_projected = float(
            s_nodes[segment]
            + alpha_best * (s_nodes[segment + 1] - s_nodes[segment])
        )
        return {
            "s_projected": s_projected,
            "point_projected": projected[best_local].copy(),
            "segment_index": segment,
            "alpha": alpha_best,
            "distance_m": float(np.sqrt(distance_squared[best_local])),
        }

    def _project_tip_to_local_path(
        self,
        *,
        tip: np.ndarray,
        geometry: dict,
        s_previous: float,
    ) -> dict:
        n_segments = geometry["segment_lengths"].size
        initialized = bool(getattr(self, "path_progress_initialized", False))

        # The first projection is global so a simulation may start at any point
        # along the lumen. Every later projection is local to prevent branch
        # switching at self-crossings or spatially close centreline sections.
        if not initialized:
            candidate_indices = np.arange(n_segments, dtype=int)
        else:
            backward = float(getattr(self, "ref_search_backward_m", 5.0e-4))
            forward = float(getattr(self, "ref_search_forward_m", 5.0e-3))
            if backward < 0.0 or forward <= 0.0:
                raise ValueError(
                    "ref_search_backward_m must be non-negative and "
                    "ref_search_forward_m must be positive."
                )

            s_nodes = geometry["s_nodes"]
            s_low = max(float(s_nodes[0]), s_previous - backward)
            s_high = min(float(s_nodes[-1]), s_previous + forward)
            active = (s_nodes[1:] >= s_low) & (s_nodes[:-1] <= s_high)
            candidate_indices = np.flatnonzero(active)
            if candidate_indices.size == 0:
                nearest_segment = int(
                    np.clip(
                        np.searchsorted(s_nodes, s_previous, side="right") - 1,
                        0,
                        n_segments - 1,
                    )
                )
                candidate_indices = np.array([nearest_segment], dtype=int)

        return self._project_tip_to_segments(
            tip=np.asarray(tip, dtype=float).reshape(3),
            geometry=geometry,
            candidate_indices=candidate_indices,
        )

    # ------------------------------------------------------------------
    # Continuous contouring reference
    # ------------------------------------------------------------------
    def _build_path_reference(self, x_now) -> dict:
        geometry = self._prepare_path_geometry()
        x_now = np.asarray(x_now, dtype=float).reshape(-1)
        if x_now.size < 3:
            raise ValueError("x_now must contain at least tip_xyz.")
        tip = x_now[:3]

        s_nodes = geometry["s_nodes"]
        s_previous = float(getattr(self, "path_progress_s", 0.0))
        projection = self._project_tip_to_local_path(
            tip=tip,
            geometry=geometry,
            s_previous=s_previous,
        )

        max_advance = float(
            getattr(self, "ref_max_progress_per_step_m", 1.0e-3)
        )
        if max_advance <= 0.0:
            raise ValueError("ref_max_progress_per_step_m must be positive.")

        allow_backward = bool(getattr(self, "allow_ref_backward", False))
        if bool(getattr(self, "path_progress_initialized", False)):
            lower = (
                max(float(s_nodes[0]), s_previous - float(
                    getattr(self, "ref_max_backward_per_step_m", 0.0)
                ))
                if allow_backward
                else s_previous
            )
            upper = min(float(s_nodes[-1]), s_previous + max_advance)
            s_progress = float(np.clip(projection["s_projected"], lower, upper))
        else:
            s_progress = float(np.clip(projection["s_projected"], 0.0, s_nodes[-1]))

        lookahead = float(getattr(self, "ref_lookahead_m", 5.0e-4))
        stride = float(getattr(self, "ref_stride_m", 5.0e-4))
        if lookahead < 0.0:
            raise ValueError("ref_lookahead_m must be non-negative.")
        if stride <= 0.0:
            raise ValueError("ref_stride_m must be positive.")

        s_ref = s_progress + lookahead + stride * np.arange(int(self.Np))
        s_ref = np.clip(s_ref, s_nodes[0], s_nodes[-1])
        C_ref, T_ref, segment_indices, alpha = self._interpolate_path(
            geometry,
            s_ref,
        )

        self.path_progress_s = s_progress
        self.path_progress_initialized = True
        self.path_projection_s_last = float(projection["s_projected"])
        self.path_projection_distance_last = float(projection["distance_m"])
        self.path_projection_segment_last = int(projection["segment_index"])
        self.path_projection_alpha_last = float(projection["alpha"])
        self.ref_s_last = s_ref.copy()
        self.ref_points_last = C_ref.copy()
        self.ref_tangents_last = T_ref.copy()
        self.ref_segment_indices_last = segment_indices.copy()

        return {
            "s_progress": s_progress,
            "s_ref": s_ref,
            "C_ref": C_ref,
            "T_ref": T_ref,
            "segment_indices": segment_indices,
            "segment_alpha": alpha,
            "projection": projection,
        }
