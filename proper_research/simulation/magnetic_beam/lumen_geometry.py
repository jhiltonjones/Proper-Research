from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .contact import LumenQuery


@dataclass(frozen=True)
class LumenBend:
    """
    One smooth centreline turn about a fixed world-frame axis.

    ``bend_start`` and ``bend_end`` are arclength positions in metres.
    ``bend_angle_rad`` is the total tangent rotation accumulated across that
    interval.
    """

    bend_angle_rad: float
    bend_start: float
    bend_end: float
    bend_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validate(self, *, lumen_length: float) -> None:
        axis = np.asarray(self.bend_axis, float).reshape(3)
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-12:
            raise ValueError("bend_axis must be a finite, nonzero 3-vector.")
        if not np.isfinite(self.bend_angle_rad):
            raise ValueError("bend_angle_rad must be finite.")
        if not 0.0 <= self.bend_start < self.bend_end <= lumen_length:
            raise ValueError(
                "Each bend must satisfy "
                "0 <= bend_start < bend_end <= lumen length."
            )


@dataclass(frozen=True)
class LumenConfig:
    """
    Configurable circular lumen geometry.

    A straight lumen has ``bends=()``. Multiple bends may use different axes,
    allowing planar S-shapes or fully three-dimensional paths.

    ``radius_profile`` contains optional ``(arclength_m, radius_m)`` control
    points. The radius is linearly interpolated between them and held constant
    outside their range. Leave it empty for a constant ``radius``.
    """

    length: float = 0.12
    n_pts: int = 160
    n_ref_pts: int = 80
    radius: float = 0.010
    bends: tuple[LumenBend, ...] = ()
    radius_profile: tuple[tuple[float, float], ...] = ()

    def validate(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("Lumen length must be positive and finite.")
        if self.n_pts < 2:
            raise ValueError("n_pts must be at least two.")
        if self.n_ref_pts < 2:
            raise ValueError("n_ref_pts must be at least two.")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Lumen radius must be positive and finite.")

        for bend in self.bends:
            if not isinstance(bend, LumenBend):
                raise TypeError("Every bends entry must be a LumenBend.")
            bend.validate(lumen_length=float(self.length))

        if self.radius_profile:
            profile = np.asarray(self.radius_profile, float)
            if profile.ndim != 2 or profile.shape[1] != 2:
                raise ValueError(
                    "radius_profile must contain (arclength, radius) pairs."
                )
            if not np.all(np.isfinite(profile)):
                raise ValueError("radius_profile contains non-finite values.")
            if np.any(profile[:, 0] < 0.0) or np.any(profile[:, 0] > self.length):
                raise ValueError(
                    "radius_profile arclengths must lie inside [0, length]."
                )
            if np.any(profile[:, 1] <= 0.0):
                raise ValueError("All radius_profile radii must be positive.")
            if np.any(np.diff(profile[:, 0]) <= 0.0):
                raise ValueError(
                    "radius_profile arclengths must be strictly increasing."
                )


def _unit(v: np.ndarray, *, name: str) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    norm = float(np.linalg.norm(v))
    if not np.all(np.isfinite(v)) or norm < 1e-12:
        raise ValueError(f"{name} must be a finite, nonzero 3-vector.")
    return v / norm


def _smoothstep(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rodrigues_rotate(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate one vector about an axis using Rodrigues' formula."""
    vector = np.asarray(vector, float).reshape(3)
    axis = _unit(axis, name="rotation axis")
    angle = float(angle)

    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def make_lumen_centerline_multi_turning(
    p_start: np.ndarray,
    t0: np.ndarray,
    *,
    length: float,
    n_pts: int,
    bends: tuple[LumenBend, ...] = (),
) -> np.ndarray:
    """
    Integrate a lumen centreline whose tangent undergoes multiple smooth turns.

    Bend axes are expressed in the world frame. Bends are applied in tuple
    order, so overlapping nonparallel bends also have a defined composition.
    Midpoint integration preserves the requested arclength accurately.
    """
    p_start = np.asarray(p_start, float).reshape(3)
    t0 = _unit(t0, name="initial tangent")
    length = float(length)
    n_pts = int(n_pts)

    if not np.all(np.isfinite(p_start)):
        raise ValueError("p_start must be finite.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if n_pts < 2:
        raise ValueError("n_pts must be at least two.")

    for bend in bends:
        bend.validate(lumen_length=length)

    s = np.linspace(0.0, length, n_pts)
    C = np.empty((n_pts, 3), dtype=float)
    C[0] = p_start

    for i, ds in enumerate(np.diff(s)):
        s_mid = 0.5 * (s[i] + s[i + 1])
        tangent = t0.copy()

        for bend in bends:
            progress = _smoothstep(
                (s_mid - bend.bend_start)
                / (bend.bend_end - bend.bend_start)
            )
            tangent = rodrigues_rotate(
                tangent,
                np.asarray(bend.bend_axis, float),
                bend.bend_angle_rad * float(progress),
            )

        tangent = _unit(tangent, name="integrated tangent")
        C[i + 1] = C[i] + float(ds) * tangent

    return C


def make_lumen_centerline_turning(
    p_start: np.ndarray,
    t0: np.ndarray,
    *,
    length: float = 0.12,
    n_pts: int = 60,
    bend_axis: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    bend_angle: float = np.deg2rad(25.0),
    bend_start: float = 0.02,
    bend_end: float = 0.09,
) -> np.ndarray:
    """Compatibility wrapper for a single smooth bend."""
    bend = LumenBend(
        bend_axis=tuple(np.asarray(bend_axis, float).reshape(3)),
        bend_angle_rad=float(bend_angle),
        bend_start=float(bend_start),
        bend_end=float(bend_end),
    )
    return make_lumen_centerline_multi_turning(
        p_start,
        t0,
        length=length,
        n_pts=n_pts,
        bends=(bend,),
    )


def resample_polyline_fixed_count(
    centreline: np.ndarray,
    *,
    n_ref_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at uniformly spaced arclength positions."""
    C = np.asarray(centreline, float)
    n_ref_pts = int(n_ref_pts)

    if C.ndim != 2 or C.shape[1] != 3 or C.shape[0] < 2:
        raise ValueError(
            f"centreline must have shape (N, 3), N >= 2; got {C.shape}."
        )
    if n_ref_pts < 2:
        raise ValueError("n_ref_pts must be at least two.")
    if not np.all(np.isfinite(C)):
        raise ValueError("centreline contains non-finite values.")

    segment_lengths = np.linalg.norm(np.diff(C, axis=0), axis=1)
    keep = np.hstack([[True], segment_lengths > 1e-12])
    C_unique = C[keep]

    if C_unique.shape[0] < 2:
        raise ValueError("centreline has zero total length.")

    s_raw = np.hstack(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(C_unique, axis=0), axis=1))]
    )
    s_path = np.linspace(0.0, float(s_raw[-1]), n_ref_pts)
    C_resampled = np.column_stack(
        [np.interp(s_path, s_raw, C_unique[:, axis]) for axis in range(3)]
    )
    return C_resampled, s_path


def build_lumen_geometry(
    *,
    p0_ur_lumen: np.ndarray,
    q0_ur_lumen: np.ndarray,
    lumen_cfg: LumenConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build centreline and radius arrays from a world pose and ``LumenConfig``.

    Quaternion convention is ``[qw, qx, qy, qz]``. Its local negative x-axis
    defines the initial lumen tangent, matching the undeformed beam convention.
    """
    lumen_cfg.validate()

    p0 = np.asarray(p0_ur_lumen, float).reshape(3)
    q0 = np.asarray(q0_ur_lumen, float).reshape(4)
    if not np.all(np.isfinite(p0)):
        raise ValueError("p0_ur_lumen must be finite.")
    if not np.all(np.isfinite(q0)) or np.linalg.norm(q0) < 1e-12:
        raise ValueError("q0_ur_lumen must be a finite, nonzero quaternion.")

    q0 = q0 / np.linalg.norm(q0)
    R0 = Rot.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])

    lumen_C_raw = make_lumen_centerline_multi_turning(
        p_start=p0,
        t0=t0,
        length=float(lumen_cfg.length),
        n_pts=int(lumen_cfg.n_pts),
        bends=lumen_cfg.bends,
    )
    lumen_C, s_path = resample_polyline_fixed_count(
        lumen_C_raw,
        n_ref_pts=int(lumen_cfg.n_ref_pts),
    )

    if lumen_cfg.radius_profile:
        profile = np.asarray(lumen_cfg.radius_profile, float)
        lumen_R = np.interp(
            s_path,
            profile[:, 0],
            profile[:, 1],
            left=float(profile[0, 1]),
            right=float(profile[-1, 1]),
        )
    else:
        lumen_R = np.full(lumen_C.shape[0], float(lumen_cfg.radius))

    lumen_path = lumen_C.copy()
    return lumen_C, lumen_R, lumen_path, s_path


def build_lumen_query(
    *,
    p0_ur_lumen: np.ndarray,
    q0_ur_lumen: np.ndarray,
    lumen_cfg: LumenConfig,
) -> LumenQuery:
    """Build the geometry and wrap it in the contact-query representation."""
    lumen_C, lumen_R, _, _ = build_lumen_geometry(
        p0_ur_lumen=p0_ur_lumen,
        q0_ur_lumen=q0_ur_lumen,
        lumen_cfg=lumen_cfg,
    )
    return LumenQuery(lumen_C, lumen_R)
