"""Read-only camera/robot/model frame validation for the magnetic beam.

This script is deliberately separate from the motion controller.  It acquires
one stationary measurement, converts every measured quantity through one
calibrated camera-plane map, evaluates the forward model at the corresponding
source-magnet pose, and plots both results in the beam and robot frames.

There are no command-line arguments.  Edit only the configuration section.
No robot motion command is sent by this file.

Transform notation
------------------
``T_A_B`` maps coordinates expressed in frame B into frame A:

    p_A = T_A_B @ p_B

For example, ``T_R_M = T_R_TCP @ T_TCP_M`` maps source-magnet coordinates into
the robot-base frame.  Pixel coordinates are not an SE(3) frame; they enter the
chain only through ``PlanarPixelCalibration``.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import proper_research.hardware.beam_hardware_experiment_v2 as experiment_v2


# =============================================================================
# USER CONFIGURATION -- edit here; there are no terminal arguments
# =============================================================================


# Reuse the exact composite-beam, contact, camera-path and model settings used by
# the v2 experiment.  This validation script overrides only motion: it never
# calls send_step(), irrespective of the value in the imported object.
EXPERIMENT_CONFIG = copy.deepcopy(experiment_v2.CONFIG)
EXPERIMENT_CONFIG.send_commands = False


# ---------------------------------------------------------------------------
# PLACEHOLDER -- physical offset from the UR TCP to the source-magnet centre.
# The magnet is rigidly mounted BELOW the end effector by a fixed amount.
# The tool points down (pivot orientation ~pi about X), so "below the tool"
# is the TCP +Z direction.  MEASURE THIS and replace the value below.
#   e.g. magnet centre 30 mm below the TCP, body axes aligned with the TCP:
#        SOURCE_MAGNET_OFFSET_BELOW_TCP_M = 0.030
# Verify with the [DEBUG] "magnet - TCP in TCP frame" line printed at run
# time: it should read approximately [0.00 0.00 +<offset>] mm, and the 3-D
# plot should show the magnet box below the TCP triad.  Flip the sign here
# if the magnet ends up on the wrong side.
# ---------------------------------------------------------------------------
SOURCE_MAGNET_OFFSET_BELOW_TCP_M: float = 0.030  # legacy "straight below" placeholder
# Extra body-axis rotation of the magnet frame M relative to the TCP frame,
# as a rotation vector (rad).  Leave at zeros if the magnet body axes are
# mounted parallel to the TCP axes.
SOURCE_MAGNET_BODY_ROTVEC_IN_TCP: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Final frame calibration (2026-09-09, from direct measurement): magnet centre
# at beam_base + [-0.23, 0, 0] in R (== [0, 0, +230] mm in beam frame B: purely
# out of the vision plane, level with the base; near edge 15.5 cm, ~15 cm long).
# Flange->magnet translation in the flange frame at the reference orientation.
# Keep in sync with simulation.simulations.initial_conditions.TCP_TO_MAGNET_POSE6.
SOURCE_MAGNET_T_TCP_M_POSE6: tuple[float, float, float, float, float, float] = (
    -0.2901545, 0.1042136, 0.3399979, 0.0, 0.0, 0.0,
)


@dataclass
class FrameValidationConfig:
    # Acquisition.  In offline mode the existing image is used and
    # offline_robot_tcp_pose6 must be filled in.
    use_live_hardware: bool = True
    capture_new_image: bool = True
    offline_robot_tcp_pose6: tuple[float, float, float, float, float, float] | None = None

    # T_R_B: pose of the FIXED beam/model frame B in robot-base frame R.
    # Baked from the reference capture (magnet in its calibrated spot, TCP at
    # [0.52615, -0.67020, 0.31343, ...], magnet 30 mm below it, pivot 300 mm
    # back along +B.x).  The beam base does NOT move -- only the source magnet
    # moves with the robot.  With derive_beam_frame_from_axes the rotation here
    # is informational; the translation is used when
    # pivot_behind_magnet_along_axial_mm is None.
    T_robot_beam_pose6: tuple[float, float, float, float, float, float] = (
        0.525575,
        -0.670028,
        -0.016567,
        0.0,
        -1.5707963,
        0.0,
    )

    # T_TCP_M: pose of source-magnet frame M in the UR TCP frame.
    # This must describe the magnet centre AND magnet body-axis orientation.
    # It intentionally has no guessed default.  If the UR TCP has physically
    # been calibrated at the magnet centre with matching axes, set
    # assume_tcp_is_magnet_frame=True and leave this as None.
    #
    # Default here: build T_TCP_M from the measured SOURCE_MAGNET_OFFSET_BELOW_TCP_M
    # placeholder above (magnet centre along TCP +Z, optional body rotation).
    T_tcp_magnet_pose6: tuple[float, float, float, float, float, float] | None = field(
        default_factory=lambda: tuple(SOURCE_MAGNET_T_TCP_M_POSE6)
    )
    assume_tcp_is_magnet_frame: bool = False

    # Camera-plane calibration.  The existing saved ex_img/ey_img values are
    # image-Cartesian vectors (x right, y up).  The signs reproduce the legacy
    # intended model convention: +B.x is opposite ex_img and +B.y follows
    # ey_img.  All sign handling occurs here exactly once.
    camera_calibration_mode: str = "basis_scale"  # "basis_scale" or "homography"
    saved_axis_convention: str = "image_cartesian"  # or "pixel_uv"
    # (-1, +1): +B.x is opposite ex_img (the saved ex_img points anti-axially,
    # from the beam back through the base), +B.y follows ey_img.  This matches
    # the documented convention above.  Check the [DEBUG] "+B.x(10mm) -> pixel
    # delta" line and camera_overlay.png (measured cyan vs model magenta must
    # overlie) before changing these.
    saved_axis_signs_for_positive_beam_xy: tuple[float, float] = (-1.0, 1.0)
    # 30 mm matches the rest of the codebase (bounds_beam.compute_mm_per_pixel
    # default and its use of this same calibration_points.json).  Only change
    # this if the two saved calibration points were physically clicked at a
    # different known separation.
    known_calibration_distance_mm: float = 38.0

    # The planar calibration origin is manual_frame["base_px"], which rarely
    # matches the detected base red marker exactly (a few px = ~1 mm).  That
    # gap biases every measured point, incl. the tip.  When True, all measured
    # B points are shifted so the DETECTED base marker sits at the B origin,
    # i.e. exactly where the forward model starts.
    anchor_beam_origin_to_detected_base: bool = True

    # Optional H_Bxy_I for perspective-aware planar calibration.  It maps
    # homogeneous [u_px, v_px, 1] to [B.x_m, B.y_m, 1].  Set mode="homography"
    # and paste a calibrated 3x3 matrix here when available.
    homography_pixel_to_beam_xy_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None

    # Marker detection and centreline reconstruction.
    marker_min_area_px2: float = 4.0
    marker_max_area_px2: float = 40000.0
    marker_saturation_min: int = 50
    marker_value_min: int = 40
    marker_red_hue1_high: int = 10
    marker_red_hue2_low: int = 170
    black_beam_threshold: int = 100
    beam_search_tube_radius_px: int = 25
    marker_bridge_radius_px: int = 8
    fallback_samples_per_marker_segment: int = 40

    # Model length.  If enabled, the calibrated measured centreline length is
    # clipped by the limits already present in EXPERIMENT_CONFIG.
    # (Set False to fix the model length and isolate frame errors from a
    # vision scale/detection error.)
    use_measured_beam_length_for_model: bool = True

    # -- Fixed beam-frame B pose in R -------------------------------------
    # T_robot_beam_pose6 is the ONLY camera<->robot calibration in the system,
    # and its saved value was wrong (wrong axial axis, wrong Z, +0.3 m in X).
    # When derive_beam_frame_from_axes is True the B frame is rebuilt from two
    # explicit robot-frame axes instead:
    #   * B.x  = beam_axial_axis_R          (the beam grows along +B.x)
    #   * B.z  = beam_plane_normal_axis_R   (out of the vision/bending plane,
    #                                        i.e. the camera viewing axis),
    #            re-orthogonalised against B.x
    #   * B.y  = B.z x B.x                  (right-handed)
    # The equivalent pose6 is printed at run time -- paste it back into
    # EXPERIMENT_CONFIG.pivot_pose6 once you are happy with it.
    derive_beam_frame_from_axes: bool = True
    # Direction the beam grows in, in robot base coordinates.  The rig has the
    # beam standing vertically, so +Z.
    beam_axial_axis_R: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Camera viewing axis / beam-plane normal in robot base coordinates.  Only
    # the component perpendicular to beam_axial_axis_R is used.  Flip the sign
    # if the modelled beam / vessel / magnet come out left-right mirrored in
    # camera_overlay.png (this sign is not observable from a planar measurement).
    beam_plane_normal_axis_R: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    # Put the B-frame ORIGIN a fixed distance behind the source magnet along
    # -B.x.  Use this ONCE, at the reference/calibration pose, to derive the
    # fixed beam base (read the printed ">>> equivalent pivot_pose6" and paste
    # it into T_robot_beam_pose6).  Then set this to None so the beam base stays
    # fixed and only the source magnet moves with the robot.
    pivot_behind_magnet_along_axial_mm: float | None = None

    # -- Model base orientation ------------------------------------------------
    # The Cosserat forward model extends its rod along  R0 @ [-1, 0, 0]  where
    # R0 is the model base rotation.  Feeding it the raw pivot_pose6 rotation
    # made the rod start OPPOSITE the real beam (the "mirror across the y axis"
    # you saw -- the modelled beam, and anything drawn from it, pointed the
    # wrong way).  When True, R0 is rebuilt from the measured beam so the rod
    # starts along the observed proximal tangent, inside the observed beam
    # plane.  Set False to use the raw pivot_pose6 rotation (old behaviour).
    align_model_base_with_measured_beam: bool = True
    # Fraction of the measured centreline (from the base) used to estimate the
    # proximal tangent that the model rod is aligned to.
    model_base_tangent_fraction: float = 0.25
    # When True the model rod starts exactly along +B.x (the nominal axial
    # axis) instead of the measured proximal tangent.  The reconstruction is
    # noisy right at the base (it is anchored to the base marker), so aligning
    # to it injects a few degrees of spurious start tilt into the model -- which
    # then shows up as a tip offset and a tangent-angle error even when the
    # source dipole is perfectly axial.  Use this to check whether an axial
    # dipole keeps the modelled beam straight (it should).
    model_rod_along_nominal_axial: bool = True

    # -- Source-magnet dipole direction -------------------------------------
    # With source_dipole_along_beam_axial=False (the normal case) the source
    # moment is source_dipole_unit_in_magnet_body expressed in the magnet body
    # (== TCP) frame, carried into the world by the LIVE TCP rotation -- so as
    # you move/rotate the robot the modelled dipole rotates with it.  The
    # magnet is poled along its body -Z (that is what "forced axial" resolved
    # to at the reference pose: dipole in TCP frame = [-0.02, 0.00, -1.00]).
    # magnet_yaw_calibration_deg is applied on top in the body XY plane.
    # Set True only to pin the dipole exactly along +B.x regardless of the TCP.
    source_dipole_along_beam_axial: bool = False
    # Body-frame dipole direction chosen so the world dipole is exactly +R.z
    # (beam axial) at the reference pose; rotating the magnet rotates it.
    source_dipole_unit_in_magnet_body: tuple[float, float, float] = (
        -0.019473, 0.001061, -0.999810,
    )

    # Source-magnet drawing dimensions in its body frame M.
    source_magnet_dimensions_mm: tuple[float, float, float] = (20.0, 20.0, 20.0)
    source_axis_display_length_mm: float = 12.0
    beam_axis_display_length_mm: float = 10.0
    dipole_display_length_mm: float = 15.0

    # Diagnostic output and thresholds.  A failed check is reported.  With
    # strict_checks=True it also makes the script exit with an error.
    output_root: str = "robotics_frame_validation"
    run_name: str = "stationary_camera_robot_model_check"
    show_interactively: bool = False
    comparison_points: int = 120
    strict_checks: bool = False
    maximum_robot_translation_during_capture_mm: float = 0.25
    maximum_robot_rotation_during_capture_deg: float = 0.10
    maximum_detected_base_error_mm: float = 3.0
    maximum_tip_error_mm: float = 10.0
    maximum_centerline_rmse_mm: float = 10.0
    maximum_tangent_error_deg: float = 15.0
    maximum_camera_round_trip_error_px: float = 1.0e-6


CONFIG = FrameValidationConfig()


# =============================================================================
# RIGID TRANSFORMS -- the only place where SE(3) composition is implemented
# =============================================================================


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}; got {array}.")
    return array


def _unit(vector: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"Cannot normalize zero-length {name}.")
    return vector / norm


def beam_frame_rotation_from_axes(
    axial_axis_R: Iterable[float],
    normal_axis_R: Iterable[float],
) -> np.ndarray:
    """Rotation R_R_B with B.x = axial, B.z ~ normal, B.y = B.z x B.x.

    ``normal_axis_R`` is re-orthogonalised against the axial axis, so only its
    perpendicular component matters.
    """

    b_x = _unit(axial_axis_R, "beam_axial_axis_R")
    normal = _finite_array(normal_axis_R, (3,), "beam_plane_normal_axis_R")
    b_z = normal - float(np.dot(normal, b_x)) * b_x
    if float(np.linalg.norm(b_z)) < 1.0e-6:
        raise ValueError(
            "beam_plane_normal_axis_R is parallel to beam_axial_axis_R; "
            "pick a normal axis that is not along the beam."
        )
    b_z = _unit(b_z, "beam-plane normal")
    b_y = np.cross(b_z, b_x)
    return np.column_stack([b_x, b_y, b_z])


def model_base_rotation_from_beam(
    tangent_R: Iterable[float],
    plane_normal_R: Iterable[float],
) -> np.ndarray:
    """Rotation R0 whose rod axis (R0 @ [-1, 0, 0]) is the measured tangent.

    The Cosserat forward model grows its rod along ``R0 @ [-1, 0, 0]``.  Given
    the measured proximal tangent and the beam-plane normal (both in R), return
    a right-handed R0 with that rod axis and its third column aligned as closely
    as possible with ``plane_normal_R``.
    """

    x_axis = -_unit(tangent_R, "measured proximal beam tangent")
    normal = _unit(plane_normal_R, "beam-plane normal")
    y_axis = np.cross(normal, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1.0e-9:
        raise ValueError("Measured beam tangent is parallel to the beam-plane normal.")
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


@dataclass(frozen=True)
class FrameTransform:
    """A named rigid transform T_parent_child mapping child into parent."""

    parent: str
    child: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = _finite_array(self.matrix, (4, 4), f"T_{self.parent}_{self.child}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
            raise ValueError("A rigid transform must end with [0, 0, 0, 1].")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
            raise ValueError(f"T_{self.parent}_{self.child} rotation is not orthogonal.")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8):
            raise ValueError(f"T_{self.parent}_{self.child} is not right-handed.")
        object.__setattr__(self, "matrix", matrix.copy())

    @classmethod
    def from_pose6(
        cls,
        parent: str,
        child: str,
        pose6: Iterable[float],
    ) -> "FrameTransform":
        pose = _finite_array(pose6, (6,), f"T_{parent}_{child} pose6")
        matrix = np.eye(4)
        matrix[:3, :3] = Rot.from_rotvec(pose[3:6]).as_matrix()
        matrix[:3, 3] = pose[:3]
        return cls(parent, child, matrix)

    @classmethod
    def identity(cls, parent: str, child: str) -> "FrameTransform":
        return cls(parent, child, np.eye(4))

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "FrameTransform":
        inverse = np.eye(4)
        inverse[:3, :3] = self.rotation.T
        inverse[:3, 3] = -self.rotation.T @ self.translation
        return FrameTransform(self.child, self.parent, inverse)

    def compose(self, next_transform: "FrameTransform") -> "FrameTransform":
        """Return self @ next_transform, for example T_R_TCP @ T_TCP_M."""

        if self.child != next_transform.parent:
            raise ValueError(
                f"Cannot compose T_{self.parent}_{self.child} with "
                f"T_{next_transform.parent}_{next_transform.child}."
            )
        return FrameTransform(
            self.parent,
            next_transform.child,
            self.matrix @ next_transform.matrix,
        )

    def apply_points(self, points_child: Any) -> np.ndarray:
        points = np.asarray(points_child, dtype=float)
        original_shape = points.shape
        points = points.reshape(-1, 3)
        mapped = points @ self.rotation.T + self.translation
        return mapped.reshape(original_shape)

    def apply_directions(self, directions_child: Any) -> np.ndarray:
        directions = np.asarray(directions_child, dtype=float)
        original_shape = directions.shape
        directions = directions.reshape(-1, 3)
        mapped = directions @ self.rotation.T
        return mapped.reshape(original_shape)

    def as_pose6(self) -> np.ndarray:
        return np.r_[self.translation, Rot.from_matrix(self.rotation).as_rotvec()]


# =============================================================================
# PIXEL-TO-BEAM PLANE CALIBRATION -- pixels are not treated as an SE(3) frame
# =============================================================================


@dataclass(frozen=True)
class PlanarPixelCalibration:
    """Calibrated mapping between image pixels and the B.z=0 beam plane."""

    mode: str
    H_Bxy_I: np.ndarray
    origin_px: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        H = _finite_array(self.H_Bxy_I, (3, 3), "H_Bxy_I")
        if abs(float(np.linalg.det(H))) < 1.0e-15:
            raise ValueError("Pixel-to-beam calibration matrix is singular.")
        origin = _finite_array(self.origin_px, (2,), "camera origin_px")
        object.__setattr__(self, "H_Bxy_I", H.copy())
        object.__setattr__(self, "origin_px", origin.copy())

    @classmethod
    def from_basis_scale(
        cls,
        *,
        origin_px: Iterable[float],
        ex_saved: Iterable[float],
        ey_saved: Iterable[float],
        metres_per_pixel: float,
        saved_axis_convention: str,
        positive_axis_signs: tuple[float, float],
    ) -> "PlanarPixelCalibration":
        origin = _finite_array(origin_px, (2,), "origin_px")
        ex = _finite_array(ex_saved, (2,), "ex_saved")
        ey = _finite_array(ey_saved, (2,), "ey_saved")
        if metres_per_pixel <= 0.0 or not np.isfinite(metres_per_pixel):
            raise ValueError("metres_per_pixel must be positive and finite.")
        if saved_axis_convention == "image_cartesian":
            ex_px = np.array([ex[0], -ex[1]])
            ey_px = np.array([ey[0], -ey[1]])
        elif saved_axis_convention == "pixel_uv":
            ex_px, ey_px = ex.copy(), ey.copy()
        else:
            raise ValueError(
                "saved_axis_convention must be 'image_cartesian' or 'pixel_uv'."
            )
        signs = _finite_array(positive_axis_signs, (2,), "positive_axis_signs")
        if np.any(np.abs(signs) < 0.5):
            raise ValueError("Each positive-axis sign must be +1 or -1.")

        raw_basis = np.column_stack(
            [signs[0] * _unit(ex_px, "saved x axis"), signs[1] * _unit(ey_px, "saved y axis")]
        )
        raw_dot = float(np.dot(raw_basis[:, 0], raw_basis[:, 1]))
        if abs(float(np.linalg.det(raw_basis))) < 0.2:
            raise ValueError("Saved camera axes are nearly parallel.")

        # Closest orthonormal pixel basis.  This prevents clicked-axis noise
        # from creating shear while retaining the handedness of the saved map.
        U, _, Vt = np.linalg.svd(raw_basis)
        basis_px_from_B = U @ Vt
        metres_per_pixel_matrix = metres_per_pixel * np.linalg.inv(basis_px_from_B)
        offset = -metres_per_pixel_matrix @ origin
        H = np.array(
            [
                [metres_per_pixel_matrix[0, 0], metres_per_pixel_matrix[0, 1], offset[0]],
                [metres_per_pixel_matrix[1, 0], metres_per_pixel_matrix[1, 1], offset[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        return cls(
            "basis_scale",
            H,
            origin,
            {
                "metres_per_pixel": float(metres_per_pixel),
                "saved_axis_convention": saved_axis_convention,
                "positive_axis_signs": signs.tolist(),
                "raw_basis_px_from_B": raw_basis.tolist(),
                "fitted_basis_px_from_B": basis_px_from_B.tolist(),
                "raw_axis_dot": raw_dot,
                "fitted_axis_dot": float(
                    np.dot(basis_px_from_B[:, 0], basis_px_from_B[:, 1])
                ),
            },
        )

    @classmethod
    def from_homography(
        cls,
        H_Bxy_I: Iterable[Iterable[float]],
    ) -> "PlanarPixelCalibration":
        H = _finite_array(H_Bxy_I, (3, 3), "homography_pixel_to_beam_xy_m")
        inverse = np.linalg.inv(H)
        homogeneous_origin = inverse @ np.array([0.0, 0.0, 1.0])
        origin = homogeneous_origin[:2] / homogeneous_origin[2]
        return cls("homography", H, origin, {"H_Bxy_I": H.tolist()})

    def pixels_to_beam(self, points_px: Any) -> np.ndarray:
        points = np.asarray(points_px, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("Pixel points must end in two coordinates [u, v].")
        original_shape = points.shape[:-1]
        points = points.reshape(-1, 2)
        homogeneous = np.column_stack([points, np.ones(len(points))])
        mapped = homogeneous @ self.H_Bxy_I.T
        if np.any(np.abs(mapped[:, 2]) < 1.0e-12):
            raise ValueError("Pixel point maps to infinity under the homography.")
        xy = mapped[:, :2] / mapped[:, 2:3]
        xyz = np.column_stack([xy, np.zeros(len(xy))])
        return xyz.reshape(*original_shape, 3)

    def beam_to_pixels(self, points_B_m: Any) -> np.ndarray:
        points = np.asarray(points_B_m, dtype=float)
        if points.shape[-1] not in (2, 3):
            raise ValueError("Beam points must end in two or three coordinates.")
        original_shape = points.shape[:-1]
        xy = points.reshape(-1, points.shape[-1])[:, :2]
        inverse = np.linalg.inv(self.H_Bxy_I)
        homogeneous = np.column_stack([xy, np.ones(len(xy))]) @ inverse.T
        if np.any(np.abs(homogeneous[:, 2]) < 1.0e-12):
            raise ValueError("Beam point maps to infinity under the homography.")
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixels.reshape(*original_shape, 2)


# =============================================================================
# CAMERA MEASUREMENT AND LUMEN GEOMETRY
# =============================================================================


def compute_metres_per_pixel(
    point_1_px: Iterable[float],
    point_2_px: Iterable[float],
    known_distance_mm: float,
) -> float:
    p1 = _finite_array(point_1_px, (2,), "calibration point 1")
    p2 = _finite_array(point_2_px, (2,), "calibration point 2")
    distance_px = float(np.linalg.norm(p2 - p1))
    if distance_px < 1.0e-6:
        raise ValueError("Calibration points are coincident.")
    if known_distance_mm <= 0.0:
        raise ValueError("known_calibration_distance_mm must be positive.")
    return known_distance_mm / (1000.0 * distance_px)


def _dense_marker_polyline(
    ordered_points_px: np.ndarray,
    samples_per_segment: int,
) -> np.ndarray:
    if samples_per_segment < 2:
        raise ValueError("fallback_samples_per_marker_segment must be at least 2.")
    pieces = []
    for start, end in zip(ordered_points_px[:-1], ordered_points_px[1:]):
        fractions = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        pieces.append(start[None, :] + fractions[:, None] * (end - start)[None, :])
    pieces.append(ordered_points_px[-1:])
    return np.vstack(pieces)


def _orient_polyline(points: np.ndarray, start_reference: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points.reshape(-1, points.shape[-1])
    reference = np.asarray(start_reference, dtype=float).reshape(-1)
    if np.linalg.norm(points[-1] - reference) < np.linalg.norm(points[0] - reference):
        return points[::-1].copy()
    return points.copy()


def measure_beam_from_image(
    *,
    image_bgr: np.ndarray,
    roi_polygon: Any,
    pivot_hint_px: Iterable[float],
    calibration: PlanarPixelCalibration,
    T_R_B: FrameTransform,
    cfg: FrameValidationConfig,
    bounds_beam: Any,
) -> dict[str, Any]:
    """Detect markers and map position, tangent and centreline consistently."""

    candidates = bounds_beam.detect_4_red_markers_in_roi(
        image_bgr,
        roi_box=None,
        roi_polygon=roi_polygon,
        min_area=cfg.marker_min_area_px2,
        max_area=cfg.marker_max_area_px2,
        sat_min=cfg.marker_saturation_min,
        val_min=cfg.marker_value_min,
        hue1_high=cfg.marker_red_hue1_high,
        hue2_low=cfg.marker_red_hue2_low,
        show_debug=False,
        min_markers=3,
        max_markers=4,
    )
    if candidates is None or len(candidates) not in (3, 4):
        raise RuntimeError(f"Expected three or four red markers; got {candidates}.")
    base_c, mag_start_c, tangent_start_c, tip_c = (
        bounds_beam.order_beam_marker_candidates(candidates, pivot_hint=pivot_hint_px)
    )
    markers_px = {
        "base": np.asarray(base_c["point"], dtype=float),
        "magnetised_start": (
            None if mag_start_c is None else np.asarray(mag_start_c["point"], dtype=float)
        ),
        "tangent_start": np.asarray(tangent_start_c["point"], dtype=float),
        "tip": np.asarray(tip_c["point"], dtype=float),
    }
    ordered_points_px = np.vstack(
        [
            markers_px[name]
            for name in ("base", "magnetised_start", "tangent_start", "tip")
            if markers_px[name] is not None
        ]
    )

    reconstruction_method = "marker_anchored_black_path"
    try:
        _, _, centreline_px = bounds_beam.compute_marker_anchored_black_beam_length_px(
            image_bgr=image_bgr,
            ordered_pts=ordered_points_px,
            threshold=cfg.black_beam_threshold,
            tube_radius_px=cfg.beam_search_tube_radius_px,
            bridge_radius_px=cfg.marker_bridge_radius_px,
        )
        centreline_px = np.asarray(centreline_px, dtype=float)
        if centreline_px.ndim != 2 or centreline_px.shape[0] < 2 or centreline_px.shape[1] != 2:
            raise ValueError("Black-path reconstruction returned an invalid polyline.")
    except Exception as error:
        reconstruction_method = f"marker_polyline_fallback: {type(error).__name__}: {error}"
        centreline_px = _dense_marker_polyline(
            ordered_points_px,
            cfg.fallback_samples_per_marker_segment,
        )

    centreline_px = _orient_polyline(centreline_px, markers_px["base"])

    # Detected base marker in B, before any anchoring.  With the calibration
    # origin at manual_frame["base_px"], this is how far the clicked base pixel
    # and the detected red marker disagree -- a constant bias on every measured
    # point.  If anchoring is on, shift all measured B points by it so the beam
    # base sits exactly at the B origin (= T_R_B.translation), matching where
    # the forward model starts.
    raw_detected_base_B = calibration.pixels_to_beam(markers_px["base"])
    origin_offset_B = (
        raw_detected_base_B if cfg.anchor_beam_origin_to_detected_base else np.zeros(3)
    )

    centreline_B_m = calibration.pixels_to_beam(centreline_px) - origin_offset_B
    centreline_R_m = T_R_B.apply_points(centreline_B_m)

    tip_B_m = calibration.pixels_to_beam(markers_px["tip"]) - origin_offset_B
    tangent_start_B_m = calibration.pixels_to_beam(markers_px["tangent_start"]) - origin_offset_B
    tangent_B = _unit(tip_B_m - tangent_start_B_m, "measured beam tangent in B")
    tangent_R = _unit(T_R_B.apply_directions(tangent_B), "measured beam tangent in R")

    length_m = float(np.sum(np.linalg.norm(np.diff(centreline_B_m, axis=0), axis=1)))
    return {
        "markers_px": markers_px,
        "ordered_points_px": ordered_points_px,
        "centreline_px": centreline_px,
        "centreline_B_m": centreline_B_m,
        "centreline_R_m": centreline_R_m,
        "detected_base_B_m": raw_detected_base_B - origin_offset_B,
        "raw_detected_base_B_m": raw_detected_base_B,
        "origin_offset_B_m": origin_offset_B,
        "tip_B_m": tip_B_m,
        "tip_R_m": T_R_B.apply_points(tip_B_m),
        "tangent_B": tangent_B,
        "tangent_R": tangent_R,
        "length_m": length_m,
        "reconstruction_method": reconstruction_method,
    }


def build_lumen_in_shared_frames(
    manual_frame: dict[str, Any],
    calibration: PlanarPixelCalibration,
    T_R_B: FrameTransform,
    origin_offset_B: Iterable[float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    left_px = np.asarray(manual_frame["left_boundary_px"], dtype=float)
    right_px = np.asarray(manual_frame["right_boundary_px"], dtype=float)
    if left_px.shape != right_px.shape or left_px.ndim != 2 or left_px.shape[1] != 2:
        raise ValueError("Left/right lumen boundaries must have matching shape (N, 2).")
    offset = _finite_array(origin_offset_B, (3,), "origin_offset_B")
    left_B = calibration.pixels_to_beam(left_px) - offset
    right_B = calibration.pixels_to_beam(right_px) - offset
    centre_B = 0.5 * (left_B + right_B)
    radius_m = 0.5 * np.linalg.norm(right_B - left_B, axis=1)
    if np.linalg.norm(centre_B[-1]) < np.linalg.norm(centre_B[0]):
        left_px, right_px = left_px[::-1], right_px[::-1]
        left_B, right_B = left_B[::-1], right_B[::-1]
        centre_B, radius_m = centre_B[::-1], radius_m[::-1]
    return {
        "left_px": left_px,
        "right_px": right_px,
        "left_B_m": left_B,
        "right_B_m": right_B,
        "centre_B_m": centre_B,
        "centre_R_m": T_R_B.apply_points(centre_B),
        "radius_m": radius_m,
    }


# =============================================================================
# SOURCE-MAGNET STATE AND FORWARD MODEL
# =============================================================================


def midpoint_pose6(pose_a: Iterable[float], pose_b: Iterable[float]) -> np.ndarray:
    a = _finite_array(pose_a, (6,), "robot pose before image")
    b = _finite_array(pose_b, (6,), "robot pose after image")
    rotation = Rot.from_rotvec(np.vstack([a[3:6], b[3:6]])).mean()
    return np.r_[0.5 * (a[:3] + b[:3]), rotation.as_rotvec()]


def pose_drift(pose_a: Iterable[float], pose_b: Iterable[float]) -> tuple[float, float]:
    a = _finite_array(pose_a, (6,), "pose_a")
    b = _finite_array(pose_b, (6,), "pose_b")
    translation_mm = 1.0e3 * float(np.linalg.norm(b[:3] - a[:3]))
    relative = Rot.from_rotvec(a[3:6]).inv() * Rot.from_rotvec(b[3:6])
    rotation_deg = float(np.degrees(relative.magnitude()))
    return translation_mm, rotation_deg


def source_transform_from_tcp(
    robot_tcp_pose6: Iterable[float],
    cfg: FrameValidationConfig,
) -> tuple[FrameTransform, FrameTransform, FrameTransform]:
    T_R_TCP = FrameTransform.from_pose6("R", "TCP", robot_tcp_pose6)
    if cfg.T_tcp_magnet_pose6 is None:
        if not cfg.assume_tcp_is_magnet_frame:
            raise ValueError(
                "T_tcp_magnet_pose6 is required. Calibrate the pose of the magnet "
                "centre/body frame in the UR TCP frame, or explicitly set "
                "assume_tcp_is_magnet_frame=True if those frames are physically identical."
            )
        T_TCP_M = FrameTransform.identity("TCP", "M")
    else:
        T_TCP_M = FrameTransform.from_pose6("TCP", "M", cfg.T_tcp_magnet_pose6)
    return T_R_TCP, T_TCP_M, T_R_TCP.compose(T_TCP_M)


def transform_to_p8(T_R_M: FrameTransform, insertion_length_m: float) -> np.ndarray:
    quaternion_xyzw = Rot.from_matrix(T_R_M.rotation).as_quat()
    quaternion_wxyz = np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]]
    return np.r_[T_R_M.translation, quaternion_wxyz, float(insertion_length_m)]


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rot.from_matrix(rotation).as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def build_forward_model_in_shared_frame(
    *,
    experiment_cfg: experiment_v2.ExperimentConfig,
    T_R_B: FrameTransform,
    lumen: dict[str, np.ndarray],
    model_base_rotation_R: np.ndarray | None = None,
    dipole_unit_in_magnet_body: Iterable[float] = (-1.0, 0.0, 0.0),
    T_R_TCP: FrameTransform | None = None,
    force_dipole_axis_R: Iterable[float] | None = None,
) -> tuple[experiment_v2.DirectForwardModelAdapter, dict[str, Any]]:
    """Build the same composite model using lumen geometry already in R.

    ``model_base_rotation_R`` overrides the rod base rotation (see
    ``model_base_rotation_from_beam``); when None the raw ``T_R_B`` rotation is
    used.  If ``force_dipole_axis_R`` is given (with ``T_R_TCP``) the source
    moment is set so it points exactly along that world axis; otherwise
    ``dipole_unit_in_magnet_body`` sets the direction in the magnet body frame
    before ``magnet_yaw_calibration_deg`` is applied.
    """

    base = experiment_v2._base_module()
    from proper_research.parameters import default_magnet_params
    from proper_research.simulation.magnetic_beam.contact import ContactParams
    from proper_research.simulation.simulations.model_factory_optimized import (
        build_forward_model,
    )

    calculated, m_local_factory, Kinv_fun = experiment_v2.calculate_composite_components(
        experiment_cfg.composite
    )
    if not callable(m_local_factory) or not callable(Kinv_fun):
        raise TypeError("Composite magnetisation and stiffness factories must be callable.")

    magnet_params = default_magnet_params()
    if force_dipole_axis_R is not None:
        if T_R_TCP is None:
            raise ValueError("force_dipole_axis_R requires T_R_TCP.")
        dipole_dir_body = _unit(
            T_R_TCP.rotation.T @ _unit(force_dipole_axis_R, "force_dipole_axis_R"),
            "forced dipole direction in magnet body frame",
        )
        nominal_m_body = float(magnet_params.mag_epm) * dipole_dir_body
        m_body = nominal_m_body
        dipole_mode = "forced along beam axial (+B.x)"
    else:
        dipole_dir_body = _unit(dipole_unit_in_magnet_body, "dipole_unit_in_magnet_body")
        nominal_m_body = float(magnet_params.mag_epm) * dipole_dir_body
        m_body = base.rotate_body_xy(
            nominal_m_body,
            experiment_cfg.magnet_yaw_calibration_deg,
        )
        dipole_mode = "from source_dipole_unit_in_magnet_body + yaw"
    use_contact = experiment_cfg.jacobian_variant == "contact"
    contact = ContactParams(
        r_beam=experiment_cfg.beam_contact_radius_m,
        k=experiment_cfg.contact_k,
        pen_switch=experiment_cfg.contact_penetration_switch_m,
        k_hard=experiment_cfg.contact_k_hard,
        smooth=True,
        smooth_eps=experiment_cfg.contact_smooth_epsilon_m,
        window=experiment_cfg.contact_window,
    )
    if model_base_rotation_R is None:
        base_rotation_R = T_R_B.rotation
    else:
        base_rotation_R = _finite_array(
            model_base_rotation_R, (3, 3), "model_base_rotation_R"
        )
    raw_model = build_forward_model(
        p0_ur=T_R_B.translation,
        q0_ur=_rotation_to_quaternion_wxyz(base_rotation_R),
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C=lumen["centre_R_m"],
        lumen_R=lumen["radius_m"],
        contact_enabled=use_contact,
        use_lumen_jac=use_contact,
        contact_params=contact if use_contact else None,
        m_local_factory=m_local_factory,
        result_detail="contact",
        store_history=False,
        store_vectors_in_info=False,
        sensitivity_workers=experiment_cfg.sensitivity_workers,
        copy_cached_results=False,
    )
    raw_model.composite_properties = dict(calculated)
    raw_model.composite_inputs = asdict(experiment_cfg.composite)
    adapter = experiment_v2.DirectForwardModelAdapter(raw_model, T_R_B.translation)
    return adapter, {
        "composite_inputs": asdict(experiment_cfg.composite),
        "composite_calculated": calculated,
        "external_source_magnet_body_moment": np.asarray(m_body, dtype=float),
        "dipole_unit_in_magnet_body": dipole_dir_body,
        "dipole_mode": dipole_mode,
        "magnet_moment_magnitude": float(magnet_params.mag_epm),
        "magnet_yaw_calibration_deg": float(experiment_cfg.magnet_yaw_calibration_deg),
        "model_base_rotation_R": base_rotation_R,
        "model_base_rod_axis_R": base_rotation_R @ np.array([-1.0, 0.0, 0.0]),
        "model_base_rotation_overridden": model_base_rotation_R is not None,
        "contact_enabled": use_contact,
        "model_type": type(raw_model).__name__,
    }


def _extract_model_centreline(
    adapter: Any,
    base_R_m: np.ndarray,
) -> np.ndarray:
    cache = getattr(adapter.raw_model, "cache", None)
    centreline = None if cache is None else getattr(cache, "centerline", None)

    if centreline is None:
        raise RuntimeError(
            "Forward-model cache does not expose a centreline after solve."
        )

    centreline = np.asarray(centreline, dtype=float)
    print(f"[MODEL CENTRELINE] raw shape: {centreline.shape}")

    if centreline.ndim != 2 or not np.all(np.isfinite(centreline)):
        raise RuntimeError(
            f"Invalid model centreline: shape={centreline.shape}"
        )

    if centreline.shape[1] == 3:
        # Already point-major: (N, 3)
        points_R_m = centreline
    elif centreline.shape[0] == 3:
        # Coordinate-major: (3, N) -> (N, 3)
        points_R_m = centreline.T
    else:
        raise RuntimeError(
            "Model centreline must have shape (N, 3) or (3, N); "
            f"received {centreline.shape}."
        )

    points_R_m = _orient_polyline(
        points_R_m,
        np.asarray(base_R_m, dtype=float),
    )

    length_mm = 1.0e3 * np.sum(
        np.linalg.norm(np.diff(points_R_m, axis=0), axis=1)
    )

    print(f"[MODEL CENTRELINE] normalized shape: {points_R_m.shape}")
    print(f"[MODEL CENTRELINE] first point: {points_R_m[0]}")
    print(f"[MODEL CENTRELINE] final point: {points_R_m[-1]}")
    print(f"[MODEL CENTRELINE] length: {length_mm:.6f} mm")

    return points_R_m


# =============================================================================
# RUN-TIME DEBUG DUMP -- verify calibration, frames and the magnet offset
# =============================================================================


def _pose6_str(transform: FrameTransform) -> str:
    pose = transform.as_pose6()
    return (
        f"t=[{pose[0]:+.5f} {pose[1]:+.5f} {pose[2]:+.5f}] m  "
        f"rotvec=[{pose[3]:+.4f} {pose[4]:+.4f} {pose[5]:+.4f}] rad"
    )


def _screen_dir(delta_px: np.ndarray) -> str:
    horizontal = "right" if delta_px[0] > 0 else "left"
    vertical = "down" if delta_px[1] > 0 else "up"
    return f"{horizontal}+{vertical}"


def print_frame_debug(
    *,
    calibration: PlanarPixelCalibration,
    T_R_B: FrameTransform,
    T_R_TCP: FrameTransform,
    T_TCP_M: FrameTransform,
    T_R_M: FrameTransform,
    measurement: dict[str, Any],
    model: dict[str, Any],
    model_metadata: dict[str, Any],
    insertion_length_m: float,
    cfg: FrameValidationConfig,
) -> None:
    """One consolidated dump so each stage of the chain can be eyeballed."""

    rule = "-" * 78
    meta = calibration.metadata

    print(f"\n{rule}\n[DEBUG] CAMERA-PLANE CALIBRATION\n{rule}")
    print(f"[DEBUG]   mode                      : {calibration.mode}")
    if meta.get("metres_per_pixel"):
        print(f"[DEBUG]   mm_per_pixel               : {1.0e3 * meta['metres_per_pixel']:.6f}")
    print(f"[DEBUG]   origin_px (maps to B 0,0)  : {calibration.origin_px}")
    print(f"[DEBUG]   raw_basis_px_from_B        : {meta.get('raw_basis_px_from_B')}")
    print(f"[DEBUG]   fitted_basis_px_from_B     : {meta.get('fitted_basis_px_from_B')}")
    print(f"[DEBUG]   H_Bxy_I                    : {calibration.H_Bxy_I.tolist()}")
    probe_B = np.array([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0], [0.0, 0.010, 0.0]])
    probe_px = calibration.beam_to_pixels(probe_B)
    dx_px = probe_px[1] - probe_px[0]
    dy_px = probe_px[2] - probe_px[0]
    print(f"[DEBUG]   +B.x 10mm -> pixel delta   : {np.round(dx_px, 2)}  (screen {_screen_dir(dx_px)})")
    print(f"[DEBUG]   +B.y 10mm -> pixel delta   : {np.round(dy_px, 2)}  (screen {_screen_dir(dy_px)})")
    print(f"[DEBUG]     (the visible beam runs base->tip; +B.x should point along it)")
    round_trip = calibration.beam_to_pixels(
        calibration.pixels_to_beam(measurement["centreline_px"])
    )
    round_trip_px = float(
        np.max(np.linalg.norm(round_trip - measurement["centreline_px"], axis=1))
    )
    print(f"[DEBUG]   pixel<->B round-trip error : {round_trip_px:.3e} px")

    print(f"\n{rule}\n[DEBUG] RIGID FRAMES  (T_A_B maps coords from B into A)\n{rule}")
    print(f"[DEBUG]   T_R_B    {_pose6_str(T_R_B)}")
    print(f"[DEBUG]   T_R_TCP  {_pose6_str(T_R_TCP)}")
    print(f"[DEBUG]   T_TCP_M  {_pose6_str(T_TCP_M)}")
    print(f"[DEBUG]   T_R_M    {_pose6_str(T_R_M)}")
    tcp_R = T_R_TCP.translation
    magnet_R = T_R_M.translation
    delta_R = magnet_R - tcp_R
    delta_tcp = T_R_TCP.rotation.T @ delta_R
    print(f"[DEBUG]   TCP position in R          : [{tcp_R[0]:+.5f} {tcp_R[1]:+.5f} {tcp_R[2]:+.5f}] m")
    print(f"[DEBUG]   magnet position in R       : [{magnet_R[0]:+.5f} {magnet_R[1]:+.5f} {magnet_R[2]:+.5f}] m")
    print(
        f"[DEBUG]   magnet - TCP in R           : "
        f"[{delta_R[0] * 1e3:+.2f} {delta_R[1] * 1e3:+.2f} {delta_R[2] * 1e3:+.2f}] mm"
    )
    print(
        f"[DEBUG]   magnet - TCP in TCP frame   : "
        f"[{delta_tcp[0] * 1e3:+.2f} {delta_tcp[1] * 1e3:+.2f} {delta_tcp[2] * 1e3:+.2f}] mm  "
        f"(expect ~[0 0 +{SOURCE_MAGNET_OFFSET_BELOW_TCP_M * 1e3:.0f}] = magnet along TCP +Z)"
    )
    print(f"[DEBUG]   magnet below TCP in base Z : {(-delta_R[2]) * 1e3:+.2f} mm")
    print(
        f"[DEBUG]   source position in B        : "
        f"[{model['source_position_B_m'][0] * 1e3:+.2f} "
        f"{model['source_position_B_m'][1] * 1e3:+.2f} "
        f"{model['source_position_B_m'][2] * 1e3:+.2f}] mm  (B.z = out-of-beam-plane)"
    )
    print(f"[DEBUG]   dipole direction in R      : {np.round(model['dipole_R'], 4)}")

    print(f"\n{rule}\n[DEBUG] SOURCE DIPOLE / MODEL BASE\n{rule}")
    dipole_R = _unit(model["dipole_R"], "dipole_R")
    dipole_B = _unit(T_R_B.rotation.T @ dipole_R, "dipole in B")
    dipole_tcp = _unit(T_R_TCP.rotation.T @ dipole_R, "dipole in TCP")
    beam_axis_R = _unit(
        np.asarray(measurement["tip_R_m"], float)
        - np.asarray(measurement["centreline_R_m"], float)[0],
        "measured beam axial in R",
    )
    field_dir_R = _unit(
        np.asarray(model["source_position_R_m"], float)
        - np.asarray(measurement["centreline_R_m"], float)[0],
        "base->magnet direction in R",
    )
    print(f"[DEBUG]   |mag_epm|                  : {model_metadata.get('magnet_moment_magnitude'):.4g} A m^2")
    print(f"[DEBUG]   dipole mode                : {model_metadata.get('dipole_mode')}")
    print(f"[DEBUG]   dipole unit in magnet body : {np.round(model_metadata.get('dipole_unit_in_magnet_body'), 4).tolist()}")
    print(f"[DEBUG]   dipole in R                : {np.round(dipole_R, 4).tolist()}")
    print(f"[DEBUG]   dipole in B                : {np.round(dipole_B, 4).tolist()}  (target from your spec: [1, 0, 0])")
    print(f"[DEBUG]   dipole in TCP frame        : {np.round(dipole_tcp, 4).tolist()}")
    print(f"[DEBUG]   angle(dipole, beam axial)  : {vector_angle_deg(dipole_R, beam_axis_R):.2f} deg  (0=along beam, 180=anti)")
    print(f"[DEBUG]   angle(dipole, base->magnet): {vector_angle_deg(dipole_R, field_dir_R):.2f} deg")
    base_to_magnet_mm = 1e3 * float(
        np.linalg.norm(
            np.asarray(model["source_position_R_m"], float)
            - np.asarray(measurement["centreline_R_m"], float)[0]
        )
    )
    mu0 = 4.0e-7 * np.pi
    m_mag = float(model_metadata.get("magnet_moment_magnitude", 0.0))
    b_axial_mt = (
        1e3 * mu0 / (4.0 * np.pi) * 2.0 * m_mag / (base_to_magnet_mm / 1e3) ** 3
        if base_to_magnet_mm > 1.0
        else float("nan")
    )
    print(
        f"[DEBUG]   base->magnet distance      : {base_to_magnet_mm:.1f} mm  "
        f"-> point-dipole |B| ~ {b_axial_mt:.3f} mT axial"
    )
    rod_axis_R = np.asarray(model_metadata.get("model_base_rod_axis_R"), float)
    print(
        f"[DEBUG]   model rod axis (R0@[-1,0,0]): {np.round(rod_axis_R, 4).tolist()}  "
        f"overridden={model_metadata.get('model_base_rotation_overridden')}"
    )
    print(
        f"[DEBUG]   angle(model rod, measured) : "
        f"{vector_angle_deg(rod_axis_R, beam_axis_R):.2f} deg  (should be ~0)"
    )

    print(f"\n{rule}\n[DEBUG] MARKERS / MEASURED BEAM\n{rule}")
    for name, point in measurement["markers_px"].items():
        print(f"[DEBUG]   marker {name:16s} px  : {None if point is None else np.round(point, 1)}")
    print(f"[DEBUG]   reconstruction_method      : {measurement['reconstruction_method']}")
    detected_base = np.asarray(measurement["detected_base_B_m"], dtype=float).reshape(-1)
    print(
        f"[DEBUG]   detected base in B          : "
        f"[{detected_base[0] * 1e3:+.2f} {detected_base[1] * 1e3:+.2f} "
        f"{detected_base[2] * 1e3:+.2f}] mm  (expect ~[0 0 0])"
    )
    print(f"[DEBUG]   detected base error norm   : {1e3 * float(np.linalg.norm(detected_base)):.2f} mm")
    print(
        f"[DEBUG]   measured tip in B           : "
        f"[{measurement['tip_B_m'][0] * 1e3:+.2f} {measurement['tip_B_m'][1] * 1e3:+.2f} "
        f"{measurement['tip_B_m'][2] * 1e3:+.2f}] mm"
    )
    print(f"[DEBUG]   measured tip in R          : {np.round(measurement['tip_R_m'], 5)} m")
    print(f"[DEBUG]   measured tangent in B      : {np.round(measurement['tangent_B'], 4)}")
    print(f"[DEBUG]   measured centreline length : {1e3 * measurement['length_m']:.2f} mm")
    print(
        f"[DEBUG]   length used for model       : {1e3 * insertion_length_m:.2f} mm  "
        f"(use_measured_beam_length_for_model={cfg.use_measured_beam_length_for_model})"
    )

    print(f"\n{rule}\n[DEBUG] MODEL vs MEASURED\n{rule}")
    tip_error_R = np.asarray(measurement["tip_R_m"], dtype=float) - np.asarray(
        model["tip_R_m"], dtype=float
    )
    print(
        f"[DEBUG]   model tip in B             : "
        f"[{model['tip_B_m'][0] * 1e3:+.2f} {model['tip_B_m'][1] * 1e3:+.2f} "
        f"{model['tip_B_m'][2] * 1e3:+.2f}] mm"
    )
    print(
        f"[DEBUG]   measured tip in B          : "
        f"[{measurement['tip_B_m'][0] * 1e3:+.2f} {measurement['tip_B_m'][1] * 1e3:+.2f} "
        f"{measurement['tip_B_m'][2] * 1e3:+.2f}] mm"
    )
    print(
        f"[DEBUG]   tip error (measured-model) : "
        f"[{tip_error_R[0] * 1e3:+.2f} {tip_error_R[1] * 1e3:+.2f} {tip_error_R[2] * 1e3:+.2f}] mm  "
        f"|.|={1e3 * float(np.linalg.norm(tip_error_R)):.2f} mm"
    )
    model_length_mm = 1e3 * float(
        np.sum(np.linalg.norm(np.diff(model["centreline_R_m"], axis=0), axis=1))
    )
    print(f"[DEBUG]   model centreline length    : {model_length_mm:.2f} mm")
    print(
        f"[DEBUG]   tangent angle error        : "
        f"{vector_angle_deg(measurement['tangent_R'], model['tangent_R']):.2f} deg"
    )
    print(f"{rule}\n")


# =============================================================================
# COMPARISON, CHECKS, PLOTTING AND LOGGING
# =============================================================================


def resample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("Polyline must contain at least two points.")
    if count < 2:
        raise ValueError("comparison_points must be at least two.")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    if cumulative[-1] < 1.0e-12:
        raise ValueError("Cannot resample a zero-length polyline.")
    samples = np.linspace(0.0, cumulative[-1], count)
    resampled = np.column_stack(
        [np.interp(samples, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )
    return resampled, samples / cumulative[-1]


def vector_angle_deg(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = _unit(vector_a, "vector_a")
    b = _unit(vector_b, "vector_b")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def compute_metrics_and_checks(
    *,
    measurement: dict[str, Any],
    model: dict[str, Any],
    calibration: PlanarPixelCalibration,
    robot_translation_drift_mm: float,
    robot_rotation_drift_deg: float,
    cfg: FrameValidationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measured_rs, normalized_s = resample_polyline(
        measurement["centreline_R_m"], cfg.comparison_points
    )
    model_rs, _ = resample_polyline(model["centreline_R_m"], cfg.comparison_points)
    residual_R_m = measured_rs - model_rs
    residual_norm_mm = 1.0e3 * np.linalg.norm(residual_R_m, axis=1)

    round_trip_px = calibration.beam_to_pixels(
        calibration.pixels_to_beam(measurement["centreline_px"])
    )
    camera_round_trip_error_px = float(
        np.max(np.linalg.norm(round_trip_px - measurement["centreline_px"], axis=1))
    )
    tip_error_R_m = measurement["tip_R_m"] - model["tip_R_m"]
    metrics = {
        "robot_translation_during_capture_mm": robot_translation_drift_mm,
        "robot_rotation_during_capture_deg": robot_rotation_drift_deg,
        "detected_base_error_mm": 1.0e3
        * float(np.linalg.norm(measurement["detected_base_B_m"])),
        "tip_error_R_mm": 1.0e3 * tip_error_R_m,
        "tip_error_norm_mm": 1.0e3 * float(np.linalg.norm(tip_error_R_m)),
        "centreline_rmse_mm": float(np.sqrt(np.mean(residual_norm_mm**2))),
        "centreline_max_error_mm": float(np.max(residual_norm_mm)),
        "tangent_error_deg": vector_angle_deg(
            measurement["tangent_R"], model["tangent_R"]
        ),
        "measured_length_mm": 1.0e3 * measurement["length_m"],
        "model_length_mm": 1.0e3
        * float(
            np.sum(
                np.linalg.norm(np.diff(model["centreline_R_m"], axis=0), axis=1)
            )
        ),
        "camera_round_trip_error_px": camera_round_trip_error_px,
        "source_position_B_mm": 1.0e3 * model["source_position_B_m"],
        "source_out_of_beam_plane_mm": 1.0e3 * float(model["source_position_B_m"][2]),
        "normalized_s": normalized_s,
        "measured_resampled_R_m": measured_rs,
        "model_resampled_R_m": model_rs,
        "residual_R_m": residual_R_m,
    }
    checks = {
        "robot_stationary_translation": {
            "passed": robot_translation_drift_mm
            <= cfg.maximum_robot_translation_during_capture_mm,
            "value": robot_translation_drift_mm,
            "limit": cfg.maximum_robot_translation_during_capture_mm,
            "units": "mm",
        },
        "robot_stationary_rotation": {
            "passed": robot_rotation_drift_deg
            <= cfg.maximum_robot_rotation_during_capture_deg,
            "value": robot_rotation_drift_deg,
            "limit": cfg.maximum_robot_rotation_during_capture_deg,
            "units": "deg",
        },
        "detected_base_matches_fixed_origin": {
            "passed": metrics["detected_base_error_mm"]
            <= cfg.maximum_detected_base_error_mm,
            "value": metrics["detected_base_error_mm"],
            "limit": cfg.maximum_detected_base_error_mm,
            "units": "mm",
        },
        "tip_matches_model": {
            "passed": metrics["tip_error_norm_mm"] <= cfg.maximum_tip_error_mm,
            "value": metrics["tip_error_norm_mm"],
            "limit": cfg.maximum_tip_error_mm,
            "units": "mm",
        },
        "centreline_matches_model": {
            "passed": metrics["centreline_rmse_mm"]
            <= cfg.maximum_centerline_rmse_mm,
            "value": metrics["centreline_rmse_mm"],
            "limit": cfg.maximum_centerline_rmse_mm,
            "units": "mm",
        },
        "tangent_matches_model": {
            "passed": metrics["tangent_error_deg"] <= cfg.maximum_tangent_error_deg,
            "value": metrics["tangent_error_deg"],
            "limit": cfg.maximum_tangent_error_deg,
            "units": "deg",
        },
        "camera_map_round_trip": {
            "passed": camera_round_trip_error_px
            <= cfg.maximum_camera_round_trip_error_px,
            "value": camera_round_trip_error_px,
            "limit": cfg.maximum_camera_round_trip_error_px,
            "units": "px",
        },
    }
    checks["all_passed"] = bool(all(item["passed"] for item in checks.values()))
    return metrics, checks


def _source_box_points_M(dimensions_mm: Iterable[float]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    half = 0.5e-3 * _finite_array(dimensions_mm, (3,), "source magnet dimensions")
    corners = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    edges = []
    for i, a in enumerate(corners):
        for j in range(i + 1, len(corners)):
            b = corners[j]
            if np.count_nonzero(np.abs(a - b) > 1.0e-12) == 1:
                edges.append((i, j))
    return corners, edges


def _draw_triad_3d(axis: Any, transform: FrameTransform, length_mm: float, label: str) -> None:
    origin = 1.0e3 * transform.translation
    colours = ("tab:red", "tab:green", "tab:blue")
    for index, (colour, suffix) in enumerate(zip(colours, ("x", "y", "z"))):
        delta = length_mm * transform.rotation[:, index]
        axis.quiver(*origin, *delta, color=colour, arrow_length_ratio=0.15)
        endpoint = origin + delta
        axis.text(*endpoint, f"{label}.{suffix}")


def _set_equal_3d(axis: Any, points_mm: np.ndarray) -> None:
    points = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    centre = 0.5 * (low + high)
    radius = max(0.5 * float(np.max(high - low)), 5.0)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)


def make_camera_overlay(
    *,
    image_bgr: np.ndarray,
    measurement: dict[str, Any],
    model: dict[str, Any],
    lumen: dict[str, np.ndarray],
    calibration: PlanarPixelCalibration,
    T_B_R: FrameTransform,
) -> np.ndarray:
    import cv2

    overlay = image_bgr.copy()

    def polyline(points_px: np.ndarray, colour: tuple[int, int, int], thickness: int) -> None:
        points = np.round(points_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], False, colour, thickness, cv2.LINE_AA)

    polyline(lumen["left_px"], (0, 180, 0), 1)
    polyline(lumen["right_px"], (0, 180, 0), 1)
    polyline(measurement["centreline_px"], (255, 255, 0), 2)
    model_B = T_B_R.apply_points(model["centreline_R_m"])
    model_px = calibration.beam_to_pixels(model_B)
    polyline(model_px, (255, 0, 255), 2)

    for name, point in measurement["markers_px"].items():
        if point is None:
            continue
        pixel = tuple(np.round(point).astype(int))
        cv2.circle(overlay, pixel, 5, (0, 0, 255), -1)
        cv2.putText(overlay, name, (pixel[0] + 6, pixel[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    origin_and_axes_B = np.array(
        [[0.0, 0.0, 0.0], [0.010, 0.0, 0.0], [0.0, 0.010, 0.0]]
    )
    axes_px = calibration.beam_to_pixels(origin_and_axes_B)
    origin = tuple(np.round(axes_px[0]).astype(int))
    cv2.arrowedLine(overlay, origin, tuple(np.round(axes_px[1]).astype(int)), (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(overlay, origin, tuple(np.round(axes_px[2]).astype(int)), (0, 180, 0), 2, tipLength=0.15)
    cv2.putText(overlay, "+B.x", tuple(np.round(axes_px[1]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(overlay, "+B.y", tuple(np.round(axes_px[2]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    source_px = calibration.beam_to_pixels(model["source_position_B_m"])
    source_pixel = tuple(np.round(source_px).astype(int))
    cv2.drawMarker(overlay, source_pixel, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(
        overlay,
        f"source projection; B.z={1e3*model['source_position_B_m'][2]:+.1f} mm",
        (source_pixel[0] + 8, source_pixel[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 165, 255),
        1,
    )
    cv2.putText(overlay, "cyan: measured   magenta: model", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return overlay


def make_validation_figure(
    *,
    camera_overlay_bgr: np.ndarray,
    measurement: dict[str, Any],
    model: dict[str, Any],
    lumen: dict[str, np.ndarray],
    metrics: dict[str, Any],
    checks: dict[str, Any],
    T_R_B: FrameTransform,
    T_R_M: FrameTransform,
    T_B_R: FrameTransform,
    cfg: FrameValidationConfig,
    output_path: Path,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 11))
    camera_axis = figure.add_subplot(2, 2, 1)
    beam_axis = figure.add_subplot(2, 2, 2)
    robot_axis = figure.add_subplot(2, 2, 3, projection="3d")
    residual_axis = figure.add_subplot(2, 2, 4)

    camera_axis.imshow(cv2.cvtColor(camera_overlay_bgr, cv2.COLOR_BGR2RGB))
    camera_axis.set_title("Raw camera frame with calibrated projections")
    camera_axis.axis("off")

    measured_B_mm = 1.0e3 * measurement["centreline_B_m"]
    model_B_mm = 1.0e3 * T_B_R.apply_points(model["centreline_R_m"])
    beam_axis.plot(1.0e3 * lumen["left_B_m"][:, 0], 1.0e3 * lumen["left_B_m"][:, 1], color="0.65", label="lumen walls")
    beam_axis.plot(1.0e3 * lumen["right_B_m"][:, 0], 1.0e3 * lumen["right_B_m"][:, 1], color="0.65")
    beam_axis.plot(measured_B_mm[:, 0], measured_B_mm[:, 1], color="tab:cyan", linewidth=2, label="measured")
    beam_axis.plot(model_B_mm[:, 0], model_B_mm[:, 1], color="tab:purple", linewidth=2, label="model")
    source_B_mm = 1.0e3 * model["source_position_B_m"]
    beam_axis.scatter(source_B_mm[0], source_B_mm[1], marker="x", s=70, color="tab:orange", label="source projection")
    beam_axis.scatter(0.0, 0.0, color="black", s=30, label="fixed B origin")
    beam_axis.set_aspect("equal", adjustable="datalim")
    beam_axis.set_xlabel("B.x (mm)")
    beam_axis.set_ylabel("B.y (mm)")
    beam_axis.set_title(f"Fixed beam frame; source B.z={source_B_mm[2]:+.2f} mm")
    beam_axis.grid(True, alpha=0.3)
    beam_axis.legend()

    measured_R_mm = 1.0e3 * measurement["centreline_R_m"]
    model_R_mm = 1.0e3 * model["centreline_R_m"]
    robot_axis.plot(*measured_R_mm.T, color="tab:cyan", linewidth=2, label="measured")
    robot_axis.plot(*model_R_mm.T, color="tab:purple", linewidth=2, label="model")
    _draw_triad_3d(robot_axis, T_R_B, cfg.beam_axis_display_length_mm, "B")
    _draw_triad_3d(robot_axis, T_R_M, cfg.source_axis_display_length_mm, "M")
    corners_M, edges = _source_box_points_M(cfg.source_magnet_dimensions_mm)
    corners_R_mm = 1.0e3 * T_R_M.apply_points(corners_M)
    for start, end in edges:
        robot_axis.plot(*np.vstack([corners_R_mm[start], corners_R_mm[end]]).T, color="tab:orange", linewidth=1)
    dipole_delta_mm = cfg.dipole_display_length_mm * model["dipole_R"]
    robot_axis.quiver(
        *(1.0e3 * model["source_position_R_m"]),
        *dipole_delta_mm,
        color="black",
        arrow_length_ratio=0.18,
        normalize=False,
    )
    robot_axis.set_xlabel("R.x (mm)")
    robot_axis.set_ylabel("R.y (mm)")
    robot_axis.set_zlabel("R.z (mm)")
    robot_axis.set_title("Robot frame: beam, source box, body axes and dipole")
    robot_axis.legend()
    _set_equal_3d(robot_axis, np.vstack([measured_R_mm, model_R_mm, corners_R_mm]))

    residual_mm = 1.0e3 * metrics["residual_R_m"]
    s = metrics["normalized_s"]
    for index, label in enumerate(("R.x", "R.y", "R.z")):
        residual_axis.plot(s, residual_mm[:, index], label=label)
    residual_axis.plot(s, np.linalg.norm(residual_mm, axis=1), color="black", linewidth=2, label="norm")
    residual_axis.axhline(0.0, color="0.4", linewidth=0.8)
    residual_axis.set_xlabel("normalized measured/model arclength")
    residual_axis.set_ylabel("measured - model (mm)")
    residual_axis.set_title(
        f"RMSE {metrics['centreline_rmse_mm']:.2f} mm; tip {metrics['tip_error_norm_mm']:.2f} mm; tangent {metrics['tangent_error_deg']:.1f} deg"
    )
    residual_axis.grid(True, alpha=0.3)
    residual_axis.legend()

    overall = "PASS" if checks["all_passed"] else "REVIEW CHECKS"
    figure.suptitle(f"Stationary frame validation — {overall}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if cfg.show_interactively:
        plt.show()
    plt.close(figure)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}.")


def save_points_csv(metrics: dict[str, Any], T_B_R: FrameTransform, path: Path) -> None:
    measured_R = metrics["measured_resampled_R_m"]
    model_R = metrics["model_resampled_R_m"]
    measured_B = T_B_R.apply_points(measured_R)
    model_B = T_B_R.apply_points(model_R)
    residual = metrics["residual_R_m"]
    fields = [
        "normalized_s",
        "measured_R_x_m", "measured_R_y_m", "measured_R_z_m",
        "model_R_x_m", "model_R_y_m", "model_R_z_m",
        "measured_B_x_m", "measured_B_y_m", "measured_B_z_m",
        "model_B_x_m", "model_B_y_m", "model_B_z_m",
        "residual_R_x_m", "residual_R_y_m", "residual_R_z_m", "residual_norm_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, s in enumerate(metrics["normalized_s"]):
            writer.writerow(
                {
                    "normalized_s": s,
                    "measured_R_x_m": measured_R[index, 0], "measured_R_y_m": measured_R[index, 1], "measured_R_z_m": measured_R[index, 2],
                    "model_R_x_m": model_R[index, 0], "model_R_y_m": model_R[index, 1], "model_R_z_m": model_R[index, 2],
                    "measured_B_x_m": measured_B[index, 0], "measured_B_y_m": measured_B[index, 1], "measured_B_z_m": measured_B[index, 2],
                    "model_B_x_m": model_B[index, 0], "model_B_y_m": model_B[index, 1], "model_B_z_m": model_B[index, 2],
                    "residual_R_x_m": residual[index, 0], "residual_R_y_m": residual[index, 1], "residual_R_z_m": residual[index, 2],
                    "residual_norm_m": np.linalg.norm(residual[index]),
                }
            )


def make_output_directory(cfg: FrameValidationConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def validate_configuration(cfg: FrameValidationConfig) -> None:
    if cfg.camera_calibration_mode not in {"basis_scale", "homography"}:
        raise ValueError("camera_calibration_mode must be 'basis_scale' or 'homography'.")
    if cfg.camera_calibration_mode == "homography" and cfg.homography_pixel_to_beam_xy_m is None:
        raise ValueError("homography mode requires homography_pixel_to_beam_xy_m.")
    if not cfg.use_live_hardware and cfg.offline_robot_tcp_pose6 is None:
        raise ValueError("Offline mode requires offline_robot_tcp_pose6.")
    # Validate the physically important tool transform before connecting to any
    # hardware or capturing a frame.
    validation_pose = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if cfg.offline_robot_tcp_pose6 is None
        else cfg.offline_robot_tcp_pose6
    )
    source_transform_from_tcp(validation_pose, cfg)
    FrameTransform.from_pose6("R", "B", cfg.T_robot_beam_pose6)
    if cfg.comparison_points < 2:
        raise ValueError("comparison_points must be at least two.")


def load_planar_calibration(
    base: Any,
    manual_frame: dict[str, Any],
    cfg: FrameValidationConfig,
) -> PlanarPixelCalibration:
    if cfg.camera_calibration_mode == "homography":
        return PlanarPixelCalibration.from_homography(
            cfg.homography_pixel_to_beam_xy_m
        )
    saved = base.bounds_beam.get_saved_2_point_calibration(
        EXPERIMENT_CONFIG.calibration_points_path
    )
    point_1, point_2 = saved["points_px"]
    metres_per_pixel = compute_metres_per_pixel(
        point_1,
        point_2,
        cfg.known_calibration_distance_mm,
    )
    distance_px = float(
        np.linalg.norm(np.asarray(point_2, float) - np.asarray(point_1, float))
    )
    print(
        f"[DEBUG] 2-point scale calib: p1={point_1} p2={point_2} "
        f"|p2-p1|={distance_px:.2f} px  known={cfg.known_calibration_distance_mm} mm "
        f"-> {1.0e3 * metres_per_pixel:.6f} mm/px"
    )
    print(
        f"[DEBUG] calib origin_px (B 0,0) = {list(manual_frame['base_px'])}  "
        f"ex_img={list(manual_frame['ex_img'])}  ey_img={list(manual_frame['ey_img'])}  "
        f"signs(Bx,By)={cfg.saved_axis_signs_for_positive_beam_xy}"
    )
    return PlanarPixelCalibration.from_basis_scale(
        origin_px=manual_frame["base_px"],
        ex_saved=manual_frame["ex_img"],
        ey_saved=manual_frame["ey_img"],
        metres_per_pixel=metres_per_pixel,
        saved_axis_convention=cfg.saved_axis_convention,
        positive_axis_signs=cfg.saved_axis_signs_for_positive_beam_xy,
    )


def main() -> None:
    import cv2

    validate_configuration(CONFIG)
    experiment_v2.validate_config(EXPERIMENT_CONFIG)
    base = experiment_v2._base_module()
    base.configure_bounds_beam_paths(EXPERIMENT_CONFIG)
    output_dir = make_output_directory(CONFIG)
    hardware = None
    try:
        if CONFIG.use_live_hardware:
            hardware = base.connect_hardware(EXPERIMENT_CONFIG)
            pose_before = hardware.get_robot_pose_once()
            if pose_before is None:
                raise RuntimeError("Could not read the UR TCP pose before image capture.")
            pose_before = _finite_array(pose_before, (6,), "UR TCP pose before image")
        else:
            pose_before = _finite_array(
                CONFIG.offline_robot_tcp_pose6,
                (6,),
                "offline_robot_tcp_pose6",
            )

        if CONFIG.capture_new_image:
            base.new_capture(filename=EXPERIMENT_CONFIG.image_filename)
        image_timestamp_utc = datetime.now(timezone.utc).isoformat()

        if CONFIG.use_live_hardware:
            pose_after = hardware.get_robot_pose_once()
            if pose_after is None:
                raise RuntimeError("Could not read the UR TCP pose after image capture.")
            pose_after = _finite_array(pose_after, (6,), "UR TCP pose after image")
        else:
            pose_after = pose_before.copy()
        robot_tcp_pose6 = midpoint_pose6(pose_before, pose_after)
        translation_drift_mm, rotation_drift_deg = pose_drift(pose_before, pose_after)
        print(
            f"[DEBUG] UR TCP pose before : {np.round(pose_before, 5).tolist()}\n"
            f"[DEBUG] UR TCP pose after  : {np.round(pose_after, 5).tolist()}\n"
            f"[DEBUG] UR TCP pose used   : {np.round(robot_tcp_pose6, 5).tolist()}\n"
            f"[DEBUG] capture drift      : {translation_drift_mm:.3f} mm / "
            f"{rotation_drift_deg:.3f} deg"
        )

        image_bgr = cv2.imread(EXPERIMENT_CONFIG.image_filename)
        if image_bgr is None:
            raise FileNotFoundError(EXPERIMENT_CONFIG.image_filename)
        manual_frame = base.load_manual_vessel_boundaries_with_frame(
            EXPERIMENT_CONFIG.manual_boundary_path
        )
        roi_polygon = base.load_polygon(EXPERIMENT_CONFIG.roi_polygon_path)
        calibration = load_planar_calibration(base, manual_frame, CONFIG)

        T_R_TCP, T_TCP_M, T_R_M = source_transform_from_tcp(robot_tcp_pose6, CONFIG)

        saved_T_R_B = FrameTransform.from_pose6("R", "B", CONFIG.T_robot_beam_pose6)
        if CONFIG.derive_beam_frame_from_axes:
            R_R_B = beam_frame_rotation_from_axes(
                CONFIG.beam_axial_axis_R,
                CONFIG.beam_plane_normal_axis_R,
            )
            orientation_note = "derived from beam_axial_axis_R + beam_plane_normal_axis_R"
        else:
            R_R_B = saved_T_R_B.rotation
            orientation_note = "using saved T_robot_beam_pose6 orientation"

        if CONFIG.pivot_behind_magnet_along_axial_mm is not None:
            b_x = R_R_B[:, 0]
            beam_origin_R = (
                T_R_M.translation
                + (CONFIG.pivot_behind_magnet_along_axial_mm / 1.0e3) * b_x
            )
            origin_note = (
                f"magnet + {CONFIG.pivot_behind_magnet_along_axial_mm:.1f} mm along +B.x"
            )
        else:
            beam_origin_R = saved_T_R_B.translation
            origin_note = "saved T_robot_beam_pose6 translation"

        beam_matrix = np.eye(4)
        beam_matrix[:3, :3] = R_R_B
        beam_matrix[:3, 3] = beam_origin_R
        T_R_B = FrameTransform("R", "B", beam_matrix)
        T_B_R = T_R_B.inverse()

        saved_rotvec = Rot.from_matrix(saved_T_R_B.rotation).as_rotvec()
        used_pose6 = np.r_[T_R_B.translation, Rot.from_matrix(T_R_B.rotation).as_rotvec()]
        magnet_B_check = T_B_R.apply_points(T_R_M.translation)
        print(
            f"[DEBUG] beam frame B: orientation {orientation_note}\n"
            f"[DEBUG]             : origin {origin_note}\n"
            f"[DEBUG]   saved pivot_pose6          : {np.round(np.asarray(CONFIG.T_robot_beam_pose6, float), 5).tolist()}\n"
            f"[DEBUG]   >>> equivalent pivot_pose6 : {np.round(used_pose6, 6).tolist()}\n"
            f"[DEBUG]       (paste into EXPERIMENT_CONFIG.pivot_pose6 to bake it in)\n"
            f"[DEBUG]   T_R_B columns  B.x/B.y/B.z : "
            f"{np.round(T_R_B.rotation[:, 0], 3).tolist()} / "
            f"{np.round(T_R_B.rotation[:, 1], 3).tolist()} / "
            f"{np.round(T_R_B.rotation[:, 2], 3).tolist()}\n"
            f"[DEBUG]   source magnet centre in B  : "
            f"[{magnet_B_check[0] * 1e3:+.1f} {magnet_B_check[1] * 1e3:+.1f} "
            f"{magnet_B_check[2] * 1e3:+.1f}] mm  (live position relative to the fixed beam base)"
        )

        measurement = measure_beam_from_image(
            image_bgr=image_bgr,
            roi_polygon=roi_polygon,
            pivot_hint_px=EXPERIMENT_CONFIG.pivot_hint_px,
            calibration=calibration,
            T_R_B=T_R_B,
            cfg=CONFIG,
            bounds_beam=base.bounds_beam,
        )
        lumen = build_lumen_in_shared_frames(
            manual_frame, calibration, T_R_B, measurement["origin_offset_B_m"]
        )
        if CONFIG.anchor_beam_origin_to_detected_base:
            print(
                f"[DEBUG] beam origin anchored to detected base marker; shift applied "
                f"{np.round(measurement['raw_detected_base_B_m'] * 1e3, 2).tolist()} mm in B"
            )

        if CONFIG.use_measured_beam_length_for_model:
            insertion_length_m = float(
                np.clip(
                    measurement["length_m"],
                    EXPERIMENT_CONFIG.min_vision_length_m,
                    EXPERIMENT_CONFIG.max_vision_length_m,
                )
            )
        else:
            insertion_length_m = float(EXPERIMENT_CONFIG.initial_beam_length_m)

        # Model base rotation.  Align the rod with the observed beam so the
        # model does not start pointing the opposite way (the mirror effect).
        centreline_R = np.asarray(measurement["centreline_R_m"], dtype=float)
        tangent_count = max(
            2,
            int(round(CONFIG.model_base_tangent_fraction * len(centreline_R))),
        )
        measured_base_tangent_R = _unit(
            centreline_R[min(tangent_count, len(centreline_R)) - 1] - centreline_R[0],
            "measured proximal beam tangent",
        )
        unaligned_rod_axis_R = T_R_B.rotation @ np.array([-1.0, 0.0, 0.0])
        nominal_axial_R = T_R_B.rotation[:, 0]
        if CONFIG.model_rod_along_nominal_axial:
            model_base_rotation_R = model_base_rotation_from_beam(
                nominal_axial_R,
                T_R_B.rotation[:, 2],
            )
            rod_source = "nominal +B.x axis"
        elif CONFIG.align_model_base_with_measured_beam:
            model_base_rotation_R = model_base_rotation_from_beam(
                measured_base_tangent_R,
                T_R_B.rotation[:, 2],
            )
            rod_source = "measured proximal tangent"
        else:
            model_base_rotation_R = None
            rod_source = "raw T_R_B rotation"
        print(
            f"[DEBUG] model rod axis  source        : {rod_source}\n"
            f"[DEBUG] model rod axis  nominal +B.x  : {np.round(nominal_axial_R, 4).tolist()}\n"
            f"[DEBUG] model rod axis  measured prox : {np.round(measured_base_tangent_R, 4).tolist()}\n"
            f"[DEBUG] measured-prox vs nominal axis : "
            f"{vector_angle_deg(nominal_axial_R, measured_base_tangent_R):.2f} deg"
            f"  (spurious start tilt if the beam is really straight)"
        )

        force_dipole_axis_R = (
            T_R_B.rotation[:, 0]
            if CONFIG.source_dipole_along_beam_axial
            else None
        )
        adapter, model_metadata = build_forward_model_in_shared_frame(
            experiment_cfg=EXPERIMENT_CONFIG,
            T_R_B=T_R_B,
            lumen=lumen,
            model_base_rotation_R=model_base_rotation_R,
            dipole_unit_in_magnet_body=CONFIG.source_dipole_unit_in_magnet_body,
            T_R_TCP=T_R_TCP,
            force_dipole_axis_R=force_dipole_axis_R,
        )
        p8 = transform_to_p8(T_R_M, insertion_length_m)
        p7, model_output = adapter.commit_nominal(p8)
        model_tip_R_m, model_tangent_R = experiment_v2._output_parts(
            model_output,
            T_R_B.translation,
        )
        model_centreline_R_m = _extract_model_centreline(adapter, T_R_B.translation)
        source_position_B_m = T_B_R.apply_points(T_R_M.translation)
        source_axes_B = T_B_R.rotation @ T_R_M.rotation
        m_body = np.asarray(model_metadata["external_source_magnet_body_moment"], dtype=float)
        dipole_R = _unit(T_R_M.rotation @ m_body, "source magnetic moment in R")
        model = {
            "tip_R_m": model_tip_R_m,
            "tip_B_m": T_B_R.apply_points(model_tip_R_m),
            "tangent_R": _unit(model_tangent_R, "model tangent in R"),
            "tangent_B": _unit(T_B_R.apply_directions(model_tangent_R), "model tangent in B"),
            "centreline_R_m": model_centreline_R_m,
            "centreline_B_m": T_B_R.apply_points(model_centreline_R_m),
            "source_position_R_m": T_R_M.translation,
            "source_position_B_m": source_position_B_m,
            "source_axes_B": source_axes_B,
            "dipole_R": dipole_R,
        }

        print_frame_debug(
            calibration=calibration,
            T_R_B=T_R_B,
            T_R_TCP=T_R_TCP,
            T_TCP_M=T_TCP_M,
            T_R_M=T_R_M,
            measurement=measurement,
            model=model,
            model_metadata=model_metadata,
            insertion_length_m=insertion_length_m,
            cfg=CONFIG,
        )

        metrics, checks = compute_metrics_and_checks(
            measurement=measurement,
            model=model,
            calibration=calibration,
            robot_translation_drift_mm=translation_drift_mm,
            robot_rotation_drift_deg=rotation_drift_deg,
            cfg=CONFIG,
        )
        camera_overlay = make_camera_overlay(
            image_bgr=image_bgr,
            measurement=measurement,
            model=model,
            lumen=lumen,
            calibration=calibration,
            T_B_R=T_B_R,
        )
        cv2.imwrite(str(output_dir / "camera_overlay.png"), camera_overlay)
        make_validation_figure(
            camera_overlay_bgr=camera_overlay,
            measurement=measurement,
            model=model,
            lumen=lumen,
            metrics=metrics,
            checks=checks,
            T_R_B=T_R_B,
            T_R_M=T_R_M,
            T_B_R=T_B_R,
            cfg=CONFIG,
            output_path=output_dir / "frame_validation.png",
        )
        save_points_csv(metrics, T_B_R, output_dir / "comparison_points.csv")

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_timestamp_utc": image_timestamp_utc,
            "hardware_motion_commanded": False,
            "hardware_execution_tested_in_development_workspace": False,
            "frame_notation": "T_A_B maps coordinates from frame B into frame A",
            "frames": {
                "R": "UR robot base / forward-model world frame",
                "TCP": "UR tool centre point returned by robot",
                "M": "source-magnet centre and body-axis frame",
                "B": "fixed beam base / model base frame",
                "I": "pixel coordinates [u right, v down]; not an SE(3) frame",
            },
            "transform_chain": "T_R_M = T_R_TCP @ T_TCP_M",
            "z_offset_handling": (
                "No scalar z_offset is applied. The complete physical tool offset is "
                "represented once by T_TCP_M. EXPERIMENT_CONFIG.z_offset_m is ignored here."
            ),
            "T_R_B": T_R_B.matrix,
            "T_R_TCP": T_R_TCP.matrix,
            "T_TCP_M": T_TCP_M.matrix,
            "T_R_M": T_R_M.matrix,
            "robot_pose6_before": pose_before,
            "robot_pose6_after": pose_after,
            "robot_pose6_midpoint_used": robot_tcp_pose6,
            "camera_calibration": {
                "mode": calibration.mode,
                "origin_px": calibration.origin_px,
                "H_Bxy_I": calibration.H_Bxy_I,
                "metadata": calibration.metadata,
            },
            "measurement": {
                "markers_px": measurement["markers_px"],
                "detected_base_B_m": measurement["detected_base_B_m"],
                "tip_B_m": measurement["tip_B_m"],
                "tip_R_m": measurement["tip_R_m"],
                "tangent_B": measurement["tangent_B"],
                "tangent_R": measurement["tangent_R"],
                "length_m": measurement["length_m"],
                "reconstruction_method": measurement["reconstruction_method"],
            },
            "model": {
                "p8_source_quaternion_state": p8,
                "p7_source_rotvec_state": p7,
                "tip_B_m": model["tip_B_m"],
                "tip_R_m": model["tip_R_m"],
                "tangent_B": model["tangent_B"],
                "tangent_R": model["tangent_R"],
                "source_position_B_m": model["source_position_B_m"],
                "source_position_R_m": model["source_position_R_m"],
                "source_axes_B_columns": model["source_axes_B"],
                "dipole_R": model["dipole_R"],
                "insertion_length_m": insertion_length_m,
                "metadata": model_metadata,
            },
            "metrics": {
                key: value
                for key, value in metrics.items()
                if key not in {"normalized_s", "measured_resampled_R_m", "model_resampled_R_m", "residual_R_m"}
            },
            "checks": checks,
            "validation_config": asdict(CONFIG),
            "experiment_config": asdict(EXPERIMENT_CONFIG),
        }
        with (output_dir / "frame_validation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)

        print("\nSTATIONARY FRAME VALIDATION")
        for name, result in checks.items():
            if name == "all_passed":
                continue
            marker = "PASS" if result["passed"] else "CHECK"
            print(
                f"[{marker:5s}] {name}: {result['value']:.4g} {result['units']} "
                f"(limit {result['limit']:.4g})"
            )
        print(f"Overall: {'PASS' if checks['all_passed'] else 'REVIEW FAILED CHECKS'}")
        print(f"Figure:  {output_dir / 'frame_validation.png'}")
        print(f"Summary: {output_dir / 'frame_validation_summary.json'}")
        print(f"Points:  {output_dir / 'comparison_points.csv'}")

        if CONFIG.strict_checks and not checks["all_passed"]:
            raise RuntimeError("One or more strict frame-validation checks failed.")
    finally:
        if hardware is not None:
            base._safe_stop(hardware)


if __name__ == "__main__":
    main()
