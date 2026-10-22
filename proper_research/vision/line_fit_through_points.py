import numpy as np


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def fit_line_direction_pca(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a best-fit 2D line through points using PCA/SVD.

    Parameters
    ----------
    points_xy : (N,2) array
        2D points in image coordinates or image-Cartesian coordinates.

    Returns
    -------
    centroid : (2,) array
        Mean of the points.
    direction : (2,) array
        Unit direction of best-fit line.
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    centroid = pts.mean(axis=0)

    X = pts - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = unit(Vt[0])   # dominant principal axis
    return centroid, direction


def make_beam_frame_from_fitted_line(
    base_px: np.ndarray,
    mag_start_px: np.ndarray,
    tangent_start_px: np.ndarray,
    tip_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a robust beam-local frame from a best-fit line through all 4 markers.

    Returns
    -------
    origin : (2,)
        Chosen as base marker.
    ex : (2,)
        Beam-axis direction in image-Cartesian coordinates.
    ey : (2,)
        Perpendicular direction, right-handed in 2D.
    """
    base = np.asarray(base_px, dtype=float).reshape(2,)
    mag_start = np.asarray(mag_start_px, dtype=float).reshape(2,)
    tangent_start = np.asarray(tangent_start_px, dtype=float).reshape(2,)
    tip = np.asarray(tip_px, dtype=float).reshape(2,)

    # Convert image coordinates to Cartesian-like coordinates:
    # x right stays the same, y up = -image_y
    pts_cart = np.array([
        [base[0],         -base[1]],
        [mag_start[0],    -mag_start[1]],
        [tangent_start[0],-tangent_start[1]],
        [tip[0],          -tip[1]],
    ], dtype=float)

    _, ex = fit_line_direction_pca(pts_cart)

    # Fix sign so ex points roughly from base toward tip
    base_cart = np.array([base[0], -base[1]], dtype=float)
    tip_cart = np.array([tip[0], -tip[1]], dtype=float)
    if np.dot(ex, tip_cart - base_cart) < 0:
        ex = -ex

    # 90 deg CCW in Cartesian coordinates
    ey = np.array([-ex[1], ex[0]], dtype=float)

    return base_cart, ex, ey


def project_point_to_beam_frame(
    point_px: np.ndarray,
    origin_cart: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
) -> np.ndarray:
    """
    Project a point into the beam-local fitted frame.

    Returns
    -------
    xy : (2,) array
        [axial, lateral] coordinates in pixels.
    """
    p = np.asarray(point_px, dtype=float).reshape(2,)
    p_cart = np.array([p[0], -p[1]], dtype=float)

    d = p_cart - origin_cart
    return np.array([
        np.dot(d, ex),
        np.dot(d, ey),
    ], dtype=float)
def make_beam_frame_from_proximal_markers(
    base_px: np.ndarray,
    mag_start_px: np.ndarray,
    tangent_start_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_px, dtype=float).reshape(2,)
    mag_start = np.asarray(mag_start_px, dtype=float).reshape(2,)
    tangent_start = np.asarray(tangent_start_px, dtype=float).reshape(2,)

    pts_cart = np.array([
        [base[0], -base[1]],
        [mag_start[0], -mag_start[1]],
        [tangent_start[0], -tangent_start[1]],
    ], dtype=float)

    centroid = pts_cart.mean(axis=0)
    X = pts_cart - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    ex = Vt[0]
    ex = ex / (np.linalg.norm(ex) + 1e-12)

    base_cart = np.array([base[0], -base[1]], dtype=float)
    tan_cart = np.array([tangent_start[0], -tangent_start[1]], dtype=float)

    if np.dot(ex, tan_cart - base_cart) < 0:
        ex = -ex

    ey = np.array([-ex[1], ex[0]], dtype=float)
    return base_cart, ex, ey
import numpy as np

def image_point_to_cartesian(pt_px: np.ndarray) -> np.ndarray:
    """
    Convert image coordinates (x right, y down) to Cartesian-like coordinates
    (x right, y up).
    """
    pt = np.asarray(pt_px, dtype=float).reshape(2,)
    return np.array([pt[0], -pt[1]], dtype=float)


def vector_angle_deg_cart(v: np.ndarray) -> float:
    """
    Absolute angle of a 2D vector in Cartesian coordinates, in degrees.
    0 deg = +x, 90 deg = +y.
    """
    v = np.asarray(v, dtype=float).reshape(2,)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def signed_angle_deg_between_2d(ref_vec: np.ndarray, test_vec: np.ndarray) -> float:
    """
    Signed angle from ref_vec to test_vec in degrees, in Cartesian coordinates.
    Positive = CCW, negative = CW.
    """
    ref_vec = np.asarray(ref_vec, dtype=float).reshape(2,)
    test_vec = np.asarray(test_vec, dtype=float).reshape(2,)

    cross_z = ref_vec[0] * test_vec[1] - ref_vec[1] * test_vec[0]
    dot = float(np.dot(ref_vec, test_vec))
    return float(np.degrees(np.arctan2(cross_z, dot)))


def measure_tip_base_angles(
    base_px: np.ndarray,
    tip_px: np.ndarray,
    ex_ref: np.ndarray | None = None,
) -> dict:
    """
    Compute useful measured angles from base to tip.

    Returns
    -------
    dict with:
      - tip_base_vec_cart
      - tip_base_abs_angle_deg
      - tip_base_angle_from_vertical_deg
      - tip_base_angle_from_ref_deg (if ex_ref provided)
    """
    base_cart = image_point_to_cartesian(base_px)
    tip_cart = image_point_to_cartesian(tip_px)

    v_bt = tip_cart - base_cart
    abs_angle_deg = vector_angle_deg_cart(v_bt)

    # angle from image vertical-up direction
    vertical_up = np.array([0.0, 1.0], dtype=float)
    angle_from_vertical_deg = signed_angle_deg_between_2d(vertical_up, v_bt)

    out = {
        "tip_base_vec_cart": v_bt,
        "tip_base_abs_angle_deg": abs_angle_deg,
        "tip_base_angle_from_vertical_deg": angle_from_vertical_deg,
    }

    if ex_ref is not None:
        out["tip_base_angle_from_ref_deg"] = signed_angle_deg_between_2d(ex_ref, v_bt)

    return out