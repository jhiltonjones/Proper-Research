import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def measured_result_to_xmeas(result):
    """
    Convert vision result into controller measurement x_meas
    expected as [x, y, z, tx, ty, tz].

    Assumes beam is planar and z = 0.
    Coordinates are in meters.
    """
    mm_per_pixel = float(result["mm_per_pixel"])

    tip_xy_mm = result["tip_result"]["tip_xy_from_base"]
    tip_x_m = float(tip_xy_mm[0]) * mm_per_pixel / 1000.0
    tip_y_m = float(tip_xy_mm[1]) * mm_per_pixel / 1000.0
    tip_z_m = 0.0

    beam_points_px = result["beam_centerline_px"]
    if len(beam_points_px) < 2:
        raise ValueError("Not enough beam points to estimate tangent.")

    p1 = np.array(beam_points_px[-2], dtype=float)
    p2 = np.array(beam_points_px[-1], dtype=float)

    # image x right, image y down -> Cartesian y up
    tangent = np.array([
        p2[0] - p1[0],
        -(p2[1] - p1[1]),
        0.0,
    ], dtype=float)
    tangent = unit(tangent)

    x_meas = np.array([
        tip_x_m,
        tip_y_m,
        tip_z_m,
        tangent[0],
        tangent[1],
        tangent[2],
    ], dtype=float)

    return x_meas