import numpy as np

from proper_research.robot.transformations import get_point
from scipy.spatial.transform import Rotation as R


# =====================================================================
# FINAL FRAME CALIBRATION  (robot read 2026-09-09, magnet geometry from
# direct physical measurement)
# ---------------------------------------------------------------------
# At the reference TCP the source magnet hangs on a bracket ~33 cm below
# and ~31 cm behind the flange (world offset [-0.314, 0.010, -0.334]).
# Its CENTRE sits at  beam_base + [-0.23, 0, 0]  in R -- 23 cm from the
# beam base / pivot, same R.y, same R.z (== [0, 0, +230] mm in beam
# frame B: purely out of the 2-D vision plane, level with the base).
# The magnet near edge is 15.5 cm from the base (it is ~15 cm long along
# -R.x).  Dipole aligned with the beam axial (+R.z) magnetisation.
# DH deltas are already fitted into urik.CONFIG.
#
#   joints (rad) = [-0.64739067, -2.05319991, -1.61638916,
#                   -1.06096228,  1.56594813, -1.84768182]
#   TCP pose6    = [0.60985444, -0.68005596, 0.31772396,
#                  -3.08577770,  0.57746171,  0.03115576]
#
# start_point = calibrated FK(live joints) @ pose6_to_T(TCP_TO_MAGNET_POSE6);
# make_robot_config IK (q_seed = live joints) returns those joints exactly.
# =====================================================================

LIVE_JOINTS_RAD = (
    -0.64739067,
    -2.05319991,
    -1.61638916,
    -1.06096228,
    1.56594813,
    -1.84768182,
)

# TCP(flange) -> magnet-centre translation, in the flange frame at the
# reference-pose orientation.
TCP_TO_MAGNET_POSE6 = (-0.2901545, 0.1042136, 0.3399979, 0.0, 0.0, 0.0)

# Source-magnet dipole direction in the magnet body frame, chosen so the
# world dipole is exactly +R.z (beam axial) at the reference pose.
SOURCE_DIPOLE_BODY_AXIS = (-0.019473, 0.001061, -0.999810)

# Magnet extent along its dipole axis (near edge 15.5 cm, centre 23 cm
# from the beam base -> ~15 cm long).  Point-dipole is marginal at this
# range; kept for reference / future distributed-source modelling.
SOURCE_MAGNET_LENGTH_M = 0.15


def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fixed initial robot/beam pose, matched to the final frame calibration.

    Returns ``(pivot_point, start_point, L_cmd, dt)``:
      * ``pivot_point`` -- beam base pose6 in R; rotvec is
        ``model_base_rotation_from_beam(+R.z, -R.x)`` so ``build_model_bundle``
        grows the rod along +R.z, bending plane R.y (in-plane) / R.x (out of
        the 2-D vision plane).
      * ``start_point`` -- source-magnet pose6 in R (calibrated-FK consistent).
      * ``L_cmd`` -- initial insertion [m] (near max).
      * ``dt`` -- controller timestep [s].
    """
    L_cmd = 0.045
    dt = 0.01

    pivot_point = np.array(
        [0.525575, -0.670028, -0.016567,
         2.221441469079183, 0.0, -2.221441469079183],
        dtype=float,
    )

    start_point = np.array(
        [
            0.295575,
            -0.670028,
            -0.016567,
            -3.08548018,
            0.57669094,
            0.03035070,
        ],
        dtype=float,
    )

    print(f"[INIT] start_point (magnet in R) = {start_point}")
    print(f"[INIT] pivot_point  (beam base)  = {pivot_point}")
    print(f"[INIT] L0={L_cmd:.4f}, dt={dt:.4f}")

    return pivot_point, start_point, L_cmd, dt
