import numpy as np

from proper_research.robot.transformations import get_point
from scipy.spatial.transform import Rotation as R


# =====================================================================
# FRAME CALIBRATION  (robot read 2026-09-10; camera-frame corrected)
# ---------------------------------------------------------------------
# GEOMETRY: the beam grows along world -X (insertion -> tip moves -X).
# Camera is OVERHEAD looking down (-Z), so it sees the world X-Y plane;
# its blind axis is world Z.  So beam frame B:
#   B.x (axial)      = world -X   (beam_axial_axis_R = (-1,0,0))
#   B.y (in-plane)   = world +Y   (left/right in the image)
#   B.z (blind)      = world -Z   (into the image; beam_plane_normal = (0,0,-1))
#
# Source magnet: N52 cylinder, 100 mm dia x 100 mm long, COAXIAL with the
# beam, dipole along the beam axis (world -X).  Centre at beam_base +
# [-0.28, 0, 0] -> 28 cm from the pivot along the axis, ~23 cm beyond the
# tip, same world Y and Z.  DH deltas in urik.CONFIG.
#
#   joints (rad) = [-0.85634357, -1.94584002, -1.76176286,
#                   -1.03319450,  1.56234264, -2.05640871]
#   TCP pose6    = [0.40795443, -0.73732753, 0.32527992,
#                  -3.07806404,  0.57586908,  0.04569585]
#
# Beam material parameters calibrated 2026-09-10 against live sweeps
# (calibration_2026-09-10/): xy-plane arc (phi +-40 deg, r = 28-38 cm),
# source-dipole rotation, and insertion length 18-36 mm.
#   -> effective_youngs_modulus_pa = 2.5e6  (in beam_hardware_experiment_v2)
#   -> in-plane RMS ~3.6 deg; camera confirms ZERO out-of-plane bend for
#      every in-plane magnet move (frame calibration is correct).
# Known model limits (fixed-permanent-magnetisation assumption):
#   * bend scales ~linearly with inserted length; the real beam's tip
#     angle is ~flat over 18-36 mm -> model only trust-worthy near ~33 mm.
#   * model is ~2x too insensitive to source-dipole rotation.
#   * model falls off too slowly with magnet distance (a bit stiff at r=38cm).
# =====================================================================

LIVE_JOINTS_RAD = (
    -0.85634357,
    -1.94584002,
    -1.76176286,
    -1.03319450,
    1.56234264,
    -2.05640871,
)

# TCP(flange) -> magnet-centre translation, in the flange frame at the
# reference-pose orientation.
TCP_TO_MAGNET_POSE6 = (-0.16542, -0.0022782, 0.3469254, 0.0, 0.0, 0.0)

# Source-magnet dipole direction in the magnet body frame, giving a world -R.x
# (beam axial: the beam grows along world -X) dipole at the reference pose.
SOURCE_DIPOLE_BODY_AXIS = (-0.932073, 0.361306, 0.026427)

SOURCE_MAGNET_DIAMETER_M = 0.10
SOURCE_MAGNET_LENGTH_M = 0.10


def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fixed initial robot/beam pose, matched to the frame calibration above.

    Returns ``(pivot_point, start_point, L_cmd, dt)``:
      * ``pivot_point`` -- beam base pose6 in R; rotvec is
        ``model_base_rotation_from_beam(+R.z, -R.x)``.
      * ``start_point`` -- source-magnet pose6 in R (calibrated-FK consistent).
      * ``L_cmd`` -- initial insertion [m] (near max).
      * ``dt`` -- controller timestep [s].
    """
    # 2026-09-10: model calibrated at E = 2.5 MPa (arc + dipole + insertion
    # sweeps).  L_cmd is the initial inserted length the offline planner starts
    # from AND the length the live beam must be set to before a planned run.
    # 2026-09-11: moved 0.030 -> 0.025 for the bigger (14 mm depth / 20 mm base)
    # apex-at-start triangle -- taller insertion range (25-39 mm instead of
    # 30-38.8 mm) at the SAME 39 mm max, to push the planned trajectory closer
    # to the velocity/acceleration limits (a deliberate stress-test: see
    # beam-lateral-authority-limit memory, MPC-vs-inverse-Jacobian clipping
    # investigation). Live beam must be set to 25 mm before a planned run on
    # this plan.
    L_cmd = 0.025
    dt = 0.01

    pivot_point = np.array(
        [0.525575, -0.670028, -0.016567,
         3.14159265, 0.0, 0.0],
        dtype=float,
    )

    start_point = np.array(
        [
            0.245575,
            -0.670028,
            -0.016567,
            -3.07793295,
            0.57537270,
            0.04503358,
        ],
        dtype=float,
    )

    print(f"[INIT] start_point (magnet in R) = {start_point}")
    print(f"[INIT] pivot_point  (beam base)  = {pivot_point}")
    print(f"[INIT] L0={L_cmd:.4f}, dt={dt:.4f}")

    return pivot_point, start_point, L_cmd, dt
