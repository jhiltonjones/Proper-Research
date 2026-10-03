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
#
# 2026-10-02 RECALIBRATION: the physical tool/magnet mount changed (user
# re-aligned the setup) and the old value above no longer matches reality --
# confirmed two independent ways: (1) the user physically measured the
# magnet sitting ~43cm straight below the TCP in world Z with the tool
# "z-axis aligned" (same world x/y as TCP); (2) computed the implied
# flange-local offset from that measurement via R_F.T @ [0,0,-0.43] using
# the robot's own calibrated DH/FK at the live joints
# [-0.6261633078204554, -1.7615310154356898, -1.8847131729125977,
#  -1.0656407636455079, 1.5728745460510254, -1.0911524931537073] (TCP pose
# [0.4957078706700621, -0.5727193984076662, 0.3903778484269186, ...]) --
# result was [-0.126mm, 0.227mm, 429.9999mm], i.e. essentially pure flange-
# local +Z at exactly 430mm, with the tiny x/y residual consistent with FK/
# measurement noise, not a real off-axis offset. Rounded to a clean value
# below for that reason. This invalidates every offline plan and Jacobian
# schedule built before this date against the OLD value -- any of those
# must be rebuilt from scratch before further closed-loop testing; don't
# mix plans/schedules built under different TCP_TO_MAGNET_POSE6 values.
TCP_TO_MAGNET_POSE6 = (0.0, 0.0, 0.43, 0.0, 0.0, 0.0)

# 2026-10-03 RECALIBRATION: the old value below predates the 2026-10-02
# TCP_TO_MAGNET_POSE6 remount and was never re-verified afterward. Found
# live to be ~176deg off (essentially the negated polarity) from a fresh
# ground-truth measurement: the user manually jogged the magnet to a
# position physically in line with the beam base along world X (same Y/Z,
# 210mm separation) with the dipole visibly pointing straight at the base,
# and read off the live joints/TCP at that pose --
#   joints = [-0.9355629126178187, -1.80503573040151, -1.8300602436065674,
#             -1.0767775636962433, 1.57289457321167, -2.0786169211017054]
#   TCP pose6 = [0.3150097798725594, -0.7199134455939546, 0.39035840819409234,
#               -3.0702563895693964, 0.665670523991651, 4.511819286522139e-05]
# FK from those joints (magnet xyz via the current TCP_TO_MAGNET_POSE6)
# matches the given TCP pose to <0.15mm. At that pose the magnet sits at
# lower world-X than the beam base, so "dipole points directly at the
# base" means world dipole direction = +X; back-solving
# R_magnet.T @ [1,0,0] through the measured magnet orientation gives the
# value below. Old value was (-0.932073, 0.361306, 0.026427) -- note both
# are nearly pure in-plane (near-zero body-Z component), consistent with
# a diametrically-poled magnet whose physical pole axis is perpendicular
# to the flange/J6 rotation axis (so a psi sweep via joint 6 alone
# meaningfully rotates the dipole -- see sweep_free_space_arc_dipole.py).
# Source-magnet dipole direction in the magnet body frame, giving a world -R.x
# (beam axial: the beam grows along world -X) dipole at the reference pose.
SOURCE_DIPOLE_BODY_AXIS = (0.910293410, -0.413963475, 0.000386004)

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
    # investigation).
    # 2026-09-15: tried moving 0.025 -> 0.02 for the real-vessel
    # (detect_blue-sourced lumen) planning campaign, per explicit request, but
    # reverted: at L0=0.02 the vessel plan's chain-rule Jacobian validation
    # fails (relative error 93.8% vs the 9% ceiling) while the identical
    # config at L0=0.025 passes cleanly and proceeds into Layer 1 -- confirmed
    # insertion-length-dependent, not a general vessel-planning bug. Root
    # cause not yet found (not investigated: bimaterial wire/tip blend
    # boundary vs. vessel contact-model geometry at short insertion). Left at
    # 0.025 (known-working) pending that investigation.
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
