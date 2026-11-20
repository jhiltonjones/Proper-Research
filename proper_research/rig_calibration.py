"""Single source of truth for this rig's physical calibration constants.

Before 2026-10-04, the beam-base position, the source-magnet's dipole
body axis, and the TCP-to-magnet flange offset were each hardcoded
independently in more than a dozen files scattered across
proper_research/hardware, proper_research/vision, proper_research/
simulation and proper_research/controllers. Every time one of these
physical quantities was recalibrated (beam-base Y on 2026-10-03,
beam-base X on 2026-10-04 -- both multi-week staleness incidents, see
build_vessel_plan.py's git history for the forensics), the fix had to be
hand-propagated to every duplicate by grepping for the old literal
value, and at least one copy was missed each time, silently
reintroducing the same bug.

Recalibrate ONCE, here. Every other module imports these values instead
of hardcoding its own copy. If you are about to write a new numeric
literal for the beam base, the source dipole axis, or the TCP-to-magnet
offset anywhere else in this repo, stop and import it from here instead.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Beam base (robot frame R)
# ---------------------------------------------------------------------------
# 2026-10-04: X remeasured directly by the user with the source magnet
# held at phi=0 (collinear with the base, where a radius error cancels
# out in bearing and so is invisible to any dipole-alignment check) --
# true distance is 255mm, not the previously-assumed 110mm. A radius
# error at the collinear pose only shows up as a growing ANGULAR error
# off-axis, which is what every phi=20/60/90deg eyeball check was
# actually detecting all along, even though it looked like a dipole-axis
# or orientation bug at first.
#
# 2026-10-03: Y remeasured directly with the source magnet jogged "in
# line" with the beam base (dipole visibly pointing straight at it, zero
# Y offset by construction) -- true value is -0.719727, not the
# previously-assumed -0.670028 (introduced 2026-09-28, commit 9380019e,
# 4 days before the 2026-10-02 TCP_TO_MAGNET_POSE6 remount, and never
# re-verified afterward).
#
# Z: recalibrated 2026-10-02 via live forward kinematics (unraised
# setup). See capture_live_start_position.py and
# RAISED_30MM_Z_OFFSET_M below for the raised-rig variant.
BEAM_BASE_XYZ_M = np.array([0.670575, -0.719727, -0.039627])

# Orientation convention for the beam-base pivot pose: a rotvec of
# [pi, 0, 0] maps the rod's own local -X axis to world -X, matching this
# rig's "beam grows along world -X" convention used throughout the
# vessel_stage_a / sweep_free_space_arc_dipole pipeline.
BEAM_BASE_ROTVEC = np.array([3.14159265, 0.0, 0.0])


def beam_base_pose6(z_m: float | None = None) -> np.ndarray:
    """6-vector [x, y, z, rx, ry, rz] for the beam-base pivot pose.

    Pass `z_m` to override the Z component (e.g. for a raised-rig
    variant: `beam_base_pose6(BEAM_BASE_XYZ_M[2] + RAISED_30MM_Z_OFFSET_M)`)
    while still sharing the single canonical X/Y and rotation.
    """
    z = BEAM_BASE_XYZ_M[2] if z_m is None else float(z_m)
    return np.array([
        BEAM_BASE_XYZ_M[0], BEAM_BASE_XYZ_M[1], z,
        BEAM_BASE_ROTVEC[0], BEAM_BASE_ROTVEC[1], BEAM_BASE_ROTVEC[2],
    ])


# Some earlier campaigns physically raised the whole rig by a fixed
# amount; their Z should be DERIVED from BEAM_BASE_XYZ_M's Z plus this
# raise, not hardcoded as an independent literal (that independence is
# exactly how the X/Y staleness incidents above happened in the first
# place). Not applied by default -- only when a script explicitly knows
# it is running the raised setup.
RAISED_30MM_Z_OFFSET_M = 0.03

# ---------------------------------------------------------------------------
# Source magnet
# ---------------------------------------------------------------------------
# TCP -> magnet-centre offset: pure flange-local +Z, zero relative
# rotation. Confirmed empirically: sweeping joint 6 (the flange's own
# rotation axis) through 180deg moves the magnet's FK position by <1um,
# i.e. the magnet's local Z axis IS the flange's local Z axis exactly.
TCP_TO_MAGNET_POSE6 = (0.0, 0.0, 0.43, 0.0, 0.0, 0.0)

# Body-frame direction of the source magnet's own dipole moment
# (diametrically poled, ~90deg from body Z). Recalibrated 2026-10-02
# after the TCP_TO_MAGNET_POSE6 remount, re-derived from the original
# ground-truth calibration pose on 2026-10-04 and confirmed the
# derivation math reproduces this value to 0.07deg -- i.e. this is NOT
# what was wrong in the Oct-4 off-axis eyeball checks; the beam-base
# distance was (see BEAM_BASE_XYZ_M above).
SOURCE_DIPOLE_BODY_AXIS = (0.910293410, -0.413963475, 0.000386004)

# A known-good, reachable magnet pose used ONLY as an IK seed / model-bundle
# placeholder when a script needs *some* valid starting pose before it
# immediately overwrites it with the real target via model.solve(p7) or a
# fresh IK call. It is not a target itself -- do not read physical meaning
# into it beyond "reachable." Found byte-identical, independently
# copy-pasted into 9 vessel_stage_a scripts on 2026-10-05; centralized here
# so the next script that needs a seed pose imports it instead of adding a
# 10th copy.
REFERENCE_MAGNET_POSE6 = np.array([
    0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
    -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929,
])

# ---------------------------------------------------------------------------
# Rig networking / hardware addresses
# ---------------------------------------------------------------------------
ROBOT_IP = "192.168.56.101"
ADVANCER_PORT = "/dev/ttyACM0"

# ---------------------------------------------------------------------------
# Vessel geometry
# ---------------------------------------------------------------------------
# The current, correctly Z-calibrated digitized vessel lumen file (frame R,
# matches today's BEAM_BASE_XYZ_M -- see build_vessel_plan.py's
# BEAM_BASE_PIVOT_Z safety check, which refuses any lumen file whose Z
# differs from this by >2mm). When the vessel is re-digitized, update ONLY
# this one path; every script that imports CURRENT_LUMEN_FILE picks up the
# change automatically instead of needing its own hardcoded filename edited.
# 2026-10-06: physical vessel wall moved; re-digitized from
# vessel_lumen_robot_frame.json (raw, stale Z=-0.016567 like every prior
# digitization) and Z-corrected the same way as the 2026-10-05 file -- see
# vessel_lumen_robot_frame_2026-10-06_zcorrected.json's own git history.
CURRENT_LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_2026-10-06_zcorrected.json"

# ---------------------------------------------------------------------------
# Safety margins
# ---------------------------------------------------------------------------
# The base/default magnet-exclusion safety radius (how close the source
# magnet is allowed to get to the beam base during transit) most campaigns
# use. Some campaigns deliberately raise this for their own reasons (see
# that script's own comment for why) -- those are legitimate per-campaign
# margins layered ON TOP of this base value, not independent
# re-measurements of the same physical quantity. Don't reuse this name for
# a script-specific margin; import it and add to it instead.
MAGNET_EXCLUSION_RADIUS_BASE_M = 0.080
