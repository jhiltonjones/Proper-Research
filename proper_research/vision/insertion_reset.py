"""Originally a VESSEL-rig tool (called vessel_stage_a's
checkpoint_beam_shape_campaign.reset_insertion_to_target, which uses the
raised +30mm pivot). CONVERTED to a RECTANGLE-rig tool on 2026-11-17, per
explicit user request, so this drives to/reports against the same
unraised pivot as the rectangle hardware run's own
check_camera_healthy / advancer_excitation.reset_insertion.

This file is NO LONGER VALID for actual vessel-rig work. For real
vessel-rig insertion reset, call
checkpoint_beam_shape_campaign.reset_insertion_to_target directly, not
this script.

Real advancer motion when run with live=True (as below) -- this is the
same convention the rectangle advancer_excitation.reset_insertion tool
uses.
"""
from proper_research.hardware.online.advancer_excitation.reset_insertion import (
    reset_insertion,
)

reset_insertion(35.0, live=True)
