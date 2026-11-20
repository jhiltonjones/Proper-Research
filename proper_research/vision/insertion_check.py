"""Originally a VESSEL-rig tool (used vessel_stage_a's raised +30mm pivot
via checkpoint_beam_shape_campaign._build_raised_camera_mapper).
CONVERTED to a RECTANGLE-rig tool on 2026-11-17, per explicit user
request, so this reports the same number as the rectangle hardware run's
own check_camera_healthy / advancer_excitation.reset_insertion -- those
two chains had been giving inconsistent readings because this file was
using the wrong (vessel) pivot reference, 30mm offset in Z from the
rectangle rig's actual pivot.

This file is NO LONGER VALID for actual vessel-rig work (it no longer
applies the raised pivot that rig needs). For real vessel-rig insertion
measurement, use checkpoint_beam_shape_campaign.measure_current_insertion_mm
with _build_raised_camera_mapper directly, not this script.
"""
import time

from proper_research.hardware.online.advancer_excitation.measure_l0 import (
    build_measurement_camera,
    measure_l_mm,
)

camera, mapper = build_measurement_camera()
camera.start()
time.sleep(1.0)
print("live insertion (mm):", float(measure_l_mm(camera, mapper, n=10, timeout_s=20.0).mean()))
camera.stop()
