from pathlib import Path
from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    measure_current_insertion_mm, _build_raised_camera_mapper,
)
camera, mapper, scfg = _build_raised_camera_mapper(Path("/tmp"))
camera.start()
import time; time.sleep(1.0)
print("live insertion (mm):", measure_current_insertion_mm(camera, scfg))
camera.stop()
