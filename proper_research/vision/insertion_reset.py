from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    reset_insertion_to_target,
)
from pathlib import Path
reset_insertion_to_target(30.44, live=True, out_dir=Path("/tmp"))
