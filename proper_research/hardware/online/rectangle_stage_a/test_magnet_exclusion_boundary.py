"""2026-11-19 (stale-default correction): guards the rectangle plan
family's shared magnet-to-beam-base exclusion boundary/margin constants
against silent drift back to the old, stale 97.0mm value (the bug this
turn fixed) or any other unreviewed change -- any future edit to these
numbers must touch this test, not slip through silently.

Run: python3 -m pytest proper_research/hardware/online/rectangle_stage_a/test_magnet_exclusion_boundary.py -v
"""
from __future__ import annotations

from proper_research.hardware.online.rectangle_stage_a import common


def test_physical_floor_is_the_reviewed_230mm_boundary():
    assert common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM == 230.0


def test_robust_margin_is_the_provisional_3mm_value():
    assert common.MAGNET_EXCLUSION_ROBUST_MARGIN_MM == 3.0


def test_commanded_boundary_is_floor_plus_margin_233mm():
    assert (
        common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + common.MAGNET_EXCLUSION_ROBUST_MARGIN_MM
        == 233.0
    )


def test_not_the_old_stale_vessel_value():
    """The bug this fixed: the rectangle runner's old default was 97.0mm,
    a value correctly derived for a different, older vessel campaign."""
    assert common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM != 97.0


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
