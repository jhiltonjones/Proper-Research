"""2026-11-19: verifies the plan-segment loader against the real marginfix
plan used by the A/B/C pilot runs. Offline, no hardware -- reads the
already-built .npz artifact only.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import _robot_kin
from proper_research.hardware.online.stopping_validation_plan_segment import (
    extract_segment, find_representative_segment, load_plan_npz, BEAM_BASE_XYZ_M,
)

PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path"


def _fk(q6):
    fk = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return fk.T_R_target[:3, 3]


def test_plan_loads_and_units_verified():
    d = load_plan_npz(PLAN_DIR)
    assert d["state"].shape == (230, 7)
    dt = np.diff(d["time_s"])
    assert np.allclose(dt, 0.1, atol=1e-6), "sample period must match the plan's own sample_period_s=0.1"


def test_rejects_wrong_column_count():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        bad = np.zeros((10, 6))  # missing the insertion column
        np.savez(Path(td) / "time_parameterized_configuration_path.npz",
                  time_s=np.arange(10) * 0.1, state=bad, state_rate=bad)
        with pytest.raises(ValueError, match="7 columns"):
            load_plan_npz(td)


def test_find_representative_segment_is_nonstationary_and_in_range():
    i0, i1 = find_representative_segment(PLAN_DIR, _fk, window_samples=30)
    d = load_plan_npz(PLAN_DIR)
    n = d["state"].shape[0]
    assert 0 <= i0 < i1 < n
    assert i1 - i0 + 1 == 30


def test_extracted_segment_matches_plan_and_preserves_clearance():
    i0, i1 = find_representative_segment(PLAN_DIR, _fk, window_samples=30)
    seg = extract_segment(PLAN_DIR, i0, i1, _fk)
    d = load_plan_npz(PLAN_DIR)

    assert np.allclose(seg.joints_rad[0], d["state"][i0, :6])
    assert np.allclose(seg.insertion_m, d["state"][i0:i1 + 1, 6])
    assert seg.n == i1 - i0 + 1
    assert seg.duration_s == pytest.approx(d["time_s"][i1] - d["time_s"][i0])

    # The whole plan must stay clear of the 233mm commanded / 230mm reviewed
    # boundary -- not asserted by assumption, computed from the real FK
    # against the same BEAM_BASE constant used throughout this review.
    full_joints = d["state"][:, :6]
    full_magnet = np.array([_fk(q) for q in full_joints])
    full_clearance = np.linalg.norm(full_magnet - BEAM_BASE_XYZ_M[None, :], axis=1)
    assert full_clearance.min() > 0.233, (
        f"plan comes within {full_clearance.min()*1e3:.1f}mm of the beam base -- "
        "closer than the 233mm commanded boundary; segment choice must account for this"
    )
    assert seg.clearance_m.min() >= full_clearance.min() * 0.999  # segment's own clearance is consistent with the plan


def test_summary_table_reports_separate_speed_figures_not_one_number():
    i0, i1 = find_representative_segment(PLAN_DIR, _fk, window_samples=30)
    seg = extract_segment(PLAN_DIR, i0, i1, _fk)
    table = seg.summary_table()
    for key in ("inward_mean_mm_s", "inward_max_mm_s", "inward_min_mm_s", "cartesian_max_mm_s"):
        assert key in table
    assert "total_speed_mm_s" not in table


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
