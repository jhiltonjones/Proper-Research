"""2026-11-19: regression test for _wrapped_build_offline_solver() itself.
Two real NameErrors have now been found at runtime in this module-level
callback function (--protective-config-reviewed referencing `args`, then
the measured-clearance monitor referencing main()'s local `_beam_base_xyz_r`)
-- both were cross-scope reference bugs that only surfaced on an actual
hardware run, because nothing exercised this function directly offline.
This test calls it directly with minimal stand-ins for the heavy runtime
state (_WORKER, _LAST_OUTPUT_DIR, pf.CONFIG, the adapter class itself),
isolating exactly the scoping question: does the function run to completion
without NameError. No hardware, no real worker, no real schedule.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.rectangle_stage_a import run_mpc_delay_aware_insertion_anchor as mod


class _DummyConfig:
    joint_velocity_limit_rad_s = 0.1
    max_joint_step_rad = 0.01
    plan_dir = "/nonexistent"


class _CapturingAdapter:
    """Stand-in for ProcessIsolatedDelayAwareAdapter: records every kwarg
    it was constructed with instead of building a real controller."""
    last_kwargs = None

    def __init__(self, **kwargs):
        _CapturingAdapter.last_kwargs = kwargs


class _DummyWorker:
    def set_frame_transform(self, r_fit, t_fit):
        pass


def test_wrapped_build_offline_solver_reaches_adapter_without_nameerror(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "ProcessIsolatedDelayAwareAdapter", _CapturingAdapter)
    monkeypatch.setattr(mod, "_METADATA_WRITTEN", True)  # skip the metadata-write block; unrelated to this bug class
    monkeypatch.setattr(mod, "_LAST_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "_WORKER", _DummyWorker())
    monkeypatch.setattr(mod.pf, "_PLANNER_TO_LIVE_TRANSFORM", (np.eye(3), np.zeros(3)))
    monkeypatch.setattr(mod.pf, "CONFIG", _DummyConfig())

    # The function continues past the adapter construction into real
    # file I/O (load_configuration_reference(cfg.plan_dir, ...)) that our
    # minimal stub config can't satisfy -- that failure is expected and
    # unrelated to the scoping bug class this test targets. What matters
    # is that the adapter got constructed (a side effect captured on
    # _CapturingAdapter) BEFORE any such unrelated failure, and that
    # nothing along the way raised NameError.
    try:
        mod._wrapped_build_offline_solver("mpc_delay_aware", reference="stub_reference", adapter_config=None)
    except NameError as exc:
        pytest.fail(f"NameError reached (the exact bug class this test guards against): {exc}")
    except Exception:
        pass  # expected: downstream file I/O on our stub plan_dir fails

    kwargs = _CapturingAdapter.last_kwargs
    assert kwargs is not None, "adapter was never constructed -- function returned/raised before reaching it"
    assert kwargs["reference"] == "stub_reference"
    assert kwargs["magnet_exclusion_lumen_C_m"].shape == (1, 3)
    # 2026-11-19: default changed from the unapproved 250mm proposal to the
    # already-reviewed 230mm boundary (mod._MEASURED_CLEARANCE_ABORT_MM,
    # set from --measured-clearance-abort-mm, defaulting to
    # common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM) -- see this module's
    # own test_measured_clearance_abort_mm.py for the boundary-case tests.
    assert kwargs["magnet_exclusion_radius_m"] == pytest.approx(
        mod._MEASURED_CLEARANCE_ABORT_MM * 1e-3
    )
    assert callable(kwargs["magnet_transform_fn"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
