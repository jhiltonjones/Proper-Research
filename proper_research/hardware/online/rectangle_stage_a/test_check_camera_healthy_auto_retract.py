"""Tests for check_camera_healthy's 2026-11-17 auto-retract behaviour.
Mocked camera/advancer -- no real hardware, no real camera device, no
real advancer motion.

Covers:
  (a) already within tolerance -> auto-retract never triggered.
  (b) out of tolerance -> auto-retract triggered, reset_insertion called
      with the correct target, and a good post-retract measurement lets
      the check pass.
  (c) still out of tolerance after retract -> a clear error is still
      raised, not silently swallowed.
"""
import sys
from unittest import mock

import numpy as np

from proper_research.hardware.online.rectangle_stage_a import common
from proper_research.hardware.online.advancer_excitation.measure_l0 import (
    PIVOT_XYZ,
    apply_affine_insertion_correction,
)


def _raw_mm_for_true(true_mm: float) -> float:
    """Invert the 2026-11-17 affine correction: these fakes set the RAW
    (uncorrected) chord length via tip_position_m, but check_camera_healthy
    now applies apply_affine_insertion_correction before comparing against
    expected_insertion_m, so the fixture has to supply the raw value that
    maps to the intended true value, not the true value itself."""
    corrected_at_zero = apply_affine_insertion_correction(0.0)
    corrected_at_one = apply_affine_insertion_correction(1.0)
    slope = corrected_at_one - corrected_at_zero
    return (true_mm - corrected_at_zero) / slope


class _FakeEstimate:
    def __init__(self, length_m: float):
        self.tip_position_m = PIVOT_XYZ + np.array([length_m, 0.0, 0.0])


class _FakeCamera:
    """Replaces CameraSource. `.latest()` always returns a valid estimate
    at whatever length `length_m_getter()` currently returns."""

    def __init__(self, *args, length_m_getter=None, **kwargs):
        self._length_m_getter = length_m_getter or (lambda: 0.025)

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self, max_age_s):
        return _FakeEstimate(self._length_m_getter()), 0.01


def _patched(length_m_getter):
    """Context manager stack: patch CameraSource, NewFrameTipMapper, and
    time.sleep (to keep the test fast -- no real waiting needed)."""
    return (
        mock.patch(
            "proper_research.hardware.online.camera_source.CameraSource",
            lambda *a, **k: _FakeCamera(*a, length_m_getter=length_m_getter, **k),
        ),
        mock.patch("proper_research.hardware.online.state_stream.NewFrameTipMapper", lambda scfg: object()),
        mock.patch("proper_research.hardware.online.rectangle_stage_a.common.time.sleep", lambda s: None),
    )


def test_a_already_within_tolerance_auto_retract_never_triggered():
    with mock.patch(
        "proper_research.hardware.online.advancer_excitation.reset_insertion.reset_insertion"
    ) as mock_reset:
        p1, p2, p3 = _patched(lambda: _raw_mm_for_true(25.0) / 1000.0)
        with p1, p2, p3:
            common.check_camera_healthy(
                n_frames=3, expected_insertion_m=0.025, insertion_tol_mm=1.0,
                auto_retract=True,
            )
        mock_reset.assert_not_called()


def test_b_out_of_tolerance_triggers_retract_with_correct_target_then_passes():
    state = {"length_m": 0.034}  # starts bad, like the real 34mm-vs-25mm case

    def fake_reset_insertion(target_mm, *, live, tolerance_mm=0.3, max_iters=5, **kw):
        # Simulate a successful physical retract: the NEXT measurement
        # will now read (after correction) at the target -- the real
        # reset_insertion closes its loop on the corrected measurement too.
        state["length_m"] = _raw_mm_for_true(target_mm) / 1000.0
        return target_mm

    with mock.patch(
        "proper_research.hardware.online.advancer_excitation.reset_insertion.reset_insertion",
        side_effect=fake_reset_insertion,
    ) as mock_reset:
        p1, p2, p3 = _patched(lambda: state["length_m"])
        with p1, p2, p3:
            common.check_camera_healthy(
                n_frames=3, expected_insertion_m=0.025, insertion_tol_mm=1.0,
                auto_retract=True, auto_retract_max_iters=5,
            )
        mock_reset.assert_called_once()
        called_target_mm = mock_reset.call_args[0][0]
        assert abs(called_target_mm - 25.0) < 1e-6, called_target_mm
        assert mock_reset.call_args.kwargs.get("live") is True


def test_c_still_out_of_tolerance_after_retract_raises_clear_error():
    def fake_reset_insertion(target_mm, *, live, tolerance_mm=0.3, max_iters=5, **kw):
        # Simulate a FAILED retract: length stays bad no matter what.
        return 34.0  # reset_insertion's own reported final length, unchanged

    with mock.patch(
        "proper_research.hardware.online.advancer_excitation.reset_insertion.reset_insertion",
        side_effect=fake_reset_insertion,
    ) as mock_reset:
        p1, p2, p3 = _patched(lambda: 0.034)  # never improves
        with p1, p2, p3:
            try:
                common.check_camera_healthy(
                    n_frames=3, expected_insertion_m=0.025, insertion_tol_mm=1.0,
                    auto_retract=True, auto_retract_max_iters=5,
                )
                raised = False
            except RuntimeError as exc:
                raised = True
                msg = str(exc)
        assert raised, "expected a RuntimeError when auto-retract does not converge"
        assert "did not converge" in msg
        mock_reset.assert_called_once()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
