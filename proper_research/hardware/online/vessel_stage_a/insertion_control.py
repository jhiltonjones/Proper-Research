"""Shared real-hardware insertion-length control.

The advancer has no encoder: nothing about the physical beam insertion
changes unless this module's `closed_loop_insertion` (or equivalent)
actually commands `AdvancerUnit.forward`/`backward`. Passing a target L
into a model or into `CameraSource(insertion_length_getter=...)` is pure
bookkeeping for software and has zero effect on the real rig -- that
confusion is what caused the Stage-1 L=30mm/L=40mm Jacobian campaigns on
2026-10-05 to silently both run at whatever insertion was already
physically present (~25mm), discovered only because the camera-measured
Jacobian columns barely changed between the two "different" L campaigns.

Mirrors the pattern already used in fixed_dipole_arc_sweep.py /
fixed_dipole_arc_pilot.py (duplicated there); this is the de-duplicated
version for reuse by any script that needs to set insertion length.
"""
import time

import numpy as np


def chord_mm(camera, beam_base_xyz, *, n=15, timeout_s=8.0, retries=3):
    """Vision-measured straight-line distance from beam base to tip (mm).

    A proxy for arc-length insertion; good to within ~1mm at this rig's
    weak lateral deflection (<=3-4mm), so adequate as a closed-loop target.
    """
    for attempt in range(retries):
        tips = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s and len(tips) < n:
            est, _age = camera.latest(0.5)
            if est is not None:
                tip = np.asarray(est.tip_position_m, dtype=float)
                if np.all(np.isfinite(tip)):
                    tips.append(tip)
            time.sleep(0.1)
        if tips:
            tip_med = np.median(np.vstack(tips), axis=0)
            return float(np.linalg.norm(tip_med - beam_base_xyz) * 1000.0)
        print(f"    [chord_mm] no detections, retry {attempt + 1}/{retries}")
        time.sleep(1.0)
    return None


def closed_loop_insertion(
    camera, advancer, target_mm, beam_base_xyz, *,
    tol_mm=0.5, max_iters=10, max_step_mm=10.0, assumed_ratio=0.85,
):
    """Drive the physical advancer until vision-measured chord_mm reaches
    target_mm. Returns (converged_L_mm, assumed_ratio) so assumed_ratio can
    be carried across calls (it adapts to the rig's true realization gain).
    """
    current = chord_mm(camera, beam_base_xyz)
    if current is None:
        print("    [insertion] WARNING: no camera detection at start -- skipping convergence")
        return None, assumed_ratio
    print(f"    [insertion] start L~{current:.2f}mm target={target_mm:.2f}mm")
    for it in range(max_iters):
        delta = target_mm - current
        if abs(delta) <= tol_mm:
            print(f"    [insertion] within tolerance after {it} correction(s)")
            break
        cmd_mm = float(np.clip(delta / assumed_ratio, -max_step_mm, max_step_mm))
        if cmd_mm > 0:
            advancer.forward(abs(cmd_mm), delay_us=25)
        else:
            advancer.backward(abs(cmd_mm), delay_us=25)
        time.sleep(1.0)
        new_current = chord_mm(camera, beam_base_xyz)
        if new_current is None:
            print("    [insertion] WARNING: lost camera detection mid-correction -- stopping here")
            break
        actual_delta = new_current - current
        if abs(cmd_mm) > 0.5:
            observed_ratio = actual_delta / cmd_mm
            assumed_ratio = float(np.clip(0.5 * assumed_ratio + 0.5 * np.clip(observed_ratio, 0.2, 1.5), 0.2, 1.5))
        current = new_current
        print(f"    [insertion] iter {it + 1}: L~{current:.2f}mm (target {target_mm:.2f}mm, assumed_ratio={assumed_ratio:.2f})")
    else:
        print(f"    [insertion] WARNING: did not converge, final L~{current:.2f}mm")
    return current, assumed_ratio
