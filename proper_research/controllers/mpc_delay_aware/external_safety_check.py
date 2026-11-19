"""Shared pre-dispatch external safety check, extracted (2026-11-19) from
`process_isolated_adapter.py`'s early-abort gate so a non-MPC controller
(e.g. the inverse-Jacobian comparator, INV) can run the SAME check without
acquiring MPC's predictive QP or any optimization layer.

This is deliberately a MEASURED-STATE check only: it reads q6 (this tick's
own measured joints, mapped through magnet_transform_fn to the magnet's
xyz) and nothing about any candidate command about to be dispatched. It
does not examine, and cannot substitute for, a predictive (candidate-
command) constraint such as MPC's in-QP magnet-exclusion rows -- those
evaluate the controller's own planned future commands; this evaluates only
where the robot measurably already is. A sampled check at the control
period (10 Hz in this rig): it can only reject what shows up at a sample
instant, so it is a breach detector for this check's own cadence, not a
continuous-time safety guarantee.

Lives in `controllers/mpc_delay_aware/` (not a `hardware/online/*` script
module) so both layers can import it: hardware scripts already import
from controllers (the existing direction throughout this codebase), and
process_isolated_adapter.py importing a hardware-script's `common.py`
would be a backwards dependency with circular-import risk.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray

__all__ = ["check_measured_external_safety"]


def check_measured_external_safety(
    q6: Array,
    magnet_transform_fn: "Callable[[Array], Array] | None",
    *,
    magnet_z_bounds_m: "tuple[float, float] | None" = None,
    magnet_exclusion_radius_m: "float | None" = None,
    lumen_C_m: "Array | None" = None,
) -> "tuple[str | None, float | None]":
    """Returns (reason, gap_m).

    `reason` is None if nothing tripped, else a short tripped-rule string
    matching the MPC adapter's own convention. `gap_m` is the measured
    clearance to the lumen centreline (None if the exclusion-radius check
    is not configured), reported unconditionally when computed, not only
    on the tripping branch, so a caller can log the approach even on ticks
    that do not trip.
    """
    if magnet_transform_fn is None:
        return None, None
    magnet_xyz = np.asarray(magnet_transform_fn(q6), dtype=float).reshape(3)
    reason = None
    gap_m = None
    if magnet_z_bounds_m is not None:
        z_min, z_max = magnet_z_bounds_m
        if magnet_xyz[2] < z_min or magnet_xyz[2] > z_max:
            reason = (
                f"magnet_z_out_of_bounds(z={magnet_xyz[2]*1e3:.1f}mm, "
                f"bounds=[{z_min*1e3:.1f},{z_max*1e3:.1f}]mm)"
            )
    if magnet_exclusion_radius_m is not None and lumen_C_m is not None:
        gap_m = float(np.min(np.linalg.norm(
            np.asarray(lumen_C_m, dtype=float) - magnet_xyz[None, :], axis=1
        )))
        if reason is None and gap_m < magnet_exclusion_radius_m:
            reason = (
                f"magnet_exclusion_violated(gap={gap_m*1e3:.1f}mm < "
                f"{magnet_exclusion_radius_m*1e3:.1f}mm)"
            )
    return reason, gap_m
