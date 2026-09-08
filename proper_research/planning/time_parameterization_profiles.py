"""Standard and time-optimal speed profiles for the same geometric path.

The existing time parameteriser is time-*optimal*: it computes the backward
controllable set at every grid point, then runs forward at the maximum
admissible path acceleration.  That is the right default, and it is also the
reason a smoother geometric path is supposed to traverse faster.

To *show* that, you need something to compare against.  This module adds a
**standard** profile — the textbook trapezoidal accelerate/cruise/decelerate
schedule most path-following implementations use — computed from the same
caps, on the same grid, and fed through the same downstream code.  Only the
choice of ``s_dot(s)`` differs, so a duration difference between the two is
attributable to the profile and nothing else.

Why a trapezoid is the honest baseline
--------------------------------------
It is what you get without a TOPP implementation: pick one cruise speed that
respects every actuator everywhere, ramp into it at one constant path
acceleration that also respects every actuator everywhere, ramp out
symmetrically.  It is feasible by construction and it is conservative by
construction — a single worst-case speed for the whole path rather than a
speed that follows the local limit.  The gap between the two is exactly the
value the optimal parameteriser adds.

Both profiles return ``(squared_speed, path_acceleration, controllable)``, the
signature of the module's own ``_reachability_parameterization``, so the swap
is one function and nothing downstream can tell the difference.

Usage
-----
    from time_parameterization_profiles import profile_scope

    with profile_scope("standard"):
        result = parameterize_configuration_path(path=path, config=config, ...)

or use ``run_time_parameterization.py --profile both``, which runs each in turn
and reports the traversal-time difference.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np

try:
    from proper_research.planning import time_parameterized_configuration_path as _tp
except ModuleNotFoundError:  # pragma: no cover - standalone review
    import time_parameterized_configuration_path as _tp  # type: ignore

Array = np.ndarray

PROFILES = ("optimal", "standard")


def standard_trapezoidal_parameterization(
    *,
    s_grid: Array,
    path_first: Array,
    path_second: Array,
    config: Any,
) -> tuple[Array, Array, list[tuple[float, float]]]:
    """Trapezoidal speed profile: one cruise speed, one ramp acceleration.

    Feasibility is enforced the same way the optimal profile enforces it, using
    the module's own cap and local-acceleration routines:

    * the cruise squared speed is the **minimum** velocity cap over the whole
      path, so no actuator velocity limit can be exceeded anywhere;
    * the ramp acceleration is the **minimum** admissible ``|u| = |d(s_dot^2)/ds| / 2``
      over the path evaluated at the cruise speed, so no actuator acceleration
      limit can be exceeded anywhere either.

    Both are single numbers for the whole path.  That is what makes this the
    standard profile rather than the optimal one: it never exploits the fact
    that most of the path is far easier than its hardest point.
    """
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    first = np.asarray(path_first, dtype=float).reshape(s_grid.size, -1)
    second = np.asarray(path_second, dtype=float).reshape(s_grid.size, -1)
    ds = np.diff(s_grid)
    if np.any(ds <= 0.0):
        raise ValueError("Timing grid must be strictly increasing.")

    velocity_limit = config.effective_velocity_limit
    acceleration_limit = config.effective_acceleration_limit
    tolerance = float(config.linear_program_tolerance)
    caps = _tp._velocity_squared_caps(
        first, velocity_limit, config.maximum_path_speed_m_s
    )
    start_x = float(config.start_path_speed_m_s) ** 2
    end_x = float(config.end_path_speed_m_s) ** 2
    if start_x > caps[0] + tolerance or end_x > caps[-1] + tolerance:
        raise RuntimeError("An endpoint path speed exceeds its velocity cap.")

    # The velocity caps only bound each actuator's *speed*. On a sharply
    # curved path the centripetal term (path_second * x, the acceleration
    # needed just to stay ON the path at squared-speed x) can already bust an
    # actuator's acceleration limit at a speed the velocity cap still allows,
    # leaving _acceleration_interval_at_speed with no feasible path
    # acceleration at all. The optimal profile handles that by slowing down
    # locally; the honest trapezoid handles it by choosing a lower global
    # cruise speed. So cap the cruise speed at each point by the largest
    # squared-speed that still admits some path acceleration, found by
    # bisection, and take the global minimum.
    def _acceleration_feasible_cap(index: int, ceiling: float) -> float:
        try:
            _tp._acceleration_interval_at_speed(
                path_first=first[index], path_second=second[index],
                squared_speed=ceiling, acceleration_limit=acceleration_limit,
                maximum_path_acceleration=config.maximum_path_acceleration_m_s2,
                tolerance=tolerance,
            )
            return ceiling
        except RuntimeError:
            pass
        low, high = 0.0, ceiling
        for _ in range(60):
            mid = 0.5 * (low + high)
            try:
                _tp._acceleration_interval_at_speed(
                    path_first=first[index], path_second=second[index],
                    squared_speed=mid, acceleration_limit=acceleration_limit,
                    maximum_path_acceleration=config.maximum_path_acceleration_m_s2,
                    tolerance=tolerance,
                )
                low = mid
            except RuntimeError:
                high = mid
        return low

    accel_caps = np.array([
        _acceleration_feasible_cap(index, float(caps[index]))
        for index in range(s_grid.size)
    ])
    caps = np.minimum(caps, accel_caps)

    cruise_x = float(np.min(caps))
    if cruise_x <= max(start_x, end_x) + tolerance:
        # The path cannot go faster anywhere than it must at its endpoints:
        # there is no cruise phase to build, so hold the endpoint speed.
        cruise_x = max(cruise_x, start_x, end_x)

    # One ramp acceleration that every point admits at the cruise speed.
    ramp = float(config.maximum_path_acceleration_m_s2)
    for index in range(s_grid.size):
        probe_x = min(cruise_x, caps[index])
        lower_u, upper_u = _tp._acceleration_interval_at_speed(
            path_first=first[index],
            path_second=second[index],
            squared_speed=probe_x,
            acceleration_limit=acceleration_limit,
            maximum_path_acceleration=config.maximum_path_acceleration_m_s2,
            tolerance=tolerance,
        )
        admissible = min(abs(lower_u), abs(upper_u))
        if not np.isfinite(admissible):
            continue
        ramp = min(ramp, float(admissible))
    if ramp <= 0.0:
        raise RuntimeError(
            "No positive path acceleration is admissible everywhere at the "
            "standard cruise speed; this path needs the optimal profile."
        )

    # x(s) = s_dot^2 obeys dx/ds = 2u, so a constant ramp is linear in x.
    distance = s_grid - s_grid[0]
    remaining = s_grid[-1] - s_grid
    squared_speed = np.minimum(
        np.minimum(start_x + 2.0 * ramp * distance, end_x + 2.0 * ramp * remaining),
        np.minimum(cruise_x, caps),
    )
    squared_speed = np.maximum(squared_speed, 0.0)
    squared_speed[0] = start_x
    squared_speed[-1] = end_x

    # A speed of exactly zero in the interior would make the traversal time
    # infinite; the caps guarantee this only at the endpoints.
    interior = squared_speed[1:-1]
    if interior.size and np.any(interior <= 0.0):
        raise RuntimeError(
            "The standard profile reached zero speed inside the path. Raise "
            "maximum_path_acceleration_m_s2 or the endpoint speeds."
        )

    path_acceleration = np.diff(squared_speed) / (2.0 * ds)
    controllable = [(float(x), float(x)) for x in squared_speed]
    return squared_speed, path_acceleration, controllable


@contextlib.contextmanager
def profile_scope(profile: str) -> Iterator[None]:
    """Run the parameteriser with the chosen speed profile.

    ``"optimal"`` is a no-op — it is what the module already does.
    ``"standard"`` swaps the reachability computation for the trapezoid and
    restores it afterwards, so nothing outside the block is affected.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}; got {profile!r}.")
    if profile == "optimal":
        yield
        return
    original = _tp._reachability_parameterization
    _tp._reachability_parameterization = standard_trapezoidal_parameterization
    try:
        yield
    finally:
        _tp._reachability_parameterization = original


def describe(profile: str) -> str:
    return {
        "optimal": (
            "time-optimal: backward controllable sets, then forward at the "
            "maximum admissible path acceleration"
        ),
        "standard": (
            "standard trapezoidal: one globally feasible cruise speed and one "
            "globally feasible ramp acceleration"
        ),
    }[profile]


__all__ = ["PROFILES", "describe", "profile_scope", "standard_trapezoidal_parameterization"]
