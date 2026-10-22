"""Measure where the analytical beam-Jacobian cost actually goes.

Purpose
-------
Before committing to a decomposed offline/online sensitivity in the paper, we
need one number: **what fraction of the analytical Jacobian's wall time is the
Hessian build?**  If it dominates, the decomposition is the contribution.  If it
does not, the honest paper is simpler and the switching claim should be dropped.

This module answers that using *your* analytical Jacobian only.  It never
finite-differences the output map.  Every Jacobian in every arm comes from

    controller_pack["plant_diagnostic_joint_adapter"].continuous_output_jacobian(state)

which is the same interface used by ``offline_inverse_configuration``'s
``_NodeObjective.output_jacobian`` and by
``precompute_reference_beam_jacobians``.  Underneath, that is
``MagneticBeamForwardModelOptimized.jacobian_output_actuation_tangent`` ->
``implicit_tip_jacobian``.  The only finite differencing that happens is the one
already inside your own sensitivity code.

Arms
----
full
    Every call rebuilds the beam Hessian.  Implemented by clearing the model's
    stored ``last_sensitivity_H`` before each call, so ``H_override`` is None and
    your existing code takes the rebuild path.  This is the *non-decomposed*
    reference cost.

reuse
    Your current behaviour, untouched: ``mode="fast"`` lets
    ``_can_reuse_sensitivity_hessian`` supply ``H_override`` when the pose has
    barely moved and the contact mask is unchanged.  This is the binary
    all-or-nothing version of the decomposition.

decomposed
    Scaffolded, not implemented, because it needs one function out of
    ``sensitivity_optimized.py`` that is not in this workspace.  See
    ``DecomposedHessianHooks`` for the exact interface required.  Everything
    else -- state walk, timing, contact bookkeeping, accuracy comparison,
    reporting -- is already wired for it.

What to read off the results
----------------------------
* ``hessian_fraction_of_jacobian`` (summary): the decision number.  High means
  build the decomposition.  Low means simplify the paper.
* ``reuse_speedup``: what the crude binary reuse already buys.  The decomposition
  can only improve on this, so it is the floor of the achievable gain.
* ``reuse_relative_jacobian_error``: what that speed costs in Jacobian accuracy.
* The per-sample CSV columns ``contact_active_nodes`` and ``jacobian_wall_s``:
  plot one against the other.  If cost or reuse-failure rises with contact-set
  size, that is the empirical basis for the contact-triggered update policy.

Usage
-----
    python -m proper_research.analysis.jacobian_cost_benchmark \
        --states time_parameterized_configuration_path \
        --max-states 60 \
        --output-dir jacobian_cost_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------------
# Validation helpers, matching the conventions used elsewhere in the project.
# ---------------------------------------------------------------------------


def _finite_vector(value: Any, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != int(size) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain {size} finite values; got {result.shape}."
        )
    return result.copy()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Configuration and per-sample record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings for one cost-benchmark run.

    ``repeats_per_state`` exists because a single timing sample is noisy.  Note
    that repeats at the *same* state would hit the Jacobian value cache, so the
    benchmark clears that cache between repeats; it does not clear the forward
    equilibrium cache, because a real controller also arrives at a state with a
    warm equilibrium.
    """

    arms: tuple[str, ...] = ("full", "reuse")
    repeats_per_state: int = 3
    warmup_states: int = 2
    maximum_states: int | None = None
    state_stride: int = 1
    compare_against_arm: str = "full"
    debug: bool = True

    def validate(self) -> None:
        if not self.arms:
            raise ValueError("At least one arm must be requested.")
        for arm in self.arms:
            if arm not in KNOWN_ARMS:
                raise ValueError(
                    f"Unknown arm {arm!r}; expected one of {sorted(KNOWN_ARMS)}."
                )
        if self.repeats_per_state < 1:
            raise ValueError("repeats_per_state must be at least one.")
        if self.warmup_states < 0:
            raise ValueError("warmup_states cannot be negative.")
        if self.state_stride < 1:
            raise ValueError("state_stride must be at least one.")
        if self.maximum_states is not None and self.maximum_states < 1:
            raise ValueError("maximum_states must be positive when supplied.")
        if self.compare_against_arm not in self.arms:
            raise ValueError(
                "compare_against_arm must be one of the requested arms; "
                f"got {self.compare_against_arm!r} with arms={self.arms}."
            )


KNOWN_ARMS = {"full", "reuse", "decomposed"}


@dataclass
class JacobianCostSample:
    """One analytical-Jacobian evaluation with its cost breakdown."""

    arm: str
    state_index: int
    repeat_index: int

    # Wall time measured around the adapter call, i.e. robot FK + chain rule +
    # beam sensitivity.  This is the number a real controller pays.
    call_wall_s: float

    # Timings reported by your own sensitivity code, surfaced through
    # MagneticBeamForwardModelOptimized._build_last_jacobian_diag_from_sensitivity.
    jacobian_wall_s: float
    hessian_time_s: float
    gtheta_time_s: float
    linear_solve_time_s: float
    kinematics_sensitivity_time_s: float
    sensitivity_time_s: float
    nominal_gradient_time_s: float
    gradient_evaluations: float
    hessian_reused: bool
    cond_H_beam: float
    stationarity_norm: float

    # Contact bookkeeping, read from the committed forward-solve parts.
    contact_active_nodes: int
    contact_total_nodes: int
    gap_min_m: float

    # Filled in after all arms have run.
    relative_jacobian_error: float = float("nan")

    @property
    def hessian_fraction(self) -> float:
        if not np.isfinite(self.jacobian_wall_s) or self.jacobian_wall_s <= 0.0:
            return float("nan")
        if not np.isfinite(self.hessian_time_s):
            return float("nan")
        return float(self.hessian_time_s / self.jacobian_wall_s)


@dataclass
class BenchmarkResult:
    samples: list[JacobianCostSample]
    jacobians: dict[tuple[str, int], Array]
    configuration: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)
    output_files: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model cache manipulation.
#
# These reach into documented attributes of MagneticBeamForwardModelOptimized.
# They are deliberately narrow: nothing here changes the physics, the solver, or
# the Jacobian definition.  They only decide whether your existing code is
# allowed to reuse a previously built Hessian.
# ---------------------------------------------------------------------------


_HESSIAN_REUSE_ATTRIBUTES = (
    "last_sensitivity_H",
    "last_sensitivity_H_p7",
    "last_sensitivity_H_eps",
    "last_sensitivity_contact_mask",
)

_JACOBIAN_VALUE_CACHE_ATTRIBUTES = (
    "last_J_output_actuation_tangent",
    "last_J_output_pose7",
    "last_J_tip_actuation_tangent",
    "last_J_tip_pose7",
    "_last_tangent_key",
    "_last_pose7_key",
)


def clear_hessian_reuse(model: Any) -> None:
    """Force the next Jacobian call to rebuild the beam Hessian.

    ``jacobian_output_actuation_tangent`` sets ``H_override = self.last_sensitivity_H``
    only when fast mode and ``_can_reuse_sensitivity_hessian`` agree.  Clearing
    the stored Hessian makes ``H_override`` None regardless, so the rebuild path
    is taken without touching your reuse logic.
    """
    for name in _HESSIAN_REUSE_ATTRIBUTES:
        if hasattr(model, name):
            setattr(model, name, None)


def clear_jacobian_value_cache(model: Any) -> None:
    """Force recomputation even when the pose and options are unchanged."""
    for name in _JACOBIAN_VALUE_CACHE_ATTRIBUTES:
        if hasattr(model, name):
            setattr(model, name, None)


def _resolve_model(adapter: Any) -> Any:
    model = getattr(adapter, "model", None)
    if model is None:
        raise AttributeError(
            "The diagnostic adapter does not expose .model, so Hessian reuse "
            "cannot be controlled. Pass the adapter built by "
            "controller_factory_joint_space.build_controller, which sets "
            "model=owned_plant_model."
        )
    return model


# ---------------------------------------------------------------------------
# Diagnostics harvesting.
# ---------------------------------------------------------------------------


def _collect_diagnostics(adapter: Any) -> dict[str, Any]:
    """Merge adapter and model diagnostics.

    This mirrors ``offline_inverse_configuration._collect_diagnostics`` so the
    benchmark reads exactly the same keys the planner does.
    """
    diagnostics: dict[str, Any] = {}

    getter = getattr(adapter, "get_last_diag", None)
    if callable(getter):
        try:
            diagnostics.update(dict(getter() or {}))
        except Exception:
            pass

    model = getattr(adapter, "model", None)
    model_getter = getattr(model, "get_last_jacobian_diag", None)
    if callable(model_getter):
        try:
            diagnostics.update(dict(model_getter() or {}))
        except Exception:
            pass

    model_diag = getattr(model, "last_jacobian_diag", None)
    if isinstance(model_diag, dict):
        diagnostics.update(model_diag)

    sens_info = getattr(model, "last_sens_info", None)
    if isinstance(sens_info, dict):
        for key, value in sens_info.items():
            diagnostics.setdefault(key, value)

    return diagnostics


def _diagnostic_float(diagnostics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in diagnostics:
            continue
        try:
            value = float(
                np.asarray(diagnostics[key], dtype=float).reshape(-1)[0]
            )
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _contact_state_from_model(model: Any) -> tuple[int, int, float]:
    """Return (active contact nodes, total nodes, minimum gap).

    ``energy_optimized._contact_energy_from_state`` stores ``gap_nodes``,
    ``contact_weights`` and ``gap_min`` in ``parts`` whenever ``result_detail``
    is "contact" or "full", and ``_commit_result`` keeps ``info["parts"]``.  A
    node counts as active when its contact weight is positive.
    """
    cache = getattr(model, "cache", None)
    info = getattr(cache, "info", None) if cache is not None else None
    if not isinstance(info, dict):
        return -1, -1, float("nan")

    parts = info.get("parts")
    if not isinstance(parts, dict):
        return -1, -1, float("nan")

    gap_min = float("nan")
    if "gap_min" in parts:
        try:
            gap_min = float(parts["gap_min"])
        except Exception:
            gap_min = float("nan")

    weights = parts.get("contact_weights")
    if weights is None:
        gaps = parts.get("gap_nodes")
        if gaps is None:
            return -1, -1, gap_min
        gaps_array = np.asarray(gaps, dtype=float).reshape(-1)
        return int(np.sum(gaps_array < 0.0)), int(gaps_array.size), gap_min

    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    return (
        int(np.sum(weights_array > 0.0)),
        int(weights_array.size),
        gap_min,
    )


# ---------------------------------------------------------------------------
# The decomposed arm: interface specification.
# ---------------------------------------------------------------------------


@dataclass
class DecomposedHessianHooks:
    """What is required to enable the ``decomposed`` arm.

    The decomposition splits the beam Hessian by energy term.  From
    ``energy_optimized.energy_from_u_optimized`` the total potential is

        W_total = W_el + W_m + W_g + W_cf

    and the corresponding Hessian split is

        H_total(u, theta) = H_elastic(L) + H_magnetic(u, theta) + H_contact(u)

    The intended scheme evaluates ``H_elastic + H_magnetic`` offline along the
    certified reference and assembles ``H_contact`` online, so only the term
    that switches is recomputed.

    To wire this up, supply the two callables below out of
    ``sensitivity_optimized.py``.  They are deliberately expressed in terms of
    your existing objects so no new differencing scheme is introduced -- the
    same code that builds ``H`` today builds both of these, with the lumen term
    switched on and off.

    build_hessian:
        ``build_hessian(solution, problem, theta_model, options, use_lumen) -> H``
        Builds the beam Hessian at the cached equilibrium using exactly the path
        ``implicit_tip_jacobian`` uses today.  ``use_lumen=False`` must suppress
        the contact term in the energy (the same flag
        ``energy_from_u_optimized`` already takes) and change nothing else.

    jacobian_from_hessian:
        ``jacobian_from_hessian(solution, problem, theta_model, options, H) -> J``
        Completes the sensitivity given a supplied Hessian.  This already exists
        in substance: it is ``implicit_tip_jacobian(..., H_override=H)``.

    With those two, the arm computes, per state:

        H_nc  = build_hessian(..., use_lumen=False)     # offline-eligible
        H_tot = build_hessian(..., use_lumen=True)      # reference only
        H_c   = H_tot - H_nc                            # the online term
        J     = jacobian_from_hessian(..., H_nc_ref + H_c)

    and times only the online portion, which is the quantity the paper claims.
    """

    build_hessian: Callable[..., Array] | None = None
    jacobian_from_hessian: Callable[..., Array] | None = None

    def available(self) -> bool:
        return callable(self.build_hessian) and callable(
            self.jacobian_from_hessian
        )

    def require(self) -> None:
        if self.available():
            return
        raise NotImplementedError(
            "The 'decomposed' arm needs two callables from "
            "sensitivity_optimized.py that are not wired yet:\n"
            "  build_hessian(solution, problem, theta_model, options, use_lumen)\n"
            "  jacobian_from_hessian(solution, problem, theta_model, options, H)\n"
            "See DecomposedHessianHooks.__doc__ for the exact contract. Until "
            "they are supplied, run with arms=('full', 'reuse'): that already "
            "measures the Hessian's share of Jacobian cost, which is the number "
            "that decides whether the decomposition is worth building."
        )


# ---------------------------------------------------------------------------
# One Jacobian evaluation.
# ---------------------------------------------------------------------------


def evaluate_analytical_jacobian(
    *,
    adapter: Any,
    state: Array,
) -> tuple[Array, dict[str, Any], float]:
    """Call the project's analytical Jacobian and time it.

    No finite differencing is performed here.  ``continuous_output_jacobian``
    chains the robot geometric Jacobian with the implicit beam sensitivity and
    returns the continuous 6x7 derivative of ``[tip_xyz, tangent_xyz]`` with
    respect to ``[q1..q6, insertion]``.
    """
    state = _finite_vector(state, 7, "benchmark state")
    started = time.perf_counter()
    jacobian = np.asarray(
        adapter.continuous_output_jacobian(state), dtype=float
    )
    elapsed = time.perf_counter() - started

    if jacobian.shape[1] != 7 or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError(
            "The analytical Jacobian returned an invalid matrix with shape "
            f"{jacobian.shape}."
        )
    return jacobian, _collect_diagnostics(adapter), float(elapsed)


def _sample_from_call(
    *,
    arm: str,
    state_index: int,
    repeat_index: int,
    call_wall_s: float,
    diagnostics: dict[str, Any],
    model: Any,
) -> JacobianCostSample:
    active, total, gap_min = _contact_state_from_model(model)
    return JacobianCostSample(
        arm=str(arm),
        state_index=int(state_index),
        repeat_index=int(repeat_index),
        call_wall_s=float(call_wall_s),
        jacobian_wall_s=_diagnostic_float(diagnostics, "jacobian_wall_s"),
        hessian_time_s=_diagnostic_float(diagnostics, "hessian_time_s"),
        gtheta_time_s=_diagnostic_float(diagnostics, "gtheta_time_s"),
        linear_solve_time_s=_diagnostic_float(
            diagnostics, "linear_solve_time_s"
        ),
        kinematics_sensitivity_time_s=_diagnostic_float(
            diagnostics, "kinematics_sensitivity_time_s"
        ),
        sensitivity_time_s=_diagnostic_float(diagnostics, "sensitivity_time_s"),
        nominal_gradient_time_s=_diagnostic_float(
            diagnostics, "nominal_gradient_time_s"
        ),
        gradient_evaluations=_diagnostic_float(
            diagnostics, "gradient_evaluations"
        ),
        hessian_reused=bool(diagnostics.get("hessian_reused", False)),
        cond_H_beam=_diagnostic_float(diagnostics, "cond_H_beam", "H_condition"),
        stationarity_norm=_diagnostic_float(diagnostics, "stationarity_norm"),
        contact_active_nodes=int(active),
        contact_total_nodes=int(total),
        gap_min_m=float(gap_min),
    )


# ---------------------------------------------------------------------------
# Arm execution.
# ---------------------------------------------------------------------------


def run_arm(
    *,
    arm: str,
    adapter: Any,
    states: Array,
    config: BenchmarkConfig,
    hooks: DecomposedHessianHooks | None = None,
) -> tuple[list[JacobianCostSample], dict[int, Array]]:
    """Walk the state list once for one arm."""
    if arm not in KNOWN_ARMS:
        raise ValueError(f"Unknown arm {arm!r}.")
    if arm == "decomposed":
        (hooks or DecomposedHessianHooks()).require()

    model = _resolve_model(adapter)
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(
            f"states must have shape (N, 7); got {states.shape}."
        )

    # Start each arm from a clean cache so the first samples are comparable.
    clear_hessian_reuse(model)
    clear_jacobian_value_cache(model)

    samples: list[JacobianCostSample] = []
    jacobians: dict[int, Array] = {}

    for state_index, state in enumerate(states):
        for repeat_index in range(int(config.repeats_per_state)):
            # The Jacobian value cache must always be cleared, otherwise a
            # repeat at the same pose returns the stored matrix and measures
            # nothing.  The Hessian reuse cache is cleared only for the "full"
            # arm, which is what makes it the non-decomposed reference.
            clear_jacobian_value_cache(model)
            if arm == "full":
                clear_hessian_reuse(model)

            jacobian, diagnostics, call_wall_s = evaluate_analytical_jacobian(
                adapter=adapter,
                state=state,
            )
            sample = _sample_from_call(
                arm=arm,
                state_index=state_index,
                repeat_index=repeat_index,
                call_wall_s=call_wall_s,
                diagnostics=diagnostics,
                model=model,
            )

            if state_index >= int(config.warmup_states):
                samples.append(sample)
                jacobians.setdefault(state_index, jacobian.copy())

        if config.debug and (
            state_index == 0
            or (state_index + 1) % 10 == 0
            or state_index + 1 == states.shape[0]
        ):
            print(
                f"[JACOBIAN COST] arm={arm} "
                f"state={state_index + 1}/{states.shape[0]} "
                f"wall={1.0e3 * sample.call_wall_s:.3f} ms "
                f"hessian={1.0e3 * sample.hessian_time_s:.3f} ms "
                f"reused={sample.hessian_reused} "
                f"contact={sample.contact_active_nodes}",
                flush=True,
            )

    return samples, jacobians


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def _arm_summary(samples: Sequence[JacobianCostSample]) -> dict[str, Any]:
    if not samples:
        return {}
    return {
        "sample_count": int(len(samples)),
        "median_call_wall_ms": 1.0e3 * _median(s.call_wall_s for s in samples),
        "p95_call_wall_ms": float(
            1.0e3
            * np.nanpercentile(
                np.asarray([s.call_wall_s for s in samples], dtype=float), 95.0
            )
        ),
        "median_jacobian_wall_ms": 1.0e3
        * _median(s.jacobian_wall_s for s in samples),
        "median_hessian_ms": 1.0e3 * _median(s.hessian_time_s for s in samples),
        "median_gtheta_ms": 1.0e3 * _median(s.gtheta_time_s for s in samples),
        "median_linear_solve_ms": 1.0e3
        * _median(s.linear_solve_time_s for s in samples),
        "median_kinematics_ms": 1.0e3
        * _median(s.kinematics_sensitivity_time_s for s in samples),
        "median_gradient_evaluations": _median(
            s.gradient_evaluations for s in samples
        ),
        "hessian_fraction_of_jacobian": _median(
            s.hessian_fraction for s in samples
        ),
        "hessian_reuse_rate": float(
            np.mean([1.0 if s.hessian_reused else 0.0 for s in samples])
        ),
        "median_cond_H_beam": _median(s.cond_H_beam for s in samples),
        "median_relative_jacobian_error": _median(
            s.relative_jacobian_error for s in samples
        ),
        "maximum_relative_jacobian_error": float(
            np.nanmax(
                np.asarray(
                    [s.relative_jacobian_error for s in samples], dtype=float
                )
            )
        )
        if any(np.isfinite(s.relative_jacobian_error) for s in samples)
        else float("nan"),
    }


def _contact_correlation(samples: Sequence[JacobianCostSample]) -> dict[str, Any]:
    """Relate cost and reuse failure to contact-set size.

    This is the empirical basis for a contact-triggered update policy: if
    neither correlation is present, the policy has no mechanism behind it.
    """
    contact = np.asarray(
        [float(s.contact_active_nodes) for s in samples], dtype=float
    )
    wall = np.asarray([float(s.call_wall_s) for s in samples], dtype=float)
    reused = np.asarray(
        [1.0 if s.hessian_reused else 0.0 for s in samples], dtype=float
    )
    valid = np.isfinite(contact) & np.isfinite(wall) & (contact >= 0.0)
    if int(np.sum(valid)) < 3 or float(np.std(contact[valid])) <= 0.0:
        return {
            "contact_cost_correlation": None,
            "contact_reuse_correlation": None,
            "note": (
                "Contact-set size did not vary across the sampled states, or "
                "contact diagnostics were unavailable. Re-run over states that "
                "span free and engaged contact to make this meaningful."
            ),
        }
    result: dict[str, Any] = {
        "contact_cost_correlation": float(
            np.corrcoef(contact[valid], wall[valid])[0, 1]
        )
    }
    if float(np.std(reused[valid])) > 0.0:
        result["contact_reuse_correlation"] = float(
            np.corrcoef(contact[valid], reused[valid])[0, 1]
        )
    else:
        result["contact_reuse_correlation"] = None
    return result


def summarize(result: BenchmarkResult) -> dict[str, Any]:
    by_arm: dict[str, list[JacobianCostSample]] = {}
    for sample in result.samples:
        by_arm.setdefault(sample.arm, []).append(sample)

    summary: dict[str, Any] = {
        "arms": {arm: _arm_summary(samples) for arm, samples in by_arm.items()},
        "contact": _contact_correlation(result.samples),
        "jacobian_source": (
            "plant_diagnostic_joint_adapter.continuous_output_jacobian "
            "(analytical implicit sensitivity; no finite-difference output "
            "Jacobian is used anywhere in this benchmark)"
        ),
    }

    full = summary["arms"].get("full", {})
    reuse = summary["arms"].get("reuse", {})
    if full and reuse:
        full_wall = float(full.get("median_call_wall_ms", np.nan))
        reuse_wall = float(reuse.get("median_call_wall_ms", np.nan))
        if np.isfinite(full_wall) and np.isfinite(reuse_wall) and reuse_wall > 0:
            summary["reuse_speedup"] = float(full_wall / reuse_wall)
        summary["reuse_relative_jacobian_error"] = reuse.get(
            "median_relative_jacobian_error", float("nan")
        )

    if full:
        fraction = float(full.get("hessian_fraction_of_jacobian", np.nan))
        summary["hessian_fraction_of_jacobian"] = fraction
        if np.isfinite(fraction):
            if fraction >= 0.6:
                verdict = (
                    "The Hessian build dominates. A term-wise decomposition is "
                    "worth building and is the natural Contribution 2."
                )
            elif fraction >= 0.3:
                verdict = (
                    "The Hessian build is significant but not dominant. Check "
                    "the Gtheta and linear-solve columns before committing; the "
                    "decomposition may buy less than expected."
                )
            else:
                verdict = (
                    "The Hessian build is NOT the bottleneck. Do not build the "
                    "decomposition on cost grounds. Find the real cost centre "
                    "in the per-term columns and simplify the paper's second "
                    "contribution accordingly."
                )
            summary["verdict"] = verdict
    return summary


def _attach_relative_errors(
    result: BenchmarkResult, *, reference_arm: str
) -> None:
    """Score every arm's Jacobian against the reference arm, state by state."""
    reference = {
        state_index: jacobian
        for (arm, state_index), jacobian in result.jacobians.items()
        if arm == reference_arm
    }
    if not reference:
        return
    for sample in result.samples:
        target = reference.get(sample.state_index)
        if target is None:
            continue
        candidate = result.jacobians.get((sample.arm, sample.state_index))
        if candidate is None or candidate.shape != target.shape:
            continue
        denominator = float(np.linalg.norm(target))
        if denominator <= 0.0:
            continue
        sample.relative_jacobian_error = float(
            np.linalg.norm(candidate - target) / denominator
        )


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    adapter: Any,
    states: Array,
    config: BenchmarkConfig | None = None,
    hooks: DecomposedHessianHooks | None = None,
) -> BenchmarkResult:
    """Run every requested arm over the same state sequence."""
    config = config or BenchmarkConfig()
    config.validate()

    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"states must have shape (N, 7); got {states.shape}.")
    states = states[:: int(config.state_stride)]
    if config.maximum_states is not None:
        states = states[: int(config.maximum_states)]
    if states.shape[0] <= int(config.warmup_states):
        raise ValueError(
            f"Only {states.shape[0]} states supplied but warmup_states="
            f"{config.warmup_states}; nothing would be recorded."
        )

    all_samples: list[JacobianCostSample] = []
    all_jacobians: dict[tuple[str, int], Array] = {}

    for arm in config.arms:
        samples, jacobians = run_arm(
            arm=arm,
            adapter=adapter,
            states=states,
            config=config,
            hooks=hooks,
        )
        all_samples.extend(samples)
        for state_index, jacobian in jacobians.items():
            all_jacobians[(arm, state_index)] = jacobian

    result = BenchmarkResult(
        samples=all_samples,
        jacobians=all_jacobians,
        configuration=_json_ready(config),
    )
    _attach_relative_errors(result, reference_arm=config.compare_against_arm)
    result.summary = summarize(result)
    result.summary["state_count"] = int(states.shape[0])
    return result


def save_benchmark(result: BenchmarkResult, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "jacobian_cost_samples.csv"
    json_path = root / "jacobian_cost_summary.json"

    rows = [asdict(sample) for sample in result.samples]
    for row, sample in zip(rows, result.samples):
        row["hessian_fraction"] = sample.hessian_fraction
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "configuration": result.configuration,
                    "summary": result.summary,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    files = {"csv": str(csv_path), "json": str(json_path)}
    result.output_files.update(files)
    return files


# ---------------------------------------------------------------------------
# State sources.
#
# These are deliberately forgiving: the benchmark's job is to produce the cost
# number, and it must not be blocked because a particular planning artefact has
# not been generated yet.  A synthetic sweep around p0 is always available.
# ---------------------------------------------------------------------------


_STATE_NPZ_KEYS = (
    "state_reference",
    "state",
    "states",
    "configuration_states",
)

_CSV_JOINT_KEYS = ("q{}_rad", "q{}", "joint{}_rad")
_CSV_INSERTION_KEYS = ("insertion_m", "insertion", "L_m", "L")


def states_from_npz(path: str | Path) -> Array:
    """Extract an (N, 7) state array from a saved NPZ."""
    path = Path(path)
    with np.load(path) as archive:
        available = list(archive.keys())
        for key in _STATE_NPZ_KEYS:
            if key not in archive:
                continue
            states = np.asarray(archive[key], dtype=float)
            if states.ndim == 2 and states.shape[1] == 7:
                return states
    raise KeyError(
        f"{path} has no (N, 7) state array. Looked for {_STATE_NPZ_KEYS}; "
        f"the archive contains {available}."
    )


def states_from_csv(path: str | Path) -> Array:
    """Extract an (N, 7) state array from a saved CSV."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no rows.")
    fields = set(rows[0].keys())

    joint_pattern = None
    for pattern in _CSV_JOINT_KEYS:
        if all(pattern.format(joint) in fields for joint in range(1, 7)):
            joint_pattern = pattern
            break
    if joint_pattern is None:
        raise KeyError(
            f"{path} has no six joint columns. Tried patterns "
            f"{_CSV_JOINT_KEYS}; columns present: {sorted(fields)}."
        )

    insertion_key = next(
        (key for key in _CSV_INSERTION_KEYS if key in fields), None
    )
    if insertion_key is None:
        raise KeyError(
            f"{path} has no insertion column. Tried {_CSV_INSERTION_KEYS}; "
            f"columns present: {sorted(fields)}."
        )

    return np.asarray(
        [
            [float(row[joint_pattern.format(joint)]) for joint in range(1, 7)]
            + [float(row[insertion_key])]
            for row in rows
        ],
        dtype=float,
    )


def states_from_directory(directory: str | Path) -> Array:
    """Scan a saved planning directory for anything holding (N, 7) states.

    Rather than assuming a filename, this tries every NPZ and then every CSV in
    the directory, so it works with the inverse path, the global path and the
    time-parameterised path without being told which one it is looking at.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    errors: list[str] = []
    for pattern, loader in ((("*.npz"), states_from_npz), (("*.csv"), states_from_csv)):
        for candidate in sorted(directory.glob(pattern)):
            try:
                states = loader(candidate)
            except Exception as exc:
                errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
                continue
            if states.shape[0] >= 2:
                print(
                    f"[JACOBIAN COST] states loaded from {candidate}"
                    f" ({states.shape[0]} rows)",
                    flush=True,
                )
                return states

    listing = sorted(item.name for item in directory.iterdir())
    detail = ("\n  " + "\n  ".join(errors)) if errors else " (no candidate files)"
    raise FileNotFoundError(
        f"No (N, 7) state array found in {directory}. Contents: {listing}."
        f" Attempts:{detail}"
    )


def states_from_controller_pack(
    controller_pack: dict[str, Any],
    *,
    count: int = 24,
    insertion_span_m: float = 0.02,
    joint_span_rad: float = 0.15,
) -> Array:
    """Build a synthetic but physically admissible state sweep around p0.

    This exists so the benchmark can always run.  It sweeps insertion, because
    insertion is what drives contact engagement, and adds a small correlated
    joint motion so the magnet pose changes too.  Every state is clipped into
    ``[p_min, p_max]``.

    It is a *cost* benchmark, not a control experiment: the states only need to
    be admissible and to span a range of contact conditions.  For the figures
    that go in the paper, prefer real planned states.
    """
    for key in ("p0", "p_min", "p_max"):
        if key not in controller_pack:
            raise KeyError(f"controller_pack is missing {key!r}.")

    p0 = _finite_vector(controller_pack["p0"], 7, "p0")
    p_min = _finite_vector(controller_pack["p_min"], 7, "p_min")
    p_max = _finite_vector(controller_pack["p_max"], 7, "p_max")
    count = int(count)
    if count < 2:
        raise ValueError("count must be at least two.")

    fraction = np.linspace(0.0, 1.0, count)
    states = np.tile(p0, (count, 1))
    states[:, 6] = p0[6] + float(insertion_span_m) * fraction
    states[:, 0] = p0[0] + float(joint_span_rad) * fraction
    states[:, 4] = p0[4] - 0.5 * float(joint_span_rad) * fraction

    states = np.clip(states, p_min, p_max)
    if np.any(states[:, 6] <= 0.0):
        raise ValueError(
            "Synthetic insertion sweep produced a non-positive insertion "
            "length; the sensitivity code requires L > 0."
        )
    return states


def _load_states(
    source: str,
    out_root: Path,
    *,
    controller_pack: dict[str, Any] | None = None,
    synthetic_count: int = 24,
) -> Array:
    """Resolve ``--states`` into an (N, 7) array."""
    if source in {"synthetic", "sweep", "p0"}:
        if controller_pack is None:
            raise ValueError(
                "Synthetic states require the controller pack."
            )
        return states_from_controller_pack(
            controller_pack, count=synthetic_count
        )

    candidate = Path(source)
    search: list[Path] = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.extend((out_root / source, candidate))

    errors: list[str] = []
    for path in search:
        if path.is_file():
            loader = states_from_npz if path.suffix == ".npz" else None
            if path.suffix == ".npy":
                states = np.asarray(np.load(path), dtype=float)
                if states.ndim == 2 and states.shape[1] == 7:
                    return states
                errors.append(f"{path}: shape {states.shape} is not (N, 7)")
                continue
            if loader is None:
                loader = states_from_csv
            try:
                return loader(path)
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
        if path.is_dir():
            try:
                return states_from_directory(path)
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
        errors.append(f"{path}: does not exist")

    if controller_pack is not None:
        print(
            "[JACOBIAN COST] Could not load saved states; falling back to a "
            "synthetic sweep around p0. Attempts:\n  "
            + "\n  ".join(errors),
            flush=True,
        )
        return states_from_controller_pack(
            controller_pack, count=synthetic_count
        )

    raise FileNotFoundError(
        "Could not load benchmark states. Attempts:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the cost breakdown of the project's analytical beam "
            "Jacobian, and compare full Hessian rebuild against Hessian reuse."
        )
    )
    parser.add_argument(
        "--states",
        type=str,
        default="time_parameterized_configuration_path",
        help=(
            "Where to get benchmark states. Either 'synthetic' for an "
            "admissible sweep around p0 (always available), or a directory "
            "under the planning output root, or a direct path to a .npz/.csv/"
            ".npy holding (N, 7) states. Directories are scanned rather than "
            "assumed, and an unreadable source falls back to synthetic."
        ),
    )
    parser.add_argument("--synthetic-count", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-states", type=int, default=40)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-states", type=int, default=2)
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=("full", "reuse"),
        choices=sorted(KNOWN_ARMS),
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_states < 1:
        parser.error("--max-states must be positive")
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.state_stride < 1:
        parser.error("--state-stride must be positive")
    if arguments.warmup_states < 0:
        parser.error("--warmup-states cannot be negative")
    if arguments.synthetic_count < 2:
        parser.error("--synthetic-count must be at least two")
    return arguments


def main() -> None:
    arguments = _arguments()

    from proper_research.planning.planning_context import build_planning_context

    _, _, controller_pack, out_root = build_planning_context()

    adapter = controller_pack.get("plant_diagnostic_joint_adapter")
    if adapter is None:
        raise KeyError(
            "controller_pack has no 'plant_diagnostic_joint_adapter'. That "
            "adapter is the analytical Jacobian interface this benchmark "
            "measures; build the controller with "
            "controller_factory_joint_space.build_controller."
        )
    if int(getattr(adapter, "n_out", 6)) != 6:
        raise ValueError(
            "The benchmark expects the full six-output diagnostic adapter, "
            f"received n_out={getattr(adapter, 'n_out', None)}."
        )

    states = _load_states(
        arguments.states,
        out_root,
        controller_pack=controller_pack,
        synthetic_count=int(arguments.synthetic_count),
    )
    config = BenchmarkConfig(
        arms=tuple(arguments.arms),
        repeats_per_state=int(arguments.repeats),
        warmup_states=int(arguments.warmup_states),
        maximum_states=int(arguments.max_states),
        state_stride=int(arguments.state_stride),
        debug=not arguments.quiet,
    )

    print(
        "[JACOBIAN COST] "
        f"states={states.shape[0]} arms={config.arms} "
        f"repeats={config.repeats_per_state}",
        flush=True,
    )

    started = time.perf_counter()
    result = run_benchmark(adapter=adapter, states=states, config=config)
    elapsed = time.perf_counter() - started

    output_directory = arguments.output_dir or (out_root / "jacobian_cost_benchmark")
    files = save_benchmark(result, output_directory)

    summary = result.summary
    print(f"\nElapsed wall time: {elapsed:.3f} s")
    for arm, arm_summary in summary.get("arms", {}).items():
        print(f"\n[{arm}]")
        print(
            "  median call wall:      "
            f"{arm_summary.get('median_call_wall_ms', float('nan')):.3f} ms"
        )
        print(
            "  median Hessian build:  "
            f"{arm_summary.get('median_hessian_ms', float('nan')):.3f} ms"
        )
        print(
            "  median Gtheta:         "
            f"{arm_summary.get('median_gtheta_ms', float('nan')):.3f} ms"
        )
        print(
            "  median linear solve:   "
            f"{arm_summary.get('median_linear_solve_ms', float('nan')):.3f} ms"
        )
        print(
            "  Hessian reuse rate:    "
            f"{arm_summary.get('hessian_reuse_rate', float('nan')):.3f}"
        )
        print(
            "  median rel. J error:   "
            f"{arm_summary.get('median_relative_jacobian_error', float('nan')):.3e}"
        )

    if "hessian_fraction_of_jacobian" in summary:
        print(
            "\nHessian share of Jacobian time: "
            f"{summary['hessian_fraction_of_jacobian']:.3f}"
        )
    if "reuse_speedup" in summary:
        print(f"Reuse speedup: {summary['reuse_speedup']:.2f}x")
    if "verdict" in summary:
        print(f"\nVerdict: {summary['verdict']}")
    print(f"\nSaved: {files['csv']}\n       {files['json']}")


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "DecomposedHessianHooks",
    "JacobianCostSample",
    "clear_hessian_reuse",
    "clear_jacobian_value_cache",
    "evaluate_analytical_jacobian",
    "run_arm",
    "run_benchmark",
    "save_benchmark",
    "states_from_controller_pack",
    "states_from_csv",
    "states_from_directory",
    "states_from_npz",
    "summarize",
]
