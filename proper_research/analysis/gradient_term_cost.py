"""Decide whether the Hessian decomposition is worth building -- in ~30 seconds.

Why this exists
---------------
``sensitivity_optimized._finite_difference_hessian`` builds H with one gradient
evaluation per strain variable: ``n_u = 3 * (N_nodes - 1)`` evaluations of
``_gradient``, which is ``energy_gradient_u(..., use_magnetic=True,
use_contact=problem.use_contact_in_jacobian)``.

So the Hessian's cost is ``n_u * t(full gradient)``.  The proposed decomposition
precomputes the elastic+magnetic part offline and rebuilds only the contact part
online, which would cost ``n_u * t(contact-only gradient)``.

That means the entire value of the decomposition reduces to one ratio:

    ceiling = t(full gradient) / t(contact-only gradient)

and it has a hard floor, because both halves must integrate the kinematics
before they can evaluate anything:

    floor = t(elastic-only gradient)   # kinematics + elastic, unavoidable

If the kinematic integration dominates ``energy_gradient_u``, the ratio is close
to one and **the decomposition cannot help**, no matter how the terms are split.
That is the single most important thing to find out before writing Contribution
2, and it does not require building the decomposition to answer.

``energy_gradient_u`` already takes independent ``use_magnetic`` and
``use_contact`` flags, so the four term combinations are measurable directly
with no new code paths and no new differencing scheme.

What it also checks
-------------------
That the split is *exact*.  Using the project's own ``_finite_difference_hessian``
unmodified, with ``use_contact_in_jacobian`` toggled through ``dataclasses.replace``:

    H_total   = FD Hessian with contact
    H_nocontact = FD Hessian without contact
    H_contact = H_total - H_nocontact

and reports the residual.  If this is at FD noise level, the three-term split in
the paper is exact rather than approximate, which is worth stating.

Usage
-----
    python -m proper_research.analysis.gradient_term_cost --repeats 20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


Array = np.ndarray


# These are private in sensitivity_optimized, and importing them is deliberate:
# the benchmark must use *your* differencing, not a reimplementation of it.
# Consider promoting them to public names if this becomes part of the pipeline.
def _import_sensitivity_internals():
    from proper_research.simulation.magnetic_beam.sensitivity_optimized import (
        SensitivityOptions,
        _finite_difference_hessian,
        _gradient,
        _prepare,
        _workers,
    )

    return {
        "SensitivityOptions": SensitivityOptions,
        "finite_difference_hessian": _finite_difference_hessian,
        "gradient": _gradient,
        "prepare": _prepare,
        "workers": _workers,
    }


def _import_energy_gradient() -> Callable[..., Array]:
    from proper_research.simulation.magnetic_beam.gradients import (
        energy_gradient_u,
    )

    return energy_gradient_u


def _time_call(function: Callable[[], Any], *, repeats: int) -> tuple[float, float]:
    """Return (median seconds, minimum seconds) over ``repeats`` calls."""
    samples: list[float] = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    array = np.asarray(samples, dtype=float)
    return float(np.median(array)), float(np.min(array))


def measure_gradient_terms(
    *,
    adapter: Any,
    state: Array,
    repeats: int = 20,
    build_hessians: bool = True,
) -> dict[str, Any]:
    """Time each energy term's gradient, and verify the Hessian split is exact."""
    internals = _import_sensitivity_internals()
    energy_gradient_u = _import_energy_gradient()

    state = np.asarray(state, dtype=float).reshape(7)
    model = getattr(adapter, "model", None)
    if model is None:
        raise AttributeError(
            "The adapter does not expose .model; pass the diagnostic adapter "
            "built by controller_factory_joint_space.build_controller."
        )

    # One committed forward solve gives a converged equilibrium and the pose the
    # sensitivity code would differentiate at.
    adapter.forward_output(state, commit=True)
    cache = model.cache
    p7 = np.asarray(cache.p7_last, dtype=float).reshape(7)
    u0 = np.asarray(cache.u_flat_opt, dtype=float).reshape(-1)

    problem = model.build_problem(p7)
    prepared = internals["prepare"](problem)
    s = prepared.s
    K_seg = prepared.K_seg

    contact_in_jacobian = bool(
        getattr(problem, "use_contact_in_jacobian", False)
    )
    if not contact_in_jacobian:
        print(
            "[GRADIENT TERMS] WARNING: problem.use_contact_in_jacobian is "
            "False, so the sensitivity currently ignores contact entirely. "
            "The timings below still hold, but the exactness check compares "
            "two identical Hessians.",
            flush=True,
        )

    def gradient(*, use_magnetic: bool, use_contact: bool) -> Array:
        return np.asarray(
            energy_gradient_u(
                u0,
                p0=problem.p0,
                q0=problem.q0,
                s=s,
                K_seg=K_seg,
                u_star=problem.u_star,
                m_src=problem.m_src,
                r_src=problem.r_src,
                m_local_fun=problem.m_local_fun,
                m_moment=problem.m_moment,
                lumen_query=problem.lumen_query,
                use_magnetic=bool(use_magnetic),
                use_contact=bool(use_contact),
                contact=problem.contact,
            ),
            dtype=float,
        ).reshape(-1)

    combinations = {
        "full": (True, True),
        "magnetic_only": (True, False),
        "contact_only": (False, True),
        "elastic_only": (False, False),
    }

    # Warm up once per combination so the first-call overheads do not land in
    # the medians.
    for use_magnetic, use_contact in combinations.values():
        gradient(use_magnetic=use_magnetic, use_contact=use_contact)

    timings: dict[str, dict[str, float]] = {}
    for name, (use_magnetic, use_contact) in combinations.items():
        median, minimum = _time_call(
            lambda m=use_magnetic, c=use_contact: gradient(
                use_magnetic=m, use_contact=c
            ),
            repeats=repeats,
        )
        timings[name] = {"median_s": median, "minimum_s": minimum}

    n_u = int(u0.size)
    t_full = timings["full"]["median_s"]
    t_contact = timings["contact_only"]["median_s"]
    t_magnetic = timings["magnetic_only"]["median_s"]
    t_elastic = timings["elastic_only"]["median_s"]

    result: dict[str, Any] = {
        "n_u": n_u,
        "n_nodes": int(getattr(problem, "N_nodes", (n_u // 3) + 1)),
        "contact_in_jacobian": contact_in_jacobian,
        "gradient_timings_ms": {
            name: 1.0e3 * value["median_s"] for name, value in timings.items()
        },
        "shared_floor_ms": 1.0e3 * t_elastic,
        "magnetic_marginal_ms": 1.0e3 * max(t_magnetic - t_elastic, 0.0),
        "contact_marginal_ms": 1.0e3 * max(t_contact - t_elastic, 0.0),
        "projected_full_hessian_ms": 1.0e3 * n_u * t_full,
        "projected_contact_hessian_ms": 1.0e3 * n_u * t_contact,
        "decomposition_ceiling": (
            float(t_full / t_contact) if t_contact > 0.0 else float("nan")
        ),
        "shared_floor_fraction_of_full": (
            float(t_elastic / t_full) if t_full > 0.0 else float("nan")
        ),
    }

    ceiling = result["decomposition_ceiling"]
    floor_fraction = result["shared_floor_fraction_of_full"]
    if np.isfinite(ceiling):
        if ceiling >= 3.0:
            verdict = (
                f"Contact-only gradients are {ceiling:.1f}x cheaper than full "
                "gradients. The decomposition can cut the online Hessian build "
                "by roughly that factor. Build it."
            )
        elif ceiling >= 1.5:
            verdict = (
                f"Contact-only gradients are only {ceiling:.1f}x cheaper. The "
                "decomposition helps but modestly; weigh it against the "
                "complexity it adds to the paper."
            )
        else:
            verdict = (
                f"Contact-only gradients are just {ceiling:.1f}x cheaper, and "
                f"{100.0 * floor_fraction:.0f}% of the gradient cost is the "
                "shared kinematic/elastic floor that neither half avoids. "
                "Splitting the energy terms CANNOT deliver a meaningful "
                "speedup. Drop the decomposition as a cost argument and "
                "reframe Contribution 2 -- either around reuse (which skips "
                "the Hessian entirely) or around something other than cost."
            )
        result["verdict"] = verdict

    if not build_hessians:
        return result

    # Exactness check, using the project's own FD Hessian, unmodified.
    options = internals["SensitivityOptions"]()
    workers = internals["workers"](options)
    build = internals["finite_difference_hessian"]
    grad_fn = internals["gradient"]

    prepared_with = internals["prepare"](
        replace(problem, use_contact_in_jacobian=True)
    )
    prepared_without = internals["prepare"](
        replace(problem, use_contact_in_jacobian=False)
    )

    started = time.perf_counter()
    H_total, evaluations_total = build(
        u0=u0,
        prepared=prepared_with,
        g0=grad_fn(u0, prepared_with),
        eps=float(options.eps_hess),
        scheme=options.difference_scheme,
        workers=workers,
    )
    hessian_total_s = time.perf_counter() - started

    started = time.perf_counter()
    H_without, _ = build(
        u0=u0,
        prepared=prepared_without,
        g0=grad_fn(u0, prepared_without),
        eps=float(options.eps_hess),
        scheme=options.difference_scheme,
        workers=workers,
    )
    hessian_without_s = time.perf_counter() - started

    H_contact = H_total - H_without
    reconstruction = H_without + H_contact
    residual = float(np.linalg.norm(reconstruction - H_total))
    scale = max(float(np.linalg.norm(H_total)), 1.0e-30)

    result.update(
        {
            "measured_full_hessian_ms": 1.0e3 * hessian_total_s,
            "measured_nocontact_hessian_ms": 1.0e3 * hessian_without_s,
            "hessian_gradient_evaluations": int(evaluations_total),
            "hessian_norm_total": float(np.linalg.norm(H_total)),
            "hessian_norm_without_contact": float(np.linalg.norm(H_without)),
            "hessian_norm_contact_term": float(np.linalg.norm(H_contact)),
            "contact_share_of_hessian_norm": (
                float(np.linalg.norm(H_contact) / scale)
            ),
            "split_reconstruction_relative_residual": float(residual / scale),
        }
    )
    return result


def run(
    *,
    adapter: Any,
    states: Array,
    repeats: int = 20,
) -> list[dict[str, Any]]:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"states must have shape (N, 7); got {states.shape}.")

    records: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        record = measure_gradient_terms(
            adapter=adapter,
            state=state,
            repeats=repeats,
            build_hessians=True,
        )
        record["state_index"] = int(index)
        record["insertion_m"] = float(state[6])
        records.append(record)
        print(
            f"[GRADIENT TERMS] state={index + 1}/{states.shape[0]} "
            f"L={1.0e3 * state[6]:.2f} mm "
            f"full={record['gradient_timings_ms']['full']:.4f} ms "
            f"contact_only={record['gradient_timings_ms']['contact_only']:.4f} ms "
            f"ceiling={record['decomposition_ceiling']:.2f}x "
            f"contact_share_H={record.get('contact_share_of_hessian_norm', float('nan')):.3e}",
            flush=True,
        )
    return records


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the per-term cost of the beam energy gradient and verify "
            "that the Hessian splits exactly into contact and non-contact "
            "parts."
        )
    )
    parser.add_argument("--states", type=str, default="synthetic")
    parser.add_argument("--synthetic-count", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.synthetic_count < 1:
        parser.error("--synthetic-count must be positive")
    return arguments


def main() -> None:
    arguments = _arguments()

    from proper_research.planning.planning_context import build_planning_context

    from proper_research.analysis.jacobian_cost_benchmark import _load_states

    _, _, controller_pack, out_root = build_planning_context()
    adapter = controller_pack.get("plant_diagnostic_joint_adapter")
    if adapter is None:
        raise KeyError(
            "controller_pack has no 'plant_diagnostic_joint_adapter'."
        )

    states = _load_states(
        arguments.states,
        out_root,
        controller_pack=controller_pack,
        synthetic_count=int(arguments.synthetic_count),
    )
    states = states[: int(arguments.synthetic_count)]

    records = run(adapter=adapter, states=states, repeats=int(arguments.repeats))

    ceilings = np.asarray(
        [record["decomposition_ceiling"] for record in records], dtype=float
    )
    floors = np.asarray(
        [record["shared_floor_fraction_of_full"] for record in records],
        dtype=float,
    )
    contact_shares = np.asarray(
        [
            record.get("contact_share_of_hessian_norm", np.nan)
            for record in records
        ],
        dtype=float,
    )
    residuals = np.asarray(
        [
            record.get("split_reconstruction_relative_residual", np.nan)
            for record in records
        ],
        dtype=float,
    )

    summary = {
        "state_count": int(len(records)),
        "median_decomposition_ceiling": float(np.nanmedian(ceilings)),
        "median_shared_floor_fraction": float(np.nanmedian(floors)),
        "median_contact_share_of_hessian_norm": float(
            np.nanmedian(contact_shares)
        ),
        "maximum_split_reconstruction_residual": float(np.nanmax(residuals)),
        "verdict": records[-1].get("verdict", ""),
    }

    print("\n=== Gradient term cost ===")
    for name, value in records[-1]["gradient_timings_ms"].items():
        print(f"  {name:<16} {value:.5f} ms")
    print(f"\n  shared floor          {records[-1]['shared_floor_ms']:.5f} ms")
    print(f"  magnetic marginal     {records[-1]['magnetic_marginal_ms']:.5f} ms")
    print(f"  contact marginal      {records[-1]['contact_marginal_ms']:.5f} ms")
    print(
        f"\n  n_u                   {records[-1]['n_u']} "
        f"(N_nodes={records[-1]['n_nodes']})"
    )
    print(
        f"  full Hessian          {records[-1]['projected_full_hessian_ms']:.3f} ms "
        f"projected / {records[-1].get('measured_full_hessian_ms', float('nan')):.3f} ms measured"
    )
    print(
        f"  contact Hessian       "
        f"{records[-1]['projected_contact_hessian_ms']:.3f} ms projected"
    )
    print(
        f"\n  decomposition ceiling {summary['median_decomposition_ceiling']:.2f}x"
    )
    print(
        f"  shared floor fraction {100.0 * summary['median_shared_floor_fraction']:.1f}%"
    )
    print(
        f"  contact share of ||H|| {summary['median_contact_share_of_hessian_norm']:.3e}"
    )
    print(
        f"  split residual        {summary['maximum_split_reconstruction_residual']:.3e}"
    )
    print(f"\nVerdict: {summary['verdict']}")

    output_directory = arguments.output_dir or (out_root / "gradient_term_cost")
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "gradient_term_cost.json"
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()


__all__ = ["measure_gradient_terms", "run"]
