"""Offline self-test for jacobian_cost_benchmark.

Runs the harness against a stub that mimics the real interfaces:

    adapter.continuous_output_jacobian(state) -> (6, 7)
    adapter.get_last_diag()                   -> dict
    adapter.model.last_jacobian_diag          -> dict with the timing keys
    adapter.model.cache.info["parts"]         -> contact weights / gap_min
    adapter.model.last_sensitivity_H          -> the reuse gate

The stub charges a large synthetic cost for a Hessian rebuild and a small one
for a reuse, and makes reuse fail whenever the contact set changes -- which is
the behaviour the real model exhibits via _can_reuse_sensitivity_hessian.  This
verifies the arm logic, the cache clearing, the contact bookkeeping, the
accuracy comparison and the summary, without touching the research package.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from proper_research.analysis.jacobian_cost_benchmark import (
    BenchmarkConfig,
    DecomposedHessianHooks,
    run_benchmark,
    summarize,
)


HESSIAN_BUILD_S = 0.004
GTHETA_S = 0.0008
LINEAR_SOLVE_S = 0.0002


class StubModel:
    def __init__(self) -> None:
        self.last_sensitivity_H = None
        self.last_sensitivity_H_p7 = None
        self.last_sensitivity_H_eps = None
        self.last_sensitivity_contact_mask = None
        self.last_J_output_actuation_tangent = None
        self.last_J_output_pose7 = None
        self.last_J_tip_actuation_tangent = None
        self.last_J_tip_pose7 = None
        self._last_tangent_key = None
        self._last_pose7_key = None
        self.last_jacobian_diag = {}
        self.last_sens_info = {}
        self.cache = SimpleNamespace(info={"parts": {}})
        self.rebuild_count = 0
        self.reuse_count = 0


def _contact_mask(state: np.ndarray) -> np.ndarray:
    """Contact engages progressively with the insertion coordinate."""
    engaged = int(np.clip(round(10.0 * float(state[6]) / 0.05), 0, 6))
    mask = np.zeros(10, dtype=float)
    mask[:engaged] = 1.0
    return mask


class StubAdapter:
    n_out = 6

    def __init__(self) -> None:
        self.model = StubModel()

    def get_last_diag(self):
        return {"joint_space_adapter": 1}

    def continuous_output_jacobian(self, state):
        state = np.asarray(state, dtype=float).reshape(7)
        model = self.model
        mask = _contact_mask(state)

        # Reuse is permitted only when a Hessian is stored AND the contact set
        # is unchanged, mirroring _can_reuse_sensitivity_hessian.
        can_reuse = (
            model.last_sensitivity_H is not None
            and model.last_sensitivity_contact_mask is not None
            and np.array_equal(model.last_sensitivity_contact_mask, mask)
        )

        started = time.perf_counter()
        if can_reuse:
            model.reuse_count += 1
            hessian_time = 0.0
            H = model.last_sensitivity_H
        else:
            model.rebuild_count += 1
            time.sleep(HESSIAN_BUILD_S)
            hessian_time = time.perf_counter() - started
            H = np.eye(30) * (1.0 + float(np.sum(mask)))

        gtheta_started = time.perf_counter()
        time.sleep(GTHETA_S)
        gtheta_time = time.perf_counter() - gtheta_started

        solve_started = time.perf_counter()
        time.sleep(LINEAR_SOLVE_S)
        linear_solve_time = time.perf_counter() - solve_started

        wall = time.perf_counter() - started

        # A reused Hessian is slightly stale, so the returned Jacobian differs.
        base = np.tile(np.arange(7, dtype=float), (6, 1)) + float(state[6])
        jacobian = base * (1.0 + 0.01 * float(np.sum(mask)))
        if can_reuse:
            jacobian = jacobian * 1.002

        model.last_sensitivity_H = H
        model.last_sensitivity_contact_mask = mask
        model.last_J_output_actuation_tangent = jacobian.copy()
        model.cache.info["parts"] = {
            "contact_weights": mask.copy(),
            "gap_min": float(0.002 - 0.0003 * np.sum(mask)),
        }
        model.last_jacobian_diag = {
            "jacobian_wall_s": wall,
            "hessian_time_s": hessian_time,
            "gtheta_time_s": gtheta_time,
            "linear_solve_time_s": linear_solve_time,
            "kinematics_sensitivity_time_s": 5.0e-5,
            "sensitivity_time_s": wall,
            "nominal_gradient_time_s": 2.0e-4,
            "gradient_evaluations": 0.0 if can_reuse else 31.0,
            "hessian_reused": bool(can_reuse),
            "cond_H_beam": 1.0e4 * (1.0 + float(np.sum(mask))),
            "stationarity_norm": 1.0e-9,
        }
        return jacobian


def main() -> None:
    adapter = StubAdapter()
    states = np.zeros((14, 7), dtype=float)
    states[:, 6] = np.linspace(0.0, 0.05, states.shape[0])
    states[:, 0] = np.linspace(0.0, 0.2, states.shape[0])

    config = BenchmarkConfig(
        arms=("full", "reuse"),
        repeats_per_state=2,
        warmup_states=1,
        debug=False,
    )
    result = run_benchmark(adapter=adapter, states=states, config=config)
    summary = result.summary

    assert summary["arms"]["full"]["sample_count"] > 0
    assert summary["arms"]["reuse"]["sample_count"] > 0

    full = summary["arms"]["full"]
    reuse = summary["arms"]["reuse"]

    # The full arm must never reuse; the reuse arm must reuse at least once.
    assert full["hessian_reuse_rate"] == 0.0, full["hessian_reuse_rate"]
    assert reuse["hessian_reuse_rate"] > 0.0, reuse["hessian_reuse_rate"]

    # The synthetic Hessian build dominates, so the fraction must be high and
    # the verdict must recommend building the decomposition.
    assert full["hessian_fraction_of_jacobian"] > 0.6
    assert "worth building" in summary["verdict"]

    # Reuse must be faster and must show a measurable Jacobian error.
    assert summary["reuse_speedup"] > 1.0, summary["reuse_speedup"]
    assert reuse["median_relative_jacobian_error"] > 0.0
    assert full["median_relative_jacobian_error"] == 0.0

    # Contact bookkeeping must have been read from the model cache.
    contact_counts = {s.contact_active_nodes for s in result.samples}
    assert contact_counts != {-1}, "contact diagnostics were not harvested"
    assert max(contact_counts) > min(contact_counts)

    # The decomposed arm must refuse clearly rather than silently doing nothing.
    try:
        run_benchmark(
            adapter=StubAdapter(),
            states=states,
            config=BenchmarkConfig(
                arms=("decomposed",),
                compare_against_arm="decomposed",
                warmup_states=1,
                debug=False,
            ),
            hooks=DecomposedHessianHooks(),
        )
    except NotImplementedError as exc:
        assert "sensitivity_optimized.py" in str(exc)
    else:
        raise AssertionError("The decomposed arm should have refused.")

    print("[SELF TEST] PASS")
    print(f"  full  median call wall : {full['median_call_wall_ms']:.3f} ms")
    print(f"  reuse median call wall : {reuse['median_call_wall_ms']:.3f} ms")
    print(f"  Hessian share          : {full['hessian_fraction_of_jacobian']:.3f}")
    print(f"  reuse speedup          : {summary['reuse_speedup']:.2f}x")
    print(f"  reuse rate             : {reuse['hessian_reuse_rate']:.3f}")
    print(f"  reuse rel. J error     : {reuse['median_relative_jacobian_error']:.3e}")
    print(f"  contact/cost corr.     : {summary['contact']['contact_cost_correlation']}")
    print(f"  rebuilds / reuses      : {adapter.model.rebuild_count} / {adapter.model.reuse_count}")


if __name__ == "__main__":
    main()
