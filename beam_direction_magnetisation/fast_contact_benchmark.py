#!/usr/bin/env python3
"""
Fast contact/lumen penalty kernel plus benchmark harness.

Usage pattern
-------------
1) Save this file next to your forward-model module.
2) In your model module, expose a small case factory, for example:

    def make_benchmark_case():
        # Reuse the setup currently in your __main__ block.
        # Return at least lumen_C, lumen_R, contact, and optionally p/s and solve_kwargs.
        return {
            "lumen_C": lumen_C,
            "lumen_R": lumen_R,
            "contact": contact,
            "p": p_for_contact_test,       # optional, shape (3, N)
            "s": s_nodes,                  # optional, shape (N,)
            "solve_kwargs": dict(...),     # optional kwargs for solve_quasistatic_insertion
        }

3) Run:

    python fast_contact_benchmark.py --module your_model_module --case-fn make_benchmark_case

What it does
------------
- Benchmarks your original contact_barrier_energy_and_force_fast.
- Monkey-patches a vectorized KD-tree/polyline implementation with the same signature.
- Compares contact energy/force/gap errors.
- Optionally runs solve_quasistatic_insertion before and after if solve_kwargs is supplied.

Notes
-----
The fast kernel computes closest points on lumen polyline segments. If your current
LumenQuery.closest uses a different geometric convention, the benchmark will reveal
that via the reported errors. Use --exact-all-segments for maximum accuracy on small
centerlines; use --candidate-k 8..32 for larger centerlines.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class FastContactConfig:
    candidate_k: int = 16
    exact_all_segments: bool = False
    validate_fallback: bool = True


class PolylineBatchLumenQuery:
    """Batch closest-point queries against a radius-labelled 3D polyline."""

    def __init__(self, C: np.ndarray, R: np.ndarray, *, candidate_k: int = 16, exact_all_segments: bool = False):
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float).reshape(-1)
        if C.ndim != 2:
            raise ValueError(f"lumen centerline C must be 2D, got shape {C.shape}")
        if C.shape[0] == 3 and C.shape[1] != 3:
            C = C.T
        if C.shape[1] != 3:
            raise ValueError(f"lumen centerline must have 3 columns, got shape {C.shape}")
        if C.shape[0] < 2:
            raise ValueError("lumen centerline must contain at least two points")
        if R.size != C.shape[0]:
            raise ValueError(f"lumen_R length {R.size} does not match C length {C.shape[0]}")

        self.C = np.ascontiguousarray(C)
        self.R = np.ascontiguousarray(R)
        self.A = np.ascontiguousarray(C[:-1])
        self.B = np.ascontiguousarray(C[1:])
        self.D = np.ascontiguousarray(self.B - self.A)
        self.seg_len2 = np.einsum("ij,ij->i", self.D, self.D)
        self.seg_len2 = np.maximum(self.seg_len2, 1e-30)
        self.R0 = self.R[:-1]
        self.R1 = self.R[1:]
        self.n_seg = self.A.shape[0]
        self.candidate_k = int(max(1, candidate_k))
        self.exact_all_segments = bool(exact_all_segments)

        if not self.exact_all_segments and self.n_seg > self.candidate_k:
            if cKDTree is None:
                raise RuntimeError(f"scipy.spatial.cKDTree unavailable: {_SCIPY_IMPORT_ERROR}")
            mid = 0.5 * (self.A + self.B)
            self.tree = cKDTree(mid)
        else:
            self.tree = None

    def closest_batch(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return delta, Rloc, q_closest for points p with shape (3,N) or (N,3)."""
        X = np.asarray(p, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"p must be 2D, got shape {X.shape}")
        if X.shape[0] == 3 and X.shape[1] != 3:
            X = X.T
        if X.shape[1] != 3:
            raise ValueError(f"p must be shape (3,N) or (N,3), got {X.shape}")
        X = np.ascontiguousarray(X)
        n = X.shape[0]

        if self.tree is None:
            ids = np.broadcast_to(np.arange(self.n_seg), (n, self.n_seg))
        else:
            k = min(self.candidate_k, self.n_seg)
            _, ids = self.tree.query(X, k=k)
            if k == 1:
                ids = ids[:, None]

        A = self.A[ids]                       # (N, K, 3)
        D = self.D[ids]
        L2 = self.seg_len2[ids]               # (N, K)
        XA = X[:, None, :] - A
        t = np.einsum("nki,nki->nk", XA, D) / L2
        t = np.clip(t, 0.0, 1.0)
        Q = A + t[:, :, None] * D
        diff = X[:, None, :] - Q
        dist2 = np.einsum("nki,nki->nk", diff, diff)
        best_local = np.argmin(dist2, axis=1)
        rows = np.arange(n)
        best_ids = ids[rows, best_local]
        best_t = t[rows, best_local]
        q = Q[rows, best_local]
        delta = np.sqrt(dist2[rows, best_local])
        Rloc = self.R0[best_ids] + best_t * (self.R1[best_ids] - self.R0[best_ids])
        return delta, Rloc, q


def _get_attr_any(obj: Any, names: tuple[str, ...]) -> Optional[Any]:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def extract_lumen_arrays(lumen_query: Any) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort extraction from common LumenQuery attribute names."""
    C = _get_attr_any(lumen_query, ("C", "centerline", "centerline_points", "lumen_C", "points", "P"))
    R = _get_attr_any(lumen_query, ("R", "radius", "radii", "lumen_R"))
    if C is None or R is None:
        raise AttributeError(
            "Could not extract centerline/radii from lumen_query. "
            "Either add attributes lumen_query.C and lumen_query.R, or pass lumen_C/lumen_R "
            "through a case factory for benchmarking."
        )
    return np.asarray(C, float), np.asarray(R, float)


def contact_barrier_energy_and_force_vectorized(
    p: np.ndarray,
    batch_query: PolylineBatchLumenQuery,
    *,
    r_beam: float = 0.0,
    k_contact: float = 3e5,
    pen_switch: float = 5e-4,
    k_hard: float = 3e5,
    eps: float = 1e-12,
    smooth: bool = False,
    smooth_eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized replacement for contact_barrier_energy_and_force_fast."""
    P = np.asarray(p, dtype=float)
    if P.ndim != 2 or P.shape[0] != 3:
        raise ValueError(f"expected p with shape (3,N), got {P.shape}")

    delta, Rloc, q_closest = batch_query.closest_batch(P)
    X = P.T

    nvec = (X - q_closest) / np.maximum(delta[:, None], eps)
    tiny = delta <= eps
    if np.any(tiny):
        nvec[tiny] = np.array([1.0, 0.0, 0.0])

    gap = Rloc - delta - float(r_beam)
    phi = -gap
    C = np.zeros(P.shape[1], dtype=float)
    F = np.zeros((P.shape[1], 3), dtype=float)

    if smooth:
        root = np.sqrt(phi * phi + smooth_eps * smooth_eps)
        phi_pos = 0.5 * (phi + root)
        dphi_pos_dphi = 0.5 * (1.0 + phi / root)
        C[:] = 0.5 * k_contact * phi_pos * phi_pos
        mag = k_contact * phi_pos * dphi_pos_dphi
        F[:] = -mag[:, None] * nvec
    else:
        active = phi > 0.0
        soft = active & (phi <= pen_switch)
        hard = active & ~soft

        C[soft] = 0.5 * k_contact * phi[soft] ** 2
        F[soft] = -(k_contact * phi[soft])[:, None] * nvec[soft]

        if np.any(hard):
            dp = phi[hard] - pen_switch
            C0 = 0.5 * k_contact * pen_switch ** 2
            F0 = k_contact * pen_switch
            C[hard] = C0 + F0 * dp + 0.5 * k_hard * dp ** 2
            F[hard] = -(F0 + k_hard * dp)[:, None] * nvec[hard]

    return C, F.T, gap


_FAST_QUERY_CACHE: dict[tuple[int, int, bool], PolylineBatchLumenQuery] = {}


def make_fast_contact_function(original_fn: Callable, config: FastContactConfig) -> Callable:
    """Create a same-signature monkey-patch wrapper."""

    def fast_contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        *,
        r_beam=0.0,
        k_contact=3e5,
        pen_switch=5e-4,
        k_hard=3e5,
        eps=1e-12,
        window=3,
        smooth=False,
        smooth_eps=1e-7,
        debug=False,
    ):
        # If the caller requests a local moving window, preserve the original semantics.
        # Your current slow run uses window=None, which is the best target for this patch.
        if window is not None:
            return original_fn(
                p, lumen_query,
                r_beam=r_beam, k_contact=k_contact, pen_switch=pen_switch,
                k_hard=k_hard, eps=eps, window=window, smooth=smooth,
                smooth_eps=smooth_eps, debug=debug,
            )

        try:
            C_arr, R_arr = extract_lumen_arrays(lumen_query)
            key = (id(lumen_query), config.candidate_k, config.exact_all_segments)
            batch_query = _FAST_QUERY_CACHE.get(key)
            if batch_query is None:
                batch_query = PolylineBatchLumenQuery(
                    C_arr, R_arr,
                    candidate_k=config.candidate_k,
                    exact_all_segments=config.exact_all_segments,
                )
                _FAST_QUERY_CACHE[key] = batch_query
            return contact_barrier_energy_and_force_vectorized(
                p, batch_query,
                r_beam=r_beam,
                k_contact=k_contact,
                pen_switch=pen_switch,
                k_hard=k_hard,
                eps=eps,
                smooth=smooth,
                smooth_eps=smooth_eps,
            )
        except Exception:
            if not config.validate_fallback:
                raise
            return original_fn(
                p, lumen_query,
                r_beam=r_beam, k_contact=k_contact, pen_switch=pen_switch,
                k_hard=k_hard, eps=eps, window=window, smooth=smooth,
                smooth_eps=smooth_eps, debug=debug,
            )

    fast_contact_barrier_energy_and_force_fast.__name__ = original_fn.__name__
    fast_contact_barrier_energy_and_force_fast.__doc__ = "Vectorized monkey-patched contact kernel."
    return fast_contact_barrier_energy_and_force_fast


def install_fast_contact(model: Any, config: FastContactConfig) -> Callable:
    original = getattr(model, "contact_barrier_energy_and_force_fast")
    setattr(model, "_original_contact_barrier_energy_and_force_fast", original)
    setattr(model, "contact_barrier_energy_and_force_fast", make_fast_contact_function(original, config))
    return original


def uninstall_fast_contact(model: Any, original: Callable) -> None:
    setattr(model, "contact_barrier_energy_and_force_fast", original)


def _time_call(fn: Callable, repeats: int) -> tuple[float, Any]:
    best = float("inf")
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, last


def _make_synthetic_p(lumen_C: np.ndarray, lumen_R: np.ndarray, n_nodes: int = 60) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).reshape(-1)
    idx = np.linspace(0, len(C) - 1, n_nodes).round().astype(int)
    base = C[idx]
    rng = np.random.default_rng(4)
    noise = rng.normal(size=base.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True) + 1e-15
    # Straddle the wall so both zero/nonzero penalty branches are exercised.
    offset = (R[idx] - np.median(R) * 0.85 + 0.35 * np.median(R) * rng.normal(size=(n_nodes,)))[:, None]
    p = (base + offset * noise).T
    s = np.linspace(0.0, 1.0, n_nodes)
    return p, s


def _contact_kwargs_from_contact(contact: Any) -> dict[str, Any]:
    return dict(
        r_beam=getattr(contact, "r_beam", 0.0),
        k_contact=getattr(contact, "k", 3e5),
        pen_switch=getattr(contact, "pen_switch", 5e-4),
        k_hard=getattr(contact, "k_hard", 3e5),
        eps=1e-12,
        window=getattr(contact, "window", None),
        smooth=getattr(contact, "smooth", False),
        smooth_eps=getattr(contact, "smooth_eps", 1e-7),
    )


def benchmark_contact_kernel(model: Any, case: dict[str, Any], config: FastContactConfig, repeats: int) -> None:
    lumen_C = np.asarray(case["lumen_C"], float)
    lumen_R = np.asarray(case["lumen_R"], float)
    lumen_query = case.get("lumen_query")
    if lumen_query is None:
        lumen_query = model.LumenQuery(lumen_C, lumen_R)

    p = case.get("p")
    s = case.get("s")
    if p is None or s is None:
        p, s = _make_synthetic_p(lumen_C, lumen_R, n_nodes=int(case.get("n_contact_nodes", 80)))

    contact = case.get("contact", model.ContactParams())
    kwargs = _contact_kwargs_from_contact(contact)
    kwargs["window"] = None  # target the global closest-query case for benchmarking

    original = getattr(model, "contact_barrier_energy_and_force_fast")
    fast = make_fast_contact_function(original, config)

    slow_t, slow_out = _time_call(lambda: original(p, lumen_query, **kwargs), repeats)
    fast_t, fast_out = _time_call(lambda: fast(p, lumen_query, **kwargs), repeats)

    C0, F0, gap0 = slow_out
    C1, F1, gap1 = fast_out
    w = model.uniform_node_weights_like_current_energy(np.asarray(s, float))
    W0 = float(np.dot(w, C0))
    W1 = float(np.dot(w, C1))

    print("\nCONTACT KERNEL BENCHMARK")
    print(f"  nodes                  : {np.asarray(p).shape[1]}")
    print(f"  lumen samples          : {len(lumen_C)}")
    print(f"  slow best              : {slow_t:.6e} s")
    print(f"  fast best              : {fast_t:.6e} s")
    print(f"  speedup                : {slow_t / fast_t:.2f}x")
    print(f"  energy slow/fast       : {W0:.12e} / {W1:.12e}")
    print(f"  abs energy error       : {abs(W1 - W0):.6e}")
    print(f"  max |C_fast-C_slow|    : {np.max(np.abs(C1 - C0)):.6e}")
    print(f"  max |F_fast-F_slow|    : {np.max(np.abs(F1 - F0)):.6e}")
    print(f"  max |gap_fast-gap_slow|: {np.max(np.abs(gap1 - gap0)):.6e}")


def benchmark_full_solve(model: Any, case: dict[str, Any], config: FastContactConfig, repeats: int) -> None:
    solve_kwargs = case.get("solve_kwargs")
    if not solve_kwargs:
        print("\nFULL SOLVE BENCHMARK SKIPPED: case did not provide solve_kwargs.")
        return

    solve_fn = getattr(model, "solve_quasistatic_insertion")

    def summarize(hist):
        final = hist[-1]
        info = final["info"]
        return dict(
            W=float(info["W"]),
            W_cf=float(info["parts"].get("W_cf", np.nan)),
            nit=int(info.get("nit", -1)),
            p_end=np.asarray(final["p"], float)[:, -1].copy(),
            u=np.asarray(info["u_flat_opt"], float).copy(),
        )

    original = getattr(model, "contact_barrier_energy_and_force_fast")

    slow_t, slow_hist = _time_call(lambda: solve_fn(**solve_kwargs), repeats)
    slow_sum = summarize(slow_hist)

    installed_original = install_fast_contact(model, config)
    try:
        fast_t, fast_hist = _time_call(lambda: solve_fn(**solve_kwargs), repeats)
    finally:
        uninstall_fast_contact(model, installed_original)
    fast_sum = summarize(fast_hist)

    print("\nFULL FORWARD SOLVE BENCHMARK")
    print(f"  slow best              : {slow_t:.6e} s")
    print(f"  fast best              : {fast_t:.6e} s")
    print(f"  speedup                : {slow_t / fast_t:.2f}x")
    print(f"  nit slow/fast          : {slow_sum['nit']} / {fast_sum['nit']}")
    print(f"  W slow/fast            : {slow_sum['W']:.12e} / {fast_sum['W']:.12e}")
    print(f"  W_cf slow/fast         : {slow_sum['W_cf']:.12e} / {fast_sum['W_cf']:.12e}")
    print(f"  |Δ end position|       : {np.linalg.norm(fast_sum['p_end'] - slow_sum['p_end']):.6e}")
    print(f"  |Δ u_flat_opt|         : {np.linalg.norm(fast_sum['u'] - slow_sum['u']):.6e}")


def load_case(model: Any, case_fn_name: str) -> dict[str, Any]:
    if not hasattr(model, case_fn_name):
        raise AttributeError(
            f"Module {model.__name__!r} does not define {case_fn_name!r}. "
            "Add a function that returns {'lumen_C': ..., 'lumen_R': ..., 'contact': ..., optionally 'solve_kwargs': ...}."
        )
    case_fn = getattr(model, case_fn_name)
    if len(inspect.signature(case_fn).parameters) != 0:
        raise TypeError(f"{case_fn_name} must take no arguments")
    case = case_fn()
    if not isinstance(case, dict):
        raise TypeError(f"{case_fn_name} must return a dict")
    if "lumen_C" not in case or "lumen_R" not in case:
        raise KeyError("case dict must contain lumen_C and lumen_R")
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Import path of your forward model module, e.g. my_model")
    parser.add_argument("--case-fn", default="make_benchmark_case", help="No-arg function in module returning benchmark case dict")
    parser.add_argument("--repeat", type=int, default=10, help="Benchmark repeats; best time is reported")
    parser.add_argument("--candidate-k", type=int, default=16, help="KD-tree segment candidates per rod node")
    parser.add_argument("--exact-all-segments", action="store_true", help="Use all lumen segments, no KD-tree approximation")
    parser.add_argument("--no-fallback", action="store_true", help="Raise instead of falling back to original on extraction errors")
    args = parser.parse_args()

    model = importlib.import_module(args.module)
    case = load_case(model, args.case_fn)
    config = FastContactConfig(
        candidate_k=args.candidate_k,
        exact_all_segments=args.exact_all_segments,
        validate_fallback=not args.no_fallback,
    )

    benchmark_contact_kernel(model, case, config, repeats=max(1, args.repeat))
    benchmark_full_solve(model, case, config, repeats=1)

    print("\nINSTALLATION SNIPPET")
    print("  from fast_contact_benchmark import FastContactConfig, install_fast_contact")
    print("  install_fast_contact(your_model_module, FastContactConfig(candidate_k=%d, exact_all_segments=%s))" % (
        config.candidate_k, config.exact_all_segments
    ))


if __name__ == "__main__":
    main()
