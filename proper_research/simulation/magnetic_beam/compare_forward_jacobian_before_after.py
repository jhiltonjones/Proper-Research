from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from proper_research.simulation.magnetic_beam.forward_model_optimized import (
    MagneticBeamForwardModelOptimized,
)
from proper_research.simulation.simulations.initial_conditions import (
    make_initial_poses,
)
from proper_research.simulation.simulations.model_factory import (
    build_model_bundle,
)
from proper_research.simulation.simulations.scenario import LumenBend, LumenConfig


def make_default_lumen_config() -> LumenConfig:
    """Same compact double-bend geometry used by the research grid."""
    return LumenConfig(
        length=0.06,
        n_pts=240,
        n_ref_pts=100,
        radius=0.006,
        ds_target=1e-3,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(0.0),
                bend_start=0.0,
                bend_end=0.01,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(-6.0),
                bend_start=0.02,
                bend_end=0.025,
            ),
        ),
    )


def load_poses(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        poses = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        for key in ("p7", "poses", "p7_sequence"):
            if key in archive:
                poses = archive[key]
                break
        else:
            raise KeyError(
                f"{path} must contain one of: p7, poses, p7_sequence."
            )
    elif path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = ["x", "y", "z", "rx", "ry", "rz", "L"]
        if not rows or not all(name in rows[0] for name in required):
            raise ValueError(
                "CSV must contain columns x,y,z,rx,ry,rz,L."
            )
        poses = np.asarray(
            [[float(row[name]) for name in required] for row in rows],
            dtype=float,
        )
    else:
        raise ValueError("Pose file must be .npy, .npz, or .csv.")

    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Pose array must have shape (N, 7), got {poses.shape}.")
    if not np.all(np.isfinite(poses)) or np.any(poses[:, 6] <= 0.0):
        raise ValueError("Pose sequence contains invalid values.")
    return poses


def generate_small_pose_trajectory(p7_start: np.ndarray, count: int) -> np.ndarray:
    """
    Generate small, smooth perturbations so both implementations receive the
    same realistic warm-start sequence. Replace this with a recorded trajectory
    for publication-quality benchmarking.
    """
    if count < 2:
        raise ValueError("count must be >= 2.")
    p7_start = np.asarray(p7_start, dtype=float).reshape(7)
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    poses = np.repeat(p7_start[None, :], count, axis=0)
    poses[:, 0] += 0.4e-3 * np.sin(phase)
    poses[:, 1] += 0.3e-3 * (1.0 - np.cos(phase))
    poses[:, 4] += np.deg2rad(0.20) * np.sin(phase)
    poses[:, 5] += np.deg2rad(0.25) * np.cos(phase)
    poses[:, 6] += 0.15e-3 * np.sin(2.0 * phase)
    return poses


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _percentile(values: list[float], q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else np.nan


def _median(values) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if arr.size else np.nan


def _maximum(values) -> float:
    arr = _finite(values)
    return float(np.max(arr)) if arr.size else np.nan


def environment_metadata() -> dict:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MAGBEAM_SENSITIVITY_WORKERS",
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "environment": {name: os.environ.get(name, "") for name in thread_vars},
    }


def _safe_float(info: dict, key: str, default=np.nan) -> float:
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def benchmark_model(
    model,
    poses: np.ndarray,
    *,
    implementation: str,
    repeats: int,
    jacobian_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    for repeat in range(repeats):
        model.reset_cache()
        if hasattr(model, "reset_performance_counters"):
            model.reset_performance_counters()

        for pose_index, p7 in enumerate(poses):
            forward_started = time.perf_counter_ns()
            result = model.solve(p7, commit=True, reuse_cache=False)
            forward_s = (time.perf_counter_ns() - forward_started) * 1e-9

            jacobian_started = time.perf_counter_ns()
            J = model.jacobian_tip_pose7(
                p7,
                solve_if_needed=False,
                mode=jacobian_mode,
                reuse_cached=False,
            )
            jacobian_s = (time.perf_counter_ns() - jacobian_started) * 1e-9

            info = dict(result.info or {})
            sens_info = dict(getattr(model, "last_sens_info", {}) or {})
            rows.append(
                {
                    "implementation": implementation,
                    "repeat": repeat,
                    "pose_index": pose_index,
                    "warmup": int(pose_index == 0),
                    "forward_s": forward_s,
                    "jacobian_s": jacobian_s,
                    "combined_s": forward_s + jacobian_s,
                    "success": int(bool(info.get("success", False))),
                    "solve_path": str(info.get("solve_path", "")),
                    "nit": int(info.get("nit", 0)),
                    "work_total_nit": int(
                        info.get("work_total_nit", info.get("total_nit", 0))
                    ),
                    "objective_calls": int(info.get("objective_calls", 0)),
                    "gradient_calls": int(info.get("gradient_calls", 0)),
                    "fixed_solve_wall_s": _safe_float(
                        info, "fixed_solve_wall_s"
                    ),
                    "sensitivity_time_s": _safe_float(
                        sens_info, "sensitivity_time_s"
                    ),
                    "hessian_time_s": _safe_float(sens_info, "hessian_time_s"),
                    "gtheta_time_s": _safe_float(sens_info, "gtheta_time_s"),
                    "linear_solve_time_s": _safe_float(
                        sens_info, "linear_solve_time_s"
                    ),
                    "hessian_reused": int(
                        bool(sens_info.get("hessian_reused", False))
                    ),
                    "tip_x": float(result.tip[0]),
                    "tip_y": float(result.tip[1]),
                    "tip_z": float(result.tip[2]),
                    **{
                        f"J_{i}_{j}": float(J[i, j])
                        for i in range(J.shape[0])
                        for j in range(J.shape[1])
                    },
                }
            )
    return rows


def add_accuracy_columns(rows: list[dict]) -> None:
    by_key: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["repeat"]), int(row["pose_index"]))
        by_key.setdefault(key, {})[str(row["implementation"])] = row

    for pair in by_key.values():
        if "legacy" not in pair or "optimized" not in pair:
            continue
        old = pair["legacy"]
        new = pair["optimized"]
        tip_old = np.array([old["tip_x"], old["tip_y"], old["tip_z"]])
        tip_new = np.array([new["tip_x"], new["tip_y"], new["tip_z"]])
        j_keys = sorted(key for key in old if key.startswith("J_"))
        J_old = np.array([old[key] for key in j_keys], dtype=float)
        J_new = np.array([new[key] for key in j_keys], dtype=float)

        tip_abs = float(np.linalg.norm(tip_new - tip_old))
        J_abs = float(np.linalg.norm(J_new - J_old))
        J_rel = J_abs / max(float(np.linalg.norm(J_old)), 1e-15)
        for row in (old, new):
            row["tip_error_vs_legacy_m"] = tip_abs
            row["jacobian_abs_error_vs_legacy"] = J_abs
            row["jacobian_rel_error_vs_legacy"] = J_rel


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for implementation in ("legacy", "optimized"):
        selected = [
            row
            for row in rows
            if row["implementation"] == implementation and not row["warmup"]
        ]
        forward = [float(row["forward_s"]) for row in selected]
        jac = [float(row["jacobian_s"]) for row in selected]
        combined = [float(row["combined_s"]) for row in selected]
        summary[implementation] = {
            "calls": len(selected),
            "success_rate": (
                float(np.mean([row["success"] for row in selected]))
                if selected
                else np.nan
            ),
            "forward_median_ms": 1e3 * _median(forward),
            "forward_p95_ms": 1e3 * _percentile(forward, 95),
            "forward_p99_ms": 1e3 * _percentile(forward, 99),
            "jacobian_median_ms": 1e3 * _median(jac),
            "jacobian_p95_ms": 1e3 * _percentile(jac, 95),
            "jacobian_p99_ms": 1e3 * _percentile(jac, 99),
            "combined_median_ms": 1e3 * _median(combined),
            "combined_p95_ms": 1e3 * _percentile(combined, 95),
            "combined_p99_ms": 1e3 * _percentile(combined, 99),
        }

    old = summary["legacy"]
    new = summary["optimized"]
    summary["speedup"] = {
        "forward_median_x": old["forward_median_ms"]
        / max(new["forward_median_ms"], 1e-15),
        "jacobian_median_x": old["jacobian_median_ms"]
        / max(new["jacobian_median_ms"], 1e-15),
        "combined_median_x": old["combined_median_ms"]
        / max(new["combined_median_ms"], 1e-15),
    }

    accuracy = [
        row
        for row in rows
        if row["implementation"] == "optimized" and not row["warmup"]
    ]
    tip_errors = [
        float(row.get("tip_error_vs_legacy_m", np.nan)) for row in accuracy
    ]
    jac_errors = [
        float(row.get("jacobian_rel_error_vs_legacy", np.nan))
        for row in accuracy
    ]
    summary["accuracy"] = {
        "tip_error_max_m": _maximum(tip_errors),
        "tip_error_median_m": _median(tip_errors),
        "jacobian_relative_error_max": _maximum(jac_errors),
        "jacobian_relative_error_median": _median(jac_errors),
    }
    return summary


def save_csv(path: Path, rows: list[dict]) -> None:
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    clean = [row for row in rows if not row["warmup"]]
    labels = ["Forward", "Jacobian", "Combined"]
    keys = ["forward_s", "jacobian_s", "combined_s"]
    legacy = [
        1e3
        * _median(
            [row[key] for row in clean if row["implementation"] == "legacy"]
        )
        for key in keys
    ]
    optimized = [
        1e3
        * _median(
            [row[key] for row in clean if row["implementation"] == "optimized"]
        )
        for key in keys
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - width / 2, legacy, width, label="Legacy")
    ax.bar(x + width / 2, optimized, width, label="Optimized")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median time [ms]")
    ax.set_title("Forward and implicit-Jacobian timing")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "timing_comparison.png", dpi=200)
    plt.close(fig)

    opt = [row for row in clean if row["implementation"] == "optimized"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].plot(
        [row["pose_index"] for row in opt],
        [1e3 * row["tip_error_vs_legacy_m"] for row in opt],
        marker="o",
    )
    axes[0].set_ylabel("Tip difference [mm]")
    axes[0].set_xlabel("Pose index")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        [row["pose_index"] for row in opt],
        [row["jacobian_rel_error_vs_legacy"] for row in opt],
        marker="o",
    )
    axes[1].set_ylabel("Relative Jacobian difference")
    axes[1].set_xlabel("Pose index")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy and optimized magnetic-beam forward/Jacobian code."
    )
    parser.add_argument("--variant", choices=("contact", "no_contact"), default="contact")
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--num-poses", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jacobian-mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--result-detail", choices=("none", "contact", "full"), default="contact")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_forward_jacobian"))
    parser.add_argument(
        "--tip-atol",
        type=float,
        default=1e-8,
        help="Maximum accepted tip-position difference in metres.",
    )
    parser.add_argument(
        "--jacobian-rtol",
        type=float,
        default=5e-3,
        help="Maximum accepted relative Frobenius-norm Jacobian difference.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when accuracy or solve-success checks fail.",
    )
    args = parser.parse_args()

    if args.repeats < 1 or args.workers < 1:
        raise ValueError("repeats and workers must be >= 1.")
    if args.tip_atol < 0.0 or args.jacobian_rtol < 0.0:
        raise ValueError("Accuracy tolerances must be non-negative.")

    pivot_point, start_point, L0, _ = make_initial_poses()
    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=make_default_lumen_config(),
        plant_contact=True,
    )
    legacy_model = bundle.models[args.variant]
    optimized_model = MagneticBeamForwardModelOptimized.from_legacy(
        legacy_model,
        result_detail=args.result_detail,
        store_history=False,
        store_vectors_in_info=False,
        sensitivity_workers=args.workers,
    )

    if args.poses is not None:
        poses = load_poses(args.poses)
    else:
        p7_start = np.concatenate(
            [np.asarray(start_point, dtype=float).reshape(6), [float(L0)]]
        )
        poses = generate_small_pose_trajectory(p7_start, args.num_poses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "poses_used.npy", poses)

    print(f"Benchmarking {poses.shape[0]} poses x {args.repeats} repeats...")
    legacy_rows = benchmark_model(
        legacy_model,
        poses,
        implementation="legacy",
        repeats=args.repeats,
        jacobian_mode=args.jacobian_mode,
    )
    optimized_rows = benchmark_model(
        optimized_model,
        poses,
        implementation="optimized",
        repeats=args.repeats,
        jacobian_mode=args.jacobian_mode,
    )
    rows = legacy_rows + optimized_rows
    add_accuracy_columns(rows)
    summary = summarize(rows)
    summary["environment"] = environment_metadata()
    tip_ok = bool(
        np.isfinite(summary["accuracy"]["tip_error_max_m"])
        and summary["accuracy"]["tip_error_max_m"] <= args.tip_atol
    )
    jac_ok = bool(
        np.isfinite(summary["accuracy"]["jacobian_relative_error_max"])
        and summary["accuracy"]["jacobian_relative_error_max"]
        <= args.jacobian_rtol
    )
    success_ok = bool(
        summary["legacy"]["success_rate"] == 1.0
        and summary["optimized"]["success_rate"] == 1.0
    )
    summary["acceptance"] = {
        "tip_atol_m": float(args.tip_atol),
        "jacobian_rtol": float(args.jacobian_rtol),
        "tip_ok": tip_ok,
        "jacobian_ok": jac_ok,
        "all_solves_successful": success_ok,
        "passed": bool(tip_ok and jac_ok and success_ok),
    }

    save_csv(args.output_dir / "per_call_results.csv", rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    save_plots(args.output_dir, rows)

    print(json.dumps(summary, indent=2))
    print(f"Results written to {args.output_dir.resolve()}")
    if args.strict and not summary["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
