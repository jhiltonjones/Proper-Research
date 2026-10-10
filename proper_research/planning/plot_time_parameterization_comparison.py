"""Plot the optimal vs standard time-parameterization comparison.

Reads the output of ``run_time_parameterization.py --profile both``:
a directory holding ``optimal/`` and ``standard/`` subdirectories (each with
``time_parameterized_configuration_path.csv``) plus ``profile_comparison.json``.

Writes ``time_parameterization_comparison.png`` into that directory.

Usage
-----
    python -m proper_research.planning.plot_time_parameterization_comparison \\
        --comparison-dir uprgrade_configuration/bends_pXX_.../time_param_profile_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOINT_QD = [f"qd{i}_rad_s" for i in range(1, 7)]
JOINT_QDD = [f"qdd{i}_rad_s2" for i in range(1, 7)]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the time-param profile comparison.")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="")
    return parser.parse_args()


def resolve_effective_limits(comparison_dir: Path) -> tuple[float, float, float]:
    """(effective joint velocity limit rad/s, effective joint accel limit rad/s^2,
    maximum path speed m/s) -- the limits the parameteriser actually respects,
    i.e. state limits x safety factor."""
    for profile in ("optimal", "standard"):
        summary = comparison_dir / profile / "time_parameterized_configuration_summary.json"
        if summary.exists():
            cfg = json.loads(summary.read_text()).get("configuration", {})
            v = float(min(cfg.get("state_velocity_limit", [0.5])[:6]))
            a = float(min(cfg.get("state_acceleration_limit", [0.5])[:6]))
            vsf = float(cfg.get("velocity_safety_factor", 1.0))
            asf = float(cfg.get("acceleration_safety_factor", 1.0))
            v_path = float(cfg.get("maximum_path_speed_m_s", 0.005)) * vsf
            return v * vsf, a * asf, v_path
    return 0.4, 0.4, 0.004


def load_profile(directory: Path) -> dict[str, np.ndarray]:
    path = directory / "time_parameterized_configuration_path.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    arc_mm = 1.0e3 * (col("path_s_m") - float(rows[0]["path_s_m"]))
    qd = np.stack([np.abs(col(name)) for name in JOINT_QD], axis=1)      # (N, 6)
    qdd = np.stack([np.abs(col(name)) for name in JOINT_QDD], axis=1)
    return {
        "arc_mm": arc_mm,
        "time_s": col("time_s"),
        "path_speed": col("path_speed_m_s"),
        "path_accel": col("path_acceleration_m_s2"),
        "qd_max": qd.max(axis=1),
        "qdd_max": qdd.max(axis=1),
        "insertion_rate": np.abs(col("insertion_rate_m_s")),
    }


def main() -> int:
    args = _arguments()
    cmp_dir = args.comparison_dir
    comparison = json.loads((cmp_dir / "profile_comparison.json").read_text())
    optimal = load_profile(cmp_dir / "optimal")
    standard = load_profile(cmp_dir / "standard")

    v_limit, a_limit, v_path_limit = resolve_effective_limits(cmp_dir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Time parameterization: optimal vs standard trapezoidal{suffix}\n"
        f"same geometric path, same limits -- "
        f"standard {comparison['standard_duration_s']:.1f} s  ->  "
        f"optimal {comparison['optimal_duration_s']:.1f} s  "
        f"({comparison['percent_faster']:.0f}% faster)",
        fontsize=13, y=0.99,
    )

    styles = {"optimal": dict(color="tab:blue", lw=1.8),
              "standard": dict(color="tab:red", lw=1.8, ls="--")}

    # (a) path speed vs arc length
    ax = axes[0, 0]
    for label, data in (("optimal", optimal), ("standard", standard)):
        ax.plot(data["arc_mm"], 1e3 * data["path_speed"], label=label, **styles[label])
    ax.axhline(1e3 * v_path_limit, color="grey", ls=":", lw=1)
    ax.text(0, 1e3 * v_path_limit, " path-speed cap", color="grey", va="bottom", fontsize=7)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("path speed [mm/s]")
    ax.set_title("(a) path speed: optimal rides the cap on the easy run, "
                 "slows only at the bend", fontsize=9)
    ax.legend(fontsize=9)

    # (b) path acceleration vs arc length
    ax = axes[0, 1]
    for label, data in (("optimal", optimal), ("standard", standard)):
        ax.plot(data["arc_mm"], 1e3 * data["path_accel"], label=label, **styles[label])
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("path acceleration [mm/s^2]")
    ax.set_title("(b) path acceleration along the path", fontsize=10)
    ax.legend(fontsize=9)

    # (c) cumulative time vs arc length -- where the standard profile spends it
    ax = axes[0, 2]
    for label, data in (("optimal", optimal), ("standard", standard)):
        ax.plot(data["arc_mm"], data["time_s"], label=label, **styles[label])
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("elapsed time [s]")
    ax.set_title("(c) elapsed time vs arc length", fontsize=10)
    ax.legend(fontsize=9)

    # (d) worst-actuator velocity utilisation vs arc length
    ax = axes[1, 0]
    for label, data in (("optimal", optimal), ("standard", standard)):
        ax.plot(data["arc_mm"], 100 * data["qd_max"] / v_limit, label=label, **styles[label])
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"max$_j$ $|\dot q_j|$ / effective limit  [%]")
    ax.set_title(f"(d) worst-joint velocity utilisation (eff. limit {v_limit:.3g} rad/s)", fontsize=10)
    ax.legend(fontsize=9)

    # (e) worst-actuator acceleration utilisation vs arc length
    ax = axes[1, 1]
    for label, data in (("optimal", optimal), ("standard", standard)):
        ax.plot(data["arc_mm"], 100 * data["qdd_max"] / a_limit, label=label, **styles[label])
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel(r"max$_j$ $|\ddot q_j|$ / effective limit  [%]")
    ax.set_title(f"(e) worst-joint acceleration utilisation (eff. limit {a_limit:.3g} rad/s^2)",
                 fontsize=10)
    ax.legend(fontsize=9)

    # (f) headline bars
    ax = axes[1, 2]
    labels = ["optimal", "standard"]
    durations = [comparison["optimal_duration_s"], comparison["standard_duration_s"]]
    peak_v = [comparison["profiles"]["optimal"]["peak_velocity_utilisation"],
              comparison["profiles"]["standard"]["peak_velocity_utilisation"]]
    peak_a = [comparison["profiles"]["optimal"]["peak_acceleration_utilisation"],
              comparison["profiles"]["standard"]["peak_acceleration_utilisation"]]
    x = np.arange(2)
    ax.bar(x - 0.2, durations, 0.4, color=["tab:blue", "tab:red"], label="duration [s]")
    ax.set_ylabel("traversal duration [s]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax2 = ax.twinx()
    ax2.plot(x, [100 * v for v in peak_v], "o-", color="tab:green", label="peak vel. util. [%]")
    ax2.plot(x, [100 * a for a in peak_a], "s--", color="tab:purple", label="peak accel. util. [%]")
    ax2.set_ylabel("peak limit utilisation [%]")
    ax2.set_ylim(0, 110)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")
    ax.set_title("(f) duration and how hard each profile pushes the limits", fontsize=10)

    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.06, right=0.95,
                        hspace=0.32, wspace=0.32)
    out = cmp_dir / "time_parameterization_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
