"""Look at an inverse-configuration (Layer 1) path in isolation and judge it.

The planner's own ``inverse_configuration_path.png`` buries the beam-tip path
in a tiny 3-D axis. This makes the one plot that answers "is this a good
path?" -- the tip path in the plane it actually lives in, against the lumen
wall, with every node marked -- plus the diagnostics that decide whether a
*controller* can then track it: tracking error, tip-path curvature (jagged
spikes), node-to-node step sizes, task-Jacobian conditioning, and the safety
margins.

Writes ``inverse_configuration_path_review.png`` into --inverse-dir and prints
a short verdict.

Usage
-----
    python -m proper_research.planning.plot_inverse_configuration_path \\
        --inverse-dir uprgrade_configuration/bends_p30_m70_.../offline_inverse_configuration_60 \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review an inverse-configuration path in isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inverse-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--no-beam-shapes", action="store_true",
                         help="Skip the beam-shape overlay (no model build).")
    try:
        from proper_research.experiments.pinned_planning_context import add_geometry_arguments
        add_geometry_arguments(parser)
    except ImportError:
        pass
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    def col(name: str) -> np.ndarray:
        return np.array([float(r.get(name, "nan")) for r in rows])

    out = {"n_nodes": len(rows)}
    for name in (
        "s_m", "insertion_m", "position_error_mm", "tangent_error_deg",
        "delta_q_norm_rad", "delta_insertion_m", "jacobian_condition",
        "jacobian_effective_rank", "minimum_joint_margin_rad",
        "minimum_state_margin", "source_magnet_lumen_margin_m",
        "source_magnet_lumen_exclusion_radius_m",
        "tip_x", "tip_y", "tip_z", "desired_x", "desired_y", "desired_z",
        "tip_tangent_x", "tip_tangent_y", "desired_tangent_x", "desired_tangent_y",
        "contact_active",
    ):
        out[name] = col(name)
    out["feasible"] = np.array([str(r.get("feasible", "")).strip().lower()
                                in ("1", "true", "yes") for r in rows])
    out["q_rad"] = np.stack([col(f"q{i}_rad") for i in range(1, 7)], axis=1)
    return out


def discrete_tip_curvature(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Menger curvature at each interior node (1/radius of the circle through 3 pts)."""
    kappa = np.zeros_like(s)
    for i in range(1, len(s) - 1):
        a = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        b = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        c = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        area2 = abs((x[i] - x[i - 1]) * (y[i + 1] - y[i - 1])
                    - (x[i + 1] - x[i - 1]) * (y[i] - y[i - 1]))
        kappa[i] = (2.0 * area2) / (a * b * c) if a * b * c > 1e-15 else 0.0
    kappa[0], kappa[-1] = kappa[1], kappa[-2]
    return kappa


def try_load_lumen_and_beams(args, data):
    """Return (lumen_C, lumen_R, beam_shapes[list of (N,3)], beam_node_indices) or (None,...)."""
    try:
        from proper_research.experiments.pinned_planning_context import (
            resolve_planning_context_from_args,
        )
        from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
        exp_cfg, bundle0, pack, out_root = resolve_planning_context_from_args(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[review] could not build the model for the lumen/beam overlay: {exc!r}",
              file=sys.stderr)
        return None, None, [], []

    lumen_C = np.asarray(bundle0.lumen_C, dtype=float)
    lumen_R = np.asarray(bundle0.lumen_R, dtype=float)
    beam_shapes: list[np.ndarray] = []
    node_indices: list[int] = []
    if not args.no_beam_shapes:
        adapter = build_diagnostic_adapter(beam_model=bundle0.models["contact"],
                                           controller_pack=pack)
        n = data["n_nodes"]
        picks = np.unique(np.linspace(0, n - 1, 7).astype(int))
        for idx in picks:
            state = np.concatenate([data["q_rad"][idx], [data["insertion_m"][idx]]])
            try:
                adapter.forward_output(state, commit=True)
                pc = adapter.beam_output_fn.forward_adapter.last_p_centerline
                if pc is not None:
                    beam_shapes.append(np.asarray(pc, dtype=float).T)
                    node_indices.append(int(idx))
            except Exception:  # noqa: BLE001
                continue
    return lumen_C, lumen_R, beam_shapes, node_indices


def draw_lumen(ax, lumen_C, lumen_R, color="0.55"):
    C = np.asarray(lumen_C, float)[:, :2]
    R = np.asarray(lumen_R, float)
    t = np.gradient(C, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    for sgn in (+1.0, -1.0):
        w = C + sgn * nrm * R[:, None]
        ax.plot(1e3 * w[:, 0], 1e3 * w[:, 1], "-", color=color, lw=1.2)
    ax.plot(1e3 * C[:, 0], 1e3 * C[:, 1], "--", color=color, lw=0.8, alpha=0.6)


def main() -> int:
    args = _arguments()
    inv_dir: Path = args.inverse_dir
    data = load_csv(inv_dir / "inverse_configuration_path.csv")
    summary = {}
    sp = inv_dir / "inverse_configuration_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text()).get("summary", {})

    s_mm = 1e3 * (data["s_m"] - data["s_m"][0])
    tip = np.stack([data["tip_x"], data["tip_y"]], axis=1)
    des = np.stack([data["desired_x"], data["desired_y"]], axis=1)
    kappa = discrete_tip_curvature(data["tip_x"], data["tip_y"], data["s_m"])
    kappa_des = discrete_tip_curvature(data["desired_x"], data["desired_y"], data["s_m"])

    lumen_C, lumen_R, beam_shapes, beam_nodes = try_load_lumen_and_beams(args, data)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    n_feas = int(np.sum(data["feasible"]))

    # ------------------------------------------------------------------
    # FIGURE 1: the tip path itself, big -- "see it in isolation"
    # ------------------------------------------------------------------
    fig1, (axp, axd) = plt.subplots(1, 2, figsize=(17, 7))
    fig1.suptitle(f"Inverse-configuration beam-tip path{suffix}", fontsize=13, y=0.98)

    if lumen_C is not None:
        draw_lumen(axp, lumen_C, lumen_R)
    axp.plot(1e3 * des[:, 0], 1e3 * des[:, 1], "--", color="tab:blue", lw=1.6,
             label="desired tip path")
    axp.plot(1e3 * tip[:, 0], 1e3 * tip[:, 1], "-o", color="tab:red", ms=4, lw=1.6,
             label="achieved (57 nodes marked)")
    if beam_shapes:
        cols = cm.plasma(np.linspace(0, 0.9, len(beam_shapes)))
        for b, c in zip(beam_shapes, cols):
            axp.plot(1e3 * b[:, 0], 1e3 * b[:, 1], "-", color=c, lw=1.1, alpha=0.65)
        axp.plot([], [], "-", color=cols[len(cols) // 2], label="beam shape at sample nodes")
    axp.set_aspect("equal")
    axp.set_xlabel("x [mm]")
    axp.set_ylabel("y [mm]")
    axp.set_title("(a) the whole path, against the lumen wall", fontsize=11)
    axp.legend(fontsize=9)

    # right: achieved-minus-desired deviation, magnified, as a function of arc length
    dev = 1e3 * (tip - des)  # mm
    dev_norm = np.hypot(dev[:, 0], dev[:, 1])
    axd.plot(s_mm, dev[:, 0], "-o", ms=3, color="tab:green", label="x deviation [mm]")
    axd.plot(s_mm, dev[:, 1], "-s", ms=3, color="tab:purple", label="y deviation [mm]")
    axd.plot(s_mm, dev_norm, "-", color="tab:red", lw=1.6, label="|deviation| [mm]")
    axd.axhline(0.0, color="grey", ls=":", lw=1)
    axd.set_xlabel("arc length [mm]")
    axd.set_ylabel("achieved tip position − desired [mm]")
    axd.set_title(f"(b) tracking deviation along the path (max {dev_norm.max():.3f} mm)",
                  fontsize=11)
    axd.legend(fontsize=9)

    fig1.subplots_adjust(top=0.90, bottom=0.11, left=0.06, right=0.97, wspace=0.22)
    out1 = inv_dir / "inverse_configuration_tip_path.png"
    fig1.savefig(out1, dpi=140)
    plt.close(fig1)
    print("wrote", out1)

    # ------------------------------------------------------------------
    # FIGURE 2: the diagnostics that decide if a controller can track it
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(
        f"Inverse-configuration path review{suffix}\n"
        f"{data['n_nodes']} nodes, {n_feas} feasible, "
        f"termination={summary.get('termination_reason','?')}, "
        f"max pos err {np.nanmax(data['position_error_mm']):.3f} mm / "
        f"max tan err {np.nanmax(data['tangent_error_deg']):.1f} deg",
        fontsize=12, y=0.99,
    )

    # (a) achieved heading vs desired heading along the path (a cleaner
    #     smoothness read than 2nd-derivative curvature on scattered nodes)
    ax = axes[0, 0]
    def heading(v):
        h = np.degrees(np.arctan2(np.gradient(v[:, 1]), np.gradient(v[:, 0])))
        return np.unwrap(np.radians(h)) * 180.0 / np.pi
    ax.plot(s_mm, heading(des), "--", color="tab:blue", lw=1.6, label="desired path heading")
    ax.plot(s_mm, heading(tip), "-o", ms=3, color="tab:red", lw=1.4, label="achieved path heading")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("path heading angle [deg]")
    ax.set_title("(a) achieved vs desired path heading  (kinks show as steps)", fontsize=9)
    ax.legend(fontsize=8)

    # (b) tracking error along the path
    ax = axes[0, 1]
    ax.plot(s_mm, data["position_error_mm"], "-o", ms=3, color="tab:blue",
            label="position error [mm]")
    ax2 = ax.twinx()
    ax2.plot(s_mm, data["tangent_error_deg"], "-s", ms=3, color="tab:orange",
             label="tangent error [deg]")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("position error [mm]", color="tab:blue")
    ax2.set_ylabel("tangent error [deg]", color="tab:orange")
    ax.set_title("(b) how well each node hits its target", fontsize=9)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    # (c) tip-path curvature -- the "jagged spike" check
    ax = axes[0, 2]
    ax.plot(s_mm, kappa_des, "--", color="tab:blue", lw=1.4, label="desired path curvature")
    ax.plot(s_mm, kappa, "-o", ms=3, color="tab:red", lw=1.4, label="achieved path curvature")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("curvature  1/R  [1/m]")
    ax.set_title("(c) tip-path curvature -- 2nd-diff of node positions, amplifies "
                 "sub-0.1 mm node scatter", fontsize=9)
    ax.legend(fontsize=8)

    # (d) node-to-node step sizes vs their caps
    ax = axes[1, 0]
    ax.plot(s_mm, np.degrees(data["delta_q_norm_rad"]), "-o", ms=3, color="tab:green",
            label=r"$\|\Delta q\|$ per node [deg]")
    ax2 = ax.twinx()
    ax2.plot(s_mm, 1e3 * data["delta_insertion_m"], "-s", ms=3, color="tab:purple",
             label=r"$\Delta$ insertion per node [mm]")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("joint step [deg]", color="tab:green")
    ax2.set_ylabel("insertion step [mm]", color="tab:purple")
    ax.set_title("(d) node-to-node step sizes  (big jumps = planner struggling)", fontsize=9)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    # (e) task-Jacobian conditioning -- can a controller track this?
    ax = axes[1, 1]
    ax.plot(s_mm, data["jacobian_condition"], "-o", ms=3, color="crimson")
    ax.set_yscale("log")
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("tolerance-scaled task Jacobian condition number")
    ax.set_title("(e) conditioning -- millions means near-singular, hard to track", fontsize=9)
    for thresh, lbl in ((1e3, "1e3"), (1e5, "1e5")):
        ax.axhline(thresh, color="grey", ls=":", lw=0.8)

    # (f) safety margins
    ax = axes[1, 2]
    ax.plot(s_mm, 1e3 * data["source_magnet_lumen_margin_m"], "-o", ms=3, color="tab:blue",
            label="source-magnet keep-out margin [mm]")
    ax.plot(s_mm, np.degrees(data["minimum_joint_margin_rad"]), "-s", ms=3, color="tab:green",
            label="min joint-limit margin [deg]")
    ax.axhline(0.0, color="crimson", ls="--", lw=1)
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("margin")
    ax.set_title("(f) safety margins  (0 = on the constraint boundary)", fontsize=9)
    ax.legend(fontsize=8)

    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.05, right=0.96,
                        hspace=0.32, wspace=0.34)
    out = inv_dir / "inverse_configuration_path_review.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)

    # short verdict
    max_pos = float(np.nanmax(data["position_error_mm"]))
    max_tan = float(np.nanmax(data["tangent_error_deg"]))
    max_dev_mm = float(np.nanmax(dev_norm))
    # heading jump between consecutive achieved segments, minus the desired's own
    ach_h = heading(tip)
    des_h = heading(des)
    heading_kink = float(np.nanmax(np.abs(np.diff(ach_h) - np.diff(des_h))))
    max_step_deg = float(np.degrees(np.nanmax(data["delta_q_norm_rad"])))
    step_spike_ratio = (
        max_step_deg / float(np.degrees(np.nanmedian(data["delta_q_norm_rad"][1:])) + 1e-9)
    )
    max_cond = float(np.nanmax(data["jacobian_condition"]))
    min_magnet_margin_mm = float(1e3 * np.nanmin(data["source_magnet_lumen_margin_m"]))
    print("\n[review] verdict")
    print(f"  complete + all feasible : {n_feas == data['n_nodes']} "
          f"({summary.get('termination_reason','?')})")
    print(f"  position tracking       : max {max_pos:.3f} mm deviation "
          f"({'good' if max_dev_mm < 0.3 else 'check'})")
    print(f"  tangent tracking        : max {max_tan:.1f} deg  "
          f"({'good' if max_tan < 15 else 'loose -- under tolerance but a controller wants tighter'})")
    print(f"  path smoothness         : worst heading kink vs desired "
          f"{heading_kink:.1f} deg/segment  ({'smooth' if heading_kink < 10 else 'a real kink'})")
    print(f"  node steps              : max |dq| {max_step_deg:.2f} deg/node, "
          f"{step_spike_ratio:.1f}x the median  "
          f"({'even' if step_spike_ratio < 4 else 'one node jumps much more than the rest'})")
    print(f"  task-Jacobian condition : max {max_cond:.2e}  "
          f"({'ok' if max_cond < 1e4 else 'NEAR-SINGULAR -- a controller will fight this'})")
    print(f"  magnet keep-out margin  : min {min_magnet_margin_mm:.4f} mm  "
          f"({'ok' if min_magnet_margin_mm > 0.1 else 'ZERO -- pinned to the boundary, no headroom for L2'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
