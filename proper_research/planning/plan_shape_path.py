#!/usr/bin/env python3
"""Plan a triangle / square tip path through the full offline stack.

Feeds a synthetic shape centreline into the *existing* planner pipeline with
the *current* planning context (``build_planning_context`` -- same beam model,
magnet transform, robot config and state bounds the working runs use):

    shape lumen_C  ->  offline inverse configuration      (Layer 1)
                   ->  global constrained optimiser       (Layer 2)
                   ->  time parameterisation              (Layer 3)
                   ->  time_parameterized_configuration_path/   (npz + csv)

The lumen *radius* is set wide (default 50 mm) so wall contact / containment is
not a factor for this experiment -- only the centreline (the desired tip path)
matters.

The shape lives in the planner's bend plane: the plane spanned by the beam's
initial forward direction (``+u``, the natural insertion direction) and the
in-plane bending direction (``+v``).  ``lumen_C[0]`` is pinned to the beam tip
at the planner's initial state ``p0`` so the path starts where the beam already
is.

    # square, all three layers
    python -m proper_research.planning.plan_shape_path --shape square --size-mm 10

    # triangle, skip Layer 2 (time-parameterise the inverse path directly)
    python -m proper_research.planning.plan_shape_path --shape triangle --skip-global

The resulting directory is what
``close_loop_path_follow.py`` consumes with ``reference_source = "plan_dir"``.
NOTE: the planner works in its own world frame; verify the planner->robot-base
alignment (supervised) before executing a planned run on hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# shape centreline
# ---------------------------------------------------------------------------
def _shape_corners(shape: str, size_m: float, closed: bool) -> np.ndarray:
    """2-D corners in the (u, v) bend plane, first corner at the origin.

    The first edge runs along +u (the beam's forward / insertion direction).
    """
    s = float(size_m)
    if shape == "square":
        pts = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    elif shape == "triangle":
        pts = [(0.0, 0.0), (s, 0.0), (0.5 * s, 0.8660254 * s)]
    elif shape == "line":
        pts = [(0.0, 0.0), (s, 0.0)]
        closed = False
    else:
        raise ValueError(f"--shape must be square|triangle|line; got {shape!r}")
    corners = np.asarray(pts, dtype=float)
    if closed and shape != "line":
        corners = np.vstack([corners, corners[:1]])
    return corners


def _resample(corners: np.ndarray, ds: float) -> np.ndarray:
    ds = max(float(ds), 1.0e-5)
    out = [corners[0].copy()]
    for a, b in zip(corners[:-1], corners[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        n = max(1, int(math.ceil(length / ds)))
        for i in range(1, n + 1):
            out.append(a + seg * (i / n))
    return np.asarray(out, dtype=float)


def _smooth_corners(path2d: np.ndarray, window: int) -> np.ndarray:
    """Moving-average the polyline to round the corners.

    A sharp corner is a tangent discontinuity: the beam tip tangent cannot turn
    it, so the inverse planner's tangent residual blows up right there (that is
    the ``tangent_error=88 deg`` failure at the first corner).  Rounding the
    corners a little makes the desired tip path physically trackable while
    leaving the straight edges essentially unchanged.  Endpoints are pinned.
    """
    window = int(window)
    if window < 3 or path2d.shape[0] < window:
        return path2d
    if window % 2 == 0:
        window += 1
    half = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(path2d, ((half, half), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(path2d.shape[1])]
    )
    smoothed[0] = path2d[0]
    smoothed[-1] = path2d[-1]
    return smoothed


def _build_shape_centreline(
    *,
    shape: str,
    size_m: float,
    ds_m: float,
    closed: bool,
    corner_smoothing_mm: float,
    tip0_world: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    corners = _shape_corners(shape, size_m, closed)
    path2d = _resample(corners, ds_m)
    # Round the corners with a fixed, light smoothing pass -- window set by
    # corner_smoothing_mm.  Just enough to kill the tangent discontinuity at a
    # corner without collapsing the shape.
    window = max(3, int(round(corner_smoothing_mm / max(ds_m * 1.0e3, 1.0e-6))))
    for _ in range(2):
        path2d = _smooth_corners(path2d, window)
    tip0 = np.asarray(tip0_world, dtype=float).reshape(3)
    u = np.asarray(u_axis, dtype=float).reshape(3)
    v = np.asarray(v_axis, dtype=float).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return tip0[None, :] + path2d[:, 0:1] * u[None, :] + path2d[:, 1:2] * v[None, :]


# ---------------------------------------------------------------------------
# bend-plane axes (robot base frame R)
# ---------------------------------------------------------------------------
def _bend_axes_R(bundle, controller_pack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tip0, u, v) in the robot base frame R.

    With ``initial_conditions.make_initial_poses`` pointing the planner at the
    hardware beam pose, the planner world frame *is* R.  The beam stands up
    along +R.z; the in-plane bending direction is +R.y.

    ``u`` = +R.z (forward / insertion / axial),
    ``v`` = +R.y (in-plane bending),
    ``tip0`` = the forward model's tip at ``p0`` (so the shape's first node has
    ~zero position error); falls back to the nominal straight tip.
    """
    from proper_research.simulation.simulations.initial_conditions import (
        make_initial_poses,
    )

    pivot_point, _start, L0, _dt = make_initial_poses()
    base_R = np.asarray(pivot_point[:3], dtype=float)
    u = np.array([0.0, 0.0, 1.0])          # +R.z, beam axial
    v = np.array([0.0, 1.0, 0.0])          # +R.y, in-plane bending
    tip0 = base_R + float(L0) * u
    try:
        model_tip = np.asarray(
            controller_pack["plant_diagnostic_joint_adapter"].forward_output(
                np.asarray(controller_pack["p0"], dtype=float)
            ),
            dtype=float,
        )[:3]
        if np.all(np.isfinite(model_tip)) and np.linalg.norm(model_tip - tip0) < 0.02:
            tip0 = model_tip
    except Exception:
        pass
    return tip0, u, v


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", choices=("square", "triangle", "line"), default="square")
    p.add_argument("--size-mm", type=float, default=6.0, help="edge length / height")
    p.add_argument("--point-spacing-mm", type=float, default=0.5)
    p.add_argument("--closed", action="store_true", help="return to the first corner")
    p.add_argument("--corner-smoothing-mm", type=float, default=2.0,
                   help="moving-average window; rounds corners so the tip tangent "
                        "stays trackable (0 = sharp corners, planner will likely fail)")
    p.add_argument("--tangent-tolerance-deg", type=float, default=150.0,
                   help="inverse-planner tangent feasibility bound; high (~150) = "
                        "position-only planning, which is what a wide lumen wants")
    p.add_argument("--position-tolerance-mm", type=float, default=1.0,
                   help="inverse/global tip-position feasibility bound. The beam "
                        "has weak lateral authority with the current magnet, so a "
                        "few-mm shape may need 2-3 mm here to be feasible at all.")
    p.add_argument("--fast", action="store_true",
                   help="fewer multistarts / evaluations in Layer 1 (quicker, "
                        "lower-quality inverse seed)")
    p.add_argument("--lumen-radius-mm", type=float, default=50.0)
    p.add_argument("--magnet-exclusion", dest="magnet_exclusion", action="store_true",
                   default=True,
                   help="hard constraint: the source magnet centre may not come "
                        "closer to the tip path than it is now (default on)")
    p.add_argument("--no-magnet-exclusion", dest="magnet_exclusion",
                   action="store_false")
    p.add_argument("--magnet-exclusion-radius-mm", type=float, default=None,
                   help="override the exclusion radius [mm]; default = the current "
                        "magnet-to-nominal-tip distance")
    p.add_argument("--skip-global", action="store_true",
                   help="time-parameterise the Layer 1 inverse path directly")
    p.add_argument("--dt", type=float, default=0.1, help="reference sample period [s]")
    p.add_argument("--joint-velocity-limit", type=float, default=0.10)
    p.add_argument("--insertion-velocity-limit", type=float, default=2.0e-3)
    p.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    p.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    p.add_argument("--max-path-speed-mm-s", type=float, default=3.0)
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _arguments()

    from proper_research.planning.planning_context import (
        build_planning_context,
        make_inverse_config,
    )
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        InverseConfigurationPlannerConfig,
        solve_from_controller_pack,
    )
    from proper_research.planning.global_constrained_configuration_path import (
        GlobalConfigurationOptimizerConfig,
        optimize_from_saved_inverse_result,
    )
    from proper_research.planning.time_parameterized_configuration_path import (
        TimeParameterizationConfig,
        time_parameterize_saved_global_path,
        time_parameterize_saved_inverse_path,
    )
    from dataclasses import fields

    _, bundle, controller_pack, out_root = build_planning_context()
    out_root = Path(args.output_root or out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tip0, u_axis, v_axis = _bend_axes_R(bundle, controller_pack)
    centreline = _build_shape_centreline(
        shape=args.shape,
        size_m=args.size_mm * 1.0e-3,
        ds_m=args.point_spacing_mm * 1.0e-3,
        closed=args.closed,
        corner_smoothing_mm=args.corner_smoothing_mm,
        tip0_world=tip0,
        u_axis=u_axis,
        v_axis=v_axis,
    )
    perimeter = float(np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))
    print(
        f"[shape] {args.shape} size={args.size_mm:.1f}mm closed={args.closed} "
        f"-> {centreline.shape[0]} points, perimeter {1e3 * perimeter:.1f} mm"
    )
    print(f"[shape] planner tip0 (world) = {np.round(tip0, 4).tolist()}")
    print(f"[shape]   +u (forward)       = {np.round(u_axis, 4).tolist()}")
    print(f"[shape]   +v (in-plane bend) = {np.round(v_axis, 4).tolist()}")
    print(f"[shape] centreline[-1] (world) = {np.round(centreline[-1], 4).tolist()}")

    lumen_R = np.full(centreline.shape[0], args.lumen_radius_mm * 1.0e-3, dtype=float)

    shape_dir = out_root / f"shape_{args.shape}"
    shape_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shape_dir / "shape_centreline.npz",
        lumen_C=centreline,
        lumen_R=lumen_R,
        tip0=tip0,
        u_axis=u_axis,
        v_axis=v_axis,
    )

    # ---- Layer 1: offline inverse configuration ----------------------------
    planner_config = make_inverse_config()
    shared = {
        f.name: getattr(planner_config, f.name)
        for f in fields(InverseConfigurationPlannerConfig)
        if hasattr(planner_config, f.name)
    }
    shared["tangent_tolerance_rad"] = math.radians(
        min(args.tangent_tolerance_deg, 179.0)
    )
    shared["position_tolerance_m"] = args.position_tolerance_mm * 1.0e-3

    if args.magnet_exclusion:
        magnet_R = np.asarray(
            controller_pack["plant_diagnostic_joint_adapter"].magnet_transform(
                np.asarray(controller_pack["p0"])
            ),
            dtype=float,
        )[:3, 3]
        # distance from the magnet to the closest point of the desired tip path
        gaps = np.linalg.norm(centreline - magnet_R[None, :], axis=1)
        exclusion_radius_m = (
            args.magnet_exclusion_radius_mm * 1.0e-3
            if args.magnet_exclusion_radius_mm is not None
            else float(gaps.min())
        )
        shared["source_magnet_lumen_exclusion_radius_m"] = exclusion_radius_m
        print(
            f"[layer1] magnet exclusion ON: source magnet centre must stay "
            f">= {1e3 * exclusion_radius_m:.1f} mm from the tip path "
            f"(magnet now at R={np.round(magnet_R, 3).tolist()}, "
            f"current min gap {1e3 * gaps.min():.1f} mm)"
        )

    if args.fast:
        shared["maximum_function_evaluations"] = 120
        shared["maximum_multistart_attempts"] = 5
    planner_config = InverseConfigurationPlannerConfig(**shared)
    print(
        f"[layer1] tangent tolerance = {args.tangent_tolerance_deg:.0f} deg "
        f"(position-dominant), position tolerance = "
        f"{1e3 * planner_config.position_tolerance_m:.2f} mm"
    )

    inverse_dir = shape_dir / "offline_inverse_configuration_60"
    print(f"\n[layer1] inverse planning -> {inverse_dir}", flush=True)
    t0 = time.perf_counter()
    inverse_result = solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=planner_config,
        output_dir=inverse_dir,
    )
    print(
        f"[layer1] done in {time.perf_counter() - t0:.1f}s  "
        f"all_nodes_feasible={inverse_result.all_nodes_feasible}",
        flush=True,
    )

    tp_config = TimeParameterizationConfig(
        sample_period_s=float(args.dt),
        state_velocity_limit=tuple(
            [args.joint_velocity_limit] * 6 + [args.insertion_velocity_limit]
        ),
        state_acceleration_limit=tuple(
            [args.joint_acceleration_limit] * 6 + [args.insertion_acceleration_limit]
        ),
        maximum_path_speed_m_s=args.max_path_speed_mm_s * 1.0e-3,
        require_nonlinear_beam_feasible=False,
        require_saved_global_feasible=False,
        require_saved_dense_feasible=False,
    )

    time_param_dir = out_root / "time_parameterized_configuration_path"

    if args.skip_global:
        if not inverse_result.all_nodes_feasible:
            raise SystemExit(
                "[layer1] did not reach a fully node-feasible path, and "
                "time-parameterising the inverse path directly requires one. "
                "Re-run without --skip-global so Layer 2 (recover_partial) can "
                "repair it, or increase --corner-smoothing-mm / --size-mm."
            )
        print(f"\n[layer3] time-parameterising the INVERSE path -> {time_param_dir}", flush=True)
        time_parameterize_saved_inverse_path(
            inverse_output_dir=inverse_dir,
            controller_pack=controller_pack,
            config=tp_config,
            output_dir=time_param_dir,
            lumen_C=centreline,
        )
    else:
        # ---- Layer 2: global constrained optimiser ------------------------
        global_dir = shape_dir / "global_configuration_converged"
        global_config = GlobalConfigurationOptimizerConfig(
            mode="refine_complete" if inverse_result.all_nodes_feasible else "recover_partial",
            position_tolerance_m=float(planner_config.position_tolerance_m),
            tangent_tolerance_rad=float(planner_config.tangent_tolerance_rad),
            path_step_m=3.0e-3,
            minimum_path_step_m=1.5e-3,
            maximum_refinement_rounds=1,
            maximum_nodes=200,
            maximum_iterations=40,
            maximum_wall_time_s=180.0,
            stagnation_function_evaluations=4000,
            dense_validation_enabled=True,
            compute_node_jacobian_diagnostics=False,
            require_contact_model=True,
            insertion_non_decreasing=False,
            fix_initial_state=True,
        )
        print(f"\n[layer2] global optimiser -> {global_dir}", flush=True)
        t0 = time.perf_counter()
        global_result = optimize_from_saved_inverse_result(
            inverse_output_dir=inverse_dir,
            controller_pack=controller_pack,
            lumen_C=centreline,
            config=global_config,
            output_dir=global_dir,
        )
        print(
            f"[layer2] done in {time.perf_counter() - t0:.1f}s  "
            f"nodes={len(getattr(global_result, 'nodes', []))}",
            flush=True,
        )

        print(f"\n[layer3] time-parameterising the GLOBAL path -> {time_param_dir}", flush=True)
        time_parameterize_saved_global_path(
            global_output_dir=global_dir,
            controller_pack=controller_pack,
            config=tp_config,
            output_dir=time_param_dir,
            lumen_C=centreline,
        )

    npz = time_param_dir / "time_parameterized_configuration_path.npz"
    data = np.load(npz)
    n = int(np.asarray(data["time_s"]).size)
    dur = float(np.asarray(data["time_s"])[-1])
    print(
        f"\n[done] {npz}\n"
        f"       {n} samples, {dur:.1f} s at dt={args.dt}s\n"
        f"       desired_position_m[0]  = {np.round(np.asarray(data['desired_position_m'])[0], 4).tolist()}\n"
        f"       desired_position_m[-1] = {np.round(np.asarray(data['desired_position_m'])[-1], 4).tolist()}"
    )
    summary = {
        "shape": args.shape,
        "size_mm": args.size_mm,
        "closed": args.closed,
        "lumen_radius_mm": args.lumen_radius_mm,
        "skipped_global": args.skip_global,
        "planner_tip0_world": tip0.tolist(),
        "u_axis_world": u_axis.tolist(),
        "v_axis_world": v_axis.tolist(),
        "centreline_points": int(centreline.shape[0]),
        "perimeter_mm": 1e3 * perimeter,
        "time_param_samples": n,
        "time_param_duration_s": dur,
        "time_param_dir": str(time_param_dir),
    }
    (shape_dir / "plan_shape_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] run online with:\n"
          f"  edit close_loop_path_follow.CONFIG: reference_source='plan_dir', "
          f"plan_dir='{time_param_dir}', control_hz={1.0 / args.dt:.1f}\n"
          f"  python -m proper_research.hardware.online.close_loop_path_follow")


if __name__ == "__main__":
    main()
