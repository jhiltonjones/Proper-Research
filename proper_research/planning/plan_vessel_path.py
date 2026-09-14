#!/usr/bin/env python3
"""Plan a tip path through a REAL, digitized vessel lumen, with the contact
model actually wired to the true wall geometry.

    vessel lumen (real digitized C, R)
        -> contact-aware model bundle (wall = real geometry, not a
           permissive placeholder -- see vessel_context.py)
        -> offline inverse configuration      (Layer 1)
        -> global constrained optimiser       (Layer 2)
        -> time parameterisation              (Layer 3)
        -> time_parameterized_configuration_path/   (npz + csv)

Unlike ``plan_shape_path.py`` (an analytic shape in the planner's bend plane,
with a permissive 50mm placeholder lumen radius so wall contact is never a
factor), this script:

  1. Loads a real digitized vessel lumen from ``--lumen-file`` (produced by
     ``proper_research.vision.detect_blue::draw_vessel_lumen_for_planner``),
     already in the robot frame R.
  2. Wires that REAL geometry into the contact-aware model's ``lumen_query``
     (``vessel_context.build_vessel_planning_context``), so the beam's
     equilibrium is solved against the true wall, not a fictional wide one.
  3. Uses the SAME real centreline as the Layer 1/2/3 target path (the
     ``lumen_C`` argument every one of those functions already expects) --
     the vessel's own centreline is what "the shape" is, there is no
     separate analytic generator.

The real per-point radius is saved alongside the centreline for reference,
but --like plan_shape_path.py's lumen_R-- is NOT itself consumed by Layer
1/2/3 (``CentrelinePath`` only uses the centreline positions/tangents); the
radius's only live physical effect is via the contact-aware model's
``lumen_query`` wired in step 2.

    python -m proper_research.planning.plan_vessel_path \\
        --lumen-file vessel_lumen_robot_frame.json --skip-global

NOTE: the planner works in its own world frame; verify the planner->robot-base
alignment (supervised) before executing a planned run on hardware, and run
the offline Z-no-decrease / magnet-base-exclusion safety check on the
resulting plan before any live test, exactly as for every other shape.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", type=str, default="vessel_lumen_robot_frame.json",
                    help="output of detect_blue.py::draw_vessel_lumen_for_planner "
                         "(robot-frame lumen_C_m / lumen_R_m + provenance).")
    p.add_argument("--insertion-max-mm", type=float, default=None,
                    help="override the robot config's insertion ceiling (library "
                         "default 50mm). Confirmed this vessel's true end needs "
                         "~64.5mm via insertion-synchronised tracking; only raise "
                         "this if the advancer/beam hardware genuinely supports it.")
    p.add_argument("--tangent-tolerance-deg", type=float, default=179.9,
                    help="inverse-planner tangent feasibility GATE. A vessel path "
                         "is open (not closed) so this rarely binds, unlike a "
                         "closed shape's antiparallel return edges.")
    p.add_argument("--position-tolerance-mm", type=float, default=1.0,
                    help="inverse/global tip-position feasibility bound.")
    p.add_argument("--tangent-error-selection-weight", type=float, default=1.0)
    p.add_argument("--fast", action="store_true",
                    help="fewer multistarts / evaluations in Layer 1")
    p.add_argument("--max-chain-rule-error", type=float, default=0.09)
    p.add_argument("--magnet-exclusion", dest="magnet_exclusion", action="store_true",
                    default=True,
                    help="hard constraint: the source magnet centre may not come "
                         "closer to the tip path than it is now (default on)")
    p.add_argument("--no-magnet-exclusion", dest="magnet_exclusion", action="store_false")
    p.add_argument("--magnet-exclusion-radius-mm", type=float, default=None)
    p.add_argument("--magnet-exclusion-reference-joints", type=str, default=None,
                    help="use --flag=value syntax (a leading '-' in the value "
                         "confuses argparse otherwise).")
    p.add_argument("--skip-global", action="store_true",
                    help="time-parameterise the Layer 1 inverse path directly")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--joint-velocity-limit", type=float, default=0.10)
    p.add_argument("--insertion-velocity-limit", type=float, default=2.0e-3)
    p.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    p.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    p.add_argument("--max-path-speed-mm-s", type=float, default=3.0)
    p.add_argument("--time-constraint-tolerance", type=float, default=1.0e-8)
    p.add_argument("--velocity-safety-factor", type=float, default=0.8)
    p.add_argument("--acceleration-safety-factor", type=float, default=0.8)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--maximum-global-nodes", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _arguments()

    from proper_research.planning.vessel_context import build_vessel_planning_context
    from proper_research.planning.planning_context import make_inverse_config
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

    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=args.lumen_file,
            insertion_max_m=(
                args.insertion_max_mm * 1.0e-3 if args.insertion_max_mm is not None else None
            ),
        )
    )
    out_root = Path(args.output_root or out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    perimeter = float(np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))
    print(
        f"[vessel] {args.lumen_file} -> {centreline.shape[0]} points, "
        f"arclength {1e3 * perimeter:.1f} mm, "
        f"radius {1e3 * lumen_R.min():.2f}-{1e3 * lumen_R.max():.2f} mm"
    )
    print(f"[vessel] centreline[0]  (world/R) = {np.round(centreline[0], 4).tolist()}")
    print(f"[vessel] centreline[-1] (world/R) = {np.round(centreline[-1], 4).tolist()}")

    vessel_dir = out_root / "vessel_lumen"
    vessel_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        vessel_dir / "vessel_centreline.npz",
        lumen_C=centreline,
        lumen_R=lumen_R,
    )
    (vessel_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # ---- Layer 1: offline inverse configuration ----------------------------
    planner_config = make_inverse_config()
    shared = {
        f.name: getattr(planner_config, f.name)
        for f in fields(InverseConfigurationPlannerConfig)
        if hasattr(planner_config, f.name)
    }
    shared["tangent_tolerance_rad"] = math.radians(min(args.tangent_tolerance_deg, 179.95))
    shared["position_tolerance_m"] = args.position_tolerance_mm * 1.0e-3
    # The beam's resting tip is 23.75mm from this vessel's literal digitized
    # entrance (centreline[0]) -- confirmed too far for an instant lateral
    # correction at near-zero insertion (every attempt to force node 0 to
    # centreline[0], with or without magnet exclusion, with or without a
    # seeded multistart, converged no closer than ~22mm). That's the SAME
    # near-zero-insertion authority limit the "tangent-safe-start" free-space
    # shapes were built to avoid, not a contact-model or exclusion bug
    # (contact_active was False throughout).
    #
    # Fix: let the plan build up gradually instead of forcing the literal
    # entrance. start_at_initial_tip_projection=True (the library default)
    # finds the CLOSEST reachable point on the vessel's own centreline to the
    # beam's resting tip (s=23.4mm here, only 4.4mm away -- a small,
    # achievable correction) and starts the tracked/optimized plan there,
    # implicitly treating the vessel's first ~23mm as plain axial insertion
    # before the beam engages the tracked centreline. solve_initial_node=True
    # still lets that now-small 4.4mm gap be optimized down toward the
    # position tolerance rather than only measured-and-rejected.
    if "solve_initial_node" in shared:
        shared["solve_initial_node"] = True
    if "tangent_error_selection_weight" in shared:
        shared["tangent_error_selection_weight"] = float(args.tangent_error_selection_weight)
    if "maximum_chain_rule_relative_error" in shared:
        shared["maximum_chain_rule_relative_error"] = float(args.max_chain_rule_error)

    if args.magnet_exclusion:
        magnet_R = np.asarray(
            controller_pack["plant_diagnostic_joint_adapter"].magnet_transform(
                np.asarray(controller_pack["p0"])
            ),
            dtype=float,
        )[:3, 3]
        gaps = np.linalg.norm(centreline - magnet_R[None, :], axis=1)
        if args.magnet_exclusion_reference_joints:
            ref_q = np.array(
                [float(x) for x in args.magnet_exclusion_reference_joints.split(",")],
                dtype=float,
            )
            from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
            from proper_research.hardware import (
                robotics_frame_measurement_validation as rfmv,
            )
            from proper_research.simulation.simulations.initial_conditions import (
                make_initial_poses,
            )
            from scipy.spatial.transform import Rotation as _Rot

            beam_base = np.asarray(make_initial_poses()[0][:3], dtype=float)
            dh = urik.corrected_dh_from_config(urik.CONFIG)
            fk = urik.forward_kinematics(ref_q, dh)
            T_ref = fk.T_R_target if fk.T_R_target is not None else fk.T_R_F
            tcp6_ref = np.r_[T_ref[:3, 3], _Rot.from_matrix(T_ref[:3, :3]).as_rotvec()]
            _, _, T_R_M_ref = rfmv.source_transform_from_tcp(tcp6_ref, rfmv.CONFIG)
            magnet_ref = np.asarray(T_R_M_ref.translation, dtype=float)
            base_radius_m = float(np.linalg.norm(magnet_ref - beam_base))
            gaps_ref = np.linalg.norm(centreline - magnet_ref[None, :], axis=1)
            exclusion_radius_m = float(gaps_ref.min())
            print(
                f"[layer1] magnet exclusion radius from --magnet-exclusion-reference-joints: "
                f"magnet@{np.round(magnet_ref, 3).tolist()}  beam_base dist {1e3 * base_radius_m:.1f} mm  "
                f"-> tip-path dist (used) {1e3 * exclusion_radius_m:.1f} mm"
            )
        else:
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

    inverse_dir = vessel_dir / "offline_inverse_configuration"
    print(f"\n[layer1] inverse planning -> {inverse_dir}", flush=True)
    t0 = time.perf_counter()
    # The beam's actual resting state (p0) is only ~4mm from the vessel's real
    # entrance -- seed the node-0 multistart with it directly, or an
    # unseeded search can wander into a much worse local optimum for what
    # should be a small, easy correction (observed: 22mm/26deg error with no
    # seed, vs the natural state's own unoptimized 4.4mm/0.8deg residual).
    inverse_result = solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=planner_config,
        output_dir=inverse_dir,
        alternative_initial_states=[np.asarray(controller_pack["p0"], dtype=float)],
    )
    print(
        f"[layer1] done in {time.perf_counter() - t0:.1f}s  "
        f"all_nodes_feasible={inverse_result.all_nodes_feasible}",
        flush=True,
    )

    tp_config = TimeParameterizationConfig(
        sample_period_s=float(args.dt),
        state_velocity_limit=tuple([args.joint_velocity_limit] * 6 + [args.insertion_velocity_limit]),
        state_acceleration_limit=tuple([args.joint_acceleration_limit] * 6 + [args.insertion_acceleration_limit]),
        maximum_path_speed_m_s=args.max_path_speed_mm_s * 1.0e-3,
        constraint_tolerance=float(args.time_constraint_tolerance),
        velocity_safety_factor=float(args.velocity_safety_factor),
        acceleration_safety_factor=float(args.acceleration_safety_factor),
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
                "repair it, or increase --position-tolerance-mm."
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
        global_dir = vessel_dir / "global_configuration_converged"
        global_config = GlobalConfigurationOptimizerConfig(
            mode="refine_complete" if inverse_result.all_nodes_feasible else "recover_partial",
            position_tolerance_m=float(planner_config.position_tolerance_m),
            tangent_tolerance_rad=float(planner_config.tangent_tolerance_rad),
            path_step_m=3.0e-3,
            minimum_path_step_m=1.5e-3,
            maximum_refinement_rounds=1,
            maximum_nodes=int(args.maximum_global_nodes),
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


if __name__ == "__main__":
    main()
