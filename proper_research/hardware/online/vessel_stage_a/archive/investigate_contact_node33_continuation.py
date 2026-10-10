"""Investigate whether the contact-aware inverse planner's stall near the
vessel bend (phi30_L30_left1mm_design.json, node 33, s=0.00825m) is caused
by per-node/local continuation bounds being too tight to reach a feasible
contact configuration that DOES exist -- vs. no feasible configuration
existing at all near that target.

Method
------
1. Replay nodes 0..32 of the EXACT same planner run (same adapter, same
   config) by intercepting `_solve_one_node` right as it is about to be
   called for node_index=33 -- this reproduces `previous_state` (node 32's
   accepted [q1..q6, L]) deterministically and fast, without ever calling
   the real (hanging) solve for node 33.
2. At that captured node-33 target, run a hard-bounded (subprocess,
   OS-level kill) baseline solve with the NORMAL local continuation bounds
   (maximum_joint_step_rad=0.075rad/joint, maximum_insertion_step_m=0.75mm)
   to confirm/measure the stall.
3. Run the same target with the SAME global hard bounds (state_min/
   state_max, exclusion constraints) but local continuation bounds relaxed
   to effectively the full global range -- i.e. "does *any* feasible
   configuration exist here if continuity is ignored".
4. If (3) finds a feasible point, report Δq1..Δq6, ΔL and the magnet-pose
   change vs. node 32, and compare each against the normal per-node caps.
5. Binary-search a scalar multiplier k on the normal local bounds to find
   the minimum relaxation that still succeeds reliably (single node, not
   the whole path).

Usage: python investigate_contact_node33_continuation.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

DESIGN_JSON = Path("plans/stage3_design/phi30_L30_left1mm_design.json")
TARGET_NODE_INDEX = 33
PER_ATTEMPT_TIMEOUT_S = 150.0


def _build_everything():
    """Build controller_pack/adapter + planner_config exactly as
    build_vessel_plan.py did for the stalled phi30_L30_left1mm contact run."""
    from proper_research.planning.vessel_context import build_vessel_planning_context
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        InverseConfigurationPlannerConfig,
    )
    from proper_research.planning.planning_context import make_inverse_config
    from proper_research.rig_calibration import beam_base_pose6, CURRENT_LUMEN_FILE

    design = json.loads(DESIGN_JSON.read_text())
    start_point = np.asarray(design["magnet_pose6_R"], dtype=float)
    exclusion_floor_m = float(design["exclusion_floor_mm"]) / 1000.0
    target_path = np.asarray(design["tracked_path_R"], dtype=float)

    BEAM_BASE_PIVOT = beam_base_pose6()
    insertion_start_m = 0.030
    DT_INIT = 0.01
    initial_poses = (BEAM_BASE_PIVOT.copy(), start_point.copy(), insertion_start_m, DT_INIT)

    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11,
            jacobian_mode="accurate", initial_poses=initial_poses, plant_contact=True,
        )
    )

    planner_config = make_inverse_config()
    from dataclasses import fields
    shared = {
        f.name: getattr(planner_config, f.name)
        for f in fields(InverseConfigurationPlannerConfig)
        if hasattr(planner_config, f.name)
    }
    shared["tangent_tolerance_rad"] = np.deg2rad(179.95)
    shared["position_tolerance_m"] = 0.008
    shared["solve_initial_node"] = True
    shared["source_magnet_lumen_exclusion_radius_m"] = None
    shared["magnet_beam_base_exclusion_radius_m"] = exclusion_floor_m
    shared["maximum_function_evaluations"] = 15
    shared["maximum_multistart_attempts"] = 2
    shared["finite_difference_joint_step_rad"] = 1.5e-2
    shared["finite_difference_insertion_step_m"] = 5.0e-3
    shared["maximum_chain_rule_relative_error"] = 0.6
    shared["node_timeout_s"] = 90.0
    planner_config = InverseConfigurationPlannerConfig(**shared)

    return controller_pack, target_path, planner_config


def _capture_node33_inputs():
    """Replay nodes 0..32 for real, stop just before node 33's real solve,
    and return everything `_solve_one_node` was about to be called with."""
    import proper_research.planning.offline_inverse_configuration_head_exclusion as oic

    controller_pack, target_path, planner_config = _build_everything()

    real_solve_one_node = oic._solve_one_node
    captured: dict = {}

    class _StopAtNode(Exception):
        pass

    def intercepting(*, node_index, s_m, desired_position, desired_tangent,
                      previous_state, extrapolated_state, jacobian_predicted_state,
                      alternative_initial_states, adapter, state_min, state_max,
                      config, exclusion_constraint):
        if node_index == TARGET_NODE_INDEX:
            captured["node_index"] = node_index
            captured["s_m"] = s_m
            captured["desired_position"] = np.array(desired_position)
            captured["desired_tangent"] = np.array(desired_tangent)
            captured["previous_state"] = np.array(previous_state)
            captured["extrapolated_state"] = (
                None if extrapolated_state is None else np.array(extrapolated_state)
            )
            captured["jacobian_predicted_state"] = (
                None if jacobian_predicted_state is None else np.array(jacobian_predicted_state)
            )
            captured["alternative_initial_states"] = [
                np.array(s) for s in alternative_initial_states
            ]
            captured["state_min"] = np.array(state_min)
            captured["state_max"] = np.array(state_max)
            raise _StopAtNode()
        return real_solve_one_node(
            node_index=node_index, s_m=s_m, desired_position=desired_position,
            desired_tangent=desired_tangent, previous_state=previous_state,
            extrapolated_state=extrapolated_state,
            jacobian_predicted_state=jacobian_predicted_state,
            alternative_initial_states=alternative_initial_states, adapter=adapter,
            state_min=state_min, state_max=state_max, config=config,
            exclusion_constraint=exclusion_constraint,
        )

    oic._solve_one_node = intercepting
    t0 = time.perf_counter()
    try:
        oic.solve_from_controller_pack(
            controller_pack=controller_pack, lumen_C=target_path, config=planner_config,
            output_dir=None,
            alternative_initial_states=[np.asarray(controller_pack["p0"], dtype=float)],
        )
        raise RuntimeError(
            f"Replay never reached node {TARGET_NODE_INDEX} -- the stalled run's "
            "own node count must have changed; re-check TARGET_NODE_INDEX."
        )
    except _StopAtNode:
        pass
    finally:
        oic._solve_one_node = real_solve_one_node
    print(f"[replay] reproduced nodes 0..{TARGET_NODE_INDEX - 1} in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    return controller_pack, planner_config, captured


def _solve_node_subprocess(result_queue, *, previous_state, extrapolated_state,
                            jacobian_predicted_state, alternative_initial_states,
                            s_m, desired_position, desired_tangent, state_min, state_max,
                            config_overrides, label):
    """Runs inside a forked child process so a hard timeout (process kill)
    can bound a genuine native-code hang, which signal.alarm cannot."""
    import proper_research.planning.offline_inverse_configuration_head_exclusion as oic

    controller_pack, target_path, planner_config = _build_everything()
    if config_overrides:
        planner_config = replace(planner_config, **config_overrides)

    adapter = controller_pack["plant_diagnostic_joint_adapter"]

    # Rebuild the SAME exclusion_constraint the real run used (magnet-z /
    # base-exclusion, no lumen-exclusion -- matches shared["source_magnet_lumen_exclusion_radius_m"]=None).
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        _MagnetPositionProvider, _MagnetZFloorConstraint, _MagnetBaseExclusionConstraint,
        _CompositeExclusionConstraint,
    )
    state0 = np.asarray(controller_pack["p0"], dtype=float)
    position_provider = _MagnetPositionProvider(
        adapter=adapter, state_min=state_min, state_max=state_max, config=planner_config,
    )
    z_floor_constraint = None
    if planner_config.enforce_magnet_z_no_decrease or planner_config.enforce_magnet_z_fixed:
        z0_m = float(position_provider.position(state0)[2])
        z_floor_constraint = _MagnetZFloorConstraint(provider=position_provider, z_floor_m=z0_m)
    base_exclusion_constraint = None
    if planner_config.magnet_beam_base_exclusion_radius_m is not None:
        from proper_research.simulation.simulations.initial_conditions import make_initial_poses
        beam_pivot, _s, _L0, _dt = make_initial_poses()
        base_exclusion_constraint = _MagnetBaseExclusionConstraint(
            provider=position_provider, base_m=np.asarray(beam_pivot[:3], dtype=float),
            radius_m=float(planner_config.magnet_beam_base_exclusion_radius_m),
        )
    exclusion_constraint = _CompositeExclusionConstraint(
        lumen=None, z_floor=z_floor_constraint, z_ceiling=None, base=base_exclusion_constraint,
    )

    t0 = time.perf_counter()
    node = oic._solve_one_node(
        node_index=TARGET_NODE_INDEX, s_m=s_m, desired_position=desired_position,
        desired_tangent=desired_tangent, previous_state=previous_state,
        extrapolated_state=extrapolated_state, jacobian_predicted_state=jacobian_predicted_state,
        alternative_initial_states=alternative_initial_states, adapter=adapter,
        state_min=state_min, state_max=state_max, config=planner_config,
        exclusion_constraint=exclusion_constraint,
    )
    elapsed = time.perf_counter() - t0
    result_queue.put({
        "label": label, "elapsed_s": elapsed, "feasible": node.feasible,
        "q_rad": node.q_rad.tolist(), "insertion_m": node.insertion_m,
        "magnet_pose6": node.magnet_pose6.tolist(),
        "position_error_m": node.position_error_m, "tangent_error_rad": node.tangent_error_rad,
        "contact_active": node.contact_active, "termination_reason": node.termination_reason,
        "within_bounds": bool(np.all(node.q_rad >= state_min[:6] - 1e-9)),
    })


def run_bounded(*, label, config_overrides, captured, timeout_s):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=_solve_node_subprocess,
        kwargs=dict(
            result_queue=q, previous_state=captured["previous_state"],
            extrapolated_state=captured["extrapolated_state"],
            jacobian_predicted_state=captured["jacobian_predicted_state"],
            alternative_initial_states=captured["alternative_initial_states"],
            s_m=captured["s_m"], desired_position=captured["desired_position"],
            desired_tangent=captured["desired_tangent"], state_min=captured["state_min"],
            state_max=captured["state_max"], config_overrides=config_overrides, label=label,
        ),
    )
    t0 = time.perf_counter()
    p.start()
    p.join(timeout_s)
    wall = time.perf_counter() - t0
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        print(f"[{label}] TIMED OUT / HUNG after {wall:.1f}s (hard-killed)", flush=True)
        return {"label": label, "hung": True, "wall_s": wall}
    try:
        result = q.get_nowait()
        result["hung"] = False
        result["wall_s"] = wall
        return result
    except Exception:
        print(f"[{label}] process exited without a result after {wall:.1f}s "
              f"(exitcode={p.exitcode})", flush=True)
        return {"label": label, "hung": True, "wall_s": wall, "exitcode": p.exitcode}


def main():
    controller_pack, planner_config, captured = _capture_node33_inputs()
    print(f"\n[node33] target s={captured['s_m']:.6f} m")
    print(f"[node33] desired_position = {captured['desired_position']}")
    print(f"[node32] previous_state (q1..q6, L) = {captured['previous_state']}")

    NORMAL_JOINT_STEP = tuple(planner_config.maximum_joint_step_rad)
    NORMAL_INS_STEP = float(planner_config.maximum_insertion_step_m)
    print(f"[config] normal maximum_joint_step_rad = {NORMAL_JOINT_STEP}")
    print(f"[config] normal maximum_insertion_step_m = {NORMAL_INS_STEP}")

    results = {}

    # --- Step 2: baseline, normal local bounds, hard-bounded ---
    print("\n=== STEP 2: baseline (normal local continuation bounds) ===", flush=True)
    results["baseline"] = run_bounded(
        label="baseline", config_overrides={}, captured=captured,
        timeout_s=PER_ATTEMPT_TIMEOUT_S,
    )
    print(json.dumps({k: v for k, v in results["baseline"].items() if k != "q_rad"}, indent=2))

    # --- Step 3: fully relaxed local bounds (global hard bounds only) ---
    print("\n=== STEP 3: fully relaxed local bounds (global constraints only) ===", flush=True)
    results["relaxed_full"] = run_bounded(
        label="relaxed_full",
        config_overrides={
            "maximum_joint_step_rad": (10.0,) * 6,
            "maximum_insertion_step_m": 10.0,
        },
        captured=captured, timeout_s=PER_ATTEMPT_TIMEOUT_S,
    )
    print(json.dumps({k: v for k, v in results["relaxed_full"].items() if k != "q_rad"}, indent=2))

    if results["relaxed_full"].get("feasible"):
        q33 = np.array(results["relaxed_full"]["q_rad"])
        L33 = results["relaxed_full"]["insertion_m"]
        q32 = captured["previous_state"][:6]
        L32 = captured["previous_state"][6]
        dq = q33 - q32
        dL = L33 - L32
        print("\n=== STEP 4-5: delta vs node 32, compared to normal per-node caps ===")
        for i in range(6):
            flag = "EXCEEDS cap" if abs(dq[i]) > NORMAL_JOINT_STEP[i] else "within cap"
            print(f"  dq{i+1} = {dq[i]: .5f} rad  (cap={NORMAL_JOINT_STEP[i]:.5f})  [{flag}]")
        flag = "EXCEEDS cap" if abs(dL) > NORMAL_INS_STEP else "within cap"
        print(f"  dL   = {dL*1e3: .4f} mm   (cap={NORMAL_INS_STEP*1e3:.4f} mm)  [{flag}]")
        mp6_32 = captured.get("previous_magnet_pose6")
        print(f"  magnet_pose6 @ node33(relaxed) = {results['relaxed_full']['magnet_pose6']}")

        # --- Step 6: binary search minimum relaxation factor ---
        print("\n=== STEP 6: binary search minimum relaxation multiplier k ===", flush=True)
        k_fail = 1.0   # normal bounds -> baseline (assume stalls/fails)
        k_succeed = 10.0 / max(max(NORMAL_JOINT_STEP), NORMAL_INS_STEP)  # the "fully relaxed" k
        for _ in range(5):
            k_mid = 0.5 * (k_fail + k_succeed)
            label = f"relax_k{k_mid:.2f}"
            print(f"--- trying k={k_mid:.3f} ---", flush=True)
            res = run_bounded(
                label=label,
                config_overrides={
                    "maximum_joint_step_rad": tuple(s * k_mid for s in NORMAL_JOINT_STEP),
                    "maximum_insertion_step_m": NORMAL_INS_STEP * k_mid,
                },
                captured=captured, timeout_s=PER_ATTEMPT_TIMEOUT_S,
            )
            print(json.dumps({k: v for k, v in res.items() if k != "q_rad"}, indent=2))
            results[label] = res
            if res.get("feasible") and not res.get("hung"):
                k_succeed = k_mid
            else:
                k_fail = k_mid
        print(f"\n[bisection] minimum working multiplier k ~= {k_succeed:.3f} "
              f"(normal bounds = k=1.0)")

    out_path = Path("plans/stage3_design/node33_continuation_investigation.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: vv for kk, vv in v.items()}
    out_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nsaved full results to {out_path}")


if __name__ == "__main__":
    main()
