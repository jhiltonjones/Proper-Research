#!/usr/bin/env python3
"""Closed-loop beam-tip tracking driven by the SIMULATED FLOW RING, not vision.

Ties together:

  * the existing building context -- ``planning_context.build_planning_context``
    (the manually-set single-30-degree-bend lumen, the contact beam plant, the
    controller pack);
  * the already-computed L1 -> L2 -> L3 reference in
    ``flow_path_planning/bends_p30_p0_.../time_parameterized_configuration_path``;
  * the DARE / Riccati inverse-Jacobian controller (``riccati_inverse_jacobian``);
  * the simulated flow-ring feedback (``flow_feedback``).

Every control step:

    true_tip           = plant beam snapshot at the measured joint state
    p_meas             = flow ring estimate  (desired point + estimated offset)
    command            = controller.solve(measured_state, p_meas, ...)
    state             += dt * command
    plant is committed

The controller never receives the true tip in the ``flow`` run -- only the
flow-reconstructed estimate.  Baselines (perfect ``vision`` feedback, ``no_dare``
resolved-rate with flow, ``feedforward`` open loop) are run through the *same*
loop and plant for comparison.

Each run writes ``configuration_mpc_simulation.csv`` in the existing schema, so
``analysis/analyse_layer4_mpc.py`` works on it unchanged, plus extra ``flow_*``
diagnostic columns.

Usage
-----
    python -m proper_research.flow.simulate_flow_closed_loop --self-test
    python -m proper_research.flow.simulate_flow_closed_loop \
        --reference flow_path_planning/bends_p30_p0_.../time_parameterized_configuration_path \
        --max-steps 400 --sigma-over-vmean 0.02
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from . import flow_observability_reference as O  # noqa: F401
    from .flow_feedback import FlowFeedbackConfig, FlowTipEstimator
    from .riccati_inverse_jacobian import (
        RiccatiConfig, RiccatiInverseJacobianController,
        build_riccati_inverse_jacobian_controller,
    )
except ImportError:  # pragma: no cover
    from flow_feedback import FlowFeedbackConfig, FlowTipEstimator  # type: ignore
    from riccati_inverse_jacobian import (  # type: ignore
        RiccatiConfig, RiccatiInverseJacobianController,
        build_riccati_inverse_jacobian_controller,
    )

Array = np.ndarray
DEFAULT_REFERENCE = Path(
    "flow_path_planning/bends_p30_p0_jac_contact_plant_contact_1_ctrl_mpc_sqp_full_"
    "rollout10_Np15_sqp50_bends_30_0/time_parameterized_configuration_path"
)


# --------------------------------------------------------------------------
# closed loop
# --------------------------------------------------------------------------
def run_flow_closed_loop(
    *,
    controller: Any,
    reference: Any,
    beam_snapshot: Callable[[Array, bool], tuple[Array, Array]],
    magnet_pose: Callable[[Array], tuple[Array, Array]],
    flow_estimator: FlowTipEstimator | None,
    record_fn: Callable[..., dict],
    loop_config: Any,
    initial_state: Array,
    reset_plant: Callable[[], None] | None = None,
    feedback: str = "flow",          # "flow" | "vision" | "none"
    verbose: bool = True,
) -> list[dict]:
    """One simulation loop.  ``feedback`` selects what the controller is told."""
    if reset_plant is not None:
        reset_plant()
    controller.reset()
    if flow_estimator is not None:
        flow_estimator.reset()

    state = np.asarray(initial_state, dtype=float).reshape(7).copy()
    previous = np.zeros(7, dtype=float)
    dt = float(loop_config.sample_period_s)

    true_tip, true_tangent = beam_snapshot(state, True)
    magnet_position, magnet_quaternion = magnet_pose(state)
    records = [record_fn(
        sample=0, reference_index=0, reference=reference, state=state,
        command=previous, previous_command=previous, output=None,
        achieved_position=true_tip, achieved_tangent=true_tangent,
        magnet_position=magnet_position, magnet_quaternion=magnet_quaternion,
        config=loop_config,
    )]
    flow_rows: list[dict] = [{}]

    steps = reference.sample_count - 1 + int(loop_config.terminal_hold_steps)
    if loop_config.maximum_control_steps is not None:
        steps = min(steps, int(loop_config.maximum_control_steps))

    for k in range(steps):
        ref_idx = min(k + 1, reference.sample_count - 1)
        desired = np.asarray(reference.desired_position_m, dtype=float)[ref_idx]

        flow_row: dict = {}
        if feedback == "flow":
            p_meas, sample = flow_estimator.estimate(true_tip, desired, k * dt)
            flow_row = sample.as_row()
        elif feedback == "vision":
            p_meas = true_tip
        else:  # open loop / feedforward only
            p_meas = desired

        step = controller.solve(
            measured_state=state, measured_beam_position=p_meas,
            control_index=k, previous_input=previous,
        )
        command = np.asarray(step.command, dtype=float).reshape(7)
        state = state + dt * command
        old_previous = previous.copy()
        previous = command.copy()

        true_tip, true_tangent = beam_snapshot(state, True)
        magnet_position, magnet_quaternion = magnet_pose(state)
        row = record_fn(
            sample=k + 1, reference_index=ref_idx, reference=reference, state=state,
            command=command, previous_command=old_previous, output=step,
            achieved_position=true_tip, achieved_tangent=true_tangent,
            magnet_position=magnet_position, magnet_quaternion=magnet_quaternion,
            config=loop_config,
        )
        row.update(flow_row)
        records.append(row)
        flow_rows.append(flow_row)

        if verbose and (k == 0 or (k + 1) % 50 == 0 or k + 1 == steps):
            spectral = getattr(getattr(controller, "diagnostics", None),
                               "spectral_radius", float("nan"))
            print(f"[{feedback:8s}] step {k + 1}/{steps} "
                  f"tip_err={row['beam_position_error_mm']:.4f} mm "
                  f"rho={spectral:.4f}"
                  + (f" flow_e_hat={1e3*flow_row.get('flow_e_hat_norm_m', 0):.4f} mm"
                     if flow_row else ""),
                  flush=True)
    return records


def summarise(records: list[dict], name: str, description: str,
              loop_config: Any = None) -> dict:
    err = np.array([r["beam_position_error_m"] for r in records], dtype=float)
    tang = np.array([r["beam_tangent_error_rad"] for r in records], dtype=float)
    feasible = np.array([r["beam_feasible"] for r in records], dtype=float)
    flow_err = np.array([r.get("flow_estimate_error_m", np.nan) for r in records],
                        dtype=float)
    held = np.array([r.get("flow_low_gain_held", 0) for r in records], dtype=float)
    out = {
        "controller": name,
        "description": description,
        "samples": len(records),
        "max_beam_position_error_mm": float(1e3 * np.nanmax(err)),
        "rms_beam_position_error_mm": float(1e3 * np.sqrt(np.nanmean(err ** 2))),
        "p95_beam_position_error_mm": float(1e3 * np.nanpercentile(err, 95)),
        "final_beam_position_error_mm": float(1e3 * err[-1]),
        "max_beam_tangent_error_deg": float(math.degrees(np.nanmax(tang))),
        "beam_feasible_fraction": float(np.mean(feasible)),
    }
    if np.any(np.isfinite(flow_err)):
        out["flow_estimate_error_rms_mm"] = float(
            1e3 * np.sqrt(np.nanmean(flow_err ** 2))
        )
        out["flow_estimate_error_max_mm"] = float(1e3 * np.nanmax(flow_err))
        out["flow_low_gain_held_fraction"] = float(np.mean(held))
    # keys analyse_layer4_mpc.py expects
    if loop_config is not None:
        out["sample_period_s"] = float(loop_config.sample_period_s)
        out["position_tolerance_m"] = float(loop_config.position_tolerance_m)
        out["mpc_configuration"] = {
            "sample_period_s": float(loop_config.sample_period_s),
            "velocity_limit": np.asarray(loop_config.velocity_limit, float).tolist(),
            "acceleration_limit": np.asarray(loop_config.acceleration_limit,
                                             float).tolist(),
        }
        out["simulation_configuration"] = {
            "position_tolerance_m": float(loop_config.position_tolerance_m),
            "tangent_tolerance_rad": float(loop_config.tangent_tolerance_rad),
        }
    return out


# --------------------------------------------------------------------------
# project entry point
# --------------------------------------------------------------------------
def _run_project(args: argparse.Namespace) -> None:
    from proper_research.analysis import compare_controllers as CC
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.simulation.simulations import (
        simulate_time_parameterized_configuration_mpc as base,
    )
    from proper_research.controllers.beam_jacobian_providers import from_model_bundle

    exp_cfg, bundle, controller_pack, out_root = build_planning_context()
    reference_source = args.reference or DEFAULT_REFERENCE
    reference = base.load_configuration_reference(
        reference_source, require_planned_beam_feasible=False,
    )
    print(f"[flow-loop] reference: {reference_source} "
          f"({reference.sample_count} samples, {reference.duration_s:.2f} s, "
          f"dt={reference.sample_period_s:.4f} s)")

    forward6d = controller_pack["forward6d_plant"]
    adapter = controller_pack["plant_diagnostic_joint_adapter"]

    mpc_config = base.make_default_mpc_config(
        reference=reference, controller_pack=controller_pack,
        prediction_horizon=15,
    )
    loop_config = CC.LoopConfig(
        sample_period_s=float(reference.sample_period_s),
        velocity_limit=np.asarray(mpc_config.velocity_limit, dtype=float),
        acceleration_limit=np.asarray(mpc_config.acceleration_limit, dtype=float),
        state_min=np.asarray(mpc_config.state_min, dtype=float),
        state_max=np.asarray(mpc_config.state_max, dtype=float),
        position_tolerance_m=1e-3 * float(args.position_tolerance_mm),
        tangent_tolerance_rad=math.radians(float(args.tangent_tolerance_deg)),
        terminal_hold_steps=int(args.terminal_hold_steps),
        maximum_control_steps=args.max_steps,
    )
    initial_state = np.asarray(reference.state[0], dtype=float).copy()
    if args.initial_joint_offset_deg:
        offs = np.zeros(7)
        offs[:6] = np.radians(np.asarray(args.initial_joint_offset_deg, dtype=float))
        initial_state = initial_state + offs
        # report the tip offset this produces
        p0 = np.asarray(base._beam_snapshot(
            forward6d=forward6d, state=initial_state, commit=False)[0], dtype=float)
        d0 = np.asarray(reference.desired_position_m[0], dtype=float)
        print(f"[flow-loop] initial joint offset -> tip starts "
              f"{1e3*np.linalg.norm(p0 - d0):.3f} mm off the centreline")

    jacobian_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack,
        contact=(args.jacobian == "contact"),
    )
    print(f"[flow-loop] Jacobian: {jacobian_provider.summary_line()}")

    # start from the tuned dataclass defaults; override only what was passed
    overrides = {}
    if args.tip_weight is not None:
        overrides["tip_tracking_weight"] = float(args.tip_weight)
    if args.integral_weight is not None:
        overrides["integral_weight"] = float(args.integral_weight)
    if args.tip_velocity_weight is not None:
        overrides["tip_velocity_weight"] = float(args.tip_velocity_weight)
    if args.nullspace_gain is not None:
        overrides["nullspace_gain"] = float(args.nullspace_gain)
    riccati_cfg = replace(RiccatiConfig(), **overrides)
    flow_cfg = FlowFeedbackConfig(
        a_ring_m=float(args.a_ring_mm) * 1e-3,
        sigma_over_vmean=float(args.sigma_over_vmean),
        seed=int(args.seed),
    )
    flow_estimator = FlowTipEstimator(bundle.lumen_C, bundle.lumen_R, flow_cfg)

    def beam_snapshot(state: Array, commit: bool) -> tuple[Array, Array]:
        return base._beam_snapshot(forward6d=forward6d, state=state, commit=commit)

    def magnet_pose(state: Array) -> tuple[Array, Array]:
        return base._magnet_pose(controller_pack=controller_pack, state=state)

    reset_plant = getattr(adapter, "reset_to_initial_baseline", None)

    inverse_module = CC._controllers_module("inverse_jacobian_controller")

    def make_controller(kind: str) -> Any:
        if kind == "no_dare":
            return inverse_module.build_inverse_jacobian_controller(
                reference=reference, jacobian_provider=jacobian_provider,
                mpc_config=mpc_config,
                position_gain=float(args.resolved_rate_gain
                                    if args.resolved_rate_gain is not None else 1.0),
                nullspace_gain=float(args.nullspace_gain
                                     if args.nullspace_gain is not None
                                     else riccati_cfg.nullspace_gain),
            )
        return build_riccati_inverse_jacobian_controller(
            reference=reference, jacobian_provider=jacobian_provider,
            mpc_config=mpc_config, config=riccati_cfg,
        )

    runs = {
        "riccati_flow": ("riccati_inverse_jacobian",
                         "DARE inverse Jacobian, simulated flow feedback", "flow"),
        "riccati_vision": ("riccati_inverse_jacobian",
                           "DARE inverse Jacobian, perfect tip feedback (upper bound)",
                           "vision"),
        "no_dare_flow": ("no_dare",
                         "resolved-rate inverse Jacobian, simulated flow feedback",
                         "flow"),
        "feedforward_only": ("riccati_inverse_jacobian",
                             "feedforward only (no feedback) -- the uncorrected mismatch",
                             "none"),
    }
    selected = args.runs or list(runs)
    output_dir = Path(args.output_dir or (out_root / "flow_closed_loop"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for tag in selected:
        kind, description, feedback = runs[tag]
        print(f"\n[flow-loop] === {tag} ({feedback} feedback) ===", flush=True)
        controller = make_controller(kind)
        records = run_flow_closed_loop(
            controller=controller, reference=reference,
            beam_snapshot=beam_snapshot, magnet_pose=magnet_pose,
            flow_estimator=flow_estimator if feedback == "flow" else None,
            record_fn=CC._record, loop_config=loop_config,
            initial_state=initial_state, reset_plant=reset_plant,
            feedback=feedback, verbose=not args.quiet,
        )
        header = list(records[0].keys()) + [
            k for k in records[-1].keys() if k not in records[0]
        ]
        run_dir = output_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(run_dir / "configuration_mpc_simulation.csv", header, records)
        summary = summarise(records, tag, description, loop_config)
        (run_dir / "configuration_mpc_simulation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries.append(summary)
        print(f"[flow-loop] {tag}: tip rms {summary['rms_beam_position_error_mm']:.4f} mm, "
              f"max {summary['max_beam_position_error_mm']:.4f} mm, "
              f"final {summary['final_beam_position_error_mm']:.4f} mm")

    # fold in any previously-completed conditions in this output dir
    known = {s["controller"] for s in summaries}
    report_tags = list(selected)
    for tag in runs:
        j = output_dir / tag / "configuration_mpc_simulation_summary.json"
        if tag not in known and j.exists():
            summaries.append(json.loads(j.read_text()))
            report_tags.append(tag)

    _write_report(output_dir, summaries, reference_source, flow_cfg, riccati_cfg,
                  jacobian_provider.summary_line())
    _write_plot(output_dir, report_tags, args.dpi)
    print(f"\n[flow-loop] outputs -> {output_dir}")


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _write_report(output_dir: Path, summaries: list[dict], reference_source: Any,
                  flow_cfg: FlowFeedbackConfig, riccati_cfg: RiccatiConfig,
                  jacobian_line: str) -> None:
    try:
        from proper_research.analysis import _stack_common as C
    except ImportError:
        return
    report = C.Report("Flow-fed closed-loop tracking (Layer 5 + DARE)")
    report.metrics = {"runs": {s["controller"]: s for s in summaries},
                      "reference": str(reference_source),
                      "jacobian": jacobian_line,
                      "flow_feedback": {
                          "a_ring_m": flow_cfg.a_ring_m,
                          "sigma_over_vmean": flow_cfg.sigma_over_vmean,
                          "heart_rate_bpm": flow_cfg.heart_rate_bpm,
                          "note": "sigma_s uncharacterised; reported as sigma_s/v_mean"},
                      "riccati": {"tip_tracking_weight": riccati_cfg.tip_tracking_weight,
                                  "integral_weight": riccati_cfg.integral_weight,
                                  "tip_velocity_weight": riccati_cfg.tip_velocity_weight,
                                  "nullspace_gain": riccati_cfg.nullspace_gain}}
    report.text(
        "Beam-tip tracking of the pre-computed L1->L2->L3 reference for the "
        "single-30-degree-bend lumen, with the tip error estimated from the "
        "**simulated four-element flow ring** rather than vision. The plant keeps "
        "contact enabled; the controller Jacobian is contact-free, so the "
        "flow-fed integral term has a genuine model mismatch to reject.",
        f"Flow feedback: a_ring = {1e3*flow_cfg.a_ring_m:.2f} mm, "
        f"sigma_s/v_mean = {flow_cfg.sigma_over_vmean:.1%} (assumption), "
        f"{flow_cfg.heart_rate_bpm:.0f} bpm.  DARE weights: tip "
        f"{riccati_cfg.tip_tracking_weight:g}, integral "
        f"{riccati_cfg.integral_weight:g}, tip-velocity "
        f"{riccati_cfg.tip_velocity_weight:g}, nullspace gain "
        f"{riccati_cfg.nullspace_gain:g}.",
    )
    report.head("Tracking results")
    report.table(
        ["run", "tip rms [mm]", "tip max [mm]", "tip final [mm]",
         "tangent max [deg]", "flow est. rms [mm]", "held %"],
        [[s["controller"], f"{s['rms_beam_position_error_mm']:.4f}",
          f"{s['max_beam_position_error_mm']:.4f}",
          f"{s['final_beam_position_error_mm']:.4f}",
          f"{s['max_beam_tangent_error_deg']:.2f}",
          f"{s.get('flow_estimate_error_rms_mm', float('nan')):.4f}",
          f"{100*s.get('flow_low_gain_held_fraction', 0):.1f}"]
         for s in summaries],
    )
    by = {s["controller"]: s for s in summaries}
    if "riccati_flow" in by and "riccati_vision" in by:
        gap = (by["riccati_flow"]["rms_beam_position_error_mm"]
               - by["riccati_vision"]["rms_beam_position_error_mm"])
        report.finding(
            "ok" if abs(gap) < 0.5 else "warn",
            "Flow feedback vs perfect vision",
            f"flow-fed tip rms {by['riccati_flow']['rms_beam_position_error_mm']:.3f} mm "
            f"vs vision {by['riccati_vision']['rms_beam_position_error_mm']:.3f} mm "
            f"(gap {gap:+.3f} mm) -- the cost of replacing the camera with the ring.",
        )
    if "riccati_flow" in by and "feedforward_only" in by:
        report.finding(
            "ok",
            "DARE + flow vs open loop",
            f"open loop (feedforward only, same start) never removes the initial "
            f"offset -- final tip error "
            f"{by['feedforward_only']['final_beam_position_error_mm']:.3f} mm; the "
            f"flow-fed DARE brings the final error to "
            f"{by['riccati_flow']['final_beam_position_error_mm']:.3f} mm.",
        )
    if "riccati_flow" in by and "no_dare_flow" in by:
        report.finding(
            "info", "DARE vs plain resolved-rate, both flow-fed",
            f"resolved-rate (no DARE) tip rms "
            f"{by['no_dare_flow']['rms_beam_position_error_mm']:.3f} mm vs the "
            f"DARE {by['riccati_flow']['rms_beam_position_error_mm']:.3f} mm.",
        )
    report.write(output_dir, "flow_closed_loop")


def _write_plot(output_dir: Path, tags: list[str], dpi: int) -> None:
    try:
        import csv
        from proper_research.analysis import _stack_common as C
    except ImportError:
        return
    figure, axes = C.new_figure(2, 1, (11, 8))
    if figure is None:
        return
    for tag in tags:
        path = output_dir / tag / "configuration_mpc_simulation.csv"
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        t = np.array([float(r["time_s"]) for r in rows])
        err = np.array([float(r["beam_position_error_mm"]) for r in rows])
        axes[0].plot(t, err, label=tag)
        fe = [r.get("flow_estimate_error_m", "") for r in rows]
        if any(v not in ("", "nan") for v in fe):
            axes[1].plot(t, [1e3 * float(v) if v not in ("", "nan") else np.nan
                             for v in fe], label=tag)
    axes[0].set_ylabel("tip position error [mm]")
    axes[0].set_title("Closed-loop beam-tip tracking")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("flow estimate error [mm]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Flow ring: |estimate - true transverse offset|")
    axes[1].legend(fontsize=8)
    C.save_figure(figure, output_dir / "flow_closed_loop.png", dpi)


# --------------------------------------------------------------------------
# self-test -- mock plant + straight lumen, no project sim imports
# --------------------------------------------------------------------------
def _self_test() -> bool:
    ok = True
    dt = 0.01
    n = 250
    # raised-cosine arclength profile: zero path speed at both ends (like L3)
    tau = np.linspace(0.0, 1.0, n)
    s_axis = 0.05 * 0.5 * (1.0 - np.cos(np.pi * tau))

    _state = np.zeros((n, 7))
    _state[:, 6] = s_axis                       # insertion advances along the path
    _input = np.zeros((n, 7))
    _input[:, 6] = np.gradient(s_axis, dt)      # feedforward insertion rate

    class _Ref:
        sample_count = n
        sample_period_s = dt
        time_s = np.arange(n) * dt
        path_coordinate_m = s_axis.copy()
        duration_s = float(time_s[-1])
        state = _state
        input = _input
        desired_position_m = np.column_stack([s_axis, np.zeros(n), np.zeros(n)])
        desired_tangent = np.tile([1.0, 0.0, 0.0], (n, 1))

    reference = _Ref()
    # mock plant: tip x follows insertion; transverse follows joints 0,1;
    # a fixed transverse mismatch the controller has not modelled.
    J0 = np.zeros((3, 7))
    J0[1, 0] = 0.03
    J0[2, 1] = 0.03
    J0[0, 6] = 1.0
    mismatch = np.array([0.0, 0.8e-3, -0.5e-3])

    def beam_snapshot(state, commit):
        p = np.array([state[6], J0[1, 0] * state[0], J0[2, 1] * state[1]]) + mismatch
        return p, np.array([1.0, 0.0, 0.0])

    def magnet_pose(state):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

    C_lumen = np.column_stack([s_axis, np.zeros(n), np.zeros(n)])
    R_lumen = np.full(n, 3.0e-3)
    estimator = FlowTipEstimator(C_lumen, R_lumen,
                                 FlowFeedbackConfig(sigma_over_vmean=0.01, seed=0))

    ctrl = RiccatiInverseJacobianController(
        reference=reference, jacobian_provider=lambda st: J0, sample_period_s=dt,
        velocity_limit=np.array([0.5] * 6 + [0.2]),
        acceleration_limit=np.array([5.0] * 6 + [0.5]),
        state_min=np.array([-3.0] * 6 + [-1.0]),
        state_max=np.array([3.0] * 6 + [1.0]),
        config=RiccatiConfig(nullspace_gain=0.3),
    )

    class _LoopCfg:
        sample_period_s = dt
        velocity_limit = np.array([0.5] * 6 + [0.2])
        acceleration_limit = np.array([5.0] * 6 + [0.5])
        state_min = np.array([-3.0] * 6 + [-1.0])
        state_max = np.array([3.0] * 6 + [1.0])
        position_tolerance_m = 1.5e-3
        tangent_tolerance_rad = math.pi / 2
        terminal_hold_steps = 20
        maximum_control_steps = None

    def record_fn(*, sample, reference_index, reference, state, command,
                  previous_command, output, achieved_position, achieved_tangent,
                  magnet_position, magnet_quaternion, config):
        desired = reference.desired_position_m[reference_index]
        err = float(np.linalg.norm(achieved_position - desired))
        return {
            "simulation_sample": sample, "time_s": sample * dt,
            "beam_position_error_m": err, "beam_position_error_mm": 1e3 * err,
            "beam_tangent_error_rad": 0.0, "beam_feasible": int(err < 1.5e-3),
        }

    records = run_flow_closed_loop(
        controller=ctrl, reference=reference, beam_snapshot=beam_snapshot,
        magnet_pose=magnet_pose, flow_estimator=estimator, record_fn=record_fn,
        loop_config=_LoopCfg(), initial_state=reference.state[0],
        feedback="flow", verbose=False,
    )
    errs = np.array([r["beam_position_error_m"] for r in records])
    t1 = errs[-1] < 3e-4 and errs[-1] < 0.35 * errs[2]
    print(f"  [{'ok  ' if t1 else 'FAIL'}] flow-fed closed loop drives the tip onto "
          f"the centreline  {1e3*errs[2]:.3f} mm -> {1e6*errs[-1]:.1f} um")
    ok &= t1

    records_v = run_flow_closed_loop(
        controller=ctrl, reference=reference, beam_snapshot=beam_snapshot,
        magnet_pose=magnet_pose, flow_estimator=None, record_fn=record_fn,
        loop_config=_LoopCfg(), initial_state=reference.state[0],
        feedback="none", verbose=False,
    )
    errs_v = np.array([r["beam_position_error_m"] for r in records_v])
    t2 = errs_v[-1] > 3e-4
    print(f"  [{'ok  ' if t2 else 'FAIL'}] feedforward-only leaves the mismatch "
          f"uncorrected  ({1e3*errs_v[-1]:.3f} mm)")
    ok &= t2

    t3 = "flow_e_hat_norm_m" in records[5]
    print(f"  [{'ok  ' if t3 else 'FAIL'}] flow diagnostic columns are recorded")
    ok &= t3

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--runs", nargs="+", default=None,
                   choices=["riccati_flow", "riccati_vision", "no_dare_flow",
                            "feedforward_only"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--terminal-hold-steps", type=int, default=25)
    p.add_argument("--sigma-over-vmean", type=float, default=0.02)
    p.add_argument("--a-ring-mm", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-joint-offset-deg", type=float, nargs=6, default=None,
                   help="perturb the initial arm joints so the tip starts off "
                        "the centreline (a clean centring demonstration)")
    p.add_argument("--jacobian", choices=["no_contact", "contact"], default="no_contact")
    p.add_argument("--tip-weight", type=float, default=None)
    p.add_argument("--integral-weight", type=float, default=None)
    p.add_argument("--tip-velocity-weight", type=float, default=None)
    p.add_argument("--nullspace-gain", type=float, default=None)
    p.add_argument("--resolved-rate-gain", type=float, default=1.0)
    p.add_argument("--position-tolerance-mm", type=float, default=1.5)
    p.add_argument("--tangent-tolerance-deg", type=float, default=90.0)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.self_test:
        raise SystemExit(0 if _self_test() else 1)
    _run_project(args)


if __name__ == "__main__":
    main()
