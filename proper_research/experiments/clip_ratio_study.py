"""Test C, generalized: does the naive resolved-rate controller's clip ratio
track path difficulty, and is the naive/MPC gap present even with a perfect
(contact-aware) Jacobian?

Reusable, geometry-agnostic version of
``results/contact_study_radius_p30_m50/scripts/test_c_clip.py`` +
``test_c_full.py`` merged into one script. Instruments
``InverseJacobianBeamController._clip`` (the clip logic itself is untouched)
to log, per control step: the surviving magnitude alpha = ||clipped||/||raw||,
the direction change cos(theta) between the raw and clipped joint command, and
which bound (velocity / acceleration / one-step state box) the raw command
broke. Runs the controller closed-loop on one or more (radius, Jacobian model)
cells and reports both the whole-path numbers and a breakdown by path segment
(pre-bend / bend / post-bend, auto-derived from the vessel's own bend
geometry -- generalizes to any number of bends).

Two use cases, one script
--------------------------
* **Quick, multi-cell**: ``--max-steps 320`` with several radii and both
  Jacobian models -- cheap, good for checking whether the contact vs
  contact-free *Jacobian* changes the clip pattern before paying for a full run.
* **Full-length, headline**: default (no ``--max-steps``) -- the whole
  reference, so the clip metrics include wherever the tracking error actually
  concentrates (usually the sharpest bend apex).

Both write the same schema; use ``--tag`` to keep separate runs from
overwriting each other in one ``--output-dir``.

Outputs (into --output-dir), one pair per --tag
------------------------------------------------
test_c_clip_<tag>.json         one record per cell: whole-path + per-segment stats
test_c_clip_<tag>_steps.csv    per-step long table (for plot_clip_ratio_study.py):
                                cell, radius_mm, jacobian_model, k, arc_mm, segment,
                                alpha_mag, cos_joint, at_vel, at_acc, at_state,
                                position_error_mm

Usage
-----
    python -m proper_research.experiments.clip_ratio_study \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.0 0.40 --jacobian-models contact no_contact \\
        --max-steps 320 --tag quick \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5

    python -m proper_research.experiments.clip_ratio_study \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.0 0.40 --jacobian-models contact --tag full \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5

    python -m proper_research.experiments.plot_clip_ratio_study \\
        --study-dir results/contact_study_radius_p30_m70 --tags quick full
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent / "controllers"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from proper_research.experiments import run_contact_study as RCS  # noqa: E402
from proper_research.experiments.pinned_planning_context import (  # noqa: E402
    add_geometry_arguments,
    resolve_planning_context_from_args,
    resolve_reference_dir,
)

_LOG: list[dict[str, Any]] = []
_CTX: dict[str, Any] = {"cell": None, "k": 0}


def _install_clip_patch(inverse_module) -> None:
    """Monkey-patch ``InverseJacobianBeamController._clip`` to log, not to change it."""
    cls = inverse_module.InverseJacobianBeamController
    if getattr(cls, "_clip_instrumented", False):
        return
    original = cls._clip

    def _instrumented(self, command, state, previous):
        raw = np.asarray(command, dtype=float).reshape(7)
        clipped = original(self, command, state, previous)
        c = np.asarray(clipped, dtype=float).reshape(7)
        nr, nc = float(np.linalg.norm(raw)), float(np.linalg.norm(c))
        vel, acc, dt = self.velocity_limit, self.acceleration_limit, self.dt
        at_vel = bool(np.any(np.abs(raw) > vel * (1 + 1e-9)))
        at_acc = bool(np.any(np.abs(raw - previous) > acc * dt * (1 + 1e-9)))
        lo, hi = (self.state_min - state) / dt, (self.state_max - state) / dt
        at_state = bool(np.any(raw < lo - 1e-12) or np.any(raw > hi + 1e-12))
        _LOG.append(dict(
            cell=_CTX["cell"], k=_CTX["k"],
            alpha_mag=(nc / nr) if nr > 1e-12 else 1.0,
            cos_joint=(float(raw @ c) / (nr * nc)) if (nr > 1e-12 and nc > 1e-12) else 1.0,
            delta_rel=float(np.linalg.norm(raw - c)) / max(nr, 1e-12),
            at_vel=at_vel, at_acc=at_acc, at_state=at_state,
            any_active=bool(at_vel or at_acc or at_state),
        ))
        _CTX["k"] += 1
        return clipped

    cls._clip = _instrumented
    cls._clip_instrumented = True


def _build_naive_environment(
    *, context, jacobian_model: str, settings, base_module, harness, inverse_module,
) -> dict[str, Any]:
    """A fast ``build_cell_environment`` for the naive controller only.

    ``run_contact_study.build_cell_environment`` unconditionally builds the
    offline Jacobian schedule and all three MPC variants (LTI/LTV-offline/
    SQP-online) over the *whole* reference before you can call ``make()`` for
    even one of them -- for a several-hundred-sample reference that dominates
    the run time, and this study never uses any of it: the clip diagnostic
    only exists on ``InverseJacobianBeamController``. This builds only what
    the naive controller needs (a couple of seconds instead of minutes).
    """
    reference = context.reference
    provider = (
        context.contact_provider if jacobian_model == "contact"
        else context.contact_free_provider
    )
    mpc_config = base_module.make_default_mpc_config(
        reference=reference,
        controller_pack=context.controller_pack,
        prediction_horizon=settings.horizon,
        joint_acceleration_limit_rad_s2=settings.joint_acceleration_limit,
        insertion_acceleration_limit_m_s2=settings.insertion_acceleration_limit,
        velocity_safety_factor=settings.velocity_safety_factor,
        acceleration_safety_factor=settings.acceleration_safety_factor,
        solver_backend=settings.solver,
        solver_verbose=False,
    )
    loop_config = harness.LoopConfig(
        sample_period_s=float(reference.sample_period_s),
        velocity_limit=np.asarray(mpc_config.velocity_limit, dtype=float),
        acceleration_limit=np.asarray(mpc_config.acceleration_limit, dtype=float),
        state_min=np.asarray(mpc_config.state_min, dtype=float),
        state_max=np.asarray(mpc_config.state_max, dtype=float),
        position_tolerance_m=1e-3 * float(settings.position_tolerance_mm),
        tangent_tolerance_rad=math.radians(float(settings.tangent_tolerance_deg)),
        terminal_hold_steps=int(settings.terminal_hold_steps),
        maximum_control_steps=settings.max_control_steps,
        stop_on_qp_failure=bool(settings.stop_on_qp_failure),
    )
    controller = inverse_module.build_inverse_jacobian_controller(
        reference=reference,
        jacobian_provider=provider,
        mpc_config=mpc_config,
        position_gain=float(settings.resolved_rate_gain),
        damping=float(settings.resolved_rate_damping),
        nullspace_gain=float(settings.nullspace_gain),
    )
    adapted = harness.ControllerAdapter(
        "naive_inverse_jacobian", controller.description, controller
    )

    forward6d = context.controller_pack.get("forward6d_plant")
    if forward6d is None:
        raise KeyError("controller_pack has no 'forward6d_plant'.")
    adapter = context.controller_pack.get("plant_diagnostic_joint_adapter")

    def beam_snapshot(state, commit):
        return base_module._beam_snapshot(forward6d=forward6d, state=state, commit=commit)

    def magnet_pose(state):
        return base_module._magnet_pose(controller_pack=context.controller_pack, state=state)

    reset_plant = getattr(adapter, "reset_to_initial_baseline", None)
    if reset_plant is None:
        forward_adapter = getattr(getattr(adapter, "beam_output_fn", None), "forward_adapter", None)
        reset_plant = getattr(forward_adapter, "reset_to_initial_baseline", None)

    return {
        "make": lambda name: adapted,
        "loop_config": loop_config,
        "beam_snapshot": beam_snapshot,
        "magnet_pose": magnet_pose,
        "reset_plant": reset_plant,
        "provider": provider,
        "mpc_config": mpc_config,
    }


def _segment_breakpoints_m(bends, pad_m: float) -> list[tuple[str, float, float]]:
    """[(label, start_m, end_m), ...] merging overlapping padded bend intervals."""
    raw = sorted(
        (float(b.bend_start) - pad_m, float(b.bend_end) + pad_m) for b in bends
    )
    merged: list[list[float]] = []
    for lo, hi in raw:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    labels = []
    prev_end = 0.0
    for i, (lo, hi) in enumerate(merged):
        if lo > prev_end:
            labels.append((f"pre_bend{i + 1}" if i > 0 else "pre_bend", prev_end, lo))
        labels.append((f"bend{i + 1}", lo, hi))
        prev_end = hi
    labels.append(("post_bend", prev_end, float("inf")))
    return labels


def _segment_of(arc_m: float, breakpoints: list[tuple[str, float, float]]) -> str:
    for label, lo, hi in breakpoints:
        if lo <= arc_m < hi:
            return label
    return breakpoints[-1][0]


def _seg_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    a = np.array([r["alpha_mag"] for r in rows])
    cj = np.array([r["cos_joint"] for r in rows])
    dr = np.array([r["delta_rel"] for r in rows])
    return dict(
        n=len(rows),
        clip_activity_rate=float(np.mean([r["any_active"] for r in rows])),
        at_vel_rate=float(np.mean([r["at_vel"] for r in rows])),
        at_acc_rate=float(np.mean([r["at_acc"] for r in rows])),
        at_state_rate=float(np.mean([r["at_state"] for r in rows])),
        alpha_mag_mean=float(np.mean(a)), alpha_mag_median=float(np.median(a)),
        alpha_mag_p05=float(np.percentile(a, 5)), alpha_mag_min=float(np.min(a)),
        cos_joint_mean=float(np.mean(cj)), cos_joint_min=float(np.min(cj)),
        cos_joint_p05=float(np.percentile(cj, 5)),
        delta_rel_mean=float(np.mean(dr)), delta_rel_max=float(np.max(dr)),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test C: naive-controller clip ratio, by path segment, "
                    "reusable across vessels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--controller", default="naive_inverse_jacobian",
                         help="Only naive_inverse_jacobian exposes _clip; other "
                              "controllers run fine but log no clip rows.")
    parser.add_argument("--radius-scales", type=float, nargs="+", default=(1.0, 0.40))
    parser.add_argument("--jacobian-models", nargs="+", default=("contact",),
                         choices=("contact", "no_contact"))
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Omit for the full reference length (the headline run).")
    parser.add_argument("--bend-pad-mm", type=float, default=2.0,
                         help="Padding added around each bend's [start,end] for segmentation.")
    parser.add_argument("--tag", default="",
                         help="Suffix for output filenames, e.g. 'quick' or 'full'.")
    parser.add_argument("--force-full-environment", action="store_true",
                         help="Build all 4 MPC variants (run_contact_study's normal path) even "
                              "for the naive controller. Much slower; only useful for parity "
                              "debugging against run_contact_study's own cells.")
    add_geometry_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    tag = f"_{args.tag}" if args.tag else ""

    harness = RCS._load("compare_controllers")
    providers_module = RCS._load("beam_jacobian_providers")
    variants_module = RCS._load("mpc_variants")
    inverse_module = RCS._load("inverse_jacobian_controller")
    base_module = harness.load_base_module()
    beam_module = harness.load_beam_output_module()
    _install_clip_patch(inverse_module)

    exp_cfg, bundle0, pack0, out_root = resolve_planning_context_from_args(args)
    RCS._assert_plant_contact(bundle0, "the baseline planning context")
    reference_dir = resolve_reference_dir(out_root, args.reference)

    base_context = RCS.resolve_planning_context(exp_cfg, pack0)
    base_context["reference_dir"] = reference_dir
    base_context.pop("resolved_from", None)

    breakpoints = _segment_breakpoints_m(exp_cfg.lumen.bends, args.bend_pad_mm * 1.0e-3)
    print("[segments]", [(label, round(1e3 * lo, 1), round(1e3 * hi, 1))
                          for label, lo, hi in breakpoints], flush=True)

    settings = RCS.StudySettings(verbose=False, max_control_steps=args.max_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for scale in args.radius_scales:
        name = f"radius_x{scale:g}".replace(".", "p")
        spec = RCS.AnatomySpec(name=name, radius_scale=float(scale))
        ctx = RCS.build_anatomy_context(
            spec=spec, base_context=base_context, providers_module=providers_module,
            base_module=base_module, require_planned_beam_feasible=False,
            shared_controller_pack=None,
        )
        radius_mm = round(1000.0 * float(np.mean(np.asarray(ctx.bundle.lumen_R, dtype=float))), 4)
        ic0 = np.asarray(ctx.reference.state, dtype=float)[0]
        arc_m = np.asarray(ctx.reference.path_coordinate_m, dtype=float)
        arc_mm_rel = 1.0e3 * (arc_m - float(arc_m[0]))

        for jacobian_model in args.jacobian_models:
            started = time.perf_counter()
            use_light = (
                args.controller == "naive_inverse_jacobian"
                and not args.force_full_environment
            )
            if use_light:
                env = _build_naive_environment(
                    context=ctx, jacobian_model=jacobian_model, settings=settings,
                    base_module=base_module, harness=harness, inverse_module=inverse_module,
                )
            else:
                env = RCS.build_cell_environment(
                    context=ctx, jacobian_model=jacobian_model, settings=settings,
                    disturbance_alpha=0.0, base_module=base_module, beam_module=beam_module,
                    harness=harness, variants_module=variants_module, inverse_module=inverse_module,
                )
            cell = f"{spec.name}__{jacobian_model}__{args.controller}{tag}"
            _CTX["cell"] = cell
            _CTX["k"] = 0
            n0 = len(_LOG)
            controller = env["make"](args.controller)
            res = harness.run_closed_loop(
                controller=controller, reference=ctx.reference,
                beam_snapshot=env["beam_snapshot"], magnet_pose=env["magnet_pose"],
                reset_plant=env["reset_plant"], config=env["loop_config"],
                initial_state=ic0, verbose=False,
            )
            rows = _LOG[n0:]
            elapsed = float(time.perf_counter() - started)

            position_error_mm = [
                float(r.get("beam_position_error_mm", float("nan"))) for r in res.records[1:]
            ]
            for row in rows:
                k = row["k"]
                arc_mm = float(arc_mm_rel[min(k, len(arc_mm_rel) - 1)])
                arc_abs_m = float(arc_m[min(k, len(arc_m) - 1)])
                seg = _segment_of(arc_abs_m, breakpoints)
                perr = position_error_mm[k] if k < len(position_error_mm) else float("nan")
                step_rows.append(dict(
                    cell=cell, anatomy=name, radius_mm=radius_mm,
                    jacobian_model=jacobian_model, controller=args.controller,
                    k=k, arc_mm=arc_mm, segment=seg,
                    alpha_mag=row["alpha_mag"], cos_joint=row["cos_joint"],
                    delta_rel=row["delta_rel"],
                    at_vel=int(row["at_vel"]), at_acc=int(row["at_acc"]),
                    at_state=int(row["at_state"]), any_active=int(row["any_active"]),
                    position_error_mm=perr,
                ))

            cell_steps = step_rows[-len(rows):] if rows else []
            by_segment: dict[str, list[dict]] = {}
            for srow in cell_steps:
                by_segment.setdefault(srow["segment"], []).append(srow)

            summary = res.summary
            rec = dict(
                cell=cell, anatomy=name, radius_mm=radius_mm,
                jacobian_model=jacobian_model, controller=args.controller,
                steps=len(rows), max_steps=args.max_steps,
                rms_mm=summary.get("rms_beam_position_error_mm"),
                max_mm=summary.get("maximum_beam_position_error_mm"),
                elapsed_s=elapsed,
                whole=_seg_stats(rows),
                by_segment={label: _seg_stats(by_segment.get(label, []))
                            for label, _, _ in breakpoints},
            )
            results.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "by_segment"}), flush=True)

            (args.output_dir / f"test_c_clip{tag}.json").write_text(json.dumps(results, indent=2))
            if step_rows:
                header = list(step_rows[0].keys())
                with (args.output_dir / f"test_c_clip{tag}_steps.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(step_rows)

    print("wrote", args.output_dir / f"test_c_clip{tag}.json")
    print("wrote", args.output_dir / f"test_c_clip{tag}_steps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
