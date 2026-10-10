"""Build a model bundle + controller pack for an EXPLICIT bend geometry.

``proper_research.planning.planning_context.build_planning_context()`` reads
its geometry from ``make_experiment_config()``, which hard-codes ONE current
bend pair (edited by hand between studies -- see its git history). That is
fine for interactive planning work, but it means any script that calls
``build_planning_context()`` silently follows whatever geometry someone last
left in that file, which is a bad foundation for a *reusable* study script:
run it a week later, after the geometry has moved on, and it studies the wrong
vessel (or, if no reference has been planned for the new one yet, fails).

``build_pinned_planning_context`` does the same construction
(``make_double_bend_lumen_config`` -> ``make_curvature_jacobian_grid`` ->
``build_model_bundle`` -> ``build_controller``) but takes the bend angles and
radius as explicit arguments, so a study is reproducible regardless of what
``planning_context.py`` currently contains. Every other setting (robot config,
design config, controller/solver knobs) still comes from
``planning_context.py`` -- only the two bend angles and the lumen radius are
pinned, because those are what changes between vessels.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any


def build_pinned_planning_context(
    *,
    first_angle_deg: float,
    second_angle_deg: float,
    radius_m: float | None = None,
    run_root: Path | None = None,
) -> tuple[Any, Any, dict[str, Any], Path]:
    """Return ``(exp_cfg, bundle, controller_pack, out_root)`` for one explicit vessel.

    ``radius_m=None`` keeps ``make_double_bend_lumen_config``'s own default.
    ``run_root=None`` uses ``planning_context.DEFAULT_RUN_ROOT``.
    """
    from proper_research.planning.planning_context import (
        DEFAULT_RUN_ROOT,
        make_design_config,
        make_double_bend_lumen_config,
        make_robot_config,
    )
    from proper_research.simulation.simulations.controller_factory_joint_space import (
        build_controller,
    )
    from proper_research.simulation.simulations.initial_conditions import (
        make_initial_poses,
    )
    from proper_research.simulation.simulations.model_factory import (
        build_model_bundle,
    )
    from proper_research.simulation.simulations.scenario import (
        make_curvature_jacobian_grid,
    )

    lumen_config = make_double_bend_lumen_config(
        first_angle_deg=float(first_angle_deg),
        second_angle_deg=float(second_angle_deg),
    )
    if radius_m is not None:
        lumen_config = replace(lumen_config, radius=float(radius_m))

    experiments = make_curvature_jacobian_grid(
        run_root=Path(run_root or DEFAULT_RUN_ROOT),
        lumen_configs=(lumen_config,),
        jacobian_variants=("contact",),
        controller_kinds=("mpc",),
        solver_modes=("sqp_full",),
        inverse_sequence_modes=("rollout_ltv",),
        inverse_kp_values=(1,),
        inverse_desired_step_max_m=4.0e-3,
        inverse_extra_damping=0.0,
        rollout_steps_values=(10,),
        Np=15,
        N_sqp=50,
        max_steps=25,
        plant_contact=True,
        adaptive_rollout_enabled=False,
    )
    if len(experiments) != 1:
        raise RuntimeError(
            f"Expected exactly one experiment for angles ({first_angle_deg}, "
            f"{second_angle_deg}); the grid produced {len(experiments)}."
        )
    exp_cfg = experiments[0]
    exp_cfg.validate()

    out_root = Path(exp_cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pivot_point, start_point, L0, dt = make_initial_poses()
    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=exp_cfg.lumen,
        plant_contact=exp_cfg.model.plant_contact,
    )
    controller_pack = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=bundle.models["plant"],
        jacobian_model=bundle.models[exp_cfg.model.jacobian_variant],
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=make_design_config(),
        robot_cfg=make_robot_config(),
    )
    return exp_cfg, bundle, controller_pack, out_root


def add_geometry_arguments(parser) -> None:
    """Shared CLI flags: pin a vessel explicitly, or fall back to planning_context.py.

    Add to any study script's ``argparse.ArgumentParser`` and pass the parsed
    args to :func:`resolve_planning_context_from_args`.
    """
    group = parser.add_argument_group(
        "vessel geometry",
        "Pin the bend geometry explicitly (recommended for a reusable/repeatable "
        "study). Omit all three to fall back to whatever "
        "proper_research/planning/planning_context.py currently builds.",
    )
    group.add_argument("--first-angle-deg", type=float, default=None,
                        help="First bend angle, degrees (e.g. 30.0).")
    group.add_argument("--second-angle-deg", type=float, default=None,
                        help="Second bend angle, degrees (e.g. -70.0).")
    group.add_argument("--nominal-radius-mm", type=float, default=None,
                        help="Nominal lumen radius, mm (e.g. 3.5). Only used "
                             "together with the two angle flags.")
    group.add_argument("--run-root", type=Path, default=None,
                        help="Override the run root directory (default: "
                             "planning_context.DEFAULT_RUN_ROOT).")


def resolve_planning_context_from_args(args) -> tuple[Any, Any, dict[str, Any], Path]:
    """Build the context per ``add_geometry_arguments``' flags, printing which path was taken."""
    pinned = (
        args.first_angle_deg is not None or args.second_angle_deg is not None
    )
    if pinned:
        if args.first_angle_deg is None or args.second_angle_deg is None:
            raise SystemExit(
                "--first-angle-deg and --second-angle-deg must be given together."
            )
        radius_m = (
            None if args.nominal_radius_mm is None
            else 1.0e-3 * float(args.nominal_radius_mm)
        )
        print(
            f"[geometry] pinned explicitly: {args.first_angle_deg:+.1f}/"
            f"{args.second_angle_deg:+.1f} deg"
            + ("" if radius_m is None else f", radius {args.nominal_radius_mm:.3f} mm"),
            flush=True,
        )
        exp_cfg, bundle, controller_pack, out_root = build_pinned_planning_context(
            first_angle_deg=args.first_angle_deg,
            second_angle_deg=args.second_angle_deg,
            radius_m=radius_m,
            run_root=args.run_root,
        )
    else:
        from proper_research.planning.planning_context import build_planning_context

        print(
            "[geometry] no --first-angle-deg/--second-angle-deg given: following "
            "planning_context.py's current geometry (may change if that file is "
            "edited between runs -- pin explicitly for a reproducible study).",
            flush=True,
        )
        exp_cfg, bundle, controller_pack, out_root = build_planning_context(
            run_root=args.run_root or Path("uprgrade_configuration")
        )
    return exp_cfg, bundle, controller_pack, out_root


def resolve_reference_dir(out_root, explicit=None):
    """Find the time-parameterised reference trajectory for a vessel.

    Tries, in order: an explicit path; the standard
    ``<out_root>/time_parameterized_configuration_path``; then the
    ``--from-inverse`` and ``--profile both`` outputs
    (``time_param_from_inverse/optimal``,
    ``time_param_profile_comparison/optimal``). Raises a clear error naming
    every location it checked and how to produce one, instead of letting the
    cryptic "Reference must be a directory, .csv file, or .npz file" surface
    from deep in the loader.
    """
    from pathlib import Path

    def _shape_ok(path: Path) -> bool:
        return path.is_dir() or path.suffix.lower() in (".csv", ".npz")

    def _loads(path: Path) -> str | None:
        """None if the reference loads and validates, else a short reason."""
        try:
            from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (  # noqa: E501
                load_configuration_reference,
            )
            load_configuration_reference(path, require_planned_beam_feasible=False)
            return None
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    if explicit is not None:
        explicit = Path(explicit)
        if not _shape_ok(explicit):
            raise SystemExit(
                f"--reference {explicit} is not a directory, .csv or .npz file."
            )
        reason = _loads(explicit)
        if reason is not None:
            raise SystemExit(f"--reference {explicit} does not load: {reason}")
        return explicit

    out_root = Path(out_root)
    candidates = [
        out_root / "time_parameterized_configuration_path",
        out_root / "time_param_from_inverse" / "optimal",
        out_root / "time_param_profile_comparison" / "optimal",
    ]
    skipped: list[str] = []
    for candidate in candidates:
        if not _shape_ok(candidate):
            continue
        reason = _loads(candidate)
        if reason is not None:
            skipped.append(f"  {candidate}\n    -> {reason}")
            continue
        if candidate != candidates[0]:
            print(f"[reference] {candidates[0]} unusable; using {candidate} instead.",
                  flush=True)
        return candidate

    message = (
        "No usable time-parameterised reference trajectory found. Looked for:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )
    if skipped:
        message += "\n\nFound but not usable (e.g. run with --skip-beam-validation):\n" + \
            "\n".join(skipped)
    message += (
        "\n\nRun the time parameteriser WITHOUT --skip-beam-validation, e.g.\n"
        "  python -m proper_research.planning.run_time_parameterization --from-inverse \\\n"
        "      --first-angle-deg <A> --second-angle-deg <B> --nominal-radius-mm <R>\n"
        "or pass --reference <path> explicitly."
    )
    raise SystemExit(message)


__all__ = [
    "build_pinned_planning_context",
    "add_geometry_arguments",
    "resolve_planning_context_from_args",
    "resolve_reference_dir",
]
