"""Planning context for a REAL, digitized vessel lumen.

``planning_context.build_planning_context()`` builds its contact-aware model
against a permissive, non-binding placeholder lumen (a straight, wide-bore
tube along the beam axis) -- deliberately, because the free-space "shape"
planning this session used up to now (triangle/rectangle/S-curve/U-shape)
never needed the contact model to actually bind against anything: the target
path was passed separately (see ``plan_shape_path.py``'s ``lumen_C`` argument
to ``solve_offline_inverse_configuration`` / ``optimize_from_saved_inverse_result``
/ ``time_parameterize_saved_*``), and that argument is ONLY a target
polyline + tracking-tolerance radius for the position objective (and the
magnet-exclusion reference) -- it does not touch the beam's own contact
physics at all.

For a real vessel, that placeholder is no longer good enough: the whole
point is for the beam's actual equilibrium to be computed against the TRUE
wall geometry, so contact energy/force is genuinely nonzero where the beam
is actually near a wall. This module builds the same planning context as
``build_planning_context()``, then swaps the contact-aware model's
``lumen_query`` for the real, digitized vessel geometry
(``detect_blue.py::create_vessel_lumen_in_robot_frame`` /
``vessel_lumen_robot_frame.json``) in place -- ``controller_pack``'s
adapters hold references to the same model objects, so this mutation is
visible to every downstream solve without rebuilding anything else.

The ``no_contact`` model is left untouched (``lumen_query=None``,
``ContactConfig.disabled()``) -- it is the deliberately contact-blind
counterpart used for the contact-vs-no-contact Jacobian comparison
(``beam_jacobian_providers.py``), and must stay that way.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


def build_vessel_planning_context(
    *,
    lumen_file: str | Path,
    run_root: Path | None = None,
    insertion_max_m: float | None = None,
    jacobian_mode: str = "fast",
    initial_poses: tuple | None = None,
):
    """Like ``planning_context.build_planning_context()``, but the
    contact-aware model's lumen is the REAL digitized vessel geometry from
    ``lumen_file`` (a ``vessel_lumen_robot_frame.json``-format file, robot
    frame R), not the permissive placeholder.

    `jacobian_mode`: passed through to the real ``build_controller`` call
    below (the one whose controller_pack is actually returned/used -- the
    one at module import inside ``build_planning_context()`` is rebuilt
    immediately after, see the 2026-09-27 bugfix comment below). Default
    "fast" preserves existing behavior; pass "accurate" for Layer 1 offline
    solving where "fast" mode's numerical-estimator artifact can make the
    optimizer chase a bad gradient -- see `build_controller`'s own
    docstring.

    `initial_poses`: optional ``(pivot_point, start_point, L0, dt)``
    override, same shape as ``initial_conditions.make_initial_poses()``
    returns. Threaded into BOTH the internal ``build_planning_context()``
    call and this function's own rebuild call below, so the two stay
    consistent -- the same guarantee module-level monkey-patching used to
    provide, without reassigning shared module state. ``None`` (default)
    preserves existing behavior exactly.

    Returns
    -------
    tuple
        ``(exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance)``.
        ``lumen_C``/``lumen_R`` are the real vessel geometry (metres, frame
        R) -- pass these as the ``lumen_C``/``lumen_R`` (target path +
        tracking tolerance) arguments to ``solve_from_controller_pack`` /
        ``optimize_from_saved_inverse_result`` / ``time_parameterize_saved_*``,
        exactly as ``plan_shape_path.py`` passes its analytic shape
        centreline -- the SAME real geometry now serves both roles (contact
        wall and tracking target), instead of a shape-specific target and an
        unrelated placeholder wall.
    """
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.simulation.magnetic_beam.contact import LumenQuery
    from proper_research.vision.detect_blue import load_vessel_lumen_robot_frame

    lumen_C, lumen_R, provenance = load_vessel_lumen_robot_frame(lumen_file)
    if lumen_C.shape[0] < 2:
        raise ValueError(f"{lumen_file} contains fewer than 2 lumen samples.")

    # 2026-10-03 safety check: found live that a lumen file digitized
    # before the 2026-10-02 BEAM_BASE_PIVOT_Z recalibration still has its
    # original (now stale) Z height baked into lumen_C -- the contact
    # model then pulls the beam toward the WRONG wall height, and its
    # huge penalty stiffness (k=1e8, k_hard=1e10) completely dominates
    # the weak magnetic forces, making the beam shape converge to that
    # wrong-height contact point almost independent of magnet pose. See
    # probe_beam_configuration.py / the 2026-10-03 commit for the full
    # diagnosis. Refuse early with a clear message rather than silently
    # building a corrupted contact model -- recreate the lumen file via
    # shift_lumen_centerline.py (any shift amount, even 0) or add the
    # offset directly to lumen_C_m's z column.
    from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
        BEAM_BASE_PIVOT_Z,
    )
    _lumen_z_offset_mm = float(np.abs(lumen_C[:, 2] - BEAM_BASE_PIVOT_Z).max()) * 1e3
    if _lumen_z_offset_mm > 2.0:
        raise ValueError(
            f"{lumen_file}: lumen_C z values differ from the current "
            f"BEAM_BASE_PIVOT_Z ({BEAM_BASE_PIVOT_Z*1e3:.3f}mm) by up to "
            f"{_lumen_z_offset_mm:.2f}mm -- this lumen file was digitized "
            f"against a different (likely stale, pre-recalibration) beam-base "
            f"height and would silently corrupt the contact model (see the "
            f"2026-10-03 stale-lumen-Z bug). Z-correct it first -- see any "
            f"vessel_lumen_robot_frame_*_zcorrected.json for the pattern."
        )

    kwargs: dict[str, Any] = {}
    if run_root is not None:
        kwargs["run_root"] = run_root
    if initial_poses is not None:
        kwargs["initial_poses"] = initial_poses
    exp_cfg, bundle, controller_pack, out_root = build_planning_context(**kwargs)

    contact_model = bundle.models.get("contact")
    if contact_model is None:
        raise RuntimeError("bundle.models['contact'] is missing; cannot wire in a real vessel lumen.")
    if getattr(contact_model, "lumen_query", "__missing__") == "__missing__":
        raise RuntimeError(
            "bundle.models['contact'] has no 'lumen_query' attribute; the "
            "MagneticBeamForwardModel API may have changed -- check "
            "forward_model.py before trusting this wiring."
        )

    real_query = LumenQuery(lumen_C, lumen_R)
    contact_model.lumen_query = real_query

    # 2026-09-27 BUGFIX: `controller_pack` (built above by
    # `build_planning_context()`) is NOT actually wired to `contact_model`
    # by reference -- `build_controller()` (which `build_planning_context()`
    # calls internally) does `owned_plant_model = copy.deepcopy(plant_model)`,
    # baking a SNAPSHOT of the contact model into the adapter at that time.
    # Setting `contact_model.lumen_query = real_query` immediately above only
    # mutates `bundle.models["contact"]`, a now-separate object from the one
    # already deep-copied into controller_pack's adapters. Every caller that
    # left `insertion_max_m=None` (i.e. every vessel plan built before this
    # fix) therefore solved against the ORIGINAL placeholder lumen_query (a
    # permissive 100-point straight tube starting at the beam base), NOT the
    # real digitized vessel -- confirmed by inspecting
    # controller_pack["plant_diagnostic_joint_adapter"].model.lumen_query.C
    # directly. The rebuild below is now unconditional so controller_pack
    # always reflects the real lumen, regardless of insertion_max_m.
    from dataclasses import replace as _dc_replace

    from proper_research.planning.planning_context import (
        make_design_config,
        make_robot_config,
    )
    from proper_research.simulation.simulations.controller_factory_joint_space import (
        build_controller,
    )
    from proper_research.simulation.simulations.initial_conditions import (
        make_initial_poses,
    )

    _, start_point, L0, dt = (
        make_initial_poses() if initial_poses is None else initial_poses
    )
    plant_key = "contact" if exp_cfg.model.plant_contact else "no_contact"
    plant_model = bundle.models[plant_key]
    jacobian_model = bundle.models[exp_cfg.model.jacobian_variant]
    effective_insertion_max_m = (
        float(insertion_max_m) if insertion_max_m is not None else make_robot_config().insertion_max_m
    )
    wide_robot_cfg = _dc_replace(make_robot_config(), insertion_max_m=effective_insertion_max_m)
    controller_pack = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant_model,
        jacobian_model=jacobian_model,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=make_design_config(),
        robot_cfg=wide_robot_cfg,
        jacobian_mode=jacobian_mode,
    )
    if insertion_max_m is not None:
        print(f"[vessel-context] insertion_max_m overridden to {1e3*insertion_max_m:.1f} mm "
              f"(library default 50.0 mm) -- controller_pack rebuilt with the real lumen wired in "
              f"(jacobian_mode={jacobian_mode!r}).")
    else:
        print(f"[vessel-context] controller_pack rebuilt with the real lumen wired in "
              f"(insertion_max_m left at the library default {1e3*effective_insertion_max_m:.1f} mm, "
              f"jacobian_mode={jacobian_mode!r}).")

    no_contact_model = bundle.models.get("no_contact")
    if no_contact_model is not None and getattr(no_contact_model, "lumen_query", None) is not None:
        raise RuntimeError(
            "bundle.models['no_contact'] unexpectedly has a lumen_query set -- "
            "refusing to proceed, since that would make the 'no contact' "
            "condition secretly contact-aware. Check build_model_bundle."
        )

    # Keep the bundle's own bookkeeping fields consistent for anything that
    # reads bundle.lumen_C/lumen_R directly (diagnostics, plotting, ...).
    # ModelBundle is frozen; build a fresh instance rather than mutate in place.
    bundle = dataclasses.replace(bundle, lumen_C=lumen_C, lumen_R=lumen_R)

    plant_key = "contact" if exp_cfg.model.plant_contact else "no_contact"
    jacobian_key = exp_cfg.model.jacobian_variant
    print(
        f"[vessel-context] real lumen wired into bundle.models['contact'] "
        f"({lumen_C.shape[0]} samples, radius {1e3*lumen_R.min():.2f}-{1e3*lumen_R.max():.2f} mm). "
        f"plant model = '{plant_key}', jacobian model = '{jacobian_key}' "
        f"(both see the real wall iff they are 'contact')."
    )

    return exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance


# NOTE: an earlier version of this module had a
# compute_wall_avoidance_geometry() that precomputed a per-reference-sample
# outward normal from the vessel centreline to the reference path. That is
# degenerate: the reference path IS the centreline (what the offline planner
# tracks), so the distance -- and hence the normal -- is ~0 almost
# everywhere (confirmed empirically: 0/234 samples had a well-defined normal
# on the first real vessel plan). The wall-avoidance term now queries the
# LumenQuery LIVE, against the actually-measured beam position each tick
# (see BeamOutputMPCConfig.wall_avoidance_gain and
# simulate_time_parameterized_beam_output_mpc.py::_wall_avoidance_qp_terms) --
# the online controller is simply handed wall_avoidance_lumen_C/lumen_R
# (this module's own lumen_C/lumen_R) directly, no precomputation needed.
