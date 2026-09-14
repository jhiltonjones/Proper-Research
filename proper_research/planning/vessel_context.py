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
):
    """Like ``planning_context.build_planning_context()``, but the
    contact-aware model's lumen is the REAL digitized vessel geometry from
    ``lumen_file`` (a ``vessel_lumen_robot_frame.json``-format file, robot
    frame R), not the permissive placeholder.

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

    kwargs: dict[str, Any] = {}
    if run_root is not None:
        kwargs["run_root"] = run_root
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

    if insertion_max_m is not None:
        # planning_context.make_robot_config() hardcodes insertion_max_m=0.05
        # (50mm) -- the project's usual advancer travel limit. This vessel's
        # own centreline can genuinely need more (confirmed: reaching this
        # vessel's true end needs ~64.5mm of insertion via the same
        # insertion-synchronised tracking that worked for the first ~50mm).
        # Rebuild controller_pack with a wider insertion ceiling rather than
        # touching build_planning_context()'s shared default, which every
        # free-space shape this session also relies on.
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

        _, start_point, L0, dt = make_initial_poses()
        plant_key = "contact" if exp_cfg.model.plant_contact else "no_contact"
        plant_model = bundle.models[plant_key]
        jacobian_model = bundle.models[exp_cfg.model.jacobian_variant]
        wide_robot_cfg = _dc_replace(make_robot_config(), insertion_max_m=float(insertion_max_m))
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
        )
        print(f"[vessel-context] insertion_max_m overridden to {1e3*insertion_max_m:.1f} mm "
              f"(library default 50.0 mm) -- controller_pack rebuilt.")

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
