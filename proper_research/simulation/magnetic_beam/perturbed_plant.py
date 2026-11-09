"""Perturbed-plant wrapper around the real quasi-static beam/magnet forward
model, for Batch 4's Monte-Carlo simulator.

This is a thin wrapper, not a new physics model: it calls
``proper_research.simulation.simulations.model_factory.build_model_bundle``
(the exact function that built plan B's own offline forward/Jacobian
models -- see ``proper_research/planning/planning_context.py:309-376``)
with plan B's own kwargs, and exposes a way to re-build the same model with
the three Batch-4 plant perturbations:

  * effective Young's modulus   scale in [0.85, 1.15]  (+-15%%)
  * source magnet moment        scale in [0.90, 1.10]  (+-10%%)
  * insertion length            scale in [0.98, 1.02]  (+-2%%)

Added 2026-11-02 for the Batch 4 perturbed-plant simulator (Round 7, K3).
Nothing in this file is imported by any existing hardware runner; it is
purely additive.


Round 9 (2026-11-02) finding: this wrapper's default pivot z was stale
by 23.06mm (fixed below, in build_perturbed_plan_b_model). After that fix,
model-vs-plan-B's-own-saved-achieved_position_m agrees to <0.002mm on
every axis for insertion <= ~47mm, but a SEPARATE, unresolved discrepancy
(growing to ~9.6mm in x/y, insertion-dependent, starting ~48mm insertion)
remains for longer insertions. Likely cause, not yet fixed: this wrapper
builds the model with a generic straight/wide-bore contact lumen (matching
planning_context.build_planning_context()'s own bare default), but plan
B's actual build goes through vessel_context.build_vessel_planning_context(),
which explicitly swaps the contact model's lumen_query for the REAL,
digitized vessel geometry after construction -- a swap this wrapper does
not perform. The onset point (~48mm insertion) is consistent with where
the beam begins actually contacting the real vessel wall; a generic,
much wider lumen would under-constrain the beam exactly past that point.
Not fixed here -- out of scope for the bounded Round 9 check that found
it. Batch 4 is parked pending this."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from proper_research.simulation.simulations import model_factory as _mf
from proper_research.simulation.simulations.model_factory import build_model_bundle
from proper_research.simulation.simulations.scenario import LumenConfig
from proper_research.simulation.simulations.initial_conditions import make_initial_poses
from proper_research.rig_calibration import SOURCE_DIPOLE_BODY_AXIS, BEAM_BASE_XYZ_M


# Matches planning_context.build_planning_context's own call to
# build_model_bundle exactly (planning_context.py:305-376) -- these are the
# kwargs that produced plan B's (and every other current vessel plan's) own
# forward/Jacobian models. NOT the vessel-wall/keep-out geometry used by the
# MPC's exclusion-sphere/height-band monitors (those are magnet-position
# checks against proper_research.rig_calibration constants, independent of
# this beam-contact model) -- this lumen_cfg is only the internal "does the
# beam touch the vessel wall" contact geometry for the Cosserat solve.
_PLAN_B_LUMEN_CFG = LumenConfig(
    length=0.06, n_pts=240, n_ref_pts=100, radius=0.05, ds_target=1.0e-3, bends=()
)
_PLAN_B_KWARGS = dict(
    plant_contact=True,
    source_dipole_body_axis=SOURCE_DIPOLE_BODY_AXIS,
    use_physical_stiffness=True,
    effective_youngs_modulus=1.60e6,
    use_bimaterial_beam=True,
    wire_youngs_modulus_pa=75.0e9,
    source_dipole_yaw_deg=0.0,
)


@dataclass(frozen=True)
class PlantPerturbation:
    modulus_scale: float = 1.0
    moment_scale: float = 1.0
    insertion_scale: float = 1.0

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "PlantPerturbation":
        return cls(
            modulus_scale=float(rng.uniform(0.85, 1.15)),
            moment_scale=float(rng.uniform(0.90, 1.10)),
            insertion_scale=float(rng.uniform(0.98, 1.02)),
        )


@contextlib.contextmanager
def _perturbed_magnet_moment(scale: float):
    """Temporarily scale model_factory's module-level mag_params.mag_epm.

    mag_params is a plain (non-frozen) dataclass instantiated once at
    model_factory import time (model_factory.py:35) and closed over by
    build_model_bundle via the module global -- there is no per-call kwarg
    for the source-magnet moment, so this is the only way to perturb it
    without editing model_factory.py itself.
    """
    original = _mf.mag_params.mag_epm
    try:
        _mf.mag_params.mag_epm = float(original) * float(scale)
        yield
    finally:
        _mf.mag_params.mag_epm = original


def build_perturbed_plan_b_model(
    perturbation: PlantPerturbation = PlantPerturbation(),
    *,
    L0_nominal: float | None = None,
    pivot_point=None,
):
    """Build plan B's own contact-aware forward model, with the three
    Batch-4 perturbations applied.

    Returns
    -------
    (forward_model, L0_perturbed, pivot_point)

    forward_model.solve(p7) -> SolveResult
        p7 = [x, y, z, rx, ry, rz, L]  (robot-base frame;
        forward_model.py:970-985). result.tip is the beam tip position,
        shape (3,). result.p is the full centerline, shape (3, N).
    """
    if pivot_point is None or L0_nominal is None:
        _pivot_point, _start_point, _L0, _dt = make_initial_poses()
        if pivot_point is None:
            # 2026-11-02 (Round 9): make_initial_poses()'s pivot z (-16.567mm)
            # predates the 2026-10-02 BEAM_BASE_XYZ_M Z recalibration and was
            # never updated -- a 23.06mm stale frame-origin offset. Confirmed
            # by direct test: using this stale z against the real plan's
            # live-FK-derived (current, correct) magnet pose produced a
            # ~21.5-21.9mm constant z gap against plan B's own saved
            # achieved_position_m at every early-path sample; replacing just
            # the z component with the current rig_calibration.BEAM_BASE_XYZ_M
            # value (keeping x/y from make_initial_poses(), which already
            # agree with BEAM_BASE_XYZ_M to sub-mm) closes that gap to <0.002mm
            # on every axis for insertion <= ~47mm. Does not fix a second,
            # separate, larger-insertion discrepancy (see module docstring
            # note below) caused by this wrapper's generic contact lumen, not
            # the pivot.
            _pivot_point = _pivot_point.copy()
            _pivot_point[2] = float(BEAM_BASE_XYZ_M[2])
            pivot_point = _pivot_point
        if L0_nominal is None:
            L0_nominal = _L0

    L0_perturbed = float(L0_nominal) * float(perturbation.insertion_scale)
    effective_E = _PLAN_B_KWARGS["effective_youngs_modulus"] * float(perturbation.modulus_scale)

    kwargs = dict(_PLAN_B_KWARGS)
    kwargs["effective_youngs_modulus"] = effective_E

    with _perturbed_magnet_moment(perturbation.moment_scale):
        bundle = build_model_bundle(
            pivot_point=pivot_point,
            L0=L0_perturbed,
            lumen_cfg=_PLAN_B_LUMEN_CFG,
            lumen_pivot_point=pivot_point,
            **kwargs,
        )

    model = bundle.models["plant"]
    return model, L0_perturbed, pivot_point
