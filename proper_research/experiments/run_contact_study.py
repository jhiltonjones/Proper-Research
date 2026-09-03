"""The contact study: run it, in three stages of increasing cost.

The question
------------
Is contact a **disturbance** (an additive offset a good observer absorbs) or is
it **model** (it changes the input-output sensitivity, so no observer can
substitute for it)?  The answer decides whether a neurovascular controller may
use a contact-free Jacobian and lean on its disturbance estimator, or must
carry contact in the model it linearises.

The design
----------
Four factors.  Only one of them is allowed to move at a time.

============  ===========================================================
factor        levels
============  ===========================================================
anatomy       lumen radius scaled down toward neurovascular calibre
Jacobian      ``contact`` vs ``no_contact``  (**control side only**)
controller    naive inverse Jacobian / MPC LTI / MPC LTV offline / MPC SQP
initial cond  the planned start, plus seeded perturbations of it
============  ===========================================================

**The plant is contact-enabled in every single cell.**  Nothing in this study
ever simulates a beam that does not touch the wall.  The Jacobian factor changes
what the *controller believes*, never what happens.  A result that came from
also disabling contact in the plant would be an artefact, and the runner refuses
to produce one: :func:`_assert_plant_contact` checks each bundle before use.

Why the anatomy sweep scales the radius and not the tortuosity
--------------------------------------------------------------
This is the design decision that makes the dose-response interpretable, and it
is worth being stubborn about.

Scaling the **radius** with the centreline held fixed leaves the desired tip
path unchanged, so *the same planned trajectory is valid on every anatomy*.
One reference, one set of limits, one initial condition; the only thing that
differs across the sweep is how hard the beam is pressed against the wall.  That
is a controlled dose.

Scaling the **tortuosity** changes the centreline, so the plan changes, so the
configurations differ, so the Jacobians differ, so the joint-limit margins
differ — and a penalty that grew across such a sweep could be blamed on any of
them.  Tortuosity is available (``--tortuosity-scales``) because reviewers ask
for it, but it requires a re-planned reference per anatomy and the runner marks
those cells ``confounded=1`` in the index so that no figure quietly mixes them
with the controlled arm.

The three stages
----------------
``--stage offline``   No simulation.  The two Jacobians at the reference states
                      on each anatomy: relative gap, principal angles, gain
                      ratio, engagement.  Minutes.  **Run this first** — if the
                      two Jacobians agree everywhere, there is nothing to find
                      and the expensive stages would only measure noise.

``--stage closed``    The factorial.  Every cell writes the same 100-column CSV
                      the existing analysis and the replay viewer already read.

``--stage observer``  The rebuttal stage.  On the tightest anatomy, with the
                      contact-free Jacobian, sweep ``disturbance_filter_alpha``
                      across its whole admissible range.  If the contact-free
                      penalty survives the best alpha, "you tuned the observer
                      badly" is no longer available as an explanation.

``--stage all``       All three, in order.

Usage
-----
    python run_contact_study.py --self-test
    python run_contact_study.py --stage offline --output-dir results/contact_study
    python run_contact_study.py --stage closed  --output-dir results/contact_study
    python run_contact_study.py --stage observer --output-dir results/contact_study
    python analyse_contact_study.py results/contact_study
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

Array = np.ndarray

# --------------------------------------------------------------------------
# sibling imports: analysis/ and controllers/ may sit beside or above us
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent / "analysis",
                   _HERE.parent / "controllers", _HERE / "analysis",
                   _HERE / "controllers"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import contact_evidence as EV  # noqa: E402


def _load(name: str) -> ModuleType:
    for module_name in (
        f"proper_research.simulation.analysis.{name}",
        f"proper_research.simulation.controllers.{name}",
        f"analysis.{name}",
        f"controllers.{name}",
        name,
    ):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    for root in (_HERE, _HERE.parent, _HERE.parent / "analysis",
                 _HERE.parent / "controllers"):
        path = root / f"{name}.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"_study_{name}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
    raise ModuleNotFoundError(
        f"Could not import {name}. Put the analysis and controllers packages "
        "beside this file, or make the project package importable."
    )


CONTROLLERS = (
    "naive_inverse_jacobian",
    "mpc_lti",
    "mpc_ltv_offline",
    "mpc_ltv_sqp_online",
)
JACOBIAN_MODELS = ("no_contact", "contact")


# ==========================================================================
# anatomy
# ==========================================================================
@dataclass(frozen=True)
class AnatomySpec:
    """One level of the anatomy factor.

    ``radius_scale`` multiplies the lumen radius; ``tortuosity_scale``
    multiplies every bend angle.  ``confounded`` records whether this level
    changed the centreline — and therefore the plan — so the analyser can keep
    it out of the controlled dose-response.
    """

    name: str
    radius_scale: float = 1.0
    tortuosity_scale: float = 1.0
    reference_dir: Path | None = None

    @property
    def confounded(self) -> bool:
        return abs(self.tortuosity_scale - 1.0) > 1e-12

    def as_dict(self) -> dict[str, Any]:
        return {
            "anatomy": self.name,
            "radius_scale": self.radius_scale,
            "tortuosity_scale": self.tortuosity_scale,
            "confounded": int(self.confounded),
            "reference_dir": None if self.reference_dir is None else str(self.reference_dir),
        }


def default_anatomies(
    radius_scales: Sequence[float], tortuosity_scales: Sequence[float]
) -> list[AnatomySpec]:
    """The controlled arm first, then any confounded tortuosity arm."""
    specs = [
        AnatomySpec(name=f"radius_x{scale:g}".replace(".", "p"), radius_scale=float(scale))
        for scale in radius_scales
    ]
    specs += [
        AnatomySpec(
            name=f"tortuosity_x{scale:g}".replace(".", "p"),
            tortuosity_scale=float(scale),
        )
        for scale in tortuosity_scales
        if abs(float(scale) - 1.0) > 1e-12
    ]
    return specs


def scale_lumen_config(lumen_cfg: Any, spec: AnatomySpec) -> Any:
    """Derive an anatomy from the project's own lumen config.

    Deliberately a *transformation of the user's configuration* rather than a
    freshly authored geometry: every field this study does not name keeps
    whatever the project set, so a cell differs from the baseline in exactly
    the way its name says and in no other way.
    """
    changes: dict[str, Any] = {}
    if abs(spec.radius_scale - 1.0) > 1e-12:
        radius = getattr(lumen_cfg, "radius", None)
        if radius is None:
            raise AttributeError(
                "The lumen config has no 'radius' field, so the radius sweep "
                "cannot be applied. Pass --anatomy-json to describe the levels "
                "explicitly, or run only the tortuosity arm."
            )
        radius = np.asarray(radius, dtype=float)
        changes["radius"] = (
            float(radius * spec.radius_scale) if radius.ndim == 0
            else radius * spec.radius_scale
        )
    if spec.confounded:
        bends = getattr(lumen_cfg, "bends", None)
        if not bends:
            raise AttributeError(
                "The lumen config has no 'bends', so the tortuosity sweep "
                "cannot be applied."
            )
        scaled = []
        for bend in bends:
            angle = float(getattr(bend, "bend_angle_rad"))
            scaled.append(replace(bend, bend_angle_rad=angle * spec.tortuosity_scale))
        changes["bends"] = tuple(scaled)
    if not changes:
        return lumen_cfg
    try:
        return replace(lumen_cfg, **changes)
    except TypeError as error:
        raise TypeError(
            f"Could not derive anatomy {spec.name!r} from the lumen config: {error}\n"
            "dataclasses.replace() needs LumenConfig to be a dataclass whose "
            "fields include the ones being scaled. If yours is not, build the "
            "levels yourself and pass them with --anatomy-json."
        ) from error


def _assert_plant_contact(bundle: Any, where: str) -> None:
    """The plant must be in contact in every cell. Refuse to run otherwise.

    This is the one assertion that protects the study's central claim. If the
    plant were ever contact-free, a 'contact matters' result would be an
    artefact of comparing two contact-free things.
    """
    plant = bundle.models.get("plant") if hasattr(bundle, "models") else None
    if plant is None:
        raise KeyError(f"{where}: the bundle has no 'plant' model.")
    contact_cfg = getattr(plant, "contact_cfg", None)
    enabled = getattr(contact_cfg, "enabled", None)
    if enabled is False:
        raise RuntimeError(
            f"{where}: the plant has contact DISABLED. Every cell of this study "
            "must simulate a contacting beam; only the controller's Jacobian is "
            "allowed to be contact-free. Rebuild the bundle with "
            "plant_contact=True."
        )
    if getattr(plant, "lumen_query", None) is None and enabled:
        raise RuntimeError(
            f"{where}: the plant has contact enabled but no lumen query, so no "
            "contact can occur. This is not a valid plant for this study."
        )


# ==========================================================================
# project context
# ==========================================================================
def resolve_planning_context(context_cfg: Any, controller_pack: Any) -> dict[str, Any]:
    """Find the experiment, design and robot configs, whatever shape the project uses.

    Two layouts exist in the wild and this has to cope with both.

    Some projects return a *context object* carrying ``.experiment``, ``.design``
    and ``.robot``.  This project does something else: ``build_planning_context``
    returns the ``ExperimentConfig`` directly and builds the design and robot
    configs inline, passing them straight into ``build_controller`` -- so they
    are not on the returned tuple at all.  But the module that builds them
    exports the factories, and calling those factories is not a guess: it is the
    *same call* ``build_planning_context`` makes, so the study gets exactly the
    configuration the planner used, and the two stay in step if either changes.

    ``robot_cfg`` is resolved as strictly as ``design_cfg``, and that is
    deliberate.  It carries ``q_seed_rad``, which chooses the IK branch.  Falling
    back to a default seed could land the arm in a different elbow/wrist
    configuration from the one the planner solved for -- the study would then be
    comparing Jacobians on a robot pose the reference never visits, and nothing
    downstream would flag it.  Silent and severe, so it is an error rather than
    a warning.
    """
    exp_cfg = getattr(context_cfg, "experiment", context_cfg)
    design_cfg = getattr(context_cfg, "design", None)
    robot_cfg = getattr(context_cfg, "robot", None)

    resolved_from = "the context object"
    if design_cfg is None or robot_cfg is None:
        module = None
        for name in (
            "proper_research.planning.planning_context",
            "planning_context",
        ):
            try:
                module = importlib.import_module(name)
                break
            except ModuleNotFoundError:
                continue
        if module is not None:
            if design_cfg is None:
                factory = getattr(module, "make_design_config", None)
                if callable(factory):
                    design_cfg = factory()
                    resolved_from = f"{module.__name__}.make_design_config()"
            if robot_cfg is None:
                factory = getattr(module, "make_robot_config", None)
                if callable(factory):
                    robot_cfg = factory()

    missing = [
        name for name, value in (("design_cfg", design_cfg), ("robot_cfg", robot_cfg))
        if value is None
    ]
    if missing:
        raise AttributeError(
            f"Could not resolve {' and '.join(missing)} for the study.\n\n"
            "Looked for '.design' / '.robot' on build_planning_context()'s first "
            "return value, then for make_design_config() / make_robot_config() in "
            "planning_context.\n\n"
            "These must be the SAME objects the planner used -- design_cfg sets "
            "the weights and trust regions, and robot_cfg carries q_seed_rad, "
            "which selects the IK branch. A default would silently put the arm "
            "in a different configuration from the reference. Either export "
            "those factories from planning_context, or set base_context "
            "explicitly in main()."
        )
    return {
        "exp_cfg": exp_cfg,
        "design_cfg": design_cfg,
        "robot_cfg": robot_cfg,
        "resolved_from": resolved_from,
    }


@dataclass
class AnatomyContext:
    """Everything one anatomy needs to run a cell."""

    spec: AnatomySpec
    bundle: Any
    controller_pack: Any
    reference: Any
    contact_provider: Any
    contact_free_provider: Any


def build_anatomy_context(
    *,
    spec: AnatomySpec,
    base_context: dict[str, Any],
    providers_module: ModuleType,
    base_module: ModuleType,
    require_planned_beam_feasible: bool = True,
    shared_controller_pack: Any | None = None,
) -> AnatomyContext:
    """Rebuild the model bundle and controller pack for one anatomy.

    Mirrors the construction in ``offline_inverse_configuration_head_exclusion``
    so the models this study drives are the same models the planner drove.

    ``shared_controller_pack`` skips the per-anatomy ``build_controller`` call.
    Only pass it for the **offline stage**, and the reason it is safe there is
    narrow: stage 1 reads nothing from the pack but robot kinematics -- the DH
    parameters and the flange-to-magnet transform, which the lumen cannot
    change. The two Jacobian providers are built fresh around this anatomy's own
    ``bundle.models``, so the contact physics is still per-anatomy.

    Do not pass it for the closed-loop stages. There the pack supplies
    ``forward6d_plant`` and the plant diagnostic adapter, both of which wrap the
    plant model and therefore *do* depend on the lumen. Reusing the baseline
    pack would step every anatomy against the baseline vessel and the study
    would quietly measure nothing.
    """
    from proper_research.simulation.simulations.controller_factory_joint_space import (
        build_controller,
    )
    from proper_research.simulation.simulations.initial_conditions import (
        make_initial_poses,
    )
    from proper_research.simulation.simulations.model_factory import build_model_bundle

    exp_cfg = base_context["exp_cfg"]
    design_cfg = base_context["design_cfg"]
    robot_cfg = base_context.get("robot_cfg")

    lumen_cfg = scale_lumen_config(exp_cfg.lumen, spec)
    pivot_point, start_point, L0, dt = make_initial_poses()
    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=lumen_cfg,
        plant_contact=True,  # never negotiable
    )
    _assert_plant_contact(bundle, f"anatomy {spec.name!r}")

    if shared_controller_pack is not None:
        controller_pack = shared_controller_pack
    else:
        # Mirror the project's own choice of jacobian_variant rather than
        # hard-coding one. The study never reads the pack's Jacobian -- it
        # builds its own declared providers below -- so this exists purely to
        # keep the pack identical to the one the planner built, removing a
        # difference that could otherwise go unnoticed.
        variant = getattr(
            getattr(exp_cfg, "model", None), "jacobian_variant", "no_contact"
        )
        kwargs = dict(
            start_point=start_point,
            L0=L0,
            dt=dt,
            plant_model=bundle.models["plant"],
            jacobian_model=bundle.models[variant],
            lumen_C=bundle.lumen_C,
            lumen_R=bundle.lumen_R,
            run_cfg=exp_cfg.controller,
            design_cfg=design_cfg,
        )
        if robot_cfg is not None:
            kwargs["robot_cfg"] = robot_cfg
        controller_pack = build_controller(**kwargs)

    reference_dir = spec.reference_dir or base_context["reference_dir"]
    if spec.confounded and spec.reference_dir is None:
        raise ValueError(
            f"Anatomy {spec.name!r} changes the centreline, so the baseline "
            "reference is no longer the right plan for it. Re-plan on this "
            "anatomy and pass its directory via --anatomy-json, or drop the "
            "tortuosity arm. Reusing the baseline plan here would measure the "
            "stale plan, not the anatomy."
        )
    reference = base_module.load_configuration_reference(
        reference_dir, require_planned_beam_feasible=require_planned_beam_feasible
    )

    contact, contact_free = providers_module.contact_and_contact_free(
        bundle=bundle, controller_pack=controller_pack
    )
    return AnatomyContext(
        spec=spec,
        bundle=bundle,
        controller_pack=controller_pack,
        reference=reference,
        contact_provider=contact,
        contact_free_provider=contact_free,
    )


# ==========================================================================
# stage 1 — offline: how the two Jacobians differ
# ==========================================================================
def _provenance(provider: Any) -> dict[str, Any]:
    """A provider's declared provenance, or an honest note that it has none."""
    describe = getattr(provider, "describe", None)
    if callable(describe):
        return dict(describe())
    return {"name": "undeclared", "source": repr(provider)}


def run_offline_stage(
    *,
    contexts: Sequence[AnatomyContext],
    output_dir: Path,
    stride: int,
    common: ModuleType,
) -> dict[str, Any]:
    """No plant stepping, no controller. The cheapest falsification available.

    If this stage finds the two Jacobians agreeing to a fraction of a percent
    on every anatomy, stop: the structural hypothesis is already in trouble and
    the expensive stages would measure nothing but solver noise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_anatomy: dict[str, Any] = {}

    for context in contexts:
        states = np.asarray(context.reference.state, dtype=float)
        sampled = states[:: max(1, int(stride))]
        started = time.perf_counter()
        divergence = EV.jacobian_divergence(
            contact_provider=context.contact_provider,
            contact_free_provider=context.contact_free_provider,
            states=sampled,
            contact_model=context.bundle.models.get("contact"),
        )
        summary = divergence.summary()
        summary["anatomy"] = context.spec.name
        summary["elapsed_s"] = float(time.perf_counter() - started)
        summary["contact_provider"] = _provenance(context.contact_provider)
        summary["contact_free_provider"] = _provenance(context.contact_free_provider)
        per_anatomy[context.spec.name] = summary

        # The per-state table: this is what the dose-response figure plots.
        for index in range(sampled.shape[0]):
            row = {
                **context.spec.as_dict(),
                "sample": int(index * max(1, int(stride))),
                "frobenius_relative": float(divergence.frobenius_relative[index]),
                "spectral_relative": float(divergence.spectral_relative[index]),
                "principal_angle_1_deg": float(divergence.principal_angles_deg[index, 0]),
                "principal_angle_2_deg": float(divergence.principal_angles_deg[index, 1]),
                "principal_angle_3_deg": float(divergence.principal_angles_deg[index, 2]),
                "gain_ratio_min": float(divergence.gain_ratio[index, -1]),
                "gain_ratio_max": float(divergence.gain_ratio[index, 0]),
                "condition_contact": float(divergence.condition_contact[index]),
                "condition_contact_free": float(divergence.condition_contact_free[index]),
            }
            for column in range(7):
                row[f"per_actuator_relative_{column + 1}"] = float(
                    divergence.per_actuator_relative[index, column]
                )
            for key, values in divergence.engagement.items():
                row[key] = float(values[index]) if index < len(values) else math.nan
            rows.append(row)

        largest = summary["largest_principal_angle_deg"]["max"]
        median = summary["frobenius_relative"]["median"]
        print(
            f"[offline] {context.spec.name}: relative gap median {median:.4f}, "
            f"largest principal angle {largest:.3f} deg",
            flush=True,
        )

    if rows:
        header = list(rows[0].keys())
        common.write_csv(
            output_dir / "jacobian_divergence.csv",
            header,
            [[row.get(key, "") for key in header] for row in rows],
        )
    (output_dir / "jacobian_divergence_summary.json").write_text(
        json.dumps(common.jsonable(per_anatomy), indent=2), encoding="utf-8"
    )
    return per_anatomy


# ==========================================================================
# stage 2 — the closed-loop factorial
# ==========================================================================
@dataclass
class CellSpec:
    anatomy: str
    jacobian_model: str
    controller: str
    initial_condition: int
    disturbance_alpha: float = 0.0
    tag: str = ""

    def directory_name(self) -> str:
        stem = (
            f"{self.anatomy}__{self.jacobian_model}__{self.controller}"
            f"__ic{self.initial_condition}"
        )
        if self.tag:
            stem += f"__{self.tag}"
        return stem

    def as_dict(self) -> dict[str, Any]:
        return {
            "anatomy": self.anatomy,
            "jacobian_model": self.jacobian_model,
            "controller": self.controller,
            "initial_condition": self.initial_condition,
            "disturbance_alpha": self.disturbance_alpha,
            "tag": self.tag,
        }


def initial_states(
    *, reference: Any, count: int, joint_spread_deg: float, insertion_spread_mm: float
) -> list[Array]:
    """The planned start, then seeded perturbations of it.

    Seeded, not random: the same physical initial condition must be handed to
    every controller and every Jacobian model, or the comparison is measuring
    luck. Index 0 is always the planned start exactly.
    """
    base = np.asarray(reference.state, dtype=float)[0]
    states = [base.copy()]
    generator = np.random.default_rng(20240517)
    for _ in range(max(0, int(count) - 1)):
        offset = np.concatenate(
            (
                np.radians(joint_spread_deg) * generator.uniform(-1.0, 1.0, 6),
                [1e-3 * insertion_spread_mm * generator.uniform(-1.0, 1.0)],
            )
        )
        states.append(base + offset)
    return states


def _state_matrix(records: Sequence[dict[str, Any]], suffix: str) -> Array:
    joints = np.asarray(
        [[float(row[f"q{i}_{suffix}"]) for i in range(1, 7)] for row in records],
        dtype=float,
    )
    insertion = np.asarray(
        [[float(row[f"insertion_{suffix}"])] for row in records], dtype=float
    )
    return np.hstack([joints, insertion])


def _command_matrix(records: Sequence[dict[str, Any]]) -> Array:
    joints = np.asarray(
        [[float(row[f"qd{i}_command"]) for i in range(1, 7)] for row in records],
        dtype=float,
    )
    insertion = np.asarray(
        [[float(row["insertion_rate_command"])] for row in records], dtype=float
    )
    return np.hstack([joints, insertion])


def _residual_matrix(records: Sequence[dict[str, Any]], prefix: str) -> Array:
    return np.asarray(
        [[float(row.get(f"{prefix}{axis}_m", math.nan)) for axis in "xyz"] for row in records],
        dtype=float,
    )


def diagnose_cell(
    *,
    records: Sequence[dict[str, Any]],
    context: AnatomyContext,
    jacobian_gap_stride: int = 25,
) -> dict[str, Any]:
    """The per-run evidence: is this run's 'disturbance' the missing model term?

    Two details here are not cosmetic.

    **The regression uses the *instantaneous* residual, not the filtered one.**
    With ``disturbance_filter_alpha > 0`` the recorded ``d_hat`` is a lagged
    version of the residual, and lag alone would depress R^2 for reasons that
    have nothing to do with the hypothesis under test. The instantaneous
    residual is the quantity the algebra is about.

    **The Jacobian gap is measured on the states this run actually visited**,
    not at the reference. If the run drifted, the relevant gap is the gap where
    it drifted to.
    """
    states = _state_matrix(records, "actual")
    reference_states = _state_matrix(records, "reference")
    commands = _command_matrix(records)

    sampled = states[:: max(1, int(jacobian_gap_stride))]
    gaps = []
    for state in sampled:
        contact = np.asarray(context.contact_provider(state), dtype=float).reshape(3, 7)
        free = np.asarray(context.contact_free_provider(state), dtype=float).reshape(3, 7)
        gaps.append(contact - free)
    jacobian_gap = np.mean(np.stack(gaps, axis=0), axis=0) if gaps else None

    # How wrong is the contact-free model about the motion actually commanded?
    directional = []
    for state, command in zip(sampled, commands[:: max(1, int(jacobian_gap_stride))]):
        if float(np.linalg.norm(command)) <= 1e-12:
            continue
        contact = np.asarray(context.contact_provider(state), dtype=float).reshape(3, 7)
        free = np.asarray(context.contact_free_provider(state), dtype=float).reshape(3, 7)
        directional.append(EV.directional_prediction_error(contact, free, command))
    directional_array = np.asarray(directional, dtype=float)
    directional_array = directional_array[np.isfinite(directional_array)]

    instantaneous = _residual_matrix(records, "beam_residual_instantaneous_")
    filtered = _residual_matrix(records, "beam_residual_")
    residual = instantaneous if np.any(np.isfinite(instantaneous)) else filtered

    verdict: dict[str, Any] | None = None
    if np.any(np.isfinite(residual)):
        mask = np.all(np.isfinite(residual), axis=1)
        if int(np.sum(mask)) >= 20:
            verdict = EV.disturbance_or_model(
                estimated_residual=residual[mask],
                state=states[mask],
                reference_state=reference_states[mask],
                jacobian_gap=jacobian_gap,
            ).as_dict()

    return {
        "jacobian_gap": None if jacobian_gap is None else jacobian_gap.tolist(),
        "jacobian_gap_norm": (
            None if jacobian_gap is None else float(np.linalg.norm(jacobian_gap))
        ),
        "directional_prediction_error_mean": (
            float(np.mean(directional_array)) if directional_array.size else math.nan
        ),
        "directional_prediction_error_p95": (
            float(np.percentile(directional_array, 95.0)) if directional_array.size else math.nan
        ),
        "residual_source": (
            "instantaneous" if np.any(np.isfinite(instantaneous)) else "filtered"
        ),
        "disturbance_verdict": verdict,
        "engagement": EV.contact_engagement(context.bundle.models.get("contact")),
    }


@dataclass
class StudySettings:
    """Everything held fixed across every cell.

    Grouped into one object on purpose: a setting that lives here is a setting
    the factorial cannot vary by accident. Anything a cell is allowed to change
    lives in :class:`CellSpec` instead.
    """

    horizon: int = 15
    terminal_hold_steps: int = 15
    position_tolerance_mm: float = 1.5
    tangent_tolerance_deg: float = 90.0
    beam_position_scale_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    beam_position_weight: float = 1.0
    beam_terminal_multiplier: float = 20.0
    state_tracking_weight: float = 1.0e-2
    input_tracking_weight: float = 1.0e-3
    input_increment_weight: float = 1.0e-3
    joint_acceleration_limit: float = 0.5
    insertion_acceleration_limit: float = 0.02
    velocity_safety_factor: float = 0.8
    acceleration_safety_factor: float = 0.8
    lti_freeze_index: int = 0
    sqp_inner_iterations: int = 1
    sqp_relinearise_horizon: bool = False
    resolved_rate_gain: float = 1.0
    resolved_rate_damping: float = 1.0e-3
    nullspace_gain: float = 1.0
    solver: str = "auto"
    max_control_steps: int | None = None
    stop_on_qp_failure: bool = False
    verbose: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def build_cell_environment(
    *,
    context: AnatomyContext,
    jacobian_model: str,
    settings: StudySettings,
    disturbance_alpha: float,
    base_module: ModuleType,
    beam_module: ModuleType,
    harness: ModuleType,
    variants_module: ModuleType,
    inverse_module: ModuleType,
) -> dict[str, Any]:
    """Build the four controllers for one (anatomy, Jacobian model) pair.

    Built once per pair and reused across controllers and initial conditions:
    the four rungs must share one config object, or a difference in limits
    could masquerade as a difference in control law.
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
    mpc_config = replace(
        mpc_config,
        state_tracking_weight=float(settings.state_tracking_weight),
        input_tracking_weight=float(settings.input_tracking_weight),
        input_increment_weight=float(settings.input_increment_weight),
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=tuple(
            1e-3 * np.asarray(settings.beam_position_scale_mm, dtype=float)
        ),
        position_tracking_weight=float(settings.beam_position_weight),
        terminal_weight_multiplier=float(settings.beam_terminal_multiplier),
        disturbance_filter_alpha=float(disturbance_alpha),
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
    schedule = variants_module.precompute_schedule(
        reference=reference, jacobian_provider=provider
    )
    mpc_variants = variants_module.build_all_mpc_variants(
        reference=reference,
        config=mpc_config,
        beam_config=beam_config,
        jacobian_provider=provider,
        schedule=schedule,
        freeze_index=int(settings.lti_freeze_index),
        sqp_inner_iterations=int(settings.sqp_inner_iterations),
        relinearise_horizon=bool(settings.sqp_relinearise_horizon),
    )

    def make(name: str) -> Any:
        if name == "naive_inverse_jacobian":
            controller = inverse_module.build_inverse_jacobian_controller(
                reference=reference,
                jacobian_provider=provider,
                mpc_config=mpc_config,
                position_gain=float(settings.resolved_rate_gain),
                damping=float(settings.resolved_rate_damping),
                nullspace_gain=float(settings.nullspace_gain),
            )
            return harness.ControllerAdapter(
                name, controller.description, controller
            )
        controller = mpc_variants[name]
        return harness.ControllerAdapter(
            name, str(controller.variant_description), controller
        )

    forward6d = context.controller_pack.get("forward6d_plant")
    if forward6d is None:
        raise KeyError("controller_pack has no 'forward6d_plant'.")
    adapter = context.controller_pack.get("plant_diagnostic_joint_adapter")

    def beam_snapshot(state: Array, commit: bool) -> tuple[Array, Array]:
        return base_module._beam_snapshot(
            forward6d=forward6d, state=state, commit=commit
        )

    def magnet_pose(state: Array) -> tuple[Array, Array]:
        return base_module._magnet_pose(
            controller_pack=context.controller_pack, state=state
        )

    reset_plant = getattr(adapter, "reset_to_initial_baseline", None)
    if reset_plant is None:
        forward_adapter = getattr(
            getattr(adapter, "beam_output_fn", None), "forward_adapter", None
        )
        reset_plant = getattr(forward_adapter, "reset_to_initial_baseline", None)

    return {
        "make": make,
        "loop_config": loop_config,
        "beam_snapshot": beam_snapshot,
        "magnet_pose": magnet_pose,
        "reset_plant": reset_plant,
        "provider": provider,
        "mpc_config": mpc_config,
        "beam_config": beam_config,
    }


def run_cells(
    *,
    cells: Sequence[CellSpec],
    contexts: dict[str, AnatomyContext],
    settings: StudySettings,
    output_dir: Path,
    modules: dict[str, ModuleType],
    initial_condition_states: dict[str, list[Array]],
    common: ModuleType,
) -> list[dict[str, Any]]:
    """Run the factorial, one cell at a time, writing as we go.

    Written incrementally rather than at the end: a long study that dies in
    cell 60 should still leave 59 usable cells and an index that says so.
    """
    harness = modules["harness"]
    environments: dict[tuple[str, str, float], dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    index_path = output_dir / "study_index.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    for number, cell in enumerate(cells, start=1):
        context = contexts[cell.anatomy]
        key = (cell.anatomy, cell.jacobian_model, cell.disturbance_alpha)
        if key not in environments:
            environments[key] = build_cell_environment(
                context=context,
                jacobian_model=cell.jacobian_model,
                settings=settings,
                disturbance_alpha=cell.disturbance_alpha,
                base_module=modules["base"],
                beam_module=modules["beam"],
                harness=harness,
                variants_module=modules["variants"],
                inverse_module=modules["inverse"],
            )
        environment = environments[key]
        controller = environment["make"](cell.controller)
        state = initial_condition_states[cell.anatomy][cell.initial_condition]

        print(
            f"[cell {number}/{len(cells)}] {cell.directory_name()}",
            flush=True,
        )
        started = time.perf_counter()
        failure = ""
        try:
            result = harness.run_closed_loop(
                controller=controller,
                reference=context.reference,
                beam_snapshot=environment["beam_snapshot"],
                magnet_pose=environment["magnet_pose"],
                reset_plant=environment["reset_plant"],
                config=environment["loop_config"],
                initial_state=state,
                verbose=settings.verbose,
            )
        except Exception as error:  # one bad cell must not lose the study
            failure = f"{type(error).__name__}: {error}"
            print(f"    FAILED: {failure}", flush=True)
            index_rows.append(
                {
                    **cell.as_dict(),
                    **context.spec.as_dict(),
                    "status": "failed",
                    "failure": failure,
                }
            )
            _write_index(index_path, index_rows, common)
            continue

        cell_dir = output_dir / "cells" / cell.directory_name()
        harness.save_loop_result(result, cell_dir)
        diagnosis = diagnose_cell(records=result.records, context=context)
        (cell_dir / "contact_diagnosis.json").write_text(
            json.dumps(common.jsonable(diagnosis), indent=2), encoding="utf-8"
        )
        (cell_dir / "cell.json").write_text(
            json.dumps(
                common.jsonable(
                    {
                        **cell.as_dict(),
                        **context.spec.as_dict(),
                        "jacobian_provenance": environment["provider"].describe(),
                        "initial_state": np.asarray(state, dtype=float).tolist(),
                        "settings": settings.as_dict(),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        verdict = (diagnosis.get("disturbance_verdict") or {})
        row = {
            **cell.as_dict(),
            **context.spec.as_dict(),
            "status": "ok",
            "failure": "",
            "elapsed_s": float(time.perf_counter() - started),
            "directory": str(cell_dir.relative_to(output_dir)),
            "contact_free_jacobian": int(cell.jacobian_model == "no_contact"),
            **{
                name: result.summary[name]
                for name in (
                    "rms_beam_position_error_mm",
                    "p95_beam_position_error_mm",
                    "maximum_beam_position_error_mm",
                    "final_beam_position_error_mm",
                    "maximum_beam_tangent_error_deg",
                    "rms_joint_error_deg",
                    "beam_feasible_fraction",
                    "solve_failures",
                    "mean_solve_time_ms",
                    "p95_solve_time_ms",
                    "maximum_solve_time_ms",
                    "deadline_overruns",
                    "stopped_reason",
                )
            },
            "jacobian_gap_norm": diagnosis.get("jacobian_gap_norm"),
            "directional_prediction_error_mean": diagnosis.get(
                "directional_prediction_error_mean"
            ),
            "directional_prediction_error_p95": diagnosis.get(
                "directional_prediction_error_p95"
            ),
            "disturbance_verdict": verdict.get("verdict", ""),
            "disturbance_r_squared": verdict.get("r_squared_total", math.nan),
            "gap_recovery_relative_error": verdict.get(
                "gap_recovery_relative_error", math.nan
            ),
            "residual_autocorrelation_lag1": verdict.get("autocorrelation_lag1", math.nan),
            **{
                f"engagement_{key}": value
                for key, value in diagnosis.get("engagement", {}).items()
            },
        }
        index_rows.append(row)
        _write_index(index_path, index_rows, common)
        print(
            f"    rms {row['rms_beam_position_error_mm']:.4f} mm | "
            f"p95 {row['p95_beam_position_error_mm']:.4f} mm | "
            f"{row['elapsed_s']:.1f} s",
            flush=True,
        )
    return index_rows


def _write_index(path: Path, rows: Sequence[dict[str, Any]], common: ModuleType) -> None:
    if not rows:
        return
    header: list[str] = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)
    common.write_csv(
        path, header, [[row.get(key, "") for key in header] for row in rows]
    )


# ==========================================================================
# cell enumeration
# ==========================================================================
def enumerate_closed_loop_cells(
    *,
    anatomies: Sequence[AnatomySpec],
    jacobian_models: Sequence[str],
    controllers: Sequence[str],
    initial_conditions: int,
) -> list[CellSpec]:
    """The full factorial, ordered so a truncated run is still balanced.

    Initial condition is the outermost loop and 0 comes first, so stopping the
    study early leaves every (anatomy, Jacobian, controller) combination
    represented at the planned start rather than a lopsided subset.
    """
    cells: list[CellSpec] = []
    for condition in range(max(1, int(initial_conditions))):
        for anatomy in anatomies:
            for jacobian_model in jacobian_models:
                for controller in controllers:
                    cells.append(
                        CellSpec(
                            anatomy=anatomy.name,
                            jacobian_model=jacobian_model,
                            controller=controller,
                            initial_condition=condition,
                        )
                    )
    return cells


def enumerate_observer_cells(
    *,
    anatomy: AnatomySpec,
    controller: str,
    alphas: Sequence[float],
) -> list[CellSpec]:
    """The rebuttal sweep.

    The contact-aware cell at alpha = 0 is included as the ceiling: it is the
    number the contact-free controller would have to reach for "the observer
    can handle it" to be true. Without that line on the plot, a flat alpha
    sweep is just a flat line with nothing to be flat *relative to*.
    """
    cells = [
        CellSpec(
            anatomy=anatomy.name,
            jacobian_model="no_contact",
            controller=controller,
            initial_condition=0,
            disturbance_alpha=float(alpha),
            tag=f"alpha{alpha:g}".replace(".", "p"),
        )
        for alpha in alphas
    ]
    cells.append(
        CellSpec(
            anatomy=anatomy.name,
            jacobian_model="contact",
            controller=controller,
            initial_condition=0,
            disturbance_alpha=0.0,
            tag="ceiling",
        )
    )
    return cells


# ==========================================================================
# self-test — mocks only, no project imports
# ==========================================================================
def run_self_test() -> int:
    """Exercise the study's own logic on a mock whose answer is known.

    Two mocks, deliberately: one where contact really does bend the sensitivity
    (the study must find it) and one where contact is a pure additive offset
    (the study must NOT call that structural). A pipeline that only ever
    confirms is not evidence.
    """
    harness = _load("compare_controllers")
    common = _load("_stack_common")
    inverse_module = _load("inverse_jacobian_controller")
    providers_module = _load("beam_jacobian_providers")

    results: list[tuple[bool, str, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((bool(condition), name, detail))

    reference = harness._MockReference(samples=90, dt=0.01)

    class _MockBeam:
        """Tip map with three selectable contact modes.

        ``mode`` picks what contact does, and the three modes are the three
        hypotheses the study has to be able to tell apart:

        ``"structural"``
            Contact **changes which joint motions move the tip**. In free space
            joints 3 and 4 leave this mock's tip where it is; under contact the
            wall reaction makes them effective. That is what rotates the row
            space of ``J`` — equivalently, it moves the nullspace, so joint
            motions the controller believed were free are not.

            Note what this rules out. A row space is a 3-dimensional subspace
            of joint space; scaling a row, or mixing rows that already span it,
            leaves it untouched. Only a genuine change in *which directions*
            reach the tip rotates it, and no additive output disturbance can
            do that.

        ``"gain"``
            Contact stiffens the existing direction without redirecting it. The
            Jacobian changes in magnitude but its row space does not rotate.
            Still not a disturbance — an observer cannot fix a gain error —
            but the principal-angle test alone will not see it, which is
            exactly why the study reports magnitude and angle separately.

        ``"additive"``
            Contact is a constant offset. ``J`` is untouched. This is the
            disturbance hypothesis, and the study must NOT call it structural.
        """

        def __init__(self, strength: float, mode: str) -> None:
            self.strength = float(strength)
            self.mode = str(mode)
            self.contact_cfg = type("C", (), {"enabled": True, "use_in_jacobian": True})()
            self.lumen_query = object()

        def tip(self, state: Array) -> Array:
            q = np.asarray(state, dtype=float)
            base = np.array([0.30 * q[0], 0.20 * q[1], 5.0 * q[6]])
            base += np.array([0.04 * q[0] ** 3, 0.02 * q[1] ** 3, 0.0])
            wall = self.strength * np.tanh(6.0 * q[0])
            if self.mode == "structural":
                # Under contact, joints 3 and 4 reach the tip through the wall
                # reaction. In free space they do not appear in the map at all.
                base += np.array([
                    -0.5 * wall + 0.6 * self.strength * q[2],
                    wall + 0.9 * self.strength * q[3],
                    0.35 * wall,
                ])
            elif self.mode == "gain":
                base += np.array([wall, 0.0, 0.0])
            else:
                base += np.array([self.strength, 0.0, 0.0])
            return base

        def jacobian(self, state: Array, contact: bool) -> Array:
            q = np.asarray(state, dtype=float)
            matrix = np.zeros((3, 7))
            matrix[0, 0] = 0.30 + 0.12 * q[0] ** 2
            matrix[1, 1] = 0.20 + 0.06 * q[1] ** 2
            matrix[2, 6] = 5.0
            if contact:
                slope = 6.0 * self.strength / np.cosh(6.0 * q[0]) ** 2
                if self.mode == "structural":
                    matrix[0, 0] += -0.5 * slope
                    matrix[1, 0] += slope
                    matrix[2, 0] += 0.35 * slope
                    matrix[0, 2] += 0.6 * self.strength
                    matrix[1, 3] += 0.9 * self.strength
                elif self.mode == "gain":
                    matrix[0, 0] += slope
            return matrix

    def make_context(spec: AnatomySpec, mode: str) -> AnatomyContext:
        # Tighter lumen -> smaller radius scale -> more contact.
        strength = 0.010 * (1.0 / max(spec.radius_scale, 1e-6) - 1.0 + 0.2)
        beam = _MockBeam(strength, mode)
        bundle = type("B", (), {"models": {"plant": beam, "contact": beam,
                                           "no_contact": beam}})()
        return AnatomyContext(
            spec=spec,
            bundle=bundle,
            controller_pack={},
            reference=reference,
            contact_provider=lambda z, b=beam: b.jacobian(z, True),
            contact_free_provider=lambda z, b=beam: b.jacobian(z, False),
        )

    # ---- anatomy plumbing ------------------------------------------------
    specs = default_anatomies([1.0, 0.7, 0.5, 0.35], [1.3])
    check("radius levels are not marked confounded",
          all(not s.confounded for s in specs if s.name.startswith("radius")))
    check("a tortuosity level IS marked confounded",
          any(s.confounded for s in specs if s.name.startswith("tortuosity")))

    cells = enumerate_closed_loop_cells(
        anatomies=specs[:2], jacobian_models=JACOBIAN_MODELS,
        controllers=CONTROLLERS, initial_conditions=2,
    )
    check("the factorial has the right cell count", len(cells) == 2 * 2 * 4 * 2,
          f"{len(cells)} cells")
    check("cell directory names are unique",
          len({c.directory_name() for c in cells}) == len(cells))
    check("initial condition 0 comes first",
          all(c.initial_condition == 0 for c in cells[: len(cells) // 2]))

    # ---- the plant-contact guard ----------------------------------------
    off = type("B", (), {"models": {"plant": type("M", (), {
        "contact_cfg": type("C", (), {"enabled": False})(), "lumen_query": None})()}})()
    try:
        _assert_plant_contact(off, "guard test")
        check("a contact-free plant is refused", False, "no error raised")
    except RuntimeError as error:
        check("a contact-free plant is refused", "contact DISABLED" in str(error))

    # ---- stage 1 on both mocks ------------------------------------------
    output = Path(__file__).resolve().parent / "_self_test_study"
    structural_contexts = [make_context(spec, "structural") for spec in specs[:4]]
    additive_contexts = [make_context(spec, "additive") for spec in specs[:4]]
    gain_contexts = [make_context(spec, "gain") for spec in specs[:4]]

    offline = run_offline_stage(
        contexts=structural_contexts, output_dir=output / "offline_structural",
        stride=5, common=common,
    )
    tightest = offline[specs[3].name]
    check("stage 1 finds a rotating row space when contact is structural",
          tightest["largest_principal_angle_deg"]["max"] > 0.5,
          f"{tightest['largest_principal_angle_deg']['max']:.3f} deg")

    offline_flat = run_offline_stage(
        contexts=additive_contexts, output_dir=output / "offline_additive",
        stride=5, common=common,
    )
    flat = offline_flat[specs[3].name]
    check("stage 1 finds NO rotation when contact is a pure offset",
          flat["largest_principal_angle_deg"]["max"] < 1e-6,
          f"{flat['largest_principal_angle_deg']['max']:.3e} deg")
    check("stage 1 finds no magnitude gap either when contact is a pure offset",
          flat["frobenius_relative"]["max"] < 1e-12,
          f"{flat['frobenius_relative']['max']:.3e}")

    # The middle case, and the reason magnitude and angle are reported apart:
    # a contact that stiffens without redirecting changes the Jacobian but not
    # its row space. Still not a disturbance; still invisible to the angles.
    offline_gain = run_offline_stage(
        contexts=gain_contexts, output_dir=output / "offline_gain",
        stride=5, common=common,
    )
    gain = offline_gain[specs[3].name]
    check("a gain-only contact shows up in the magnitude gap",
          gain["frobenius_relative"]["median"] > 1e-3,
          f"median {gain['frobenius_relative']['median']:.4f}")
    check("a gain-only contact does NOT rotate the row space",
          gain["largest_principal_angle_deg"]["max"] < 1e-6,
          f"{gain['largest_principal_angle_deg']['max']:.3e} deg — magnitude and "
          "angle must be read together")

    # ---- stage 2, on the baseline controller only ------------------------
    loop_config = harness.LoopConfig(
        sample_period_s=reference.sample_period_s,
        velocity_limit=np.array([2.0] * 6 + [0.5]),
        acceleration_limit=np.array([20.0] * 6 + [5.0]),
        state_min=np.array([-3.0] * 6 + [-1.0]),
        state_max=np.array([3.0] * 6 + [1.0]),
        position_tolerance_m=1.5e-3,
        tangent_tolerance_rad=math.pi / 2,
        terminal_hold_steps=5,
        progress_stride=10_000,
    )

    def run_one(context: AnatomyContext, contact_aware: bool) -> tuple[Any, dict[str, Any]]:
        beam = context.bundle.models["plant"]
        provider = context.contact_provider if contact_aware else context.contact_free_provider
        controller = harness.ControllerAdapter(
            "naive_inverse_jacobian", "self-test",
            inverse_module.InverseJacobianBeamController(
                reference=reference, jacobian_provider=provider,
                sample_period_s=loop_config.sample_period_s,
                velocity_limit=loop_config.velocity_limit,
                acceleration_limit=loop_config.acceleration_limit,
                state_min=loop_config.state_min, state_max=loop_config.state_max,
                allow_undeclared_jacobian=True,
            ),
        )
        result = harness.run_closed_loop(
            controller=controller, reference=reference,
            beam_snapshot=lambda z, commit: (beam.tip(z), np.array([1.0, 0.0, 0.0])),
            magnet_pose=lambda z: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
            reset_plant=None, config=loop_config,
            initial_state=np.asarray(reference.state[0], dtype=float), verbose=False,
        )
        return result, diagnose_cell(records=result.records, context=context)

    engagement, free_metric, aware_metric = [], [], []
    for context in structural_contexts:
        aware, _ = run_one(context, True)
        free, diagnosis = run_one(context, False)
        engagement.append(float(context.bundle.models["plant"].strength))
        aware_metric.append(aware.summary["rms_beam_position_error_mm"])
        free_metric.append(free.summary["rms_beam_position_error_mm"])
        if context is structural_contexts[-1]:
            check("the diagnosis records a Jacobian gap",
                  diagnosis["jacobian_gap_norm"] > 0.0,
                  f"|gap| = {diagnosis['jacobian_gap_norm']:.4f}")
            check("the diagnosis records a directional prediction error",
                  np.isfinite(diagnosis["directional_prediction_error_mean"]),
                  f"{diagnosis['directional_prediction_error_mean']:.4f}")

    check("the contact-free controller is worse on the tightest anatomy",
          free_metric[-1] > aware_metric[-1],
          f"{free_metric[-1]:.4f} vs {aware_metric[-1]:.4f} mm rms")

    dose = EV.engagement_dose_response(
        engagement=engagement, contact_free_metric=free_metric,
        contact_aware_metric=aware_metric,
    )
    check("the penalty rises with engagement", dose["penalty_rises_with_engagement"],
          f"slope {dose['slope']:.2f}, 90% {[round(v, 3) for v in dose['slope_90_percent_interval']]}")

    # ---- the index writer -------------------------------------------------
    rows = [
        {**cell.as_dict(), "status": "ok", "rms_beam_position_error_mm": 0.1}
        for cell in cells[:4]
    ]
    rows.append({**cells[4].as_dict(), "status": "failed", "failure": "boom"})
    _write_index(output / "index_test.csv", rows, common)
    text = (output / "index_test.csv").read_text(encoding="utf-8")
    check("the index keeps a column that only some rows have", "failure" in text)
    check("the index has one line per cell plus a header",
          len(text.strip().splitlines()) == len(rows) + 1)

    # ---- provider provenance is enforced ---------------------------------
    try:
        providers_module.declare_provider(lambda z: np.zeros((3, 7)))
        check("a bare callable is refused as a Jacobian source", False, "no error")
    except TypeError as error:
        check("a bare callable is refused as a Jacobian source",
              "provenance" in str(error))

    print("\ncontact study self-test")
    print("=" * 74)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    failures = sum(1 for ok, _, _ in results if not ok)
    print("=" * 74)
    print(f"  {len(results) - failures}/{len(results)} checks passed")
    print(f"  artefacts in {output}")
    print(
        "\n  Note: this self-test covers the study's own logic. Rebuilding the\n"
        "  model bundle per anatomy and constructing the MPC rungs are exercised\n"
        "  only by a real run, because they need the project package."
    )
    return 1 if failures else 0


# ==========================================================================
# entry point
# ==========================================================================
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the contact-vs-disturbance study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=("offline", "closed", "observer", "all"), default="offline",
        help="offline is cheap and comes first; run it before paying for the rest.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--radius-scales", type=float, nargs="+", default=(1.0, 0.75, 0.55, 0.40),
        help="Lumen radius multipliers. The centreline is untouched, so one "
             "plan stays valid across the whole sweep.",
    )
    parser.add_argument(
        "--tortuosity-scales", type=float, nargs="+", default=(),
        help="Bend-angle multipliers. These change the plan, so each needs its "
             "own re-planned reference via --anatomy-json; cells are marked "
             "confounded.",
    )
    parser.add_argument(
        "--anatomy-json", type=Path, default=None,
        help="A JSON list of {name, radius_scale, tortuosity_scale, "
             "reference_dir} objects, replacing the scale sweeps.",
    )
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    parser.add_argument("--jacobian-models", nargs="+", default=list(JACOBIAN_MODELS))
    parser.add_argument("--initial-conditions", type=int, default=1)
    parser.add_argument("--initial-joint-spread-deg", type=float, default=0.5)
    parser.add_argument("--initial-insertion-spread-mm", type=float, default=1.0)
    parser.add_argument("--offline-stride", type=int, default=5)
    parser.add_argument(
        "--observer-alphas", type=float, nargs="+",
        default=(0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98),
        help="disturbance_filter_alpha values. Must lie in [0, 1).",
    )
    parser.add_argument("--observer-controller", default="mpc_ltv_offline")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--terminal-hold-steps", type=int, default=15)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    parser.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    parser.add_argument("--lti-freeze-index", type=int, default=0)
    parser.add_argument("--sqp-inner-iterations", type=int, default=1)
    parser.add_argument("--sqp-relinearise-horizon", action="store_true")
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--allow-planned-beam-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the cells and stop. Do this before a long study.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _load_anatomies(args: argparse.Namespace) -> list[AnatomySpec]:
    if args.anatomy_json is not None:
        entries = json.loads(Path(args.anatomy_json).read_text(encoding="utf-8"))
        return [
            AnatomySpec(
                name=str(entry["name"]),
                radius_scale=float(entry.get("radius_scale", 1.0)),
                tortuosity_scale=float(entry.get("tortuosity_scale", 1.0)),
                reference_dir=(
                    Path(entry["reference_dir"]) if entry.get("reference_dir") else None
                ),
            )
            for entry in entries
        ]
    return default_anatomies(args.radius_scales, args.tortuosity_scales)


def main() -> int:
    args = _arguments()
    if args.self_test:
        return run_self_test()

    anatomies = _load_anatomies(args)
    if args.dry_run:
        # Deliberately before any project import: listing the cells is exactly
        # the thing you want to be able to do on a laptop, before committing a
        # machine to a study that may take hours.
        cells = enumerate_closed_loop_cells(
            anatomies=anatomies, jacobian_models=args.jacobian_models,
            controllers=args.controllers, initial_conditions=args.initial_conditions,
        )
        observer = enumerate_observer_cells(
            anatomy=min(anatomies, key=lambda spec: spec.radius_scale),
            controller=args.observer_controller, alphas=args.observer_alphas,
        )
        confounded = [spec for spec in anatomies if spec.confounded]
        print(f"[dry run] anatomies: {', '.join(spec.name for spec in anatomies)}")
        print(f"[dry run] {len(cells)} closed-loop cells, {len(observer)} observer cells")
        for cell in cells + observer:
            print("   " + cell.directory_name())
        for spec in confounded:
            if spec.reference_dir is None:
                print(
                    f"[dry run] WARNING: {spec.name} changes the centreline but has "
                    "no reference_dir. The real run will refuse it — re-plan on "
                    "that anatomy and pass --anatomy-json."
                )
        return 0

    common = _load("_stack_common")
    harness = _load("compare_controllers")
    providers_module = _load("beam_jacobian_providers")
    variants_module = _load("mpc_variants")
    inverse_module = _load("inverse_jacobian_controller")
    base_module = harness.load_base_module()
    beam_module = harness.load_beam_output_module()

    from proper_research.planning.planning_context import build_planning_context

    context_cfg, bundle, controller_pack, out_root = build_planning_context()
    _assert_plant_contact(bundle, "the baseline planning context")

    reference_dir = args.reference or (out_root / "time_parameterized_configuration_path")
    output_dir = Path(args.output_dir or (out_root / "contact_study"))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_context = resolve_planning_context(context_cfg, controller_pack)
    base_context["reference_dir"] = reference_dir
    print(
        f"[study] design config resolved from {base_context.pop('resolved_from')}",
        flush=True,
    )

    # Checked here, before any anatomy is built. Each anatomy assembles a full
    # beam model, so discovering a missing reference four bundles later would
    # waste minutes and tell you nothing you could not have known now.
    if not reference_dir.exists():
        present = (
            sorted(entry.name for entry in out_root.iterdir() if entry.is_dir())
            if out_root.exists() else []
        )
        raise SystemExit(
            f"\nNo reference trajectory at:\n  {reference_dir}\n\n"
            "Stage 1 evaluates both Jacobians AT THE STATES THE PLANNER PRODUCED, "
            "so layers 1-3 (inverse configuration, global smoothing, time "
            "parameterisation) must have run before any stage of this study.\n\n"
            f"Directories under {out_root}:\n  "
            + ("\n  ".join(present) if present else "(none)")
            + "\n\nIf your reference lives elsewhere, point at it with --reference."
        )

    print(f"[study] anatomies: {', '.join(spec.name for spec in anatomies)}", flush=True)

    # The offline stage touches no plant, so every anatomy can share the
    # baseline controller pack -- see build_anatomy_context. That turns N calls
    # to build_controller (each running IK and chain-rule validation) into one,
    # which matters when sweeping six radii to choose phantom diameters.
    shared_pack = controller_pack if args.stage == "offline" else None
    if shared_pack is not None:
        print(
            "[study] offline stage: sharing one controller pack across anatomies "
            "(robot kinematics only; contact physics is still per-anatomy)",
            flush=True,
        )

    contexts: dict[str, AnatomyContext] = {}
    for spec in anatomies:
        print(f"[study] building anatomy {spec.name}", flush=True)
        contexts[spec.name] = build_anatomy_context(
            spec=spec, base_context=base_context, providers_module=providers_module,
            base_module=base_module,
            require_planned_beam_feasible=not args.allow_planned_beam_failure,
            shared_controller_pack=shared_pack,
        )

    manifest: dict[str, Any] = {
        "reference_dir": str(reference_dir),
        "anatomies": [spec.as_dict() for spec in anatomies],
        "controllers": list(args.controllers),
        "jacobian_models": list(args.jacobian_models),
        "initial_conditions": int(args.initial_conditions),
        "plant": "contact enabled in every cell; only the controller Jacobian varies",
        "stage": args.stage,
    }

    settings = StudySettings(
        horizon=int(args.horizon),
        terminal_hold_steps=int(args.terminal_hold_steps),
        position_tolerance_mm=float(args.position_tolerance_mm),
        joint_acceleration_limit=float(args.joint_acceleration_limit),
        insertion_acceleration_limit=float(args.insertion_acceleration_limit),
        lti_freeze_index=int(args.lti_freeze_index),
        sqp_inner_iterations=int(args.sqp_inner_iterations),
        sqp_relinearise_horizon=bool(args.sqp_relinearise_horizon),
        solver=str(args.solver),
        max_control_steps=args.max_control_steps,
        verbose=bool(args.verbose),
    )
    manifest["settings"] = settings.as_dict()

    modules = {
        "harness": harness, "base": base_module, "beam": beam_module,
        "variants": variants_module, "inverse": inverse_module,
    }
    conditions = {
        name: initial_states(
            reference=context.reference, count=int(args.initial_conditions),
            joint_spread_deg=float(args.initial_joint_spread_deg),
            insertion_spread_mm=float(args.initial_insertion_spread_mm),
        )
        for name, context in contexts.items()
    }

    if args.stage in {"offline", "all"}:
        print("\n[study] stage 1 — offline Jacobian divergence", flush=True)
        manifest["offline"] = run_offline_stage(
            contexts=list(contexts.values()), output_dir=output_dir / "offline",
            stride=int(args.offline_stride), common=common,
        )

    if args.stage in {"closed", "all"}:
        print("\n[study] stage 2 — closed-loop factorial", flush=True)
        cells = enumerate_closed_loop_cells(
            anatomies=anatomies, jacobian_models=args.jacobian_models,
            controllers=args.controllers, initial_conditions=args.initial_conditions,
        )
        run_cells(
            cells=cells, contexts=contexts, settings=settings,
            output_dir=output_dir, modules=modules,
            initial_condition_states=conditions, common=common,
        )

    if args.stage in {"observer", "all"}:
        print("\n[study] stage 3 — disturbance-observer ceiling", flush=True)
        tightest = min(anatomies, key=lambda spec: spec.radius_scale)
        manifest["observer_anatomy"] = tightest.name
        observer_cells = enumerate_observer_cells(
            anatomy=tightest, controller=args.observer_controller,
            alphas=args.observer_alphas,
        )
        run_cells(
            cells=observer_cells, contexts=contexts, settings=settings,
            output_dir=output_dir / "observer", modules=modules,
            initial_condition_states=conditions, common=common,
        )

    (output_dir / "study_manifest.json").write_text(
        json.dumps(common.jsonable(manifest), indent=2), encoding="utf-8"
    )
    print(f"\n[study] -> {output_dir}")
    print("[study] next: python analyse_contact_study.py " + str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
