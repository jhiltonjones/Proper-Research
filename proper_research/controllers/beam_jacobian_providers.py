"""Contact-aware and contact-free beam Jacobians, from one shared plant.

The comparison you want — does relinearising online beat a scheduled offline
Jacobian? — is only meaningful if every controller is handed a Jacobian that is
wrong in the *same known way*.  This module builds those Jacobians.

Two models already exist
------------------------
``build_model_bundle`` constructs both and hands them back together::

    models = {"plant": ..., "contact": ..., "no_contact": ...}

``contact`` has ``ContactConfig(enabled=True, use_in_jacobian=True)`` and a
``LumenQuery``; ``no_contact`` has ``ContactConfig.disabled()`` and no lumen
query at all.  They are separate instances with separate equilibrium caches, so
differentiating one cannot perturb the other.  That is the route this module
prefers, and it is why ``jacobian_model`` is a separate argument to
``controller_factory_joint_space.build_controller`` in the first place.

Three ways to get a contact-free Jacobian, in descending order of preference:

1. **A separate model instance** (``from_model_bundle``).  Structurally
   contact-free: the contact energy is never assembled, never differentiated,
   and the beam equilibrium the derivative is taken at is the no-contact one.
   This is the honest "the controller does not know about contact" experiment.

2. **The flag on a shared model** (``contact_free_jacobian_scope``).  Sets
   ``model.contact_cfg.use_in_jacobian = False`` and restores it afterwards.
   The forward solve still includes contact — ``energy_from_u`` uses
   ``use_contact`` — while the sensitivity drops it, because
   ``energy_gradient_u`` is called with ``use_contact=use_contact_in_jacobian``.
   So the derivative is taken *at the contact equilibrium* but *of the
   contact-free energy*.  That is a different experiment from 1, and often the
   more realistic one: the plant really is in contact, and only the model used
   to linearise is wrong.

3. **The pack's own adapters** (``from_controller_pack``).
   ``plant_diagnostic_joint_adapter`` differentiates the plant model;
   ``jacobian_joint_adapter`` differentiates whatever was passed as
   ``jacobian_model``.  Convenient, but what you get depends on how the pack
   was built, so the provider reports which it found rather than assuming.

Which to use
------------
For the four-controller comparison: **1 or 2, applied identically to all four**.
Mixing them, or letting one controller see contact while another does not, makes
the comparison meaningless. Every provider carries a ``describe()`` that names
the model and the contact settings it is actually differentiating, and the
comparison harness prints it — so a mismatch shows up in the report rather than
in a conclusion.

All providers return the 3x7 tip-position Jacobian
``d p_tip / d [q1..q6, insertion]``, taken from the first three rows of the
adapter's 6x7 ``continuous_output_jacobian`` — the same interface the inverse
planner and the beam-output MPC already use.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def position_jacobian_from_output(value: Any) -> Array:
    """Take the 3x7 tip-position block out of a 6x7 output Jacobian."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (6, 7):
        matrix = matrix[:3]
    elif matrix.shape != (3, 7):
        raise ValueError(
            f"Expected a (3, 7) or (6, 7) beam Jacobian; received {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("The beam Jacobian contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=float)


def _contact_settings(model: Any) -> dict[str, Any]:
    contact = getattr(model, "contact_cfg", None)
    if contact is None:
        return {"enabled": None, "use_in_jacobian": None}
    return {
        "enabled": bool(getattr(contact, "enabled", False)),
        "use_in_jacobian": bool(getattr(contact, "use_in_jacobian", False)),
    }


@dataclass
class BeamJacobianProvider:
    """A named 3x7 tip-position Jacobian with its provenance attached.

    ``name`` and ``describe()`` exist so a run can record *which* model it
    differentiated. A comparison whose provenance is not recorded is not a
    comparison.
    """

    name: str
    jacobian_at: Callable[[Array], Array]
    adapter: Any
    model: Any
    source: str

    def __call__(self, state: Any) -> Array:
        return self.jacobian_at(np.asarray(state, dtype=float).reshape(7))

    def describe(self) -> dict[str, Any]:
        settings = _contact_settings(self.model)
        return {
            "name": self.name,
            "source": self.source,
            "contact_enabled": settings["enabled"],
            "contact_used_in_jacobian": settings["use_in_jacobian"],
            "lumen_query_present": bool(getattr(self.model, "lumen_query", None)),
            "model_id": hex(id(self.model)),
        }

    def summary_line(self) -> str:
        info = self.describe()
        return (
            f"{info['name']}: {info['source']} | contact enabled="
            f"{info['contact_enabled']} used_in_jacobian="
            f"{info['contact_used_in_jacobian']} lumen_query="
            f"{info['lumen_query_present']}"
        )


def _adapter_provider(name: str, adapter: Any, source: str) -> BeamJacobianProvider:
    method = getattr(adapter, "continuous_output_jacobian", None)
    if not callable(method):
        raise TypeError(
            f"{source} does not expose continuous_output_jacobian(state), which "
            "is the analytical Jacobian interface the inverse planner uses."
        )

    def jacobian_at(state: Array) -> Array:
        return position_jacobian_from_output(method(np.asarray(state, dtype=float)))

    return BeamJacobianProvider(
        name=name,
        jacobian_at=jacobian_at,
        adapter=adapter,
        model=getattr(adapter, "model", None),
        source=source,
    )


# --------------------------------------------------------------------------
# 1. separate model instances (preferred)
# --------------------------------------------------------------------------
def build_diagnostic_adapter(
    *,
    beam_model: Any,
    controller_pack: dict[str, Any],
    jacobian_mode: str = "fast",
) -> Any:
    """A six-output ``JointSpaceBeamMPCAdapter`` around an arbitrary beam model.

    The robot kinematics are taken from the pack (``robot_dh`` and ``T_F_M``),
    so this adapter and the plant's share exactly the same magnet forward
    kinematics and geometric Jacobian.  Only the beam model differs — which is
    the point.

    The returned adapter owns its own forward/Jacobian callbacks and therefore
    its own equilibrium cache, so evaluating it never disturbs the plant.
    """
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.simulation.simulations import (
        controller_factory_joint_space as joint_space,
    )
    from proper_research.simulation.simulations.joint_space_beam_mpc_adapter import (
        JointSpaceBeamMPCAdapter,
    )

    for key in ("robot_dh", "T_F_M"):
        if key not in controller_pack:
            raise KeyError(
                f"controller_pack has no {key!r}. Build the pack with "
                "controller_factory_joint_space.build_controller, which returns "
                "the robot DH table and the flange-to-magnet transform."
            )
    dh = controller_pack["robot_dh"]
    T_F_M = np.asarray(controller_pack["T_F_M"], dtype=float).reshape(4, 4)

    def magnet_transform_fn(q6: Any) -> Array:
        return urik.forward_kinematics(q6, dh, T_F_M).T_R_target

    def magnet_geometric_jacobian_fn(q6: Any) -> Array:
        return urik.geometric_jacobian(q6, dh, T_F_M)

    beam_output, forward_adapter = joint_space._make_beam_forward_callback(beam_model)
    beam_jacobian = joint_space._make_beam_jacobian_callback(
        beam_model,
        forward_adapter=forward_adapter,
        jacobian_mode=jacobian_mode,
    )
    return JointSpaceBeamMPCAdapter(
        magnet_transform_fn=magnet_transform_fn,
        magnet_geometric_jacobian_fn=magnet_geometric_jacobian_fn,
        beam_output_fn=beam_output,
        beam_output_jacobian_fn=beam_jacobian,
        output_indices=(0, 1, 2, 3, 4, 5),
        model=beam_model,
    )


def from_model_bundle(
    *,
    bundle: Any,
    controller_pack: dict[str, Any],
    contact: bool,
    jacobian_mode: str = "fast",
) -> BeamJacobianProvider:
    """The structurally contact-free (or contact-aware) Jacobian.

    ``contact=False`` differentiates ``bundle.models["no_contact"]``: the
    contact energy is never assembled, so the equilibrium the derivative is
    taken at is itself contact-free.  This is the strongest form of "the
    controller does not know the beam touches anything".
    """
    models = getattr(bundle, "models", None)
    if not isinstance(models, dict):
        raise TypeError("bundle.models must be the dict built by build_model_bundle.")
    key = "contact" if contact else "no_contact"
    if key not in models:
        raise KeyError(
            f"bundle.models has no {key!r} entry; available: {sorted(models)}."
        )
    model = models[key]
    settings = _contact_settings(model)
    if not contact and settings["enabled"]:
        raise ValueError(
            "bundle.models['no_contact'] reports contact enabled. The bundle was "
            "not built with a genuinely contact-free model."
        )
    adapter = build_diagnostic_adapter(
        beam_model=model,
        controller_pack=controller_pack,
        jacobian_mode=jacobian_mode,
    )
    return _adapter_provider(
        name="contact" if contact else "no_contact",
        adapter=adapter,
        source=f"bundle.models[{key!r}] via a dedicated diagnostic adapter",
    )


def contact_and_contact_free(
    *, bundle: Any, controller_pack: dict[str, Any], jacobian_mode: str = "fast"
) -> tuple[BeamJacobianProvider, BeamJacobianProvider]:
    """Both providers, sharing robot kinematics and differing only in the beam."""
    return (
        from_model_bundle(
            bundle=bundle,
            controller_pack=controller_pack,
            contact=True,
            jacobian_mode=jacobian_mode,
        ),
        from_model_bundle(
            bundle=bundle,
            controller_pack=controller_pack,
            contact=False,
            jacobian_mode=jacobian_mode,
        ),
    )


# --------------------------------------------------------------------------
# 2. the flag on a shared model
# --------------------------------------------------------------------------
@contextlib.contextmanager
def contact_free_jacobian_scope(model: Any, *, required: bool = True) -> Iterator[bool]:
    """Differentiate the contact-free energy at the contact equilibrium.

    The forward solve is untouched: ``energy_from_u`` uses ``use_contact``,
    while ``energy_gradient_u`` — which builds both the Hessian and G_theta
    inside ``implicit_tip_jacobian`` — is called with
    ``use_contact=use_contact_in_jacobian``.  So the plant stays in contact and
    only the linearisation forgets about it.

    Yields True when the flag was actually changed.  With ``required=False`` and
    no such flag it yields False and changes nothing, so a caller can report
    "this is not the contact-free experiment" rather than silently running a
    different one.
    """
    contact = getattr(model, "contact_cfg", None)
    if contact is None or not hasattr(contact, "use_in_jacobian"):
        if required:
            raise AttributeError(
                "Cannot switch this model's Jacobian to contact-free: "
                "model.contact_cfg.use_in_jacobian was not found. Use "
                "from_model_bundle(contact=False) for a separate contact-free "
                "model instead."
            )
        yield False
        return
    previous = bool(contact.use_in_jacobian)
    contact.use_in_jacobian = False
    try:
        yield True
    finally:
        contact.use_in_jacobian = previous


# --------------------------------------------------------------------------
# 3. whatever the pack already holds
# --------------------------------------------------------------------------
def from_controller_pack(
    controller_pack: dict[str, Any], *, prefer: str = "jacobian"
) -> BeamJacobianProvider:
    """The pack's own adapter.

    ``prefer="jacobian"`` uses ``jacobian_joint_adapter`` (the model passed as
    ``jacobian_model``); ``prefer="plant"`` uses
    ``plant_diagnostic_joint_adapter`` (the plant itself).  What contact
    settings you get depends on how the pack was built, so check
    ``describe()`` before drawing conclusions.
    """
    order = (
        ("jacobian_joint_adapter", "plant_diagnostic_joint_adapter")
        if prefer == "jacobian"
        else ("plant_diagnostic_joint_adapter", "jacobian_joint_adapter")
    )
    for key in order:
        adapter = controller_pack.get(key)
        if adapter is not None:
            output_count = int(getattr(adapter, "n_out", 6))
            if output_count != 6 and key == "jacobian_joint_adapter":
                # The jacobian adapter may be built with n_out=3 for the MPC.
                # That still yields a valid 3x7 position Jacobian.
                pass
            return _adapter_provider(
                name=f"pack:{key}",
                adapter=adapter,
                source=f"controller_pack[{key!r}]",
            )
    raise KeyError(
        "controller_pack contains neither 'jacobian_joint_adapter' nor "
        "'plant_diagnostic_joint_adapter'."
    )


# --------------------------------------------------------------------------
# provenance enforcement
# --------------------------------------------------------------------------
UNDECLARED = {
    "name": "undeclared",
    "source": "a bare callable with no provenance",
    "contact_enabled": None,
    "contact_used_in_jacobian": None,
    "lumen_query_present": None,
    "model_id": None,
}


def declare_provider(
    provider: Any, *, allow_undeclared: bool = False
) -> tuple[Callable[[Array], Array], dict[str, Any]]:
    """Split a Jacobian source into ``(callable, provenance)``.

    Every controller in this package calls this on construction, so a run can
    always answer "was this Jacobian contact-free?" from its own record rather
    than from whoever remembers what they passed.

    A bare callable carries no provenance, and a contact study whose provenance
    is a guess is not a study.  Passing one therefore raises unless
    ``allow_undeclared=True`` — which is fine for a unit test with a mock, and
    is not fine for an experiment you intend to publish.
    """
    if isinstance(provider, BeamJacobianProvider):
        return provider.jacobian_at, provider.describe()
    if not callable(provider):
        raise TypeError("A Jacobian provider must be callable.")
    if not allow_undeclared:
        raise TypeError(
            "This Jacobian source has no provenance, so the run could not record "
            "whether it is contact-free. Pass a BeamJacobianProvider (see "
            "from_model_bundle / contact_and_contact_free), or set "
            "allow_undeclared=True if this is a mock in a unit test."
        )
    return provider, dict(UNDECLARED)


def provenance_line(provenance: dict[str, Any]) -> str:
    """One line naming the model and whether contact is in the derivative."""
    used = provenance.get("contact_used_in_jacobian")
    verdict = (
        "CONTACT-AWARE"
        if used
        else "CONTACT-FREE"
        if used is False
        else "UNKNOWN (undeclared source)"
    )
    return (
        f"{verdict} | name={provenance.get('name')} "
        f"source={provenance.get('source')} "
        f"contact_enabled={provenance.get('contact_enabled')} "
        f"lumen_query={provenance.get('lumen_query_present')}"
    )


# --------------------------------------------------------------------------
# agreement check
# --------------------------------------------------------------------------
def compare_providers(
    first: BeamJacobianProvider,
    second: BeamJacobianProvider,
    states: Any,
) -> dict[str, Any]:
    """How far apart two Jacobians are over a set of states.

    Run this on the reference trajectory before the controller comparison. If
    the contact and contact-free Jacobians barely differ, the model mismatch
    you are asking the controllers to reject is not there, and the comparison
    will show four near-identical controllers for an uninteresting reason.
    """
    states = np.asarray(states, dtype=float).reshape(-1, 7)
    differences, relative = [], []
    for state in states:
        a = first(state)
        b = second(state)
        gap = float(np.linalg.norm(a - b))
        scale = max(float(np.linalg.norm(a)), 1e-30)
        differences.append(gap)
        relative.append(gap / scale)
    differences = np.asarray(differences)
    relative = np.asarray(relative)
    return {
        "first": first.describe(),
        "second": second.describe(),
        "samples": int(states.shape[0]),
        "frobenius_difference_max": float(np.max(differences)),
        "frobenius_difference_rms": float(np.sqrt(np.mean(differences**2))),
        "relative_difference_max": float(np.max(relative)),
        "relative_difference_rms": float(np.sqrt(np.mean(relative**2))),
        "note": (
            "A relative difference well below ~1% means the two Jacobians are "
            "nearly the same model, and a contact-free-versus-contact "
            "comparison has little to reject."
        ),
    }


__all__ = [
    "BeamJacobianProvider",
    "UNDECLARED",
    "declare_provider",
    "provenance_line",
    "build_diagnostic_adapter",
    "compare_providers",
    "contact_and_contact_free",
    "contact_free_jacobian_scope",
    "from_controller_pack",
    "from_model_bundle",
    "position_jacobian_from_output",
]
