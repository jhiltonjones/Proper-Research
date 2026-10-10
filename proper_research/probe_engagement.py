"""One-shot probe: find out what get_last_diag() actually is and when it fills in.

Run this in place of the earlier scrap.py. It answers three questions in one
pass instead of another round of guessing:

  1. Which class actually defines get_last_diag / get_last_jacobian_diag, and
     what does its source say? (inspect.getsource -- no more guessing field
     names from a mock that may not match the real class.)

  2. Are store_history / store_vectors_in_info / copy_cached_results set the
     way build_model_bundle's defaults leave them? If store_vectors_in_info is
     False, any diagnostic that needs a stored vector (force array, per-node
     contact weights) is off by construction, not a naming problem.

  3. Does diag populate on the SAME object after a real evaluation through the
     exact call path stage 1 uses, or does copy_cached_results mean the solve
     ran on a throwaway copy and the bundle's own model never sees it?

Paste the full output back -- whatever it says, it settles this without
another round trip.
"""

import inspect

from proper_research.planning.planning_context import build_planning_context

exp_cfg, bundle, controller_pack, out_root = build_planning_context()
model = bundle.models["contact"]

print("=" * 70)
print("1. WHERE THE METHODS ARE DEFINED")
print("=" * 70)
print("model class:", type(model).__module__ + "." + type(model).__name__)
print("MRO:", [f"{c.__module__}.{c.__name__}" for c in type(model).__mro__])
print()

for name in ("get_last_diag", "get_last_jacobian_diag"):
    method = getattr(model, name, None)
    if method is None:
        print(f"{name}: NOT PRESENT on this object")
        continue
    owner = None
    for cls in type(model).__mro__:
        if name in cls.__dict__:
            owner = cls
            break
    print(f"{name}: defined on {owner.__module__}.{owner.__name__}" if owner else name)
    try:
        print(inspect.getsource(method))
    except (OSError, TypeError) as error:
        print(f"  (source unavailable: {error})")
    print("-" * 70)

print()
print("=" * 70)
print("2. DIAGNOSTIC-RELEVANT FLAGS ON THIS MODEL INSTANCE")
print("=" * 70)
for attr in ("result_detail", "store_history", "store_vectors_in_info",
             "copy_cached_results", "sensitivity_workers"):
    print(f"  {attr} = {getattr(model, attr, '<absent>')!r}")

print()
print("=" * 70)
print("3. DOES DIAG POPULATE AFTER A REAL EVALUATION?")
print("=" * 70)

import beam_jacobian_providers as providers  # from the delivered controllers/ package

contact_provider, contact_free_provider = providers.contact_and_contact_free(
    bundle=bundle, controller_pack=controller_pack
)

import numpy as np
from proper_research.simulation.simulations import (
    simulate_time_parameterized_configuration_mpc as base,
)

reference = base.load_configuration_reference(
    out_root / "time_parameterized_configuration_path"
)
state = np.asarray(reference.state, dtype=float)[len(reference.state) // 2]

print("model id before evaluation:", hex(id(model)))
print(f"cache before: {model.cache!r}" if hasattr(model, "cache") else "no .cache attr")

J = contact_provider(state)
print("Jacobian evaluated, shape:", J.shape)
print("model id after evaluation: ", hex(id(model)), "(same object?" ,
      id(model) == id(bundle.models["contact"]), ")")

for name in ("get_last_diag", "get_last_jacobian_diag"):
    method = getattr(model, name, None)
    result = method() if callable(method) else "<not callable>"
    print(f"{name}() after evaluation -> {result!r}")

cache = getattr(model, "cache", None)
info = getattr(cache, "info", None) if cache is not None else None
print("model.cache.info after evaluation ->", info)

# If the provider wraps its own adapter around a DIFFERENT model instance
# (e.g. a copy made once at construction time, not per-call), the id check
# above will still say "same object" but the object that actually solved may
# be one level deeper inside the provider. Surface that too.
print()
print("provider internals (only useful if the id check above looked wrong):")
print("  contact_provider type:", type(contact_provider))
for attr in ("model", "adapter", "source"):
    if hasattr(contact_provider, attr):
        value = getattr(contact_provider, attr)
        print(f"  contact_provider.{attr} = {value!r}"
              + (f"  id={hex(id(value))}" if attr in ("model", "adapter") else ""))
