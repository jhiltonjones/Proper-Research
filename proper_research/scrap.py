from proper_research.planning.planning_context import build_planning_context
exp_cfg, bundle, controller_pack, out_root = build_planning_context()
model = bundle.models["contact"]

for name in ("get_last_diag", "get_last_jacobian_diag"):
    method = getattr(model, name, None)
    print(name, "->", method() if callable(method) else "absent")

cache = getattr(model, "cache", None)
print("cache.info keys:", list(getattr(cache, "info", {}).keys()) if cache else "no cache")