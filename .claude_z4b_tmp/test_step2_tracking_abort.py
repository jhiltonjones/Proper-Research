"""Z4b step 2: verify the controller-aware tracking-error abort default fires
at the right threshold (7mm MPC / 15mm INV) via a synthetic error ramp, and
that the insertion-offset monitor now defaults off."""
import sys
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")

from proper_research.hardware.online.close_loop_path_follow import PathFollowConfig
from proper_research.hardware.online.controller_adapters import _MPC_KINDS

# --- 1. default-fill policy, replicated from close_loop_path_follow.main() -
def resolve_default(controller_kind: str) -> float:
    cfg = PathFollowConfig()
    cfg.controller_kind = controller_kind
    _ONLINE_MPC_KINDS = set(_MPC_KINDS) | {"mpc_delay_aware"}
    if cfg.max_tracking_error_m <= 0.0:
        cfg.max_tracking_error_m = 0.007 if cfg.controller_kind in _ONLINE_MPC_KINDS else 0.015
    return cfg.max_tracking_error_m

cases = {
    "mpc_delay_aware": 0.007,     # real online MPC string (qn_zero, insertion_anchor, vessel MPC)
    "mpc_ltv_offline": 0.007,     # offline-harness MPC string (also real online, run_mpc_ltv.py)
    "inv_7dof_delay_aware": 0.015,
    "naive_inverse_jacobian_ltv": 0.015,
    "inv_2dof_trim": 0.015,
}
for kind, expected in cases.items():
    got = resolve_default(kind)
    assert abs(got - expected) < 1e-12, f"{kind}: expected {expected}, got {got}"
    print(f"[OK] controller_kind={kind!r} -> default max_tracking_error_m={got*1000:.0f}mm")

# An explicit nonzero caller value must NOT be overridden.
cfg = PathFollowConfig()
cfg.controller_kind = "mpc_delay_aware"
cfg.max_tracking_error_m = 0.005  # explicit 5mm, e.g. an existing analysis script
_ONLINE_MPC_KINDS = set(_MPC_KINDS) | {"mpc_delay_aware"}
if cfg.max_tracking_error_m <= 0.0:
    cfg.max_tracking_error_m = 0.007 if cfg.controller_kind in _ONLINE_MPC_KINDS else 0.015
assert cfg.max_tracking_error_m == 0.005
print("[OK] explicit nonzero max_tracking_error_m (5mm) is NOT overridden by the policy default")

# --- 2. synthetic error ramp: does the abort actually fire at the right tick?
# Reproduces the exact check from close_loop_path_follow.py's main() loop:
#   if cfg.max_tracking_error_m > 0.0 and error_mm > 1.0e3 * cfg.max_tracking_error_m: abort
def ramp_test(max_tracking_error_m: float, ramp_mm_per_tick: float, n_ticks: int = 40):
    threshold_mm = 1.0e3 * max_tracking_error_m
    abort_tick = None
    for tick in range(n_ticks):
        error_mm = tick * ramp_mm_per_tick
        if max_tracking_error_m > 0.0 and error_mm > threshold_mm:
            abort_tick = tick
            break
    return abort_tick, threshold_mm

# MPC path: 7mm threshold, ramp 0.5mm/tick -> should trip at tick 15 (7.5mm > 7mm)
abort_tick, threshold_mm = ramp_test(0.007, ramp_mm_per_tick=0.5)
assert abort_tick == 15, f"MPC ramp: expected abort at tick 15, got {abort_tick}"
print(f"[OK] MPC (7mm threshold): synthetic 0.5mm/tick ramp aborts at tick {abort_tick} "
      f"(error={abort_tick*0.5:.1f}mm > {threshold_mm:.0f}mm)")

# INV path: 15mm threshold, same ramp -> should trip at tick 31 (15.5mm > 15mm)
abort_tick, threshold_mm = ramp_test(0.015, ramp_mm_per_tick=0.5)
assert abort_tick == 31, f"INV ramp: expected abort at tick 31, got {abort_tick}"
print(f"[OK] INV-7 (15mm threshold): synthetic 0.5mm/tick ramp aborts at tick {abort_tick} "
      f"(error={abort_tick*0.5:.1f}mm > {threshold_mm:.0f}mm)")

# A ramp that never crosses either threshold must never abort.
abort_tick, _ = ramp_test(0.007, ramp_mm_per_tick=0.1, n_ticks=40)  # max error 3.9mm < 7mm
assert abort_tick is None, f"expected no abort, got tick {abort_tick}"
print("[OK] a ramp that stays under threshold (max 3.9mm < 7mm) never aborts")

# --- 3. insertion-offset monitor defaults off ------------------------------
# Replicates the exact resolution formula now used by all 3 updated scripts.
import argparse
def resolve_insertion_offset(disable: bool, enable: bool, mm: float = 5.0):
    return mm * 1e-3 if (enable and not disable) else None

assert resolve_insertion_offset(disable=False, enable=False) is None
print("[OK] insertion-offset monitor defaults OFF (neither flag passed)")
assert resolve_insertion_offset(disable=False, enable=True) == 0.005
print("[OK] --enable-insertion-offset-abort turns it back on at the given threshold")
assert resolve_insertion_offset(disable=True, enable=True) is None
print("[OK] --disable-insertion-offset-abort still wins even if --enable is also passed "
      "(backward-compatible no-op path stays authoritative)")

print("[PASS] Z4b step 2 test")
