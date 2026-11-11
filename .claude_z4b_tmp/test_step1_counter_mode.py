"""Z4b step 1: verify progress_mode defaults to "counter" everywhere, and that
counter-mode index == raw tick count under mocked 125ms ticks."""
import sys
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")

from dataclasses import dataclass
from types import SimpleNamespace

from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    _BaseProcessIsolatedAdapter,
)
from proper_research.hardware.online.close_loop_path_follow import PathFollowConfig

# --- 1. defaults -----------------------------------------------------------
cfg = OfflineControllerConfig()
assert cfg.progress_mode == "counter", f"OfflineControllerConfig default is {cfg.progress_mode!r}"
print(f"[OK] OfflineControllerConfig().progress_mode == {cfg.progress_mode!r}")

pf_cfg = PathFollowConfig()
assert pf_cfg.progress_mode == "counter", f"PathFollowConfig default is {pf_cfg.progress_mode!r}"
print(f"[OK] PathFollowConfig().progress_mode == {pf_cfg.progress_mode!r} "
      f"(shared by every runner going through close_loop_path_follow.py: "
      f"vessel, shape/rectangle, INV-7)")

# --- 2. index == tick count at mocked 125ms ticks --------------------------
# Call the real _progress_index logic via a minimal stand-in carrying only
# the attributes that method reads (self.config, self._counter, self.reference,
# self._t0), rather than constructing a full worker-backed adapter.
reference = SimpleNamespace(sample_count=1000, sample_period_s=0.1)
fake_self = SimpleNamespace(
    config=OfflineControllerConfig(terminal_hold_steps=50),
    reference=reference,
    _counter=0,
    _t0=None,
)
progress_index_fn = _BaseProcessIsolatedAdapter._progress_index

t = 0.0
dt_tick = 0.125  # 125ms mocked tick period (realized period != reference dt=0.1s)
mismatches = []
for tick in range(40):
    estimate = SimpleNamespace(t_monotonic=t)
    idx = progress_index_fn(fake_self, estimate)
    if idx != tick:
        mismatches.append((tick, idx))
    fake_self._counter = tick + 1  # mirrors the adapter's own self._counter += 1 after each call
    t += dt_tick

if mismatches:
    print(f"[FAIL] counter-mode index != tick count at {len(mismatches)}/40 ticks: {mismatches[:5]}")
    sys.exit(1)
print(f"[OK] counter-mode index == tick count for all 40 mocked 125ms ticks "
      f"(reference.sample_period_s=0.1s, realized tick period=0.125s -- "
      f"exactly the drift case 'wallclock' would have mis-tracked, which "
      f"counter-mode is immune to by construction)")

print("[PASS] Z4b step 1 test")
