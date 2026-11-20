"""Z4b step 4, revised 2026-11-11: test check_against_protocol with one
deliberately wrong value per field group -- confirm each gets refused, and
that an all-correct resolved config passes. Rooted entirely in
/home/jack/Proper-Research -- no worktree path, no robot/camera import,
no motion. Run from /home/jack/Proper-Research.
"""
import sys
sys.path.insert(0, "/home/jack/Proper-Research")

from proper_research.hardware.online.stage_a_common import check_against_protocol, ProtocolMismatchError

PROTOCOL = "/home/jack/Proper-Research/protocol_campaign2.yaml"
COND = "rectangle_mpc_sj_n15"   # real, current condition key (post-rebuild)

# A fully-correct resolved config for rectangle_mpc_sj_n15 (mirrors the
# protocol entry's own current values -- re-read directly from
# protocol_campaign2.yaml at the time this test was written).
CORRECT = {
    "weights": {"R": 7.0, "R_d": 1.0e-3},
    "rate_limit_rad_s": {"joint": 0.1},
    "increment_rad": {"joint": 0.01},
    "tcp_box": {"min_m": [0.24469, -0.89361, 0.36544], "max_m": [0.58971, -0.49576, 0.46496]},
    "robust_margins": {"tcp_m": 0.015},
    "tracking_error_abort_m": 0.007,
    "insertion_offset_abort": "off",
    "reference_mode": "counter",
    "streaming": "on",
    "deadline_ms": 70.0,
    "delay_samples": 2,
    "osqp": {"eps_abs": 1.0e-7, "eps_rel": 1.0e-7, "max_iter": 20000, "polish": True,
              "time_limit_s": 0.03, "accept_time_limit_solution": True},
}

log = check_against_protocol(COND, CORRECT, protocol_path=PROTOCOL)
n_matched = sum(1 for r in log if r["status"] == "match")
print(f"[OK] fully-correct config: {n_matched} fields matched, no exception raised")

import copy

CASES = {
    "weights": ("weights.R", 99.0),
    "rate_limit": ("rate_limit_rad_s.joint", 0.5),
    "increment": ("increment_rad.joint", 0.5),
    "tcp_box": ("tcp_box.min_m.0", 999.0),  # handled specially below (list index)
    "robust_margins": ("robust_margins.tcp_m", 0.5),
    "tracking_error_abort": ("tracking_error_abort_m", 0.999),
    "insertion_offset_abort": ("insertion_offset_abort", "on"),
    "reference_mode": ("reference_mode", "wallclock"),
    "streaming": ("streaming", "off"),
    "deadline": ("deadline_ms", 9999.0),
    "delay_samples": ("delay_samples", 7),
    "osqp": ("osqp.max_iter", 1),
}

n_refused = 0
for group, (field, wrong_value) in CASES.items():
    wrong = copy.deepcopy(CORRECT)
    if field == "tcp_box.min_m.0":
        wrong["tcp_box"]["min_m"] = [wrong_value] + wrong["tcp_box"]["min_m"][1:]
    else:
        parts = field.split(".")
        d = wrong
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = wrong_value
    try:
        check_against_protocol(COND, wrong, protocol_path=PROTOCOL)
        print(f"[FAIL] group={group!r}: expected ProtocolMismatchError, got none!")
    except ProtocolMismatchError as exc:
        n_refused += 1
        print(f"[OK] group={group!r}: refused as expected ({str(exc)[:90]}...)")

assert n_refused == len(CASES), f"only {n_refused}/{len(CASES)} wrong-value cases were refused"
print(f"\n[PASS] Z4b step 4 test: all {len(CASES)} deliberately-wrong-value cases refused; "
      f"the correct config passed clean")
