"""Generate synthetic path_follow.jsonl-shaped logs with a KNOWN, injected
joint-command delay, to validate delay_identification_5fold.py's fold/stop
logic independently of any real hardware log. Writes three files, one per
injected delay d in {1,2,3}, 300 ticks each (well above the ~150-tick
practical minimum), 6 joints, with small Gaussian noise so it's not a
trivial zero-noise case.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
N_TICKS = 300
N_JOINTS = 6
NOISE_STD = 1e-5  # rad, small relative to the commanded step scale below

# Commanded delta per tick: a slowly-varying sinusoid per joint, scale
# similar to a real max-joint-step-rad command (~1e-2 rad/tick).
t = np.arange(N_TICKS)
q_target_delta = np.stack([
    0.01 * np.sin(2 * np.pi * (t / 40.0) + j) for j in range(N_JOINTS)
], axis=1)

for d_true in (1, 2, 3):
    q_meas = np.zeros((N_TICKS, N_JOINTS))
    for k in range(1, N_TICKS):
        # realized step from k-1 to k equals the commanded delta issued
        # d_true ticks before k-1 (i.e. applied with a d_true-tick lag),
        # plus small noise.
        src = k - 1 - d_true
        step = q_target_delta[src] if src >= 0 else 0.0
        q_meas[k] = q_meas[k - 1] + step + rng.normal(0, NOISE_STD, N_JOINTS)

    path = f"/home/jack/Proper-Research/.claude_z4b_tmp/synthetic_d{d_true}_path_follow.jsonl"
    with open(path, "w") as f:
        for k in range(N_TICKS):
            row = {
                "step": k,
                "q_meas_rad": q_meas[k].tolist(),
                "q_target_delta_rad": q_target_delta[k].tolist(),
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {path} ({N_TICKS} ticks, injected d_true={d_true})")
