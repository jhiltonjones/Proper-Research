"""H3b follow-up: pipeline-stage null-space decomposition.

Reviewer concern on the existing h3b_null_redundant_projection.csv (built by
h3b_workspace_and_nullspace.py): that analysis projects u = out["u0"] -- the
LOGGED, EXECUTED command, i.e. after BOTH the velocity/acceleration/state-box
clip AND the external selective gate -- into task/null components. A damped
least-squares task-space solution should itself lie almost entirely in the
row-space of J by construction (low E_N). If E_N's observed growth
(~0.02-0.03 early -> 0.21-0.38 pre-failure) is actually injected by clipping
or gating rather than being intrinsic to the controller's own task-space
solve, that is a different (and important) distinction.

This script reconstructs, at every tick of the same 6 selective-gate runs
h3b_workspace_and_nullspace.py used, THREE pipeline stages:
    u_raw     -- damped-least-squares task_velocity, before ANY clip
                 (nullspace_gain=0, feedforward=False -- confirmed from each
                 run's own controller_metadata/summary.json config, same as
                 h3_replay.py's replay_raw_commands)
    u_clipped -- after the velocity/acceleration/state-box clip (exact same
                 clip_command as h3_replay.py, using the CLIPPED previous
                 command recursively, matching the real control loop)
    u_gated   -- the actually logged/executed out["u0"] (post external
                 selective gate) -- ground truth, no replay needed

and projects each onto the task Jacobian's null space (P_N = I - J^+ J,
J = live_jac.live_jacobian(state7, contact=<run's own pairing>) -- the same
J_p convention h3b_workspace_and_nullspace.py used) to get E_N for each
stage, at the same early (s<30mm) / pre-failure (last 12 ticks) regions.

Read-only against hardware logs, manifest/loader/live_jac, and h3_replay.py's
OWN logic (reproduced here, not imported, since h3_replay.py returns only
the post-clip array, not the pre-clip intermediate). Writes only new
table/figure/script files; never modifies h3_replay.py, h3b_workspace_and_
nullspace.py, proper_research/**, or any run log.
"""
import sys, time
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest, loader, live_jac

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
REPO = loader.REPO

DT = 0.1
VEL_LIMIT = np.array([0.1] * 6 + [0.002])
ACC_LIMIT = np.array([0.4] * 6 + [10.0 * 0.002])
STATE_MIN = np.array([-2 * np.pi] * 6 + [-0.05])
STATE_MAX = np.array([2 * np.pi] * 6 + [0.20])

SCHED_C_REPAIRED = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy"
SCHED_C_ORIG = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy"
SCHED_NC = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy"


def clip_command(command, state, previous):
    command = np.clip(command, -VEL_LIMIT, VEL_LIMIT)
    command = np.clip(command, previous - ACC_LIMIT * DT, previous + ACC_LIMIT * DT)
    command = np.clip(command, (STATE_MIN - state) / DT, (STATE_MAX - state) / DT)
    return np.clip(command, -VEL_LIMIT, VEL_LIMIT)


def replay_raw_and_clipped(out, meta, schedule, desired_position_m):
    """Exact reproduction of h3_replay.py's replay_raw_commands, but
    returning BOTH the pre-clip task_velocity (u_raw) and the post-clip
    command (u_clipped) per tick, using the real clipped-previous-command
    recursion (matching the actual control loop, which only ever sees its
    own clipped output as "previous")."""
    N = len(out["step"])
    position_gain = float(meta["position_gain"])
    damping = float(meta["damping"])
    n_sched = schedule.shape[0]
    n_ref = desired_position_m.shape[0]

    u_raw = np.zeros((N, 7))
    u_clipped = np.zeros((N, 7))
    prev_clipped = np.zeros(7)
    q = out["q_meas_rad"]
    L = out["insertion_length_m"]
    tip_m = out["tip_mm"] / 1000.0
    ref_idx = out["ref_index"]

    for k in range(N):
        state = np.concatenate([q[k], [L[k]]])
        measured = tip_m[k]
        index = int(np.clip(ref_idx[k] + 1, 0, min(n_sched, n_ref) - 1))
        J = schedule[index]
        desired = desired_position_m[index]

        gram = J @ J.T + (damping ** 2) * np.eye(3)
        pseudo = J.T @ np.linalg.solve(gram, np.eye(3))
        task_velocity = pseudo @ (position_gain * (desired - measured) / DT)

        command = clip_command(task_velocity, state, prev_clipped)
        u_raw[k] = task_velocity
        u_clipped[k] = command
        prev_clipped = command.copy()

    return u_raw, u_clipped


sel_c = sorted([rm for rm in manifest.RUNS if rm["group"] == "invjac_selective"], key=lambda r: r["rep"])
sel_nc = sorted([rm for rm in manifest.RUNS if rm["group"] == "invjac_clip_nc"], key=lambda r: r["rep"])
PAIRINGS = [("selective_contact", sel_c, SCHED_C_REPAIRED, True),
            ("selective_nocontact", sel_nc, SCHED_NC, False)]

t0 = time.time()
live_jac.get_context()
print(f"[pipeline-stage] live context ready at {time.time()-t0:.0f}s", flush=True)

rows = []
for pairing_name, runs, sched_path, contact_model in PAIRINGS:
    schedule = np.load(sched_path)
    for rm in runs:
        out, meta = loader.enrich_run(rm)
        ref = loader.get_reference(meta["plan_dir"])
        desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)

        u_raw, u_clipped = replay_raw_and_clipped(out, meta, schedule, desired_position_m)
        u_gated = out["u0"]

        # sanity: does u_clipped match the logged u0 wherever the gate did
        # NOT intervene (i.e. gate had nothing to zero)? Use this as a
        # consistency check on the replay, not a pass/fail gate.
        gate_untouched = np.all(np.isclose(u_clipped, u_gated, atol=1e-6), axis=1)
        frac_untouched = float(np.mean(gate_untouched))

        N = len(out["step"])
        s = out["s_ref_mm"]
        q6 = out["q_meas_rad"]
        L = out["insertion_length_m"]
        early_idx = np.where(s < 30.0)[0]
        tail_idx = np.arange(max(0, N - 12), N)
        pick = sorted(set(list(early_idx[::max(1, len(early_idx) // 10)]) + list(tail_idx)))

        for k in pick:
            state7 = np.concatenate([q6[k], [L[k]]])
            Jp = live_jac.live_jacobian(state7, contact=contact_model)
            Jp_pinv = np.linalg.pinv(Jp)
            P_N = np.eye(7) - Jp_pinv @ Jp
            region = "early" if k in early_idx else "pre_failure"

            for stage_name, uk in (("u_raw", u_raw[k]), ("u_clipped", u_clipped[k]), ("u_gated", u_gated[k])):
                u_N = P_N @ uk
                denom = max(float(np.dot(uk, uk)), 1e-18)
                E_N = float(np.dot(u_N, u_N) / denom)
                rows.append(dict(
                    pairing=pairing_name, dirname=meta["dirname"], rep=rm["rep"], tick=int(out["step"][k]),
                    s_mm=float(s[k]), region=region, stage=stage_name, E_N=E_N,
                    norm_u=float(np.linalg.norm(uk)), frac_gate_untouched_wholerun=frac_untouched,
                ))
        print(f"[pipeline-stage] {meta['dirname']}: {len(pick)} samples, "
              f"gate left {frac_untouched:.1%} of ticks byte-identical to clipped command "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/h3b_pipeline_stage_nullspace.csv", index=False)
print(f"\nwrote {OUT}/tables/h3b_pipeline_stage_nullspace.csv ({len(df)} rows)")

summary = df.groupby(["pairing", "stage", "region"])["E_N"].agg(["mean", "max", "count"])
pd.set_option("display.width", 160)
print("\n=== E_N by pipeline stage, pairing, region ===")
print(summary.to_string())

# --- figure: one panel per pairing, E_N vs s for all 3 stages ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
stage_style = {"u_raw": ("o", "#1b7f3b"), "u_clipped": ("s", "#2f8fd1"), "u_gated": ("^", "#b3331d")}
for ax, (pairing_name, runs, sched_path, contact_model) in zip(axes, PAIRINGS):
    sub = df[df.pairing == pairing_name]
    for stage_name, (mk, color) in stage_style.items():
        ssub = sub[sub.stage == stage_name]
        ax.scatter(ssub.s_mm, ssub.E_N, marker=mk, color=color, s=22, alpha=0.7, label=stage_name)
    ax.set_title(pairing_name, fontsize=10)
    ax.set_xlabel("path progress s (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r"$E_N = \|u_N\|^2/\|u\|^2$")
fig.suptitle("H3b: null-space energy fraction at each pipeline stage (raw DLS solve / post-clip / post-gate=logged)", fontsize=11)
fig.tight_layout()
f = f"{OUT}/figures/h3b_pipeline_stage_nullspace.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print(f"\nsaved {f}")
