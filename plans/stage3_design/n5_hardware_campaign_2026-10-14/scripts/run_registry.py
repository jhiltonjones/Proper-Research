"""Shared registry of the 15 floor240 hardware runs used throughout the
n=5 three-condition comparison (plan: snug-bouncing-giraffe.md,
post-speed-fix round). SUPERSEDES the earlier n=3 registry (that
campaign was invalidated by the timing regression fixed in
contact_jacobian_hybrid.py's project_s_q -- see this plan's Context).
These 15 runs were given directly by the user as "these 5 reps" per
condition, confirmed read-only (all path_complete, zero
deadline_miss) before being registered here. No top-level execution
besides this registry -- safe to import from any script.
"""
import json
import os

RUN_ROOT = "/home/jack/Proper-Research/close_loop_logs/myrun"

CONTACT_PLAN_DIR = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_2026-10-12/time_parameterized_configuration_path"
NOCONTACT_PLAN_DIR = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_nocontact_2026-10-12/time_parameterized_configuration_path"

# condition -> list of (rep_number, run_dir_name), rep numbers assigned
# in EXACTLY the order the user listed each condition's 5 runs -- not
# reordered by timestamp or renamed. NEW's "v4" appears twice (two
# distinct real runs at different timestamps); kept as rep3/rep4 in
# listed order, per the user's own framing of "these are my 5 reps."
RUNS = {
    "NEW": [
        (1, "mpc_closedloop_contact_floor240_hybrid_j1j3_zfloor5mm_v2_20261014T172324Z"),
        (2, "mpc_closedloop_contact_floor240_hybrid_j1j3_zfloor5mm_v3_20261014T172608Z"),
        (3, "mpc_closedloop_contact_floor240_hybrid_j1j3_zfloor5mm_v4_20261014T172844Z"),
        (4, "mpc_closedloop_contact_floor240_hybrid_j1j3_zfloor5mm_v4_20261014T174000Z"),
        (5, "mpc_closedloop_contact_floor240_hybrid_j1j3_zfloor5mm_v5_20261014T175843Z"),
    ],
    "OLD-C": [
        (1, "mpc_closedloop_contact_floor240_schedule_compare_zfloor5mm_20261014T174329Z"),
        (2, "mpc_closedloop_contact_floor240_schedule_compare_zfloor5mm_v2_20261014T174605Z"),
        (3, "mpc_closedloop_contact_floor240_schedule_compare_zfloor5mm_v3_20261014T174902Z"),
        (4, "mpc_closedloop_contact_floor240_schedule_compare_zfloor5mm_v4_20261014T175144Z"),
        (5, "mpc_closedloop_contact_floor240_schedule_compare_zfloor5mm_v5_20261014T180141Z"),
    ],
    "NC": [
        (1, "mpc_closedloop_nocontact_floor240_schedule_compare_20261014T173148Z"),
        (2, "mpc_closedloop_nocontact_floor240_schedule_compare_v2_20261014T173500Z"),
        (3, "mpc_closedloop_nocontact_floor240_schedule_compare_v3_20261014T173728Z"),
        (4, "mpc_closedloop_nocontact_floor240_schedule_compare_v4_20261014T175546Z"),
        (5, "mpc_closedloop_nocontact_floor240_schedule_compare_v5_20261014T182151Z"),
    ],
}

# Hybrid map constants (from the committed production map, not refit here).
HYBRID_MAP_DIR = "/home/jack/Proper-Research/plans/stage3_design/contact_jacobian_maps/floor240_2026-10-12"
R_A = 1.7910914672700304
R_B = 11.1579367241426

DT = 0.1  # common.CONTROL_HZ = 10.0 Hz, confirmed for all 9 runs
HORIZON = 15
DELAY_SAMPLES = 2

PLAN_ROOT_CONTACT = "/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_2026-10-12"
PLAN_ROOT_NOCONTACT = "/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_nocontact_2026-10-12"


def run_dir(condition, rep):
    for r, d in RUNS[condition]:
        if r == rep:
            return os.path.join(RUN_ROOT, d)
    raise KeyError((condition, rep))


def all_runs():
    """Yields (condition, rep, run_path) for all 15 runs."""
    for cond, reps in RUNS.items():
        for rep, d in reps:
            yield cond, rep, os.path.join(RUN_ROOT, d)


def load_summary(run_path):
    with open(os.path.join(run_path, "summary.json")) as f:
        return json.load(f)


def load_controller_metadata(run_path):
    with open(os.path.join(run_path, "controller_metadata.json")) as f:
        return json.load(f)


def load_path_follow(run_path):
    """Returns a list of per-tick dicts from path_follow.jsonl, in order."""
    rows = []
    with open(os.path.join(run_path, "path_follow.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plan_root_for(condition):
    return PLAN_ROOT_NOCONTACT if condition == "NC" else PLAN_ROOT_CONTACT
