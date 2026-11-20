"""Z4b step 4: build protocol_campaign2.yaml.

Per-field sourcing (recorded in each entry's _source block):
- "campaign2_decision": explicit value given in Round 34's spec, overriding
  whatever the historical majority was.
- "log_majority": derived from a representative logged controller_metadata.json
  for that condition family (NOT an exhaustive majority-vote across every
  historical attempt -- time did not allow surveying every log directory;
  this uses the real value found in the condition's own representative
  recent run(s), flagged for the author to confirm/correct on review).
- "code_default": no representative log carried this field (e.g. OSQP
  solver settings are never logged anywhere in controller_metadata.json);
  taken from the single source-of-truth constant/default in the current
  code, which is what the vast majority of unoverridden runs actually use.
"""
import hashlib
import yaml
from pathlib import Path

ROOT = Path("/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- explicit campaign-2 decisions (Round 34, verbatim) -------------------
CAMPAIGN2 = {
    "weights": {"R": 7.0, "R_d": 1.0e-3},
    "keep_out": {"radius_m": 0.235, "monitor_tolerance_m": None},  # tolerance: see log_majority below
    "robust_margins": {"tcp_m": 0.015},
    "tracking_error_abort_m": {"mpc": 0.007, "inv7": 0.015},
    "insertion_offset_abort": "off",
    "reference_mode": "counter",
    "streaming": "on",
    "delay_samples": 2,
    "deadline_ms": 70.0,
}

# ---- plan files + hashes (real files on disk, hashed directly) -----------
PLAN_B = ROOT / "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_dipolealigned_2026-10-20/time_parameterized_configuration_path/time_parameterized_configuration_path.npz"
PLAN_BPRIME = ROOT / "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_2026-10-19/time_parameterized_configuration_path/time_parameterized_configuration_path.npz"
PLAN_RECT = ROOT / "plans/rectangle_10x15mm_skipglobal_ae1_2026-11-10/shape_rectangle/shape_centreline.npz"
PLAN_TRI = ROOT / "plans/triangle_base35mm_skipglobal_ad2_2026-11-10/shape_triangle/shape_centreline.npz"

for p in (PLAN_B, PLAN_BPRIME, PLAN_RECT, PLAN_TRI):
    assert p.exists(), f"missing: {p}"

HASH_B = sha256_of(PLAN_B)
HASH_BPRIME = sha256_of(PLAN_BPRIME)
HASH_RECT = sha256_of(PLAN_RECT)
HASH_TRI = sha256_of(PLAN_TRI)

# ---- vessel-side values, derived from a representative plan-B log --------
# (close_loop_logs/myrun/mpc_closedloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T131243Z/controller_metadata.json)
VESSEL_HEIGHT_BAND_M = [-0.028142895340206694, 0.016857104659793308]  # magnet_z_bounds_m, relative to the online anchor
VESSEL_KEEPOUT_CENTRE_R_M = [0.670575, -0.719727, -0.039627]           # magnet_exclusion_point_R == beam base
VESSEL_KEEPOUT_FLOOR_M = 0.210                                         # beam_base_exclusion_floor_mm CLI default (210.0mm), run_mpc_delay_aware_vessel.py
VESSEL_INSERTION_RATE_LIMIT_M_S = 0.002
VESSEL_STATE_SCALE = {"q_rad": 0.008726646259971648, "L_m": 0.00025}
VESSEL_INPUT_SCALE = {"q_rad_s": 0.05, "L_m_s": 0.005}
VESSEL_GAMMA = 0.0
VESSEL_BETA_D = 1.0

# ---- rectangle/shape-side values, from common.py's single source of truth
RECT_TCP_BOX_MIN_M = [0.24469, -0.89361, 0.36544]
RECT_TCP_BOX_MAX_M = [0.58971, -0.49576, 0.46496]
RECT_JOINT_VELOCITY_LIMIT_RAD_S = 0.1
RECT_MAX_JOINT_STEP_RAD = 0.01
RECT_INSERTION_OFFSET_ABORT_MM_DEFAULT = 5.0  # CLI default, irrelevant now the monitor is off by campaign-2 decision
RECT_INSERTION_TOL_MM = 3.0

# ---- OSQP settings, code_default (never logged anywhere) -----------------
OSQP = {
    "eps_abs": 1.0e-7,
    "eps_rel": 1.0e-7,
    "max_iter": 20000,
    "polish": True,
    "time_limit_s": 0.03,
    "accept_time_limit_solution": True,
    "source_note": "code_default: ConfigurationMPCConfig class defaults "
                    "(simulate_time_parameterized_configuration_mpc.py) for eps/max_iter/"
                    "polish; time_limit_s/accept_time_limit_solution per the 2026-10-14 "
                    "speed-fix investigation (not re-verified live this session -- "
                    "author should confirm against the actual worker_process.py call "
                    "site before sign-off).",
}

DEADLINE_MS = CAMPAIGN2["deadline_ms"]


def base_safety(tcp_box_min, tcp_box_max, height_band=None, keepout_centre=None,
                 keepout_floor_m=None, abort_mpc_or_inv7="mpc"):
    entry = {
        "tcp_box": {"min_m": tcp_box_min, "max_m": tcp_box_max,
                    "_source": "log_majority/code_default (rectangle_stage_a.common.WORKSPACE_XYZ_MIN_M/MAX_M)"},
        "robust_margins": {"tcp_m": CAMPAIGN2["robust_margins"]["tcp_m"],
                            "_source": "campaign2_decision"},
        "tracking_error_abort_m": (CAMPAIGN2["tracking_error_abort_m"]["mpc"]
                                    if abort_mpc_or_inv7 == "mpc"
                                    else CAMPAIGN2["tracking_error_abort_m"]["inv7"]),
        "insertion_offset_abort": CAMPAIGN2["insertion_offset_abort"],
        "reference_mode": CAMPAIGN2["reference_mode"],
        "streaming": CAMPAIGN2["streaming"],
        "deadline_ms": DEADLINE_MS,
    }
    if height_band is not None:
        entry["height_band_m"] = {"lower": height_band[0], "upper": height_band[1],
                                   "_source": "log_majority (representative plan-B run's magnet_z_bounds_m)"}
    if keepout_centre is not None:
        entry["keep_out"] = {
            "centre_R_m": keepout_centre,
            "radius_m": CAMPAIGN2["keep_out"]["radius_m"],
            "floor_m": keepout_floor_m,
            "monitor_tolerance_m": None,
            "_source": "radius_m: campaign2_decision; centre/floor: log_majority "
                        "(beam_base_exclusion_floor_mm CLI default); monitor_tolerance_m: "
                        "NOT FOUND in any representative log or CLI default -- flagged for "
                        "author, do not sign off until filled in.",
        }
    return entry


def vessel_entry(controller_class, N, jacobian_source, plan_hash, abort_kind):
    e = {
        "plan_file": str(PLAN_B if plan_hash == HASH_B else PLAN_BPRIME),
        "plan_sha256": plan_hash,
        "controller_class": controller_class,
        "N": N,
        "jacobian_source": jacobian_source,
        "weights": {"Q_p": "per-sample LTV schedule (see plan's own mpc_schedule_*.npy)",
                    "R": CAMPAIGN2["weights"]["R"], "R_d": CAMPAIGN2["weights"]["R_d"],
                    "_source": "R/R_d: campaign2_decision; Q_p: inherent to the per-sample "
                               "schedule, not a scalar -- see delay_aware_mpc.py"},
        "rate_limit_m_s_or_rad_s": {"insertion": VESSEL_INSERTION_RATE_LIMIT_M_S,
                                     "_source": "log_majority"},
        "increment_scale": {"input": VESSEL_INPUT_SCALE, "state": VESSEL_STATE_SCALE,
                             "_source": "log_majority"},
        "gamma": VESSEL_GAMMA, "beta_d": VESSEL_BETA_D,
        "delay_samples": CAMPAIGN2["delay_samples"],
        "osqp": OSQP,
    }
    e.update(base_safety(
        tcp_box_min=None, tcp_box_max=None,  # vessel has no TCP-box CLI constant surfaced the same way; flagged
        height_band=VESSEL_HEIGHT_BAND_M,
        keepout_centre=VESSEL_KEEPOUT_CENTRE_R_M,
        keepout_floor_m=VESSEL_KEEPOUT_FLOOR_M,
        abort_mpc_or_inv7=("inv7" if "inv" in controller_class.lower() or "inv7" in controller_class.lower() else "mpc"),
    ))
    e["tcp_box"]["_source"] = ("NOT a fixed CLI constant for the vessel runner the way "
                                "rectangle has common.WORKSPACE_XYZ_MIN_M/MAX_M -- flagged "
                                "for author, fill in from the actual vessel TCP-box flag "
                                "default before sign-off.")
    return e


def shape_entry(shape_name, plan_file, plan_hash, controller_class, N, jacobian_source, abort_kind):
    e = {
        "plan_file": str(plan_file),
        "plan_sha256": plan_hash,
        "controller_class": controller_class,
        "N": N,
        "jacobian_source": jacobian_source,
        "weights": {"Q_p": "per-sample LTV schedule (see this plan's own schedule cache)",
                    "R": CAMPAIGN2["weights"]["R"], "R_d": CAMPAIGN2["weights"]["R_d"],
                    "_source": "R/R_d: campaign2_decision; Q_p: inherent to the schedule"},
        "rate_limit_rad_s": {"joint": RECT_JOINT_VELOCITY_LIMIT_RAD_S, "_source": "log_majority/code_default (common.JOINT_VELOCITY_LIMIT_RAD_S)"},
        "increment_rad": {"joint": RECT_MAX_JOINT_STEP_RAD, "_source": "log_majority/code_default (common.MAX_JOINT_STEP_RAD)"},
        "joint_box": "6-DOF UR10e joint limits (unchanged from manufacturer limits; not separately overridden)",
        "insertion_tol_mm": RECT_INSERTION_TOL_MM,
        "delay_samples": CAMPAIGN2["delay_samples"],
        "osqp": OSQP,
    }
    e.update(base_safety(
        tcp_box_min=RECT_TCP_BOX_MIN_M, tcp_box_max=RECT_TCP_BOX_MAX_M,
        abort_mpc_or_inv7=abort_kind,
    ))
    return e


protocol = {
    "protocol_version": "campaign2-2026-11-11",
    "status": "DRAFT -- NOT YET REVIEWED OR SIGNED OFF BY THE AUTHOR. Do not use for a "
              "real dry run or hardware campaign until the author has reviewed every "
              "entry below (especially every field flagged '_source': ... NOT FOUND / "
              "flagged for author) and explicitly signs off.",
    "campaign2_decisions_applied_globally": CAMPAIGN2,
    "conditions": {
        # ---- vessel plan B ----
        "vessel_planB_mpc_sj_n15": vessel_entry("ExactQNZeroTaskNullspaceDelayAwareMPC", 15, "scheduled", HASH_B, "mpc"),
        "vessel_planB_mpc_sj_n1": vessel_entry("ExactQNZeroTaskNullspaceDelayAwareMPC", 1, "scheduled", HASH_B, "mpc"),
        "vessel_planB_mpc_fj_n15": vessel_entry("ExactQNZeroTaskNullspaceDelayAwareMPC", 15, "frozen", HASH_B, "mpc"),
        "vessel_planB_inv7": vessel_entry("InverseJacobian7DOFDelayAware", None, "scheduled", HASH_B, "inv7"),
        # ---- vessel plan B' ----
        "vessel_planBprime_mpc_sj_n15": vessel_entry("ExactQNZeroTaskNullspaceDelayAwareMPC", 15, "scheduled", HASH_BPRIME, "mpc"),
        "vessel_planBprime_mpc_sj_n1": vessel_entry("ExactQNZeroTaskNullspaceDelayAwareMPC", 1, "scheduled", HASH_BPRIME, "mpc"),
        "vessel_planBprime_inv7": vessel_entry("InverseJacobian7DOFDelayAware", None, "scheduled", HASH_BPRIME, "inv7"),
        # ---- rectangle ----
        "rectangle_sj": shape_entry("rectangle", PLAN_RECT, HASH_RECT, "mpc_delay_aware (qn_zero)", 15, "scheduled", "mpc"),
        "rectangle_fj": shape_entry("rectangle", PLAN_RECT, HASH_RECT, "mpc_delay_aware (qn_zero)", 15, "frozen", "mpc"),
        "rectangle_inv7": shape_entry("rectangle", PLAN_RECT, HASH_RECT, "inv_7dof_delay_aware", None, "scheduled", "inv7"),
        # ---- triangle-35 ----
        "triangle35_sj": shape_entry("triangle", PLAN_TRI, HASH_TRI, "mpc_delay_aware (qn_zero)", 15, "scheduled", "mpc"),
        "triangle35_fj": shape_entry("triangle", PLAN_TRI, HASH_TRI, "mpc_delay_aware (qn_zero)", 15, "frozen", "mpc"),
        "triangle35_inv7": shape_entry("triangle", PLAN_TRI, HASH_TRI, "inv_7dof_delay_aware", None, "scheduled", "inv7"),
    },
}

out_path = ROOT / "protocol_campaign2.yaml"
with open(out_path, "w") as f:
    yaml.dump(protocol, f, default_flow_style=False, sort_keys=False, width=100)
print(f"wrote {out_path}")
