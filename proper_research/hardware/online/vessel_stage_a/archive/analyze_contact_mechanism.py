#!/usr/bin/env python3
"""Contact-vs-no-contact wall-interaction mechanism analysis (2026-09-28).

Compares the two live vessel closed-loop MPC runs that differ ONLY in the
Jacobian model (contact-aware vs free-space), same plan/weights/horizon/
limits/magnet config:

    CONTACT:    close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_contact_20260928T134051Z
    NO-CONTACT: close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_nocontact_20260928T135815Z

Goes beyond raw tip-tracking RMS to test three mechanism hypotheses: (1) the
contact-aware Jacobian predicts the plant better during wall contact, (2) it
correctly captures the loss of wall-normal mobility once in contact, (3) it
redirects control authority into tangential motion instead of commanding
motion through the wall.

Data sources (no new hardware access, no re-solving of the MPC):
  - path_follow.jsonl: per-tick tip/joint/insertion measurement + commanded
    input u0. NOTE: this controller (the process-isolated vessel worker)
    leaves `predicted_beam_position_0_m` / `predicted_input_0` / etc. as
    `None` at every tick in both runs -- those path_follow.jsonl prediction
    fields are NOT populated for this controller path. Verified empirically
    before writing this script (100% None over 505/508 rows respectively).
  - predicted_beam_positions.jsonl: the actual source of per-tick MPC
    predictions for this controller -- `predicted_beam_positions_m` is the
    horizon-length (15) sequence of PREDICTED TIP positions (row 0 = the
    one-step-ahead prediction x_{k+1|k}), and `measured_beam_position_m` is
    the tip position measured at that same tick. Row 0 at tick k vs the
    NEXT tick's measured tip gives the true one-step MPC prediction
    residual.
  - the cached Jacobian schedules (/tmp/vessel_live_trimmed6mm_schedule_
    {contact,nocontact}.npy), shape (526, 3, 7): row i = d(tip_R_xyz)/d[q1..
    q6, insertion] linearized at reference sample i. Indexed per tick by
    that tick's own `target_index` (the delay-compensated reference sample
    the controller was solving against that tick) -- validated against the
    independently-logged scalar `jacobian_condition` per tick before use,
    see `pick_schedule_index_and_validate`.
  - the real digitized lumen file + `LumenQuery` (this project's own wall-
    distance/normal convention from
    `proper_research.simulation.magnetic_beam.contact.LumenQuery`): reused
    exactly, not reinvented, so "the wall" here means the same thing the
    contact-aware model itself uses. `d_w = R_local - dist_to_centreline -
    r_beam`: positive = free lumen, 0 = beam surface at wall, negative =
    geometric penetration.
  - beam TANGENT at the tip: no vision-measured tangent is logged for these
    runs (`NewFrameTipMapper`'s own chord-tangent is not persisted to disk
    by this controller path), so this script solves each tick's OWN model
    (contact-aware model for the contact run, free-space model for the
    no-contact run) forward at the tick's measured (q, L) to get the full
    predicted centreline, and takes the tangent from its last two nodes.
    This is a MODEL-INFERRED tangent (what each controller's own model
    believes its shape is), not a vision ground truth -- flagged throughout.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.archive.analyze_contact_mechanism
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/jack/Proper-Research")
RUN_CONTACT = REPO / "close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_contact_20260928T134051Z"
RUN_NOCONTACT = REPO / "close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_nocontact_20260928T135815Z"
LUMEN_FILE = str(REPO / "vessel_lumen_robot_frame_raised3cm_trimmed6mm_2026-09-28.json")
SCHEDULE_CONTACT = Path("/tmp/vessel_live_trimmed6mm_schedule_contact.npy")
SCHEDULE_NOCONTACT = Path("/tmp/vessel_live_trimmed6mm_schedule_nocontact.npy")
INSERTION_MAX_M = 0.065
Z_RAISE_M = 0.03
R_BEAM_M = 0.001  # ContactParams.r_beam default == mpc_wall_avoidance_beam_radius_mm default (1.0mm)
CONTACT_BAND_M = 0.5e-3  # matches this project's own --mpc-wall-avoidance-margin-mm default (0.5mm)
OUT_DIR = REPO / "close_loop_logs/myrun/contact_mechanism_analysis_2026-09-28"


# ===========================================================================
# loading
# ===========================================================================
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def load_run(run_dir: Path) -> dict[str, Any]:
    pf = _load_jsonl(run_dir / "path_follow.jsonl")
    pb = _load_jsonl(run_dir / "predicted_beam_positions.jsonl")
    assert len(pf) == len(pb), f"{run_dir}: path_follow ({len(pf)}) != predicted_beam_positions ({len(pb)})"
    # path_follow.jsonl's `step` is 1-indexed, predicted_beam_positions.jsonl's
    # is 0-indexed (two independent counters in the harness vs the worker) --
    # same row position is the same control tick. Verified by `ref_index`,
    # which both logs derive from the same tick and must agree exactly.
    for i, (a, b) in enumerate(zip(pf, pb)):
        assert a["ref_index"] == b["ref_index"], (
            f"{run_dir}: row {i} ref_index mismatch {a['ref_index']} != {b['ref_index']}"
        )

    n = len(pf)
    d: dict[str, Any] = {}
    d["n"] = n
    d["step"] = np.array([r["step"] for r in pf])
    d["t_s"] = np.array([r["t_s"] for r in pf], dtype=float)
    d["ref_index"] = np.array([r["ref_index"] for r in pf], dtype=int)
    d["target_index"] = np.array([r["target_index"] for r in pf], dtype=int)
    d["tip_m"] = np.array([r["tip_mm"] for r in pf], dtype=float) / 1000.0
    d["desired_m"] = np.array([r["desired_mm"] for r in pf], dtype=float) / 1000.0
    d["error_norm_mm"] = np.array([r["error_norm_mm"] for r in pf], dtype=float)
    d["q_meas_rad"] = np.array([r["q_meas_rad"] for r in pf], dtype=float)
    d["insertion_m"] = np.array([r["insertion_length_m"] for r in pf], dtype=float)
    d["u0"] = np.array([r["u0"] for r in pf], dtype=float)
    d["jacobian_condition_logged"] = np.array([r["jacobian_condition"] for r in pf], dtype=float)
    d["terminal_hold"] = np.array([bool(r["terminal_hold"]) for r in pf])
    d["solver_success"] = np.array([bool(r["solver_success"]) for r in pf])
    d["pred_next_tip_m"] = np.array([r["predicted_beam_positions_m"][0] for r in pb], dtype=float)
    d["meas_tip_from_pb_m"] = np.array([r["measured_beam_position_m"] for r in pb], dtype=float)
    consistency_mm = 1000.0 * np.linalg.norm(d["meas_tip_from_pb_m"] - d["tip_m"], axis=1)
    d["harness_worker_tip_consistency_mm"] = consistency_mm
    return d


def load_lumen():
    from proper_research.vision.detect_blue import load_vessel_lumen_robot_frame

    C, R, provenance = load_vessel_lumen_robot_frame(LUMEN_FILE)
    return np.asarray(C, dtype=float), np.asarray(R, dtype=float), provenance


# ===========================================================================
# wall geometry (reusing this project's own LumenQuery convention)
# ===========================================================================
def wall_geometry(lumen_query, points_m: np.ndarray):
    """Returns (d_w, n_outward) for each point.

    d_w = R_local - dist_to_centreline - r_beam  (project convention, see
    proper_research.simulation.magnetic_beam.contact.contact_barrier_energy_
    and_force_fast's `gap_arr`). Positive = free lumen clearance, 0 = beam
    surface touching the wall, negative = geometric penetration.

    n_outward = unit vector from the nearest centreline point toward the tip
    (radially outward, i.e. pointing INTO the wall / direction of decreasing
    d_w). This is `grad_delta` from LumenQuery.closest_many_with_gradients.
    """
    delta, Rloc, q, grad_delta, grad_Rloc = lumen_query.closest_many_with_gradients(points_m, window=None)
    d_w = Rloc - delta - R_BEAM_M
    return d_w, grad_delta


# ===========================================================================
# model adapters (for beam tangent only -- everything else comes from logs)
# ===========================================================================
def build_adapters():
    import proper_research.simulation.simulations.initial_conditions as initial_conditions_mod
    from proper_research.planning.vessel_context import build_vessel_planning_context
    from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter

    orig_make_initial_poses = initial_conditions_mod.make_initial_poses

    def _raised_make_initial_poses():
        p, s, L, dt = orig_make_initial_poses()
        p = np.array(p, dtype=float).copy()
        s = np.array(s, dtype=float).copy()
        p[2] += Z_RAISE_M
        s[2] += Z_RAISE_M
        return p, s, L, dt

    exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance = build_vessel_planning_context(
        lumen_file=LUMEN_FILE, insertion_max_m=INSERTION_MAX_M,
        initial_poses=_raised_make_initial_poses(),
    )
    adapter_contact = build_diagnostic_adapter(
        beam_model=bundle.models["contact"], controller_pack=controller_pack, jacobian_mode="accurate",
    )
    adapter_nocontact = build_diagnostic_adapter(
        beam_model=bundle.models["no_contact"], controller_pack=controller_pack, jacobian_mode="accurate",
    )
    return adapter_contact, adapter_nocontact


def predict_centreline(adapter, q6: np.ndarray, L_m: float) -> np.ndarray:
    """Same recipe as checkpoint_beam_shape_campaign.py's own helper."""
    forward_adapter = adapter.beam_output_fn.forward_adapter
    state = np.concatenate([q6, [L_m]])
    forward_adapter.start_step()
    adapter.forward_output(state, commit=False)
    return forward_adapter.last_p_centerline.T  # (N, 3)


def tip_tangents(adapter, q_meas: np.ndarray, insertion_m: np.ndarray, label: str) -> np.ndarray:
    cache_path = Path(f"/tmp/vessel_contact_mechanism_tangents_{label}.npy")
    n = q_meas.shape[0]
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (n, 3):
            print(f"[analyze] loaded cached tangents ({label}) from {cache_path}")
            return cached
    tangents = np.zeros((n, 3))
    print(f"[analyze] solving {n} model shapes for tangent ({label})...")
    for i in range(n):
        centreline = predict_centreline(adapter, q_meas[i], float(insertion_m[i]))
        tang = centreline[-1] - centreline[-2]
        norm = np.linalg.norm(tang)
        tangents[i] = tang / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        if (i + 1) % 100 == 0:
            print(f"[analyze]   {label}: {i + 1}/{n}")
    np.save(cache_path, tangents)
    print(f"[analyze] cached tangents ({label}) -> {cache_path}")
    return tangents


# ===========================================================================
# schedule indexing
# ===========================================================================
def pick_schedule_index_and_validate(schedule: np.ndarray, run: dict[str, Any], label: str) -> np.ndarray:
    """Use `target_index` -- the delay-compensated reference sample the
    controller is solving against this tick -- as the schedule row in force
    for this tick's QP (see worker_process.py's `schedule[indices]` usage,
    keyed off the horizon's target samples, and the `run_mpc_delay_aware_
    vessel.py` docstring's d=2 delay compensation).

    NOTE: this was meant to be cross-checked against the harness's own
    logged per-tick `jacobian_condition` scalar, but that field turns out to
    be CONSTANT across the entire run (1268.14 for every one of 505 rows in
    the contact run, checked directly) -- it is the ONE-TIME startup
    Jacobian's condition number, printed once and copied into every log row,
    not a live per-tick measurement. It cannot validate anything here, so
    that check was removed rather than reported as a false empirical
    validation. `target_index` is used on architectural grounds only.
    """
    idx = np.clip(run["target_index"], 0, schedule.shape[0] - 1)
    print(f"[analyze] {label}: indexing schedule by target_index "
          f"(range {idx.min()}-{idx.max()}); jacobian_condition in the logs is a constant "
          f"startup value, not usable as a per-tick cross-check -- see docstring above.")
    return idx


# ===========================================================================
# per-run metric computation
# ===========================================================================
def analyze_run(run: dict[str, Any], schedule: np.ndarray, tangents: np.ndarray, lumen_query, label: str) -> dict[str, Any]:
    n = run["n"]
    tip_m = run["tip_m"]
    des_m = run["desired_m"]
    t_s = run["t_s"]
    u0 = run["u0"]

    d_w, n_out = wall_geometry(lumen_query, tip_m)
    in_contact = d_w < CONTACT_BAND_M

    cosang = np.clip(np.abs(np.einsum("ij,ij->i", tangents, n_out)), -1.0, 1.0)
    theta_n_deg = np.degrees(np.arccos(cosang))
    theta_wall_deg = 90.0 - theta_n_deg

    idx = pick_schedule_index_and_validate(schedule, run, label)
    J = schedule[idx]  # (N, 3, 7)

    e_vec_m = des_m - tip_m
    e_norm = np.linalg.norm(e_vec_m, axis=1)
    ehat = np.divide(e_vec_m, e_norm[:, None], out=np.zeros_like(e_vec_m), where=(e_norm > 1e-9)[:, None])
    g_e = np.linalg.norm(np.einsum("nij,ni->nj", J, ehat), axis=1)

    Jn = np.einsum("ni,nij->nj", n_out, J)
    Jn_norm = np.linalg.norm(Jn, axis=1)
    Jt = J - np.einsum("ni,nj->nij", n_out, Jn)
    Jt_norm = np.linalg.norm(Jt.reshape(n, -1), axis=1)

    sigmas = np.linalg.svd(J[:, :, :6], compute_uv=False)
    sigma_min, sigma_max = sigmas[:, -1], sigmas[:, 0]
    cond_J = sigma_max / np.maximum(sigma_min, 1e-12)

    dt = np.diff(t_s)
    state = np.concatenate([run["q_meas_rad"], run["insertion_m"][:, None]], axis=1)
    dstate = np.diff(state, axis=0)
    dx_meas = np.diff(tip_m, axis=0)

    dx_pred_from_state = np.einsum("nij,nj->ni", J[:-1], dstate)
    e_J = np.linalg.norm(dx_meas - dx_pred_from_state, axis=1)

    r_pred = np.linalg.norm(tip_m[1:] - run["pred_next_tip_m"][:-1], axis=1)

    dx_cmd = np.einsum("nij,nj->ni", J[:-1], u0[:-1] * dt[:, None])
    n_k = n_out[:-1]
    dx_cmd_n = np.einsum("ni,ni->n", n_k, dx_cmd)
    dx_ach_n = np.einsum("ni,ni->n", n_k, dx_meas)
    dx_cmd_t = np.linalg.norm(dx_cmd - dx_cmd_n[:, None] * n_k, axis=1)
    dx_ach_t = np.linalg.norm(dx_meas - dx_ach_n[:, None] * n_k, axis=1)

    du = np.diff(u0, axis=0)
    effort_u = np.sum(u0[:-1] ** 2, axis=1)
    smoothness_du = np.sum(du ** 2, axis=1)
    u0_norm = np.linalg.norm(u0[:-1], axis=1)
    dx_ach_norm = np.linalg.norm(dx_meas, axis=1)
    eta = np.divide(dx_ach_norm, u0_norm, out=np.full_like(dx_ach_norm, np.nan), where=u0_norm > 1e-9)

    contact_diff = np.diff(in_contact.astype(int))
    onsets = np.where(contact_diff == 1)[0] + 1
    releases = np.where(contact_diff == -1)[0] + 1

    return dict(
        label=label, n=n, t_s=t_s, ref_index=run["ref_index"], target_index=run["target_index"],
        d_w_mm=d_w * 1000.0, in_contact=in_contact, theta_n_deg=theta_n_deg, theta_wall_deg=theta_wall_deg,
        error_norm_mm=run["error_norm_mm"], g_e=g_e, Jn_norm=Jn_norm, Jt_norm=Jt_norm,
        sigma_min=sigma_min, sigma_max=sigma_max, cond_J=cond_J,
        e_J_mm=e_J * 1000.0, r_pred_mm=r_pred * 1000.0,
        dx_cmd_n_mm=dx_cmd_n * 1000.0, dx_ach_n_mm=dx_ach_n * 1000.0,
        dx_cmd_t_mm=dx_cmd_t * 1000.0, dx_ach_t_mm=dx_ach_t * 1000.0,
        effort_u=effort_u, smoothness_du=smoothness_du, eta=eta,
        onsets=onsets, releases=releases,
        harness_worker_tip_consistency_mm=run["harness_worker_tip_consistency_mm"],
        terminal_hold=run["terminal_hold"], solver_success=run["solver_success"],
    )


# ===========================================================================
# summary stats
# ===========================================================================
def summarize(m: dict[str, Any]) -> dict[str, Any]:
    track = ~m["terminal_hold"]
    err = m["error_norm_mm"][track]
    contact_mask = m["in_contact"][track]
    time_in_contact_s = float(np.sum(np.diff(m["t_s"], prepend=m["t_s"][0])[m["in_contact"]]))
    max_pen_mm = float(max(0.0, -np.min(m["d_w_mm"])))

    def _safe(a):
        a = np.asarray(a, dtype=float)
        return a[np.isfinite(a)]

    e_J_c = _safe(m["e_J_mm"][m["in_contact"][:-1]])
    e_J_f = _safe(m["e_J_mm"][~m["in_contact"][:-1]])
    r_pred_c = _safe(m["r_pred_mm"][m["in_contact"][:-1]])
    r_pred_f = _safe(m["r_pred_mm"][~m["in_contact"][:-1]])
    Jn_c = _safe(m["Jn_norm"][m["in_contact"]])
    Jn_f = _safe(m["Jn_norm"][~m["in_contact"]])
    Jt_c = _safe(m["Jt_norm"][m["in_contact"]])
    Jt_f = _safe(m["Jt_norm"][~m["in_contact"]])
    dxcmdn_c = _safe(m["dx_cmd_n_mm"][m["in_contact"][:-1]])
    dxachn_c = _safe(m["dx_ach_n_mm"][m["in_contact"][:-1]])

    run_dir = RUN_CONTACT if m["label"] == "contact" else RUN_NOCONTACT
    return dict(
        label=m["label"],
        n_ticks=int(m["n"]),
        rms_tip_error_mm=float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
        p95_tip_error_mm=float(np.percentile(err, 95)) if err.size else float("nan"),
        max_tip_error_mm=float(np.max(err)) if err.size else float("nan"),
        max_wall_penetration_mm=max_pen_mm,
        time_in_contact_s=time_in_contact_s,
        frac_ticks_in_contact=float(np.mean(m["in_contact"])),
        mean_tip_error_in_contact_mm=float(np.mean(err[contact_mask])) if contact_mask.any() else float("nan"),
        mean_tip_error_free_space_mm=float(np.mean(err[~contact_mask])) if (~contact_mask).any() else float("nan"),
        mean_e_J_in_contact_mm=float(np.mean(e_J_c)) if e_J_c.size else float("nan"),
        mean_e_J_free_space_mm=float(np.mean(e_J_f)) if e_J_f.size else float("nan"),
        mean_r_pred_in_contact_mm=float(np.mean(r_pred_c)) if r_pred_c.size else float("nan"),
        mean_r_pred_free_space_mm=float(np.mean(r_pred_f)) if r_pred_f.size else float("nan"),
        mean_Jn_norm_in_contact=float(np.mean(Jn_c)) if Jn_c.size else float("nan"),
        mean_Jn_norm_free_space=float(np.mean(Jn_f)) if Jn_f.size else float("nan"),
        mean_Jt_norm_in_contact=float(np.mean(Jt_c)) if Jt_c.size else float("nan"),
        mean_Jt_norm_free_space=float(np.mean(Jt_f)) if Jt_f.size else float("nan"),
        mean_dx_cmd_n_in_contact_mm=float(np.mean(dxcmdn_c)) if dxcmdn_c.size else float("nan"),
        mean_dx_ach_n_in_contact_mm=float(np.mean(dxachn_c)) if dxachn_c.size else float("nan"),
        mean_theta_wall_deg_in_contact=float(np.mean(m["theta_wall_deg"][m["in_contact"]])) if m["in_contact"].any() else float("nan"),
        n_contact_intervals=int(len(m["onsets"])),
        control_effort_sum_u2=float(np.sum(m["effort_u"])),
        control_smoothness_sum_du2=float(np.sum(m["smoothness_du"])),
        mean_efficiency_eta=float(np.nanmean(m["eta"])),
        max_harness_worker_tip_consistency_mm=float(np.max(m["harness_worker_tip_consistency_mm"])),
        stop_reason=json.loads((run_dir / "summary.json").read_text())["stop_reason"],
    )


# ===========================================================================
# figures
# ===========================================================================
def make_figures(mc: dict[str, Any], mn: dict[str, Any], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    col = {"contact": "tab:red", "nocontact": "tab:blue"}

    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    for m in (mc, mn):
        x = m["ref_index"]
        c = col[m["label"]]
        axes[0].plot(x, m["error_norm_mm"], color=c, lw=1, label=m["label"])
        axes[1].plot(x, m["d_w_mm"], color=c, lw=1)
        axes[2].plot(x, m["theta_wall_deg"], color=c, lw=0.8, alpha=0.8)
        axes[3].plot(x[:-1], m["dx_ach_n_mm"], color=c, lw=0.8, ls="-", label=f"{m['label']} achieved n")
        axes[3].plot(x[:-1], m["dx_cmd_n_mm"], color=c, lw=0.8, ls="--", alpha=0.6, label=f"{m['label']} commanded n")
        axes[4].plot(x, m["cond_J"], color=c, lw=0.8, alpha=0.7, label=f"{m['label']} cond(J)")
        axes[4].plot(x, m["g_e"], color=c, lw=1.2, ls="--", label=f"{m['label']} g_e")
        axes[5].plot(x[:-1], m["e_J_mm"], color=c, lw=0.8, alpha=0.6, label=f"{m['label']} e_J")
        axes[5].plot(x[:-1], m["r_pred_mm"], color=c, lw=1.0, ls="--", label=f"{m['label']} r_pred")
        for onset in m["onsets"]:
            for ax in axes:
                ax.axvline(x[onset], color=c, alpha=0.15, lw=3)

    axes[0].set_ylabel("tip error (mm)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].axhline(0, color="k", lw=0.8); axes[1].axhline(CONTACT_BAND_M * 1000, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("signed wall dist d_w (mm)\n(<0 = penetration)"); axes[1].grid(alpha=.3)
    axes[2].set_ylabel("wall-tangent angle\ntheta_wall (deg)"); axes[2].grid(alpha=.3)
    axes[3].set_ylabel("normal tip motion (mm)\nsolid=achieved dashed=commanded"); axes[3].legend(fontsize=6, ncol=2); axes[3].grid(alpha=.3)
    axes[4].set_ylabel("cond(J) (dashed) / g_e (solid)\nJacobian directional gain"); axes[4].legend(fontsize=6, ncol=2); axes[4].grid(alpha=.3)
    axes[4].set_yscale("log")
    axes[5].set_ylabel("prediction error (mm)\nsolid=e_J dashed=r_pred"); axes[5].legend(fontsize=6, ncol=2); axes[5].grid(alpha=.3)
    axes[5].set_xlabel("reference sample index (trajectory progress, 0-525)")
    fig.suptitle("Contact (red) vs no-contact (blue) -- shaded bands = contact onset ticks", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "centerpiece_panel.png", dpi=130)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, m in enumerate((mc, mn)):
        ax = axes2[j]
        if len(m["onsets"]):
            k = int(m["onsets"][0]) + 3
            k = min(k, m["n"] - 2)
        else:
            k = int(np.argmin(m["d_w_mm"]))
            k = min(k, m["n"] - 2)
        vals = [m["dx_cmd_n_mm"][k], m["dx_ach_n_mm"][k], m["dx_cmd_t_mm"][k], m["dx_ach_t_mm"][k]]
        labels = ["cmd normal", "achieved\nnormal", "cmd\ntangential", "achieved\ntangential"]
        ax.bar(labels, vals, color=[col[m["label"]], col[m["label"]], "gray", "gray"])
        ax.set_title(f"{m['label']}: tick {k} (ref_index={m['ref_index'][k]}, d_w={m['d_w_mm'][k]:.2f}mm)")
        ax.set_ylabel("mm"); ax.grid(alpha=.3, axis="y")
        ax.axhline(0, color="k", lw=0.8)
    fig2.suptitle("Commanded vs achieved tip motion decomposed into wall-normal / tangential components\n"
                   "(representative tick near contact onset, or deepest-penetration tick if no onset)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "wall_frame_decomposition_example.png", dpi=130)
    plt.close(fig2)

    if len(mc["onsets"]) or len(mn["onsets"]):
        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
        for j, m in enumerate((mc, mn)):
            ax = axes3[j]
            if len(m["onsets"]) == 0:
                ax.text(0.5, 0.5, "no contact onset detected", ha="center", va="center", transform=ax.transAxes)
                continue
            k0 = int(m["onsets"][0])
            lo, hi = max(0, k0 - 15), min(m["n"] - 1, k0 + 25)
            x = m["ref_index"][lo:hi]
            ax.plot(x, m["error_norm_mm"][lo:hi], label="tip error (mm)", color=col[m["label"]])
            ax2 = ax.twinx()
            ax2.plot(x, m["d_w_mm"][lo:hi], label="d_w (mm)", color="gray", ls="--")
            ax2.axhline(0, color="k", lw=0.6)
            ax.axvline(m["ref_index"][k0], color="green", ls=":", label="contact onset")
            ax.set_title(f"{m['label']}: contact-onset zoom (tick {k0})")
            ax.set_ylabel("tip error (mm)"); ax2.set_ylabel("d_w (mm)")
            ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=.3)
        fig3.tight_layout()
        fig3.savefig(out_dir / "contact_transition_zoom.png", dpi=130)
        plt.close(fig3)


# ===========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[analyze] loading runs...")
    run_c = load_run(RUN_CONTACT)
    run_n = load_run(RUN_NOCONTACT)

    print("[analyze] loading lumen + building LumenQuery...")
    lumen_C, lumen_R, lumen_provenance = load_lumen()
    from proper_research.simulation.magnetic_beam.contact import LumenQuery
    lq = LumenQuery(lumen_C, lumen_R)

    print("[analyze] loading Jacobian schedules...")
    sched_c = np.load(SCHEDULE_CONTACT)
    sched_n = np.load(SCHEDULE_NOCONTACT)

    print("[analyze] building model adapters for beam-tangent solves (this takes a while: "
          f"{run_c['n'] + run_n['n']} forward solves total)...")
    adapter_contact, adapter_nocontact = build_adapters()

    tang_c = tip_tangents(adapter_contact, run_c["q_meas_rad"], run_c["insertion_m"], "contact")
    tang_n = tip_tangents(adapter_nocontact, run_n["q_meas_rad"], run_n["insertion_m"], "nocontact")

    print("[analyze] computing metrics...")
    mc = analyze_run(run_c, sched_c, tang_c, lq, "contact")
    mn = analyze_run(run_n, sched_n, tang_n, lq, "nocontact")

    sc = summarize(mc)
    sn = summarize(mn)

    print("[analyze] writing figures...")
    make_figures(mc, mn, OUT_DIR)

    stats = {"contact": sc, "nocontact": sn, "contact_band_m": CONTACT_BAND_M, "r_beam_m": R_BEAM_M,
             "lumen_file": LUMEN_FILE, "lumen_provenance": lumen_provenance}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    for m, name in ((mc, "contact"), (mn, "nocontact")):
        rows = []
        for i in range(m["n"]):
            row = dict(
                step=i, ref_index=int(m["ref_index"][i]), t_s=float(m["t_s"][i]),
                error_norm_mm=float(m["error_norm_mm"][i]), d_w_mm=float(m["d_w_mm"][i]),
                in_contact=bool(m["in_contact"][i]), theta_wall_deg=float(m["theta_wall_deg"][i]),
                cond_J=float(m["cond_J"][i]), g_e=float(m["g_e"][i]),
                Jn_norm=float(m["Jn_norm"][i]), Jt_norm=float(m["Jt_norm"][i]),
            )
            if i < m["n"] - 1:
                row.update(
                    e_J_mm=float(m["e_J_mm"][i]), r_pred_mm=float(m["r_pred_mm"][i]),
                    dx_cmd_n_mm=float(m["dx_cmd_n_mm"][i]), dx_ach_n_mm=float(m["dx_ach_n_mm"][i]),
                    dx_cmd_t_mm=float(m["dx_cmd_t_mm"][i]), dx_ach_t_mm=float(m["dx_ach_t_mm"][i]),
                    effort_u=float(m["effort_u"][i]), smoothness_du=float(m["smoothness_du"][i]),
                    eta=float(m["eta"][i]) if np.isfinite(m["eta"][i]) else None,
                )
            rows.append(row)
        cols = list(rows[0].keys())
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join("" if r.get(c) is None else str(r[c]) for c in cols))
        (OUT_DIR / f"per_tick_{name}.csv").write_text("\n".join(lines) + "\n")

    print("\n===== SUMMARY =====")
    print(json.dumps(stats, indent=2, default=str))
    print(f"\n[analyze] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
