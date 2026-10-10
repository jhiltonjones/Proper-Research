"""Read a contact study and answer the question it was run to answer.

    python analyse_contact_study.py results/contact_study

Four verdicts, each stated so that a reader can see what would have falsified
it.  The analyser is deliberately willing to return "no" — a script that can
only confirm is not evidence, and the ``NOT SHOWN`` outcomes below are real
outcomes, not error paths.

**V1 — the sensitivity itself changes** (offline stage)
    Falsified if the contact and contact-free Jacobians agree everywhere.
    Three outcomes rather than two, because the middle one is easy to
    mis-report:

    ``structural``  the row spaces rotate.  Under contact a different set of
                    joint motions reaches the tip, so the controller's idea of
                    which directions are free is wrong.  No additive
                    disturbance can produce this.
    ``gain-only``   the Jacobians differ in magnitude but the row spaces do
                    not rotate.  Still not a disturbance — an estimator cannot
                    correct a gain error — but a weaker claim, and it must be
                    reported as the weaker claim.
    ``none``        they agree.  The study has nothing to find; stop here.

**V2 — what the observer calls a disturbance is the missing model term**
    (closed-loop stage, contact-free cells)
    ``d_hat`` is regressed on the configuration tracking error.  Falsified if
    that regression is weak (the residual really is exogenous) or if it is
    strong but its coefficient matrix does not match the independently measured
    Jacobian gap (something else is driving it).  Both failure modes are
    reported separately, because they mean different things.

**V3 — the penalty grows with contact** (closed-loop stage, radius arm)
    Falsified if the contact-free penalty is flat in measured engagement.  Uses
    only the radius sweep: those anatomies share one plan, so engagement is the
    only thing that varied.  Confounded (tortuosity) cells are excluded by
    construction and the report says how many were set aside.

**V4 — no observer setting rescues it** (observer stage)
    Falsified if some ``disturbance_filter_alpha`` brings the contact-free
    controller to the contact-aware ceiling.  This is the one that closes off
    "you just tuned the estimator badly", so it is worth running even when V2
    is already decisive.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent / "analysis", _HERE / "analysis"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import contact_evidence as EV  # noqa: E402

Array = np.ndarray

ROTATION_DEGREES = 1.0        # below this, call it gain-only, not structural
MAGNITUDE_FRACTION = 0.01     # below this, call it no difference at all
R_SQUARED_FLOOR = 0.5
RECOVERY_CEILING = 0.5
CEILING_FRACTION = 0.20       # "reaches the ceiling" = within 20% of it


def _load(name: str) -> ModuleType:
    for module_name in (f"analysis.{name}", name):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    for root in (_HERE, _HERE.parent, _HERE.parent / "analysis"):
        path = root / f"{name}.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"_an_{name}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
    raise ModuleNotFoundError(f"Could not import {name}.")


C = _load("_stack_common")


# ==========================================================================
# loading
# ==========================================================================
def _rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _ok(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


# ==========================================================================
# V1 — offline
# ==========================================================================
def verdict_jacobians_differ(summary: dict[str, Any]) -> dict[str, Any]:
    per_anatomy = {}
    for name, entry in summary.items():
        rotation = float(entry["largest_principal_angle_deg"]["max"])
        magnitude = float(entry["frobenius_relative"]["median"])
        if magnitude < MAGNITUDE_FRACTION and rotation < ROTATION_DEGREES:
            classification = "none"
        elif rotation >= ROTATION_DEGREES:
            classification = "structural"
        else:
            classification = "gain-only"
        per_anatomy[name] = {
            "classification": classification,
            "largest_principal_angle_deg": rotation,
            "median_relative_gap": magnitude,
            "max_relative_gap": float(entry["frobenius_relative"]["max"]),
            "gain_ratio_smallest_singular_value": entry[
                "gain_ratio_smallest_singular_value"
            ],
        }
    classifications = [v["classification"] for v in per_anatomy.values()]
    overall = (
        "structural" if "structural" in classifications
        else "gain-only" if "gain-only" in classifications
        else "none"
    )
    return {"overall": overall, "per_anatomy": per_anatomy}


# ==========================================================================
# V2 — the disturbance regression
# ==========================================================================
def verdict_disturbance_or_model(index_rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Aggregate the per-run regression over every contact-free MPC cell.

    Restricted to MPC cells on purpose: the resolved-rate baseline runs no
    disturbance estimator, so it has no ``d_hat`` to test and including it
    would only dilute the count with NaN.
    """
    candidates = [
        row for row in _ok(index_rows)
        if row.get("jacobian_model") == "no_contact"
        and row.get("controller", "").startswith("mpc")
        and row.get("disturbance_verdict", "")
    ]
    if not candidates:
        return {"outcome": "not tested", "cells": 0,
                "detail": "No contact-free MPC cell recorded a disturbance verdict."}

    r_squared = np.array([_number(row, "disturbance_r_squared") for row in candidates])
    recovery = np.array(
        [_number(row, "gap_recovery_relative_error") for row in candidates]
    )
    verdicts = [row["disturbance_verdict"] for row in candidates]
    structural = sum(1 for v in verdicts if v.startswith("structural"))
    confirmed = int(np.sum((r_squared >= R_SQUARED_FLOOR) & (recovery <= RECOVERY_CEILING)))
    explained_not_recovered = int(
        np.sum((r_squared >= R_SQUARED_FLOOR) & (recovery > RECOVERY_CEILING))
    )

    if confirmed >= max(1, len(candidates) // 2):
        outcome = "confirmed"
        detail = (
            f"In {confirmed} of {len(candidates)} contact-free MPC cells the "
            "estimated disturbance is explained by the configuration tracking "
            "error AND the fitted coefficient matrix reproduces the "
            "independently measured Jacobian gap. The estimator is not "
            "tracking an exogenous signal; it is absorbing the model term the "
            "controller left out."
        )
    elif explained_not_recovered > confirmed:
        outcome = "explained but not recovered"
        detail = (
            f"{explained_not_recovered} cells show d_hat strongly explained by "
            "tracking error, but the fitted matrix does not match the measured "
            "Jacobian gap. Something input-dependent is driving the residual, "
            "and contact is not established as the whole of it. Check whether "
            "nominal_reference_positions_m was passed — the planner's own tip "
            "error lands in d_hat otherwise."
        )
    else:
        outcome = "not shown"
        detail = (
            "d_hat is not well explained by the configuration tracking error. "
            "On this evidence it behaves like an exogenous disturbance, which "
            "is the disturbance hypothesis, not the structural one."
        )
    return {
        "outcome": outcome,
        "cells": len(candidates),
        "confirmed_cells": confirmed,
        "explained_not_recovered_cells": explained_not_recovered,
        "structural_labelled": structural,
        "median_r_squared": float(np.nanmedian(r_squared)),
        "median_gap_recovery_error": float(np.nanmedian(recovery)),
        "detail": detail,
    }


# ==========================================================================
# V3 — dose response
# ==========================================================================
def _engagement_by_anatomy(
    index_rows: Sequence[dict[str, str]], offline: dict[str, Any]
) -> tuple[dict[str, float], str]:
    """Prefer a measured dose; fall back to the knob, and say which was used."""
    measured: dict[str, float] = {}
    for name, entry in offline.items():
        engagement = entry.get("engagement", {})
        for key in ("contact_energy_J", "contacting_node_fraction", "gap_min_m"):
            stats = engagement.get(key)
            if not stats:
                continue
            value = stats.get("median")
            if value is not None and np.isfinite(value):
                measured[name] = abs(float(value))
                break
    if len(measured) >= 3:
        return measured, "measured contact engagement from the beam's own diagnostics"

    knob: dict[str, float] = {}
    for row in index_rows:
        scale = _number(row, "radius_scale")
        if np.isfinite(scale) and scale > 0:
            knob[row["anatomy"]] = 1.0 / scale - 1.0
    return knob, (
        "1/radius_scale - 1, a KNOB not a measurement — the beam models "
        "reported no contact diagnostics, so the dose axis is nominal"
    )


def verdict_dose_response(
    index_rows: Sequence[dict[str, str]], offline: dict[str, Any], metric: str
) -> dict[str, Any]:
    rows = [row for row in _ok(index_rows) if row.get("confounded", "0") != "1"]
    dropped = len(_ok(index_rows)) - len(rows)
    engagement_map, dose_note = _engagement_by_anatomy(rows, offline)

    per_controller: dict[str, Any] = {}
    controllers = sorted({row["controller"] for row in rows})
    for controller in controllers:
        anatomies, free, aware = [], [], []
        for anatomy in sorted({row["anatomy"] for row in rows}):
            def pick(model: str) -> float:
                values = [
                    _number(row, metric) for row in rows
                    if row["anatomy"] == anatomy
                    and row["controller"] == controller
                    and row["jacobian_model"] == model
                ]
                values = [v for v in values if np.isfinite(v)]
                return float(np.mean(values)) if values else math.nan

            free_value, aware_value = pick("no_contact"), pick("contact")
            dose = engagement_map.get(anatomy, math.nan)
            if np.isfinite(free_value) and np.isfinite(aware_value) and np.isfinite(dose):
                anatomies.append(dose)
                free.append(free_value)
                aware.append(aware_value)
        if len(anatomies) < 3:
            per_controller[controller] = {
                "outcome": "not tested",
                "anatomies": len(anatomies),
                "detail": "Fewer than three anatomies have both Jacobian models.",
            }
            continue
        order = np.argsort(anatomies)
        fit = EV.engagement_dose_response(
            engagement=np.asarray(anatomies)[order],
            contact_free_metric=np.asarray(free)[order],
            contact_aware_metric=np.asarray(aware)[order],
            metric_name=metric,
        )
        fit["outcome"] = "rises" if fit["penalty_rises_with_engagement"] else "flat"
        per_controller[controller] = fit

    rising = [k for k, v in per_controller.items() if v.get("outcome") == "rises"]
    tested = [k for k, v in per_controller.items() if v.get("outcome") in {"rises", "flat"}]
    return {
        "outcome": (
            "rises" if rising else "flat" if tested else "not tested"
        ),
        "rising_controllers": rising,
        "tested_controllers": tested,
        "confounded_cells_excluded": dropped,
        "dose_axis": dose_note,
        "per_controller": per_controller,
    }


# ==========================================================================
# V4 — observer ceiling
# ==========================================================================
def verdict_observer_ceiling(
    observer_rows: Sequence[dict[str, str]], metric: str
) -> dict[str, Any]:
    rows = _ok(observer_rows)
    if not rows:
        return {"outcome": "not run", "detail": "No observer stage in this study."}
    ceiling_rows = [row for row in rows if row.get("jacobian_model") == "contact"]
    sweep = [row for row in rows if row.get("jacobian_model") == "no_contact"]
    if not ceiling_rows or len(sweep) < 2:
        return {"outcome": "incomplete",
                "detail": "The observer stage needs the contact-aware ceiling cell "
                          "and at least two alpha values."}
    ceiling = float(np.nanmean([_number(row, metric) for row in ceiling_rows]))
    points = sorted(
        ((_number(row, "disturbance_alpha"), _number(row, metric)) for row in sweep),
        key=lambda pair: pair[0],
    )
    best_alpha, best_value = min(points, key=lambda pair: pair[1])
    shortfall = (best_value - ceiling) / max(abs(ceiling), 1e-12)
    reaches = shortfall <= CEILING_FRACTION
    return {
        "outcome": "observer closes the gap" if reaches else "no alpha rescues it",
        "ceiling_metric": ceiling,
        "best_alpha": best_alpha,
        "best_metric": best_value,
        "shortfall_relative": float(shortfall),
        "alpha_sweep": [{"alpha": a, metric: v} for a, v in points],
        "detail": (
            f"The best filter setting (alpha = {best_alpha:g}) puts the "
            "contact-free controller "
            + (
                f"{100.0 * shortfall:.1f}% above the contact-aware ceiling. "
                if shortfall > 0
                else f"{abs(100.0 * shortfall):.1f}% BELOW the contact-aware "
                "ceiling — it matched or beat the model that knows about "
                "contact. "
            )
            + (
                "That is inside the tolerance for calling the observer "
                "sufficient, so on this evidence contact CAN be treated as a "
                "disturbance here."
                if reaches
                else "Tuning the estimator does not substitute for modelling "
                "the contact."
            )
        ),
    }


# ==========================================================================
# the ladder table
# ==========================================================================
def controller_ladder(
    index_rows: Sequence[dict[str, str]], metric: str
) -> list[dict[str, Any]]:
    rows = _ok(index_rows)
    out = []
    for anatomy in sorted({row["anatomy"] for row in rows}):
        for model in sorted({row["jacobian_model"] for row in rows}):
            entry: dict[str, Any] = {"anatomy": anatomy, "jacobian_model": model}
            for controller in (
                "naive_inverse_jacobian", "mpc_lti", "mpc_ltv_offline",
                "mpc_ltv_sqp_online",
            ):
                values = [
                    _number(row, metric) for row in rows
                    if row["anatomy"] == anatomy
                    and row["jacobian_model"] == model
                    and row["controller"] == controller
                ]
                values = [v for v in values if np.isfinite(v)]
                entry[controller] = float(np.mean(values)) if values else math.nan
                times = [
                    _number(row, "p95_solve_time_ms") for row in rows
                    if row["anatomy"] == anatomy
                    and row["jacobian_model"] == model
                    and row["controller"] == controller
                ]
                times = [v for v in times if np.isfinite(v)]
                entry[f"{controller}_p95_ms"] = float(np.mean(times)) if times else math.nan
            out.append(entry)
    return out


# ==========================================================================
# figures
# ==========================================================================
def draw_figures(
    *,
    study_dir: Path,
    divergence_rows: Sequence[dict[str, str]],
    index_rows: Sequence[dict[str, str]],
    dose: dict[str, Any],
    observer: dict[str, Any],
    metric: str,
) -> list[Path]:
    figure, axes = C.new_figure(2, 2, (13.0, 9.0))
    if figure is None:
        return []
    written: list[Path] = []

    # (a) the two Jacobians, per anatomy: magnitude and rotation together.
    axis = axes[0]
    anatomies = sorted({row["anatomy"] for row in divergence_rows})
    for anatomy in anatomies:
        subset = [row for row in divergence_rows if row["anatomy"] == anatomy]
        gap = [100.0 * _number(row, "frobenius_relative") for row in subset]
        axis.plot(range(len(gap)), gap, linewidth=1.4, label=anatomy)
    axis.set_title("(a) contact vs contact-free Jacobian, along the path")
    axis.set_xlabel("reference sample (strided)")
    axis.set_ylabel("relative gap  ‖ΔJ‖/‖J‖  [%]")
    axis.legend(fontsize=7)

    axis = axes[1]
    for anatomy in anatomies:
        subset = [row for row in divergence_rows if row["anatomy"] == anatomy]
        angle = [_number(row, "principal_angle_3_deg") for row in subset]
        axis.plot(range(len(angle)), angle, linewidth=1.4, label=anatomy)
    axis.axhline(ROTATION_DEGREES, color="crimson", linestyle="--", linewidth=1.0)
    axis.set_title("(b) largest principal angle — a disturbance cannot rotate this")
    axis.set_xlabel("reference sample (strided)")
    axis.set_ylabel("angle [deg]")
    axis.legend(fontsize=7)

    # (c) the dose response: the headline figure.
    axis = axes[2]
    drawn = False
    for controller, fit in dose.get("per_controller", {}).items():
        if fit.get("outcome") not in {"rises", "flat"}:
            continue
        engagement = np.asarray(fit["engagement"], dtype=float)
        axis.plot(engagement, fit["contact_free"], "o-", linewidth=1.6,
                  label=f"{controller} · contact-free")
        axis.plot(engagement, fit["contact_aware"], "s--", linewidth=1.2, alpha=0.7,
                  label=f"{controller} · contact-aware")
        drawn = True
    axis.set_title("(c) does the contact-free penalty grow with contact?")
    axis.set_xlabel(dose.get("dose_axis", "contact engagement")[:60])
    axis.set_ylabel(metric.replace("_", " "))
    if drawn:
        axis.legend(fontsize=6)
    else:
        axis.text(0.5, 0.5, "not enough anatomies", ha="center", transform=axis.transAxes)

    # (d) the observer ceiling.
    axis = axes[3]
    sweep = observer.get("alpha_sweep") or []
    if sweep:
        alphas = [point["alpha"] for point in sweep]
        values = [point[metric] for point in sweep]
        axis.plot(alphas, values, "o-", linewidth=1.6, label="contact-free, swept α")
        axis.axhline(observer["ceiling_metric"], color="seagreen", linestyle="--",
                     linewidth=1.4, label="contact-aware ceiling")
        axis.set_xlabel("disturbance_filter_alpha")
        axis.set_ylabel(metric.replace("_", " "))
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "observer stage not run", ha="center", transform=axis.transAxes)
    axis.set_title("(d) can any filter setting reach the contact-aware ceiling?")

    path = study_dir / "contact_study_figures.png"
    C.save_figure(figure, path)
    written.append(path)
    return written


# ==========================================================================
# report
# ==========================================================================
def build_report(
    *,
    manifest: dict[str, Any],
    v1: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
    v4: dict[str, Any],
    ladder: Sequence[dict[str, Any]],
    metric: str,
    figures: Sequence[Path],
) -> Any:
    report = C.Report("Contact study — is contact a disturbance, or is it model?")
    report.metrics = {
        "manifest": manifest,
        "v1_jacobians_differ": v1,
        "v2_disturbance_or_model": v2,
        "v3_dose_response": v3,
        "v4_observer_ceiling": v4,
        "ladder": list(ladder),
        "metric": metric,
    }

    # ---- verdicts --------------------------------------------------------
    if v1["overall"] == "structural":
        worst = max(
            v1["per_anatomy"].items(),
            key=lambda item: item[1]["largest_principal_angle_deg"],
        )
        report.finding(
            "ok", "V1 — contact changes the sensitivity, not just the output",
            f"On {worst[0]} the row spaces of the two Jacobians differ by up to "
            f"{worst[1]['largest_principal_angle_deg']:.2f} deg, with a median "
            f"relative gap of {100 * worst[1]['median_relative_gap']:.2f}%. A "
            "rotated row space means a different set of joint motions reaches "
            "the tip under contact. No additive output disturbance can do that.",
        )
    elif v1["overall"] == "gain-only":
        report.finding(
            "warn", "V1 — the Jacobians differ, but only in magnitude",
            "The row spaces do not rotate measurably, so the claim available "
            "here is the weaker one: contact is a gain error, not a "
            "redirection. An observer still cannot correct a gain error, but "
            "do not describe this result as a change in the reachable "
            "directions.",
        )
    else:
        report.finding(
            "fail", "V1 — the two Jacobians agree",
            "There is no measurable contact effect on the sensitivity in this "
            "anatomy set. The rest of the study cannot establish the claim, "
            "and the honest report is that contact behaves as a disturbance "
            "here. Check that the radius sweep actually reaches contact.",
        )

    level = {"confirmed": "ok", "explained but not recovered": "warn",
             "not shown": "fail", "not tested": "info"}[v2["outcome"]]
    report.finding(
        level, f"V2 — the estimated disturbance ({v2['outcome']})", v2["detail"]
    )

    if v3["outcome"] == "rises":
        report.finding(
            "ok", "V3 — the penalty grows with contact engagement",
            f"The contact-free penalty rises with engagement for "
            f"{', '.join(v3['rising_controllers'])}. A disturbance an observer "
            "absorbs would give a flat line once converged.",
        )
    elif v3["outcome"] == "flat":
        report.finding(
            "warn", "V3 — no detectable dose response",
            "The contact-free penalty does not measurably grow with "
            "engagement. Either the anatomy sweep did not vary contact enough, "
            "or the estimator is coping at these levels.",
        )
    else:
        report.finding(
            "info", "V3 — dose response not tested",
            "Fewer than three anatomies carry both Jacobian models.",
        )

    if v4["outcome"] == "no alpha rescues it":
        report.finding("ok", "V4 — tuning the observer does not substitute", v4["detail"])
    elif v4["outcome"] == "observer closes the gap":
        report.finding("fail", "V4 — the observer IS sufficient here", v4["detail"])
    else:
        report.finding("info", f"V4 — observer stage {v4['outcome']}",
                       v4.get("detail", ""))

    # ---- body ------------------------------------------------------------
    report.head("What this study compared")
    report.text(
        "Four controllers, two Jacobian models, several anatomies. **The plant "
        "is contact-enabled in every cell.** Nothing here simulates a beam that "
        "does not touch the wall; the Jacobian factor changes only what the "
        "controller believes. A result obtained by also disabling contact in "
        "the plant would be an artefact, and the runner refuses to produce one.",
        "The anatomy sweep scales the lumen radius with the centreline held "
        "fixed, so one planned trajectory stays valid across the whole sweep "
        "and engagement is the only thing that varies. Any cell whose "
        "centreline changed is marked `confounded` and excluded from the "
        f"dose response — {v3.get('confounded_cells_excluded', 0)} were set aside.",
    )
    report.bullets(
        [
            f"anatomies: {', '.join(a['anatomy'] for a in manifest.get('anatomies', []))}",
            f"controllers: {', '.join(manifest.get('controllers', []))}",
            f"Jacobian models: {', '.join(manifest.get('jacobian_models', []))}",
            f"initial conditions per cell: {manifest.get('initial_conditions', 1)}",
            f"dose axis: {v3.get('dose_axis', 'n/a')}",
        ]
    )

    report.head("V1 — the two Jacobians, per anatomy")
    report.table(
        ["anatomy", "classification", "median gap [%]", "max gap [%]",
         "largest principal angle [deg]"],
        [
            [name, entry["classification"], 100 * entry["median_relative_gap"],
             100 * entry["max_relative_gap"], entry["largest_principal_angle_deg"]]
            for name, entry in v1["per_anatomy"].items()
        ],
    )
    report.text(
        "Read magnitude and angle together. A large gap with no rotation is a "
        "gain error; a rotation is a change in which joint motions reach the "
        "tip. Only the second rules out every additive-disturbance model, but "
        "neither is something a disturbance estimator can fix."
    )

    report.head("V2 — is d_hat the missing model term?")
    report.text(
        "The beam-output MPC estimates `d_hat = p_meas - (p_nom + J_free (z - z_ref))`. "
        "If the plant is really `p_nom + J_contact (z - z_ref)`, then algebraically "
        "`d_hat = (J_contact - J_free)(z - z_ref)`: not an exogenous signal at all, "
        "but a linear function of the controller's own tracking error, with the "
        "Jacobian gap as its coefficient matrix. The regression tests exactly that, "
        "and the recovery check compares the fitted matrix against the gap measured "
        "independently along the same run."
    )
    report.bullets(
        [
            f"contact-free MPC cells tested: {v2.get('cells', 0)}",
            f"median R² of d_hat on tracking error: {v2.get('median_r_squared', float('nan')):.4f}",
            f"median relative error recovering the Jacobian gap: "
            f"{v2.get('median_gap_recovery_error', float('nan')):.4f}",
        ]
    )

    report.head("V3 — dose response")
    for controller, fit in v3.get("per_controller", {}).items():
        if fit.get("outcome") not in {"rises", "flat"}:
            continue
        interval = fit["slope_90_percent_interval"]
        report.text(
            f"**{controller}** — slope {fit['slope']:.4g} "
            f"(90% bootstrap {interval[0]:.4g} to {interval[1]:.4g}), "
            f"R² {fit['r_squared']:.3f} over {fit['anatomies']} anatomies. "
            f"{fit['verdict']}"
        )

    report.head("The controller ladder")
    report.text(
        "Each rung differs from the one below in exactly one thing. Read "
        "accuracy and worst-case solve time together: an accuracy win you "
        "cannot compute inside one sample period is not a win."
    )
    report.table(
        ["anatomy", "Jacobian", "naive", "LTI", "LTV offline", "SQP online",
         "SQP p95 [ms]"],
        [
            [entry["anatomy"], entry["jacobian_model"],
             entry["naive_inverse_jacobian"], entry["mpc_lti"],
             entry["mpc_ltv_offline"], entry["mpc_ltv_sqp_online"],
             entry["mpc_ltv_sqp_online_p95_ms"]]
            for entry in ladder
        ],
    )
    report.text(f"Values are {metric.replace('_', ' ')}, averaged over initial conditions.")

    report.head("V4 — the observer ceiling")
    if v4.get("alpha_sweep"):
        report.table(
            ["disturbance_filter_alpha", metric],
            [[point["alpha"], point[metric]] for point in v4["alpha_sweep"]],
        )
        report.text(
            f"Contact-aware ceiling: {v4['ceiling_metric']:.4f}. "
            f"Best contact-free result: {v4['best_metric']:.4f} at alpha "
            f"{v4['best_alpha']:g}."
        )
    else:
        report.text(v4.get("detail", "Observer stage not run."))

    report.head("What would have falsified each claim")
    report.bullets(
        [
            "**V1** — the two Jacobians agreeing to a fraction of a percent, "
            "with no row-space rotation, on every anatomy.",
            "**V2** — d_hat poorly explained by the configuration tracking "
            "error, or well explained but by a matrix unlike the measured "
            "Jacobian gap.",
            "**V3** — a flat penalty in engagement, with a bootstrap slope "
            "interval containing zero.",
            "**V4** — some filter setting bringing the contact-free controller "
            f"within {100 * CEILING_FRACTION:.0f}% of the contact-aware ceiling.",
        ]
    )
    if figures:
        report.head("Figures")
        report.bullets([f"`{path.name}`" for path in figures])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a contact study.")
    parser.add_argument("study_dir", type=Path)
    parser.add_argument(
        "--metric", default="rms_beam_position_error_mm",
        help="The tracking metric the verdicts are computed on.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        raise SystemExit(f"No such study directory: {study_dir}")
    output_dir = Path(args.output_dir or study_dir)

    manifest_path = study_dir / "study_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    offline_path = study_dir / "offline" / "jacobian_divergence_summary.json"
    offline = (
        json.loads(offline_path.read_text(encoding="utf-8"))
        if offline_path.exists() else {}
    )
    divergence_rows = _rows(study_dir / "offline" / "jacobian_divergence.csv")
    index_rows = _rows(study_dir / "study_index.csv")
    observer_rows = _rows(study_dir / "observer" / "study_index.csv")

    if not offline and not index_rows:
        raise SystemExit(
            f"{study_dir} holds neither an offline stage nor a closed-loop index. "
            "Run run_contact_study.py first."
        )

    v1 = (
        verdict_jacobians_differ(offline) if offline
        else {"overall": "none", "per_anatomy": {},
              "note": "offline stage not run"}
    )
    v2 = verdict_disturbance_or_model(index_rows)
    v3 = verdict_dose_response(index_rows, offline, args.metric)
    v4 = verdict_observer_ceiling(observer_rows, args.metric)
    ladder = controller_ladder(index_rows, args.metric)

    figures = draw_figures(
        study_dir=output_dir, divergence_rows=divergence_rows, index_rows=index_rows,
        dose=v3, observer=v4, metric=args.metric,
    )
    report = build_report(
        manifest=manifest, v1=v1, v2=v2, v3=v3, v4=v4, ladder=ladder,
        metric=args.metric, figures=figures,
    )
    paths = report.write(output_dir, "contact_study")
    print(f"\nContact study -> {paths['report']}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
