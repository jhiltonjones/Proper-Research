"""End-to-end test of the study pipeline, on data whose answer is known.

Two synthetic studies are built and analysed:

* a **structural** one, where contact really does change which joint motions
  reach the tip — the analyser must reach the structural verdict on all four
  questions;
* a **disturbance** one, where contact is a pure additive offset an observer
  can absorb — the analyser must NOT claim structure, and must say so plainly.

The second study is the point of this file. Any pipeline can confirm; the test
of an evidence pipeline is whether it will tell you no.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# Layout-tolerant: works whether this file sits beside the modules it tests
# (experiments/) or in a separate tests/ directory beside them.
for _root in (_HERE, _HERE.parent, _HERE.parent.parent):
    for _sub in ("", "experiments", "analysis", "controllers"):
        _path = _root / _sub if _sub else _root
        if _path.is_dir() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import analyse_contact_study as AN  # noqa: E402
import run_contact_study as RS  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((bool(condition), name, detail))


ANATOMIES = [
    {"anatomy": "radius_x1", "radius_scale": 1.0, "engagement": 0.10},
    {"anatomy": "radius_x0p75", "radius_scale": 0.75, "engagement": 0.40},
    {"anatomy": "radius_x0p55", "radius_scale": 0.55, "engagement": 0.85},
    {"anatomy": "radius_x0p4", "radius_scale": 0.40, "engagement": 1.60},
]
CONTROLLERS = list(RS.CONTROLLERS)


def _divergence_summary(structural: bool) -> dict:
    """The offline stage's summary, as run_offline_stage would have written it."""
    summary = {}
    for entry in ANATOMIES:
        dose = entry["engagement"]
        gap = 0.02 * dose if structural else 0.0
        angle = 4.0 * dose if structural else 0.0

        def stat(value):
            return {"max": value * 1.4, "rms": value * 1.1, "median": value}

        summary[entry["anatomy"]] = {
            "samples": 40,
            "frobenius_relative": stat(gap),
            "spectral_relative": stat(gap),
            "largest_principal_angle_deg": stat(angle),
            "smallest_principal_angle_deg": stat(0.0),
            "gain_ratio_smallest_singular_value": stat(1.0 + gap),
            "condition_contact": stat(30.0),
            "condition_contact_free": stat(30.0),
            "per_actuator_relative_max": [gap] * 7,
            "engagement": {"contact_energy_J": stat(dose * 1e-6),
                           "contacting_node_fraction": stat(min(1.0, 0.3 * dose))},
            "anatomy": entry["anatomy"],
        }
    return summary


def _index_rows(structural: bool) -> list[dict]:
    """One row per cell, in the schema run_cells writes."""
    rows = []
    generator = np.random.default_rng(3)
    for entry in ANATOMIES:
        dose = entry["engagement"]
        for model in ("no_contact", "contact"):
            for rung, controller in enumerate(CONTROLLERS):
                base = 0.40 - 0.06 * rung          # each rung helps a little
                if model == "no_contact":
                    # Structural: the penalty scales with contact.
                    # Disturbance: a fixed offset the estimator flattens out.
                    penalty = 0.55 * dose if structural else 0.05
                else:
                    penalty = 0.0
                value = base + penalty + 0.002 * generator.normal()
                is_mpc = controller.startswith("mpc")
                rows.append(
                    {
                        "anatomy": entry["anatomy"],
                        "jacobian_model": model,
                        "controller": controller,
                        "initial_condition": 0,
                        "disturbance_alpha": 0.0,
                        "tag": "",
                        "radius_scale": entry["radius_scale"],
                        "tortuosity_scale": 1.0,
                        "confounded": 0,
                        "status": "ok",
                        "failure": "",
                        "rms_beam_position_error_mm": value,
                        "p95_beam_position_error_mm": value * 1.6,
                        "maximum_beam_position_error_mm": value * 2.2,
                        "p95_solve_time_ms": 3.0 + 2.0 * rung,
                        "contact_free_jacobian": int(model == "no_contact"),
                        "jacobian_gap_norm": 0.02 * dose if structural else 0.0,
                        "disturbance_verdict": (
                            ("structural" if structural else "disturbance-like")
                            if (is_mpc and model == "no_contact") else ""
                        ),
                        "disturbance_r_squared": (
                            (0.93 if structural else 0.04)
                            if (is_mpc and model == "no_contact") else ""
                        ),
                        "gap_recovery_relative_error": (
                            (0.06 if structural else 0.95)
                            if (is_mpc and model == "no_contact") else ""
                        ),
                    }
                )
    return rows


def _observer_rows(structural: bool) -> list[dict]:
    """The alpha sweep plus the contact-aware ceiling."""
    rows = []
    ceiling = 0.28
    for alpha in (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98):
        if structural:
            # Filtering trades lag against noise and never reaches the ceiling.
            value = 1.16 - 0.10 * alpha + 0.35 * alpha**4
        else:
            # A real disturbance: a well-tuned observer absorbs it.
            value = 0.33 - 0.055 * alpha
        rows.append(
            {"anatomy": "radius_x0p4", "jacobian_model": "no_contact",
             "controller": "mpc_ltv_offline", "initial_condition": 0,
             "disturbance_alpha": alpha, "tag": f"alpha{alpha:g}",
             "confounded": 0, "status": "ok",
             "rms_beam_position_error_mm": value}
        )
    rows.append(
        {"anatomy": "radius_x0p4", "jacobian_model": "contact",
         "controller": "mpc_ltv_offline", "initial_condition": 0,
         "disturbance_alpha": 0.0, "tag": "ceiling", "confounded": 0,
         "status": "ok", "rms_beam_position_error_mm": ceiling}
    )
    return rows


def build_study(root: Path, structural: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "offline").mkdir(exist_ok=True)
    (root / "observer").mkdir(exist_ok=True)

    (root / "offline" / "jacobian_divergence_summary.json").write_text(
        json.dumps(_divergence_summary(structural), indent=2), encoding="utf-8"
    )
    common = AN.C
    divergence_rows = []
    for entry in ANATOMIES:
        dose = entry["engagement"]
        for sample in range(20):
            wave = 0.5 + 0.5 * np.sin(sample / 3.0)
            divergence_rows.append(
                {"anatomy": entry["anatomy"], "sample": sample,
                 "frobenius_relative": (0.02 * dose * wave) if structural else 0.0,
                 "principal_angle_3_deg": (4.0 * dose * wave) if structural else 0.0}
            )
    header = list(divergence_rows[0].keys())
    common.write_csv(root / "offline" / "jacobian_divergence.csv", header,
                     [[r[k] for k in header] for r in divergence_rows])

    rows = _index_rows(structural)
    RS._write_index(root / "study_index.csv", rows, common)
    RS._write_index(root / "observer" / "study_index.csv", _observer_rows(structural), common)

    (root / "study_manifest.json").write_text(
        json.dumps(
            {"anatomies": [{"anatomy": e["anatomy"], "radius_scale": e["radius_scale"],
                            "confounded": 0} for e in ANATOMIES],
             "controllers": CONTROLLERS,
             "jacobian_models": list(RS.JACOBIAN_MODELS),
             "initial_conditions": 1,
             "plant": "contact enabled in every cell"},
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def analyse(root: Path) -> dict:
    argv = sys.argv
    sys.argv = ["analyse_contact_study.py", str(root)]
    try:
        AN.main()
    finally:
        sys.argv = argv
    return json.loads((root / "contact_study_metrics.json").read_text(encoding="utf-8"))


# ==========================================================================
root = _HERE / "_self_test_pipeline"
structural = analyse(build_study(root / "structural", True))
additive = analyse(build_study(root / "disturbance", False))

# ---- the structural study: all four must land ----------------------------
check("structural: V1 says the sensitivity changes",
      structural["v1_jacobians_differ"]["overall"] == "structural",
      structural["v1_jacobians_differ"]["overall"])
check("structural: V2 confirms d_hat is the missing model term",
      structural["v2_disturbance_or_model"]["outcome"] == "confirmed",
      structural["v2_disturbance_or_model"]["outcome"])
check("structural: V3 finds a rising dose response",
      structural["v3_dose_response"]["outcome"] == "rises",
      f"rising for {structural['v3_dose_response']['rising_controllers']}")
check("structural: V4 says no alpha rescues it",
      structural["v4_observer_ceiling"]["outcome"] == "no alpha rescues it",
      f"best alpha {structural['v4_observer_ceiling']['best_alpha']}, "
      f"shortfall {100 * structural['v4_observer_ceiling']['shortfall_relative']:.0f}%")
check("structural: the dose axis is the measured one, not the knob",
      "measured" in structural["v3_dose_response"]["dose_axis"],
      structural["v3_dose_response"]["dose_axis"])
check("structural: the report has no FAIL finding",
      not any(f["level"] == "fail" for f in structural["findings"]),
      str([f["title"] for f in structural["findings"] if f["level"] == "fail"]))

# ---- the disturbance study: the pipeline must decline to claim ------------
check("disturbance: V1 reports no Jacobian difference",
      additive["v1_jacobians_differ"]["overall"] == "none",
      additive["v1_jacobians_differ"]["overall"])
check("disturbance: V2 does NOT claim the residual is model error",
      additive["v2_disturbance_or_model"]["outcome"] == "not shown",
      additive["v2_disturbance_or_model"]["outcome"])
check("disturbance: V3 finds no dose response",
      additive["v3_dose_response"]["outcome"] == "flat",
      additive["v3_dose_response"]["outcome"])
check("disturbance: V4 says the observer is sufficient",
      additive["v4_observer_ceiling"]["outcome"] == "observer closes the gap",
      f"shortfall {100 * additive['v4_observer_ceiling']['shortfall_relative']:.0f}%")
check("disturbance: the report carries FAIL findings",
      any(f["level"] == "fail" for f in additive["findings"]),
      str([f["title"] for f in additive["findings"] if f["level"] == "fail"]))

# ---- the ladder table ----------------------------------------------------
ladder = structural["ladder"]
check("the ladder covers every anatomy x Jacobian pair",
      len(ladder) == len(ANATOMIES) * 2, f"{len(ladder)} rows")
tight = [r for r in ladder
         if r["anatomy"] == "radius_x0p4" and r["jacobian_model"] == "contact"][0]
check("the ladder is monotone across the rungs on the synthetic data",
      tight["naive_inverse_jacobian"] > tight["mpc_lti"] > tight["mpc_ltv_offline"]
      > tight["mpc_ltv_sqp_online"],
      f"{tight['naive_inverse_jacobian']:.3f} > {tight['mpc_ltv_sqp_online']:.3f}")
check("the ladder carries solve times beside accuracy",
      np.isfinite(tight["mpc_ltv_sqp_online_p95_ms"]))

# ---- reports and figures exist -------------------------------------------
for name in ("structural", "disturbance"):
    report = root / name / "contact_study_report.md"
    check(f"{name}: a markdown report was written",
          report.exists() and "## Verdict" in report.read_text(encoding="utf-8"))
figure = root / "structural" / "contact_study_figures.png"
check("a figure was written (or matplotlib is absent)",
      figure.exists() or AN.C.new_figure(1, 1, (1, 1))[0] is None,
      "present" if figure.exists() else "matplotlib unavailable")

# ---- the analyser refuses an empty directory -----------------------------
empty = root / "empty"
empty.mkdir(parents=True, exist_ok=True)
completed = subprocess.run(
    # AN.__file__, not a guessed path: the analyser may sit in a sibling
    # directory depending on how the bundle was unpacked.
    [sys.executable, str(Path(AN.__file__).resolve()), str(empty)],
    capture_output=True, text=True,
)
check("an empty study directory is refused, not silently analysed",
      completed.returncode != 0 and "neither an offline stage" in completed.stderr,
      completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "")

print("\ncontact study pipeline test")
print("=" * 74)
for ok, name, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
failures = sum(1 for ok, _, _ in results if not ok)
print("=" * 74)
print(f"  {len(results) - failures}/{len(results)} checks passed")
sys.exit(1 if failures else 0)
