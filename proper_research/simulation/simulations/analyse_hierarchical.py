
"""
analyse_contact_sqp4_features.py

Feature-evidence analysis for contact-aware SQP-full SQP4 fixed-rollout experiments.

Typical use:
    python analyse_contact_sqp4_features.py \
        --run-root evidence_contact_sqp4_fixed_rollouts \
        --out-dir evidence_contact_sqp4_fixed_rollouts_analysis

What it does:
    1. Recursively finds event-level CSVs under --run-root, or reads --input-csv.
    2. Filters to contact + sqp_full + SQP4 where those columns exist.
    3. Screens candidate features:
         - Spearman with next-spike label
         - AUC for next-spike label
         - Spearman with log adaptive error
         - Spearman with log one-step error
    4. Fits GLM and GEE spike models for each feature:
         next_spike ~ z_feature + z_rollout [+ bend controls]
       and saves odds ratios, standard errors, p-values, convergence flags.
    5. Fits GLM and GEE continuous-error models:
         log(adaptive_error) ~ z_feature + z_rollout [+ bend controls]
         log(one_step_error) ~ z_feature + z_rollout [+ bend controls]
    6. Builds feature-tertile × rollout risk tables.
    7. Saves plots comparing GLM vs GEE p-values / odds ratios and rollout-risk evidence.

Notes:
    - GEE grouping defaults to run_name if available.
    - If run_name is absent, a cluster ID is inferred from source file and metadata.
    - AUC is computed using rank statistics, so sklearn is not required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy. Install with: pip install scipy") from exc

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires statsmodels. Install with: pip install statsmodels") from exc


DEFAULT_FEATURES = [
    # Main beam-Hessian candidates
    "risk_low_lambda_min_H_beam",
    "lambda_min_H_beam",
    "log10_cond_H_beam",
    "cond_H_beam",
    # Composite / clearance geometry
    "geometry_conditioning_risk_score",
    "risk_low_clearance_mm",
    "risk_inv_clearance",
    "min_clearance_mm",
    # Curvature / angle features
    "curvature_max_ahead_1pm",
    "curvature_mean_ahead_1pm",
    "curvature_local_1pm",
    "adapt_tip_vessel_angle_deg",
    # SQP-probe and final tolerance features
    "sqp_du_abs_first",
    "sqp_du_rel_first",
    "sqp_du_abs_final",
    "sqp_du_rel_final",
    # Other conditioning names sometimes used
    "norm_H_mpc",
    "mpc_eig_cond",
    "log10_mpc_eig_cond",
]

EVENT_HINT_COLUMNS = {
    "adapt_pred_err_xy_mm",
    "pred1_err_xy_mm",
    "is_next_error_spike",
    "is_next_error_spike_int",
    "next_spike",
    "rollout_steps",
    "risk_low_lambda_min_H_beam",
    "lambda_min_H_beam",
    "sqp_du_abs_first",
    "solver_mode",
    "jacobian_variant",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_colname(c: str) -> str:
    return str(c).strip()


def read_csv_safely(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, nrows=nrows)
        df.columns = [normalise_colname(c) for c in df.columns]
        return df
    except Exception:
        return None


def looks_like_event_csv(path: Path) -> bool:
    sample = read_csv_safely(path, nrows=5)
    if sample is None or sample.empty:
        return False

    cols = set(sample.columns)
    if len(cols & EVENT_HINT_COLUMNS) >= 2:
        return True

    # Common generic names in logs.
    if "step" in cols and ("x_tip" in cols or "adapt_pred_err_xy_mm" in cols or "rollout_steps" in cols):
        return True

    return False


def discover_event_csvs(run_root: Path) -> list[Path]:
    if run_root.is_file() and run_root.suffix.lower() == ".csv":
        return [run_root]

    all_csvs = sorted(run_root.rglob("*.csv"))

    # Prefer already-assembled event tables if present.
    preferred_patterns = [
        "composite_score_events",
        "geometry_event_table",
        "events_with",
        "event_table",
        "per_step_events",
    ]
    preferred = []
    for p in all_csvs:
        name = p.name.lower()
        if any(pattern in name for pattern in preferred_patterns) and looks_like_event_csv(p):
            preferred.append(p)

    if preferred:
        return preferred

    return [p for p in all_csvs if looks_like_event_csv(p)]


def load_events(run_root: Optional[Path], input_csv: Optional[Path]) -> pd.DataFrame:
    if input_csv is not None:
        csvs = [input_csv]
    elif run_root is not None:
        csvs = discover_event_csvs(run_root)
    else:
        raise ValueError("Provide either --run-root or --input-csv")

    if not csvs:
        raise FileNotFoundError(
            f"No event-like CSV files found. Checked: {run_root or input_csv}"
        )

    frames = []
    for path in csvs:
        df = read_csv_safely(path)
        if df is None or df.empty:
            continue

        df["source_csv"] = str(path)
        if "run_name" not in df.columns:
            # Use parent folder as a conservative run identifier.
            df["run_name"] = path.parent.name

        frames.append(df)

    if not frames:
        raise RuntimeError("CSV files were found, but none could be read successfully.")

    # Use sort=False because event tables may differ slightly by run.
    out = pd.concat(frames, ignore_index=True, sort=False)

    # Drop exact duplicate rows that can occur if composite and geometry event tables are both found.
    likely_keys = [c for c in ["run_name", "step", "rollout_steps", "bend_angle_deg", "solver_mode", "jacobian_variant"] if c in out.columns]
    if likely_keys:
        before = len(out)
        out = out.drop_duplicates(subset=likely_keys + [c for c in ["adapt_pred_err_xy_mm", "pred1_err_xy_mm"] if c in out.columns])
        after = len(out)
        if after < before:
            print(f"Dropped {before-after} exact/near duplicate event rows.")

    return out.reset_index(drop=True)


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_engineered_columns(df: pd.DataFrame, spike_threshold: Optional[float]) -> pd.DataFrame:
    # Standard numeric conversions for likely columns.
    numeric_like = [
        c for c in df.columns
        if any(token in c.lower() for token in [
            "err", "risk", "lambda", "cond", "clearance", "curvature",
            "angle", "rollout", "sqp", "budget", "step", "bend", "score", "norm"
        ])
    ]
    df = coerce_numeric(df, numeric_like)

    # Spike label.
    target_candidates = [
        "is_next_error_spike_int",
        "is_next_error_spike",
        "next_spike",
        "next_error_spike",
        "is_spike_next",
    ]
    target_col = find_first_existing(df, target_candidates)

    if target_col is not None:
        raw = df[target_col]
        if raw.dtype == bool:
            df["target_next_spike"] = raw.astype(int)
        else:
            # Handles 0/1, True/False strings, yes/no strings.
            if raw.dtype == object:
                mapped = raw.astype(str).str.lower().map({
                    "true": 1, "false": 0,
                    "yes": 1, "no": 0,
                    "1": 1, "0": 0,
                })
                df["target_next_spike"] = pd.to_numeric(mapped.fillna(raw), errors="coerce")
            else:
                df["target_next_spike"] = pd.to_numeric(raw, errors="coerce")
        df["target_next_spike"] = (df["target_next_spike"] > 0).astype(float)
    else:
        # Fallback: infer a spike from one-step error or adaptive error.
        err_col = find_first_existing(df, ["next_error_delta_mm", "pred1_err_xy_mm", "adapt_pred_err_xy_mm"])
        if err_col is None:
            raise ValueError(
                "Could not find a next-spike column and could not infer one from error columns."
            )
        if spike_threshold is None:
            # Conservative fallback: top decile as spike label.
            spike_threshold = float(df[err_col].quantile(0.90))
            print(f"Warning: no spike label found. Using {err_col} >= 90th percentile = {spike_threshold:.6g}")
        df["target_next_spike"] = (df[err_col] >= spike_threshold).astype(float)

    # Standard outcome aliases.
    adapt_col = find_first_existing(df, ["adapt_pred_err_xy_mm", "adaptive_error_mm", "adapt_error_xy_mm"])
    pred1_col = find_first_existing(df, ["pred1_err_xy_mm", "one_step_error_mm", "pred1_error_xy_mm"])

    if adapt_col:
        df["adaptive_error"] = pd.to_numeric(df[adapt_col], errors="coerce")
        df["log_adaptive_error"] = np.log(df["adaptive_error"].clip(lower=1e-9))
    if pred1_col:
        df["one_step_error"] = pd.to_numeric(df[pred1_col], errors="coerce")
        df["log_one_step_error"] = np.log(df["one_step_error"].clip(lower=1e-9))

    # Rollout alias.
    rollout_col = find_first_existing(df, [
        "runtime_rollout_steps",
        "selected_rollout_steps",
        "selected_rollout",
        "effective_rollout_steps",
        "rollout_steps_used",
        "rollout_steps",
    ])
    if rollout_col:
        df["rollout_for_model"] = pd.to_numeric(df[rollout_col], errors="coerce")

    # SQP budget alias.
    budget_col = find_first_existing(df, ["sqp_budget", "N_sqp", "n_sqp", "sqp_iters", "sqp_iterations"])
    if budget_col:
        df["sqp_budget_for_filter"] = pd.to_numeric(df[budget_col], errors="coerce")

    # Bend alias.
    bend_col = find_first_existing(df, ["bend_angle_deg", "bend_deg", "angle_deg"])
    if bend_col:
        df["bend_for_model"] = pd.to_numeric(df[bend_col], errors="coerce")

    # Derived conditioning features if possible.
    if "cond_H_beam" in df.columns and "log10_cond_H_beam" not in df.columns:
        df["log10_cond_H_beam"] = np.log10(pd.to_numeric(df["cond_H_beam"], errors="coerce").clip(lower=1e-300))
    if "lambda_min_H_beam" in df.columns and "risk_low_lambda_min_H_beam" not in df.columns:
        df["risk_low_lambda_min_H_beam"] = -np.log10(pd.to_numeric(df["lambda_min_H_beam"], errors="coerce").clip(lower=1e-12))

    return df


def filter_contact_sqp4(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()

    if args.filter_contact and "jacobian_variant" in out.columns:
        out = out[out["jacobian_variant"].astype(str).str.lower().eq("contact")]

    if args.filter_sqp_full and "solver_mode" in out.columns:
        out = out[out["solver_mode"].astype(str).str.lower().isin(["sqp_full", "sqp", "full_sqp"])]

    if args.filter_sqp_budget is not None and "sqp_budget_for_filter" in out.columns:
        out = out[np.isclose(out["sqp_budget_for_filter"], args.filter_sqp_budget)]

    return out.reset_index(drop=True)


def standardise_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / sd


def auc_rank(y: pd.Series, x: pd.Series) -> float:
    """AUC for y in {0,1}, using rank-sum / Mann-Whitney statistic."""
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    if tmp.empty:
        return np.nan
    yv = tmp["y"].astype(int)
    if yv.nunique() != 2:
        return np.nan

    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    ranks = tmp["x"].rank(method="average")
    rank_sum_pos = ranks[yv == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def spearman_pair(y: pd.Series, x: pd.Series) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan, len(tmp)
    rho, p = stats.spearmanr(tmp["x"], tmp["y"])
    return float(rho), float(p), len(tmp)


def available_features(df: pd.DataFrame, requested: Optional[list[str]]) -> list[str]:
    candidates = requested if requested else DEFAULT_FEATURES
    found = []
    for f in candidates:
        if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().sum() >= 10:
            found.append(f)
    # Unique while preserving order.
    return list(dict.fromkeys(found))


def screen_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce")
        rho_spike, p_spike, n_spike = spearman_pair(df["target_next_spike"], x)
        auc = auc_rank(df["target_next_spike"], x)

        row = {
            "feature": f,
            "n": int(x.notna().sum()),
            "n_spike_model": n_spike,
            "spearman_next_spike": rho_spike,
            "spearman_next_spike_p": p_spike,
            "auc_next_spike": auc,
            "auc_directional": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
            "higher_feature_higher_spike_risk": bool(auc >= 0.5) if np.isfinite(auc) else np.nan,
        }

        if "log_adaptive_error" in df.columns:
            rho, p, n = spearman_pair(df["log_adaptive_error"], x)
            row.update({
                "spearman_log_adaptive_error": rho,
                "spearman_log_adaptive_error_p": p,
                "n_log_adaptive_error": n,
            })

        if "log_one_step_error" in df.columns:
            rho, p, n = spearman_pair(df["log_one_step_error"], x)
            row.update({
                "spearman_log_one_step_error": rho,
                "spearman_log_one_step_error_p": p,
                "n_log_one_step_error": n,
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["auc_directional", "spearman_next_spike_p"], ascending=[False, True])
    return out


def build_formula(outcome: str, feature_z: str, include_rollout: bool, include_bend: bool) -> str:
    terms = [feature_z]
    if include_rollout:
        terms.append("z_rollout_for_model")
    if include_bend:
        terms.append("C(bend_factor)")
    return outcome + " ~ " + " + ".join(terms)


@dataclass
class ModelResult:
    feature: str
    model_type: str
    outcome: str
    formula: str
    n: int
    n_clusters: int
    coef: float
    se: float
    z_or_t: float
    p_value: float
    odds_ratio: float
    ci_low_or: float
    ci_high_or: float
    converged: bool
    error: str


def get_cluster_col(df: pd.DataFrame) -> str:
    if "run_name" in df.columns:
        return "run_name"
    if "source_csv" in df.columns:
        return "source_csv"
    df["_single_cluster"] = "cluster_0"
    return "_single_cluster"


def fit_glm_or_gee(
    data: pd.DataFrame,
    feature: str,
    outcome: str,
    family: str,
    use_gee: bool,
    include_rollout: bool,
    include_bend: bool,
) -> ModelResult:
    d = data.copy()
    zcol = f"z__{re.sub('[^0-9a-zA-Z_]+', '_', feature)}"
    d[zcol] = standardise_series(d[feature])

    needed = [outcome, zcol]
    if include_rollout:
        needed.append("z_rollout_for_model")
    if include_bend:
        needed.append("bend_factor")
    if use_gee:
        cluster_col = get_cluster_col(d)
        needed.append(cluster_col)
    else:
        cluster_col = get_cluster_col(d)

    d = d.dropna(subset=needed).copy()
    n = len(d)
    n_clusters = d[cluster_col].nunique() if cluster_col in d.columns else 1

    formula = build_formula(outcome, zcol, include_rollout, include_bend)

    if n < 20 or d[outcome].nunique() < 2 and family == "binomial":
        return ModelResult(
            feature, "GEE" if use_gee else "GLM", outcome, formula, n, n_clusters,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            False, "not enough observations or outcome variation"
        )

    try:
        if family == "binomial":
            fam = sm.families.Binomial()
        elif family == "gaussian":
            fam = sm.families.Gaussian()
        else:
            raise ValueError(f"Unknown family: {family}")

        if use_gee:
            # Exchangeable is the intended working correlation for runs.
            try:
                model = smf.gee(
                    formula=formula,
                    groups=d[cluster_col],
                    data=d,
                    family=fam,
                    cov_struct=sm.cov_struct.Exchangeable(),
                )
                res = model.fit(maxiter=100)
            except Exception:
                # Fallback to independent working correlation if exchangeable fails.
                model = smf.gee(
                    formula=formula,
                    groups=d[cluster_col],
                    data=d,
                    family=fam,
                    cov_struct=sm.cov_struct.Independence(),
                )
                res = model.fit(maxiter=100)
        else:
            model = smf.glm(formula=formula, data=d, family=fam)
            res = model.fit(maxiter=100)

        coef = float(res.params.get(zcol, np.nan))
        se = float(res.bse.get(zcol, np.nan))
        stat_val = float((res.tvalues if hasattr(res, "tvalues") else res.params / res.bse).get(zcol, np.nan))
        pval = float(res.pvalues.get(zcol, np.nan))
        if family == "binomial":
            odds_ratio = math.exp(coef) if np.isfinite(coef) else np.nan
            ci_low_or = math.exp(coef - 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan
            ci_high_or = math.exp(coef + 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan
        else:
            odds_ratio = np.nan
            ci_low_or = np.nan
            ci_high_or = np.nan

        conv = bool(getattr(res, "converged", True))
        return ModelResult(
            feature, "GEE" if use_gee else "GLM", outcome, formula, n, n_clusters,
            coef, se, stat_val, pval, odds_ratio, ci_low_or, ci_high_or,
            conv, ""
        )
    except Exception as exc:
        return ModelResult(
            feature, "GEE" if use_gee else "GLM", outcome, formula, n, n_clusters,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            False, repr(exc)
        )


def fit_feature_models(df: pd.DataFrame, features: list[str], include_bend: bool) -> pd.DataFrame:
    data = df.copy()
    data["z_rollout_for_model"] = standardise_series(data["rollout_for_model"]) if "rollout_for_model" in data.columns else np.nan
    include_rollout = "rollout_for_model" in data.columns and data["rollout_for_model"].notna().sum() > 10
    if "bend_for_model" in data.columns:
        data["bend_factor"] = data["bend_for_model"].astype("category")
    else:
        include_bend = False

    results = []

    # Binary spike models.
    for f in features:
        for use_gee in [False, True]:
            results.append(fit_glm_or_gee(
                data, f, "target_next_spike", "binomial", use_gee,
                include_rollout=include_rollout,
                include_bend=include_bend,
            ))

    # Continuous error models.
    for outcome in ["log_adaptive_error", "log_one_step_error"]:
        if outcome not in data.columns:
            continue
        for f in features:
            for use_gee in [False, True]:
                results.append(fit_glm_or_gee(
                    data, f, outcome, "gaussian", use_gee,
                    include_rollout=include_rollout,
                    include_bend=include_bend,
                ))

    return pd.DataFrame([r.__dict__ for r in results])


def make_tertile_bins(s: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    x = pd.to_numeric(s, errors="coerce")
    qs = x.quantile([1/3, 2/3]).to_dict()
    q1, q2 = float(qs[1/3]), float(qs[2/3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        b = pd.Series(pd.NA, index=s.index, dtype="object")
        return b, {"low_medium": np.nan, "medium_high": np.nan}
    labels = pd.cut(x, bins=[-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"], include_lowest=True)
    return labels.astype("object"), {"low_medium": q1, "medium_high": q2}


def rollout_risk_tables(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "rollout_for_model" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    threshold_rows = []
    for f in features:
        bins, thresholds = make_tertile_bins(df[f])
        threshold_rows.append({"feature": f, **thresholds})
        d = df.assign(feature_bin=bins).dropna(subset=["feature_bin", "rollout_for_model"])
        if d.empty:
            continue
        group_cols = ["feature", "feature_bin", "rollout_for_model"]
        d["feature"] = f
        agg = d.groupby(group_cols, dropna=False).agg(
            events=("target_next_spike", "size"),
            spikes=("target_next_spike", "sum"),
            spike_rate=("target_next_spike", "mean"),
            mean_adaptive_error=("adaptive_error", "mean") if "adaptive_error" in d.columns else ("target_next_spike", "mean"),
            median_adaptive_error=("adaptive_error", "median") if "adaptive_error" in d.columns else ("target_next_spike", "median"),
            mean_one_step_error=("one_step_error", "mean") if "one_step_error" in d.columns else ("target_next_spike", "mean"),
        ).reset_index()
        summary_rows.append(agg)

    return (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(),
        pd.DataFrame(threshold_rows),
    )


def plot_feature_screen(screen: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    if screen.empty:
        return
    d = screen.head(top_n).sort_values("auc_directional")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(d))))
    ax.barh(d["feature"], d["auc_directional"])
    ax.axvline(0.5, linestyle="--")
    ax.set_xlabel("Directional AUC for next spike")
    ax.set_ylabel("Feature")
    ax.set_title("Feature ranking by spike AUC")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_screen_directional_auc.png", dpi=220)
    plt.close(fig)


def plot_glm_gee_pvalues(models: pd.DataFrame, out_dir: Path, outcome: str, top_features: list[str]) -> None:
    d = models[(models["outcome"] == outcome) & (models["feature"].isin(top_features))].copy()
    d = d[np.isfinite(d["p_value"])]
    if d.empty:
        return

    # Pivot: rows feature, columns model_type.
    piv = d.pivot_table(index="feature", columns="model_type", values="p_value", aggfunc="first")
    piv = piv.reindex(top_features)
    plot_data = -np.log10(piv.clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(11, max(5, 0.4 * len(plot_data))))
    y = np.arange(len(plot_data.index))
    width = 0.35

    if "GLM" in plot_data.columns:
        ax.barh(y - width/2, plot_data["GLM"], height=width, label="GLM")
    if "GEE" in plot_data.columns:
        ax.barh(y + width/2, plot_data["GEE"], height=width, label="GEE")

    ax.axvline(-np.log10(0.05), linestyle="--", label="p=0.05")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data.index)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(f"GLM vs GEE feature p-values: {outcome}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"glm_vs_gee_pvalues_{outcome}.png", dpi=220)
    plt.close(fig)


def plot_odds_ratios(models: pd.DataFrame, out_dir: Path, top_features: list[str]) -> None:
    d = models[
        (models["outcome"] == "target_next_spike")
        & (models["feature"].isin(top_features))
        & np.isfinite(models["odds_ratio"])
    ].copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(top_features))))
    y_base = np.arange(len(top_features))
    offset = {"GLM": -0.12, "GEE": 0.12}

    for model_type, sub in d.groupby("model_type"):
        sub = sub.set_index("feature").reindex(top_features)
        y = y_base + offset.get(model_type, 0.0)
        x = sub["odds_ratio"].astype(float)
        xerr = np.vstack([
            (x - sub["ci_low_or"]).clip(lower=0),
            (sub["ci_high_or"] - x).clip(lower=0)
        ])
        ax.errorbar(x, y, xerr=xerr, fmt="o", label=model_type, capsize=3)

    ax.axvline(1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y_base)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Odds ratio for one SD feature increase, log scale")
    ax.set_title("GLM vs GEE spike-model odds ratios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "glm_vs_gee_spike_odds_ratios.png", dpi=220)
    plt.close(fig)


def plot_rollout_risk_tables(tables: pd.DataFrame, out_dir: Path, features: list[str]) -> None:
    if tables.empty:
        return

    for f in features:
        d = tables[tables["feature"] == f].copy()
        if d.empty:
            continue
        # Spike rate plot
        piv = d.pivot_table(index="rollout_for_model", columns="feature_bin", values="spike_rate", aggfunc="first")
        if not piv.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            piv.sort_index().plot(kind="bar", ax=ax)
            ax.set_ylabel("Spike rate")
            ax.set_xlabel("Rollout steps")
            ax.set_title(f"Spike rate by rollout and tertile bin: {f}")
            ax.legend(title="Feature bin")
            fig.tight_layout()
            safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", f)
            fig.savefig(out_dir / f"rollout_spike_rate_by_{safe_name}.png", dpi=220)
            plt.close(fig)

        # Mean adaptive error plot
        if "mean_adaptive_error" in d.columns:
            piv = d.pivot_table(index="rollout_for_model", columns="feature_bin", values="mean_adaptive_error", aggfunc="first")
            if not piv.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                piv.sort_index().plot(kind="bar", ax=ax)
                ax.set_ylabel("Mean adaptive error [mm]")
                ax.set_xlabel("Rollout steps")
                ax.set_title(f"Mean adaptive error by rollout and tertile bin: {f}")
                ax.legend(title="Feature bin")
                fig.tight_layout()
                safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", f)
                fig.savefig(out_dir / f"rollout_mean_adaptive_error_by_{safe_name}.png", dpi=220)
                plt.close(fig)


def write_markdown_report(
    out_dir: Path,
    df: pd.DataFrame,
    screen: pd.DataFrame,
    models: pd.DataFrame,
    rollout_tables: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_n: int,
) -> None:
    lines = []
    lines.append("# Contact SQP4 fixed-rollout feature/GEE analysis\n")
    lines.append("## Dataset\n")
    lines.append(f"- Events after filtering: **{len(df)}**")
    lines.append(f"- Runs/clusters: **{df['run_name'].nunique() if 'run_name' in df.columns else 'unknown'}**")
    lines.append(f"- Next-spike events: **{int(df['target_next_spike'].sum())}**")
    lines.append(f"- Next-spike rate: **{df['target_next_spike'].mean():.4%}**")
    if "adaptive_error" in df.columns:
        lines.append(f"- Mean adaptive error: **{df['adaptive_error'].mean():.6g} mm**")
    if "one_step_error" in df.columns:
        lines.append(f"- Mean one-step error: **{df['one_step_error'].mean():.6g} mm**")

    lines.append("\n## Top feature screen by directional AUC\n")
    if not screen.empty:
        cols = ["feature", "spearman_next_spike", "spearman_next_spike_p", "auc_next_spike", "auc_directional"]
        lines.append(screen[cols].head(top_n).to_markdown(index=False))
    else:
        lines.append("No feature-screen rows available.")

    lines.append("\n## GLM vs GEE spike-model p-values\n")
    if not models.empty:
        spike = models[models["outcome"] == "target_next_spike"].copy()
        if not spike.empty:
            piv = spike.pivot_table(index="feature", columns="model_type", values="p_value", aggfunc="first")
            order = screen["feature"].head(top_n).tolist() if not screen.empty else sorted(piv.index.tolist())
            piv = piv.reindex(order)
            lines.append(piv.to_markdown())
    else:
        lines.append("No model rows available.")

    lines.append("\n## GLM vs GEE spike-model odds ratios\n")
    if not models.empty:
        spike = models[models["outcome"] == "target_next_spike"].copy()
        if not spike.empty:
            cols = ["feature", "model_type", "odds_ratio", "ci_low_or", "ci_high_or", "p_value", "converged", "error"]
            order = screen["feature"].head(top_n).tolist() if not screen.empty else sorted(spike["feature"].unique())
            table = spike[spike["feature"].isin(order)][cols]
            lines.append(table.to_markdown(index=False))

    lines.append("\n## Tertile thresholds\n")
    if not thresholds.empty:
        lines.append(thresholds.head(top_n).to_markdown(index=False))

    lines.append("\n## How to interpret\n")
    lines.append(
        "- GLM treats event rows as independent.\n"
        "- GEE groups rows by `run_name` and reports cluster-robust inference under a working correlation.\n"
        "- A feature can be weak for binary spikes but strong for continuous error when spike count is low.\n"
        "- Pre-SQP geometry features should be interpreted as rollout selectors/vetoes; SQP first-change features are probe signals; final SQP residuals are acceptance/tolerance signals.\n"
    )

    (out_dir / "feature_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=None, help="Root directory of fixed-rollout experiment.")
    p.add_argument("--input-csv", type=Path, default=None, help="Optional already-assembled event CSV.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output analysis directory.")
    p.add_argument("--spike-threshold", type=float, default=None, help="Used only if no spike label column exists.")
    p.add_argument("--features", nargs="*", default=None, help="Optional explicit feature list.")
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--include-bend-control", action="store_true", help="Add C(bend_angle) to GLM/GEE models.")
    p.add_argument("--filter-contact", action="store_true", default=True)
    p.add_argument("--no-filter-contact", dest="filter_contact", action="store_false")
    p.add_argument("--filter-sqp-full", action="store_true", default=True)
    p.add_argument("--no-filter-sqp-full", dest="filter_sqp_full", action="store_false")
    p.add_argument("--filter-sqp-budget", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)
    safe_mkdir(args.out_dir / "plots")

    df_raw = load_events(args.run_root, args.input_csv)
    df = ensure_engineered_columns(df_raw, args.spike_threshold)
    df = filter_contact_sqp4(df, args)

    if df.empty:
        raise RuntimeError("No rows left after contact/SQP/SQP-budget filtering. Try disabling filters.")

    features = available_features(df, args.features)
    if not features:
        raise RuntimeError("No requested/default feature columns were found after filtering.")

    # Save filtered data for audit/reuse.
    df.to_csv(args.out_dir / "filtered_contact_sqp4_events.csv", index=False)

    # Basic summary.
    basic = {
        "events": int(len(df)),
        "runs": int(df["run_name"].nunique()) if "run_name" in df.columns else None,
        "spikes": int(df["target_next_spike"].sum()),
        "spike_rate": float(df["target_next_spike"].mean()),
        "mean_adaptive_error": float(df["adaptive_error"].mean()) if "adaptive_error" in df.columns else None,
        "mean_one_step_error": float(df["one_step_error"].mean()) if "one_step_error" in df.columns else None,
        "features_used": features,
    }
    (args.out_dir / "basic_summary.json").write_text(json.dumps(basic, indent=2), encoding="utf-8")
    pd.DataFrame([basic]).to_csv(args.out_dir / "basic_summary.csv", index=False)

    screen = screen_features(df, features)
    screen.to_csv(args.out_dir / "feature_screen.csv", index=False)

    models = fit_feature_models(df, features, include_bend=args.include_bend_control)
    models.to_csv(args.out_dir / "glm_gee_feature_models.csv", index=False)

    rollout_tables, thresholds = rollout_risk_tables(df, features)
    rollout_tables.to_csv(args.out_dir / "feature_tertile_by_rollout_risk.csv", index=False)
    thresholds.to_csv(args.out_dir / "feature_tertile_thresholds.csv", index=False)

    top_features = screen["feature"].head(args.top_n).tolist() if not screen.empty else features[:args.top_n]
    plot_feature_screen(screen, args.out_dir / "plots", args.top_n)
    plot_glm_gee_pvalues(models, args.out_dir / "plots", "target_next_spike", top_features)
    if "log_adaptive_error" in models["outcome"].unique():
        plot_glm_gee_pvalues(models, args.out_dir / "plots", "log_adaptive_error", top_features)
    if "log_one_step_error" in models["outcome"].unique():
        plot_glm_gee_pvalues(models, args.out_dir / "plots", "log_one_step_error", top_features)
    plot_odds_ratios(models, args.out_dir / "plots", top_features)
    plot_rollout_risk_tables(rollout_tables, args.out_dir / "plots", top_features[:6])

    write_markdown_report(args.out_dir, df, screen, models, rollout_tables, thresholds, args.top_n)

    print(f"Done. Wrote feature analysis to: {args.out_dir}")
    print(f"Key files:")
    print(f"  {args.out_dir / 'feature_analysis_report.md'}")
    print(f"  {args.out_dir / 'feature_screen.csv'}")
    print(f"  {args.out_dir / 'glm_gee_feature_models.csv'}")
    print(f"  {args.out_dir / 'plots'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()