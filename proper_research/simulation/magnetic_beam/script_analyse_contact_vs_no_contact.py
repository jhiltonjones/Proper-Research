#!/usr/bin/env python3
"""
Analyze MPC and inverse-Jacobian experiment grids.

The script recursively discovers experiment directories beneath --root and
analyzes:

1. Linear prediction versus nonlinear rollout error.
2. Predicted/reference/applied tracking error.
3. Planned commands and normalized command utilization.
4. Per-channel Jacobian authority.
5. Per-channel selected tip-motion contribution B[:, j] * u_j.
6. Tangential versus lateral contribution relative to the reference path.
7. Jacobian change over the prediction horizon.
8. Contact-aware versus no-contact Jacobian runs when matched pairs exist.
9. Cross-controller summary tables and plots.

Expected experiment layout
--------------------------
root/
    experiment_name/
        experiment_config.json
        log.csv
        lumen_C.npy
        rollout_frames/
            rollout_0000.npz
            rollout_0001.npz
            ...

The loader is intentionally tolerant. Missing keys are skipped and recorded
in warnings rather than terminating the complete grid analysis.

Recommended NPZ keys
--------------------
U_seq
U_applied
X_ref_plan or X_ref
X_pred
X_nl_plan or X_nl_candidate
x_rollout
idx_ref
reference_s_plan_m
p_start
B_sequence_linearisation
B_sequence_solution
B_drift_relative_stage
jacobian_diag_input_scale

Example
-------
python analyze_controller_grid.py \
    --root predicition_controller \
    --output predicition_controller/analysis_results

The default input scales correspond to:
    [vx, vy, vz, wx, wy, wz, dL]
    [3, 3, 3, 60 deg/s, 60 deg/s, 2*pi rad/s, 0.1 m/s]

Override them when your physical limits or characteristic commands differ:
python analyze_controller_grid.py \
    --root predicition_controller \
    --input-scales 3 3 3 1.0472 1.0472 6.2832 0.1
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
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNEL_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz", "dL")
SOURCE_CHANNEL_NAMES = CHANNEL_NAMES[:6]
EPS = 1.0e-12

CHANNEL_METRIC_COLUMNS = (
    "experiment",
    "bend_deg",
    "jacobian_contact",
    "plant_contact",
    "controller",
    "solver",
    "frame",
    "stage",
    "stage_1based",
    "channel_index",
    "channel",
    "command",
    "input_scale",
    "command_utilization_abs",
    "B_x",
    "B_y",
    "B_z",
    "jacobian_gain_raw",
    "jacobian_authority_scaled_m",
    "contribution_x_m",
    "contribution_y_m",
    "contribution_z_m",
    "contribution_norm_m",
    "tangential_contribution_m",
    "lateral_contribution_m",
    "gain_ratio_from_stage0",
    "direction_change_from_stage0_deg",
    "solution_to_controller_gain_ratio",
    "solution_to_controller_angle_deg",
    "progress",
)


@dataclass(frozen=True)
class Experiment:
    path: Path
    name: str
    bend_deg: float | None
    jacobian_contact: bool | None
    plant_contact: bool | None
    controller: str
    solver: str
    rollout_steps: int | None
    Np: int | None
    kp: float | None
    failed: bool
    failure_text: str
    config: dict[str, Any]
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MPC and inverse-Jacobian experiment grids."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/analysis_results",
    )
    parser.add_argument(
        "--input-scales",
        type=float,
        nargs=7,
        default=[
            3.0,
            3.0,
            3.0,
            math.radians(60.0),
            math.radians(60.0),
            2.0 * math.pi,
            0.1,
        ],
        metavar=("VX", "VY", "VZ", "WX", "WY", "WZ", "DL"),
        help=(
            "Characteristic command magnitudes for normalized command use and "
            "scaled Jacobian authority."
        ),
    )
    parser.add_argument(
        "--heatmap-quantile",
        type=float,
        default=0.98,
        help=(
            "Upper quantile used only for heatmap display limits. "
            "Data are not modified. Default: 0.98"
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include partial data from directories containing failure.txt.",
    )
    parser.add_argument(
        "--max-rollout-files",
        type=int,
        default=None,
        help="Optional maximum number of rollout NPZ files loaded per experiment.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved figure resolution. Default: 180",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print discovered keys and additional diagnostics.",
    )
    return parser.parse_args()


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        warnings.warn(f"Could not read JSON {path}: {exc}")
        return {}


def parse_experiment_name(path: Path) -> Experiment:
    name = path.name

    bend_deg: float | None = None
    bend_match = re.search(r"(?:^|_)bend_([mp])(\d+(?:p\d+)?)", name)
    if bend_match:
        sign = -1.0 if bend_match.group(1) == "m" else 1.0
        value = float(bend_match.group(2).replace("p", "."))
        bend_deg = sign * value

    jacobian_contact: bool | None = None
    if "jac_no_contact" in name:
        jacobian_contact = False
    elif "jac_contact" in name:
        jacobian_contact = True

    plant_contact: bool | None = None
    plant_match = re.search(r"plant_contact_([01])", name)
    if plant_match:
        plant_contact = bool(int(plant_match.group(1)))

    if "ctrl_inverse_jacobian" in name:
        controller = "inverse_jacobian"
    elif "ctrl_mpc" in name:
        controller = "mpc"
    else:
        controller = "unknown"

    if "ctrl_mpc_sqp_full" in name:
        solver = "sqp_full"
    elif "ctrl_mpc_ltv_oneshot" in name:
        solver = "ltv_oneshot"
    elif "ctrl_mpc_lti" in name:
        solver = "lti"
    elif "ctrl_inverse_jacobian_rollout_ltv" in name:
        solver = "rollout_ltv"
    elif "ctrl_inverse_jacobian_rollout_lti" in name:
        solver = "rollout_lti"
    elif "ctrl_inverse_jacobian_held" in name:
        solver = "held"
    else:
        solver = "unknown"

    rollout_steps: int | None = None
    rollout_match = re.search(r"(?:^|_)rollout(\d+)(?:_|$)", name)
    if rollout_match:
        rollout_steps = int(rollout_match.group(1))

    Np: int | None = None
    np_match = re.search(r"(?:^|_)Np(\d+)(?:_|$)", name)
    if np_match:
        Np = int(np_match.group(1))

    kp: float | None = None
    kp_match = re.search(r"(?:^|_)kp(\d+(?:p\d+)?)", name)
    if kp_match:
        kp = float(kp_match.group(1).replace("p", "."))

    failure_path = path / "failure.txt"
    failed = failure_path.exists()
    failure_text = ""
    if failed:
        try:
            failure_text = failure_path.read_text().strip()
        except Exception:
            failure_text = "failure.txt exists but could not be read."

    return Experiment(
        path=path,
        name=name,
        bend_deg=bend_deg,
        jacobian_contact=jacobian_contact,
        plant_contact=plant_contact,
        controller=controller,
        solver=solver,
        rollout_steps=rollout_steps,
        Np=Np,
        kp=kp,
        failed=failed,
        failure_text=failure_text,
        config=safe_json(path / "experiment_config.json"),
        summary=safe_json(path / "run_summary.json"),
    )


def discover_experiments(root: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        has_rollouts = (path / "rollout_frames").is_dir()
        has_log = (path / "log.csv").exists()
        has_config = (path / "experiment_config.json").exists()
        if has_rollouts or has_log or has_config:
            experiments.append(parse_experiment_name(path))
    return experiments


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def npz_to_dict(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]).copy() for key in data.files}
    except Exception as exc:
        warnings.warn(f"Could not load {path}: {exc}")
        return {}


def load_rollouts(
    experiment: Experiment,
    *,
    max_files: int | None,
) -> list[tuple[int, Path, dict[str, np.ndarray]]]:
    rollout_dir = experiment.path / "rollout_frames"
    if not rollout_dir.exists():
        return []

    paths = sorted(rollout_dir.glob("rollout_*.npz"))
    if max_files is not None:
        paths = paths[: max(0, int(max_files))]

    loaded: list[tuple[int, Path, dict[str, np.ndarray]]] = []
    for order_index, path in enumerate(paths):
        match = re.search(r"(\d+)$", path.stem)
        frame = int(match.group(1)) if match else order_index
        data = npz_to_dict(path)
        if data:
            loaded.append((frame, path, data))
    return loaded


def first_array(
    data: dict[str, np.ndarray],
    keys: Sequence[str],
    *,
    ndim: int | None = None,
) -> np.ndarray | None:
    for key in keys:
        if key not in data:
            continue
        try:
            value = np.asarray(data[key], float)
        except Exception:
            continue
        if value.size == 0:
            continue
        if ndim is not None and value.ndim != ndim:
            continue
        return value.copy()
    return None


def infer_horizon(data: dict[str, np.ndarray]) -> int:
    candidate_keys = (
        "U_seq",
        "X_pred",
        "X_nl_plan",
        "X_nl_candidate",
        "X_ref_plan",
        "X_ref",
        "B_sequence_linearisation",
        "B_sequence_solution",
    )
    lengths: list[int] = []
    for key in candidate_keys:
        value = data.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] > 0:
            lengths.append(int(array.shape[0]))
    return max(lengths) if lengths else 0


def reference_tangents(X_ref: np.ndarray | None, Np: int) -> np.ndarray:
    tangents = np.full((Np, 3), np.nan, float)
    if X_ref is None or X_ref.ndim != 2 or X_ref.shape[1] < 3:
        return tangents

    X = X_ref[:, :3]
    K = min(Np, X.shape[0])
    if K == 1:
        return tangents

    for stage in range(K):
        if stage < K - 1:
            delta = X[stage + 1] - X[stage]
        else:
            delta = X[stage] - X[stage - 1]
        norm = float(np.linalg.norm(delta))
        if norm > EPS:
            tangents[stage] = delta / norm
    return tangents


def safe_row_norm_difference(
    A: np.ndarray | None,
    B: np.ndarray | None,
    stage: int,
    dims: int = 3,
) -> float:
    if A is None or B is None:
        return np.nan
    if A.ndim != 2 or B.ndim != 2:
        return np.nan
    if stage >= A.shape[0] or stage >= B.shape[0]:
        return np.nan
    n = min(dims, A.shape[1], B.shape[1])
    if n <= 0:
        return np.nan
    return float(np.linalg.norm(A[stage, :n] - B[stage, :n]))


def quaternion_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, float).reshape(4)
    q2 = np.asarray(q2, float).reshape(4)
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 <= EPS or n2 <= EPS:
        return np.nan
    q1 = q1 / n1
    q2 = q2 / n2
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def extract_progress(
    data: dict[str, np.ndarray],
    lumen_C: np.ndarray | None,
) -> float:
    s_plan = first_array(data, ("reference_s_plan_m",))
    if s_plan is not None and s_plan.size > 0:
        return float(np.ravel(s_plan)[0])

    idx_ref = first_array(data, ("idx_ref",))
    if idx_ref is not None and idx_ref.size > 0:
        idx = int(np.ravel(idx_ref)[0])
        if lumen_C is not None and lumen_C.ndim == 2 and lumen_C.shape[0] > 0:
            C = lumen_C[:, :3]
            ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
            s = np.concatenate(([0.0], np.cumsum(ds)))
            idx = int(np.clip(idx, 0, len(s) - 1))
            return float(s[idx])
        return float(idx)

    return np.nan


def build_metric_tables(
    experiment: Experiment,
    rollouts: list[tuple[int, Path, dict[str, np.ndarray]]],
    *,
    default_input_scales: np.ndarray,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_records: list[dict[str, Any]] = []
    channel_records: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []

    lumen_C: np.ndarray | None = None
    lumen_path = experiment.path / "lumen_C.npy"
    if lumen_path.exists():
        try:
            lumen_C = np.asarray(np.load(lumen_path), float)
        except Exception as exc:
            warnings.warn(f"Could not load {lumen_path}: {exc}")

    for frame, path, data in rollouts:
        if verbose:
            print(f"[KEYS] {experiment.name} frame={frame}: {sorted(data)}")

        U_seq = first_array(data, ("U_seq",), ndim=2)
        U_applied = first_array(data, ("U_applied", "u_applied"), ndim=2)
        X_ref = first_array(data, ("X_ref_plan", "X_ref"), ndim=2)
        X_lin = first_array(data, ("X_pred", "X_lin_candidate"), ndim=2)
        X_nl = first_array(data, ("X_nl_plan", "X_nl_candidate"), ndim=2)
        X_applied = first_array(data, ("x_rollout",), ndim=2)

        B_ctrl = first_array(
            data,
            ("B_sequence_linearisation", "B_sequence"),
            ndim=3,
        )
        B_sol = first_array(
            data,
            ("B_sequence_solution",),
            ndim=3,
        )
        if B_sol is None:
            B_sol = B_ctrl

        p_start = first_array(data, ("p_start",))
        progress = extract_progress(data, lumen_C)
        Np = infer_horizon(data)

        scale_from_file = first_array(data, ("jacobian_diag_input_scale",))
        if scale_from_file is not None and scale_from_file.size == 7:
            input_scales = scale_from_file.reshape(7)
        else:
            input_scales = default_input_scales.copy()

        tangents = reference_tangents(X_ref, Np)

        frame_record: dict[str, Any] = {
            "experiment": experiment.name,
            "bend_deg": experiment.bend_deg,
            "jacobian_contact": experiment.jacobian_contact,
            "plant_contact": experiment.plant_contact,
            "controller": experiment.controller,
            "solver": experiment.solver,
            "frame": frame,
            "npz_path": str(path),
            "Np_inferred": Np,
            "progress": progress,
            "failed_run": experiment.failed,
        }

        if p_start is not None and p_start.size >= 8:
            p_flat = p_start.reshape(-1)
            frame_record.update(
                {
                    "source_x": float(p_flat[0]),
                    "source_y": float(p_flat[1]),
                    "source_z": float(p_flat[2]),
                    "source_qw": float(p_flat[3]),
                    "source_qx": float(p_flat[4]),
                    "source_qy": float(p_flat[5]),
                    "source_qz": float(p_flat[6]),
                    "source_L": float(p_flat[7]),
                }
            )

        first_lin_nl = safe_row_norm_difference(X_nl, X_lin, 0)
        first_applied_lin = safe_row_norm_difference(X_applied, X_lin, 0)
        first_applied_nl = safe_row_norm_difference(X_applied, X_nl, 0)
        frame_record.update(
            {
                "first_lin_nl_error_m": first_lin_nl,
                "first_applied_lin_error_m": first_applied_lin,
                "first_applied_nl_error_m": first_applied_nl,
            }
        )
        frame_records.append(frame_record)

        for stage in range(Np):
            stage_record: dict[str, Any] = {
                "experiment": experiment.name,
                "bend_deg": experiment.bend_deg,
                "jacobian_contact": experiment.jacobian_contact,
                "plant_contact": experiment.plant_contact,
                "controller": experiment.controller,
                "solver": experiment.solver,
                "frame": frame,
                "stage": stage,
                "stage_1based": stage + 1,
                "applied_stage": bool(
                    X_applied is not None and stage < X_applied.shape[0]
                ),
                "progress": progress,
            }

            stage_record["lin_nl_error_m"] = safe_row_norm_difference(
                X_nl, X_lin, stage
            )
            stage_record["tracking_linear_error_m"] = safe_row_norm_difference(
                X_lin, X_ref, stage
            )
            stage_record["tracking_nonlinear_error_m"] = safe_row_norm_difference(
                X_nl, X_ref, stage
            )
            stage_record["applied_linear_error_m"] = safe_row_norm_difference(
                X_applied, X_lin, stage
            )
            stage_record["applied_nonlinear_error_m"] = safe_row_norm_difference(
                X_applied, X_nl, stage
            )
            stage_record["applied_tracking_error_m"] = safe_row_norm_difference(
                X_applied, X_ref, stage
            )

            if B_ctrl is not None and stage < B_ctrl.shape[0]:
                Bc = B_ctrl[stage]
                if B_sol is not None and stage < B_sol.shape[0]:
                    Bs = B_sol[stage]
                    denom = max(float(np.linalg.norm(Bc)), EPS)
                    stage_record["B_drift_relative"] = float(
                        np.linalg.norm(Bs - Bc) / denom
                    )
                if stage == 0:
                    stage_record["B_change_from_stage0_relative"] = 0.0
                elif B_ctrl.shape[0] > 0:
                    B0 = B_ctrl[0]
                    stage_record["B_change_from_stage0_relative"] = float(
                        np.linalg.norm(Bc - B0) / max(np.linalg.norm(B0), EPS)
                    )

            stage_records.append(stage_record)

            # Build command-channel rows whenever U_seq exists, even when the
            # rollout NPZ does not contain a Jacobian sequence. This allows the
            # normalized-command plots to work on older datasets. Jacobian,
            # authority, and contribution fields remain NaN until B is saved.
            if U_seq is None or stage >= U_seq.shape[0]:
                continue

            u_stage = np.asarray(U_seq[stage], float).reshape(-1)
            m = min(u_stage.size, len(CHANNEL_NAMES))
            if m <= 0:
                continue

            B_stage = None
            if B_sol is not None and stage < B_sol.shape[0]:
                candidate = np.asarray(B_sol[stage], float)
                if candidate.ndim == 2:
                    B_stage = candidate
                    m = min(m, B_stage.shape[1])

            Bc_stage = None
            if B_ctrl is not None and stage < B_ctrl.shape[0]:
                candidate = np.asarray(B_ctrl[stage], float)
                if candidate.ndim == 2:
                    Bc_stage = candidate

            tangent = (
                tangents[stage]
                if stage < tangents.shape[0]
                else np.full(3, np.nan)
            )
            tangent_valid = bool(np.all(np.isfinite(tangent)))

            B0_sol = None
            if B_sol is not None and B_sol.shape[0] > 0:
                candidate = np.asarray(B_sol[0], float)
                if candidate.ndim == 2:
                    B0_sol = candidate

            for channel in range(m):
                u = float(u_stage[channel])
                scale = float(input_scales[channel])

                # Defaults for datasets that contain commands but not B.
                b = np.full(3, np.nan, dtype=float)
                contribution = np.full(3, np.nan, dtype=float)
                authority_raw = np.nan
                authority_scaled = np.nan
                contribution_norm = np.nan
                tangential = np.nan
                lateral = np.nan
                gain_ratio = np.nan
                direction_change_deg = np.nan
                controller_solution_angle_deg = np.nan
                controller_solution_gain_ratio = np.nan

                if B_stage is not None and channel < B_stage.shape[1]:
                    b_raw = np.asarray(B_stage[:, channel], float).reshape(-1)
                    b[: min(3, b_raw.size)] = b_raw[:3]

                    if np.all(np.isfinite(b)):
                        contribution = b * u
                        authority_raw = float(np.linalg.norm(b))
                        authority_scaled = float(np.linalg.norm(b * scale))
                        contribution_norm = float(np.linalg.norm(contribution))

                        if tangent_valid:
                            tangential = float(np.dot(tangent, contribution))
                            lateral_vector = contribution - tangential * tangent
                            lateral = float(np.linalg.norm(lateral_vector))

                        if (
                            B0_sol is not None
                            and channel < B0_sol.shape[1]
                        ):
                            b0_raw = np.asarray(
                                B0_sol[:, channel],
                                float,
                            ).reshape(-1)
                            b0 = np.full(3, np.nan, dtype=float)
                            b0[: min(3, b0_raw.size)] = b0_raw[:3]

                            if np.all(np.isfinite(b0)):
                                norm0 = float(np.linalg.norm(b0))
                                if norm0 > EPS:
                                    gain_ratio = authority_raw / norm0
                                if authority_raw > EPS and norm0 > EPS:
                                    cosine = float(
                                        np.clip(
                                            np.dot(b, b0)
                                            / (authority_raw * norm0),
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                    direction_change_deg = float(
                                        np.degrees(np.arccos(cosine))
                                    )

                        if (
                            Bc_stage is not None
                            and channel < Bc_stage.shape[1]
                        ):
                            bc_raw = np.asarray(
                                Bc_stage[:, channel],
                                float,
                            ).reshape(-1)
                            bc = np.full(3, np.nan, dtype=float)
                            bc[: min(3, bc_raw.size)] = bc_raw[:3]

                            if np.all(np.isfinite(bc)):
                                norm_bc = float(np.linalg.norm(bc))
                                if norm_bc > EPS:
                                    controller_solution_gain_ratio = (
                                        authority_raw / norm_bc
                                    )
                                if norm_bc > EPS and authority_raw > EPS:
                                    cosine = float(
                                        np.clip(
                                            np.dot(b, bc)
                                            / (authority_raw * norm_bc),
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                    controller_solution_angle_deg = float(
                                        np.degrees(np.arccos(cosine))
                                    )

                channel_records.append(
                    {
                        "experiment": experiment.name,
                        "bend_deg": experiment.bend_deg,
                        "jacobian_contact": experiment.jacobian_contact,
                        "plant_contact": experiment.plant_contact,
                        "controller": experiment.controller,
                        "solver": experiment.solver,
                        "frame": frame,
                        "stage": stage,
                        "stage_1based": stage + 1,
                        "channel_index": channel,
                        "channel": CHANNEL_NAMES[channel],
                        "command": u,
                        "input_scale": scale,
                        "command_utilization_abs": abs(u)
                        / max(abs(scale), EPS),
                        "B_x": float(b[0]),
                        "B_y": float(b[1]),
                        "B_z": float(b[2]),
                        "jacobian_gain_raw": authority_raw,
                        "jacobian_authority_scaled_m": authority_scaled,
                        "contribution_x_m": float(contribution[0]),
                        "contribution_y_m": float(contribution[1]),
                        "contribution_z_m": float(contribution[2]),
                        "contribution_norm_m": contribution_norm,
                        "tangential_contribution_m": tangential,
                        "lateral_contribution_m": lateral,
                        "gain_ratio_from_stage0": gain_ratio,
                        "direction_change_from_stage0_deg": direction_change_deg,
                        "solution_to_controller_gain_ratio": (
                            controller_solution_gain_ratio
                        ),
                        "solution_to_controller_angle_deg": (
                            controller_solution_angle_deg
                        ),
                        "progress": progress,
                    }
                )

    channel_df = pd.DataFrame(
        channel_records,
        columns=CHANNEL_METRIC_COLUMNS,
    )

    if rollouts:
        has_commands = (
            not channel_df.empty
            and np.any(
                np.isfinite(
                    pd.to_numeric(
                        channel_df["command"],
                        errors="coerce",
                    ).to_numpy(float)
                )
            )
        )
        has_jacobians = (
            not channel_df.empty
            and np.any(
                np.isfinite(
                    pd.to_numeric(
                        channel_df["jacobian_gain_raw"],
                        errors="coerce",
                    ).to_numpy(float)
                )
            )
        )

        if not has_commands:
            warnings.warn(
                f"{experiment.name}: no per-channel command metrics were built. "
                "The rollout NPZ files likely do not contain a usable U_seq."
            )
        elif not has_jacobians:
            warnings.warn(
                f"{experiment.name}: command metrics were built, but no "
                "per-channel Jacobian metrics were available. The rollout NPZ "
                "files likely do not contain B_sequence_solution, "
                "B_sequence_linearisation, or B_sequence. Normalized-command "
                "plots will be generated; Jacobian authority and B*u "
                "contribution plots will be skipped."
            )

    return (
        pd.DataFrame(stage_records),
        channel_df,
        pd.DataFrame(frame_records),
    )


def finite_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q = float(np.clip(q, 0.5, 1.0))
    value = float(np.quantile(values, q))
    if value <= 0.0:
        value = float(np.max(values))
    return value if value > 0.0 else None


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_by_stage(
    stage_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    if stage_df.empty:
        return

    metrics = (
        ("lin_nl_error_m", "Nonlinear plan − linear prediction"),
        ("tracking_nonlinear_error_m", "Nonlinear plan − reference"),
        ("applied_linear_error_m", "Applied rollout − linear prediction"),
        ("applied_nonlinear_error_m", "Applied rollout − nonlinear plan"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = False
    for column, label in metrics:
        if column not in stage_df:
            continue
        grouped = stage_df.groupby("stage_1based")[column]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        valid = np.isfinite(median.to_numpy(float))
        if not np.any(valid):
            continue

        x = median.index.to_numpy(float)
        y = 1e3 * median.to_numpy(float)
        lower = 1e3 * q25.to_numpy(float)
        upper = 1e3 * q75.to_numpy(float)

        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, lower, upper, alpha=0.18)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Prediction stage")
    ax.set_ylabel("Position error [mm]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_first_stage_errors(
    frame_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    if frame_df.empty:
        return

    metrics = (
        ("first_lin_nl_error_m", "Nonlinear plan − linear prediction"),
        ("first_applied_lin_error_m", "Applied rollout − linear prediction"),
        ("first_applied_nl_error_m", "Applied rollout − nonlinear plan"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for column, label in metrics:
        if column not in frame_df:
            continue
        values = pd.to_numeric(frame_df[column], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(values)):
            continue
        ax.plot(
            frame_df["frame"].to_numpy(int),
            1e3 * values,
            marker="o",
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Feedback frame")
    ax.set_ylabel("First-stage position error [mm]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path, dpi)


def pivot_channel_stage(
    channel_df: pd.DataFrame,
    value_column: str,
    *,
    channels: Sequence[str],
) -> pd.DataFrame:
    required = {"channel", "stage_1based", value_column}
    if channel_df.empty or not required.issubset(channel_df.columns):
        return pd.DataFrame()

    subset = channel_df[channel_df["channel"].isin(channels)]
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["channel", "stage_1based"])[value_column]
        .median()
        .unstack("stage_1based")
    )
    return grouped.reindex(channels)


def plot_heatmap(
    matrix_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    dpi: int,
    heatmap_quantile: float,
    value_multiplier: float = 1.0,
) -> None:
    if matrix_df.empty:
        return

    values = value_multiplier * matrix_df.to_numpy(float)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return

    vmax = finite_quantile(np.abs(values), heatmap_quantile)
    fig, ax = plt.subplots(figsize=(8, 5))
    kwargs: dict[str, Any] = {
        "origin": "lower",
        "aspect": "auto",
    }
    if vmax is not None:
        kwargs["vmin"] = 0.0
        kwargs["vmax"] = vmax

    image = ax.imshow(values, **kwargs)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index.tolist())
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns.tolist())
    ax.set_xlabel("Prediction stage")
    ax.set_ylabel("Control channel")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_first_stage_channel_lines(
    channel_df: pd.DataFrame,
    value_column: str,
    path: Path,
    *,
    channels: Sequence[str],
    title: str,
    ylabel: str,
    dpi: int,
    value_multiplier: float = 1.0,
) -> None:
    required = {"stage", "channel", "frame", value_column}
    if channel_df.empty or not required.issubset(channel_df.columns):
        return

    subset = channel_df[
        (channel_df["stage"] == 0) & channel_df["channel"].isin(channels)
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for channel in channels:
        data = subset[subset["channel"] == channel].sort_values("frame")
        values = pd.to_numeric(data[value_column], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(values)):
            continue
        ax.plot(
            data["frame"].to_numpy(int),
            value_multiplier * values,
            marker="o",
            label=channel,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Feedback frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_insertion_first_stage(
    channel_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    required = {"stage", "channel", "frame", "command_utilization_abs"}
    if channel_df.empty or not required.issubset(channel_df.columns):
        return

    subset = channel_df[
        (channel_df["stage"] == 0) & (channel_df["channel"] == "dL")
    ].sort_values("frame")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        subset["frame"].to_numpy(int),
        subset["command_utilization_abs"].to_numpy(float),
        marker="o",
    )
    ax.axhline(1.0, linestyle="--", label="Characteristic/maximum input")
    ax.set_xlabel("Feedback frame")
    ax.set_ylabel("Absolute insertion utilization")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_contribution_fraction_heatmap(
    channel_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    required = {"frame", "stage", "channel", "contribution_norm_m"}
    if channel_df.empty or not required.issubset(channel_df.columns):
        return

    grouped = (
        channel_df.groupby(["frame", "stage", "channel"])["contribution_norm_m"]
        .median()
        .reset_index()
    )
    totals = grouped.groupby(["frame", "stage"])["contribution_norm_m"].transform("sum")
    grouped["fraction"] = grouped["contribution_norm_m"] / np.maximum(
        totals.to_numpy(float), EPS
    )

    matrix = (
        grouped.groupby(["channel", "stage"])["fraction"]
        .median()
        .unstack("stage")
        .reindex(CHANNEL_NAMES)
    )
    matrix.columns = [int(value) + 1 for value in matrix.columns]

    plot_heatmap(
        matrix,
        path,
        title=title,
        colorbar_label="Median fraction of summed channel contributions",
        dpi=dpi,
        heatmap_quantile=1.0,
        value_multiplier=1.0,
    )


def plot_jacobian_change(
    stage_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    if stage_df.empty:
        return

    columns = (
        ("B_change_from_stage0_relative", "Controller B change from stage 1"),
        ("B_drift_relative", "Selected-trajectory B versus controller B"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for column, label in columns:
        if column not in stage_df:
            continue
        grouped = stage_df.groupby("stage_1based")[column]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        values = median.to_numpy(float)
        if not np.any(np.isfinite(values)):
            continue
        x = median.index.to_numpy(float)
        ax.plot(x, values, marker="o", label=label)
        ax.fill_between(
            x,
            q25.to_numpy(float),
            q75.to_numpy(float),
            alpha=0.18,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Prediction stage")
    ax.set_ylabel("Relative Frobenius-norm change")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path, dpi)


def save_per_experiment_plots(
    experiment: Experiment,
    stage_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
    heatmap_quantile: float,
) -> None:
    exp_dir = output_dir / "per_experiment" / sanitize_filename(experiment.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = (
        f"{experiment.name}\n"
        f"bend={experiment.bend_deg}, controller={experiment.controller}, "
        f"solver={experiment.solver}"
    )

    plot_prediction_error_by_stage(
        stage_df,
        exp_dir / "prediction_error_by_stage.png",
        title=f"{title_prefix}\nPrediction and tracking error by horizon stage",
        dpi=dpi,
    )
    plot_first_stage_errors(
        frame_df,
        exp_dir / "first_stage_prediction_error_vs_frame.png",
        title=f"{title_prefix}\nFirst-stage prediction error",
        dpi=dpi,
    )

    source_util = pivot_channel_stage(
        channel_df,
        "command_utilization_abs",
        channels=SOURCE_CHANNEL_NAMES,
    )
    plot_heatmap(
        source_util,
        exp_dir / "source_command_utilization_by_stage.png",
        title=f"{title_prefix}\nSource-magnet command utilization",
        colorbar_label="Median |u| / characteristic input",
        dpi=dpi,
        heatmap_quantile=heatmap_quantile,
    )

    plot_first_stage_channel_lines(
        channel_df,
        "command_utilization_abs",
        exp_dir / "source_command_utilization_vs_frame.png",
        channels=SOURCE_CHANNEL_NAMES,
        title=f"{title_prefix}\nFirst-stage source-magnet command utilization",
        ylabel="Absolute command utilization",
        dpi=dpi,
    )

    plot_insertion_first_stage(
        channel_df,
        exp_dir / "insertion_utilization_vs_frame.png",
        title=f"{title_prefix}\nFirst-stage insertion utilization",
        dpi=dpi,
    )

    source_contribution = pivot_channel_stage(
        channel_df,
        "contribution_norm_m",
        channels=SOURCE_CHANNEL_NAMES,
    )
    plot_heatmap(
        source_contribution,
        exp_dir / "source_tip_contribution_by_stage.png",
        title=f"{title_prefix}\nSelected source-magnet tip contribution",
        colorbar_label="Median ||B[:,j] u_j|| [mm]",
        dpi=dpi,
        heatmap_quantile=heatmap_quantile,
        value_multiplier=1e3,
    )

    insertion_contribution = pivot_channel_stage(
        channel_df,
        "contribution_norm_m",
        channels=("dL",),
    )
    plot_heatmap(
        insertion_contribution,
        exp_dir / "insertion_tip_contribution_by_stage.png",
        title=f"{title_prefix}\nSelected insertion tip contribution",
        colorbar_label="Median ||B[:,dL] u_dL|| [mm]",
        dpi=dpi,
        heatmap_quantile=heatmap_quantile,
        value_multiplier=1e3,
    )

    plot_first_stage_channel_lines(
        channel_df,
        "contribution_norm_m",
        exp_dir / "source_tip_contribution_vs_frame.png",
        channels=SOURCE_CHANNEL_NAMES,
        title=f"{title_prefix}\nFirst-stage source-magnet tip contribution",
        ylabel="||B[:,j] u_j|| [mm]",
        dpi=dpi,
        value_multiplier=1e3,
    )

    plot_contribution_fraction_heatmap(
        channel_df,
        exp_dir / "fractional_tip_contribution_by_stage.png",
        title=f"{title_prefix}\nFractional channel contribution",
        dpi=dpi,
    )

    authority = pivot_channel_stage(
        channel_df,
        "jacobian_authority_scaled_m",
        channels=CHANNEL_NAMES,
    )
    plot_heatmap(
        authority,
        exp_dir / "scaled_jacobian_authority_by_stage.png",
        title=f"{title_prefix}\nScaled Jacobian authority",
        colorbar_label="Median ||B[:,j] input_scale_j|| [mm]",
        dpi=dpi,
        heatmap_quantile=heatmap_quantile,
        value_multiplier=1e3,
    )

    tangential = pivot_channel_stage(
        channel_df,
        "tangential_contribution_m",
        channels=CHANNEL_NAMES,
    )
    if not tangential.empty:
        # Signed values need symmetric display limits.
        values = 1e3 * tangential.to_numpy(float)
        vmax = finite_quantile(np.abs(values), heatmap_quantile)
        if np.any(np.isfinite(values)):
            fig, ax = plt.subplots(figsize=(8, 5))
            kwargs: dict[str, Any] = {"origin": "lower", "aspect": "auto"}
            if vmax is not None:
                kwargs.update({"vmin": -vmax, "vmax": vmax})
            image = ax.imshow(values, **kwargs)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("Median signed tangential contribution [mm]")
            ax.set_yticks(np.arange(tangential.shape[0]))
            ax.set_yticklabels(tangential.index.tolist())
            ax.set_xticks(np.arange(tangential.shape[1]))
            ax.set_xticklabels(tangential.columns.tolist())
            ax.set_xlabel("Prediction stage")
            ax.set_ylabel("Control channel")
            ax.set_title(f"{title_prefix}\nTangential tip contribution")
            fig.tight_layout()
            save_figure(
                fig,
                exp_dir / "tangential_tip_contribution_by_stage.png",
                dpi,
            )

    lateral = pivot_channel_stage(
        channel_df,
        "lateral_contribution_m",
        channels=CHANNEL_NAMES,
    )
    plot_heatmap(
        lateral,
        exp_dir / "lateral_tip_contribution_by_stage.png",
        title=f"{title_prefix}\nLateral steering contribution",
        colorbar_label="Median lateral contribution [mm]",
        dpi=dpi,
        heatmap_quantile=heatmap_quantile,
        value_multiplier=1e3,
    )

    plot_jacobian_change(
        stage_df,
        exp_dir / "jacobian_change_by_stage.png",
        title=f"{title_prefix}\nJacobian change over the prediction horizon",
        dpi=dpi,
    )


def rms(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def summarize_experiment(
    experiment: Experiment,
    stage_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment": experiment.name,
        "bend_deg": experiment.bend_deg,
        "jacobian_contact": experiment.jacobian_contact,
        "plant_contact": experiment.plant_contact,
        "controller": experiment.controller,
        "solver": experiment.solver,
        "rollout_steps": experiment.rollout_steps,
        "Np_name": experiment.Np,
        "kp": experiment.kp,
        "failed": experiment.failed,
        "failure_text": experiment.failure_text,
        "num_frames": int(frame_df["frame"].nunique()) if not frame_df.empty else 0,
        "num_stage_rows": int(len(stage_df)),
        "num_channel_rows": int(len(channel_df)),
    }

    stage_metrics = (
        "lin_nl_error_m",
        "tracking_linear_error_m",
        "tracking_nonlinear_error_m",
        "applied_linear_error_m",
        "applied_nonlinear_error_m",
        "applied_tracking_error_m",
        "B_drift_relative",
        "B_change_from_stage0_relative",
    )
    for metric in stage_metrics:
        if metric not in stage_df:
            summary[f"{metric}_rms"] = np.nan
            summary[f"{metric}_median"] = np.nan
            summary[f"{metric}_max"] = np.nan
            continue
        values = pd.to_numeric(stage_df[metric], errors="coerce")
        summary[f"{metric}_rms"] = rms(values)
        summary[f"{metric}_median"] = float(values.median())
        summary[f"{metric}_max"] = float(values.max())

    first_stage = stage_df[stage_df["stage"] == 0] if not stage_df.empty else stage_df
    for metric in (
        "lin_nl_error_m",
        "tracking_nonlinear_error_m",
        "applied_linear_error_m",
        "applied_nonlinear_error_m",
        "applied_tracking_error_m",
    ):
        values = (
            pd.to_numeric(first_stage[metric], errors="coerce")
            if metric in first_stage
            else pd.Series(dtype=float)
        )
        summary[f"first_{metric}_rms"] = rms(values)
        summary[f"first_{metric}_median"] = float(values.median())

    if not channel_df.empty:
        first_channel = channel_df[channel_df["stage"] == 0]
        for channel in CHANNEL_NAMES:
            data = first_channel[first_channel["channel"] == channel]
            summary[f"first_utilization_median_{channel}"] = float(
                pd.to_numeric(data["command_utilization_abs"], errors="coerce").median()
            )
            summary[f"first_contribution_median_mm_{channel}"] = float(
                1e3
                * pd.to_numeric(data["contribution_norm_m"], errors="coerce").median()
            )
            summary[f"first_authority_median_mm_{channel}"] = float(
                1e3
                * pd.to_numeric(
                    data["jacobian_authority_scaled_m"], errors="coerce"
                ).median()
            )

        dL_data = first_channel[first_channel["channel"] == "dL"]
        dL_util = pd.to_numeric(
            dL_data["command_utilization_abs"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(dL_util)
        summary["dL_near_limit_fraction"] = (
            float(np.mean(dL_util[valid] >= 0.99)) if np.any(valid) else np.nan
        )
    else:
        summary["dL_near_limit_fraction"] = np.nan

    return summary


def experiment_label(row: pd.Series) -> str:
    bend = row.get("bend_deg")
    bend_text = "bend?" if pd.isna(bend) else f"{float(bend):g}°"
    jac = row.get("jacobian_contact")
    if pd.isna(jac):
        jac_text = "jac?"
    else:
        jac_text = "jac-contact" if bool(jac) else "jac-no-contact"
    return f"{bend_text} | {row.get('controller')} | {row.get('solver')} | {jac_text}"


def plot_cross_experiment_bars(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    if summary_df.empty:
        return

    plots = (
        (
            "first_lin_nl_error_m_rms",
            "First-stage linear versus nonlinear prediction error",
            "RMS error [mm]",
            1e3,
            "comparison_first_stage_linearisation_error.png",
        ),
        (
            "first_applied_linear_error_m_rms",
            "First-stage applied rollout versus linear prediction",
            "RMS error [mm]",
            1e3,
            "comparison_first_stage_execution_prediction_error.png",
        ),
        (
            "tracking_nonlinear_error_m_rms",
            "Nonlinear planned tracking error",
            "RMS error [mm]",
            1e3,
            "comparison_nonlinear_tracking_error.png",
        ),
        (
            "dL_near_limit_fraction",
            "Fraction of frames with insertion near characteristic limit",
            "Fraction",
            1.0,
            "comparison_dL_near_limit_fraction.png",
        ),
    )

    labels = [experiment_label(row) for _, row in summary_df.iterrows()]

    for column, title, ylabel, multiplier, filename in plots:
        if column not in summary_df:
            continue
        values = pd.to_numeric(summary_df[column], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(values)):
            continue

        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(labels))))
        positions = np.arange(len(labels))
        ax.barh(positions, multiplier * values)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="x")
        fig.tight_layout()
        save_figure(fig, output_dir / "cross_experiment" / filename, dpi)


def contact_pair_key(experiment: Experiment) -> tuple[Any, ...]:
    return (
        experiment.bend_deg,
        experiment.plant_contact,
        experiment.controller,
        experiment.solver,
        experiment.rollout_steps,
        experiment.Np,
        experiment.kp,
    )


def first_stage_B_samples(
    experiment: Experiment,
    rollouts: list[tuple[int, Path, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    """
    Extract the local first-stage Jacobian from every feedback frame.

    B[0] is the Jacobian at the beginning of the receding-horizon solve.
    Storing U_seq[0] as well lets the analysis distinguish:

        Jacobian authority:
            ||B[:, j] s_j||

        Selected closed-loop contribution:
            ||B[:, j] u_j||

    The first is a property of the local model and chosen characteristic
    input scale. The second also depends on the command selected by the
    controller.
    """
    lumen_C: np.ndarray | None = None
    lumen_path = experiment.path / "lumen_C.npy"
    if lumen_path.exists():
        try:
            lumen_C = np.asarray(np.load(lumen_path), float)
        except Exception:
            lumen_C = None

    samples: list[dict[str, Any]] = []

    for frame, path, data in rollouts:
        B = first_array(
            data,
            (
                "B_sequence_solution",
                "B_sequence_linearisation",
                "B_sequence",
            ),
            ndim=3,
        )
        if B is None or B.shape[0] == 0:
            continue

        p_start = first_array(data, ("p_start",))
        progress = extract_progress(data, lumen_C)

        U_seq = first_array(data, ("U_seq",), ndim=2)
        u0 = None
        if U_seq is not None and U_seq.shape[0] > 0:
            u0 = np.asarray(U_seq[0], float).reshape(-1)

        input_scale = first_array(
            data,
            ("jacobian_diag_input_scale",),
        )
        if input_scale is not None:
            input_scale = np.asarray(
                input_scale,
                float,
            ).reshape(-1)

        sample: dict[str, Any] = {
            "frame": int(frame),
            "path": str(path),
            "progress": float(progress),
            "B": np.asarray(B[0], float),
            "p_start": None,
            "u0": u0,
            "input_scale": input_scale,
        }

        if p_start is not None and p_start.size >= 8:
            sample["p_start"] = np.asarray(
                p_start,
                float,
            ).reshape(-1)[:8]

        samples.append(sample)

    return samples


def pair_samples_by_progress(
    contact_samples: list[dict[str, Any]],
    no_contact_samples: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not contact_samples or not no_contact_samples:
        return []

    available = set(range(len(no_contact_samples)))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for contact in contact_samples:
        if not available:
            break

        progress_c = float(contact["progress"])
        candidate_indices = sorted(available)

        if np.isfinite(progress_c):
            distances = []
            for index in candidate_indices:
                progress_n = float(no_contact_samples[index]["progress"])
                distance = (
                    abs(progress_c - progress_n)
                    if np.isfinite(progress_n)
                    else np.inf
                )
                distances.append(distance)
            chosen = candidate_indices[int(np.argmin(distances))]
        else:
            chosen = min(
                candidate_indices,
                key=lambda index: abs(
                    int(contact["frame"]) - int(no_contact_samples[index]["frame"])
                ),
            )

        available.remove(chosen)
        pairs.append((contact, no_contact_samples[chosen]))

    return pairs


def _shared_channel_scale(
    contact_sample: dict[str, Any],
    no_contact_sample: dict[str, Any],
    channel: int,
) -> tuple[float, float]:
    """
    Return one common scale for comparing contact and no-contact authority.

    The no-contact scale is used as the reference when available. Using the
    same scale on both Jacobians ensures that any plotted difference comes
    from B rather than from different plotting scales.

    Returns
    -------
    scale:
        Positive common characteristic input.
    relative_scale_mismatch:
        |s_contact - s_no_contact| / |s_no_contact| when both are available.
    """
    scale_contact = contact_sample.get("input_scale")
    scale_no_contact = no_contact_sample.get("input_scale")

    value_contact = np.nan
    value_no_contact = np.nan

    if scale_contact is not None and channel < len(scale_contact):
        candidate = float(scale_contact[channel])
        if np.isfinite(candidate) and abs(candidate) > EPS:
            value_contact = abs(candidate)

    if scale_no_contact is not None and channel < len(scale_no_contact):
        candidate = float(scale_no_contact[channel])
        if np.isfinite(candidate) and abs(candidate) > EPS:
            value_no_contact = abs(candidate)

    if np.isfinite(value_no_contact):
        scale = value_no_contact
    elif np.isfinite(value_contact):
        scale = value_contact
    else:
        scale = 1.0

    mismatch = np.nan
    if np.isfinite(value_contact) and np.isfinite(value_no_contact):
        mismatch = abs(value_contact - value_no_contact) / max(
            value_no_contact,
            EPS,
        )

    return float(scale), float(mismatch)


def compare_contact_pair(
    contact_exp: Experiment,
    no_contact_exp: Experiment,
    contact_rollouts: list[tuple[int, Path, dict[str, np.ndarray]]],
    no_contact_rollouts: list[tuple[int, Path, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    """
    Compare first-stage Jacobians over feedback frames.

    Samples are paired by nearest reference progress. This is appropriate for
    comparing two closed-loop runs, but it is not a strict same-pose model
    comparison because the source poses can differ. Pose differences are
    therefore saved in every row.

    Pure Jacobian metrics
    ---------------------
    gain_contact / gain_no_contact
        Raw column magnitudes.

    scaled_authority_contact_m / scaled_authority_no_contact_m
        ||B[:, j] s_j|| using one common characteristic input scale.

    gain_ratio_contact_to_no_contact
        ||b_contact|| / ||b_no_contact||.

    direction_difference_deg
        Angle between b_contact and b_no_contact.

    column_relative_change
        ||b_contact - b_no_contact|| / ||b_no_contact||.

    Selected-command metrics
    ------------------------
    contribution_contact_m / contribution_no_contact_m
        ||B[:, j] u_j||. These combine the Jacobian and the command selected
        by each controller and must not be interpreted as a pure Jacobian
        difference.
    """
    contact_samples = first_stage_B_samples(
        contact_exp,
        contact_rollouts,
    )
    no_contact_samples = first_stage_B_samples(
        no_contact_exp,
        no_contact_rollouts,
    )
    pairs = pair_samples_by_progress(
        contact_samples,
        no_contact_samples,
    )

    records: list[dict[str, Any]] = []

    for pair_index, (contact, no_contact) in enumerate(pairs):
        Bc = np.asarray(contact["B"], float)
        Bn = np.asarray(no_contact["B"], float)

        if Bc.ndim != 2 or Bn.ndim != 2:
            continue

        m = min(
            Bc.shape[1],
            Bn.shape[1],
            len(CHANNEL_NAMES),
        )

        progress_contact = float(contact["progress"])
        progress_no_contact = float(no_contact["progress"])

        mean_progress = np.nan
        if (
            np.isfinite(progress_contact)
            and np.isfinite(progress_no_contact)
        ):
            mean_progress = 0.5 * (
                progress_contact
                + progress_no_contact
            )

        pose_position_difference_mm = np.nan
        pose_rotation_difference_deg = np.nan
        pose_length_difference_mm = np.nan

        if (
            contact["p_start"] is not None
            and no_contact["p_start"] is not None
        ):
            pc = np.asarray(
                contact["p_start"],
                float,
            )
            pn = np.asarray(
                no_contact["p_start"],
                float,
            )

            pose_position_difference_mm = float(
                1e3 * np.linalg.norm(
                    pc[:3] - pn[:3]
                )
            )
            pose_rotation_difference_deg = (
                quaternion_angle_deg(
                    pc[3:7],
                    pn[3:7],
                )
            )
            pose_length_difference_mm = float(
                1e3 * abs(pc[7] - pn[7])
            )

        B_difference = Bc - Bn

        full_relative_change = float(
            np.linalg.norm(B_difference)
            / max(np.linalg.norm(Bn), EPS)
        )

        source_columns = min(6, m)
        source_relative_change = np.nan
        if source_columns > 0:
            source_relative_change = float(
                np.linalg.norm(
                    B_difference[:, :source_columns]
                )
                / max(
                    np.linalg.norm(
                        Bn[:, :source_columns]
                    ),
                    EPS,
                )
            )

        dL_relative_change = np.nan
        if m > 6:
            dL_relative_change = float(
                np.linalg.norm(
                    B_difference[:, 6]
                )
                / max(
                    np.linalg.norm(Bn[:, 6]),
                    EPS,
                )
            )

        # Use a scale relative to the complete matrix to suppress direction
        # angles for numerically zero columns.
        matrix_scale = max(
            float(np.linalg.norm(Bc)),
            float(np.linalg.norm(Bn)),
            EPS,
        )
        direction_eps = max(
            EPS,
            1e-8 * matrix_scale,
        )

        u_contact = contact.get("u0")
        u_no_contact = no_contact.get("u0")

        for channel in range(m):
            bc = np.asarray(
                Bc[:3, channel],
                float,
            )
            bn = np.asarray(
                Bn[:3, channel],
                float,
            )

            gain_contact = float(
                np.linalg.norm(bc)
            )
            gain_no_contact = float(
                np.linalg.norm(bn)
            )

            gain_ratio = np.nan
            relative_gain_difference = np.nan
            column_relative_change = np.nan

            if gain_no_contact > direction_eps:
                gain_ratio = (
                    gain_contact
                    / gain_no_contact
                )
                relative_gain_difference = (
                    gain_contact
                    - gain_no_contact
                ) / gain_no_contact

                column_relative_change = float(
                    np.linalg.norm(bc - bn)
                    / gain_no_contact
                )

            angle_deg = np.nan
            if (
                gain_contact > direction_eps
                and gain_no_contact > direction_eps
            ):
                cosine = float(
                    np.clip(
                        np.dot(bc, bn)
                        / (
                            gain_contact
                            * gain_no_contact
                        ),
                        -1.0,
                        1.0,
                    )
                )
                angle_deg = float(
                    np.degrees(
                        np.arccos(cosine)
                    )
                )

            common_scale, scale_mismatch = (
                _shared_channel_scale(
                    contact,
                    no_contact,
                    channel,
                )
            )

            authority_contact = (
                gain_contact * common_scale
            )
            authority_no_contact = (
                gain_no_contact * common_scale
            )

            command_contact = np.nan
            command_no_contact = np.nan
            contribution_contact = np.nan
            contribution_no_contact = np.nan
            contribution_ratio = np.nan

            if (
                u_contact is not None
                and channel < len(u_contact)
            ):
                command_contact = float(
                    u_contact[channel]
                )
                contribution_contact = float(
                    np.linalg.norm(
                        bc * command_contact
                    )
                )

            if (
                u_no_contact is not None
                and channel < len(u_no_contact)
            ):
                command_no_contact = float(
                    u_no_contact[channel]
                )
                contribution_no_contact = float(
                    np.linalg.norm(
                        bn * command_no_contact
                    )
                )

            if (
                np.isfinite(contribution_contact)
                and np.isfinite(contribution_no_contact)
                and contribution_no_contact > EPS
            ):
                contribution_ratio = (
                    contribution_contact
                    / contribution_no_contact
                )

            records.append(
                {
                    "pair_key": str(
                        contact_pair_key(contact_exp)
                    ),
                    "contact_experiment": (
                        contact_exp.name
                    ),
                    "no_contact_experiment": (
                        no_contact_exp.name
                    ),
                    "pair_index": int(pair_index),
                    "contact_frame": int(
                        contact["frame"]
                    ),
                    "no_contact_frame": int(
                        no_contact["frame"]
                    ),
                    "contact_progress": (
                        progress_contact
                    ),
                    "no_contact_progress": (
                        progress_no_contact
                    ),
                    "mean_progress": (
                        mean_progress
                    ),
                    "progress_difference": (
                        abs(
                            progress_contact
                            - progress_no_contact
                        )
                        if (
                            np.isfinite(progress_contact)
                            and np.isfinite(
                                progress_no_contact
                            )
                        )
                        else np.nan
                    ),
                    "pose_position_difference_mm": (
                        pose_position_difference_mm
                    ),
                    "pose_rotation_difference_deg": (
                        pose_rotation_difference_deg
                    ),
                    "pose_length_difference_mm": (
                        pose_length_difference_mm
                    ),
                    "full_matrix_relative_change": (
                        full_relative_change
                    ),
                    "source_matrix_relative_change": (
                        source_relative_change
                    ),
                    "dL_column_relative_change": (
                        dL_relative_change
                    ),
                    "channel_index": int(channel),
                    "channel": CHANNEL_NAMES[channel],
                    "common_input_scale": (
                        common_scale
                    ),
                    "input_scale_relative_mismatch": (
                        scale_mismatch
                    ),
                    "gain_contact": (
                        gain_contact
                    ),
                    "gain_no_contact": (
                        gain_no_contact
                    ),
                    "scaled_authority_contact_m": (
                        authority_contact
                    ),
                    "scaled_authority_no_contact_m": (
                        authority_no_contact
                    ),
                    "gain_ratio_contact_to_no_contact": (
                        gain_ratio
                    ),
                    "relative_gain_difference": (
                        relative_gain_difference
                    ),
                    "direction_difference_deg": (
                        angle_deg
                    ),
                    "column_relative_change": (
                        column_relative_change
                    ),
                    "command_contact": (
                        command_contact
                    ),
                    "command_no_contact": (
                        command_no_contact
                    ),
                    "contribution_contact_m": (
                        contribution_contact
                    ),
                    "contribution_no_contact_m": (
                        contribution_no_contact
                    ),
                    "contribution_ratio_contact_to_no_contact": (
                        contribution_ratio
                    ),
                }
            )

    return pd.DataFrame(records)


def _contact_comparison_x(
    pair_df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """
    Prefer physical reference progress for the x-axis.

    If progress is unavailable, fall back to the paired-sample index.
    """
    progress = pd.to_numeric(
        pair_df["mean_progress"],
        errors="coerce",
    )

    if (
        np.any(np.isfinite(progress.to_numpy(float)))
        and float(progress.max() - progress.min()) > EPS
    ):
        return 1e3 * progress, "Mean matched reference progress [mm]"

    return (
        pd.to_numeric(
            pair_df["pair_index"],
            errors="coerce",
        ),
        "Matched feedback-frame pair",
    )


def _plot_pair_metric_lines(
    *,
    pair_df: pd.DataFrame,
    value_column: str,
    channels: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    dpi: int,
    reference_value: float | None = None,
) -> None:
    if value_column not in pair_df.columns:
        return

    colours = {
        "vx": "#0072B2",
        "vy": "#E69F00",
        "vz": "#009E73",
        "wx": "#D55E00",
        "wy": "#CC79A7",
        "wz": "#56B4E9",
        "dL": "#222222",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False

    for channel in channels:
        data = pair_df[
            pair_df["channel"] == channel
        ].sort_values("pair_index")

        if data.empty:
            continue

        values = pd.to_numeric(
            data[value_column],
            errors="coerce",
        ).to_numpy(float)

        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        x, xlabel = _contact_comparison_x(data)

        ax.plot(
            x.to_numpy(float)[valid],
            values[valid],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=channel,
            color=colours.get(channel),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    if reference_value is not None:
        ax.axhline(
            float(reference_value),
            linestyle="--",
            linewidth=1.0,
            color="0.35",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def _plot_contact_no_contact_channel_pair(
    *,
    pair_df: pd.DataFrame,
    channel: str,
    contact_column: str,
    no_contact_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    dpi: int,
    multiplier: float = 1.0,
) -> None:
    data = pair_df[
        pair_df["channel"] == channel
    ].sort_values("pair_index")

    if data.empty:
        return

    contact_values = (
        multiplier
        * pd.to_numeric(
            data[contact_column],
            errors="coerce",
        ).to_numpy(float)
    )
    no_contact_values = (
        multiplier
        * pd.to_numeric(
            data[no_contact_column],
            errors="coerce",
        ).to_numpy(float)
    )

    if not (
        np.any(np.isfinite(contact_values))
        or np.any(np.isfinite(no_contact_values))
    ):
        return

    x, xlabel = _contact_comparison_x(data)
    x_values = x.to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    valid_contact = np.isfinite(contact_values)
    if np.any(valid_contact):
        ax.plot(
            x_values[valid_contact],
            contact_values[valid_contact],
            marker="o",
            linewidth=2.0,
            label="Contact Jacobian",
        )

    valid_no_contact = np.isfinite(no_contact_values)
    if np.any(valid_no_contact):
        ax.plot(
            x_values[valid_no_contact],
            no_contact_values[valid_no_contact],
            marker="s",
            linewidth=2.0,
            label="No-contact Jacobian",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_contact_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    """
    Save aggregate and per-frame contact/no-contact Jacobian comparisons.

    Source-pose channels and dL are separated so insertion cannot compress the
    steering-channel plots.
    """
    if comparison_df.empty:
        return

    for pair_key, pair_df in comparison_df.groupby(
        "pair_key"
    ):
        pair_name = sanitize_filename(
            str(pair_key)
        )
        pair_dir = (
            output_dir
            / "contact_vs_no_contact"
            / pair_name
        )
        pair_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        grouped = pair_df.groupby("channel")

        gain_ratio = (
            grouped[
                "gain_ratio_contact_to_no_contact"
            ]
            .median()
            .reindex(CHANNEL_NAMES)
        )
        angle = (
            grouped[
                "direction_difference_deg"
            ]
            .median()
            .reindex(CHANNEL_NAMES)
        )
        relative_gain = (
            grouped[
                "relative_gain_difference"
            ]
            .median()
            .reindex(CHANNEL_NAMES)
        )

        # Existing aggregate summaries.
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            np.arange(len(CHANNEL_NAMES)),
            gain_ratio.to_numpy(float),
        )
        ax.axhline(
            1.0,
            linestyle="--",
            label="Equal gain",
        )
        ax.set_xticks(
            np.arange(len(CHANNEL_NAMES))
        )
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylabel(
            "Contact gain / no-contact gain"
        )
        ax.set_title(
            "Median contact versus no-contact "
            "Jacobian gain"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(
            fig,
            pair_dir
            / "jacobian_gain_ratio_median.png",
            dpi,
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            np.arange(len(CHANNEL_NAMES)),
            angle.to_numpy(float),
        )
        ax.set_xticks(
            np.arange(len(CHANNEL_NAMES))
        )
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylabel(
            "Column direction difference [deg]"
        )
        ax.set_title(
            "Median contact versus no-contact "
            "Jacobian direction difference"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        save_figure(
            fig,
            pair_dir
            / "jacobian_direction_difference_median.png",
            dpi,
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            np.arange(len(CHANNEL_NAMES)),
            relative_gain.to_numpy(float),
        )
        ax.axhline(
            0.0,
            linestyle="--",
        )
        ax.set_xticks(
            np.arange(len(CHANNEL_NAMES))
        )
        ax.set_xticklabels(CHANNEL_NAMES)
        ax.set_ylabel(
            "(gain_contact − gain_no_contact) "
            "/ gain_no_contact"
        )
        ax.set_title(
            "Median relative Jacobian gain change"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        save_figure(
            fig,
            pair_dir
            / "jacobian_relative_gain_change_median.png",
            dpi,
        )

        # Per-frame/progress pure Jacobian comparisons.
        _plot_pair_metric_lines(
            pair_df=pair_df,
            value_column=(
                "gain_ratio_contact_to_no_contact"
            ),
            channels=SOURCE_CHANNEL_NAMES,
            output_path=(
                pair_dir
                / "source_gain_ratio_vs_progress.png"
            ),
            title=(
                "Source-channel Jacobian gain ratio "
                "over the closed-loop trajectory"
            ),
            ylabel=(
                "Contact gain / no-contact gain"
            ),
            dpi=dpi,
            reference_value=1.0,
        )

        _plot_pair_metric_lines(
            pair_df=pair_df,
            value_column=(
                "direction_difference_deg"
            ),
            channels=SOURCE_CHANNEL_NAMES,
            output_path=(
                pair_dir
                / "source_direction_difference_vs_progress.png"
            ),
            title=(
                "Source-channel Jacobian direction "
                "difference over the trajectory"
            ),
            ylabel=(
                "Direction difference [deg]"
            ),
            dpi=dpi,
        )

        _plot_pair_metric_lines(
            pair_df=pair_df,
            value_column=(
                "column_relative_change"
            ),
            channels=SOURCE_CHANNEL_NAMES,
            output_path=(
                pair_dir
                / "source_column_relative_change_vs_progress.png"
            ),
            title=(
                "Source-channel relative Jacobian "
                "change over the trajectory"
            ),
            ylabel=(
                "||b_contact − b_no_contact|| "
                "/ ||b_no_contact||"
            ),
            dpi=dpi,
            reference_value=0.0,
        )

        # dL is deliberately separate.
        _plot_pair_metric_lines(
            pair_df=pair_df,
            value_column=(
                "gain_ratio_contact_to_no_contact"
            ),
            channels=("dL",),
            output_path=(
                pair_dir
                / "dL_gain_ratio_vs_progress.png"
            ),
            title=(
                "Insertion-column Jacobian gain ratio"
            ),
            ylabel=(
                "Contact gain / no-contact gain"
            ),
            dpi=dpi,
            reference_value=1.0,
        )

        _plot_pair_metric_lines(
            pair_df=pair_df,
            value_column=(
                "direction_difference_deg"
            ),
            channels=("dL",),
            output_path=(
                pair_dir
                / "dL_direction_difference_vs_progress.png"
            ),
            title=(
                "Insertion-column Jacobian direction "
                "difference"
            ),
            ylabel=(
                "Direction difference [deg]"
            ),
            dpi=dpi,
        )

        # Complete-matrix, source-only, and dL-only relative differences.
        pair_level = (
            pair_df.sort_values("pair_index")
            .drop_duplicates("pair_index")
        )
        x, xlabel = _contact_comparison_x(
            pair_level
        )
        x_values = x.to_numpy(float)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        matrix_metrics = (
            (
                "full_matrix_relative_change",
                "Full 7-channel Jacobian",
            ),
            (
                "source_matrix_relative_change",
                "Six source-pose channels",
            ),
            (
                "dL_column_relative_change",
                "Insertion column",
            ),
        )

        plotted = False
        for column, label in matrix_metrics:
            values = pd.to_numeric(
                pair_level[column],
                errors="coerce",
            ).to_numpy(float)
            valid = np.isfinite(values)
            if not np.any(valid):
                continue

            ax.plot(
                x_values[valid],
                values[valid],
                marker="o",
                linewidth=2.0,
                label=label,
            )
            plotted = True

        if plotted:
            ax.axhline(
                0.0,
                linestyle="--",
                linewidth=0.9,
                color="0.35",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(
                "Relative Frobenius-norm difference"
            )
            ax.set_title(
                "Contact versus no-contact Jacobian "
                "difference over the trajectory"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            save_figure(
                fig,
                pair_dir
                / "matrix_relative_change_vs_progress.png",
                dpi,
            )
        else:
            plt.close(fig)

        # Per-channel authority: pure B comparison using the same input scale.
        authority_dir = pair_dir / "authority_by_channel"
        contribution_dir = (
            pair_dir
            / "closed_loop_contribution_by_channel"
        )

        for channel in CHANNEL_NAMES:
            _plot_contact_no_contact_channel_pair(
                pair_df=pair_df,
                channel=channel,
                contact_column=(
                    "scaled_authority_contact_m"
                ),
                no_contact_column=(
                    "scaled_authority_no_contact_m"
                ),
                output_path=(
                    authority_dir
                    / f"{channel}_authority_vs_progress.png"
                ),
                title=(
                    f"{channel}: contact versus no-contact "
                    "Jacobian authority"
                ),
                ylabel=(
                    "||B[:,j] common_scale_j|| [mm]"
                ),
                dpi=dpi,
                multiplier=1e3,
            )

            # This plot combines B with each run's selected command.
            _plot_contact_no_contact_channel_pair(
                pair_df=pair_df,
                channel=channel,
                contact_column=(
                    "contribution_contact_m"
                ),
                no_contact_column=(
                    "contribution_no_contact_m"
                ),
                output_path=(
                    contribution_dir
                    / f"{channel}_contribution_vs_progress.png"
                ),
                title=(
                    f"{channel}: selected closed-loop "
                    "tip contribution"
                ),
                ylabel=(
                    "||B[:,j] u_j|| [mm]"
                ),
                dpi=dpi,
                multiplier=1e3,
            )

        # Save one compact pair-level diagnostics table.
        pair_level_columns = [
            "pair_index",
            "contact_frame",
            "no_contact_frame",
            "contact_progress",
            "no_contact_progress",
            "mean_progress",
            "progress_difference",
            "pose_position_difference_mm",
            "pose_rotation_difference_deg",
            "pose_length_difference_mm",
            "full_matrix_relative_change",
            "source_matrix_relative_change",
            "dL_column_relative_change",
        ]

        pair_level[
            pair_level_columns
        ].to_csv(
            pair_dir
            / "matched_frame_pairs.csv",
            index=False,
        )


def build_contact_comparisons(
    experiments: list[Experiment],
    rollout_cache: dict[str, list[tuple[int, Path, dict[str, np.ndarray]]]],
    output_dir: Path,
    *,
    dpi: int,
) -> pd.DataFrame:
    grouped: dict[tuple[Any, ...], dict[bool, Experiment]] = {}

    for experiment in experiments:
        if experiment.jacobian_contact is None:
            continue
        key = contact_pair_key(experiment)
        grouped.setdefault(key, {})[bool(experiment.jacobian_contact)] = experiment

    comparisons: list[pd.DataFrame] = []
    for key, variants in grouped.items():
        if True not in variants or False not in variants:
            continue
        contact_exp = variants[True]
        no_contact_exp = variants[False]
        comparison = compare_contact_pair(
            contact_exp,
            no_contact_exp,
            rollout_cache.get(contact_exp.name, []),
            rollout_cache.get(no_contact_exp.name, []),
        )
        if not comparison.empty:
            comparisons.append(comparison)

    if not comparisons:
        return pd.DataFrame()

    result = pd.concat(comparisons, ignore_index=True)
    result.to_csv(output_dir / "contact_vs_no_contact_metrics.csv", index=False)
    plot_contact_comparison(result, output_dir, dpi=dpi)
    return result


def write_readme(
    output_dir: Path,
    *,
    experiments: list[Experiment],
    contact_comparison_available: bool,
) -> None:
    lines = [
        "# Controller Grid Analysis",
        "",
        f"Experiments discovered: {len(experiments)}",
        "",
        "## Main outputs",
        "",
        "- `experiment_summary.csv`: one row per experiment.",
        "- `stage_metrics.csv`: one row per frame and prediction stage.",
        "- `channel_metrics.csv`: one row per frame, stage, and input channel.",
        "- `frame_metrics.csv`: one row per feedback frame.",
        "- `per_experiment/`: detailed figures for each run.",
        "- `cross_experiment/`: controller-level comparison figures.",
        "",
        "## Interpretation",
        "",
        "- `lin_nl_error_m` measures local linearization error.",
        "- `applied_nonlinear_error_m` measures nonlinear-plan versus executed rollout mismatch.",
        "- `command_utilization_abs` is `abs(u_j) / input_scale_j`.",
        "- `jacobian_authority_scaled_m` is `||B[:,j] input_scale_j||`.",
        "- `contribution_norm_m` is `||B[:,j] u_j||`.",
        "- `tangential_contribution_m` is signed progress along the reference tangent.",
        "- `lateral_contribution_m` is steering contribution normal to the reference tangent.",
        "",
        "Heatmap clipping is display-only. CSV values are never clipped.",
        "",
    ]

    if contact_comparison_available:
        lines.extend(
            [
                "## Contact versus no-contact comparison",
                "",
                "Matched contact/no-contact runs were found. Samples are paired by nearest",
                "reference progress. The script reports source-pose differences because",
                "trajectory differences can confound the Jacobian comparison.",
                "",
                "For a strict same-pose comparison, evaluate both Jacobian models at the",
                "same saved `p_start` using your model factory.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Contact versus no-contact comparison",
                "",
                "No matched contact/no-contact Jacobian experiment pairs were found.",
                "Generate runs that differ only in `jac_contact` while keeping the plant,",
                "controller, bend, horizon, rollout, and weights fixed.",
                "",
            ]
        )

    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        return 2

    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "analysis_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_scales = np.asarray(args.input_scales, float).reshape(7)
    if np.any(~np.isfinite(input_scales)) or np.any(input_scales <= 0.0):
        print(
            "[ERROR] All --input-scales must be finite and positive.",
            file=sys.stderr,
        )
        return 2

    experiments = discover_experiments(root)
    if not experiments:
        print(f"[ERROR] No experiment directories found beneath {root}.")
        return 1

    print(f"[ANALYSIS] Root: {root}")
    print(f"[ANALYSIS] Output: {output_dir}")
    print(f"[ANALYSIS] Experiments discovered: {len(experiments)}")

    all_stage: list[pd.DataFrame] = []
    all_channel: list[pd.DataFrame] = []
    all_frame: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    rollout_cache: dict[str, list[tuple[int, Path, dict[str, np.ndarray]]]] = {}

    included_experiments: list[Experiment] = []

    for index, experiment in enumerate(experiments, start=1):
        if experiment.failed and not args.include_failed:
            print(
                f"[SKIP {index}/{len(experiments)}] {experiment.name}: "
                "failure.txt present (use --include-failed for partial analysis)"
            )
            continue

        print(f"[LOAD {index}/{len(experiments)}] {experiment.name}")
        rollouts = load_rollouts(
            experiment,
            max_files=args.max_rollout_files,
        )
        rollout_cache[experiment.name] = rollouts

        if not rollouts:
            warnings.warn(f"No rollout NPZ files found for {experiment.name}")

        stage_df, channel_df, frame_df = build_metric_tables(
            experiment,
            rollouts,
            default_input_scales=input_scales,
            verbose=args.verbose,
        )

        save_per_experiment_plots(
            experiment,
            stage_df,
            channel_df,
            frame_df,
            output_dir,
            dpi=args.dpi,
            heatmap_quantile=args.heatmap_quantile,
        )

        summary_records.append(
            summarize_experiment(experiment, stage_df, channel_df, frame_df)
        )

        if not stage_df.empty:
            all_stage.append(stage_df)
        if not channel_df.empty:
            all_channel.append(channel_df)
        if not frame_df.empty:
            all_frame.append(frame_df)

        included_experiments.append(experiment)

    stage_all = (
        pd.concat(all_stage, ignore_index=True)
        if all_stage
        else pd.DataFrame()
    )
    channel_all = (
        pd.concat(all_channel, ignore_index=True)
        if all_channel
        else pd.DataFrame()
    )
    frame_all = (
        pd.concat(all_frame, ignore_index=True)
        if all_frame
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(summary_records)

    stage_all.to_csv(output_dir / "stage_metrics.csv", index=False)
    channel_all.to_csv(output_dir / "channel_metrics.csv", index=False)
    frame_all.to_csv(output_dir / "frame_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)

    plot_cross_experiment_bars(summary_df, output_dir, dpi=args.dpi)

    contact_df = build_contact_comparisons(
        included_experiments,
        rollout_cache,
        output_dir,
        dpi=args.dpi,
    )

    write_readme(
        output_dir,
        experiments=included_experiments,
        contact_comparison_available=not contact_df.empty,
    )

    print("")
    print("[DONE] Analysis complete.")
    print(f"[DONE] Summary: {output_dir / 'experiment_summary.csv'}")
    print(f"[DONE] Stage metrics: {output_dir / 'stage_metrics.csv'}")
    print(f"[DONE] Channel metrics: {output_dir / 'channel_metrics.csv'}")
    print(f"[DONE] Figures: {output_dir / 'per_experiment'}")

    if contact_df.empty:
        print(
            "[NOTE] No matched contact/no-contact Jacobian pairs were found. "
            "Current contact-only runs are still fully analyzed."
        )
    else:
        print(
            f"[DONE] Contact comparison: "
            f"{output_dir / 'contact_vs_no_contact_metrics.csv'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
