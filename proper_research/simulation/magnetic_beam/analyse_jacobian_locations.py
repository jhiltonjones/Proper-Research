from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


PACKAGE_NAME = __package__ or (
    f"proper_research.simulation.{Path(__file__).resolve().parent.name}"
)
_smoke = importlib.import_module(f"{PACKAGE_NAME}.run_solver_smoke_test")


CONTROL_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz", "Ldot")
TIP_NAMES = ("tip_x", "tip_y", "tip_z")
POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz", "L")
AXIS_INDEX = {name: index for index, name in enumerate(POSE_FIELDS)}


@dataclass(frozen=True)
class ProbePose:
    name: str
    p7: np.ndarray
    offset: float
    offset_unit: str


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_.") or "probe"


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _clear_sensitivity_history(model) -> None:
    """
    Retain the nominal forward equilibrium but remove all reusable Jacobian data.

    This produces the ``fast_fresh`` calculation: forward differences without
    inheriting a Hessian from an earlier location.
    """
    for name in (
        "last_J_tip_pose7",
        "last_J_tip_actuation_tangent",
        "last_sens_info",
        "last_jacobian_diag",
        "last_sensitivity_H",
        "last_sensitivity_H_p7",
        "last_sensitivity_H_eps",
        "last_sensitivity_contact_mask",
        "_last_tangent_key",
        "_last_pose7_key",
    ):
        if hasattr(model, name):
            if name == "last_jacobian_diag":
                setattr(model, name, {})
            else:
                setattr(model, name, None)


def _solve_checked(model, p7: np.ndarray, *, label: str):
    started = time.perf_counter()
    result = model.solve(np.asarray(p7, float), commit=True)
    elapsed_s = time.perf_counter() - started
    if not result.success:
        message = result.info.get("message", "unknown solver failure")
        raise RuntimeError(f"{label} forward solve failed: {message}")
    return result, elapsed_s


def _implicit_jacobian(
    model,
    p7: np.ndarray,
    *,
    mode: str,
    eps_theta: float,
    eps_hess: float,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], float]:
    _, solve_time_s = _solve_checked(model, p7, label=label)
    started = time.perf_counter()
    J = np.asarray(
        model.jacobian_tip_actuation_tangent(
            p7,
            solve_if_needed=False,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
            mode=mode,
            reuse_cached=False,
        ),
        float,
    )
    jacobian_time_s = time.perf_counter() - started
    if J.shape != (3, 7):
        raise ValueError(f"{label} Jacobian has shape {J.shape}; expected (3, 7).")
    if not np.all(np.isfinite(J)):
        raise FloatingPointError(f"{label} Jacobian contains non-finite values.")

    diag = {}
    if hasattr(model, "get_last_jacobian_diag"):
        diag.update(model.get_last_jacobian_diag())
    diag.update(
        {
            "requested_mode": mode,
            "solve_time_s": solve_time_s,
            "jacobian_time_s": jacobian_time_s,
        }
    )
    return J, diag, solve_time_s + jacobian_time_s


def _perturb_tangent_pose(
    p7: np.ndarray,
    column: int,
    delta: float,
) -> np.ndarray:
    """
    Apply a perturbation in the tangent coordinates returned by
    ``jacobian_tip_actuation_tangent``.

    Rotation columns are left/world-frame increments:

        R_perturbed = Exp(delta_phi_world) @ R_source.
    """
    p7_new = np.asarray(p7, float).reshape(7).copy()
    if column < 3:
        p7_new[column] += float(delta)
    elif column < 6:
        axis = np.zeros(3)
        axis[column - 3] = float(delta)
        R_source = Rot.from_rotvec(p7_new[3:6])
        p7_new[3:6] = (Rot.from_rotvec(axis) * R_source).as_rotvec()
    elif column == 6:
        p7_new[6] += float(delta)
    else:
        raise IndexError(f"Jacobian column must be in [0, 6], got {column}.")
    return p7_new


def _apply_tangent_delta(
    p7: np.ndarray,
    delta_theta_world: np.ndarray,
) -> np.ndarray:
    """
    Apply ``[dr_world, dphi_world, dL]`` to a source pose.

    This is the same first-order coordinate convention used by
    ``jacobian_tip_actuation_tangent``. Rotation is applied on the left:

        R_next = Exp(dphi_world) @ R_source.
    """
    p7 = np.asarray(p7, float).reshape(7)
    delta = np.asarray(delta_theta_world, float).reshape(7)

    p7_next = p7.copy()
    p7_next[0:3] += delta[0:3]
    p7_next[3:6] = (
        Rot.from_rotvec(delta[3:6])
        * Rot.from_rotvec(p7[3:6])
    ).as_rotvec()
    p7_next[6] += float(delta[6])
    return p7_next


def _controller_command_to_tangent_delta(
    p7: np.ndarray,
    command: np.ndarray,
    *,
    dt: float,
    angular_velocity_frame: str,
) -> np.ndarray:
    """
    Convert a controller command to the beam Jacobian's tangent increment.

    ``command`` is ordered ``[v_world, omega_body_or_world, Ldot]``.
    """
    command = np.asarray(command, float).reshape(7)
    delta = float(dt) * command
    if angular_velocity_frame == "body":
        R_source = Rot.from_rotvec(np.asarray(p7, float)[3:6]).as_matrix()
        delta[3:6] = float(dt) * (R_source @ command[3:6])
    elif angular_velocity_frame != "world":
        raise ValueError(
            "angular_velocity_frame must be 'body' or 'world'."
        )
    return delta


def _prediction_metrics(
    actual_motion: np.ndarray,
    predicted_motion: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual_motion, float).reshape(3)
    predicted = np.asarray(predicted_motion, float).reshape(3)
    error = predicted - actual

    actual_norm = float(np.linalg.norm(actual))
    predicted_norm = float(np.linalg.norm(predicted))
    error_norm = float(np.linalg.norm(error))
    return {
        "actual_motion_mm": 1e3 * actual_norm,
        "predicted_motion_mm": 1e3 * predicted_norm,
        "absolute_error_mm": 1e3 * error_norm,
        "relative_error": error_norm / max(actual_norm, 1e-12),
        "direction_cosine": float(
            np.dot(actual, predicted)
            / max(actual_norm * predicted_norm, 1e-12)
        ),
        "error_x_mm": 1e3 * float(error[0]),
        "error_y_mm": 1e3 * float(error[1]),
        "error_z_mm": 1e3 * float(error[2]),
    }


def _evaluate_fixed_command_prediction(
    model,
    p7: np.ndarray,
    jacobians: dict[str, np.ndarray],
    *,
    command: np.ndarray,
    dt: float,
    scales: tuple[float, ...],
    angular_velocity_frame: str,
) -> tuple[list[dict[str, Any]], float]:
    """
    Compare all Jacobians using the same command and nonlinear contact plant.

    For every scale:

        delta_theta = scale * dt * command
        delta_tip_pred = J @ delta_theta
        delta_tip_actual = f_contact(p7 + delta_theta) - f_contact(p7)
    """
    nominal, nominal_time_s = _solve_checked(
        model,
        p7,
        label="prediction-test nominal contact plant",
    )
    nominal_cache = model.get_cache_copy()
    base_delta = _controller_command_to_tangent_delta(
        p7,
        command,
        dt=dt,
        angular_velocity_frame=angular_velocity_frame,
    )

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for scale in scales:
            delta = float(scale) * base_delta
            p7_next = _apply_tangent_delta(p7, delta)

            model.set_cache(nominal_cache)
            result_next = model.solve(
                p7_next,
                commit=True,
                reuse_cache=True,
            )
            if not result_next.success:
                message = result_next.info.get(
                    "message",
                    "unknown solver failure",
                )
                raise RuntimeError(
                    "Prediction-test nonlinear solve failed at "
                    f"scale={scale:g}: {message}"
                )

            actual_motion = result_next.tip - nominal.tip
            next_contact = _contact_state(result_next, model)
            for model_name, J in jacobians.items():
                predicted_motion = np.asarray(J, float) @ delta
                records.append(
                    {
                        "scale": float(scale),
                        "model": model_name,
                        "p7_next": p7_next.copy(),
                        "delta_theta_world": delta.copy(),
                        "actual_motion": actual_motion.copy(),
                        "predicted_motion": predicted_motion.copy(),
                        "next_gap_min_m": next_contact["gap_min_m"],
                        "next_penetrating_nodes": next_contact[
                            "penetrating_nodes"
                        ],
                        **_prediction_metrics(
                            actual_motion,
                            predicted_motion,
                        ),
                    }
                )
    finally:
        model.set_cache(nominal_cache)

    elapsed_s = time.perf_counter() - started + nominal_time_s
    return records, elapsed_s


def _plot_prediction_at_pose(
    records: list[dict[str, Any]],
    *,
    probe: ProbePose,
    output_path: Path,
    show: bool,
) -> None:
    if not records:
        return

    model_names = list(dict.fromkeys(record["model"] for record in records))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )

    for model_name in model_names:
        selected = [
            record for record in records if record["model"] == model_name
        ]
        scales = [record["scale"] for record in selected]
        absolute = [record["absolute_error_mm"] for record in selected]
        relative = [100.0 * record["relative_error"] for record in selected]
        axes[0].plot(scales, absolute, marker="o", label=model_name)
        axes[1].plot(scales, relative, marker="o", label=model_name)

    axes[0].set_ylabel("One-step tip prediction error [mm]")
    axes[1].set_ylabel("Error / actual tip motion [%]")
    for ax in axes:
        ax.set_xlabel("Command scale")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Absolute prediction error")
    axes[1].set_title("Relative prediction error")
    fig.suptitle(f"{probe.name}: fixed-command contact-plant prediction")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_prediction_across_locations(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    show: bool,
) -> None:
    """Plot full-scale fixed-command prediction errors at every probe pose."""
    if not records:
        return

    full_scale = max(float(record["scale"]) for record in records)
    selected = [
        record
        for record in records
        if np.isclose(float(record["scale"]), full_scale)
    ]
    probe_names = list(
        dict.fromkeys(str(record["probe_name"]) for record in selected)
    )
    model_names = list(dict.fromkeys(record["model"] for record in selected))
    positions = np.arange(len(probe_names))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )
    for model_name in model_names:
        by_probe = {
            str(record["probe_name"]): record
            for record in selected
            if record["model"] == model_name
        }
        absolute = [
            _as_float(by_probe.get(name, {}).get("absolute_error_mm"))
            for name in probe_names
        ]
        relative = [
            100.0
            * _as_float(by_probe.get(name, {}).get("relative_error"))
            for name in probe_names
        ]
        axes[0].plot(positions, absolute, marker="o", label=model_name)
        axes[1].plot(positions, relative, marker="o", label=model_name)

    for ax in axes:
        ax.set_xticks(positions, probe_names, rotation=30, ha="right")
        ax.set_xlabel("Magnet probe location")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Tip prediction error [mm]")
    axes[1].set_ylabel("Error / actual tip motion [%]")
    axes[0].set_title("Absolute error")
    axes[1].set_title("Relative error")
    fig.suptitle(
        f"Fixed-command prediction across locations (scale={full_scale:g})"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _finite_difference_contact_jacobian(
    model,
    p7: np.ndarray,
    *,
    columns: tuple[int, ...],
    translation_step_m: float,
    rotation_step_rad: float,
    length_step_m: float,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """
    Central finite-difference reference of the complete nonlinear contact plant.

    Every perturbed solve begins from the same nominal equilibrium cache. This
    avoids plus/minus derivatives following different warm-start histories.
    Unchecked columns are stored as NaN.
    """
    nominal, nominal_solve_time_s = _solve_checked(
        model,
        p7,
        label="finite-difference nominal contact plant",
    )
    nominal_cache = model.get_cache_copy()
    J = np.full((3, 7), np.nan, dtype=float)
    column_times: dict[str, float] = {}

    steps = np.array(
        [
            translation_step_m,
            translation_step_m,
            translation_step_m,
            rotation_step_rad,
            rotation_step_rad,
            rotation_step_rad,
            length_step_m,
        ],
        dtype=float,
    )

    started_total = time.perf_counter()
    try:
        for column in columns:
            h = float(steps[column])
            started_column = time.perf_counter()

            tips = []
            for sign in (+1.0, -1.0):
                model.set_cache(nominal_cache)
                p7_perturbed = _perturb_tangent_pose(
                    p7,
                    column,
                    sign * h,
                )
                result = model.solve(
                    p7_perturbed,
                    commit=True,
                    reuse_cache=True,
                )
                if not result.success:
                    message = result.info.get("message", "unknown solver failure")
                    raise RuntimeError(
                        "Finite-difference solve failed for "
                        f"{CONTROL_NAMES[column]} ({sign:+.0f}h): {message}"
                    )
                tips.append(result.tip.copy())

            J[:, column] = (tips[0] - tips[1]) / (2.0 * h)
            column_times[CONTROL_NAMES[column]] = (
                time.perf_counter() - started_column
            )
    finally:
        model.set_cache(nominal_cache)

    elapsed_s = time.perf_counter() - started_total + nominal_solve_time_s
    diag = {
        "requested_mode": "nonlinear_central_finite_difference",
        "checked_columns": [CONTROL_NAMES[j] for j in columns],
        "translation_step_m": translation_step_m,
        "rotation_step_rad": rotation_step_rad,
        "length_step_m": length_step_m,
        "nominal_tip": nominal.tip.copy(),
        "column_times_s": column_times,
        "jacobian_time_s": elapsed_s,
    }
    return J, diag, elapsed_s


def _matrix_metrics(
    J_test: np.ndarray,
    J_reference: np.ndarray,
    *,
    columns: tuple[int, ...],
) -> dict[str, Any]:
    test = np.asarray(J_test, float)[:, columns]
    reference = np.asarray(J_reference, float)[:, columns]
    difference = test - reference

    test_norm = float(np.linalg.norm(test))
    reference_norm = float(np.linalg.norm(reference))
    abs_error = float(np.linalg.norm(difference))
    relative_error = abs_error / max(reference_norm, 1e-12)

    column_abs = np.linalg.norm(difference, axis=0)
    column_reference = np.linalg.norm(reference, axis=0)
    column_relative = column_abs / np.maximum(column_reference, 1e-12)

    cosine = float(
        np.sum(test * reference)
        / max(test_norm * reference_norm, 1e-12)
    )
    return {
        "abs_error": abs_error,
        "relative_error": relative_error,
        "max_abs_element_error": float(np.max(np.abs(difference))),
        "cosine": cosine,
        "column_abs_error": column_abs,
        "column_relative_error": column_relative,
    }


def _contact_state(result, model) -> dict[str, Any]:
    parts = result.parts
    gaps = np.asarray(parts.get("gap_nodes", []), float).reshape(-1)
    forces = np.asarray(parts.get("F_nodes", []), float)
    smooth_eps = np.nan
    params = getattr(getattr(model, "contact_cfg", None), "params", None)
    if params is not None:
        smooth_eps = _as_float(getattr(params, "smooth_eps", np.nan))

    return {
        "gap_min_m": (
            float(np.min(gaps))
            if gaps.size
            else _as_float(parts.get("gap_min", np.nan))
        ),
        "penetrating_nodes": int(np.sum(gaps < 0.0)) if gaps.size else -1,
        "smoothing_band_nodes": (
            int(np.sum(gaps <= 3.0 * smooth_eps))
            if gaps.size and np.isfinite(smooth_eps)
            else -1
        ),
        "contact_energy": _as_float(parts.get("W_cf", np.nan)),
        "contact_force_norm": (
            float(np.linalg.norm(forces)) if forces.size else np.nan
        ),
    }


def _column_scales(args: argparse.Namespace) -> np.ndarray:
    return np.array(
        [
            args.display_translation_mm * 1e-3,
            args.display_translation_mm * 1e-3,
            args.display_translation_mm * 1e-3,
            np.deg2rad(args.display_rotation_deg),
            np.deg2rad(args.display_rotation_deg),
            np.deg2rad(args.display_rotation_deg),
            args.display_insertion_mm * 1e-3,
        ],
        dtype=float,
    )


def _scaled_jacobian_mm(
    J: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return 1e3 * np.asarray(J, float) * _column_scales(args)[None, :]


def _plot_jacobian_matrices(
    matrices: dict[str, np.ndarray],
    *,
    probe: ProbePose,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    names = list(matrices)
    n_panels = len(names)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    scaled = {
        name: _scaled_jacobian_mm(matrix, args)
        for name, matrix in matrices.items()
    }
    finite_values = np.concatenate(
        [
            values[np.isfinite(values)].reshape(-1)
            for values in scaled.values()
            if np.any(np.isfinite(values))
        ]
    )
    vmax = max(float(np.max(np.abs(finite_values))), 1e-12)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 3.7 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    for panel, name in enumerate(names):
        row, col = divmod(panel, n_cols)
        ax = axes[row, col]
        values = scaled[name]
        image = ax.imshow(
            values,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(name)
        ax.set_xticks(range(7), CONTROL_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(3), TIP_NAMES)

        for i in range(3):
            for j in range(7):
                value = values[i, j]
                label = "—" if not np.isfinite(value) else f"{value:.3g}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    for panel in range(n_panels, n_rows * n_cols):
        row, col = divmod(panel, n_cols)
        axes[row, col].axis("off")

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.84,
            label="Predicted tip displacement [mm]",
        )
    fig.suptitle(
        f"{probe.name}: Jacobian response to "
        f"{args.display_translation_mm:g} mm translation, "
        f"{args.display_rotation_deg:g}° rotation, and "
        f"{args.display_insertion_mm:g} mm insertion",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


def _plot_summary(
    records: list[dict[str, Any]],
    matrices_by_model: dict[str, list[np.ndarray]],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not records:
        return

    positions = np.arange(len(records))
    labels = [str(record["name"]) for record in records]
    J_accurate = np.stack(matrices_by_model["contact_accurate"], axis=0)
    scales = _column_scales(args)
    authority_mm = 1e3 * np.linalg.norm(
        J_accurate * scales[None, None, :],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for column, control_name in enumerate(CONTROL_NAMES):
        ax.plot(
            positions,
            authority_mm[:, column],
            marker="o",
            label=control_name,
        )
    ax.set_xticks(positions, labels, rotation=30, ha="right")
    ax.set_ylabel("Accurate contact-Jacobian authority [mm]")
    ax.set_xlabel("Magnet probe location")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    ax.set_title("Control authority of the accurate contact Jacobian")
    fig.savefig(
        output_dir / "jacobian_authority_across_locations.png",
        dpi=180,
        bbox_inches="tight",
    )
    if args.show:
        plt.show()
    plt.close(fig)

    error_keys = [
        key
        for key in (
            "accurate_relative_error",
            "fast_relative_error",
            "fast_fresh_relative_error",
            "no_contact_relative_error",
        )
        if any(np.isfinite(_as_float(record.get(key))) for record in records)
    ]
    if error_keys:
        labels_by_key = {
            "accurate_relative_error": "accurate contact",
            "fast_relative_error": "fast contact (history)",
            "fast_fresh_relative_error": "fast contact (fresh H)",
            "no_contact_relative_error": "accurate no-contact",
        }
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        for key in error_keys:
            values = 100.0 * np.array(
                [_as_float(record.get(key)) for record in records],
                dtype=float,
            )
            ax.plot(
                positions,
                values,
                marker="o",
                label=labels_by_key[key],
            )
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_ylabel("Frobenius error relative to reference [%]")
        ax.set_xlabel("Magnet probe location")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(
            "Jacobian error against "
            + (
                "nonlinear contact finite differences"
                if args.finite_difference
                else "accurate contact Jacobian"
            )
        )
        fig.savefig(
            output_dir / "jacobian_error_across_locations.png",
            dpi=180,
            bbox_inches="tight",
        )
        if args.show:
            plt.show()
        plt.close(fig)


def _read_pose_csv(path: Path) -> list[ProbePose]:
    poses: list[ProbePose] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in POSE_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing columns {missing}; required columns are "
                f"{list(POSE_FIELDS)}. An optional 'name' column is allowed."
            )
        for index, row in enumerate(reader):
            p7 = np.array([float(row[field]) for field in POSE_FIELDS], dtype=float)
            name = row.get("name") or f"pose_{index:02d}"
            poses.append(
                ProbePose(
                    name=str(name),
                    p7=p7,
                    offset=float(index),
                    offset_unit="CSV row",
                )
            )
    if not poses:
        raise ValueError(f"{path} contains no poses.")
    return poses


def _make_sweep_poses(
    base_p7: np.ndarray,
    *,
    axis: str,
    offsets: list[float],
) -> list[ProbePose]:
    poses = []
    index = AXIS_INDEX[axis]
    is_rotation = axis in {"rx", "ry", "rz"}
    unit = "deg" if is_rotation else "mm"

    for offset in offsets:
        p7 = np.asarray(base_p7, float).reshape(7).copy()
        p7[index] += (
            np.deg2rad(float(offset))
            if is_rotation
            else 1e-3 * float(offset)
        )
        poses.append(
            ProbePose(
                name=f"{axis}_{float(offset):+.3f}_{unit}",
                p7=p7,
                offset=float(offset),
                offset_unit=unit,
            )
        )
    return poses


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = []
    keys: list[str] = []
    for row in rows:
        scalar_row = {
            key: value
            for key, value in row.items()
            if np.isscalar(value) or value is None or isinstance(value, str)
        }
        scalar_rows.append(scalar_row)
        for key in scalar_row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scalar_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the source magnet, contact-aware beam, and lumen at several "
            "poses, then compare accurate, fast, and optional nonlinear "
            "finite-difference tip Jacobians at those identical poses."
        )
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_INDEX),
        default="y",
        help=(
            "Pose component swept around make_run_plot_p7(). Translation and "
            "length offsets use millimetres; rotation offsets use degrees."
        ),
    )
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=(-2.0, -1.0, 0.0, 1.0, 2.0),
        help="Offsets from the shared run_plot pose.",
    )
    parser.add_argument(
        "--poses-csv",
        type=Path,
        help=(
            "Use exact poses from CSV instead of a sweep. Required columns: "
            "x,y,z,rx,ry,rz,L; optional column: name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("jacobian_location_analysis"),
    )
    parser.add_argument(
        "--finite-difference",
        action="store_true",
        help=(
            "Compute the expensive central finite-difference Jacobian of the "
            "complete nonlinear contact plant."
        ),
    )
    parser.add_argument(
        "--fd-columns",
        nargs="+",
        choices=CONTROL_NAMES,
        default=None,
        help="Finite-difference only these columns. Default: all seven.",
    )
    parser.add_argument(
        "--skip-no-contact",
        action="store_true",
        help="Do not calculate the accurate no-contact comparison.",
    )
    parser.add_argument(
        "--prediction-u",
        type=float,
        nargs=7,
        metavar=("VX", "VY", "VZ", "WX", "WY", "WZ", "LDOT"),
        help=(
            "Enable a fixed-command prediction test. Supply one controller "
            "command [v_world (m/s), omega (rad/s), Ldot (m/s)]. Every "
            "Jacobian predicts this same command and the nonlinear contact "
            "plant supplies the ground truth."
        ),
    )
    parser.add_argument(
        "--prediction-dt",
        type=float,
        default=0.01,
        help="Controller sample period used with --prediction-u.",
    )
    parser.add_argument(
        "--prediction-scales",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5, 1.0),
        help=(
            "Multipliers applied to dt*u. Several scales reveal the local "
            "linear validity region."
        ),
    )
    parser.add_argument(
        "--prediction-angular-frame",
        choices=("body", "world"),
        default="body",
        help=(
            "Frame of the angular components supplied to --prediction-u. "
            "Use 'body' for the controller convention described in this model."
        ),
    )
    parser.add_argument("--eps-theta", type=float, default=1e-6)
    parser.add_argument("--eps-hess", type=float, default=1e-4)
    parser.add_argument(
        "--fd-translation-step-mm",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--fd-rotation-step-deg",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--fd-length-step-mm",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--display-translation-mm",
        type=float,
        default=1.0,
        help="Translation used to scale Jacobian heatmaps.",
    )
    parser.add_argument(
        "--display-rotation-deg",
        type=float,
        default=1.0,
        help="Rotation used to scale Jacobian heatmaps.",
    )
    parser.add_argument(
        "--display-insertion-mm",
        type=float,
        default=1.0,
        help="Insertion change used to scale Jacobian heatmaps.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively as well as saving them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first invalid pose or failed equilibrium.",
    )
    args = parser.parse_args()

    for name in (
        "eps_theta",
        "eps_hess",
        "fd_translation_step_mm",
        "fd_rotation_step_deg",
        "fd_length_step_mm",
        "display_translation_mm",
        "display_rotation_deg",
        "display_insertion_mm",
        "prediction_dt",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if any(float(scale) <= 0.0 for scale in args.prediction_scales):
        parser.error("--prediction-scales values must all be positive.")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_p7 = np.asarray(_smoke.make_run_plot_p7(), float).reshape(7)
    poses = (
        _read_pose_csv(args.poses_csv)
        if args.poses_csv is not None
        else _make_sweep_poses(
            base_p7,
            axis=args.axis,
            offsets=list(args.offsets),
        )
    )

    fd_columns = tuple(
        range(7)
        if args.fd_columns is None
        else (CONTROL_NAMES.index(name) for name in args.fd_columns)
    )
    comparison_columns = fd_columns if args.finite_difference else tuple(range(7))

    # Separate instances prevent accurate-mode Hessians leaking into fast mode.
    scene_model = _smoke.build_run_plot_model(contact_enabled=True)
    accurate_model = _smoke.build_run_plot_model(contact_enabled=True)
    fast_model = _smoke.build_run_plot_model(contact_enabled=True)
    fd_model = (
        _smoke.build_run_plot_model(contact_enabled=True)
        if args.finite_difference
        else None
    )
    no_contact_model = (
        None
        if args.skip_no_contact
        else _smoke.build_run_plot_model(contact_enabled=False)
    )
    prediction_model = (
        _smoke.build_run_plot_model(contact_enabled=True)
        if args.prediction_u is not None
        else None
    )

    records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    matrices_by_model: dict[str, list[np.ndarray]] = {
        "contact_accurate": [],
        "contact_fast_history": [],
        "contact_fast_fresh": [],
    }
    if fd_model is not None:
        matrices_by_model["contact_nonlinear_fd"] = []
    if no_contact_model is not None:
        matrices_by_model["no_contact_accurate"] = []
    successful_p7: list[np.ndarray] = []
    successful_tips: list[np.ndarray] = []
    failures: list[dict[str, str]] = []

    print(f"Package under test: {PACKAGE_NAME}")
    print("Base run_plot p7:", base_p7)
    print(f"Probe count: {len(poses)}")
    print(f"Output directory: {output_dir}")
    if args.finite_difference:
        print(
            "Finite-difference columns:",
            [CONTROL_NAMES[index] for index in fd_columns],
        )
        print(
            "Warning: nonlinear finite differences require two extra forward "
            "solves per selected column and may take several minutes."
        )
    if args.prediction_u is not None:
        print("Prediction-test command u:", list(args.prediction_u))
        print(
            f"Prediction dt={args.prediction_dt:g} s, "
            f"angular frame={args.prediction_angular_frame}, "
            f"scales={list(args.prediction_scales)}"
        )

    for pose_index, probe in enumerate(poses):
        print("\n" + "=" * 88)
        print(f"[{pose_index + 1}/{len(poses)}] {probe.name}")
        print("p7:", probe.p7)

        try:
            scene_result, scene_solve_time_s = _solve_checked(
                scene_model,
                probe.p7,
                label="scene contact model",
            )
            contact_state = _contact_state(scene_result, scene_model)

            scene_path = output_dir / (
                f"{pose_index:02d}_{_slug(probe.name)}_scene.png"
            )
            fig, _ = scene_model.plot_solution(
                probe.p7,
                result=scene_result,
                show=args.show,
                save_path=scene_path,
                title=(
                    f"{probe.name} | gap="
                    f"{1e3 * contact_state['gap_min_m']:.3f} mm"
                ),
            )
            plt.close(fig)

            J_accurate, diag_accurate, accurate_time_s = _implicit_jacobian(
                accurate_model,
                probe.p7,
                mode="accurate",
                eps_theta=args.eps_theta,
                eps_hess=args.eps_hess,
                label="accurate contact model",
            )
            J_fast, diag_fast, fast_time_s = _implicit_jacobian(
                fast_model,
                probe.p7,
                mode="fast",
                eps_theta=args.eps_theta,
                eps_hess=args.eps_hess,
                label="persistent fast contact model",
            )

            # This model inherits the exact nominal scene equilibrium but no
            # previous sensitivity Hessian.
            fast_fresh_model = copy.deepcopy(scene_model)
            _clear_sensitivity_history(fast_fresh_model)
            J_fast_fresh, diag_fast_fresh, fast_fresh_time_s = (
                _implicit_jacobian(
                    fast_fresh_model,
                    probe.p7,
                    mode="fast",
                    eps_theta=args.eps_theta,
                    eps_hess=args.eps_hess,
                    label="fresh fast contact model",
                )
            )

            matrices = {
                "contact accurate": J_accurate,
                "contact fast (history)": J_fast,
                "contact fast (fresh H)": J_fast_fresh,
            }

            J_fd = None
            diag_fd: dict[str, Any] = {}
            fd_time_s = np.nan
            if fd_model is not None:
                J_fd, diag_fd, fd_time_s = (
                    _finite_difference_contact_jacobian(
                        fd_model,
                        probe.p7,
                        columns=fd_columns,
                        translation_step_m=(
                            args.fd_translation_step_mm * 1e-3
                        ),
                        rotation_step_rad=np.deg2rad(
                            args.fd_rotation_step_deg
                        ),
                        length_step_m=args.fd_length_step_mm * 1e-3,
                    )
                )
                matrices = {
                    "nonlinear contact FD": J_fd,
                    **matrices,
                }

            J_no_contact = None
            diag_no_contact: dict[str, Any] = {}
            no_contact_time_s = np.nan
            if no_contact_model is not None:
                (
                    J_no_contact,
                    diag_no_contact,
                    no_contact_time_s,
                ) = _implicit_jacobian(
                    no_contact_model,
                    probe.p7,
                    mode="accurate",
                    eps_theta=args.eps_theta,
                    eps_hess=args.eps_hess,
                    label="accurate no-contact model",
                )
                matrices["accurate no-contact"] = J_no_contact

            reference = J_fd if J_fd is not None else J_accurate
            reference_name = (
                "nonlinear contact finite difference"
                if J_fd is not None
                else "accurate contact implicit Jacobian"
            )

            metrics_accurate = (
                _matrix_metrics(
                    J_accurate,
                    reference,
                    columns=comparison_columns,
                )
                if J_fd is not None
                else None
            )
            metrics_fast = _matrix_metrics(
                J_fast,
                reference,
                columns=comparison_columns,
            )
            metrics_fast_fresh = _matrix_metrics(
                J_fast_fresh,
                reference,
                columns=comparison_columns,
            )
            metrics_no_contact = (
                _matrix_metrics(
                    J_no_contact,
                    reference,
                    columns=comparison_columns,
                )
                if J_no_contact is not None
                else None
            )

            pose_prediction_records: list[dict[str, Any]] = []
            prediction_time_s = np.nan
            if prediction_model is not None:
                prediction_jacobians = {
                    "contact accurate": J_accurate,
                    "contact fast (history)": J_fast,
                    "contact fast (fresh H)": J_fast_fresh,
                }
                if J_no_contact is not None:
                    prediction_jacobians[
                        "accurate no-contact"
                    ] = J_no_contact
                if J_fd is not None and np.all(np.isfinite(J_fd)):
                    prediction_jacobians[
                        "nonlinear FD Jacobian"
                    ] = J_fd

                (
                    pose_prediction_records,
                    prediction_time_s,
                ) = _evaluate_fixed_command_prediction(
                    prediction_model,
                    probe.p7,
                    prediction_jacobians,
                    command=np.asarray(args.prediction_u, float),
                    dt=args.prediction_dt,
                    scales=tuple(float(x) for x in args.prediction_scales),
                    angular_velocity_frame=args.prediction_angular_frame,
                )
                for prediction_record in pose_prediction_records:
                    prediction_record["probe_index"] = pose_index
                    prediction_record["probe_name"] = probe.name

                prediction_plot_path = output_dir / (
                    f"{pose_index:02d}_{_slug(probe.name)}_prediction.png"
                )
                _plot_prediction_at_pose(
                    pose_prediction_records,
                    probe=probe,
                    output_path=prediction_plot_path,
                    show=args.show,
                )
                prediction_records.extend(pose_prediction_records)

            heatmap_path = output_dir / (
                f"{pose_index:02d}_{_slug(probe.name)}_jacobians.png"
            )
            _plot_jacobian_matrices(
                matrices,
                probe=probe,
                output_path=heatmap_path,
                args=args,
            )

            print("tip [m]:", scene_result.tip)
            print(
                "contact: "
                f"gap={1e3 * contact_state['gap_min_m']:+.6f} mm, "
                f"penetrating_nodes={contact_state['penetrating_nodes']}, "
                f"force_norm={contact_state['contact_force_norm']:.6e}"
            )
            print(
                "accurate: "
                f"scheme={diag_accurate.get('difference_scheme')}, "
                f"H_reused={diag_accurate.get('hessian_reused')}, "
                f"cond(H)={_as_float(diag_accurate.get('cond_H_beam')):.3e}, "
                f"time={accurate_time_s:.3f} s"
            )
            print(
                "fast history: "
                f"scheme={diag_fast.get('difference_scheme')}, "
                f"H_reused={diag_fast.get('hessian_reused')}, "
                f"cond(H)={_as_float(diag_fast.get('cond_H_beam')):.3e}, "
                f"time={fast_time_s:.3f} s, "
                f"error={100.0 * metrics_fast['relative_error']:.3f}%"
            )
            print(
                "fast fresh:   "
                f"scheme={diag_fast_fresh.get('difference_scheme')}, "
                f"H_reused={diag_fast_fresh.get('hessian_reused')}, "
                f"cond(H)="
                f"{_as_float(diag_fast_fresh.get('cond_H_beam')):.3e}, "
                f"time={fast_fresh_time_s:.3f} s, "
                f"error={100.0 * metrics_fast_fresh['relative_error']:.3f}%"
            )
            if metrics_accurate is not None:
                print(
                    "accurate vs nonlinear FD: "
                    f"error={100.0 * metrics_accurate['relative_error']:.3f}%"
                )
            if metrics_no_contact is not None:
                print(
                    "no-contact vs reference: "
                    f"error={100.0 * metrics_no_contact['relative_error']:.3f}%"
                )
            if pose_prediction_records:
                full_scale = max(
                    float(item["scale"])
                    for item in pose_prediction_records
                )
                print(
                    f"fixed-command prediction at scale={full_scale:g}:"
                )
                for item in pose_prediction_records:
                    if not np.isclose(float(item["scale"]), full_scale):
                        continue
                    print(
                        f"  {item['model']}: "
                        f"actual={item['actual_motion_mm']:.6f} mm, "
                        f"error={item['absolute_error_mm']:.6f} mm "
                        f"({100.0 * item['relative_error']:.3f}%), "
                        f"cos={item['direction_cosine']:.6f}"
                    )
            print("comparison reference:", reference_name)
            print("scene:", scene_path)
            print("Jacobian heatmap:", heatmap_path)

            row = {
                "index": pose_index,
                "name": probe.name,
                "offset": probe.offset,
                "offset_unit": probe.offset_unit,
                **{
                    f"p7_{name}": float(probe.p7[index])
                    for index, name in enumerate(POSE_FIELDS)
                },
                **{
                    f"tip_{axis}": float(scene_result.tip[index])
                    for index, axis in enumerate(("x", "y", "z"))
                },
                **contact_state,
                "scene_solve_time_s": scene_solve_time_s,
                "accurate_time_s": accurate_time_s,
                "fast_time_s": fast_time_s,
                "fast_fresh_time_s": fast_fresh_time_s,
                "fd_time_s": fd_time_s,
                "no_contact_time_s": no_contact_time_s,
                "prediction_time_s": prediction_time_s,
                "accurate_cond_H": _as_float(
                    diag_accurate.get("cond_H_beam")
                ),
                "fast_cond_H": _as_float(diag_fast.get("cond_H_beam")),
                "fast_fresh_cond_H": _as_float(
                    diag_fast_fresh.get("cond_H_beam")
                ),
                "accurate_hessian_reused": diag_accurate.get(
                    "hessian_reused"
                ),
                "fast_hessian_reused": diag_fast.get("hessian_reused"),
                "fast_fresh_hessian_reused": diag_fast_fresh.get(
                    "hessian_reused"
                ),
                "accurate_relative_error": (
                    np.nan
                    if metrics_accurate is None
                    else metrics_accurate["relative_error"]
                ),
                "fast_relative_error": metrics_fast["relative_error"],
                "fast_fresh_relative_error": (
                    metrics_fast_fresh["relative_error"]
                ),
                "no_contact_relative_error": (
                    np.nan
                    if metrics_no_contact is None
                    else metrics_no_contact["relative_error"]
                ),
                "comparison_reference": reference_name,
            }
            if pose_prediction_records:
                full_scale = max(
                    float(item["scale"])
                    for item in pose_prediction_records
                )
                for item in pose_prediction_records:
                    if not np.isclose(float(item["scale"]), full_scale):
                        continue
                    prefix = _slug(str(item["model"])).replace("-", "_")
                    row[f"prediction_{prefix}_absolute_error_mm"] = item[
                        "absolute_error_mm"
                    ]
                    row[f"prediction_{prefix}_relative_error"] = item[
                        "relative_error"
                    ]
            for column, control_name in enumerate(CONTROL_NAMES):
                row[f"accurate_authority_{control_name}"] = float(
                    np.linalg.norm(
                        _scaled_jacobian_mm(J_accurate, args)[:, column]
                    )
                )
                row[f"fast_col_rel_error_{control_name}"] = float(
                    metrics_fast["column_relative_error"][
                        comparison_columns.index(column)
                    ]
                    if column in comparison_columns
                    else np.nan
                )

            records.append(row)
            successful_p7.append(probe.p7.copy())
            successful_tips.append(scene_result.tip.copy())
            matrices_by_model["contact_accurate"].append(J_accurate)
            matrices_by_model["contact_fast_history"].append(J_fast)
            matrices_by_model["contact_fast_fresh"].append(J_fast_fresh)
            if J_fd is not None:
                matrices_by_model["contact_nonlinear_fd"].append(J_fd)
            if J_no_contact is not None:
                matrices_by_model["no_contact_accurate"].append(J_no_contact)

            diagnostic_records.append(
                {
                    "index": pose_index,
                    "name": probe.name,
                    "p7": probe.p7,
                    "contact_state": contact_state,
                    "accurate": {
                        "diag": diag_accurate,
                        "metrics": metrics_accurate,
                    },
                    "fast_history": {
                        "diag": diag_fast,
                        "metrics": metrics_fast,
                    },
                    "fast_fresh": {
                        "diag": diag_fast_fresh,
                        "metrics": metrics_fast_fresh,
                    },
                    "finite_difference": {
                        "diag": diag_fd,
                    },
                    "no_contact": {
                        "diag": diag_no_contact,
                        "metrics": metrics_no_contact,
                    },
                    "fixed_command_prediction": {
                        "command": args.prediction_u,
                        "dt": args.prediction_dt,
                        "angular_velocity_frame": (
                            args.prediction_angular_frame
                        ),
                        "records": pose_prediction_records,
                    },
                }
            )

        except Exception as exc:
            failure = {
                "index": str(pose_index),
                "name": probe.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[FAILED] {failure['error']}")
            if args.fail_fast:
                raise

    if records:
        _write_summary_csv(output_dir / "jacobian_location_summary.csv", records)
        with (output_dir / "jacobian_diagnostics.json").open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(_jsonable(diagnostic_records), stream, indent=2)

        npz_payload: dict[str, np.ndarray] = {
            "p7": np.stack(successful_p7, axis=0),
            "tip": np.stack(successful_tips, axis=0),
            "control_names": np.asarray(CONTROL_NAMES),
            "probe_names": np.asarray([row["name"] for row in records]),
        }
        for model_name, values in matrices_by_model.items():
            if values:
                npz_payload[model_name] = np.stack(values, axis=0)
        np.savez_compressed(
            output_dir / "jacobian_matrices.npz",
            **npz_payload,
        )

        _plot_summary(
            records,
            matrices_by_model,
            output_dir=output_dir,
            args=args,
        )
        if prediction_records:
            _write_summary_csv(
                output_dir / "prediction_accuracy_summary.csv",
                prediction_records,
            )
            _plot_prediction_across_locations(
                prediction_records,
                output_path=(
                    output_dir
                    / "prediction_accuracy_across_locations.png"
                ),
                show=args.show,
            )

    if failures:
        with (output_dir / "failures.json").open("w", encoding="utf-8") as stream:
            json.dump(failures, stream, indent=2)

    print("\n" + "=" * 88)
    print(f"Completed {len(records)} of {len(poses)} probe poses.")
    print(f"Results: {output_dir}")
    if failures:
        print(f"Failures: {len(failures)} (see failures.json)")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
