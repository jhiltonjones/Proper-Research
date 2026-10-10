"""Shared helpers for the four-layer control-stack analysis scripts.

Every script in this package is *output-only*: it reads files that the planning
and simulation layers already wrote and never imports the robot/beam model, so
it can be run long after an experiment on a machine that cannot rebuild the
planning context.

Layer numbering used throughout::

    L1  offline inverse configuration    -> inverse_configuration_path.csv
    L2  global constrained smoothing     -> global_configuration_path.csv
    L3  time parameterisation            -> time_parameterized_configuration_path.csv
    L4  MPC tracking simulation          -> configuration_mpc_simulation.csv

The shared vocabulary is deliberately small:

``budget``
    A task error divided by its hard tolerance.  ``budget = 1`` means the
    solution sits exactly on the constraint boundary.  L1 minimises task error
    as a *cost*, so its budget is small; L2 treats the same quantity as a hard
    *constraint*, so its budget is free to rise to 1 at no objective penalty.
    Comparing budgets across layers is the single most informative thing this
    package does.

``jaggedness``
    Discrete second difference of a geometric path with respect to arc length,
    which is the quantity a time parameteriser and a tracking controller
    actually pay for.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

Array = np.ndarray

COORDINATE_NAMES: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5", "q6", "L")
COORDINATE_UNITS: tuple[str, ...] = ("rad", "rad", "rad", "rad", "rad", "rad", "m")

# Default scaling used when a saved configuration does not provide one.  It
# matches GlobalConfigurationOptimizerConfig.configuration_scale so that
# derivative numbers printed by different layers are directly comparable.
DEFAULT_CONFIGURATION_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-2])


# --------------------------------------------------------------------------
# tabular input
# --------------------------------------------------------------------------
@dataclass
class Table:
    """A CSV read into named float columns, with the raw strings kept."""

    path: Path
    header: list[str]
    columns: dict[str, Array]
    text_columns: dict[str, list[str]]
    row_count: int

    def has(self, *names: str) -> bool:
        return all(name in self.columns for name in names)

    def column(self, name: str, default: float | None = None) -> Array:
        if name in self.columns:
            return self.columns[name]
        if default is None:
            raise KeyError(
                f"{self.path.name} has no column {name!r}. "
                f"Available: {', '.join(sorted(self.columns))}"
            )
        return np.full(self.row_count, float(default))

    def first_present(self, names: Sequence[str]) -> str | None:
        for name in names:
            if name in self.columns:
                return name
        return None

    def vector(
        self,
        prefix: str,
        suffixes: Sequence[str] = ("x", "y", "z"),
        unit_suffixes: Sequence[str] = ("", "_m"),
    ) -> Array | None:
        """Stack ``prefix_x[, _m]`` style columns, tolerating a unit suffix."""
        for unit in unit_suffixes:
            keys = [f"{prefix}_{suffix}{unit}" for suffix in suffixes]
            if self.has(*keys):
                return np.stack([self.columns[key] for key in keys], axis=1)
        return None

    def state_matrix(self, joint_pattern: str, insertion: str) -> Array | None:
        """Stack ``[q1..q6, insertion]`` given naming patterns.

        ``joint_pattern`` uses ``{i}`` for the 1-based joint index, e.g.
        ``"q{i}_rad"`` (L1/L2) or ``"q{i}_actual"`` (L4).
        """
        keys = [joint_pattern.format(i=i) for i in range(1, 7)]
        if not self.has(*keys, insertion):
            return None
        return np.stack(
            [self.columns[key] for key in keys] + [self.columns[insertion]], axis=1
        )


def read_table(path: str | Path) -> Table:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError(f"{path} contains no data rows.")
    header = [cell.strip() for cell in rows[0]]
    body = rows[1:]
    count = len(body)
    columns: dict[str, Array] = {}
    text_columns: dict[str, list[str]] = {}
    for index, name in enumerate(header):
        raw = [row[index] if index < len(row) else "" for row in body]
        values = np.empty(count, dtype=float)
        numeric = True
        for i, cell in enumerate(raw):
            cell = cell.strip()
            if cell == "":
                values[i] = np.nan
                continue
            try:
                values[i] = float(cell)
            except ValueError:
                numeric = False
                break
        if numeric:
            columns[name] = values
        else:
            text_columns[name] = [cell.strip() for cell in raw]
    return Table(
        path=path,
        header=header,
        columns=columns,
        text_columns=text_columns,
        row_count=count,
    )


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_json_if_present(path: str | Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (FileNotFoundError, ValueError):
        return None


def find_one(directory: Path, *patterns: str) -> Path | None:
    """First existing file matching any glob pattern, searched in order."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


# --------------------------------------------------------------------------
# small statistics
# --------------------------------------------------------------------------
def finite(values: Any) -> Array:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def rms(values: Any) -> float:
    array = finite(values)
    return float(np.sqrt(np.mean(array * array))) if array.size else math.nan


def percentile(values: Any, q: float) -> float:
    array = finite(values)
    return float(np.percentile(array, q)) if array.size else math.nan


def maximum(values: Any) -> float:
    array = finite(values)
    return float(np.max(array)) if array.size else math.nan


def minimum(values: Any) -> float:
    array = finite(values)
    return float(np.min(array)) if array.size else math.nan


def max_abs(values: Any, axis: int | None = None) -> Any:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan if axis is None else np.full(array.shape[1] if array.ndim == 2 else 0, math.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.abs(array), axis=axis)


def improvement_percent(before: float, after: float) -> float:
    """Positive means ``after`` is better for a lower-is-better quantity."""
    if not np.isfinite(before) or not np.isfinite(after) or abs(before) <= 1e-30:
        return math.nan
    return 100.0 * (before - after) / abs(before)


def fraction(mask: Any) -> float:
    array = np.asarray(mask)
    return float(np.mean(array.astype(bool))) if array.size else math.nan


# --------------------------------------------------------------------------
# tolerance-budget analysis  (the core idea of this package)
# --------------------------------------------------------------------------
@dataclass
class BudgetReport:
    """How much of a hard tolerance a layer actually spends."""

    name: str
    tolerance: float
    unit: str
    display_scale: float
    maximum: float
    rms: float
    p95: float
    median: float
    utilisation_max: float
    utilisation_rms: float
    utilisation_p95: float
    over_budget_fraction: float
    near_boundary_fraction: float
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "maximum": self.maximum,
            "rms": self.rms,
            "p95": self.p95,
            "median": self.median,
            "utilisation_max": self.utilisation_max,
            "utilisation_rms": self.utilisation_rms,
            "utilisation_p95": self.utilisation_p95,
            "over_budget_fraction": self.over_budget_fraction,
            "near_boundary_fraction": self.near_boundary_fraction,
            "sample_count": self.sample_count,
        }

    def line(self) -> str:
        s = self.display_scale
        return (
            f"{self.name}: max {s * self.maximum:.3f} {self.unit}, "
            f"p95 {s * self.p95:.3f} {self.unit}, rms {s * self.rms:.3f} {self.unit} "
            f"| budget p95 {self.utilisation_p95:.2f}x, "
            f"over tolerance {100 * self.over_budget_fraction:.1f}% of samples"
        )


def budget_report(
    name: str,
    errors: Any,
    tolerance: float,
    *,
    unit: str = "mm",
    display_scale: float = 1.0e3,
    near_boundary: float = 0.8,
) -> BudgetReport:
    values = finite(errors)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name}: tolerance must be finite and positive.")
    utilisation = values / tolerance if values.size else values
    return BudgetReport(
        name=name,
        tolerance=float(tolerance),
        unit=unit,
        display_scale=float(display_scale),
        maximum=maximum(values),
        rms=rms(values),
        p95=percentile(values, 95.0),
        median=percentile(values, 50.0),
        utilisation_max=maximum(utilisation),
        utilisation_rms=rms(utilisation),
        utilisation_p95=percentile(utilisation, 95.0),
        over_budget_fraction=fraction(utilisation > 1.0 + 1e-12) if values.size else math.nan,
        near_boundary_fraction=fraction(utilisation >= near_boundary) if values.size else math.nan,
        sample_count=int(values.size),
    )


# --------------------------------------------------------------------------
# geometric-path quality
# --------------------------------------------------------------------------
def path_derivatives(
    s: Any, states: Any, scale: Any = None
) -> tuple[Array, Array, Array, Array]:
    """``(ds, d(chi/scale)/ds, d2(chi/scale)/ds2, raw node-to-node deltas)``.

    The second-difference formula is the non-uniform-grid one used by the
    global optimiser, so the numbers here are directly comparable with the
    weights in ``GlobalConfigurationOptimizerConfig``.
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("states must be a 2-D array of shape (nodes, coordinates)")
    width = states.shape[1]
    scale_vector = (
        np.ones(width)
        if scale is None
        else np.asarray(scale, dtype=float).reshape(-1)[:width]
    )
    if s.size != states.shape[0]:
        raise ValueError("s and state row counts differ")
    if s.size < 2:
        empty = np.empty((0, width))
        return np.empty(0), empty, empty, empty
    ds = np.diff(s)
    if np.any(~np.isfinite(ds)) or np.any(ds <= 0.0):
        raise ValueError("Path coordinate must be finite and strictly increasing.")
    delta = np.diff(states, axis=0)
    first = delta / ds[:, None] / scale_vector[None, :]
    if first.shape[0] >= 2:
        second = 2.0 * np.diff(first, axis=0) / (ds[:-1] + ds[1:])[:, None]
    else:
        second = np.empty((0, width))
    return ds, first, second, delta


def polyline_jaggedness(points: Any, s: Any = None) -> dict[str, Any]:
    """Quantify how 'jagged' a 3-D path is, independently of its length.

    ``turning_angle_*`` is the metric that matches the visual impression of a
    ragged path: the angle between consecutive chords.  A smooth curve sampled
    finely has small turning angles; a path that jitters inside a tolerance
    tube has large ones even though its total excursion is tiny.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    output: dict[str, Any] = {"sample_count": int(points.shape[0])}
    if points.shape[0] < 3:
        return output
    chords = np.diff(points, axis=0)
    lengths = np.linalg.norm(chords, axis=1)
    usable = lengths > 1e-15
    unit = np.zeros_like(chords)
    unit[usable] = chords[usable] / lengths[usable, None]
    cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    angles = np.arccos(cosine)
    angles = angles[np.isfinite(angles)]
    total_length = float(np.sum(lengths))
    straight = float(np.linalg.norm(points[-1] - points[0]))
    output.update(
        {
            "arc_length_m": total_length,
            "end_to_end_m": straight,
            "tortuosity": total_length / straight if straight > 1e-15 else math.nan,
            "turning_angle_max_deg": math.degrees(maximum(angles)),
            "turning_angle_rms_deg": math.degrees(rms(angles)),
            "turning_angle_p95_deg": math.degrees(percentile(angles, 95.0)),
            "total_absolute_turning_deg": math.degrees(float(np.sum(angles))),
        }
    )
    if s is not None:
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size == points.shape[0]:
            _, first, second, _ = path_derivatives(s, points)
            output.update(
                {
                    "max_abs_d2p_ds2": float(max_abs(second)) if second.size else math.nan,
                    "rms_d2p_ds2": rms(np.linalg.norm(second, axis=1)) if second.size else math.nan,
                    "max_abs_dp_ds": float(max_abs(first)) if first.size else math.nan,
                }
            )
    return output


def nearest_polyline_distance(points: Any, polyline: Any) -> Array:
    """Distance from every point to a 3-D polyline, by segment projection."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    polyline = np.asarray(polyline, dtype=float).reshape(-1, 3)
    if polyline.shape[0] < 2:
        return np.full(points.shape[0], math.nan)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    length_sq = np.sum(vectors * vectors, axis=1)
    keep = length_sq > 1e-24
    starts, vectors, length_sq = starts[keep], vectors[keep], length_sq[keep]
    if starts.shape[0] == 0:
        return np.full(points.shape[0], math.nan)
    out = np.empty(points.shape[0], dtype=float)
    for index, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            out[index] = math.nan
            continue
        t = np.clip(np.sum((point - starts) * vectors, axis=1) / length_sq, 0.0, 1.0)
        closest = starts + t[:, None] * vectors
        out[index] = float(np.min(np.linalg.norm(closest - point, axis=1)))
    return out


def limit_utilisation(values: Any, limits: Any) -> Array:
    """``|value| / limit`` per column, with non-positive limits mapped to NaN."""
    values = np.asarray(values, dtype=float)
    limits = np.asarray(limits, dtype=float).reshape(-1)
    if values.ndim == 1:
        values = values[:, None]
    width = min(values.shape[1], limits.size)
    out = np.full(values.shape, math.nan)
    for column in range(width):
        limit = limits[column]
        if np.isfinite(limit) and limit > 0.0:
            out[:, column] = np.abs(values[:, column]) / limit
    return out


def binding_axis(utilisation: Array, names: Sequence[str] = COORDINATE_NAMES) -> dict[str, Any]:
    """Which coordinate is closest to its limit, and how often it is the worst."""
    utilisation = np.asarray(utilisation, dtype=float)
    if utilisation.size == 0:
        return {"available": False}
    with np.errstate(invalid="ignore"):
        per_axis_peak = np.nanmax(utilisation, axis=0)
        worst_per_row = np.nanargmax(np.nan_to_num(utilisation, nan=-1.0), axis=1)
    counts = np.bincount(worst_per_row, minlength=utilisation.shape[1])
    order = int(np.nanargmax(per_axis_peak))
    return {
        "available": True,
        "peak_per_axis": per_axis_peak.tolist(),
        "binding_axis": names[order] if order < len(names) else str(order),
        "binding_axis_index": order,
        "binding_peak": float(per_axis_peak[order]),
        "share_of_samples_where_axis_is_worst": {
            (names[i] if i < len(names) else str(i)): float(counts[i]) / float(utilisation.shape[0])
            for i in range(utilisation.shape[1])
        },
    }


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
@dataclass
class Finding:
    """One judged statement about a layer, for the report's verdict block."""

    level: str  # "ok" | "warn" | "fail" | "info"
    title: str
    detail: str

    ICONS = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "info": "NOTE"}

    def line(self) -> str:
        return f"- **{self.ICONS.get(self.level, '?')}** — {self.title}. {self.detail}"


@dataclass
class Report:
    title: str
    lines: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def head(self, text: str, level: int = 2) -> None:
        self.lines.extend(["", "#" * level + " " + text, ""])

    def text(self, *paragraphs: str) -> None:
        for paragraph in paragraphs:
            self.lines.extend([paragraph, ""])

    def bullets(self, items: Iterable[str]) -> None:
        for item in items:
            self.lines.append(f"- {item}")
        self.lines.append("")

    def table(self, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
        self.lines.append("| " + " | ".join(str(h) for h in header) + " |")
        self.lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            self.lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
        self.lines.append("")

    def finding(self, level: str, title: str, detail: str) -> None:
        self.findings.append(Finding(level, title, detail))

    def render(self) -> str:
        body = [f"# {self.title}", ""]
        if self.findings:
            body.extend(["## Verdict", ""])
            body.extend(f.line() for f in self.findings)
            body.append("")
        body.extend(self.lines)
        return "\n".join(body).rstrip() + "\n"

    def write(self, output_dir: Path, stem: str) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{stem}_report.md"
        metrics_path = output_dir / f"{stem}_metrics.json"
        report_path.write_text(self.render(), encoding="utf-8")
        payload = dict(self.metrics)
        payload["findings"] = [
            {"level": f.level, "title": f.title, "detail": f.detail} for f in self.findings
        ]
        metrics_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
        return {"report": report_path, "metrics": metrics_path}

    @property
    def worst_level(self) -> str:
        for level in ("fail", "warn", "ok"):
            if any(f.level == level for f in self.findings):
                return level
        return "info"


def _cell(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
PALETTE = {
    "achieved": "#0089A0",
    "desired": "#E2669E",
    "magnet": "#C46A00",
    "before": "#8A9BA0",
    "limit": "#B3261E",
    "grid": "#C7D6D6",
}


def new_figure(rows: int, columns: int, size: tuple[float, float]):
    """Matplotlib figure with a consistent house style; None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plots are always optional
        return None, None
    figure, axes = plt.subplots(rows, columns, figsize=size, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.grid(True, alpha=0.25, color=PALETTE["grid"])
    return figure, axes


def save_figure(figure, path: Path, dpi: int = 170) -> None:
    if figure is None:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


__all__ = [
    "Array", "BudgetReport", "COORDINATE_NAMES", "COORDINATE_UNITS",
    "DEFAULT_CONFIGURATION_SCALE", "Finding", "PALETTE", "Report", "Table",
    "binding_axis", "budget_report", "finite", "find_one", "fraction",
    "improvement_percent", "jsonable", "limit_utilisation", "max_abs",
    "maximum", "minimum", "nearest_polyline_distance", "new_figure",
    "path_derivatives", "percentile", "polyline_jaggedness", "read_json",
    "read_json_if_present", "read_table", "rms", "save_figure", "write_csv",
]
