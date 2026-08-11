from __future__ import annotations

"""
Create a single diagnostic GIF from one MPC experiment run directory.

The script is intentionally configured by editing RUN_DIR below rather than by
command-line arguments. It reuses the PNGs that your simulation already writes,
so the 3-D beam/lumen/source-magnet rendering keeps exactly the visual style of
`plot_magnetic_beam_scene`.

Expected run layout
-------------------
RUN_DIR/
    log.csv
    frames/
        with_source_magnet/frame_00000.png
        without_source_magnet/frame_00000.png
    diagnostic_frames/
        actual_jacobian_authority_split/frame_00000.png
        prediction_errors/prediction_errors_00000.png
        rollout_trajectory_xy/rollout_trajectory_xy_00000.png
        command_visibility_split/frame_00000.png
        selected_channel_contributions_split/frame_00000.png

The five diagnostic panels are configurable in DIAGNOSTIC_PANELS. The sixth
right-hand panel is a compact status card populated from log.csv.
"""

from pathlib import Path
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "2026-08-08-v5-final-integer-frame-index"

# Edit this path for each run. Relative paths are resolved from the directory
# from which you launch this script (normally the Proper-Research repo root).
RUN_DIR = Path(
    "experiment1_sim_2/"
    "bends_p0_m30_jac_no_contact_plant_contact_1_ctrl_mpc_lti_"
    "rollout10_Np15_sqp50_bends_0_-30"
)

# Output is written inside the run directory unless you change this.
OUTPUT_GIF = RUN_DIR / "diagnostic_summary.gif"

# Optional: write every composed GIF frame as a PNG for closer inspection.
SAVE_COMPOSITE_FRAMES = False
COMPOSITE_FRAME_DIR = RUN_DIR / "diagnostic_summary_frames"

# Animation controls.
FPS = 2.0
FRAME_START: int | None = None
FRAME_END: int | None = None       # inclusive
FRAME_STEP = 1
HOLD_LAST_SECONDS = 1.0

# Large enough that the embedded diagnostic plots remain legible, but still a
# practical GIF size. Increase to (3000, 1688) for presentation-quality output.
CANVAS_SIZE = (2400, 1350)
BACKGROUND = (245, 247, 250)
PANEL_BACKGROUND = (255, 255, 255)
PANEL_BORDER = (198, 204, 212)
TEXT = (32, 36, 42)
SUBTLE_TEXT = (95, 103, 115)
ACCENT = (31, 119, 180)
DANGER = (196, 55, 55)
WARNING = (210, 139, 35)
GOOD = (40, 145, 83)

# Left-hand scene renders. These are the frames produced from your existing
# `plot_magnetic_beam_scene` style.
SCENE_PANELS = [
    ("With source magnet", RUN_DIR / "frames" / "with_source_magnet"),
    ("Without source magnet", RUN_DIR / "frames" / "without_source_magnet"),
]

# Right-hand diagnostics. Reorder, remove, or replace entries here without
# touching the rest of the script.
DIAGNOSTIC_PANELS = [
    (
        "Actual Jacobian authority",
        RUN_DIR / "diagnostic_frames" / "actual_jacobian_authority_split",
    ),
    (
        "Prediction errors",
        RUN_DIR / "diagnostic_frames" / "prediction_errors",
    ),
    (
        "Rollout trajectory XY",
        RUN_DIR / "diagnostic_frames" / "rollout_trajectory_xy",
    ),
    (
        "Command visibility split",
        RUN_DIR / "diagnostic_frames" / "command_visibility_split",
    ),
    (
        "Selected channel contributions",
        RUN_DIR / "diagnostic_frames" / "selected_channel_contributions_split",
    ),
]

# If a panel is missing one frame, use the nearest earlier frame from that
# panel. This is useful if a diagnostic is emitted less often than the scene.
# Alternatives: "nearest" or "blank".
MISSING_FRAME_POLICY = "previous"

# Status thresholds used only for colouring the summary card; they do not alter
# controller data or the simulation.
ANGLE_WARNING_DEG = 40.0
CLEARANCE_WARNING_MM = 1.5
PREDICTION_ERROR_WARNING_MM = 0.5


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

_FRAME_RE = re.compile(r"(\d+)$")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a sensible local font on macOS/Linux, with a Pillow fallback."""
    candidates = []
    if bold:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = _font(38, bold=True)
FONT_PANEL = _font(25, bold=True)
FONT_BODY = _font(24)
FONT_BODY_BOLD = _font(24, bold=True)
FONT_SMALL = _font(20)
FONT_TINY = _font(17)


def _frame_map(folder: Path) -> dict[int, Path]:
    """
    Map animation frame number -> PNG path.

    The frame number is the final integer in the PNG filename stem, so all of
    these naming conventions are accepted:

        frame_00014.png
        prediction_errors_00014.png
        rollout_trajectory_xy_00014.png
        arbitrary_diagnostic_name_00014.png
    """
    out: dict[int, Path] = {}
    if not folder.is_dir():
        return out

    for path in sorted(folder.glob("*.png")):
        if not path.is_file():
            continue

        match = _FRAME_RE.search(path.stem)
        if match is None:
            continue

        frame_id = int(match.group(1))
        if frame_id in out:
            raise RuntimeError(
                "Multiple PNGs map to the same frame number "
                f"{frame_id} in {folder}:\n"
                f"  {out[frame_id].name}\n"
                f"  {path.name}"
            )

        out[frame_id] = path

    return dict(sorted(out.items()))


def _choose_frame(frame_map: dict[int, Path], frame_id: int) -> Path | None:
    if frame_id in frame_map:
        return frame_map[frame_id]
    if not frame_map or MISSING_FRAME_POLICY == "blank":
        return None

    ids = np.asarray(list(frame_map.keys()), dtype=int)
    if MISSING_FRAME_POLICY == "previous":
        eligible = ids[ids <= frame_id]
        if eligible.size == 0:
            return None
        return frame_map[int(eligible[-1])]
    if MISSING_FRAME_POLICY == "nearest":
        nearest = int(ids[np.argmin(np.abs(ids - frame_id))])
        return frame_map[nearest]
    raise ValueError(f"Unknown MISSING_FRAME_POLICY={MISSING_FRAME_POLICY!r}")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def _fit_image(
    source: Image.Image,
    width: int,
    height: int,
    *,
    pad_color: tuple[int, int, int] = PANEL_BACKGROUND,
) -> Image.Image:
    """Letterbox an image without cropping or distorting plot axes/text."""
    if width <= 0 or height <= 0:
        raise ValueError("Panel dimensions must be positive.")
    scale = min(width / source.width, height / source.height)
    new_size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    resized = source.resize(new_size, Image.Resampling.LANCZOS)
    dst = Image.new("RGB", (width, height), pad_color)
    x = (width - new_size[0]) // 2
    y = (height - new_size[1]) // 2
    dst.paste(resized, (x, y))
    return dst


def _draw_panel(
    canvas: Image.Image,
    *,
    box: tuple[int, int, int, int],
    title: str,
    image_path: Path | None,
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        box,
        radius=14,
        fill=PANEL_BACKGROUND,
        outline=PANEL_BORDER,
        width=2,
    )

    title_h = 44
    draw.text((x0 + 16, y0 + 10), title, fill=TEXT, font=FONT_PANEL)
    image_box = (x0 + 8, y0 + title_h, x1 - 8, y1 - 8)
    iw = max(1, image_box[2] - image_box[0])
    ih = max(1, image_box[3] - image_box[1])

    if image_path is None:
        draw.rectangle(image_box, fill=(250, 250, 250))
        msg = "Frame not available"
        bbox = draw.textbbox((0, 0), msg, font=FONT_BODY)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = image_box[0] + (iw - tw) // 2
        ty = image_box[1] + (ih - th) // 2
        draw.text((tx, ty), msg, fill=SUBTLE_TEXT, font=FONT_BODY)
        return

    fitted = _fit_image(_open_rgb(image_path), iw, ih)
    canvas.paste(fitted, (image_box[0], image_box[1]))


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _get(row: pd.Series | None, key: str, default=np.nan):
    if row is None or key not in row.index:
        return default
    return row[key]


def _fmt_number(value, digits=3, suffix="") -> str:
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _status_colour(label: str) -> tuple[int, int, int]:
    u = label.upper()
    if any(token in u for token in ("INFEAS", "VIOLATION", "IMPOSSIBLE", "BLOCKED")):
        return DANGER
    if any(token in u for token in ("WARNING", "LIMITED", "REPOSITION")):
        return WARNING
    if "ADVANCE" in u or "SOLVED" in u:
        return GOOD
    return ACCENT


def _draw_metric(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    label: str,
    value: str,
    colour: tuple[int, int, int] = TEXT,
    label_width: int = 255,
) -> int:
    draw.text((x, y), label, fill=SUBTLE_TEXT, font=FONT_SMALL)
    draw.text((x + label_width, y), value, fill=colour, font=FONT_BODY_BOLD)
    return y + 34


def _draw_status_card(
    canvas: Image.Image,
    *,
    box: tuple[int, int, int, int],
    frame_id: int,
    row: pd.Series | None,
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        box,
        radius=14,
        fill=PANEL_BACKGROUND,
        outline=PANEL_BORDER,
        width=2,
    )
    draw.text((x0 + 16, y0 + 10), "Controller / geometry state", fill=TEXT, font=FONT_PANEL)

    state = str(_get(row, "controller_motion_state", "n/a"))
    status = str(_get(row, "status", "n/a"))
    state_colour = _status_colour(state)

    y = y0 + 58
    draw.text((x0 + 18, y), f"frame {frame_id:05d}", fill=SUBTLE_TEXT, font=FONT_SMALL)
    draw.text((x0 + 190, y), state, fill=state_colour, font=FONT_BODY_BOLD)
    y += 36
    draw.text((x0 + 18, y), status[:68], fill=SUBTLE_TEXT, font=FONT_TINY)
    y += 38

    current_m = _get(row, "progress_current_m")
    safe_mm = _get(row, "progress_predicted_safe_mm")
    slack = _get(row, "progress_slack_fraction")
    clearance = _get(row, "clearance_mm")
    angle = _get(row, "tip_vessel_angle_deg")
    pred_err = _get(row, "pred1_err_xy_mm")
    epm_activity = _get(row, "progress_epm_reposition_activity_fraction")
    n_active = _get(row, "tip_tangent_constraint_num_active")

    angle_colour = DANGER if _finite(angle) and float(angle) > ANGLE_WARNING_DEG else TEXT
    clearance_colour = (
        DANGER
        if _finite(clearance) and float(clearance) < CLEARANCE_WARNING_MM
        else TEXT
    )
    err_colour = (
        WARNING
        if _finite(pred_err) and float(pred_err) > PREDICTION_ERROR_WARNING_MM
        else TEXT
    )

    left = x0 + 18
    y = _draw_metric(
        draw,
        x=left,
        y=y,
        label="Path progress",
        value=_fmt_number(float(current_m) * 1e3 if _finite(current_m) else np.nan, 3, " mm"),
    )
    y = _draw_metric(draw, x=left, y=y, label="Predicted safe progress", value=_fmt_number(safe_mm, 3, " mm"))
    y = _draw_metric(draw, x=left, y=y, label="Progress slack", value=_fmt_number(float(slack) * 100 if _finite(slack) else np.nan, 2, " %"))
    y = _draw_metric(draw, x=left, y=y, label="Wall clearance", value=_fmt_number(clearance, 3, " mm"), colour=clearance_colour)
    y = _draw_metric(draw, x=left, y=y, label="Tip / lumen angle", value=_fmt_number(angle, 2, " deg"), colour=angle_colour)
    y = _draw_metric(draw, x=left, y=y, label="1-step prediction error", value=_fmt_number(pred_err, 3, " mm"), colour=err_colour)
    y = _draw_metric(draw, x=left, y=y, label="EPM activity", value=_fmt_number(float(epm_activity) * 100 if _finite(epm_activity) else np.nan, 1, " %"))
    y = _draw_metric(draw, x=left, y=y, label="Active tangent rows", value=_fmt_number(n_active, 0))

    # Compact u0 command row. This is useful when inspecting how Jacobian
    # authority/visibility changes translate into the command actually sent.
    command_names = ["vx", "vy", "vz", "wx", "wy", "wz", "dL"]
    command_values = [_get(row, f"u0_{name}") for name in command_names]
    y += 8
    draw.line((x0 + 18, y, x1 - 18, y), fill=PANEL_BORDER, width=1)
    y += 12
    draw.text((x0 + 18, y), "u0", fill=SUBTLE_TEXT, font=FONT_SMALL)
    y += 30

    available_w = max(1, x1 - x0 - 36)
    col_w = available_w // len(command_names)
    for i, (name, value) in enumerate(zip(command_names, command_values)):
        cx = x0 + 18 + i * col_w
        draw.text((cx, y), name, fill=SUBTLE_TEXT, font=FONT_TINY)
        draw.text((cx, y + 24), _fmt_number(value, 3), fill=TEXT, font=FONT_TINY)


def _load_log(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "log.csv"
    if not path.is_file():
        print(f"[warning] log.csv not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "k" not in df.columns:
        print("[warning] log.csv has no 'k' column; status card will be blank.")
        return df
    return df


def _row_for_frame(df: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if df.empty or "k" not in df.columns:
        return None
    matches = df.loc[pd.to_numeric(df["k"], errors="coerce") == frame_id]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _run_label(run_dir: Path) -> str:
    # Keep the full folder name available, but wrap long names into a readable
    # two-line title by splitting around the middle underscore.
    name = run_dir.name
    if len(name) <= 95:
        return name
    midpoint = len(name) // 2
    candidates = [m.start() for m in re.finditer("_", name)]
    if not candidates:
        return name
    split_at = min(candidates, key=lambda p: abs(p - midpoint))
    return name[:split_at] + "\n" + name[split_at + 1 :]


def _compose_frame(
    *,
    frame_id: int,
    scene_maps: list[tuple[str, dict[int, Path]]],
    diagnostic_maps: list[tuple[str, dict[int, Path]]],
    log_df: pd.DataFrame,
) -> Image.Image:
    W, H = CANVAS_SIZE
    canvas = Image.new("RGB", (W, H), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    margin = 18
    header_h = 90
    gap = 14

    title = _run_label(RUN_DIR)
    draw.multiline_text(
        (margin, 10),
        title,
        fill=TEXT,
        font=FONT_TITLE,
        spacing=4,
    )
    frame_text = f"frame {frame_id:05d}"
    frame_bbox = draw.textbbox((0, 0), frame_text, font=FONT_PANEL)
    draw.text(
        (W - margin - (frame_bbox[2] - frame_bbox[0]), 28),
        frame_text,
        fill=ACCENT,
        font=FONT_PANEL,
    )

    content_y0 = header_h
    content_y1 = H - margin
    content_h = content_y1 - content_y0

    left_w = int(W * 0.39)
    right_x0 = margin + left_w + gap
    right_w = W - right_x0 - margin

    # Left: two large beam/lumen renders.
    scene_h = (content_h - gap) // 2
    for i, (title_text, fmap) in enumerate(scene_maps[:2]):
        y0 = content_y0 + i * (scene_h + gap)
        y1 = y0 + scene_h
        _draw_panel(
            canvas,
            box=(margin, y0, margin + left_w, y1),
            title=title_text,
            image_path=_choose_frame(fmap, frame_id),
        )

    # Right: 3 x 2. Five image diagnostics plus one numerical state card.
    cols = 2
    rows = 3
    cell_w = (right_w - gap) // cols
    cell_h = (content_h - 2 * gap) // rows

    cells: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = right_x0 + c * (cell_w + gap)
            y0 = content_y0 + r * (cell_h + gap)
            cells.append((x0, y0, x0 + cell_w, y0 + cell_h))

    for cell, (title_text, fmap) in zip(cells[:5], diagnostic_maps[:5]):
        _draw_panel(
            canvas,
            box=cell,
            title=title_text,
            image_path=_choose_frame(fmap, frame_id),
        )

    _draw_status_card(
        canvas,
        box=cells[5],
        frame_id=frame_id,
        row=_row_for_frame(log_df, frame_id),
    )

    return canvas


def _select_frame_ids(reference_map: dict[int, Path]) -> list[int]:
    ids = sorted(reference_map)
    if FRAME_START is not None:
        ids = [i for i in ids if i >= FRAME_START]
    if FRAME_END is not None:
        ids = [i for i in ids if i <= FRAME_END]
    ids = ids[:: max(1, int(FRAME_STEP))]
    return ids


def _validate_structure() -> None:
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(
            f"RUN_DIR does not exist: {RUN_DIR}\n"
            "Edit RUN_DIR near the top of this script."
        )


def main() -> None:
    _validate_structure()

    scene_maps = [(title, _frame_map(folder)) for title, folder in SCENE_PANELS]
    diagnostic_maps = [
        (title, _frame_map(folder)) for title, folder in DIAGNOSTIC_PANELS
    ]

    if not scene_maps[0][1]:
        raise FileNotFoundError(
            "No frame_XXXXX.png files found in:\n"
            f"  {SCENE_PANELS[0][1]}\n"
            "The with-source-magnet scene folder is used as the animation clock."
        )

    frame_ids = _select_frame_ids(scene_maps[0][1])
    if not frame_ids:
        raise RuntimeError("No frames selected after FRAME_START/END/STEP filtering.")

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Script file: {Path(__file__).resolve()}")
    print(f"Run: {RUN_DIR.resolve()}")
    print(f"Frames: {frame_ids[0]} .. {frame_ids[-1]}  (n={len(frame_ids)})")
    for title, fmap in scene_maps + diagnostic_maps:
        print(f"  {title:34s}: {len(fmap):4d} PNGs")

    print("\nPanel sources:")
    for title, folder in SCENE_PANELS + DIAGNOSTIC_PANELS:
        print(f"  {title}:")
        print(f"    {folder.resolve()}")

    log_df = _load_log(RUN_DIR)

    if SAVE_COMPOSITE_FRAMES:
        COMPOSITE_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    gif_frames: list[Image.Image] = []
    for j, frame_id in enumerate(frame_ids, start=1):
        print(f"Composing {j:3d}/{len(frame_ids):3d}: frame {frame_id:05d}", end="\r")
        composed = _compose_frame(
            frame_id=frame_id,
            scene_maps=scene_maps,
            diagnostic_maps=diagnostic_maps,
            log_df=log_df,
        )

        if SAVE_COMPOSITE_FRAMES:
            composed.save(
                COMPOSITE_FRAME_DIR / f"frame_{frame_id:05d}.png",
                dpi=(150, 150),
            )

        # GIF is palette-based. Quantizing each frame here sharply reduces
        # memory and final file size while retaining plot/text readability.
        palette_frame = composed.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        gif_frames.append(palette_frame)

    print()

    frame_duration_ms = max(20, int(round(1000.0 / max(FPS, 1e-6))))
    durations = [frame_duration_ms] * len(gif_frames)
    if HOLD_LAST_SECONDS > 0 and durations:
        durations[-1] += int(round(1000.0 * HOLD_LAST_SECONDS))

    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=gif_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )

    size_mb = OUTPUT_GIF.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved GIF: {OUTPUT_GIF}")
    print(f"Size: {size_mb:.1f} MiB")
    if SAVE_COMPOSITE_FRAMES:
        print(f"Composite PNGs: {COMPOSITE_FRAME_DIR}")


if __name__ == "__main__":
    main()
