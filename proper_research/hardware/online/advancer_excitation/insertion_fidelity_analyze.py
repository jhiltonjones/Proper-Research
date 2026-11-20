#!/usr/bin/env python3
"""Offline (no hardware) analysis of an insertion_fidelity_acquire.py log:
does the advancer's actual insertion gain / smoothness degrade under
magnetic bending load? See this package's README for the metric
definitions and the predeclared interpretation.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.insertion_fidelity_analyze \\
        --log advancer_excitation_logs/rectangle_2026-09-18_load/insertion_fidelity_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .measure_l0 import PIVOT_XYZ

STICK_THRESHOLD_FRAC = 0.2   # |v_meas| < this * v_cmd counts as "stuck"
JUMP_THRESHOLD_MM = 0.05     # a single inter-sample forward jump bigger than this = a "burst"
SMOOTH_WINDOW = 3            # samples, for the velocity estimate only


def load_rows(log_path: str):
    meta = None
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_meta"):
                meta = row
            else:
                rows.append(row)
    if meta is None:
        raise ValueError(f"{log_path} has no _meta header (not an insertion_fidelity_acquire.py log?)")
    return meta, rows


def _phase_rows(rows, L_index: int, phase: str, repeat_index: int):
    out = [r for r in rows if r["L_index"] == L_index and r["phase"] == phase and r["repeat_index"] == repeat_index]
    out.sort(key=lambda r: r["t_monotonic"])
    return out


def _l_meas_mm(rows: list) -> np.ndarray:
    tip = np.array([r["tip_position_m"] for r in rows], dtype=float)
    return np.linalg.norm(tip - PIVOT_XYZ[None, :], axis=1) * 1e3


def _l_cmd_mm(rows: list) -> np.ndarray:
    return np.array([r["insertion_est_m"] for r in rows], dtype=float) * 1e3


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if x.size < window:
        return x.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: x.size]


@dataclass
class RepeatFidelity:
    L_index: int
    repeat: int
    delta_L_cmd_mm: float
    delta_L_meas_mm: float
    G_L: float
    rms_eL_mm: float
    max_eL_mm: float
    eL_end_mm: float
    cv_v: float
    p_stick: float
    n_bursts: int
    mean_burst_mm: float


def analyze_repeat(rows_all: list, L_index: int, repeat: int, *, advance_rate_mm_s: float) -> RepeatFidelity:
    pre = _phase_rows(rows_all, L_index, "hold_pre", repeat)
    adv = _phase_rows(rows_all, L_index, "advance", repeat)
    post = _phase_rows(rows_all, L_index, "hold_post", repeat)
    if len(pre) < 3 or len(adv) < 3 or len(post) < 3:
        raise ValueError(f"L{L_index} rep{repeat}: too few samples "
                          f"(pre={len(pre)} adv={len(adv)} post={len(post)})")

    t_pre = np.array([r["t_monotonic"] for r in pre])
    t_adv = np.array([r["t_monotonic"] for r in adv])
    t_post = np.array([r["t_monotonic"] for r in post])

    L_meas_pre, L_cmd_pre = _l_meas_mm(pre), _l_cmd_mm(pre)
    L_meas_adv, L_cmd_adv = _l_meas_mm(adv), _l_cmd_mm(adv)
    L_meas_post, L_cmd_post = _l_meas_mm(post), _l_cmd_mm(post)

    final_pre_mask = t_pre >= max(t_pre[-1] - 0.5, t_pre[0])
    final_post_mask = t_post >= max(t_post[-1] - 0.5, t_post[0])
    L_meas_before = float(L_meas_pre[final_pre_mask].mean())
    L_meas_after = float(L_meas_post[final_post_mask].mean())
    L_cmd_start = float(L_cmd_pre[-1])
    L_cmd_end = float(L_cmd_adv[-1])

    delta_L_cmd = L_cmd_end - L_cmd_start
    delta_L_meas = L_meas_after - L_meas_before
    g_l = delta_L_meas / delta_L_cmd if abs(delta_L_cmd) > 1e-6 else float("nan")

    # e_L relative to the start of THIS move -- cancels any constant vision/
    # L0-calibration offset between L_meas and L_cmd, isolating the genuine
    # per-move insertion mismatch (see module docstring / README).
    t_cat = np.concatenate([t_adv, t_post]) - t_adv[0]
    L_meas_cat = np.concatenate([L_meas_adv, L_meas_post]) - L_meas_before
    L_cmd_cat = np.concatenate([L_cmd_adv, L_cmd_post]) - L_cmd_start
    e_L = L_meas_cat - L_cmd_cat
    rms_eL = float(np.sqrt(np.mean(e_L ** 2)))
    max_eL = float(np.max(np.abs(e_L)))
    eL_end = float(e_L[-1])

    # stick-slip, steady-insertion window only (the "advance" phase)
    L_meas_adv_smooth = _smooth(L_meas_adv, SMOOTH_WINDOW)
    v_meas = np.gradient(L_meas_adv_smooth, t_adv)  # mm/s
    mean_v = float(np.mean(v_meas))
    cv_v = float(np.std(v_meas) / abs(mean_v)) if abs(mean_v) > 1e-9 else float("nan")
    p_stick = float(np.mean(np.abs(v_meas) < STICK_THRESHOLD_FRAC * advance_rate_mm_s))

    # Burst threshold scales with the EXPECTED per-sample advance at this
    # camera rate, not a fixed mm value: a fixed threshold below the smooth
    # per-sample step (rate/fps) flags ordinary motion as "bursts" at low
    # fps and misses real bursts at high fps. Caught on synthetic ground
    # truth: a fixed 0.05mm threshold gave MORE "bursts" for a smooth ramp
    # (~18Hz camera, 0.072mm/sample) than for an injected stick-then-release
    # signal, the wrong direction entirely.
    dt_median = float(np.median(np.diff(t_adv))) if t_adv.size > 1 else 1.0 / 18.0
    expected_step_mm = advance_rate_mm_s * dt_median
    burst_threshold_mm = max(JUMP_THRESHOLD_MM, 3.0 * expected_step_mm)
    raw_diffs = np.diff(L_meas_adv)  # unsmoothed -- bursts are a sample-to-sample event
    bursts = raw_diffs[raw_diffs > burst_threshold_mm]
    n_bursts = int(bursts.size)
    mean_burst = float(bursts.mean()) if n_bursts else 0.0

    return RepeatFidelity(
        L_index=L_index, repeat=repeat, delta_L_cmd_mm=delta_L_cmd, delta_L_meas_mm=delta_L_meas,
        G_L=g_l, rms_eL_mm=rms_eL, max_eL_mm=max_eL, eL_end_mm=eL_end, cv_v=cv_v,
        p_stick=p_stick, n_bursts=n_bursts, mean_burst_mm=mean_burst,
    )


def step_interval_check(rows_all: list, L_index: int, repeat: int) -> None:
    """Coarse (camera-poll-rate-limited) inter-step-count-change intervals
    during the advance phase -- flags gross bursting, not true per-pulse
    timing (see README's caveat on this)."""
    adv = _phase_rows(rows_all, L_index, "advance", repeat)
    t = np.array([r["t_monotonic"] for r in adv])
    steps = np.array([r["steps_dispatched_total"] for r in adv], dtype=float)
    changed = np.where(np.diff(steps) != 0)[0]
    if changed.size < 2:
        print(f"    step-interval: too few step-count changes to characterize ({changed.size})")
        return
    intervals = np.diff(t[changed])
    print(f"    step-interval (coarse, camera-poll-limited): n_events={changed.size} "
          f"mean={intervals.mean()*1e3:.1f}ms std={intervals.std()*1e3:.1f}ms "
          f"CV={intervals.std()/intervals.mean():.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    args = p.parse_args()

    meta, rows = load_rows(args.log)
    poses = meta["poses"]
    advance_rate_mm_s = meta["config"]["advance_rate_mm_s"]
    print(f"[analyze] {args.log}: {len(rows)} rows, {len(poses)} load poses, "
          f"advance_rate={advance_rate_mm_s:.3f}mm/s")

    print("\n=== INSERTION-FIDELITY SUMMARY (mean +/- std across repeats) ===")
    header = (f"{'load':>8} {'b(mm)':>7} {'G_L':>14} {'RMS(eL)mm':>12} {'max|eL|mm':>11} "
              f"{'eL_end mm':>11} {'CV_v':>10} {'P(stick)':>9} {'bursts/rep':>10}")
    print(header)
    all_results: dict[int, list[RepeatFidelity]] = {}
    for load_index, pose in enumerate(poses, start=1):
        results = []
        for rep in range(1, meta["config"]["repeats"] + 1):
            try:
                r = analyze_repeat(rows, load_index, rep, advance_rate_mm_s=advance_rate_mm_s)
                results.append(r)
            except ValueError as exc:
                print(f"[analyze] {exc}")
        all_results[load_index] = results
        if not results:
            print(f"{pose['label']:>8} {pose['b_mm']:7.2f} {'--':>14} {'--':>12} {'--':>11} "
                  f"{'--':>11} {'--':>10} {'--':>9} {'--':>10}")
            continue
        g_l = np.array([r.G_L for r in results])
        rms_eL = np.array([r.rms_eL_mm for r in results])
        max_eL = np.array([r.max_eL_mm for r in results])
        eL_end = np.array([r.eL_end_mm for r in results])
        cv_v = np.array([r.cv_v for r in results])
        p_stick = np.array([r.p_stick for r in results])
        n_bursts = np.array([r.n_bursts for r in results], dtype=float)
        print(f"{pose['label']:>8} {pose['b_mm']:7.2f} "
              f"{np.nanmean(g_l):5.3f}+-{np.nanstd(g_l):<5.3f} "
              f"{np.nanmean(rms_eL):7.4f}+-{np.nanstd(rms_eL):<4.4f} "
              f"{np.nanmean(max_eL):11.4f} {np.nanmean(eL_end):11.4f} "
              f"{np.nanmean(cv_v):10.3f} {np.nanmean(p_stick):9.3f} {np.nanmean(n_bursts):10.2f}")

    print("\n=== G_L / RMS(e_L) vs bending proxy b ===")
    for load_index, pose in enumerate(poses, start=1):
        results = all_results[load_index]
        if not results:
            continue
        g_l = np.array([r.G_L for r in results])
        rms_eL = np.array([r.rms_eL_mm for r in results])
        print(f"  b={pose['b_mm']:6.2f}mm ({pose['label']:>6}): "
              f"G_L={np.nanmean(g_l):.3f}  RMS(eL)={np.nanmean(rms_eL):.4f}mm")
    labels_b = [poses[i]['b_mm'] for i in range(len(poses)) if all_results[i + 1]]
    labels_g = [np.nanmean([r.G_L for r in all_results[i + 1]]) for i in range(len(poses)) if all_results[i + 1]]
    labels_r = [np.nanmean([r.rms_eL_mm for r in all_results[i + 1]]) for i in range(len(poses)) if all_results[i + 1]]
    if len(labels_b) >= 2:
        g_trend = "decreases" if labels_g[-1] < labels_g[0] else "increases/flat"
        r_trend = "increases" if labels_r[-1] > labels_r[0] else "decreases/flat"
        print(f"\n[analyze] G_L(b) {g_trend} from b={labels_b[0]:.2f}mm to b={labels_b[-1]:.2f}mm "
              f"({labels_g[0]:.3f} -> {labels_g[-1]:.3f})")
        print(f"[analyze] RMS(eL)(b) {r_trend} from b={labels_b[0]:.2f}mm to b={labels_b[-1]:.2f}mm "
              f"({labels_r[0]:.4f} -> {labels_r[-1]:.4f} mm)")

    print("\n=== step-interval check (repeat 1 at each load) ===")
    for load_index, pose in enumerate(poses, start=1):
        print(f"  {pose['label']}:")
        step_interval_check(rows, load_index, 1)


if __name__ == "__main__":
    main()
