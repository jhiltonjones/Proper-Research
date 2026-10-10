#!/usr/bin/env python3
"""Offline (no robot, no camera) analysis of an acquire.py log: ring-down
metrics, G_excitation, dominant oscillation frequency, damping ratio,
settling time -- and the length-dependence trend. See this package's README
for the metric definitions (taken verbatim from the test's predeclared
specification) and the decision rule.

Also writes a resampled-at-10Hz vibration record per length (`r_k`, the
post-advance residual sampled at the controller's own tick rate) to an npz
for `causality.py` to consume.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.analyze \\
        --log advancer_excitation_logs/rectangle_2026-09-18/advancer_excitation_<stamp>.jsonl \\
        --out advancer_excitation_logs/rectangle_2026-09-18/analysis.npz
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

CONTROLLER_DT_S = 0.1  # 10Hz -- close_loop_path_follow.CONFIG.control_hz


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
        raise ValueError(f"{log_path} has no _meta header row (not an acquire.py log?)")
    return meta, rows


def _phase_rows(rows, L_index: int, phase: str, repeat_index: int | None = None):
    out = [r for r in rows if r["L_index"] == L_index and r["phase"] == phase
           and (repeat_index is None or r["repeat_index"] == repeat_index)]
    out.sort(key=lambda r: r["t_monotonic"])
    return out


def _t_p(rows) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([r["t_monotonic"] for r in rows], dtype=float)
    p = np.array([r["tip_position_m"] for r in rows], dtype=float)
    if t.size:
        t = t - t[0]
    return t, p


def resample_uniform(t: np.ndarray, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate a (possibly multi-column) signal onto a uniform grid."""
    if t.size < 2:
        return t.copy(), x.copy()
    t_uniform = np.arange(t[0], t[-1], dt)
    if x.ndim == 1:
        return t_uniform, np.interp(t_uniform, t, x)
    cols = [np.interp(t_uniform, t, x[:, j]) for j in range(x.shape[1])]
    return t_uniform, np.stack(cols, axis=1)


def psd_peak(t: np.ndarray, x: np.ndarray, *, exclude_dc_hz: float = 0.5) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Rough PSD via a Hann-windowed rFFT on a uniformly resampled signal.
    Returns (f_osc, P_osc, freqs, psd)."""
    if t.size < 8:
        return float("nan"), float("nan"), np.array([]), np.array([])
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return float("nan"), float("nan"), np.array([]), np.array([])
    t_u, x_u = resample_uniform(t, x, dt)
    n = x_u.size
    if n < 8:
        return float("nan"), float("nan"), np.array([]), np.array([])
    window = np.hanning(n)
    x_win = (x_u - np.mean(x_u)) * window
    spec = np.fft.rfft(x_win)
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = (np.abs(spec) ** 2) / (np.sum(window ** 2) + 1e-30)
    mask = freqs > exclude_dc_hz
    if not np.any(mask):
        return float("nan"), float("nan"), freqs, psd
    idx = np.argmax(psd[mask])
    f_osc = float(freqs[mask][idx])
    band = (freqs >= 0.8 * f_osc) & (freqs <= 1.2 * f_osc)
    p_osc = float(np.trapz(psd[band], freqs[band])) if np.sum(band) > 1 else float(psd[mask][idx])
    return f_osc, p_osc, freqs, psd


def log_decrement_zeta(t: np.ndarray, r3d: np.ndarray) -> tuple[float, float, int]:
    """Peak-pick |r(t)| and return (delta, zeta, n_peaks_used). NaN if fewer
    than 2 usable peaks."""
    if r3d.size < 5:
        return float("nan"), float("nan"), 0
    peaks = []
    for i in range(1, r3d.size - 1):
        if r3d[i] > r3d[i - 1] and r3d[i] >= r3d[i + 1] and r3d[i] > 0:
            peaks.append(r3d[i])
    if len(peaks) < 2:
        return float("nan"), float("nan"), len(peaks)
    ratios = [np.log(peaks[i] / peaks[i + 1]) for i in range(len(peaks) - 1) if peaks[i + 1] > 0]
    ratios = [r for r in ratios if np.isfinite(r)]
    if not ratios:
        return float("nan"), float("nan"), len(peaks)
    delta = float(np.mean(ratios))
    zeta = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2) if delta > 0 else float("nan")
    return delta, zeta, len(peaks)


def settling_time_10pct(t: np.ndarray, r3d: np.ndarray, a_peak: float) -> float:
    if a_peak <= 0 or t.size == 0:
        return float("nan")
    threshold = 0.1 * a_peak
    below = r3d < threshold
    for i in range(t.size):
        if below[i] and np.all(below[i:]):
            return float(t[i])
    return float("nan")  # never settled within the recorded window


def _dominant_signed_component(r: np.ndarray) -> np.ndarray:
    """r: (n,3) residual matrix -> (n,) projection onto its dominant PCA
    direction (sign arbitrary, zero-crossings preserved)."""
    if r.shape[0] < 3:
        return np.linalg.norm(r, axis=1)
    _u, _s, vt = np.linalg.svd(r - r.mean(axis=0, keepdims=True), full_matrices=False)
    return r @ vt[0]


@dataclass
class RepeatMetrics:
    L_index: int
    repeat: int
    a_peak: float
    r_ring: float
    f_osc: float
    p_osc: float
    delta: float
    zeta: float
    n_peaks: int
    t10: float


def analyze_repeat(
    rows_hold_post: list, *, peak_window_s: float = 1.0, psd_window_s: float = 2.0,
) -> tuple[RepeatMetrics, np.ndarray, np.ndarray]:
    t, p = _t_p(rows_hold_post)
    if t.size < 5:
        raise ValueError("too few hold_post samples for this repeat")
    final_mask = t >= max(t[-1] - 0.5, 0.0)
    p_inf = p[final_mask].mean(axis=0) if np.any(final_mask) else p[-1]
    r = p - p_inf[None, :]
    r3d = np.linalg.norm(r, axis=1)

    win = t <= peak_window_s
    a_peak = float(np.max(r3d[win])) if np.any(win) else float("nan")
    r_ring = float(np.sqrt(np.mean(r3d[win] ** 2))) if np.any(win) else float("nan")
    # PSD over a window covering the active transient but excluding most of
    # the long noise-only tail (which otherwise dilutes the real peak's SNR
    # for a fast-decaying mode) -- capped at what's actually available.
    psd_win = t <= min(psd_window_s, t[-1])
    # f_osc must come from a SIGNED proxy, not r3d: r3d = ||r(t)|| rectifies
    # any single-axis (or planar) oscillation, which folds the fundamental
    # onto 2x its true frequency in the power spectrum (confirmed on
    # synthetic ground truth -- r3d gave exactly 2x the injected frequency).
    # Project onto the dominant PCA direction of r(t) over the PSD window
    # instead, which preserves sign/zero-crossings; A_peak/R_ring correctly
    # keep using r3d (the test's own amplitude definition, sign-irrelevant).
    r_signed = _dominant_signed_component(r[psd_win])
    f_osc, p_osc, _freqs, _psd = psd_peak(t[psd_win], r_signed)
    # Same rectification issue affects log-decrement: consecutive r3d peaks
    # are half a period apart (|cos| has two lobes per cycle), which halves
    # the apparent delta/zeta. Positive-going peaks of the signed component
    # recur once per full period, matching the classical definition.
    r_signed_win = _dominant_signed_component(r[win])
    delta, zeta, n_peaks = log_decrement_zeta(t[win], r_signed_win)
    t10 = settling_time_10pct(t, r3d, a_peak)
    return (
        RepeatMetrics(L_index=-1, repeat=-1, a_peak=a_peak, r_ring=r_ring,
                       f_osc=f_osc, p_osc=p_osc, delta=delta, zeta=zeta, n_peaks=n_peaks, t10=t10),
        t, r,
    )


def baseline_noise(rows_baseline: list) -> float:
    t, p = _t_p(rows_baseline)
    if t.size < 3:
        return float("nan")
    p_mean = p.mean(axis=0)
    r3d = np.linalg.norm(p - p_mean[None, :], axis=1)
    return float(np.sqrt(np.mean(r3d ** 2)))


def fit_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Linear fit y ~ slope*x + intercept, returns (slope, intercept, r2)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    y_hat = slope * x[mask] + intercept
    ss_res = np.sum((y[mask] - y_hat) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    p.add_argument("--out", default=None, help="npz path for per-repeat metrics + resampled r_k")
    args = p.parse_args()

    meta, rows = load_rows(args.log)
    test_lengths_mm = meta["test_lengths_mm"]
    n_repeats = meta["config"]["repeats"]
    print(f"[analyze] {args.log}: {len(rows)} rows, lengths(mm)={[round(v,2) for v in test_lengths_mm]}")

    per_L_repeat_metrics: dict[int, list[RepeatMetrics]] = {}
    per_L_noise: dict[int, float] = {}
    per_L_resampled_r: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for L_index, L_mm in enumerate(test_lengths_mm, start=1):
        baseline_rows = _phase_rows(rows, L_index, "baseline")
        r_noise = baseline_noise(baseline_rows)
        per_L_noise[L_index] = r_noise

        metrics_list = []
        all_t_abs, all_r = [], []
        t_offset = 0.0
        for rep in range(1, n_repeats + 1):
            hp_rows = _phase_rows(rows, L_index, "hold_post", rep)
            if len(hp_rows) < 5:
                print(f"[analyze] L{L_index} rep{rep}: too few hold_post samples ({len(hp_rows)}), skipping")
                continue
            m, t_rel, r = analyze_repeat(hp_rows)
            m.L_index, m.repeat = L_index, rep
            metrics_list.append(m)
            all_t_abs.append(t_rel + t_offset)
            all_r.append(r)
            t_offset += t_rel[-1] + CONTROLLER_DT_S  # keep repeats from overlapping in the concatenated record
        per_L_repeat_metrics[L_index] = metrics_list

        if all_t_abs:
            t_cat = np.concatenate(all_t_abs)
            r_cat = np.concatenate(all_r, axis=0)
            t_k, r_k = resample_uniform(t_cat, r_cat, CONTROLLER_DT_S)
            per_L_resampled_r[L_index] = (t_k, r_k)

    print("\n=== ADVANCER-EXCITATION SUMMARY (mean +/- std across repeats) ===")
    header = f"{'L(mm)':>8} {'A_peak(um)':>12} {'R_ring(um)':>12} {'R_noise(um)':>12} {'G_exc':>8} {'f_osc(Hz)':>10} {'zeta':>8} {'t10%(s)':>8}"
    print(header)
    L_mm_arr, a_peak_mean, f_osc_mean = [], [], []
    for L_index, L_mm in enumerate(test_lengths_mm, start=1):
        ms = per_L_repeat_metrics[L_index]
        r_noise = per_L_noise[L_index]
        if not ms:
            print(f"{L_mm:8.2f} {'--':>12} {'--':>12} {r_noise*1e6:12.2f} {'--':>8} {'--':>10} {'--':>8} {'--':>8}")
            continue
        a_peak = np.array([m.a_peak for m in ms])
        r_ring = np.array([m.r_ring for m in ms])
        f_osc = np.array([m.f_osc for m in ms])
        zeta = np.array([m.zeta for m in ms])
        t10 = np.array([m.t10 for m in ms])
        g_exc = r_ring / max(r_noise, 1e-12)
        print(f"{L_mm:8.2f} {np.nanmean(a_peak)*1e6:8.1f}+-{np.nanstd(a_peak)*1e6:<4.1f} "
              f"{np.nanmean(r_ring)*1e6:8.1f}+-{np.nanstd(r_ring)*1e6:<4.1f} "
              f"{r_noise*1e6:12.2f} {np.nanmean(g_exc):8.2f} "
              f"{np.nanmean(f_osc):6.2f}+-{np.nanstd(f_osc):<4.2f} "
              f"{np.nanmean(zeta):8.4f} {np.nanmean(t10):8.3f}")
        L_mm_arr.append(L_mm)
        a_peak_mean.append(np.nanmean(a_peak))
        f_osc_mean.append(np.nanmean(f_osc))

    L_mm_arr, a_peak_mean, f_osc_mean = map(np.array, (L_mm_arr, a_peak_mean, f_osc_mean))
    if L_mm_arr.size >= 2:
        slope_a, _, r2_a = fit_trend(L_mm_arr, a_peak_mean)
        slope_f, _, r2_f = fit_trend(L_mm_arr, f_osc_mean)
        print(f"\n[analyze] A_peak(L) trend: slope={slope_a*1e6:.3f} um/mm, R^2={r2_a:.3f} "
              f"({'increases' if slope_a > 0 else 'decreases'} with L)")
        print(f"[analyze] f_osc(L) trend:  slope={slope_f:.4f} Hz/mm, R^2={r2_f:.3f} "
              f"({'increases' if slope_f > 0 else 'decreases'} with L)")

    if args.out:
        save_kwargs = {"test_lengths_mm": np.array(test_lengths_mm)}
        for L_index in per_L_repeat_metrics:
            ms = per_L_repeat_metrics[L_index]
            save_kwargs[f"L{L_index}_a_peak"] = np.array([m.a_peak for m in ms])
            save_kwargs[f"L{L_index}_r_ring"] = np.array([m.r_ring for m in ms])
            save_kwargs[f"L{L_index}_f_osc"] = np.array([m.f_osc for m in ms])
            save_kwargs[f"L{L_index}_zeta"] = np.array([m.zeta for m in ms])
            save_kwargs[f"L{L_index}_t10"] = np.array([m.t10 for m in ms])
            save_kwargs[f"L{L_index}_r_noise"] = np.array([per_L_noise[L_index]])
            if L_index in per_L_resampled_r:
                t_k, r_k = per_L_resampled_r[L_index]
                save_kwargs[f"L{L_index}_t_k"] = t_k
                save_kwargs[f"L{L_index}_r_k"] = r_k  # (K,3) resampled vibration @ 10Hz, meters
        np.savez(args.out, **save_kwargs)
        print(f"\n[analyze] saved -> {args.out}")


if __name__ == "__main__":
    main()
