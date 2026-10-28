#!/usr/bin/env python3
"""Trim a fixed length off one end of a detect_blue.py-format vessel lumen
file (robot-frame lumen_C_m / lumen_R_m + provenance), by arc length,
interpolating exactly at the cut point rather than just dropping the
nearest sample.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.trim_vessel_lumen \\
        --lumen-file vessel_lumen_robot_frame_raised3cm_2026-09-27.json \\
        --trim-mm 6.0 --end end \\
        --out vessel_lumen_robot_frame_raised3cm_trimmed6mm_2026-09-28.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def trim_lumen(C: np.ndarray, R: np.ndarray, trim_m: float, end: str):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if trim_m <= 0 or trim_m >= total:
        raise ValueError(f"trim_m={trim_m} must be in (0, total_length={total}).")

    if end == "end":
        target = total - trim_m
        idx = np.searchsorted(s, target)
        s0, s1 = s[idx - 1], s[idx]
        t = (target - s0) / (s1 - s0)
        C_cut = C[idx - 1] + t * (C[idx] - C[idx - 1])
        R_cut = R[idx - 1] + t * (R[idx] - R[idx - 1])
        C_new = np.vstack([C[:idx], C_cut[None, :]])
        R_new = np.concatenate([R[:idx], [R_cut]])
    elif end == "start":
        target = trim_m
        idx = np.searchsorted(s, target)
        s0, s1 = s[idx - 1], s[idx]
        t = (target - s0) / (s1 - s0)
        C_cut = C[idx - 1] + t * (C[idx] - C[idx - 1])
        R_cut = R[idx - 1] + t * (R[idx] - R[idx - 1])
        C_new = np.vstack([C_cut[None, :], C[idx:]])
        R_new = np.concatenate([[R_cut], R[idx:]])
    else:
        raise ValueError("end must be 'start' or 'end'")

    return C_new, R_new, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--trim-mm", type=float, required=True)
    p.add_argument("--end", choices=["start", "end"], default="end")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.lumen_file) as f:
        d = json.load(f)

    C = np.asarray(d["lumen_C_m"], dtype=float)
    R = np.asarray(d["lumen_R_m"], dtype=float)
    C_new, R_new, total = trim_lumen(C, R, args.trim_mm / 1000.0, args.end)
    new_total = float(np.sum(np.linalg.norm(np.diff(C_new, axis=0), axis=1)))

    d["lumen_C_m"] = C_new.tolist()
    d["lumen_R_m"] = R_new.tolist()
    d.setdefault("provenance", {})["trimmed_note"] = (
        f"{args.trim_mm}mm trimmed from the {args.end} of {args.lumen_file}"
    )
    d["provenance"]["original_arc_length_mm"] = total * 1000.0
    d["provenance"]["trimmed_arc_length_mm"] = new_total * 1000.0

    with open(args.out, "w") as f:
        json.dump(d, f, indent=2)

    print(f"[trim] original arc length = {total * 1000:.2f}mm ({len(C)} points)")
    print(f"[trim] trimmed arc length  = {new_total * 1000:.2f}mm ({len(C_new)} points)")
    print(f"[trim] saved -> {args.out}")


if __name__ == "__main__":
    main()
