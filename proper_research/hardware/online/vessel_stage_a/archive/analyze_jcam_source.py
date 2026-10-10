import json
from pathlib import Path

import numpy as np

from proper_research.hardware.online.vessel_stage_a.archive.jnc_source_jacobian import J_NC_source

RESULTS_PATH = Path(__file__).resolve().parents[5] / "calibration_2026-10-05" / "jcam_source_phi40_r225_L25mm.json"  # archive/ adds one level
with open(RESULTS_PATH) as f:
    r = json.load(f)

LABELS = ["x", "y", "z", "rx", "ry", "rz"]
J_cam_source = np.zeros((3, 6))
for i, col in enumerate(r["columns"]):
    J_cam_source[:, i] = col["J_col"]

print("J_NC_source (model, full 3x6):")
print(np.round(J_NC_source, 6))
print("\nJ_cam_source (camera, full 3x6):")
print(np.round(J_cam_source, 6))

# Fair comparison: XY rows only (camera is blind to Z).
J_NC_xy = J_NC_source[:2, :]
J_cam_xy = J_cam_source[:2, :]

D_J = 2 * np.linalg.norm(J_cam_xy - J_NC_xy, "fro") / (np.linalg.norm(J_cam_xy, "fro") + np.linalg.norm(J_NC_xy, "fro"))
print(f"\nD_J (XY-only) = {D_J:.4f}")

print(f"\n{'axis':>4s} {'theta_deg':>10s} {'gain(NC/cam)':>13s} {'|NC|':>10s} {'|cam|':>10s}")
for i in range(6):
    a, b = J_NC_xy[:, i], J_cam_xy[:, i]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos_t = np.clip(np.dot(a, b) / (na * nb), -1, 1) if na > 1e-9 and nb > 1e-9 else np.nan
    theta = np.degrees(np.arccos(cos_t)) if not np.isnan(cos_t) else float("nan")
    gain = na / nb if nb > 1e-9 else float("nan")
    print(f"{LABELS[i]:>4s} {theta:10.2f} {gain:13.3f} {na:10.6f} {nb:10.6f}")

print("\n=== repeatability ===")
for ret in r["returns"]:
    print(f"{ret['axis']}: drift1={ret['drift1_mm']:.3f}mm  drift2={ret['drift2_mm']:.3f}mm")

U_f, S_f, _ = np.linalg.svd(J_NC_xy)
U_c, S_c, _ = np.linalg.svd(J_cam_xy)
print(f"\nJ_NC_source singular values (XY): {np.round(S_f, 5)}")
print(f"J_cam_source singular values (XY): {np.round(S_c, 5)}")
cos_u = np.clip(abs(np.dot(U_f[:, 0], U_c[:, 0])), -1, 1)
print(f"angle between dominant directions: {np.degrees(np.arccos(cos_u)):.2f}deg")
