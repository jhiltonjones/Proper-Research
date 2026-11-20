import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parents[5] / "calibration_2026-10-05" / "jcam_joint_space_phi40_r225_L25mm.json"  # archive/ adds one level
with open(RESULTS_PATH) as f:
    r = json.load(f)

J_fast = np.array(r["J_fast"])  # 3x6
dqs = np.array(r["dqs_rad"])

J_cam = np.zeros((3, 6))
for col in r["columns"]:
    i = col["joint"]
    J_cam[:, i] = col["J_cam_col"]

print("J_fast:\n", np.round(J_fast, 6))
print("\nJ_cam:\n", np.round(J_cam, 6))

D_J = 2 * np.linalg.norm(J_cam - J_fast, "fro") / (np.linalg.norm(J_cam, "fro") + np.linalg.norm(J_fast, "fro"))
print(f"\nD_J (normalized matrix disagreement) = {D_J:.4f}")

print(f"\n{'col':>4s} {'theta_deg':>10s} {'gain(fast/cam)':>15s} {'|J_fast_col|':>14s} {'|J_cam_col|':>14s}")
for i in range(6):
    a = J_fast[:, i]
    b = J_cam[:, i]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos_t = np.clip(np.dot(a, b) / (na * nb), -1, 1) if na > 1e-9 and nb > 1e-9 else np.nan
    theta = np.degrees(np.arccos(cos_t)) if not np.isnan(cos_t) else float("nan")
    gain = na / nb if nb > 1e-9 else float("nan")
    print(f"q{i+1:>3d} {theta:10.2f} {gain:15.3f} {na:14.6f} {nb:14.6f}")

print("\n=== repeatability (return-to-nominal) ===")
tip0 = np.array(r["tip0"])
for ret in r["returns"]:
    for key in ("tip_ret1", "tip_ret2"):
        if ret[key] is not None:
            d = np.linalg.norm(np.array(ret[key]) - tip0) * 1000
            print(f"joint q{ret['joint']+1} {key}: drift_from_tip0_mm={d:.3f}")

print("\n=== singular values / directions ===")
U_f, S_f, Vt_f = np.linalg.svd(J_fast)
U_c, S_c, Vt_c = np.linalg.svd(J_cam)
print(f"J_fast singular values: {np.round(S_f, 5)}")
print(f"J_cam  singular values: {np.round(S_c, 5)}")
print(f"J_fast dominant output direction (U[:,0]): {np.round(U_f[:,0], 4)}")
print(f"J_cam  dominant output direction (U[:,0]): {np.round(U_c[:,0], 4)}")
cos_u0 = np.clip(abs(np.dot(U_f[:,0], U_c[:,0])), -1, 1)
print(f"angle between dominant output directions: {np.degrees(np.arccos(cos_u0)):.2f}deg")

print("\n=== held-out probes ===")
for i, p in enumerate(r["probes"]):
    u = np.array(p["u_scaled_rad"])
    pred_fast = np.array(p["pred_fast"])
    delta_real = np.array(p["delta_x_real"]) if p["delta_x_real"] is not None else None
    pred_cam_via_Jcam = J_cam @ u
    print(f"\nprobe {i+1}: |u|={np.linalg.norm(u):.4f}rad")
    print(f"  pred_fast (J_fast @ u):     {pred_fast*1000} mm")
    print(f"  pred via J_cam (J_cam @ u): {pred_cam_via_Jcam*1000} mm")
    if delta_real is not None:
        print(f"  ACTUAL measured delta_x:   {delta_real*1000} mm")
        err_fast = np.linalg.norm(pred_fast - delta_real) * 1000
        err_cam = np.linalg.norm(pred_cam_via_Jcam - delta_real) * 1000
        cos_fast = np.clip(np.dot(pred_fast, delta_real) / (np.linalg.norm(pred_fast)*np.linalg.norm(delta_real)+1e-12), -1, 1)
        cos_camJ = np.clip(np.dot(pred_cam_via_Jcam, delta_real) / (np.linalg.norm(pred_cam_via_Jcam)*np.linalg.norm(delta_real)+1e-12), -1, 1)
        print(f"  |pred_fast - actual|  = {err_fast:.3f}mm   angle = {np.degrees(np.arccos(cos_fast)):.2f}deg")
        print(f"  |pred_camJ - actual|  = {err_cam:.3f}mm   angle = {np.degrees(np.arccos(cos_camJ)):.2f}deg")
