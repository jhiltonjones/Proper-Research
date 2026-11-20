import numpy as np, json, io, contextlib, dataclasses
np.set_printoptions(precision=3, suppress=True)

@contextlib.contextmanager
def q():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

D = json.load(open("/home/jack/.claude/jobs/3710eca5/tmp/corner_sweep_data.json"))
rows = [d for d in D if d.get("cam_tipB") is not None]
tcp = [np.array(d["tcp"]) for d in rows]
camB = np.array([d["cam_tipB"] for d in rows])
magB = np.array([d["magnetB"] for d in rows])
cam_v = camB[:, 1] * 1e3
cam_u = camB[:, 0] * 1e3
mag_y = magB[:, 1] * 1e3
mag_x = magB[:, 0] * 1e3
INS = 0.038

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.simulation.magnetic_beam.run_solver_smoke_test as SST

_omm = SST.make_uniform_axial_m_local_factory
E0 = 2.5e6
S = {"m": 1.0}

def mm(*, moment_per_length, local_axis):
    return _omm(moment_per_length=moment_per_length * S["m"], local_axis=local_axis)

with q():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
T_R_B = mapper.T_R_B

def online_model_v(yaw_deg, E_s, msc):
    S["m"] = msc
    SST.make_uniform_axial_m_local_factory = mm
    oc = ev2.CONFIG.composite
    oy = ev2.CONFIG.magnet_yaw_calibration_deg
    ev2.CONFIG.composite = dataclasses.replace(oc, effective_youngs_modulus_pa=E0 * E_s)
    ev2.CONFIG.magnet_yaw_calibration_deg = float(yaw_deg)
    try:
        with q():
            ad, _ = rfmv.build_forward_model_in_shared_frame(
                experiment_cfg=ev2.CONFIG, T_R_B=T_R_B, lumen=lumen, model_base_rotation_R=mbr,
                dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
            out = []
            for p in tcp:
                _, _, T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
                _, o = ad.commit_nominal(rfmv.transform_to_p8(T_R_M, INS))
                t = T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
                out.append(t * 1e3)
    finally:
        SST.make_uniform_axial_m_local_factory = _omm
        ev2.CONFIG.composite = oc
        ev2.CONFIG.magnet_yaw_calibration_deg = oy
    return np.array(out)

# ---- current online model ----
cur = online_model_v(-10.0, 1.0, 1.0)
print("=== corner sweep: cam vs CURRENT online model (yaw -10, E 2.5MPa) ===")
print(" mag_y  cam_v  mod_v  d(mod-cam)   cam_u  mod_u")
for i in np.argsort(mag_y):
    print(f"  {mag_y[i]:+5.0f}  {cam_v[i]:+6.2f} {cur[i,1]:+6.2f}   {cur[i,1]-cam_v[i]:+6.2f}    {cam_u[i]:5.1f} {cur[i,0]:5.1f}")

# ---- slope + bias of cam and model vs mag_y ----
mask_p = mag_y > 3
mask_n = mag_y < -3
def slope_bias(y, x):
    A = np.vstack([x, np.ones_like(x)]).T
    s, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return s, c
sc, cc = slope_bias(cam_v, mag_y)
print(f"\ncam_v  = {sc:.4f}*mag_y + {cc:+.3f}   (slope mm/mm, bias mm)")
sm, cm = slope_bias(cur[:, 1], mag_y)
print(f"mod_v  = {sm:.4f}*mag_y + {cm:+.3f}")
print(f"  -> slope ratio model/cam = {sm/sc:.3f}   bias diff (model-cam) = {cm-cc:+.3f} mm")

# ---- grid: yaw + E + moment ----
print("\n=== grid search (min RMS model_v - cam_v over all 25 poses) ===")
print("yaw   E_s  msc | RMS_v  bias  slopeR | RMS(+y) RMS(-y)")
best = None
for yaw in (-10, -6, -3, 0, 3):
    for E_s in (0.85, 1.0, 1.2, 1.45):
        for msc in (1.0,):
            m = online_model_v(yaw, E_s, msc)
            r = m[:, 1] - cam_v
            rms = np.sqrt(np.mean(r**2))
            s2, c2 = slope_bias(m[:, 1], mag_y)
            rp = np.sqrt(np.mean(r[mask_p]**2)); rn = np.sqrt(np.mean(r[mask_n]**2))
            print(f"{yaw:+4.0f}  {E_s:<4} {msc:<3} | {rms:5.2f}  {c2-cc:+5.2f}  {s2/sc:5.3f} | {rp:5.2f}  {rn:5.2f}")
            if best is None or rms < best[0]:
                best = (rms, yaw, E_s, msc, m)
print(f"\nBEST: yaw={best[1]}  E_s={best[2]} (E={E0*best[2]:.0f} Pa)  msc={best[3]}  RMS_v={best[0]:.2f} mm")

# ---- OFFLINE model (build_model_bundle, no yaw) at the same poses ----
print("\n=== OFFLINE planner model (build_planning_context bundle) at the swept magnet poses ===")
from proper_research.planning.planning_context import build_planning_context
with q():
    _, bundle, cp, _ = build_planning_context()
mdl = bundle.models["no_contact"]
off = []
for p in tcp:
    _, _, T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
    p8 = rfmv.transform_to_p8(T_R_M, INS)
    from proper_research.hardware.beam_hardware_experiment_v2 import p8_to_p7
    p7 = p8_to_p7(p8)
    try:
        with q():
            res = mdl(p7)
        tip = np.asarray(getattr(res, "tip_position", getattr(res, "y", res))).reshape(-1)[:3]
    except Exception as e:
        tip = np.array([np.nan]*3)
    off.append(T_R_B.inverse().apply_points(tip) * 1e3)
off = np.array(off)
so, co = slope_bias(off[np.isfinite(off[:, 1]), 1], mag_y[np.isfinite(off[:, 1])])
print(f"offline mod_v = {so:.4f}*mag_y + {co:+.3f}    slopeR={so/sc:.3f}  biasdiff={co-cc:+.3f}")
print(" mag_y  cam_v  offline_v  d")
for i in np.argsort(mag_y):
    print(f"  {mag_y[i]:+5.0f}  {cam_v[i]:+6.2f}  {off[i,1]:+7.2f}   {off[i,1]-cam_v[i]:+6.2f}")

json.dump({"cam_slope": float(sc), "cam_bias": float(cc),
           "online_cur_slope": float(sm), "online_cur_bias": float(cm),
           "best_yaw": best[1], "best_E_scale": best[2], "best_E_pa": E0*best[2], "best_rms_v_mm": best[0],
           "offline_slope": float(so), "offline_bias": float(co)},
          open("/home/jack/.claude/jobs/3710eca5/tmp/corner_fit.json", "w"), indent=1)
