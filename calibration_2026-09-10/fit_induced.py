import numpy as np, json, io, contextlib, time
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True)

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

# ---- load the two insertion=30mm datasets (arc+z+dipole+combined, and dipole-vs-radius) ----
BASE = "/home/jack/Proper-Research/calibration_2026-09-10/"
D1 = json.load(open(BASE + "calib_sweep_20260911/data.json"))
D2 = json.load(open(BASE + "calib_sweep_dipole_vs_radius/data.json"))
rows = [d for d in (D1 + D2) if d.get("cam_tipB") is not None]
tcp = [np.array(d["tcp"]) for d in rows]
ins = [d["ins"] for d in rows]
cam_v = np.array([d["cam_tipB"][1] for d in rows]) * 1e3
kinds = [d["kind"] for d in rows]
print(f"n={len(rows)}  kinds={ {k: kinds.count(k) for k in set(kinds)} }")

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.simulation.magnetic_beam.run_solver_smoke_test as SST

with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
T_R_B = mapper.T_R_B
_orig_factory = SST.make_uniform_axial_m_local_factory

def build_induced(chi_ax, chi_tr, m_sat):
    def factory(*, moment_per_length, local_axis):
        return SST.make_induced_axial_transverse_m_local_factory(
            chi_axial=chi_ax, chi_transverse=chi_tr, local_axis=local_axis, m_sat_A_m=m_sat)
    SST.make_uniform_axial_m_local_factory = factory
    try:
        with quiet():
            ad, _ = rfmv.build_forward_model_in_shared_frame(
                experiment_cfg=ev2.CONFIG, T_R_B=T_R_B, lumen=lumen, model_base_rotation_R=mbr,
                dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
    finally:
        SST.make_uniform_axial_m_local_factory = _orig_factory
    return ad

def evalm(ad):
    out = []
    for p, L in zip(tcp, ins):
        _, _, T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
        with quiet():
            _, o = ad.commit_nominal(rfmv.transform_to_p8(T_R_M, float(L)))
        t = T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
        out.append(1e3 * t[1])
    return np.array(out)

t0 = time.perf_counter()

def objective(x):
    chi_ax, chi_tr_frac = x
    chi_tr = chi_ax * np.clip(chi_tr_frac, 0.0, 1.0)
    ad = build_induced(chi_ax, chi_tr, m_sat=8.0)
    mod_v = evalm(ad)
    rms = float(np.sqrt(np.mean((mod_v - cam_v) ** 2)))
    print(f"  chi_ax={chi_ax:7.3f} chi_tr={chi_tr:7.3f} (frac={chi_tr_frac:.3f})  RMS={rms:.3f} mm  "
          f"[{time.perf_counter()-t0:5.0f}s]")
    return rms

res = minimize(objective, x0=[6.0, 0.5], method="Nelder-Mead",
               bounds=[(0.5, 20.0), (0.0, 1.0)],
               options={"xatol": 0.05, "fatol": 0.02, "maxiter": 25})
chi_ax, chi_tr_frac = res.x
chi_tr = chi_ax * chi_tr_frac
print(f"\nBEST: chi_axial={chi_ax:.3f}  chi_transverse={chi_tr:.3f} (ratio {chi_tr_frac:.3f})  RMS={res.fun:.3f} mm")

# ---- compare against the OLD fixed-axial model (current calibrated E=3.0MPa/yaw=-6) ----
with quiet():
    ad_old, _ = rfmv.build_forward_model_in_shared_frame(
        experiment_cfg=ev2.CONFIG, T_R_B=T_R_B, lumen=lumen, model_base_rotation_R=mbr,
        dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
mod_old = evalm(ad_old)
rms_old = float(np.sqrt(np.mean((mod_old - cam_v) ** 2)))
ad_new = build_induced(chi_ax, chi_tr, m_sat=8.0)
mod_new = evalm(ad_new)

print(f"\nOLD fixed-axial (E=3.0MPa, yaw=-6):        RMS = {rms_old:.3f} mm")
print(f"NEW induced (chi_ax={chi_ax:.2f}, chi_tr={chi_tr:.2f}): RMS = {res.fun:.3f} mm")

print("\nper-kind RMS (old vs new):")
for kind in sorted(set(kinds)):
    idx = [i for i, k in enumerate(kinds) if k == kind]
    r_old = float(np.sqrt(np.mean((mod_old[idx] - cam_v[idx]) ** 2)))
    r_new = float(np.sqrt(np.mean((mod_new[idx] - cam_v[idx]) ** 2)))
    print(f"  {kind:10s} n={len(idx):3d}  old={r_old:6.2f}  new={r_new:6.2f}")

json.dump({
    "chi_axial": chi_ax, "chi_transverse": chi_tr, "chi_tr_over_chi_ax": chi_tr_frac,
    "m_sat_A_m": 8.0, "rms_old_mm": rms_old, "rms_new_mm": res.fun,
    "n_poses": len(rows),
}, open("/home/jack/.claude/jobs/3710eca5/tmp/fit_induced.json", "w"), indent=1)
print("\nsaved fit_induced.json")
