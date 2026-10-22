import numpy as np, json, io, contextlib, dataclasses
np.set_printoptions(precision=4, suppress=True)

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

D = json.load(open("/home/jack/.claude/jobs/3710eca5/tmp/calib910_data.json"))
tcp = [np.array(x["tcp"]) for x in D]
kind = [x["kind"] for x in D]
labels = [x["label"] for x in D]
camB = np.array([x["cam_tipB"] for x in D])
magB = np.array([x["magnetB"] for x in D])
cam_in = np.degrees(np.arctan2(camB[:, 1], camB[:, 0]))
magY = 1e3 * magB[:, 1]

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
from proper_research.hardware.robotics_frame_measurement_validation import FrameTransform
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.parameters as PP
import proper_research.simulation.magnetic_beam.run_solver_smoke_test as SST

_omp = PP.default_magnet_params
_omm = SST.make_uniform_axial_m_local_factory
BEPM = _omp().mag_epm
E0 = ev2.CONFIG.composite.effective_youngs_modulus_pa
S = {"mscale": 1.0}

def mm(*, moment_per_length, local_axis):
    return _omm(moment_per_length=moment_per_length * S["mscale"], local_axis=local_axis)

with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=27.0, beam_axial_axis_R=(0, 0, 1.0), beam_plane_normal_axis_R=(-1.0, 0, 0)))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
T_R_B = mapper.T_R_B
DIP = tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body)

def build(E_s, mscale):
    S["mscale"] = mscale
    SST.make_uniform_axial_m_local_factory = mm
    oc = ev2.CONFIG.composite
    ev2.CONFIG.composite = dataclasses.replace(oc, effective_youngs_modulus_pa=E0 * E_s)
    try:
        with quiet():
            ad, _ = rfmv.build_forward_model_in_shared_frame(
                experiment_cfg=ev2.CONFIG, T_R_B=T_R_B, lumen=lumen,
                model_base_rotation_R=mbr, dipole_unit_in_magnet_body=DIP)
    finally:
        SST.make_uniform_axial_m_local_factory = _omm
        ev2.CONFIG.composite = oc
    return ad

def evalm(ad, dz, ins=0.045):
    ip = []; op = []
    for p in tcp:
        _, _, T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
        mR = np.asarray(T_R_M.translation) + T_R_B.rotation @ np.array([dz, 0.0, 0.0])  # shift along beam axial B.x
        T2 = FrameTransform("R", "M", np.block([[T_R_M.rotation, mR.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]))
        with quiet():
            _, o = ad.commit_nominal(rfmv.transform_to_p8(T2, ins))
        t = T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
        ip.append(np.degrees(np.arctan2(t[1], t[0])))
        op.append(np.degrees(np.arctan2(t[2], np.hypot(t[0], t[1]))))
    return np.array(ip), np.array(op)

ymask = np.array([k == "y" or k == "ref" for k in kind])
onaxis = np.abs(magY) < 5    # B.y ~ 0 poses: out-of-plane should be ~0

print("E_s  mscale  dz(mm) | in-plane RMS(y-arc)  |out@onaxis|  (cam y-arc: up to 24 deg)")
best = None
for E_s in (1.0, 0.5, 0.25, 0.12, 0.06):
    for msc in (1.0, 2.0, 3.5, 6.0):
        ad = build(E_s, msc)
        for dz in (0.0, 0.02, 0.035, 0.05, 0.065):
            ip, op = evalm(ad, dz)
            pre = np.median(cam_in[ymask] - ip[ymask])
            rms = np.sqrt(np.mean((cam_in[ymask] - ip[ymask] - pre) ** 2))
            oout = np.sqrt(np.mean(op[onaxis] ** 2))
            print(f"  {E_s:<4} {msc:<5} {1e3*dz:5.0f} | {rms:8.2f}          {oout:7.2f}")
            score = rms + 0.6 * oout
            if best is None or score < best[0]:
                best = (score, E_s, msc, dz, pre, ip, op, rms, oout)

print(f"\nBEST  E_s {best[1]} (E={E0*best[1]:.0f} Pa)  mscale {best[2]}  dz {1e3*best[3]:.0f} mm  "
      f"prebend {best[4]:+.2f}  in-RMS {best[7]:.2f}  |out@onaxis| {best[8]:.2f}")
ip, op = best[5], best[6]
print("\nlabel        kind  magB.y   cam_in   model_in+pre   model_out")
for k in np.argsort(magY):
    print(f"  {labels[k]:12s} {kind[k]:4s} {magY[k]:+7.1f}  {cam_in[k]:+7.2f}   {ip[k]+best[4]:+7.2f}      {op[k]:+7.2f}")
json.dump({"E_pa": E0*best[1], "E_scale": best[1], "beam_moment_scale": best[2],
           "magnet_axial_offset_m": best[3], "prebend_deg": best[4],
           "in_plane_rms_deg": best[7], "out_of_plane_onaxis_rms_deg": best[8]},
          open("/home/jack/.claude/jobs/3710eca5/tmp/calib910_fit.json", "w"), indent=1)
