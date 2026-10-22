import numpy as np, json, io, contextlib, dataclasses
np.set_printoptions(precision=4, suppress=True)
@contextlib.contextmanager
def quiet():
    f=io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f): yield

D = json.load(open("/home/jack/.claude/jobs/3710eca5/tmp/ysweep_data.json"))
tcp = [np.array(x["tcp"]) for x in D]
magY = np.array([1e3*x["magnetB"][1] for x in D])
camB = np.array([x["cam_tipB"] for x in D])
cam_in = np.degrees(np.arctan2(camB[:,1], camB[:,0]))
cam_iny_mm = 1e3*camB[:,1]

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.parameters as PP
_omp = PP.default_magnet_params; BASE_EPM=_omp().mag_epm
E0 = ev2.CONFIG.composite.effective_youngs_modulus_pa
with quiet():
    base=ev2._base_module(); base.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=27.0, beam_axial_axis_R=(0,0,1.0), beam_plane_normal_axis_R=(-1.0,0,0)))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:,0]), mapper.T_R_B.rotation[:,2])
T_R_B = mapper.T_R_B
DIP = tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body)

def build(epm_s, E_s):
    def patched():
        m=_omp()
        try: m.mag_epm = BASE_EPM*epm_s
        except Exception: pass
        return m
    PP.default_magnet_params=patched
    oc = ev2.CONFIG.composite
    ev2.CONFIG.composite = dataclasses.replace(oc, effective_youngs_modulus_pa=E0*E_s)
    try:
        with quiet():
            ad,_ = rfmv.build_forward_model_in_shared_frame(experiment_cfg=ev2.CONFIG, T_R_B=T_R_B,
                lumen=lumen, model_base_rotation_R=mbr, dipole_unit_in_magnet_body=DIP)
    finally:
        PP.default_magnet_params=_omp; ev2.CONFIG.composite=oc
    return ad

def evalm(ad, ins=0.045):
    ip=[]; op=[]; ax=[]
    for p in tcp:
        _,_,T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
        with quiet(): _,out = ad.commit_nominal(rfmv.transform_to_p8(T_R_M, ins))
        tB = T_R_B.inverse().apply_points(np.asarray(out).reshape(-1)[:3])
        ip.append(np.degrees(np.arctan2(tB[1],tB[0]))); op.append(np.degrees(np.arctan2(tB[2],np.hypot(tB[0],tB[1])))); ax.append(1e3*tB[0])
    return np.array(ip),np.array(op),np.array(ax)

print(f"BASE_EPM={BASE_EPM:.1f}  E0={E0:.0f} Pa")
print("\nepm_s  E_s  | RMS(cam_in)  slope_ratio  out@ref  ax@ref")
best=None
for es in (0.05,0.1,0.2,0.35,0.6,1.0):
    for Es in (1.0,0.5,0.25,0.12):
        ad=build(es,Es); mi,mo,ma = evalm(ad)
        # match in-plane deg curve, allow a prebend offset
        pre=np.median(cam_in - mi)
        rms=np.sqrt(np.mean((cam_in-mi-pre)**2))
        # slope near origin (|magY|<40)
        msk=np.abs(magY)<45
        sr = (np.polyfit(magY[msk],cam_in[msk],1)[0]) / (np.polyfit(magY[msk],mi[msk],1)[0]+1e-9)
        i0=np.argmin(np.abs(magY))
        print(f"  {es:<5} {Es:<4} | {rms:8.2f}   {sr:8.2f}    {mo[i0]:+6.1f}  {ma[i0]:5.1f}")
        if best is None or rms<best[0]: best=(rms,es,Es,pre,mi,mo,ma)
print(f"\nBEST epm_s {best[1]} E_s {best[2]}  RMS {best[0]:.2f} deg  prebend {best[3]:+.2f}")
mi,mo,ma=best[4],best[5],best[6]
print("\nmagB.y   cam_in   model_in+pre   model_out   model_ax")
for k in np.argsort(magY):
    print(f"  {magY[k]:+7.1f}   {cam_in[k]:+7.2f}   {mi[k]+best[3]:+7.2f}      {mo[k]:+6.1f}    {ma[k]:5.1f}")
