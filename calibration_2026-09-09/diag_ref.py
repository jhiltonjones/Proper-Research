import numpy as np, io, contextlib
np.set_printoptions(precision=4, suppress=True)
@contextlib.contextmanager
def quiet():
    f=io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f): yield
from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.parameters as PP
_omp=PP.default_magnet_params; BEPM=_omp().mag_epm
with quiet():
    b=ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper=NewFrameTipMapper(StateStreamConfig(exposure=27.0,beam_axial_axis_R=(0,0,1.0),beam_plane_normal_axis_R=(-1.0,0,0)))
    lumen=rfmv.build_lumen_in_shared_frames(mapper.manual_frame,mapper.calibration,mapper.T_R_B)
    mbr=rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:,0]),mapper.T_R_B.rotation[:,2])
T_R_B=mapper.T_R_B; BASE=np.asarray(T_R_B.translation)
DIP=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body)

def bend(ad, magnet_B_mm, dipole_R, ins=0.045):
    # place magnet at BASE + T_R_B.rotation @ (magnet_B in m); dipole along dipole_R (world)
    from proper_research.hardware.robotics_frame_measurement_validation import FrameTransform
    mR = BASE + T_R_B.rotation @ (np.asarray(magnet_B_mm)/1e3)
    z = -np.asarray(dipole_R,float); z/=np.linalg.norm(z)
    x = np.cross([0,1,0],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    Rm=np.column_stack([x,y,z])
    T=FrameTransform("R","M",np.block([[Rm,mR.reshape(3,1)],[np.zeros((1,3)),np.ones((1,1))]]))
    with quiet(): _,o=ad.commit_nominal(rfmv.transform_to_p8(T,ins))
    t=T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
    ip=np.degrees(np.arctan2(t[1],t[0])); op=np.degrees(np.arctan2(t[2],np.hypot(t[0],t[1])))
    return t, ip, op

def build(epm_s):
    def pat():
        m=_omp()
        try: m.mag_epm=BEPM*epm_s
        except: pass
        return m
    PP.default_magnet_params=pat
    try:
        with quiet(): ad,_=rfmv.build_forward_model_in_shared_frame(experiment_cfg=ev2.CONFIG,T_R_B=T_R_B,lumen=lumen,model_base_rotation_R=mbr,dipole_unit_in_magnet_body=DIP)
    finally: PP.default_magnet_params=_omp
    return ad

for epm in (1.0, 0.3, 0.1, 0.02, 0.0):
    ad=build(max(epm,1e-9))
    t,ip,op = bend(ad, [0,0,230], [0,0,1.0])
    t2,ip2,op2 = bend(ad, [0,0,2000], [0,0,1.0])
    print(f"epm*{epm:<4}: magnet@230mm  tipB {np.round(1e3*t,1)}  in {ip:+.2f} out {op:+.2f} deg  |  magnet@2000mm out {op2:+.2f}")
# dipole along the beam-magnet line (radial, -B.z toward camera... actually +R.x from magnet toward beam)
ad=build(1.0)
for name,dip in [("dipole +Rz (axial)",[0,0,1.0]), ("dipole +Rx (toward beam)",[1,0,0]), ("dipole -Rx",[-1,0,0])]:
    t,ip,op = bend(ad,[0,0,230],dip)
    print(f"  {name:26s}: in {ip:+.2f}  out {op:+.2f} deg")
