import numpy as np, json, io, contextlib, dataclasses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

BASE = "/home/jack/.claude/jobs/3710eca5/tmp/"
DA = json.load(open(BASE + "sweep_dipole_arc_data.json"))
IN = json.load(open(BASE + "sweep_insertion_data.json"))

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2

E_NEW = 2.5e6
with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
    ev2.CONFIG.composite = dataclasses.replace(ev2.CONFIG.composite, effective_youngs_modulus_pa=E_NEW)
    ad, _ = rfmv.build_forward_model_in_shared_frame(
        experiment_cfg=ev2.CONFIG, T_R_B=mapper.T_R_B, lumen=lumen, model_base_rotation_R=mbr,
        dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
T_R_B = mapper.T_R_B

def model_in(tcp, L):
    _, _, T_R_M = rfmv.source_transform_from_tcp(np.asarray(tcp), rfmv.CONFIG)
    with quiet():
        _, o = ad.commit_nominal(rfmv.transform_to_p8(T_R_M, float(L)))
    t = T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
    return (np.degrees(np.arctan2(t[1], t[0])),
            np.degrees(np.arctan2(t[2], np.hypot(t[0], t[1]))))

# ---- arc: group by radius ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for r_tag, col in (("r28", "tab:blue"), ("r32", "tab:orange"), ("r38", "tab:green")):
    P = [d for d in DA if d["label"].startswith("arc_" + r_tag)]
    phi = [int(d["label"].split("phi")[1]) for d in P]
    cam = [np.degrees(np.arctan2(np.array(d["cam_tipB"])[1], np.array(d["cam_tipB"])[0])) for d in P]
    mod = [model_in(d["tcp"], d["ins"])[0] for d in P]
    o = np.argsort(phi); phi = np.array(phi)[o]; cam = np.array(cam)[o]; mod = np.array(mod)[o]
    ax[0].plot(phi, cam, "o-", color=col, label=f"cam {r_tag[1:]}cm")
    ax[0].plot(phi, mod, "x--", color=col, alpha=0.6, label=f"model {r_tag[1:]}cm")
ax[0].set_xlabel("magnet azimuth around beam base [deg]"); ax[0].set_ylabel("in-plane tip bend [deg]")
ax[0].set_title("xy-plane ARC  (E=2.5 MPa)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=3)

# ---- dipole rotation (coax) ----
P = [d for d in DA if d["label"].startswith("dipZ_coax")]
az = [int(d["label"].split("_")[-1]) for d in P]
cam = [np.degrees(np.arctan2(np.array(d["cam_tipB"])[1], np.array(d["cam_tipB"])[0])) for d in P]
mod = [model_in(d["tcp"], d["ins"])[0] for d in P]
o = np.argsort(az); az = np.array(az)[o]; cam = np.array(cam)[o]; mod = np.array(mod)[o]
ax[1].plot(az, cam, "o-", label="camera")
ax[1].plot(az, mod, "x--", label="model")
Po = [d for d in DA if d["label"].startswith("dipZ_off")]
az2 = [int(d["label"].split("_")[-1]) for d in Po]
cam2 = [np.degrees(np.arctan2(np.array(d["cam_tipB"])[1], np.array(d["cam_tipB"])[0])) for d in Po]
mod2 = [model_in(d["tcp"], d["ins"])[0] for d in Po]
o = np.argsort(az2); az2 = np.array(az2)[o]; cam2 = np.array(cam2)[o]; mod2 = np.array(mod2)[o]
ax[1].plot(az2, cam2, "s-", color="tab:red", label="camera (+off-axis bend)")
ax[1].plot(az2, mod2, "+--", color="tab:red", label="model (+off-axis bend)")
ax[1].set_xlabel("source-dipole rotation about world Z [deg]"); ax[1].set_ylabel("in-plane tip bend [deg]")
ax[1].set_title("DIPOLE ROTATION"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)

# ---- insertion length ----
L = np.array([d["cam_len_m"] * 1e3 for d in IN])
cam = np.array([np.degrees(np.arctan2(np.array(d["cam_tipB"])[1], np.array(d["cam_tipB"])[0])) for d in IN])
mod = np.array([model_in(d["tcp"], d["cam_len_m"])[0] for d in IN])
o = np.argsort(L)
ax[2].plot(L[o], cam[o], "o-", label="camera")
ax[2].plot(L[o], mod[o], "x--", label="model")
ax[2].set_xlabel("inserted beam length [mm]"); ax[2].set_ylabel("in-plane tip bend [deg]")
ax[2].set_title("INSERTION LENGTH  (magnet fixed off-axis)"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
ax[2].axvspan(18, 36, color="grey", alpha=0.08)

fig.tight_layout()
fig.savefig(BASE + "sweep_all.png", dpi=110)
print("saved", BASE + "sweep_all.png")

# numeric summary
def rms(a, b): return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))
allc, allm, allco, allmo = [], [], [], []
for d in DA:
    if d.get("cam_tipB") is None: continue
    c = np.array(d["cam_tipB"])
    allc.append(np.degrees(np.arctan2(c[1], c[0])))
    allco.append(np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1]))))
    mi, mo = model_in(d["tcp"], d["ins"]); allm.append(mi); allmo.append(mo)
print(f"dipole_arc  in-plane RMS {rms(allc, allm):.2f} deg   cam out-of-plane max |{np.max(np.abs(allco)):.2f}|  model out max |{np.max(np.abs(allmo)):.2f}|")
