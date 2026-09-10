import numpy as np, json, io, contextlib, dataclasses
np.set_printoptions(precision=3, suppress=True)

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

BASE = "/home/jack/.claude/jobs/3710eca5/tmp/"
DA = json.load(open(BASE + "sweep_dipole_arc_data.json"))
IN = json.load(open(BASE + "sweep_insertion_data.json"))

rows = []  # (tcp, ins, cam_in, cam_out, kind, label)
for d in DA:
    if d.get("cam_tipB") is None:
        continue
    c = np.array(d["cam_tipB"])
    rows.append((np.array(d["tcp"]), d["ins"],
                 np.degrees(np.arctan2(c[1], c[0])),
                 np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1]))),
                 d["kind"], d["label"]))
for d in IN:
    c = np.array(d["cam_tipB"])
    rows.append((np.array(d["tcp"]), d["cam_len_m"],
                 np.degrees(np.arctan2(c[1], c[0])),
                 np.degrees(np.arctan2(c[2], np.hypot(c[0], c[1]))),
                 "ins_" + d["phase"], d["tag"]))

tcp = [r[0] for r in rows]; ins = np.array([r[1] for r in rows])
cam_in = np.array([r[2] for r in rows]); cam_out = np.array([r[3] for r in rows])
kind = [r[4] for r in rows]; lab = [r[5] for r in rows]

from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
from proper_research.hardware import robotics_frame_measurement_validation as rfmv
import proper_research.hardware.beam_hardware_experiment_v2 as ev2
import proper_research.simulation.magnetic_beam.run_solver_smoke_test as SST

_omm = SST.make_uniform_axial_m_local_factory
E0 = ev2.CONFIG.composite.effective_youngs_modulus_pa
S = {"m": 1.0}

def mm(*, moment_per_length, local_axis):
    return _omm(moment_per_length=moment_per_length * S["m"], local_axis=local_axis)

with quiet():
    b = ev2._base_module(); b.configure_bounds_beam_paths(ev2.CONFIG)
    mapper = NewFrameTipMapper(StateStreamConfig(exposure=29.0))
    lumen = rfmv.build_lumen_in_shared_frames(mapper.manual_frame, mapper.calibration, mapper.T_R_B)
    mbr = rfmv.model_base_rotation_from_beam(np.asarray(mapper.T_R_B.rotation[:, 0]), mapper.T_R_B.rotation[:, 2])
T_R_B = mapper.T_R_B

def build(E_s, msc):
    S["m"] = msc
    SST.make_uniform_axial_m_local_factory = mm
    oc = ev2.CONFIG.composite
    ev2.CONFIG.composite = dataclasses.replace(oc, effective_youngs_modulus_pa=E0 * E_s)
    try:
        with quiet():
            ad, _ = rfmv.build_forward_model_in_shared_frame(
                experiment_cfg=ev2.CONFIG, T_R_B=T_R_B, lumen=lumen, model_base_rotation_R=mbr,
                dipole_unit_in_magnet_body=tuple(rfmv.CONFIG.source_dipole_unit_in_magnet_body))
    finally:
        SST.make_uniform_axial_m_local_factory = _omm
        ev2.CONFIG.composite = oc
    return ad

def evalm(ad):
    ip = []; op = []
    for p, L in zip(tcp, ins):
        _, _, T_R_M = rfmv.source_transform_from_tcp(p, rfmv.CONFIG)
        with quiet():
            _, o = ad.commit_nominal(rfmv.transform_to_p8(T_R_M, float(L)))
        t = T_R_B.inverse().apply_points(np.asarray(o).reshape(-1)[:3])
        ip.append(np.degrees(np.arctan2(t[1], t[0])))
        op.append(np.degrees(np.arctan2(t[2], np.hypot(t[0], t[1]))))
    return np.array(ip), np.array(op)

K = np.array(kind)
m_arc = np.array([k == "arc" for k in kind])
m_dipz = np.array([k in ("dipZ", "dipZoff") for k in kind])
m_ins = np.array([k.startswith("ins_") for k in kind])

print(f"E0={E0:.0f} Pa    n={len(rows)}  (arc {m_arc.sum()}, dipZ {m_dipz.sum()}, ins {m_ins.sum()})")
print("E_s  msc  | RMS all  arc   dipZ   ins  | slope(arc)  offset")
best = None
for E_s in (1.0, 1.6, 2.5, 4.0, 6.0, 9.0):
    for msc in (0.4, 0.7, 1.0, 1.5):
        ad = build(E_s, msc)
        ip, op = evalm(ad)
        def rms(msk):
            return float(np.sqrt(np.mean((cam_in[msk] - ip[msk]) ** 2)))
        A = np.vstack([cam_in[m_arc], np.ones(m_arc.sum())]).T
        sl, off = np.linalg.lstsq(A, ip[m_arc], rcond=None)[0]
        rall = rms(np.ones(len(rows), bool))
        print(f"{E_s:<4} {msc:<4} | {rall:7.2f} {rms(m_arc):6.2f} {rms(m_dipz):6.2f} {rms(m_ins):6.2f} | {sl:8.2f}  {off:+7.2f}")
        if best is None or rall < best[0]:
            best = (rall, E_s, msc, ip, op)

rall, E_s, msc, ip, op = best
print(f"\nBEST E_s={E_s} (E={E0*E_s:.0f} Pa)  moment_scale={msc}  RMS={rall:.2f}")
print("\n label            kind      ins   cam_in  mod_in   cam_out mod_out")
order = sorted(range(len(rows)), key=lambda i: (kind[i], lab[i]))
for i in order:
    print(f"  {lab[i]:16s} {kind[i]:9s} {1e3*ins[i]:5.1f}  {cam_in[i]:+6.2f}  {ip[i]:+6.2f}   {cam_out[i]:+5.2f}  {op[i]:+5.2f}")

json.dump({"E_pa": E0 * E_s, "E_scale": E_s, "moment_scale": msc, "rms_all_deg": rall},
          open(BASE + "fit_all.json", "w"), indent=1)

# length-scaling check on the insertion sweep
print("\n--- insertion sweep: bend vs length ---")
mi = np.where(m_ins)[0]
mi = sorted(mi, key=lambda i: ins[i])
for i in mi:
    print(f"  {lab[i]:12s} L={1e3*ins[i]:5.1f}mm  cam_in {cam_in[i]:+6.2f}  mod_in {ip[i]:+6.2f}")
