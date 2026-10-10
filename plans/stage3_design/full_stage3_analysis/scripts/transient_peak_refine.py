import sys, time, contextlib, io
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
import numpy as np, pandas as pd
import manifest, loader, live_jac
from proper_research.simulation.simulations.controller_factory_joint_space import _pose7_from_transform

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
runs = sorted([rm for rm in manifest.RUNS if rm["group"]=="closedloop" and rm["condition"]=="mpc_C" and rm["radius_intended_mm"]==210], key=lambda r:r["rep"])

live_jac.get_context()
prov = live_jac.get_provider(True)
model = prov.model

rows = []
t0=time.time()
for rm in runs:
    out, meta = loader.enrich_run(rm)
    s = out["s_ref_mm"]; e = out["error_norm_mm"]
    mask = (s>=58)&(s<=62)
    idx = np.where(mask)[0]
    peak_k = idx[np.argmax(e[idx])]
    lo = max(peak_k-3, 0); hi = min(peak_k+4, len(s)-1)
    for k in range(lo, hi):
        state7 = np.concatenate([out["q_meas_rad"][k], [out["insertion_length_m"][k]]])
        T_R_M = prov.adapter.magnet_transform(state7)
        p7 = _pose7_from_transform(T_R_M, float(state7[6]))
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.solve(p7, commit=False, reuse_cache=False)
        parts = (result.info or {}).get("parts", {})
        gap_nodes = parts.get("gap_nodes"); F_nodes = parts.get("F_nodes")
        row = dict(rep=rm["rep"], step=int(out["step"][k]), s_mm=float(s[k]),
                   e_track_mm=float(e[k]), is_peak=bool(k==peak_k),
                   gap_min_m=float(np.min(gap_nodes)) if gap_nodes is not None else np.nan,
                   contact_force_norm=float(np.linalg.norm(F_nodes)) if F_nodes is not None else np.nan,
                   W_cf=float(parts.get("W_cf", np.nan)))
        rows.append(row)
        print(f"rep{rm['rep']} s={s[k]:.2f} e={e[k]:.3f} peak={k==peak_k} gap_min={row['gap_min_m']*1e3:.3f}mm F={row['contact_force_norm']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/tables/transient_peak_refine.csv", index=False)
print("DONE", time.time()-t0)
