import pickle
import numpy as np

with open("../data/authority_abc_replay.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]
MODES = data["modes"]
LAMBDAS = data["lambdas"]
N = len(results)
print(f"n_ticks={N}, modes={MODES}\n")


def collect(field, mode, sub=None):
    vals = []
    for r in results:
        v = r["per_mode"][mode][field]
        if sub is not None:
            v = v[sub]
        if v is not None:
            vals.append(v)
    return np.array(vals, dtype=float)


print("=" * 100)
print("TASK TRACKING: E_chosen (mm) at +3/+5/+10 prediction horizon, mean +/- sd, vs mode A")
print("=" * 100)
for h in (3, 5, 10):
    row = f"  h=+{h:2d}: "
    a_vals = collect("e_chosen", "A_reference_nullspace", sub=h)
    row += f"A={a_vals.mean():.4f}+/-{a_vals.std():.4f}  "
    b_vals = collect("e_chosen", "B_none", sub=h)
    row += f"B={b_vals.mean():.4f}+/-{b_vals.std():.4f}  "
    for lam in LAMBDAS:
        mode = f"C_manip_lam{lam:g}"
        c_vals = collect("e_chosen", mode, sub=h)
        row += f"C(l={lam:g})={c_vals.mean():.4f}  "
    print(row)

print()
print("=" * 100)
print("AUTHORITY: phi_true(z1_pred) mean, and Delta-phi_true vs mode B baseline (same tick)")
print("=" * 100)
phi_by_mode = {m: collect("phi_true1", m) for m in MODES}
for m in MODES:
    print(f"  {m:28s}: phi_true mean={phi_by_mode[m].mean():.4f}  sd={phi_by_mode[m].std():.4f}")

# paired delta vs B (same ticks, same order)
print("\n  Paired Delta phi_true = phi_true(mode) - phi_true(B_none), per tick, mean +/- sd:")
b_phi = np.array([r["per_mode"]["B_none"]["phi_true1"] for r in results])
a_phi = np.array([r["per_mode"]["A_reference_nullspace"]["phi_true1"] for r in results])
print(f"    A - B: mean={np.mean(a_phi-b_phi):+.4f}  sd={np.std(a_phi-b_phi):.4f}  frac>0={np.mean(a_phi>b_phi):.2f}")
for lam in LAMBDAS:
    c_phi = np.array([r["per_mode"][f"C_manip_lam{lam:g}"]["phi_true1"] for r in results])
    d = c_phi - b_phi
    print(f"    C(l={lam:g}) - B: mean={d.mean():+.4f}  sd={d.std():.4f}  frac>0={np.mean(d>0):.2f}")

print()
print("=" * 100)
print("SECONDARY-DIRECTION DEPARTURE: ||P_N * dz~_1|| (nullspace departure at first predicted stage)")
print("=" * 100)
for m in MODES:
    v = collect("PN_dz1", m)
    print(f"  {m:28s}: mean={v.mean():.4f}  sd={v.std():.4f}  max={v.max():.4f}")

print()
print("=" * 100)
print("LINEAR-SURROGATE VALIDITY: r_phi = dphi_true/dphi_lin  (relative to B_none, i.e. isolating the")
print("manip term's own effect) -- fraction with WRONG SIGN (dphi_lin>0 but dphi_true<=0)")
print("=" * 100)
for lam in LAMBDAS:
    mode = f"C_manip_lam{lam:g}"
    wrong_sign = 0
    total = 0
    dphi_lin_pos_true_neg = []
    for r in results:
        c = r["per_mode"][mode]
        b = r["per_mode"]["B_none"]
        if c["phi_true1"] is None or b["phi_true1"] is None:
            continue
        dphi_true_rel = c["phi_true1"] - b["phi_true1"]  # true effect of adding manip term vs none
        dphi_lin_rel = c["dphi_lin"] - b["dphi_lin"]
        total += 1
        if dphi_lin_rel > 0 and dphi_true_rel <= 0:
            wrong_sign += 1
            dphi_lin_pos_true_neg.append((r["ref_index"], dphi_lin_rel, dphi_true_rel))
    print(f"  lambda={lam:5g}: wrong-sign fraction = {wrong_sign}/{total} = {wrong_sign/total:.2f}")

print()
print("=" * 100)
print("CONTROL EFFORT (fraction of velocity limit used): joints and insertion channel")
print("=" * 100)
for m in MODES:
    ej = collect("effort_joints", m)
    ei = collect("effort_insertion", m)
    print(f"  {m:28s}: joints={ej.mean():.3f}  insertion={ei.mean():.3f}")

print()
print("=" * 100)
print("DISTANCE TO BOUNDS (min over ticks -- how close does any mode get to a hard limit)")
print("=" * 100)
for m in MODES:
    ds = collect("dist_state", m)
    dv = collect("dist_vel", m)
    print(f"  {m:28s}: min dist_state={ds.min():.5f}  min dist_vel={dv.min():.5f}")

print()
print("=" * 100)
print("CONFIGURATION DEPARTURE FROM REFERENCE-LINEARIZED POINT: ||(z1_pred - z_lin1)/S_z||")
print("=" * 100)
for m in MODES:
    v = collect("config_departure", m)
    print(f"  {m:28s}: mean={v.mean():.4f}  sd={v.std():.4f}  max={v.max():.4f}")

print()
print("=" * 100)
print("BREAKDOWN BY PATH REGION: ordinary (norm_gN0 in [0.1,0.3]) vs opposition-tail (norm_gN0<0.08)")
print("using E_chosen(+5) and Delta-phi_true(C_lam10 - B) as the two headline numbers")
print("=" * 100)
gN0 = np.array([r["norm_gN0"] for r in results])
ordinary_mask = (gN0 >= 0.10) & (gN0 <= 0.30)
tail_mask = gN0 < 0.08
print(f"  n_ordinary={ordinary_mask.sum()}  n_tail={tail_mask.sum()}")
for label, mask in (("ordinary", ordinary_mask), ("tail", tail_mask)):
    if mask.sum() == 0:
        continue
    a_e5 = collect("e_chosen", "A_reference_nullspace", sub=5)[mask]
    b_e5 = collect("e_chosen", "B_none", sub=5)[mask]
    c10_e5 = collect("e_chosen", "C_manip_lam10", sub=5)[mask]
    c10_phi = np.array([r["per_mode"]["C_manip_lam10"]["phi_true1"] for r in results])[mask]
    b_phi_m = np.array([r["per_mode"]["B_none"]["phi_true1"] for r in results])[mask]
    print(f"  [{label}] E_chosen+5(mm): A={a_e5.mean():.3f} B={b_e5.mean():.3f} C(l=10)={c10_e5.mean():.3f}   "
          f"Delta-phi_true(C10-B)={np.mean(c10_phi-b_phi_m):+.4f}")

with open("../data/authority_abc_summary.pkl", "wb") as f:
    pickle.dump(dict(phi_by_mode={m: phi_by_mode[m].tolist() for m in MODES}), f)
