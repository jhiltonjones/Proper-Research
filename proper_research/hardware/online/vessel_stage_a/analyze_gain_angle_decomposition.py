"""Gain/angle decomposition of the dual-Jacobian replay (2026-10-01).

Extends the existing Section-3 vector prediction error

    e = ||p - m||,   p = J @ du,   m = dx_meas,   du = dstate

(computed once per tick for both the contact-aware (C) and no-contact (NC)
Jacobians, against the SAME measured state trajectory -- see
dual_jacobian_replay.py) into a gain/direction decomposition, per the
user's 2026-10-01 request. e mixes two different failure modes:

    gain error      -- did the model predict too much/little MOTION?
    direction error -- did it predict motion in the wrong DIRECTION?

Per tick, for each model X in {C, NC}:

    g_X        = ||p_X|| / ||m||                      overall gain ratio
    theta_X    = angle(p_X, m), degrees                direction error
    g_par_X    = (m_hat . p_X) / ||m||                 gain along the REAL
                                                        direction of motion
    e_perp_X   = ||p_X - (m_hat . p_X) m_hat||          wrong-direction
                                                        component magnitude

and the wall-normal/tangential-plane authority check (the most physically
direct test of "does the no-contact model believe it has wall-normal
mobility the real system doesn't have"):

    g_n_real = |n . m|   / ||du||      g_n_X = |n . p_X| / ||du||
    g_t_real = ||m_perp||/ ||du||      g_t_X = ||p_X_perp|| / ||du||

Noise floor: ticks with ||m|| below MIN_MOTION_M are excluded from every
gain/angle/g_parallel/e_perp/g_n/g_t computation (division by a near-zero
measured motion blows up the ratio and the angle becomes noise-dominated).
MIN_MOTION_M = kf_measurement_noise_std_m (0.3mm, read from this project's
own run configs -- summary.json's "kf_measurement_noise_std_m"); see the
constant's own comment below for why a stricter 3x floor was tried and
discarded.

IMPORTANT finding reused from this analysis: the existing 0.5mm
CONTACT_BAND_M tag (dual_jacobian_replay.py) marks EVERY SINGLE TICK of
ALL 10 runs as "in contact" (c_min_mm ranges +0.08 to -0.18mm throughout --
confirmed not a stale/buggy replay, see c_min_mm's own variation). So the
requested free-space-vs-near-wall split is EMPTY under that definition --
this IS itself evidence for the "beam stays continuously close to/against
the wall" finding already established elsewhere in this project. In its
place this script reports a clearance-based split (TOUCHING: c_min<=0mm
vs CLOSE: 0<c_min<=0.5mm) and a continuous Spearman correlation of each
metric against c_min_mm, which is the physically meaningful substitute.

Reuses the already-computed dual_jacobian_replay_results.pkl (z_meas,
tip_meas, JC, JNC, normal_c, c_min_mm, stop_reason per run) -- no adapter
rebuild, no hardware/robot access, pure numpy post-processing.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPLAY_PKL = Path("/home/jack/.claude/jobs/043c0b8b/tmp/dual_jacobian_replay_results.pkl")
OUT_PKL = Path("/home/jack/.claude/jobs/043c0b8b/tmp/gain_angle_decomposition_results.pkl")

MIN_MOTION_M = 0.3e-3  # 1x the vision pipeline's own kf_measurement_noise_std_m (0.3mm,
# summary.json's "kf_measurement_noise_std_m"). A 3x floor (0.9mm) was tried first and
# discarded: at this controller's 10Hz tick rate and slow magnetic-steering authority, the
# median per-tick motion is only ~0.17mm (p90=0.42mm, p99=0.78mm) -- a 3-sigma floor left
# only 31/5088 ticks total (and ZERO from the failed no-contact reps specifically), nowhere
# near enough for the per-split statistics below. 1x std keeps ~1200/5088 ticks (70-166 per
# run, well-balanced across failed/successful and contact/no-contact), at the cost of
# somewhat noisier individual-tick angle/gain values -- acceptable since every number below
# is itself an aggregate (mean/median/IQR) over 70+ ticks, not a single-tick read.


def decompose_run(run: dict) -> dict:
    z_meas = run["z_meas"]
    tip_meas = run["tip_meas"]
    JC = run["JC"]
    JNC = run["JNC"]
    normal_c = run["normal_c"][:-1]
    c_min_mm = run["c_min_mm"][:-1]

    dstate = np.diff(z_meas, axis=0)          # (n-1,7), "du"
    m = np.diff(tip_meas, axis=0)              # (n-1,3), "m" = dx_meas
    m_norm = np.linalg.norm(m, axis=1)
    du_norm = np.linalg.norm(dstate, axis=1)

    valid = m_norm >= MIN_MOTION_M

    p_C = np.einsum("nij,nj->ni", JC[:-1], dstate)
    p_NC = np.einsum("nij,nj->ni", JNC[:-1], dstate)

    def per_model(p):
        p_norm = np.linalg.norm(p, axis=1)
        e_vec = np.linalg.norm(p - m, axis=1)

        g = np.full(len(p), np.nan)
        theta_deg = np.full(len(p), np.nan)
        g_par = np.full(len(p), np.nan)
        e_perp = np.full(len(p), np.nan)

        g[valid] = p_norm[valid] / m_norm[valid]
        cos_t = np.einsum("ni,ni->n", p, m)[valid] / (p_norm[valid] * m_norm[valid])
        cos_t = np.clip(cos_t, -1.0, 1.0)
        theta_deg[valid] = np.degrees(np.arccos(cos_t))

        m_hat = m[valid] / m_norm[valid, None]
        proj = np.einsum("ni,ni->n", m_hat, p[valid])
        g_par[valid] = proj / m_norm[valid]
        perp_vec = p[valid] - proj[:, None] * m_hat
        e_perp[valid] = np.linalg.norm(perp_vec, axis=1)

        # wall-normal / tangential authority, normalized by ||du|| (not ||m||)
        du_valid = du_norm > 1e-12
        g_n = np.full(len(p), np.nan)
        g_t = np.full(len(p), np.nan)
        proj_n_all = np.einsum("ni,ni->n", normal_c, p)
        g_n[du_valid] = np.abs(proj_n_all[du_valid]) / du_norm[du_valid]
        p_perp_wall = p - proj_n_all[:, None] * normal_c
        g_t[du_valid] = np.linalg.norm(p_perp_wall[du_valid], axis=1) / du_norm[du_valid]

        return dict(e_vector_mm=e_vec * 1e3, g=g, theta_deg=theta_deg,
                     g_parallel=g_par, e_perp_mm=e_perp * 1e3, g_n=g_n, g_t=g_t)

    out_C = per_model(p_C)
    out_NC = per_model(p_NC)

    # real (measured) wall-normal/tangential "gain" -- what the beam ACTUALLY
    # did, normalized the same way, for direct comparison against g_n_C/g_n_NC
    du_valid = du_norm > 1e-12
    g_n_real = np.full(len(m), np.nan)
    g_t_real = np.full(len(m), np.nan)
    proj_n_real = np.einsum("ni,ni->n", normal_c, m)
    g_n_real[du_valid] = np.abs(proj_n_real[du_valid]) / du_norm[du_valid]
    m_perp_wall = m - proj_n_real[:, None] * normal_c
    g_t_real[du_valid] = np.linalg.norm(m_perp_wall[du_valid], axis=1) / du_norm[du_valid]

    return dict(
        valid=valid, c_min_mm=c_min_mm, m_norm_mm=m_norm * 1e3,
        g_n_real=g_n_real, g_t_real=g_t_real,
        C=out_C, NC=out_NC,
        stop_reason=run["stop_reason"],
    )


def summarize(values: np.ndarray) -> str:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return "n=0"
    return (f"n={v.size:4d}  mean={np.mean(v):7.3f}  median={np.median(v):7.3f}  "
            f"std={np.std(v):7.3f}  IQR=[{np.percentile(v,25):7.3f},{np.percentile(v,75):7.3f}]")


def print_split(label: str, mask: np.ndarray, dec: dict) -> None:
    print(f"\n  -- {label} (n_ticks={int(mask.sum())}) --")
    for model_name, out in (("C", dec["C"]), ("NC", dec["NC"])):
        print(f"    [{model_name}] g        : {summarize(out['g'][mask])}")
        print(f"    [{model_name}] theta_deg: {summarize(out['theta_deg'][mask])}")
        print(f"    [{model_name}] g_par    : {summarize(out['g_parallel'][mask])}")
        print(f"    [{model_name}] e_perp_mm: {summarize(out['e_perp_mm'][mask])}")
        print(f"    [{model_name}] e_vec_mm : {summarize(out['e_vector_mm'][mask])}")
    print(f"    [real] g_n: {summarize(dec['g_n_real'][mask])}   "
          f"[C] g_n: {summarize(dec['C']['g_n'][mask])}   "
          f"[NC] g_n: {summarize(dec['NC']['g_n'][mask])}")
    print(f"    [real] g_t: {summarize(dec['g_t_real'][mask])}   "
          f"[C] g_t: {summarize(dec['C']['g_t'][mask])}   "
          f"[NC] g_t: {summarize(dec['NC']['g_t'][mask])}")


def main() -> None:
    with open(REPLAY_PKL, "rb") as f:
        replay = pickle.load(f)

    decomposed = {key: decompose_run(run) for key, run in replay.items()}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(decomposed, f)
    print(f"wrote {OUT_PKL}\n")

    # ---- per-run summary (contact vs no-contact, split into failed/successful) ----
    nc_failed_reps = [rep for (cond, rep), d in decomposed.items()
                       if cond == "nocontact" and d["stop_reason"] != "path_complete"]
    nc_success_reps = [rep for (cond, rep), d in decomposed.items()
                        if cond == "nocontact" and d["stop_reason"] == "path_complete"]
    print(f"no-contact reps: failed={nc_failed_reps}  successful={nc_success_reps}")

    def pooled(keys):
        pools = {"C": {k: [] for k in ("g", "theta_deg", "g_parallel", "e_perp_mm", "e_vector_mm", "g_n", "g_t")},
                 "NC": {k: [] for k in ("g", "theta_deg", "g_parallel", "e_perp_mm", "e_vector_mm", "g_n", "g_t")},
                 "g_n_real": [], "g_t_real": [], "c_min_mm": [], "valid": []}
        for key in keys:
            d = decomposed[key]
            pools["valid"].append(d["valid"])
            pools["c_min_mm"].append(d["c_min_mm"])
            pools["g_n_real"].append(d["g_n_real"])
            pools["g_t_real"].append(d["g_t_real"])
            for m in ("C", "NC"):
                for k in pools[m]:
                    pools[m][k].append(d[m][k])
        out = {"valid": np.concatenate(pools["valid"]), "c_min_mm": np.concatenate(pools["c_min_mm"]),
               "g_n_real": np.concatenate(pools["g_n_real"]), "g_t_real": np.concatenate(pools["g_t_real"])}
        out["C"] = {k: np.concatenate(v) for k, v in pools["C"].items()}
        out["NC"] = {k: np.concatenate(v) for k, v in pools["NC"].items()}
        return out

    all_contact_keys = [k for k in decomposed if k[0] == "contact"]
    all_nc_keys = [k for k in decomposed if k[0] == "nocontact"]
    nc_failed_keys = [("nocontact", r) for r in nc_failed_reps]
    nc_success_keys = [("nocontact", r) for r in nc_success_reps]

    print("\n" + "=" * 100)
    print("POOLED: all contact-condition reps (5 reps, every tick in-contact by 0.5mm band)")
    print("=" * 100)
    pc = pooled(all_contact_keys)
    print_split("all ticks", pc["valid"] & np.isfinite(pc["c_min_mm"]), pc)

    print("\n" + "=" * 100)
    print("POOLED: all no-contact-condition reps (5 reps)")
    print("=" * 100)
    pnc = pooled(all_nc_keys)
    print_split("all ticks", pnc["valid"], pnc)

    print("\n" + "=" * 100)
    print("CLEARANCE SPLIT (substitute for free-space/near-wall -- EVERY tick is <=0.5mm from")
    print("the wall in this data, so splitting on c_min_mm sign/magnitude instead)")
    print("=" * 100)
    touching = pnc["c_min_mm"] <= 0.0
    close = (pnc["c_min_mm"] > 0.0) & (pnc["c_min_mm"] <= 0.5)
    print_split("no-contact reps, TOUCHING (c_min<=0mm)", pnc["valid"] & touching, pnc)
    print_split("no-contact reps, CLOSE (0<c_min<=0.5mm)", pnc["valid"] & close, pnc)

    print("\n" + "=" * 100)
    print("FAILED vs SUCCESSFUL no-contact reps")
    print("=" * 100)
    pf = pooled(nc_failed_keys)
    ps = pooled(nc_success_keys)
    print_split("FAILED no-contact reps", pf["valid"], pf)
    print_split("SUCCESSFUL no-contact reps", ps["valid"], ps)

    print("\n" + "=" * 100)
    print("Spearman correlation of each metric against c_min_mm (continuous clearance), pooled no-contact")
    print("(negative correlation of e.g. g_NC with c_min_mm means: tighter clearance -> WORSE gain error)")
    print("=" * 100)
    v = pnc["valid"]
    cmin = pnc["c_min_mm"][v]
    for model_name in ("C", "NC"):
        for metric in ("g", "theta_deg", "e_perp_mm", "g_n", "g_t"):
            x = pnc[model_name][metric][v]
            ok = np.isfinite(x) & np.isfinite(cmin)
            if ok.sum() < 10:
                continue
            rho, p = spearmanr(cmin[ok], x[ok])
            print(f"  [{model_name}] {metric:12s} vs c_min_mm: rho={rho:+.3f}  p={p:.2e}  (n={ok.sum()})")

    # wall-normal authority gap specifically for failed no-contact reps
    print("\n" + "=" * 100)
    print("Wall-normal authority: g_n_real vs g_n_C vs g_n_NC -- FAILED no-contact reps only")
    print("hypothesis: g_n_real ~ g_n_C  but  g_n_NC > g_n_real  (NC falsely believes wall-normal authority)")
    print("=" * 100)
    v = pf["valid"]
    print(f"  g_n_real : {summarize(pf['g_n_real'][v])}")
    print(f"  g_n_C    : {summarize(pf['C']['g_n'][v])}")
    print(f"  g_n_NC   : {summarize(pf['NC']['g_n'][v])}")
    excess_NC = pf["NC"]["g_n"][v] - pf["g_n_real"][v]
    excess_C = pf["C"]["g_n"][v] - pf["g_n_real"][v]
    print(f"  excess (g_n_NC - g_n_real): {summarize(excess_NC)}")
    print(f"  excess (g_n_C  - g_n_real): {summarize(excess_C)}")


if __name__ == "__main__":
    main()
