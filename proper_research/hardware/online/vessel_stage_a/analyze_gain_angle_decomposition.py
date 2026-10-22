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


def longest_true_streak(bad: np.ndarray) -> int:
    """Longest run of consecutive True values in `bad` (raw tick-index order).
    A sub-threshold (invalid, ||m||<MIN_MOTION_M) tick is treated as `False`
    here (i.e. it breaks a streak) -- we have no directional information for
    it, so it cannot be counted as "bad", but silently skipping it instead of
    breaking the streak would understate how CONSECUTIVE the real controller
    ticks actually were. This is a deliberate, conservative convention."""
    best = cur = 0
    for b in bad:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def tail_metrics(theta_deg: np.ndarray, valid: np.ndarray, window_mask: np.ndarray | None = None) -> dict:
    """p50/p90/p95/p99/max, %>60, %>90, longest >90-streak for one
    (run, model, insertion-window) cell. `window_mask` further restricts
    which raw ticks are eligible (insertion-depth subsetting); streaks are
    computed over the RAW tick order within that window (not compressed),
    so a window boundary can break a streak that straddles it -- intentional,
    since a streak split across e.g. the 48mm boundary isn't really evidence
    about what happens inside the 56-65mm window specifically."""
    mask = valid if window_mask is None else (valid & window_mask)
    theta_in_window = np.where(window_mask, theta_deg, np.nan) if window_mask is not None else theta_deg
    bad90 = mask & (theta_deg > 90.0)
    bad90_windowed = bad90 if window_mask is None else (bad90 & window_mask)
    v = theta_in_window[mask] if window_mask is not None else theta_deg[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, p50=np.nan, p90=np.nan, p95=np.nan, p99=np.nan, max=np.nan,
                    pct_gt60=np.nan, pct_gt90=np.nan, longest_streak_gt90=0)
    return dict(
        n=int(v.size),
        p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)),
        p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
        max=float(np.max(v)),
        pct_gt60=float(100.0 * np.mean(v > 60.0)),
        pct_gt90=float(100.0 * np.mean(v > 90.0)),
        longest_streak_gt90=longest_true_streak(bad90_windowed if window_mask is not None else bad90),
    )


def s_wrong_scores(theta_deg: np.ndarray, m_norm_mm: np.ndarray, valid: np.ndarray,
                    window_mask: np.ndarray | None = None) -> tuple[float, float]:
    """S_wrong = sum(max(0,-cos theta)), S_wrong_w = sum(||m|| * max(0,-cos theta)).
    cos(theta) derived from the already-computed theta_deg (round-trips
    through arccos/cos to ~1e-10, negligible for this diagnostic) rather than
    recomputing the dot products from scratch."""
    mask = valid if window_mask is None else (valid & window_mask)
    th = theta_deg[mask]
    finite = np.isfinite(th)
    th = th[finite]
    if th.size == 0:
        return 0.0, 0.0
    cos_t = np.cos(np.radians(th))
    wrong = np.maximum(0.0, -cos_t)
    m = m_norm_mm[mask][finite]
    return float(np.sum(wrong)), float(np.sum(m * wrong))


def insertion_mm_for_run(run: dict) -> np.ndarray:
    """Insertion depth (mm) at the START of each per-tick diff interval --
    z_meas[:,6] is insertion in metres (state7's last component); the diffs
    p/m/theta are indexed like z_meas[:-1] (see decompose_run)."""
    return np.asarray(run["z_meas"], dtype=float)[:-1, 6] * 1e3


def run_outcome(stop_reason: str) -> str:
    return "success" if stop_reason == "path_complete" else "fail"


def print_tail_table(title: str, keys: list, decomposed: dict, replay: dict,
                      window: tuple[float, float] | None = None, rep_label_prefix: str = "") -> None:
    print(f"\n  -- {title} --")
    if window is not None:
        print(f"     insertion window: {window[0]:.0f}-{window[1]:.0f}mm")
    header = (f"{'rep':<14}{'outcome':<9}{'model':<5}{'n':>5}  {'p50':>6} {'p90':>6} {'p95':>6} "
              f"{'p99':>6} {'max':>7}  {'%>60':>6} {'%>90':>6}  {'streak':>7}  {'S_wrong':>9} {'S_wrong_w(mm)':>14}")
    print("     " + header)
    for key in keys:
        cond, rep = key
        d = decomposed[key]
        outcome = run_outcome(d["stop_reason"])
        run = replay[key]
        ins_mm = insertion_mm_for_run(run)
        wmask = None
        if window is not None:
            wmask = (ins_mm >= window[0]) & (ins_mm <= window[1])
        for model_name in ("C", "NC"):
            theta = d[model_name]["theta_deg"]
            valid = d["valid"]
            tm = tail_metrics(theta, valid, wmask)
            sw, sww = s_wrong_scores(theta, d["m_norm_mm"], valid, wmask)
            label = f"{rep_label_prefix}{rep}"
            print(f"     {label:<14}{outcome:<9}{model_name:<5}{tm['n']:>5}  "
                  f"{tm['p50']:>6.1f} {tm['p90']:>6.1f} {tm['p95']:>6.1f} {tm['p99']:>6.1f} "
                  f"{tm['max']:>7.1f}  {tm['pct_gt60']:>6.1f} {tm['pct_gt90']:>6.1f}  "
                  f"{tm['longest_streak_gt90']:>7d}  {sw:>9.2f} {sww:>14.3f}")


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

    # ---- tail / persistence analysis (2026-10-01 user follow-up) ----
    # Question: is NC failure caused by a generally worse Jacobian (higher
    # median/typical error), or by a small number of catastrophic predictions
    # -- and specifically, does PERSISTENCE (consecutive >90 deg ticks) rather
    # than peak severity (raw max) distinguish failed from successful runs?
    print("\n" + "=" * 100)
    print("TAIL + PERSISTENCE ANALYSIS (p90/p95/p99, max, %>60, %>90, longest >90deg streak, S_wrong)")
    print("=" * 100)

    nc_keys_ordered = [("nocontact", r) for r in range(1, 6)]
    contact_keys_ordered = [("contact", r) for r in range(1, 6)]

    print_tail_table("Whole-run, all 5 no-contact reps (real outcomes, not assumed)",
                      nc_keys_ordered, decomposed, replay, window=None, rep_label_prefix="NC")
    print_tail_table("Whole-run, all 5 contact reps (for comparison)",
                      contact_keys_ordered, decomposed, replay, window=None, rep_label_prefix="C")
    print_tail_table("Insertion window 48-65mm, no-contact reps",
                      nc_keys_ordered, decomposed, replay, window=(48.0, 65.0), rep_label_prefix="NC")
    print_tail_table("Insertion window 56-65mm (critical sub-window), no-contact reps",
                      nc_keys_ordered, decomposed, replay, window=(56.0, 65.0), rep_label_prefix="NC")

    # persist tail metrics alongside the existing per-tick decomposition
    for key in decomposed:
        run = replay[key]
        ins_mm = insertion_mm_for_run(run)
        d = decomposed[key]
        d["tail"] = {}
        for win_name, win in (("whole_run", None), ("ins_48_65mm", (48.0, 65.0)), ("ins_56_65mm", (56.0, 65.0))):
            wmask = None if win is None else ((ins_mm >= win[0]) & (ins_mm <= win[1]))
            d["tail"][win_name] = {}
            for model_name in ("C", "NC"):
                theta = d[model_name]["theta_deg"]
                tm = tail_metrics(theta, d["valid"], wmask)
                sw, sww = s_wrong_scores(theta, d["m_norm_mm"], d["valid"], wmask)
                tm["S_wrong"] = sw
                tm["S_wrong_w_mm"] = sww
                d["tail"][win_name][model_name] = tm
    with open(OUT_PKL, "wb") as f:
        pickle.dump(decomposed, f)
    print(f"\n(re-wrote {OUT_PKL} with tail metrics added under decomposed[key]['tail'])")

    # explicit persistence-vs-extremity verdict for NC, whole run
    print("\n" + "-" * 100)
    print("PERSISTENCE-VS-EXTREMITY VERDICT (no-contact model, whole run)")
    print("-" * 100)
    for key in nc_keys_ordered:
        d = decomposed[key]
        t = d["tail"]["whole_run"]["NC"]
        outcome = run_outcome(d["stop_reason"])
        print(f"  NC{key[1]} ({outcome:7s}): p95={t['p95']:6.1f}  p99={t['p99']:6.1f}  "
              f"max={t['max']:6.1f}  longest>90streak={t['longest_streak_gt90']:3d}")


if __name__ == "__main__":
    main()
