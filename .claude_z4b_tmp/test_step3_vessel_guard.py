"""Z4b step 3: verify the vessel beam-base guard added to build_or_load_schedule
does not alter the returned schedule array in any branch (no-metadata/backfill,
matching-metadata/PASSED, mismatched-metadata/WARNING) -- exercises the REAL,
modified function against a copy of a real cached vessel schedule, not a
reimplementation."""
import sys, os, shutil, json
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")
import numpy as np

import proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel as m

REAL_CACHE = (
    "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis/plans/"
    "vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_"
    "dipolealigned_2026-10-20/mpc_schedule_contact.npy"
)
assert os.path.exists(REAL_CACHE), f"real cache file not found: {REAL_CACHE}"
real_schedule = np.load(REAL_CACHE)
print(f"[setup] real cache: {REAL_CACHE}, shape={real_schedule.shape}")

tmp_dir = "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis/.claude_z4b_tmp/guard_test"
os.makedirs(tmp_dir, exist_ok=True)
tmp_cache = os.path.join(tmp_dir, "schedule_copy.npy")
tmp_meta = tmp_cache + ".meta.json"
for p in (tmp_cache, tmp_meta):
    if os.path.exists(p):
        os.remove(p)
shutil.copy(REAL_CACHE, tmp_cache)

current_beam_base = np.asarray(m._recalibrated_make_initial_poses()[0], dtype=float)[:3]
print(f"[setup] current beam base (xyz) = {current_beam_base.tolist()}")

# --- Case 1: no metadata file yet -> loads unchanged, backfills metadata --
assert not os.path.exists(tmp_meta)
out1 = m.build_or_load_schedule(
    plan_dir="unused", cache_path=tmp_cache,
    lumen_file="unused", insertion_max_mm=0.0, contact=True,
)
assert np.array_equal(out1, real_schedule), "case 1: returned array differs from the real cache!"
assert os.path.exists(tmp_meta), "case 1: metadata was not backfilled"
with open(tmp_meta) as f:
    backfilled = json.load(f)["beam_base_xyz_m"]
assert np.allclose(backfilled, current_beam_base.tolist()), "case 1: backfilled beam base wrong"
print("[OK] case 1 (no metadata): returned array is byte-identical to the real cache; "
      "metadata backfilled correctly")

# --- Case 2: matching metadata -> PASSED, still unchanged -----------------
out2 = m.build_or_load_schedule(
    plan_dir="unused", cache_path=tmp_cache,
    lumen_file="unused", insertion_max_mm=0.0, contact=True,
)
assert np.array_equal(out2, real_schedule), "case 2: returned array differs from the real cache!"
print("[OK] case 2 (matching metadata): returned array still byte-identical to the real cache")

# --- Case 3: mismatched metadata -> WARNING, but STILL returns the cache --
with open(tmp_meta, "w") as f:
    json.dump({"beam_base_xyz_m": (current_beam_base + 0.05).tolist()}, f)  # 50mm off
out3 = m.build_or_load_schedule(
    plan_dir="unused", cache_path=tmp_cache,
    lumen_file="unused", insertion_max_mm=0.0, contact=True,
)
assert np.array_equal(out3, real_schedule), "case 3: returned array differs from the real cache!"
print("[OK] case 3 (mismatched metadata, 50mm off): WARNING path still returns the "
      "byte-identical real cache array -- no solve change, no silent data loss")

shutil.rmtree(tmp_dir)
print("[PASS] Z4b step 3 vessel-guard test")
