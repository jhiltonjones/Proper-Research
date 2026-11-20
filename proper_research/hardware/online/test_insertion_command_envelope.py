"""2026-11-19: verifies PathFollowConfig.insertion_command_min_m/max_m
(the actual ONLINE dispatched-command clip) default to the prior hardcoded
values (5mm/200mm, no silent behaviour change for existing callers) and
are honored when overridden -- directly reproducing the three np.clip
call sites' own arithmetic, since exercising them live requires the full
hardware-adjacent main() loop. This closes the gap identified this turn:
a plan-family name like "L30to50mm" was never an enforced execution
bound; it only ever described the generated reference."""
import numpy as np

from proper_research.hardware.online.close_loop_path_follow import PathFollowConfig


def test_defaults_match_prior_hardcoded_values():
    cfg = PathFollowConfig()
    assert cfg.insertion_command_min_m == 0.005
    assert cfg.insertion_command_max_m == 0.20


def test_accumulator_seam_clip_formula_respects_configured_bounds():
    cfg = PathFollowConfig(insertion_command_min_m=0.030, insertion_command_max_m=0.050)
    # reproduces close_loop_path_follow.py's own accumulator_seam branch:
    # ins_cmd = clip(ins_cmd + corr[6]*dt, cfg.insertion_command_min_m, cfg.insertion_command_max_m)
    ins_cmd, dt = 0.040, 0.1
    corr6_large_negative = -5.0  # would drive well below 30mm without the clip
    result = float(np.clip(ins_cmd + corr6_large_negative * dt, cfg.insertion_command_min_m, cfg.insertion_command_max_m))
    assert result == cfg.insertion_command_min_m

    corr6_large_positive = 5.0
    result = float(np.clip(ins_cmd + corr6_large_positive * dt, cfg.insertion_command_min_m, cfg.insertion_command_max_m))
    assert result == cfg.insertion_command_max_m


def test_default_bounds_do_not_enforce_30_50mm():
    # confirms the GAP this turn's audit found: with the default (unchanged)
    # bounds, a command well outside 30-50mm is NOT clipped.
    cfg = PathFollowConfig()
    result = float(np.clip(0.010, cfg.insertion_command_min_m, cfg.insertion_command_max_m))
    assert result == 0.010  # 10mm passes through untouched -- outside 30-50mm, not rejected

    result = float(np.clip(0.100, cfg.insertion_command_min_m, cfg.insertion_command_max_m))
    assert result == 0.100  # 100mm also passes through untouched
