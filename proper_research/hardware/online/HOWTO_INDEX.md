# Index: which script runs what (vessel vs shape, MPC vs INV-7)

> **Campaign 2 (2026-11-11):** see `HOWTO_CAMPAIGN2.md` in this
> same directory for the current campaign's exact commands and
> protocol (plan B' and trap paths deferred/out of scope for this
> campaign). This index's own runner-choice ambiguity (below) is
> resolved there with a best-justified, author-to-confirm choice.


Written 2026-11-09, after an audit against the current code (not the older
per-package HOWTOs, which predate several fixes made this session). Read
this first if you're not sure which script to invoke; it links to the
fuller per-package docs for everything else.

## The 4-row answer

| Portion | Controller | Offline planning script | Online runner script | Uses `close_loop_path_follow.py`? |
|---|---|---|---|---|
| Vessel | MPC | `vessel_stage_a/build_vessel_plan.py` | `vessel_stage_a/run_mpc_delay_aware_vessel.py` | Yes (`import ... as pf`, confirmed line 138) |
| Vessel | INV-7 (inverse-Jacobian) | `vessel_stage_a/build_vessel_plan.py` | `vessel_stage_a/run_inverse_jacobian_online_vessel.py` | Yes (confirmed line 93) |
| Shape (rectangle/triangle) | MPC | `proper_research/planning/plan_shape_path.py` | `rectangle_stage_a/run_mpc_delay_aware_qn_zero.py` **or** `rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py` -- see "Open ambiguity" below, the real campaign logs don't record which | Yes (confirmed lines 54/83 respectively) |
| Shape (rectangle/triangle) | INV-7 (inverse-Jacobian) | `proper_research/planning/plan_shape_path.py` | `rectangle_stage_a/run_inv_7dof_delay_aware.py` | Yes (confirmed line 27) |

`close_loop_path_follow.py` is never run directly -- it's a shared library
module (scheduler, accumulator, delay handling, monitors, logging) that
every online runner script above imports and drives. Changes to it (this
session: the background-streaming fix, the new TCP-box-in-QP wiring for
shapes) take effect in whichever runner script is actually invoked, vessel
and shape alike.

## Open ambiguity, found during this audit -- a real gap, not resolved here

The rectangle-shape MPC family has at least 4 scripts in
`rectangle_stage_a/`: `run_mpc_delay_aware.py` (the plain baseline),
`run_mpc_delay_aware_qn_zero.py`, `run_mpc_delay_aware_insertion_anchor.py`,
`run_mpc_delay_aware_nullq_r700.py` -- the package's own `README.md` calls
the latter three "frozen ablation variants... each isolates exactly one
change for a specific recorded result." Each sets a DIFFERENT internal
factory kwarg (`qn_zero.py:209` passes `controller_kind="exact_qn0"`,
`insertion_anchor.py:259` passes `controller_kind="nullspace_r700"`) -- but
ALL of them, including the plain baseline, set the SAME generic
`cfg.controller_kind = "mpc_delay_aware"` (confirmed at `run_mpc_delay_aware.py:188`,
`qn_zero.py:450`, `insertion_anchor.py:495`), and it's this generic string,
not the inner factory kwarg, that gets logged into `summary.json`'s
`config` block. Checked two real campaign logs directly
(`rectangle_rerun1023_mpc_SJ_rep1_20261023T125622Z`,
`..._mpc_N1_rep1_20261023T171842Z`): both show `controller_kind:
"mpc_delay_aware"` and every script-specific override field (`mpc_state_
tracking_weight_override`, `mpc_input_tracking_weight_override`, etc.) at
`None`/`False` -- i.e. **the log alone cannot tell you which of the 4
scripts produced it.** The only fields that differ between the two checked
logs are `mpc_prediction_horizon` (15 vs 1) -- expected, that's the N=15 vs
N=1 condition, not the script choice. This needs settling by one of: (a)
checking saved argv/shell history from the actual campaign if it exists
somewhere, (b) adding the inner factory-kwarg string to the logged config
going forward (a one-line fix, consistent with this session's broader
provenance push), or (c) asking the author directly which script was used
for which named condition. Not resolved in this pass -- flagging it plainly
rather than guessing.

INV-7 logs have no `controller_metadata.json` at all (a separately-known
gap from earlier this session) -- their `summary.json["config"]` is the
only source, and it does cleanly show `controller_kind: "inv_7dof_delay_aware"`
for the real campaign logs checked, so the INV-7 side does NOT have this
particular ambiguity.

## Flags added this session, not in any existing HOWTO

These apply to the shape planner/runners specifically and postdate the
three `vessel_stage_a/HOWTO_*.md` docs (which only cover the vessel side
and were not edited by this audit, since they remain accurate for what
they describe):

- `plan_shape_path.py --use-calibrated-beam-base` -- corrects the planner's
  pivot to `rig_calibration.BEAM_BASE_XYZ_M`; default off (legacy stale
  pivot) for backward compatibility, but the current campaign plans require
  it.
- `plan_shape_path.py --magnet-distance-weight` / `--magnet-distance-max-m`
  / `--magnet-distance-scale-m` -- the new L1 cost term bounding how far the
  magnet may drift from the tip on near-straight path segments.
- `plan_shape_path.py --tcp-box-weight` / `--tcp-box-margin-mm` /
  `--tcp-box-scale-mm` -- the new L1 cost term keeping the planned TCP inside
  the free-space workspace box with margin.
- `run_mpc_delay_aware_qn_zero.py` / `run_mpc_delay_aware_insertion_anchor.py
  --disable-insertion-offset-abort` -- forces the insertion-offset monitor
  threshold to `None` and logs that it was off by design, rather than the
  monitor silently not being part of the protocol.
- `run_mpc_delay_aware_qn_zero.py` / `run_mpc_delay_aware_insertion_anchor.py
  --tcp-workspace-robust-margin-mm` -- the shape side's new in-QP TCP-box
  margin, mirroring the vessel runner's own flag of the same name.
- `--allow-unverified-plan` (being added by a concurrent fix as of this
  writing) -- bypass for the beam-base mismatch guard; not for campaign use.

## One claim in the existing vessel HOWTO that is now shape-side-stale

`HOWTO_CLOSED_LOOP_MPC.md` section 0 states "a generic TCP-workspace box...
this one is not wired into the QP at all" -- **this is still true for the
vessel runner** (not edited, still accurate there), but it is **no longer
true for the shape runners** as of this session's TCP-box-in-QP fix. If
anyone reads that line and assumes it also describes the shape side, it
won't. No shape-specific HOWTO existed to correct, since one didn't exist
before this audit; this index is the correction until a full shape-side
HOWTO is written.

## Where the fuller docs live

- Vessel MPC: `vessel_stage_a/HOWTO_CLOSED_LOOP_MPC.md`
- Vessel INV-7: `vessel_stage_a/HOWTO_INVERSE_JACOBIAN.md`
- Vessel planning: `vessel_stage_a/HOWTO_VESSEL_PLANNING.md`
- Shape (all controllers, offline+online, 2026-09-17/18 state): `rectangle_stage_a/README.md`
  -- accurate for what it covers, but predates the delay-aware family's
  real campaign use and this session's new flags; read this index first,
  then that README's "delay-aware controller family" table for per-script
  one-liners.

None of the three vessel HOWTOs needed in-place correction -- they describe
the vessel runners accurately. This index exists because no equivalent
top-level summary existed for the shape side, and because the open
ambiguity above is real and needs the author's input, not a guess.
