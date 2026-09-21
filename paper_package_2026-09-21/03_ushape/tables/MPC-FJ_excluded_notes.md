# MPC-FJ on the U-shape: excluded from quantitative analysis (NA)

Two live attempts were made at MPC-FJ (MPC, frozen index-0 Jacobian) on the
10x15mm bimaterial U-shape. Both diverged at the same location in the path
(ref_index ~156-181) and neither produced a usable full-path RMS number.

| run | outcome | n_ticks completed | final error | max error |
|---|---|---|---|---|
| `mpc_ushape_FJ_20260921T153423Z` | manually stopped by user before becoming unsafe | 190 / 213 | 11.90mm | 11.99mm |
| `mpc_ushape_FJ_20260921T160158Z` | automatically caught by the tightened workspace bound (`z_min=0.20m`) | 174 / 213 | 7.39mm | 7.39mm |

After the first divergence, the live workspace safety bounds in
`close_loop_path_follow.py` were tightened (`workspace_xyz_min_m` z-component
0.30 -> validated -> further loosened to 0.20 after checking against a clean
FJ run's legitimate operating range; `workspace_xyz_max_m` y-component
tightened to -0.483) specifically so a repeat of this failure mode is caught
automatically rather than requiring a human to intervene. The second attempt
confirms this works: the same divergence recurred at the same path location
and was caught automatically at ~7.4mm instead of running to ~12mm.

**Root-cause note**: both INV7-FJ and MPC-FJ show trouble at the same path
location (ref_index ~156-181) -- INV-7-FJ degrades but stays bounded (see
`mechanism_figure_ushape_INV7_schedule.png`, Panel 1, where the divergence is
visible as a widening band from index ~150 onward), while MPC-FJ actively
diverges rather than degrading gracefully. This is the most notable
qualitative finding on this shape: MPC's usual robustness to Jacobian
staleness (seen clearly on the rectangle and the 25mm-base triangle) is not
universal -- it can fail outright, and more abruptly than the matched-authority
inverse, when the frozen-Jacobian mismatch is severe enough at a given point
in the path.

**Recommendation for the paper**: report MPC-FJ as NA for this shape with
this explanation, rather than omitting the condition silently. The 2 diverged
runs are informative failure-mode evidence, not missing data.
