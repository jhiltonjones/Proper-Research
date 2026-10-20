"""Delay-aware beam-output MPC (2026-09-18).

Subclasses `BeamOutputTrackingMPC` rather than mutating it -- the original
must stay behaviourally intact for the `current MPC` vs `delay-aware MPC`
ablation this was built for. Reuses everything that genuinely doesn't
change (weights Qbar/Rbar/Rdbar/Qpbar, the OSQP interface, velocity/rate
limits, the beam Jacobian schedule, `_estimate_output_residual`) and
overrides only the prediction-dependent pieces.

Three explicitly separate objects, kept separate all the way through (see
this package's earlier design discussion -- the whole point of NOT reusing
the old single `measured_state` argument for two different meanings):

    z_meas (7,)   = [q_meas(6), L_meas(1)]   -- feeds ONLY the disturbance
                                                 estimate d_k (unchanged
                                                 from the base class: "given
                                                 where the robot actually is
                                                 now, how wrong is the beam
                                                 model")
    x_exec (13,)  = [q_cmd(6), q_cmd_prev(6), L(1)]  -- feeds ONLY the
                                                 condensed prediction (what
                                                 the optimizer predicts will
                                                 happen), via the validated
                                                 `mpc_delay_aware.prediction`
                                                 module
    previous_input (7,) = u_applied,k-1        -- feeds ONLY the rate
                                                 (Delta-u) constraint and
                                                 cost, unchanged, exactly the
                                                 `u_prev` bookkeeping already
                                                 audited earlier in this
                                                 investigation

Two condensed representations of the SAME decision vector v are built from
x_exec, for two DIFFERENT purposes (this is the one part of the design that
is easy to get backwards, so it is spelled out everywhere it's used):

    (Ep, Sp)  physical-state stack  z_phys_stack = Ep@x_exec + Sp@v
              -- feeds the Q state-tracking cost AND the beam prediction
              (delay_samples=2: q_phys lags the command by 2 samples)

    (Ec, Sc)  command-state stack   q_cmd_stack  = Ec@x_exec + Sc@v
              -- feeds ONLY the joint-position box (safety) constraint,
              because q_cmd is the absolute target actually sent to
              servoJ (delay_samples=0: no shift, standard accumulator)

No DARE terminal cost (V_f=0) -- `beam_config.use_dare_terminal_cost` MUST
be False; this class raises otherwise rather than trying to reinterpret the
augmented-state terminal Riccati machinery for a 13D state it was never
derived for. `directional_damping` and `wall_avoidance_gain` must also be 0
(unused optional cost terms this development controller doesn't touch --
raising rather than silently ignoring them if set).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from proper_research.simulation.simulations import (
    simulate_time_parameterized_beam_output_mpc as beam_module,
)

from .prediction import build_delay_prediction_matrices

Array = np.ndarray
_finite_vector = beam_module._finite_vector  # noqa: SLF001 -- reuse, don't duplicate

__all__ = ["DelayAwareBeamOutputMPCStep", "DelayAwareBeamOutputTrackingMPC"]


@dataclass
class DelayAwareBeamOutputMPCStep(beam_module.BeamOutputMPCStep):
    predicted_commands: Array = None  # (N, 7) -- q_cmd_stack, NOT the physical prediction
    # (2026-09-28 instrumentation pass) the exact (3, n) tip-Jacobian this
    # tick's QP was actually built with -- jacobians[0] from
    # `_beam_prediction_terms_exec`, read back out of `beam_terms["Jbar"]`'s
    # top-left block rather than threading a new value through every
    # `_dynamic_qp_terms_exec` override (see `solve_delay_aware`). Purely a
    # read of an already-computed matrix; does not affect the solve.
    jacobian_used: Array = None
    # (2026-09-28) raw OSQP dual vector for this solve, or None off the osqp
    # backend / on solver failure -- see `solve_delay_aware`'s osqp branch.
    dual_y: Array = None


class DelayAwareBeamOutputTrackingMPC(beam_module.BeamOutputTrackingMPC):
    """See module docstring. `delay_samples` in {0, 1, 2} (see
    `mpc_delay_aware.prediction`'s docstring for why 2 is the only
    validated, and 0-is-a-regression-test-only, value); `beta_d` scales the
    disturbance term uniformly across the horizon (beta_d=1 for the first
    ablation, per the frozen 2026-09-18 investigation sequence; beta_d=0.82
    is the earned-but-not-yet-adopted refinement -- see this package's
    README)."""

    def __init__(
        self,
        *,
        reference: Any,
        config: Any,
        beam_config: Any,
        reference_position_jacobians: Any,
        nominal_reference_positions_m: Any | None = None,
        delay_samples: int = 2,
        beta_d: float = 1.0,
        magnet_exclusion: dict | None = None,
        magnet_workspace: dict | None = None,
        magnet_exclusion_clearance: dict | None = None,
        magnet_workspace_clearance: dict | None = None,
        tcp_workspace: dict | None = None,
        magnetic_force_preference: dict | None = None,
        dipole_alignment_preference: dict | None = None,
    ) -> None:
        if beam_config.use_dare_terminal_cost:
            raise ValueError(
                "DelayAwareBeamOutputTrackingMPC requires "
                "beam_config.use_dare_terminal_cost=False (V_f=0) -- see module "
                "docstring for why this development controller does not adapt "
                "the terminal Riccati machinery to the augmented 13D state."
            )
        if float(beam_config.directional_damping) != 0.0:
            raise ValueError("directional_damping must be 0 for this development controller.")
        if float(beam_config.wall_avoidance_gain) != 0.0:
            raise ValueError("wall_avoidance_gain must be 0 for this development controller.")
        super().__init__(
            reference=reference, config=config, beam_config=beam_config,
            reference_position_jacobians=reference_position_jacobians,
            nominal_reference_positions_m=nominal_reference_positions_m,
        )
        self.delay_samples = int(delay_samples)
        self.beta_d = float(beta_d)
        n_joints = self.n - 1  # 6

        self._magnet_excl_radius: float | None = None
        self._magnet_excl_live_relinearize: bool = False
        if magnet_exclusion is not None:
            self._configure_magnet_exclusion(**magnet_exclusion)

        self._magnet_ws_z_min: float | None = None
        self._magnet_ws_z_max: float | None = None
        self._magnet_ws_live_relinearize: bool = False
        if magnet_workspace is not None:
            self._configure_magnet_workspace(**magnet_workspace)

        # 2026-09-30: soft clearance-cost term, added SEPARATELY from (and
        # never replacing) the hard magnet_exclusion constraint above -- see
        # this class's README/session notes for why. The hard constraint
        # answers "is the commanded state feasible"; this answers "does the
        # optimizer have any REASON to prefer more margin before the hard
        # constraint binds." Requires magnet_exclusion to already be
        # configured (reuses its (d_nom, normal, jacobians) schedule
        # unchanged -- this term never re-derives its own geometry).
        self._magnet_clear_gain: float | None = None
        self._magnet_clear_soft_radius: float | None = None
        if magnet_exclusion_clearance is not None:
            self._configure_magnet_exclusion_clearance(**magnet_exclusion_clearance)

        # 2026-10-12: z-workspace analogue of the exclusion clearance cost
        # above, added SEPARATELY from (and never replacing) the hard
        # magnet_workspace constraint -- same rationale: the hard bound only
        # answers "is the commanded z feasible," it gives the optimizer no
        # reason to prefer staying away from the edge before the hard bound
        # binds. Root cause this closes: a live floor240 run with
        # --zero-input-reference-in-r (which removes the R-cost's pull
        # toward the offline plan's own v_ref -- the only thing that had
        # been discouraging a redundant null-space direction from climbing
        # in z) rode the hard z ceiling, margined or not, for its entire
        # run -- magnet z climbed smoothly and monotonically from tick 0
        # (no oscillation, ~1.3mm/tick), so no amount of HARD-bound margin
        # alone can fix this: it only buys a few extra ticks before the
        # same monotonic climb consumes that margin too. Requires
        # magnet_workspace to already be configured (reuses its (z_nom,
        # jacobians, z_min, z_max) schedule unchanged -- this term never
        # derives its own geometry).
        self._magnet_ws_clear_gain: float | None = None
        self._magnet_ws_clear_soft_margin: float | None = None
        if magnet_workspace_clearance is not None:
            self._configure_magnet_workspace_clearance(**magnet_workspace_clearance)

        # 2026-10-14: proactive, in-optimizer counterpart to close_loop_
        # path_follow.py's post-hoc tcp_out_of_workspace monitor -- see
        # _configure_tcp_workspace's own docstring for why this is a
        # SEPARATE mechanism from magnet_workspace above (different
        # frame: the literal UR robot-base frame "R" that
        # getActualTCPPose() reports in, not the beam-centric frame
        # magnet_workspace uses) and does not modify it.
        self._tcp_ws_min: Array | None = None
        self._tcp_ws_max: Array | None = None
        self._tcp_ws_live_relinearize: bool = False
        if tcp_workspace is not None:
            self._configure_tcp_workspace(**tcp_workspace)

        # 2026-10-15: soft "actuation preference" tie-breaker -- NOT a hard
        # constraint, NEVER rules out a configuration the task cost would
        # otherwise pick. Reuses the SAME command-stack (Ec, Sc) anchoring as
        # magnet_exclusion/magnet_workspace/tcp_workspace above (all three
        # reason about "where the magnet rigidly on the end effector is
        # being told to go right now"), not the delayed physical stack
        # (Ep, Sp) the beam-tip-tracking cost uses -- see
        # `_configure_magnetic_force_preference`'s own docstring for why.
        self._force_pref_lambda_F: float = 0.0
        self._force_pref_lambda_tau: float = 0.0
        if magnetic_force_preference is not None:
            self._configure_magnetic_force_preference(**magnetic_force_preference)

        # 2026-10-19: see this class's _configure_dipole_alignment_
        # preference docstring for why this REPLACES magnetic_force_
        # preference for "make the magnet point somewhere" purposes --
        # minimizing torque (m x B) can be satisfied by moving the magnet
        # FARTHER away (B -> 0) just as well as by genuine alignment;
        # this term is blind to distance by construction (both terms in
        # its residual are unit vectors), so it cannot be gamed that way.
        self._dipole_align_weight: float = 0.0
        if dipole_alignment_preference is not None:
            self._configure_dipole_alignment_preference(**dipole_alignment_preference)

        self.Ep, self.Sp = build_delay_prediction_matrices(
            N=self.N, dt=self.dt, delay_samples=self.delay_samples, n_joints=n_joints,
        )
        self.Ec, self.Sc = build_delay_prediction_matrices(
            N=self.N, dt=self.dt, delay_samples=0, n_joints=n_joints,
        )

        # Rebuild H using Sp (physical-state stack) in place of the base
        # class's S (which used the old undelayed n=7 model for EVERYTHING,
        # cost included) -- the R/Rd terms (Rbar, D.T@Rdbar@D) are pure
        # input-side costs, unaffected by the state augmentation, so they
        # carry over unchanged.
        base_hessian = 2.0 * (
            self.Sp.T @ self.Qbar @ self.Sp + self.Rbar + self.D.T @ self.Rdbar @ self.D
        )
        base_hessian += float(self.config.hessian_regularization) * np.eye(self.nu)
        self.H = 0.5 * (base_hessian + base_hessian.T)
        self._base_hessian = np.asarray(self.H, dtype=float).copy()
        if self.backend == "osqp":
            # Rebuild the OSQP sparsity pattern for the corrected H -- see
            # the Q-ablation NullAwareMPC precedent in
            # hardware/online/rectangle_stage_a/q_ablation.py for why
            # `_setup_osqp()` (sparsity from H alone) is wrong here and
            # `_setup_variable_hessian_osqp()` (includes the dense runtime
            # G.T@Qpbar@G term) is required.
            if (
                self._magnet_excl_radius is not None
                or self._magnet_ws_z_min is not None
                or self._tcp_ws_min is not None
            ):
                self._setup_osqp_with_magnet_exclusion()
            else:
                self._setup_variable_hessian_osqp()

    # ------------------------------------------------------------------
    # magnet-to-point exclusion, wired directly into the QP (2026-09-28;
    # repurposed 2026-09-28 from a vessel-centreline point set to a single
    # fixed beam-base point -- the method itself is unchanged, generic over
    # any (K,3) point set, K=1 included; only the caller-supplied
    # `lumen_C_m`/`radius_m` changed, in `run_mpc_delay_aware_vessel.py`)
    # ------------------------------------------------------------------
    def _configure_magnet_exclusion(
        self, *, position_jacobians: Any, nominal_positions_m: Any,
        lumen_C_m: Any, radius_m: float,
        live_relinearize: bool = False, dh: Any = None, T_F_M: Any = None,
    ) -> None:
        """Add a linearized "magnet stays >= radius_m from the nearest point
        in lumen_C_m" inequality to every horizon stage of the QP, so the
        optimizer itself can see this limit -- previously it was enforced
        ONLY as a post-hoc monitor in `process_isolated_adapter.py`,
        computed from the MEASURED joints after the command had already
        been applied. That monitor stays on unchanged as a second line of
        defense; this is a first line of defense inside the optimization.
        `lumen_C_m` is generic: a many-point vessel centreline (the
        original 2026-09-28 use) or a single fixed point such as the beam
        base (K=1) both work unchanged -- `closest = np.argmin(dists,
        axis=1)` is trivial but correct for K=1.

        Root cause this closes: the frozen-Jacobian run
        (close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_frozen_
        20260928T140358Z) aborted at 143/525 ticks on
        `magnet_exclusion_violated`, yet EVERY logged QP solve that whole
        run reports status='ok'/success=True -- the optimizer was never
        infeasible, it simply had no term in its own constraint set that
        knew the wall existed, so a fully valid-by-its-own-lights solution
        walked the real magnet into the abort radius.

        Fixed schedule (like `reference_position_jacobians` for the beam-
        tracking term): for each reference sample, find the closest point
        on `lumen_C_m` to the magnet's own NOMINAL (planned) position
        there, and linearize the distance-to-that-point about that nominal
        magnet position and its analytic position Jacobian w.r.t. q
        (`position_jacobians`, (S,3,6), NOT a function of insertion L --
        the magnet is rigidly on the end effector). Re-anchored every solve
        to the ACTUAL commanded state via (Ec, Sc) -- the same command
        stack the joint-position box already uses, deliberately not the
        delayed physical prediction -- so the constraint tracks where the
        robot is actually being told to go, tick to tick, even though the
        (index -> nominal point/Jacobian/normal) schedule itself is fixed
        offline from the reference trajectory.
        """
        jacobians = np.asarray(position_jacobians, dtype=float)
        positions = np.asarray(nominal_positions_m, dtype=float)
        lumen = np.asarray(lumen_C_m, dtype=float).reshape(-1, 3)
        n_joints = self.n - 1
        expected_j = (self.reference.sample_count, 3, n_joints)
        expected_p = (self.reference.sample_count, 3)
        if jacobians.shape != expected_j or not np.all(np.isfinite(jacobians)):
            raise ValueError(
                "magnet_exclusion['position_jacobians'] must have shape "
                f"{expected_j}; received {jacobians.shape}."
            )
        if positions.shape != expected_p or not np.all(np.isfinite(positions)):
            raise ValueError(
                "magnet_exclusion['nominal_positions_m'] must have shape "
                f"{expected_p}; received {positions.shape}."
            )
        if lumen.ndim != 2 or lumen.shape[1] != 3 or not np.all(np.isfinite(lumen)):
            raise ValueError("magnet_exclusion['lumen_C_m'] must be finite with shape (K,3).")
        radius = float(radius_m)
        if radius <= 0.0:
            raise ValueError("magnet_exclusion['radius_m'] must be positive.")

        diffs = positions[:, None, :] - lumen[None, :, :]              # (S,K,3)
        dists = np.linalg.norm(diffs, axis=2)                           # (S,K)
        closest = np.argmin(dists, axis=1)                              # (S,)
        rows_idx = np.arange(positions.shape[0])
        d_nom = dists[rows_idx, closest]                                 # (S,)
        normal = diffs[rows_idx, closest] / np.maximum(d_nom, 1.0e-9)[:, None]  # (S,3)

        self._magnet_excl_jacobians = jacobians
        self._magnet_excl_d_nom = d_nom
        self._magnet_excl_normal = normal
        self._magnet_excl_radius = radius
        self._magnet_excl_min_nominal_margin_m = float(np.min(d_nom - radius))
        self._magnet_excl_lumen = lumen

        # 2026-10-12: same fix as _configure_magnet_workspace's own
        # live_relinearize -- see that method's docstring for the full
        # root-cause story (confirmed live on the z-workspace constraint;
        # this constraint shares the IDENTICAL linearize-around-offline-
        # nominal-q pattern, so it has the SAME failure mode under
        # --zero-input-reference-in-r's accumulated drift). When enabled,
        # re-anchors (d_nom, normal, c) to the CURRENT actual commanded
        # magnet position/Jacobian every solve, re-resolving the closest
        # lumen_C_m point against that live position too (not the stale
        # offline-nominal one) -- rather than the fixed per-reference-index
        # schedule computed once at construction time.
        self._magnet_excl_live_relinearize = bool(live_relinearize)
        if self._magnet_excl_live_relinearize:
            if dh is None or T_F_M is None:
                raise ValueError(
                    "magnet_exclusion['live_relinearize']=True requires both "
                    "'dh' and 'T_F_M' (the robot's own kinematic parameters, "
                    "needed to compute a fresh FK/Jacobian anchor every solve)."
                )
            self._magnet_excl_dh = dh
            self._magnet_excl_T_F_M = np.asarray(T_F_M, dtype=float)

        # Placeholder magnet rows: force full sparsity in every column so a
        # later OSQP Ax-update can write ANY real coefficient (including an
        # exact zero) into every slot -- same trick `_setup_variable_hessian_
        # osqp` already uses for the Hessian's own upper-triangular pattern.
        placeholder = np.full((self.N, self.nu), 1.0e-30, dtype=float)
        self._magnet_excl_row_start = self.A.shape[0]
        self.A = np.vstack((self.A, placeholder))

    def _setup_osqp_with_magnet_exclusion(self) -> None:
        """Same as the base class's `_setup_variable_hessian_osqp`, except
        it does NOT reuse that method's own dummy-bounds call (which is
        hard-sized to the base 3-block `self.A` and would mismatch any
        extra magnet-exclusion/workspace rows) -- it builds correctly-sized
        dummy bounds itself and caches `self._a_pattern` so later solves
        can push real row values in via `update(Ax=...)`.

        Dummy-row count is derived from `self.A.shape[0] - base row count`
        rather than assumed -- `self.A` may carry ZERO, ONE, or more
        extra N-row blocks by this point (exclusion: 1 block; magnet
        workspace: 1 block; tcp_workspace: 3 blocks, one per axis),
        appended in `_configure_magnet_exclusion`/`_configure_magnet_
        workspace`/`_configure_tcp_workspace` (called, if at all,
        before this setup runs -- see `__init__`), so hardcoding one
        block's worth of dummy rows here would silently under/over-
        size `l`/`u` whenever more than one of these mechanisms (or
        none, in a future refactor) are active at once."""
        import osqp as _osqp

        self._p_pattern = sp.triu(
            sp.csc_matrix(np.ones((self.nu, self.nu), dtype=float)), format="csc",
        )
        setup_values = self._upper_values(self._base_hessian)
        setup_values[np.abs(setup_values) < 1.0e-30] = 1.0e-30
        P = sp.csc_matrix(
            (setup_values, self._p_pattern.indices.copy(), self._p_pattern.indptr.copy()),
            shape=(self.nu, self.nu),
        )
        base_lower, base_upper = self._constraint_bounds(
            state=np.zeros(7), previous_input=np.zeros(7), validate_state=False,
        )
        n_extra_rows = self.A.shape[0] - base_lower.shape[0]
        assert n_extra_rows >= 0 and n_extra_rows % self.N == 0, (
            f"unexpected extra row count {n_extra_rows} (self.A has "
            f"{self.A.shape[0]} rows, base constraint set has "
            f"{base_lower.shape[0]}) -- expected a whole multiple of N={self.N} "
            "from the magnet-exclusion/workspace constraint blocks"
        )
        dummy_lower = np.full(n_extra_rows, -1.0e6, dtype=float)
        dummy_upper = np.full(n_extra_rows, 1.0e6, dtype=float)
        lower = np.concatenate([base_lower, dummy_lower])
        upper = np.concatenate([base_upper, dummy_upper])
        self._a_pattern = sp.csc_matrix(self.A)
        self._solver = _osqp.OSQP()
        setup_kwargs = dict(
            P=P, q=np.zeros(self.nu), A=self._a_pattern, l=lower, u=upper,
            eps_abs=float(self.config.solver_absolute_tolerance),
            eps_rel=float(self.config.solver_relative_tolerance),
            max_iter=int(self.config.solver_maximum_iterations),
            polish=bool(self.config.solver_polish),
            verbose=bool(self.config.solver_verbose),
            warm_start=True,
        )
        if float(self.config.solver_time_limit_s) > 0.0:
            setup_kwargs["time_limit"] = float(self.config.solver_time_limit_s)
        self._solver.setup(**setup_kwargs)

    def _a_values(self, A_dense: Array) -> Array:
        values = np.empty(self._a_pattern.nnz, dtype=float)
        for column in range(self.nu):
            start = self._a_pattern.indptr[column]
            stop = self._a_pattern.indptr[column + 1]
            rows = self._a_pattern.indices[start:stop]
            values[start:stop] = A_dense[rows, column]
        return values

    def _magnet_exclusion_bounds(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        if self._magnet_excl_radius is None:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

        if self._magnet_excl_live_relinearize:
            from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
            q_now = free_cmd[0]
            pos_live = urik.forward_kinematics(q_now, self._magnet_excl_dh, self._magnet_excl_T_F_M).T_R_target[:3, 3]
            jac_live = urik.geometric_jacobian(q_now, self._magnet_excl_dh, self._magnet_excl_T_F_M)[:3, :6]
            diffs_live = pos_live[None, :] - self._magnet_excl_lumen      # (K,3)
            dists_live = np.linalg.norm(diffs_live, axis=1)               # (K,)
            closest_live = int(np.argmin(dists_live))
            d_live = float(dists_live[closest_live])
            normal_live = diffs_live[closest_live] / max(d_live, 1.0e-9)
            c_live = normal_live @ jac_live                              # (6,)
            q_nom = np.tile(q_now, (self.N, 1))
            d_nom = np.full(self.N, d_live, dtype=float)
            c = np.tile(c_live, (self.N, 1))                             # (N,6)
        else:
            indices = self._reference_indices(control_index, future=True)
            J = self._magnet_excl_jacobians[indices]                       # (N,3,6)
            q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
            d_nom = self._magnet_excl_d_nom[indices]                        # (N,)
            normal = self._magnet_excl_normal[indices]                      # (N,3)
            c = np.einsum("nj,njk->nk", normal, J)                          # (N,6): n_j^T J_j
        rows = np.zeros((self.N, self.nu), dtype=float)
        lower = np.empty(self.N, dtype=float)
        for j in range(self.N):
            stage_rows = slice(j * self.n, j * self.n + 6)
            rows[j, :] = c[j] @ self.Sc[stage_rows, :]
            lower[j] = self._magnet_excl_radius - d_nom[j] + c[j] @ q_nom[j] - c[j] @ free_cmd[j]
        upper = np.full(self.N, np.inf, dtype=float)
        self.A[self._magnet_excl_row_start:self._magnet_excl_row_start + self.N, :] = rows
        return lower, upper

    # ------------------------------------------------------------------
    # soft clearance-cost term (2026-09-30): a SEPARATE mechanism from the
    # hard magnet_exclusion constraint above. Root cause this closes: live
    # instrumentation (2026-09-30 session) showed the hard constraint
    # correctly guarantees d(q_cmd) >= radius_m -- it was NEVER violated in
    # its own linearized sense, and the linearization error against the
    # true nonlinear distance was consistently sub-millimetre -- yet the
    # controller still rode the boundary for 15+ consecutive ticks with
    # zero margin, because nothing in the cost function distinguishes
    # d=0.01mm from d=10mm as long as the hard constraint is satisfied.
    # Meanwhile a genuine redundant escape direction was measured
    # (||N @ grad_d|| ~= 25-33mm/rad, N = tip-Jacobian nullspace projector)
    # that the controller had no incentive to use. This term supplies that
    # incentive: a per-horizon-stage quadratic penalty, active ONLY when
    # this tick's own linearized clearance prediction falls inside
    # [radius_m, soft_radius_m) -- gated fresh every solve against the
    # SAME (Ec, Sc)-anchored d_hat the hard constraint itself uses, never
    # against a stale/offline value. Does not touch the hard constraint's
    # own radius_m or its rows in self.A at all.
    def _configure_magnet_exclusion_clearance(
        self, *, gain: float, soft_radius_m: float,
    ) -> None:
        if self._magnet_excl_radius is None:
            raise ValueError(
                "magnet_exclusion_clearance requires magnet_exclusion to be "
                "configured too (this term reuses its (d_nom, normal, "
                "jacobians) schedule unchanged; it never derives its own "
                "geometry)."
            )
        gain = float(gain)
        soft_radius = float(soft_radius_m)
        if gain < 0.0 or not np.isfinite(gain):
            raise ValueError("magnet_exclusion_clearance['gain'] must be finite and >= 0.")
        if soft_radius <= self._magnet_excl_radius:
            raise ValueError(
                f"magnet_exclusion_clearance['soft_radius_m']={soft_radius} must exceed "
                f"the hard magnet_exclusion radius_m={self._magnet_excl_radius} -- "
                "otherwise the avoidance band is empty or inverted."
            )
        self._magnet_clear_gain = gain
        self._magnet_clear_soft_radius = soft_radius

    def _magnet_exclusion_clearance_qp_terms(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        zero = (np.zeros((self.nu, self.nu), dtype=float), np.zeros(self.nu, dtype=float))
        if not self._magnet_clear_gain:
            return zero
        indices = self._reference_indices(control_index, future=True)
        J = self._magnet_excl_jacobians[indices]                       # (N,3,6)
        q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
        d_nom = self._magnet_excl_d_nom[indices]                        # (N,)
        normal = self._magnet_excl_normal[indices]                      # (N,3)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)
        c = np.einsum("nj,njk->nk", normal, J)                          # (N,6)

        hessian = np.zeros((self.nu, self.nu), dtype=float)
        linear = np.zeros(self.nu, dtype=float)
        for j in range(self.N):
            d_hat = d_nom[j] + c[j] @ (free_cmd[j] - q_nom[j])
            if d_hat >= self._magnet_clear_soft_radius:
                continue  # this stage's own commanded trajectory is already clear -- no penalty
            stage_rows = slice(j * self.n, j * self.n + 6)
            a = c[j] @ self.Sc[stage_rows, :]                            # (nu,): d(d_j)/du
            b = self._magnet_clear_soft_radius - d_hat                  # > 0 here, by the check above
            # cost_j = gain * (soft_radius - d_j)^2 = gain * (b - a@u)^2
            hessian += 2.0 * self._magnet_clear_gain * np.outer(a, a)
            linear += -2.0 * self._magnet_clear_gain * b * a
        return hessian, linear

    # ------------------------------------------------------------------
    # magnet z-workspace bounds, wired directly into the QP (2026-09-28,
    # same day as the point-exclusion constraint above, closing the
    # identical blind spot for general magnet-workspace excursions rather
    # than only wall-approach)
    # ------------------------------------------------------------------
    def _configure_magnet_workspace(
        self, *, position_jacobians: Any, nominal_positions_m: Any,
        z_min_m: float, z_max_m: float,
        live_relinearize: bool = False, dh: Any = None, T_F_M: Any = None,
    ) -> None:
        """Add a linearized "magnet z stays within [z_min_m, z_max_m]"
        two-sided inequality to every horizon stage of the QP -- the
        proactive, in-optimizer counterpart to
        `run_mpc_delay_aware_vessel.py`'s post-hoc `_MAGNET_Z_BOUNDS_M`
        monitor (pass the SAME bounds here; that monitor stays on
        unchanged as a second line of defense, computed from MEASURED
        joints after a command has already been applied).

        Root cause this closes: a live frozen-Jacobian run this session
        (no relinearization, Q_N=0 removing all posture anchoring) let
        the redundant joint DOF drift the arm's flange to z=0.262m --
        ~25cm above normal operating height -- while EVERY logged QP
        solve that whole run reported status='ok'/success=True. None of
        the QP's existing constraints (per-joint position box, per-joint
        velocity/rate limits, the point-exclusion constraint above) cover
        a general Cartesian/TCP-workspace bound, so this kind of
        excursion was invisible to the optimizer by construction, only
        ever catchable after the fact by the external `tcp_out_of_
        workspace` monitor -- exactly what happened.

        Same linearization recipe as `_configure_magnet_exclusion`: for
        each reference sample, the magnet's nominal z and the z-row of
        its position Jacobian w.r.t. q (`position_jacobians[:, 2, :]`)
        give a first-order estimate of achieved z as a function of the
        commanded joints, re-anchored every solve via (Ec, Sc) exactly
        like the exclusion constraint. Unlike the exclusion (one-sided,
        distance >= radius), this is TWO-sided on the SAME linear row
        (z_min <= z <= z_max), so it costs only N extra rows, not 2N.
        """
        jacobians = np.asarray(position_jacobians, dtype=float)
        positions = np.asarray(nominal_positions_m, dtype=float)
        n_joints = self.n - 1
        expected_j = (self.reference.sample_count, 3, n_joints)
        expected_p = (self.reference.sample_count, 3)
        if jacobians.shape != expected_j or not np.all(np.isfinite(jacobians)):
            raise ValueError(
                "magnet_workspace['position_jacobians'] must have shape "
                f"{expected_j}; received {jacobians.shape}."
            )
        if positions.shape != expected_p or not np.all(np.isfinite(positions)):
            raise ValueError(
                "magnet_workspace['nominal_positions_m'] must have shape "
                f"{expected_p}; received {positions.shape}."
            )
        z_min = float(z_min_m)
        z_max = float(z_max_m)
        if not (z_min < z_max):
            raise ValueError("magnet_workspace['z_min_m'] must be < z_max_m.")

        self._magnet_ws_jacobians = jacobians[:, 2, :].copy()  # (S,6) -- z-row only
        self._magnet_ws_nominal_z = positions[:, 2].copy()      # (S,)
        self._magnet_ws_z_min = z_min
        self._magnet_ws_z_max = z_max

        # 2026-10-12: optional re-anchoring of the linearization point from
        # the OFFLINE PLAN's nominal joint trajectory (q_nom[j]/z_nom[j]
        # above, fixed at schedule-build time) to the CURRENT actual
        # commanded state, recomputed fresh every solve. Root cause this
        # closes: under --zero-input-reference-in-r, the whole point is to
        # let the redundant DOF drift away from q_nom -- confirmed live
        # (floor240 run, 2026-10-12) that after ~10deg of accumulated joint
        # drift from q_nom, the first-order estimate using the STALE (q_nom,
        # Jacobian-at-q_nom) anchor was already wrong by 5.5mm with ZERO
        # further motion (verified by direct FK comparison, not guessed) --
        # the hard constraint was being satisfied exactly against its own
        # (by-then-inaccurate) linear model, while the TRUE z blew past the
        # bound by several mm per tick. Re-anchoring to the current state
        # each solve keeps the Taylor expansion valid over only the (much
        # smaller) intra-horizon drift, not the full accumulated drift
        # since the plan's own start. None (default for dh/T_F_M, or
        # live_relinearize=False) preserves the exact original behaviour.
        self._magnet_ws_live_relinearize = bool(live_relinearize)
        if self._magnet_ws_live_relinearize:
            if dh is None or T_F_M is None:
                raise ValueError(
                    "magnet_workspace['live_relinearize']=True requires both "
                    "'dh' and 'T_F_M' (the robot's own kinematic parameters, "
                    "needed to compute a fresh FK/Jacobian anchor every solve)."
                )
            self._magnet_ws_dh = dh
            self._magnet_ws_T_F_M = np.asarray(T_F_M, dtype=float)

        placeholder = np.full((self.N, self.nu), 1.0e-30, dtype=float)
        self._magnet_ws_row_start = self.A.shape[0]
        self.A = np.vstack((self.A, placeholder))

    def _magnet_workspace_bounds(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        if self._magnet_ws_z_min is None:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        indices = self._reference_indices(control_index, future=True)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

        if self._magnet_ws_live_relinearize:
            # Re-anchor ALL stages to a SINGLE fresh (z, Jacobian) pair
            # evaluated at the CURRENT actual commanded joints (free_cmd[0]),
            # not the offline plan's nominal q_nom -- see
            # _configure_magnet_workspace's docstring for why. One extra
            # FK + Jacobian call per solve (not per horizon stage); the
            # Taylor expansion only has to stay valid over the horizon's own
            # (much smaller) intra-horizon drift, not the drift accumulated
            # since the plan's own start.
            from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
            q_now = free_cmd[0]
            z_live = float(urik.forward_kinematics(q_now, self._magnet_ws_dh, self._magnet_ws_T_F_M).T_R_target[2, 3])
            c_live = urik.geometric_jacobian(q_now, self._magnet_ws_dh, self._magnet_ws_T_F_M)[2, :6]
            c = np.tile(c_live, (self.N, 1))                            # (N,6)
            z_nom = np.full(self.N, z_live, dtype=float)
            q_nom = np.tile(q_now, (self.N, 1))                         # (N,6)
        else:
            c = self._magnet_ws_jacobians[indices]                          # (N,6)
            q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
            z_nom = self._magnet_ws_nominal_z[indices]                       # (N,)

        rows = np.zeros((self.N, self.nu), dtype=float)
        lower = np.empty(self.N, dtype=float)
        upper = np.empty(self.N, dtype=float)
        for j in range(self.N):
            stage_rows = slice(j * self.n, j * self.n + 6)
            rows[j, :] = c[j] @ self.Sc[stage_rows, :]
            offset = z_nom[j] - c[j] @ q_nom[j] + c[j] @ free_cmd[j]
            lower[j] = self._magnet_ws_z_min - offset
            upper[j] = self._magnet_ws_z_max - offset
        self.A[self._magnet_ws_row_start:self._magnet_ws_row_start + self.N, :] = rows
        return lower, upper

    # ------------------------------------------------------------------
    # TCP Cartesian workspace box, wired directly into the QP (2026-10-14)
    # -- proactive counterpart to close_loop_path_follow.py's post-hoc
    # tcp_out_of_workspace monitor. SEPARATE from _configure_magnet_
    # workspace above: that one bounds the MAGNET's z in a beam-centric
    # frame (_resolve_robot_kinematics(make_robot_config())'s dh/T_F_M);
    # this one bounds the TCP's full (x,y,z) in the literal UR robot-base
    # frame "R" that close_loop_path_follow.py's reader.latest_pose()
    # (= getActualTCPPose()) reports in. Do not conflate the two frames
    # or reuse one mechanism's dh/offset for the other.
    # ------------------------------------------------------------------
    def _configure_tcp_workspace(
        self, *, position_jacobians: Any, nominal_positions_m: Any,
        xyz_min: Any, xyz_max: Any,
        live_relinearize: bool = False, dh: Any = None, T_F_TCP: Any = None,
    ) -> None:
        """Add a linearized, two-sided "TCP stays within [xyz_min, xyz_max]"
        inequality -- one row per axis per horizon stage (3*N extra rows
        total) -- to every horizon stage of the QP.

        `position_jacobians`/`nominal_positions_m` must be built OFFLINE
        using `urik.corrected_dh_from_config(urik.CONFIG)` (the DH this
        project has already validated against the robot's real FK -- see
        `ur_magnet_ik_jacobian_validation.py`) and the FLANGE-TO-TCP
        offset (`pose6_to_T(urik.CONFIG.offline_active_T_flange_tcp_
        pose6)`), NOT `_resolve_robot_kinematics(make_robot_config())`
        (the wrong, beam-centric frame -- using it here would silently
        constrain a different point than the one `tcp_out_of_workspace`
        actually checks, which is worse than no constraint at all since
        it would look protective without being so).

        Accuracy caveat, found by direct comparison against real
        hardware abort poses before this was wired in (2026-10-14): this
        FK agreed with the logged `tcp_out_of_workspace([...])` pose to a
        few mm, not the ~0.11mm this DH is validated to elsewhere --
        likely `CONFIG.offline_active_T_flange_tcp_pose6` defaulting to
        identity (no real flange-to-TCP translation applied) plus
        `pose`/`q_meas` not being read perfectly atomically on hardware.
        Callers MUST pass `xyz_min`/`xyz_max` with a safety margin large
        enough to absorb this (several mm, not sub-mm) inward from
        whatever the post-hoc monitor's own bounds are -- see this
        mechanism's caller in `run_mpc_delay_aware_vessel.py` for the
        actual margin used.

        Same linearization recipe as `_configure_magnet_workspace`, just
        three rows (x,y,z) per horizon stage instead of one.
        """
        jacobians = np.asarray(position_jacobians, dtype=float)
        positions = np.asarray(nominal_positions_m, dtype=float)
        n_joints = self.n - 1
        expected_j = (self.reference.sample_count, 3, n_joints)
        expected_p = (self.reference.sample_count, 3)
        if jacobians.shape != expected_j or not np.all(np.isfinite(jacobians)):
            raise ValueError(
                "tcp_workspace['position_jacobians'] must have shape "
                f"{expected_j}; received {jacobians.shape}."
            )
        if positions.shape != expected_p or not np.all(np.isfinite(positions)):
            raise ValueError(
                "tcp_workspace['nominal_positions_m'] must have shape "
                f"{expected_p}; received {positions.shape}."
            )
        xyz_min = np.asarray(xyz_min, dtype=float).reshape(3)
        xyz_max = np.asarray(xyz_max, dtype=float).reshape(3)
        if not np.all(xyz_min < xyz_max):
            raise ValueError(
                "tcp_workspace['xyz_min']/['xyz_max'] must satisfy min < max on every axis."
            )

        self._tcp_ws_jacobians = jacobians.copy()   # (S,3,6)
        self._tcp_ws_nominal = positions.copy()      # (S,3)
        self._tcp_ws_min = xyz_min
        self._tcp_ws_max = xyz_max

        self._tcp_ws_live_relinearize = bool(live_relinearize)
        if self._tcp_ws_live_relinearize:
            if dh is None or T_F_TCP is None:
                raise ValueError(
                    "tcp_workspace['live_relinearize']=True requires both 'dh' "
                    "and 'T_F_TCP' (the robot's own calibrated kinematics and "
                    "flange-to-TCP offset, needed to compute a fresh FK/"
                    "Jacobian anchor every solve)."
                )
            self._tcp_ws_dh = dh
            self._tcp_ws_T_F_TCP = np.asarray(T_F_TCP, dtype=float)

        placeholder = np.full((3 * self.N, self.nu), 1.0e-30, dtype=float)
        self._tcp_ws_row_start = self.A.shape[0]
        self.A = np.vstack((self.A, placeholder))

    def _tcp_workspace_bounds(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        if self._tcp_ws_min is None:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        indices = self._reference_indices(control_index, future=True)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

        if self._tcp_ws_live_relinearize:
            from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
            q_now = free_cmd[0]
            p_live = urik.forward_kinematics(q_now, self._tcp_ws_dh, self._tcp_ws_T_F_TCP).T_R_target[:3, 3]
            J_live = urik.geometric_jacobian(q_now, self._tcp_ws_dh, self._tcp_ws_T_F_TCP)[:3, :6]
            c = np.tile(J_live[None, :, :], (self.N, 1, 1))              # (N,3,6)
            p_nom = np.tile(p_live[None, :], (self.N, 1))                # (N,3)
            q_nom = np.tile(q_now, (self.N, 1))                          # (N,6)
        else:
            c = self._tcp_ws_jacobians[indices]                          # (N,3,6)
            q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
            p_nom = self._tcp_ws_nominal[indices]                        # (N,3)

        rows = np.zeros((3 * self.N, self.nu), dtype=float)
        lower = np.empty(3 * self.N, dtype=float)
        upper = np.empty(3 * self.N, dtype=float)
        for axis in range(3):
            for j in range(self.N):
                stage_rows = slice(j * self.n, j * self.n + 6)
                row_idx = axis * self.N + j
                rows[row_idx, :] = c[j, axis, :] @ self.Sc[stage_rows, :]
                offset = (
                    p_nom[j, axis] - c[j, axis, :] @ q_nom[j] + c[j, axis, :] @ free_cmd[j]
                )
                lower[row_idx] = self._tcp_ws_min[axis] - offset
                upper[row_idx] = self._tcp_ws_max[axis] - offset
        self.A[self._tcp_ws_row_start:self._tcp_ws_row_start + 3 * self.N, :] = rows
        return lower, upper

    def _configure_magnet_workspace_clearance(
        self, *, gain: float, soft_margin_m: float,
    ) -> None:
        if self._magnet_ws_z_min is None:
            raise ValueError(
                "magnet_workspace_clearance requires magnet_workspace to be "
                "configured too (this term reuses its (z_nom, jacobians, "
                "z_min, z_max) schedule unchanged; it never derives its own "
                "geometry)."
            )
        gain = float(gain)
        soft_margin = float(soft_margin_m)
        if gain < 0.0 or not np.isfinite(gain):
            raise ValueError("magnet_workspace_clearance['gain'] must be finite and >= 0.")
        if soft_margin <= 0.0 or not np.isfinite(soft_margin):
            raise ValueError("magnet_workspace_clearance['soft_margin_m'] must be finite and > 0.")
        if 2.0 * soft_margin >= (self._magnet_ws_z_max - self._magnet_ws_z_min):
            raise ValueError(
                f"magnet_workspace_clearance['soft_margin_m']={soft_margin} is too large: "
                f"the floor and ceiling soft bands would overlap/invert given hard bounds "
                f"[{self._magnet_ws_z_min},{self._magnet_ws_z_max}]."
            )
        self._magnet_ws_clear_gain = gain
        self._magnet_ws_clear_soft_margin = soft_margin

    def _magnet_workspace_clearance_qp_terms(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        """Soft, two-sided clearance cost around the z-workspace hard bound
        -- see this class's docstring note above `_magnet_ws_clear_gain`'s
        init for why. Mirrors `_magnet_exclusion_clearance_qp_terms`'s own
        derivation exactly, just with "distance from ceiling" (z_max - z)
        and "distance from floor" (z - z_min) in place of the single
        wall-distance quantity there -- each only incurs a cost when its
        own distance drops below `soft_margin_m`.
        """
        zero = (np.zeros((self.nu, self.nu), dtype=float), np.zeros(self.nu, dtype=float))
        if not self._magnet_ws_clear_gain:
            return zero
        indices = self._reference_indices(control_index, future=True)
        c = self._magnet_ws_jacobians[indices]                          # (N,6)
        q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
        z_nom = self._magnet_ws_nominal_z[indices]                       # (N,)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

        hessian = np.zeros((self.nu, self.nu), dtype=float)
        linear = np.zeros(self.nu, dtype=float)
        margin = self._magnet_ws_clear_soft_margin
        gain = self._magnet_ws_clear_gain
        for j in range(self.N):
            stage_rows = slice(j * self.n, j * self.n + 6)
            a = c[j] @ self.Sc[stage_rows, :]                            # (nu,): d(z_j)/du
            z_hat = z_nom[j] - c[j] @ q_nom[j] + c[j] @ free_cmd[j]

            d_ceil = self._magnet_ws_z_max - z_hat  # distance below ceiling at u=0
            if d_ceil < margin:
                b_ceil = margin - d_ceil  # > 0
                # cost = gain*(margin - (z_max - z_j(u)))^2 = gain*(b_ceil + a@u)^2
                hessian += 2.0 * gain * np.outer(a, a)
                linear += 2.0 * gain * b_ceil * a

            d_floor = z_hat - self._magnet_ws_z_min  # distance above floor at u=0
            if d_floor < margin:
                b_floor = margin - d_floor  # > 0
                # cost = gain*(margin - (z_j(u) - z_min))^2 = gain*(b_floor - a@u)^2
                hessian += 2.0 * gain * np.outer(a, a)
                linear += -2.0 * gain * b_floor * a
        return hessian, linear

    # ------------------------------------------------------------------
    # magnetic force/torque actuation preference (2026-10-15): a SOFT
    # tie-breaker, not a constraint -- see module-level design note in
    # `simulation/magnetic_beam/force_preference.py`. For a rigid point
    # dipole, aligning the moment with the local field drives TORQUE toward
    # zero while FORCE depends on the field gradient too -- the two are not
    # generally maximized by the same configuration, so this term tracks
    # F_des and tau_des SEPARATELY (with independent weights) rather than
    # folding them into one "alignment" score. The task cost (beam-tip
    # tracking) remains the primary objective; this only breaks ties among
    # configurations the task cost is otherwise indifferent between.
    # ------------------------------------------------------------------
    def _configure_magnetic_force_preference(
        self, *, force_jacobians: Any, torque_jacobians: Any,
        nominal_force_N: Any, nominal_torque_Nm: Any,
        force_desired_N: Any, torque_desired_Nm: Any,
        lambda_force: float, lambda_torque: float,
        weight_force: Any = None, weight_torque: Any = None,
    ) -> None:
        """Linearized F_mag(q,L)/tau_mag(q,L) schedule, one row-set per
        reference sample -- same fixed-schedule-with-command-stack-
        reanchoring architecture as `_configure_magnet_exclusion`/
        `_configure_magnet_workspace`/`_configure_tcp_workspace` above, just
        with the resultant magnetic wrench on the whole beam (see
        `simulation.magnetic_beam.force_preference.total_magnetic_wrench`)
        in place of a position. `force_jacobians`/`torque_jacobians`:
        (S,3,7) d(F_total)/d[q1..q6,L] and d(tau_total)/d[q1..q6,L] at each
        reference sample's own nominal state (built OFFLINE, e.g. via
        `simulation.magnetic_beam.force_preference.
        magnetic_wrench_joint_jacobian` -- a finite-difference Jacobian of
        the beam model's own resultant wrench, NOT yet chain-rule-validated
        against an analytic derivative the way this project's tip-position
        Jacobians are; appropriate for a soft tie-breaker, not a safety
        constraint). `nominal_force_N`/`nominal_torque_Nm`: (S,3) the
        resultant wrench AT that same nominal state. Anchored every solve to
        the actual COMMANDED state via (Ec, Sc) -- deliberately NOT the
        delayed physical prediction (Ep, Sp) the beam-tip-tracking cost
        uses: the magnet's pose equals what is commanded (the robot
        realizes joint commands with negligible lag relative to this
        control rate, exactly why magnet_exclusion/workspace/tcp_workspace
        all use the command stack too), and the wrench model itself is a
        QUASISTATIC equilibrium at a given q -- it has no separate notion of
        "the beam's actual, still-lagging shape" to linearize about, so
        there is no more-correct alternative anchor available here.
        """
        force_jacobians = np.asarray(force_jacobians, dtype=float)
        torque_jacobians = np.asarray(torque_jacobians, dtype=float)
        nominal_force_N = np.asarray(nominal_force_N, dtype=float)
        nominal_torque_Nm = np.asarray(nominal_torque_Nm, dtype=float)
        expected_j = (self.reference.sample_count, 3, self.n)
        expected_f = (self.reference.sample_count, 3)
        for name, arr, expected in (
            ("force_jacobians", force_jacobians, expected_j),
            ("torque_jacobians", torque_jacobians, expected_j),
            ("nominal_force_N", nominal_force_N, expected_f),
            ("nominal_torque_Nm", nominal_torque_Nm, expected_f),
        ):
            if arr.shape != expected or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"magnetic_force_preference[{name!r}] must have shape "
                    f"{expected}; received {arr.shape}."
                )
        lambda_force = float(lambda_force)
        lambda_torque = float(lambda_torque)
        if lambda_force < 0.0 or not np.isfinite(lambda_force):
            raise ValueError("magnetic_force_preference['lambda_force'] must be finite and >= 0.")
        if lambda_torque < 0.0 or not np.isfinite(lambda_torque):
            raise ValueError("magnetic_force_preference['lambda_torque'] must be finite and >= 0.")

        self._force_pref_force_jacobians = force_jacobians
        self._force_pref_torque_jacobians = torque_jacobians
        self._force_pref_nominal_force = nominal_force_N
        self._force_pref_nominal_torque = nominal_torque_Nm
        self._force_pref_F_des = np.asarray(force_desired_N, dtype=float).reshape(3)
        self._force_pref_tau_des = np.asarray(torque_desired_Nm, dtype=float).reshape(3)
        self._force_pref_lambda_F = lambda_force
        self._force_pref_lambda_tau = lambda_torque
        self._force_pref_W_F = (
            np.eye(3) if weight_force is None else np.asarray(weight_force, dtype=float).reshape(3, 3)
        )
        self._force_pref_W_tau = (
            np.eye(3) if weight_torque is None else np.asarray(weight_torque, dtype=float).reshape(3, 3)
        )

    def _magnetic_force_preference_qp_terms(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        """cost_j = lambda_F*(F_hat_j-F_des)^T W_F (F_hat_j-F_des)
                   + lambda_tau*(tau_hat_j-tau_des)^T W_tau (tau_hat_j-tau_des),
        summed over the horizon, where F_hat_j/tau_hat_j are the first-order
        wrench estimate at this stage's own COMMANDED state. Derivation
        mirrors `_magnet_exclusion_clearance_qp_terms`'s own (linear-in-u
        residual -> quadratic QP terms), just with a (3,3) weight matrix
        and two residuals (force, torque) instead of one scalar distance.
        """
        zero = (np.zeros((self.nu, self.nu), dtype=float), np.zeros(self.nu, dtype=float))
        if not self._force_pref_lambda_F and not self._force_pref_lambda_tau:
            return zero
        indices = self._reference_indices(control_index, future=True)
        q_nom = np.asarray(self.reference.state, dtype=float)[indices]        # (N,7)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)                  # (N,7)
        c_F_all = self._force_pref_force_jacobians[indices]                   # (N,3,7)
        nom_F_all = self._force_pref_nominal_force[indices]                    # (N,3)
        c_tau_all = self._force_pref_torque_jacobians[indices]                # (N,3,7)
        nom_tau_all = self._force_pref_nominal_torque[indices]                 # (N,3)

        hessian = np.zeros((self.nu, self.nu), dtype=float)
        linear = np.zeros(self.nu, dtype=float)
        for j in range(self.N):
            stage_rows = slice(j * self.n, (j + 1) * self.n)
            delta_q = free_cmd[j] - q_nom[j]
            if self._force_pref_lambda_F:
                c_F = c_F_all[j]                                               # (3,7)
                A = c_F @ self.Sc[stage_rows, :]                               # (3,nu)
                const = nom_F_all[j] + c_F @ delta_q - self._force_pref_F_des
                WA = self._force_pref_W_F @ A
                hessian += 2.0 * self._force_pref_lambda_F * (A.T @ WA)
                linear += 2.0 * self._force_pref_lambda_F * (A.T @ (self._force_pref_W_F @ const))
            if self._force_pref_lambda_tau:
                c_tau = c_tau_all[j]                                          # (3,7)
                A = c_tau @ self.Sc[stage_rows, :]                             # (3,nu)
                const = nom_tau_all[j] + c_tau @ delta_q - self._force_pref_tau_des
                WA = self._force_pref_W_tau @ A
                hessian += 2.0 * self._force_pref_lambda_tau * (A.T @ WA)
                linear += 2.0 * self._force_pref_lambda_tau * (A.T @ (self._force_pref_W_tau @ const))
        return hessian, linear

    # ------------------------------------------------------------------
    # dipole-to-point alignment preference (2026-10-19): a SOFT tie-
    # breaker like magnetic_force_preference above, but purely KINEMATIC
    # (see controllers.dipole_alignment's module docstring) -- cheap
    # enough to relinearize fresh EVERY SOLVE, unconditionally, no
    # schedule file at all. One (r_src, R_src)-based (3,6) Jacobian is
    # computed at the CURRENT commanded joints (free_cmd[0]) and reused
    # across the whole horizon, exactly the same live-relinearize pattern
    # `_configure_tcp_workspace`/`_configure_magnet_workspace` already use
    # for their own hard constraints -- the only difference here is this
    # is a cost, not a constraint, so there is no separate "schedule vs
    # live" choice to make: live is simply always cheap enough.
    # ------------------------------------------------------------------
    def _configure_dipole_alignment_preference(
        self, *, dh: Any, T_F_M: Any, m_body: Any, target_point_m: Any,
        weight: float, sign: float = 1.0,
    ) -> None:
        weight = float(weight)
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("dipole_alignment_preference['weight'] must be finite and >= 0.")
        if abs(float(sign)) != 1.0:
            raise ValueError("dipole_alignment_preference['sign'] must be exactly +1.0 or -1.0.")
        self._dipole_align_dh = dh
        self._dipole_align_T_F_M = np.asarray(T_F_M, dtype=float)
        self._dipole_align_m_body = np.asarray(m_body, dtype=float).reshape(3)
        self._dipole_align_target_point_m = np.asarray(target_point_m, dtype=float).reshape(3)
        self._dipole_align_weight = weight
        self._dipole_align_sign = float(sign)

    def _dipole_alignment_preference_qp_terms(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        zero = (np.zeros((self.nu, self.nu), dtype=float), np.zeros(self.nu, dtype=float))
        if not self._dipole_align_weight:
            return zero
        from proper_research.controllers.dipole_alignment import (
            dipole_alignment_residual_and_jacobian_from_q,
        )

        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]          # (N,6)
        q_now = free_cmd[0]
        e_now, c = dipole_alignment_residual_and_jacobian_from_q(
            q_now, self._dipole_align_dh, self._dipole_align_T_F_M,
            self._dipole_align_m_body, self._dipole_align_target_point_m,
            sign=self._dipole_align_sign,
        )  # e_now: (3,), c: (3,6), both evaluated once at q_now, reused every stage

        hessian = np.zeros((self.nu, self.nu), dtype=float)
        linear = np.zeros(self.nu, dtype=float)
        weight = self._dipole_align_weight
        for j in range(self.N):
            stage_rows = slice(j * self.n, j * self.n + 6)
            A = c @ self.Sc[stage_rows, :]                                    # (3,nu)
            const = e_now + c @ (free_cmd[j] - q_now)                         # (3,)
            hessian += 2.0 * weight * (A.T @ A)
            linear += 2.0 * weight * (A.T @ const)
        return hessian, linear

    # ------------------------------------------------------------------
    # overridden prediction-dependent pieces
    # ------------------------------------------------------------------
    def _constraint_bounds_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
    ) -> tuple[Array, Array]:
        """Same as the base class's `_constraint_bounds`, EXCEPT the joint-
        position box uses (Ec, Sc) -- the COMMAND stack -- not the physical
        prediction. Velocity/rate terms are untouched (still act on the
        decision v and previous_input exactly as before). Appends the
        magnet-exclusion rows (see `_magnet_exclusion_bounds`) when
        `magnet_exclusion` was configured at construction, magnet
        z-workspace rows (see `_magnet_workspace_bounds`) when
        `magnet_workspace` was configured, and TCP Cartesian-workspace
        rows (see `_tcp_workspace_bounds`) when `tcp_workspace` was
        configured; all are no-ops (0-length concatenation) otherwise,
        so every existing caller that never passes any of them is
        completely unaffected."""
        previous_input = _finite_vector(previous_input, self.m, "previous_input")
        input_lower = np.tile(-self.velocity_limit, self.N)
        input_upper = np.tile(self.velocity_limit, self.N)
        free_cmd = self.Ec @ x_exec
        state_lower = np.tile(self.state_min, self.N) - free_cmd
        state_upper = np.tile(self.state_max, self.N) - free_cmd
        change = np.zeros(self.nu, dtype=float)
        change[: self.m] = previous_input
        delta = np.tile(self.dt * self.acceleration_limit, self.N)
        increment_lower = change - delta
        increment_upper = change + delta
        magnet_lower, magnet_upper = self._magnet_exclusion_bounds(
            x_exec=x_exec, control_index=control_index,
        )
        ws_lower, ws_upper = self._magnet_workspace_bounds(
            x_exec=x_exec, control_index=control_index,
        )
        tcp_lower, tcp_upper = self._tcp_workspace_bounds(
            x_exec=x_exec, control_index=control_index,
        )
        return (
            np.concatenate((input_lower, state_lower, increment_lower, magnet_lower, ws_lower, tcp_lower)),
            np.concatenate((input_upper, state_upper, increment_upper, magnet_upper, ws_upper, tcp_upper)),
        )

    def _linear_cost_exec(
        self, *, x_exec: Array, previous_input: Array,
        state_reference: Array, input_reference: Array,
    ) -> Array:
        free_state = self.Ep @ x_exec  # PHYSICAL stack, not command stack
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input
        return 2.0 * (
            self.Sp.T @ self.Qbar @ (free_state - state_reference_vector)
            - self.Rbar @ input_reference_vector
            - self.D.T @ self.Rdbar @ previous_vector
        )

    def _beam_prediction_terms_exec(
        self, *, x_exec: Array, control_index: int, estimated_residual: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        indices = self._reference_indices(control_index, future=True)
        state_reference = np.asarray(self.reference.state, dtype=float)[indices]
        desired_position = np.asarray(self.reference.desired_position_m, dtype=float)[indices]
        nominal_position = self.nominal_reference_positions_m[indices]
        jacobians = self.reference_position_jacobians[indices]
        Jbar = sp.block_diag(list(jacobians), format="csc").toarray()
        state_reference_vector = state_reference.reshape(self.nu)
        free_state = self.Ep @ x_exec  # PHYSICAL stack
        G = Jbar @ self.Sp
        constant_error = (
            nominal_position.reshape(3 * self.N)
            - desired_position.reshape(3 * self.N)
            + Jbar @ (free_state - state_reference_vector)
            + self.beta_d * np.tile(estimated_residual, self.N)
        )
        return G, constant_error, state_reference, desired_position, nominal_position, Jbar

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)
        base_linear = self._linear_cost_exec(
            x_exec=x_exec, previous_input=previous_input,
            state_reference=state_reference, input_reference=input_reference,
        )
        G, constant_error, _, desired_position, nominal_position, Jbar = (
            self._beam_prediction_terms_exec(
                x_exec=x_exec, control_index=control_index,
                estimated_residual=estimated_residual,
            )
        )
        hessian = self._base_hessian + 2.0 * (G.T @ self.Qpbar @ G)
        linear = base_linear + 2.0 * (G.T @ self.Qpbar @ constant_error)
        if self._magnet_clear_gain:
            clear_hessian, clear_linear = self._magnet_exclusion_clearance_qp_terms(
                x_exec=x_exec, control_index=control_index,
            )
            hessian = hessian + clear_hessian
            linear = linear + clear_linear
        if self._magnet_ws_clear_gain:
            ws_clear_hessian, ws_clear_linear = self._magnet_workspace_clearance_qp_terms(
                x_exec=x_exec, control_index=control_index,
            )
            hessian = hessian + ws_clear_hessian
            linear = linear + ws_clear_linear
        if self._force_pref_lambda_F or self._force_pref_lambda_tau:
            force_pref_hessian, force_pref_linear = self._magnetic_force_preference_qp_terms(
                x_exec=x_exec, control_index=control_index,
            )
            hessian = hessian + force_pref_hessian
            linear = linear + force_pref_linear
        if self._dipole_align_weight:
            dipole_hessian, dipole_linear = self._dipole_alignment_preference_qp_terms(
                x_exec=x_exec, control_index=control_index,
            )
            hessian = hessian + dipole_hessian
            linear = linear + dipole_linear
        hessian = 0.5 * (hessian + hessian.T)
        return hessian, linear, {
            "G": G, "constant_error": constant_error,
            "desired_position": desired_position, "nominal_position": nominal_position,
            "Jbar": Jbar, "input_reference": input_reference,
        }

    # ------------------------------------------------------------------
    # public entry point -- deliberately NOT `measured_state` alone
    # ------------------------------------------------------------------
    def solve_delay_aware(
        self,
        *,
        z_meas: Any,
        q_cmd: Any,
        q_cmd_prev: Any,
        insertion_m: float,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> DelayAwareBeamOutputMPCStep:
        """`z_meas`: [q_meas(6), L(1)] -- feeds ONLY the disturbance estimate.
        `q_cmd`/`q_cmd_prev`: the execution-C accumulator's OWN current and
        previous commanded joints -- feeds ONLY the prediction, via x_exec.
        `previous_input`: u_applied,k-1 -- feeds ONLY rate continuity."""
        z_meas = _finite_vector(z_meas, 7, "z_meas")
        q_cmd = _finite_vector(q_cmd, 6, "q_cmd")
        q_cmd_prev = _finite_vector(q_cmd_prev, 6, "q_cmd_prev")
        beam_position = _finite_vector(measured_beam_position, 3, "measured_beam_position")
        previous = _finite_vector(previous_input, 7, "previous_input")
        x_exec = np.concatenate([q_cmd, q_cmd_prev, [float(insertion_m)]])

        instantaneous, estimated = self._estimate_output_residual(
            measured_state=z_meas, measured_beam_position=beam_position,
            control_index=control_index,
        )
        hessian, linear_cost, beam_terms = self._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=previous, control_index=control_index,
            estimated_residual=estimated,
        )
        input_reference = beam_terms["input_reference"]
        lower, upper = self._constraint_bounds_exec(
            x_exec=x_exec, previous_input=previous, control_index=control_index,
        )
        warm_start = self._feasible_warm_start(
            input_reference=input_reference, previous_input=previous,
        )

        if self.backend == "osqp":
            assert self._solver is not None
            update_kwargs = dict(Px=self._upper_values(hessian), q=linear_cost, l=lower, u=upper)
            if self._magnet_excl_radius is not None or self._magnet_ws_z_min is not None:
                update_kwargs["Ax"] = self._a_values(self.A)
            self._solver.update(**update_kwargs)
            self._solver.warm_start(x=warm_start)
            result = self._solver.solve()
            status = str(result.info.status).lower()
            success = status in {"solved", "solved inaccurate"}
            # See ConfigurationMPCConfig.accept_time_limit_solution's
            # docstring (2026-09-29 fix) -- this class has its own,
            # separate OSQP call from simulate_time_parameterized_beam_
            # output_mpc.py's base-class one (same fix applied there too),
            # so needs the same opt-in acceptance of a solver_time_limit_s
            # early exit repeated here.
            if (
                not success and status == "run time limit reached"
                and result.x is not None
                and bool(getattr(self.config, "accept_time_limit_solution", False))
            ):
                success = True
            solution = None if result.x is None else np.asarray(result.x, dtype=float).reshape(self.nu)
            diagnostic = {
                "status": status, "success": success,
                "iterations": int(result.info.iter), "solve_time_s": float(result.info.run_time),
                "primal_residual": float(getattr(result.info, "prim_res", getattr(result.info, "pri_res", np.nan))),
                "dual_residual": float(getattr(result.info, "dual_res", getattr(result.info, "dua_res", np.nan))),
                # (2026-09-28 instrumentation pass) raw OSQP dual variables --
                # already computed by `.solve()`, just read out here. Row
                # ordering matches `self.A` (input-rate, state-box,
                # increment, then magnet-exclusion rows if configured -- see
                # `_constraint_bounds_exec`); mapping row index to semantic
                # constraint name is NOT done here, left for post-hoc
                # analysis against that same row layout.
                "dual_y": None if result.y is None else np.asarray(result.y, dtype=float).copy(),
            }
        else:
            solution, diagnostic = self._solve_scipy_dynamic(
                hessian=hessian, linear_cost=linear_cost, lower=lower, upper=upper,
                warm_start=warm_start,
            )

        if solution is None or not bool(diagnostic["success"]):
            command = np.zeros(7, dtype=float)
            predicted_inputs = np.zeros((self.N, self.m), dtype=float)
            predicted_states = np.tile(np.concatenate([q_cmd, [insertion_m]]), (self.N, 1))
            predicted_commands = predicted_states.copy()
            objective = np.inf
        else:
            predicted_inputs = solution.reshape(self.N, self.m)
            predicted_states = (self.Ep @ x_exec + self.Sp @ solution).reshape(self.N, self.n)
            predicted_commands = (self.Ec @ x_exec + self.Sc @ solution).reshape(self.N, self.n)
            command = predicted_inputs[0].copy()
            objective = float(0.5 * solution @ hessian @ solution + linear_cost @ solution)
            self._warm_start = solution.copy()

        predicted_beam_vector = (
            beam_terms["nominal_position"].reshape(3 * self.N)
            + beam_terms["Jbar"] @ (
                predicted_states.reshape(self.nu)
                - self.reference.state_window(control_index, self.N).reshape(self.nu)
            )
            + self.beta_d * np.tile(estimated, self.N)
        )
        predicted_beam = predicted_beam_vector.reshape(self.N, 3)
        predicted_error = predicted_beam - beam_terms["desired_position"]
        first_error = float(np.linalg.norm(predicted_error[0]))
        # Current-tick (3, n) Jacobian: block_diag's first block is exactly
        # jacobians[0] from `_beam_prediction_terms_exec` -- a read of the
        # already-built Jbar, not a new computation.
        jacobian_used = np.asarray(beam_terms["Jbar"][:3, : self.n], dtype=float).copy()

        return DelayAwareBeamOutputMPCStep(
            command=command,
            planned_input=np.asarray(input_reference[0], dtype=float).copy(),
            predicted_states=predicted_states,
            predicted_commands=predicted_commands,
            jacobian_used=jacobian_used,
            dual_y=diagnostic.get("dual_y"),
            predicted_inputs=predicted_inputs,
            objective=objective,
            status=str(diagnostic["status"]), success=bool(diagnostic["success"]),
            iterations=int(diagnostic["iterations"]), solve_time_s=float(diagnostic["solve_time_s"]),
            primal_residual=float(diagnostic["primal_residual"]), dual_residual=float(diagnostic["dual_residual"]),
            measured_beam_position=beam_position.copy(),
            instantaneous_output_residual=instantaneous.copy(),
            estimated_output_residual=estimated.copy(),
            predicted_beam_positions=predicted_beam,
            predicted_beam_errors=predicted_error,
            first_predicted_beam_error_m=first_error,
        )
