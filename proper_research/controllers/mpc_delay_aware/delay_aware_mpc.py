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
        if magnet_exclusion is not None:
            self._configure_magnet_exclusion(**magnet_exclusion)

        self._magnet_ws_z_min: float | None = None
        self._magnet_ws_z_max: float | None = None
        if magnet_workspace is not None:
            self._configure_magnet_workspace(**magnet_workspace)

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
            if self._magnet_excl_radius is not None or self._magnet_ws_z_min is not None:
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
        rather than assumed -- `self.A` may carry ZERO, ONE (exclusion
        only, workspace only) or TWO (both) extra N-row blocks by this
        point, appended in `_configure_magnet_exclusion`/`_configure_
        magnet_workspace` (called, if at all, before this setup runs --
        see `__init__`), so hardcoding one block's worth of dummy rows
        here would silently under/over-size `l`/`u` whenever both (or
        neither, in a future refactor) are active at once."""
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
        indices = self._reference_indices(control_index, future=True)
        J = self._magnet_excl_jacobians[indices]                       # (N,3,6)
        q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
        d_nom = self._magnet_excl_d_nom[indices]                        # (N,)
        normal = self._magnet_excl_normal[indices]                      # (N,3)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

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
    # magnet z-workspace bounds, wired directly into the QP (2026-09-28,
    # same day as the point-exclusion constraint above, closing the
    # identical blind spot for general magnet-workspace excursions rather
    # than only wall-approach)
    # ------------------------------------------------------------------
    def _configure_magnet_workspace(
        self, *, position_jacobians: Any, nominal_positions_m: Any,
        z_min_m: float, z_max_m: float,
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

        placeholder = np.full((self.N, self.nu), 1.0e-30, dtype=float)
        self._magnet_ws_row_start = self.A.shape[0]
        self.A = np.vstack((self.A, placeholder))

    def _magnet_workspace_bounds(
        self, *, x_exec: Array, control_index: int,
    ) -> tuple[Array, Array]:
        if self._magnet_ws_z_min is None:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        indices = self._reference_indices(control_index, future=True)
        c = self._magnet_ws_jacobians[indices]                          # (N,6)
        q_nom = np.asarray(self.reference.state, dtype=float)[indices][:, :6]  # (N,6)
        z_nom = self._magnet_ws_nominal_z[indices]                       # (N,)
        free_cmd = (self.Ec @ x_exec).reshape(self.N, self.n)[:, :6]     # (N,6)

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
        `magnet_exclusion` was configured at construction, and magnet
        z-workspace rows (see `_magnet_workspace_bounds`) when
        `magnet_workspace` was configured; both are no-ops (0-length
        concatenation) otherwise, so every existing caller that never
        passes either is completely unaffected."""
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
        return (
            np.concatenate((input_lower, state_lower, increment_lower, magnet_lower, ws_lower)),
            np.concatenate((input_upper, state_upper, increment_upper, magnet_upper, ws_upper)),
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
