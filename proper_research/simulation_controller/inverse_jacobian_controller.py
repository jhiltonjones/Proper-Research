from __future__ import annotations

from typing import Literal

import numpy as np

# Adjust this import to the module containing your existing controller.
from proper_research.simulation_controller.controller import (
    MPCControllerTipXY,
)


SequenceMode = Literal[
    "held",
    "rollout_lti",
    "rollout_ltv",
]


class InverseJacobianControllerTipXY(MPCControllerTipXY):
    """
    Drop-in inverse-Jacobian baseline.

    It reuses the existing controller's:
        - reference selection,
        - actuator-pose integration,
        - nonlinear forward model,
        - open-loop command application,
        - logging interface.

    It replaces only the MPC/SQP sequence optimiser with a local
    weighted damped-least-squares inverse-Jacobian policy.

    Parameters
    ----------
    sequence_mode:
        "held":
            Compute one command and repeat it over the horizon.

        "rollout_lti":
            Roll out the feedback policy on predicted states while using
            the initial Jacobian at every stage.

        "rollout_ltv":
            Roll out the feedback policy and recompute the Jacobian at
            every predicted actuator configuration.

    kp:
        Scalar or output-sized vector. It maps output error into desired
        one-step output displacement.

    desired_step_max:
        Optional maximum norm of the desired one-step tip displacement,
        in metres. This limits the task-space request before inversion.

    extra_damping:
        Additional nonnegative diagonal damping added to the existing
        input-effort matrix R.
    """

    def __init__(
        self,
        *args,
        sequence_mode: SequenceMode = "rollout_ltv",
        kp: float | np.ndarray = 1,
        desired_step_max: float | None = None,
        extra_damping: float | np.ndarray = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        valid_modes = {
            "held",
            "rollout_lti",
            "rollout_ltv",
        }
        if sequence_mode not in valid_modes:
            raise ValueError(
                f"Unknown sequence_mode={sequence_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )

        self.sequence_mode = sequence_mode

        kp_arr = np.asarray(kp, dtype=float).reshape(-1)
        if kp_arr.size == 1:
            kp_arr = np.full(self.n, float(kp_arr[0]))
        if kp_arr.size != self.n:
            raise ValueError(
                f"kp must be scalar or length {self.n}, "
                f"got length {kp_arr.size}."
            )
        if np.any(kp_arr < 0.0):
            raise ValueError("kp entries must be nonnegative.")

        self.Kp_ij = np.diag(kp_arr)

        if desired_step_max is not None and desired_step_max <= 0.0:
            raise ValueError("desired_step_max must be positive or None.")

        self.desired_step_max = (
            None
            if desired_step_max is None
            else float(desired_step_max)
        )
        print(f"Desired step is {self.desired_step_max}")
        damping = np.asarray(extra_damping, dtype=float).reshape(-1)
        if damping.size == 1:
            damping = np.full(self.m, float(damping[0]))
        if damping.size != self.m:
            raise ValueError(
                f"extra_damping must be scalar or length {self.m}, "
                f"got length {damping.size}."
            )
        if np.any(damping < 0.0):
            raise ValueError("extra_damping must be nonnegative.")

        self.extra_damping_matrix = np.diag(damping)

        self._last_inverse_jacobian_info: dict = {}

    def _desired_output_increment(
        self,
        *,
        x_now: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
        """Construct the desired one-step tip displacement."""
        x_now = np.asarray(x_now, float).reshape(self.n)
        x_ref = np.asarray(x_ref, float).reshape(self.n)

        error = x_ref - x_now
        delta_des = self.Kp_ij @ error

        if self.desired_step_max is not None:
            norm_delta = float(np.linalg.norm(delta_des))
            if norm_delta > self.desired_step_max:
                delta_des *= self.desired_step_max / max(
                    norm_delta,
                    1e-12,
                )

        return delta_des

    def _solve_damped_inverse(
        self,
        *,
        B: np.ndarray,
        delta_des: np.ndarray,
        u_previous: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Solve:

            min_u ||B u - delta_des||_Q^2
                  + ||u||_R^2
                  + ||u-u_previous||_Rd^2.
        """
        B = np.asarray(B, float).reshape(self.n, self.m)
        delta_des = np.asarray(delta_des, float).reshape(self.n)
        u_previous = np.asarray(
            u_previous,
            float,
        ).reshape(self.m)

        Q = np.asarray(self.Q, float).reshape(self.n, self.n)
        R = np.asarray(self.R, float).reshape(self.m, self.m)
        Rd = np.asarray(self.Rd, float).reshape(self.m, self.m)

        H = (
            B.T @ Q @ B
            + R
            + Rd
            + self.extra_damping_matrix
        )
        rhs = (
            B.T @ Q @ delta_des
            + Rd @ u_previous
        )

        # Symmetrise against small numerical asymmetries.
        H = 0.5 * (H + H.T)

        try:
            u = np.linalg.solve(H, rhs)
            solve_method = "solve"
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(
                H,
                rhs,
                rcond=None,
            )[0]
            solve_method = "lstsq"

        u = np.asarray(u, float).reshape(self.m)

        if not np.all(np.isfinite(u)):
            raise FloatingPointError(
                "Inverse-Jacobian solve produced non-finite input."
            )

        # Apply per-channel physical command bounds.
        u_max = np.asarray(self.u_max, float).reshape(self.m)
        finite = np.isfinite(u_max)

        u_bounded = u.copy()
        u_bounded[finite] = np.clip(
            u_bounded[finite],
            -u_max[finite],
            u_max[finite],
        )

        # Apply the dedicated insertion-rate limits as well.
        dL_idx = int(getattr(self, "dL_index", 6))
        if 0 <= dL_idx < self.m:
            lower_dL = -float(
                getattr(self, "dL_back_max", np.inf)
            )
            upper_dL = float(
                getattr(self, "dL_fwd_max", np.inf)
            )
            u_bounded[dL_idx] = np.clip(
                u_bounded[dL_idx],
                lower_dL,
                upper_dL,
            )

        diag = {
            "ij_solve_method": solve_method,
            "ij_H_cond": float(np.linalg.cond(H)),
            "ij_B_cond": self._safe_condition_number(B),
            "ij_unbounded_norm": float(np.linalg.norm(u)),
            "ij_bounded_norm": float(np.linalg.norm(u_bounded)),
            "ij_saturated": int(
                np.any(np.abs(u_bounded - u) > 1e-12)
            ),
            "ij_task_residual_norm": float(
                np.linalg.norm(B @ u_bounded - delta_des)
            ),
        }

        return u_bounded, diag

    @staticmethod
    def _safe_condition_number(B: np.ndarray) -> float:
        B = np.asarray(B, float)
        try:
            singular_values = np.linalg.svd(
                B,
                compute_uv=False,
            )
        except np.linalg.LinAlgError:
            return np.inf

        if singular_values.size == 0:
            return np.inf

        return float(
            singular_values[0]
            / max(singular_values[-1], 1e-12)
        )

    def _reference_array(
        self,
        idx_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Reuse the MPC reference builder so the baseline receives exactly
        the same reference points as the MPC.
        """
        X_ref = np.asarray(
            self._build_reference_stack(idx_ref),
            float,
        )

        return X_ref.reshape(self.Np, self.n)

    def _predicted_pose_at_stage(
        self,
        *,
        p0: np.ndarray,
        U_partial: np.ndarray,
        stage: int,
    ) -> np.ndarray:
        """
        Obtain actuator pose at the start of `stage`.

        The existing RolloutMixin performs the exact same actuator
        integration used by MPC, including quaternion updates.

        U_partial contains already selected controls followed by zeros.
        """
        U_partial = np.asarray(
            U_partial,
            float,
        ).reshape(self.Np, self.m)

        p_nodes = self._p_nodes_from_U(
            p0,
            U_partial,
        )

        return np.asarray(
            p_nodes[stage],
            float,
        ).reshape(self.np)

    def _generate_held_sequence(
        self,
        *,
        p0: np.ndarray,
        x0: np.ndarray,
        X_ref: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        """Compute one inverse-Jacobian command and repeat it."""
        B0 = np.asarray(
            self.Jxy_fn(p0),
            float,
        ).reshape(self.n, self.m)

        delta_des = self._desired_output_increment(
            x_now=x0,
            x_ref=X_ref[0],
        )

        u0, diag = self._solve_damped_inverse(
            B=B0,
            delta_des=delta_des,
            u_previous=self.u_prev,
        )

        diag.update({
            "ij_stage": 0,
            "ij_error_norm": float(
                np.linalg.norm(X_ref[0] - x0)
            ),
            "ij_desired_step_norm": float(
                np.linalg.norm(delta_des)
            ),
        })

        U_seq = np.tile(
            u0.reshape(1, self.m),
            (self.Np, 1),
        )
        diag.update(
            {
                "B_matrix": B0.copy(),
                "B_column_norm_xyz": np.linalg.norm(
                    B0,
                    axis=0,
                ),
                "B_column_norm_xy": np.linalg.norm(
                    B0[:2, :],
                    axis=0,
                ),
            }
        )
        return U_seq, [diag]

    def _generate_policy_rollout(
        self,
        *,
        p0: np.ndarray,
        x0: np.ndarray,
        X_ref: np.ndarray,
        use_ltv_jacobian: bool,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Roll out the inverse-Jacobian feedback law on predicted states.

        Each stage performs:

            1. Predict the actuator pose at the start of the stage.
            2. Evaluate the predicted tip state.
            3. Evaluate or select the stage Jacobian.
            4. Compute the desired tip displacement.
            5. Solve the inverse-Jacobian allocation.
            6. Store the command before predicting the next stage.
        """
        p0 = np.asarray(
            p0,
            float,
        ).reshape(self.np)

        x0 = np.asarray(
            x0,
            float,
        ).reshape(self.n)

        X_ref = np.asarray(
            X_ref,
            float,
        ).reshape(self.Np, self.n)

        U_seq = np.zeros(
            (self.Np, self.m),
            dtype=float,
        )

        stage_diagnostics: list[dict] = []

        # Used at every stage for the LTI rollout.
        B_fixed = np.asarray(
            self.Jxy_fn(p0),
            float,
        ).reshape(self.n, self.m)

        u_previous = np.asarray(
            self.u_prev,
            float,
        ).reshape(self.m)

        for stage in range(self.Np):
            # --------------------------------------------------------
            # Commands before this stage have been filled.
            # Future commands remain zero.
            # --------------------------------------------------------
            p_stage = self._predicted_pose_at_stage(
                p0=p0,
                U_partial=U_seq,
                stage=stage,
            )

            # --------------------------------------------------------
            # Current/predicted tip output
            # --------------------------------------------------------
            if stage == 0:
                # Use the measured/current output at stage zero.
                x_stage = x0.copy()
            else:
                # Use the nonlinear predicted output after applying
                # the previously selected prefix commands.
                x_stage = np.asarray(
                    self._eval_forward_tip(p_stage),
                    float,
                ).reshape(self.n)

            # --------------------------------------------------------
            # Stage Jacobian
            # --------------------------------------------------------
            if use_ltv_jacobian:
                B_stage = np.asarray(
                    self.Jxy_fn(p_stage),
                    float,
                ).reshape(self.n, self.m)
            else:
                B_stage = B_fixed.copy()

            # --------------------------------------------------------
            # Desired one-step tip displacement
            # --------------------------------------------------------
            delta_des = (
                self._desired_output_increment(
                    x_now=x_stage,
                    x_ref=X_ref[stage],
                )
            )

            # --------------------------------------------------------
            # Inverse-Jacobian allocation
            # --------------------------------------------------------
            u_stage, diag = (
                self._solve_damped_inverse(
                    B=B_stage,
                    delta_des=delta_des,
                    u_previous=u_previous,
                )
            )

            # Store this command before moving to the next stage.
            U_seq[stage] = u_stage
            u_previous = u_stage.copy()

            diag.update(
                {
                    "B_matrix": B_stage.copy(),

                    "B_column_norm_xyz": (
                        np.linalg.norm(
                            B_stage,
                            axis=0,
                        )
                    ),

                    "B_column_norm_xy": (
                        np.linalg.norm(
                            B_stage[:2, :],
                            axis=0,
                        )
                    ),

                    "ij_stage": int(stage),

                    "ij_error_vector": (
                        X_ref[stage] - x_stage
                    ).copy(),

                    "ij_error_norm": float(
                        np.linalg.norm(
                            X_ref[stage] - x_stage
                        )
                    ),

                    "ij_desired_step": (
                        delta_des.copy()
                    ),

                    "ij_desired_step_norm": float(
                        np.linalg.norm(delta_des)
                    ),

                    "ij_predicted_x": (
                        x_stage.copy()
                    ),

                    "ij_reference_x": (
                        X_ref[stage].copy()
                    ),

                    "ij_control_unapplied": (
                        u_stage.copy()
                    ),

                    "ij_predicted_linear_increment": (
                        B_stage @ u_stage
                    ),
                }
            )

            stage_diagnostics.append(diag)

        if len(stage_diagnostics) != self.Np:
            raise RuntimeError(
                "Inverse-Jacobian rollout generated "
                f"{len(stage_diagnostics)} stage diagnostics, "
                f"but Np={self.Np}."
            )

        return U_seq, stage_diagnostics

    def _build_prediction_diagnostics(
        self,
        *,
        p0: np.ndarray,
        U_seq: np.ndarray,
    ) -> dict:
        """
        Construct X_aff and Mc so the existing step()/simulation logging
        can still form X_pred and rollout-error diagnostics.

        At the selected U_seq, the affine expression is matched to the
        nonlinear nominal rollout.
        """
        U_seq = np.asarray(
            U_seq,
            float,
        ).reshape(self.Np, self.m)

        old_mode = self.model_mode

        try:
            if self.sequence_mode == "rollout_ltv":
                self.model_mode = "ltv"
            else:
                self.model_mode = "lti"

            p_seq, _, Mc, B0 = self._build_prediction_mats(
                p0,
                U_seq,
            )

            X_nom = np.asarray(
                [
                    self._eval_forward_tip(p_stage)
                    for p_stage in p_seq
                ],
                float,
            ).reshape(self.Np, self.n)

            U_vec = U_seq.reshape(
                self.Np * self.m,
                1,
            )
            X_nom_vec = X_nom.reshape(
                self.Np * self.n,
                1,
            )

            X_aff = X_nom_vec - Mc @ U_vec

            return {
                "X_aff": X_aff,
                "Mc": Mc,
                "B0": B0,
                "X_nom_inverse": X_nom,
            }

        finally:
            self.model_mode = old_mode
    def _linear_policy_rollout(
        self,
        *,
        p0,
        x0,
        U_seq,
        use_ltv_jacobian,
    ):
        U_seq = np.asarray(
            U_seq,
            float,
        ).reshape(self.Np, self.m)

        p_nodes = self._p_nodes_from_U(
            p0,
            U_seq,
        )

        B0 = np.asarray(
            self.Jxy_fn(p_nodes[0]),
            float,
        ).reshape(self.n, self.m)

        x_linear = np.asarray(
            x0,
            float,
        ).reshape(self.n)

        X_linear = []
        B_sequence = []

        for stage in range(self.Np):
            if use_ltv_jacobian:
                B_stage = np.asarray(
                    self.Jxy_fn(p_nodes[stage]),
                    float,
                ).reshape(self.n, self.m)
            else:
                B_stage = B0

            x_linear = (
                x_linear
                + B_stage @ U_seq[stage]
            )

            X_linear.append(x_linear.copy())
            B_sequence.append(B_stage.copy())

        return (
            np.asarray(X_linear, float),
            np.asarray(B_sequence, float),
        )
    def _solve_mpc_sequence(
        self,
        *,
        p0,
        x0,
        U_init,
        idx_ref,
        solver_mode,
    ):
        """
        Replacement for the MPC/SQP sequence optimiser.

        The name is retained only because the inherited step() method
        calls this interface.
        """
        del U_init
        del solver_mode

        p0 = np.asarray(p0, float).reshape(self.np)
        x0 = np.asarray(x0, float).reshape(self.n)

        X_ref = self._reference_array(idx_ref)

        if self.sequence_mode == "held":
            U_seq, stage_diag = self._generate_held_sequence(
                p0=p0,
                x0=x0,
                X_ref=X_ref,
            )

        elif self.sequence_mode == "rollout_lti":
            U_seq, stage_diag = self._generate_policy_rollout(
                p0=p0,
                x0=x0,
                X_ref=X_ref,
                use_ltv_jacobian=False,
            )

        elif self.sequence_mode == "rollout_ltv":
            U_seq, stage_diag = self._generate_policy_rollout(
                p0=p0,
                x0=x0,
                X_ref=X_ref,
                use_ltv_jacobian=True,
            )

        prediction_diag = self._build_prediction_diagnostics(
            p0=p0,
            U_seq=U_seq,
        )

        maximum_B_condition = max(
            (
                float(d.get("ij_B_cond", np.nan))
                for d in stage_diag
            ),
            default=np.nan,
        )

        maximum_task_residual = max(
            (
                float(
                    d.get(
                        "ij_task_residual_norm",
                        np.nan,
                    )
                )
                for d in stage_diag
            ),
            default=np.nan,
        )
        X_linear, B_sequence = (
            self._linear_policy_rollout(
                p0=p0,
                x0=x0,
                U_seq=U_seq,
                use_ltv_jacobian=(
                    self.sequence_mode == "rollout_ltv"
                ),
            )
        )
        if self.sequence_mode == "held":
            B_sequence = np.repeat(
                np.asarray(
                    stage_diag[0]["B_matrix"],
                    float,
                ).reshape(1, self.n, self.m),
                self.Np,
                axis=0,
            )
        else:
            B_sequence = np.stack(
                [
                    np.asarray(
                        d["B_matrix"],
                        float,
                    )
                    for d in stage_diag
                ],
                axis=0,
            )
        X_nonlinear = self._rollout_tip_stack_for_U(
            p0,
            U_seq,
        )

        inverse_lin_error = np.linalg.norm(
            X_nonlinear - X_linear,
            axis=1,
        )

        self._last_inverse_jacobian_info = {
            "controller_type": "inverse_jacobian",
            "inverse_sequence_mode": self.sequence_mode,

            "inverse_stage_diagnostics": stage_diag,
            "inverse_max_B_cond": maximum_B_condition,
            "inverse_max_task_residual": maximum_task_residual,

            # Predicted trajectories.
            "X_linear_plan": X_linear.copy(),
            "X_nl_plan": X_nonlinear.copy(),

            # Jacobian sequence used by the inverse controller.
            "B_sequence": B_sequence.copy(),

            # Common diagnostic key shared with MPC.
            "B_sequence_linearisation": B_sequence.copy(),

            # Stagewise linear-versus-nonlinear error.
            "inverse_linearisation_error_stage_m": (
                inverse_lin_error.copy()
            ),
        }

        solve_info = {
            "status": "solved_inverse_jacobian",
            "infeasible": 0,
            "sqp_hist": [],
            "failed_sqp_iter": -1,
            "used_previous_feasible": False,
            **prediction_diag,
        }

        return U_seq, solve_info

    def step(
        self,
        x_meas=None,
        rollout_steps=1,
        solver_mode=None,
    ):
        p_before = np.asarray(
            self.p,
            float,
        ).copy()

        p_post, x_post, info = super().step(
            x_meas=x_meas,
            rollout_steps=rollout_steps,
            solver_mode=solver_mode,
        )

        info.update(
            self._last_inverse_jacobian_info
        )

        info["X_pred"] = np.asarray(
            self._last_inverse_jacobian_info[
                "X_linear_plan"
            ],
            float,
        ).copy()

        info["X_nl_plan"] = np.asarray(
            self._last_inverse_jacobian_info[
                "X_nl_plan"
            ],
            float,
        ).copy()

        info["solver_mode"] = "inverse_jacobian"
        info["sqp_hist"] = []

        U_seq = np.asarray(
            info.get("U_seq", []),
            float,
        )

        U_applied = np.asarray(
            info.get("U_applied", []),
            float,
        )

        print(
            "\n[INVERSE STEP OUTPUT]"
        )

        print(
            "U_seq shape:",
            U_seq.shape,
        )

        print(
            "U_applied shape:",
            U_applied.shape,
        )

        if U_seq.ndim == 2 and U_seq.shape[0] > 0:
            print(
                "planned first command:",
                U_seq[0],
            )

        if (
            U_applied.ndim == 2
            and U_applied.shape[0] > 0
        ):
            print(
                "actually applied command:",
                U_applied[0],
            )

        print(
            "pose change:",
            np.asarray(p_post) - p_before,
        )

        return p_post, x_post, info