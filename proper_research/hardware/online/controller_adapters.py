"""Wire the offline Jacobian-based controllers into the ``OnlineMPCRunner`` seam.

``OnlineMPCRunner`` takes a ``solve(estimate, dt) -> SolveResult`` callback and
otherwise knows nothing about controller maths.  The three offline controllers
already built and validated in simulation —

    naive_inverse_jacobian   InverseJacobianBeamController  (resolved-rate)
    mpc_lti                  BeamOutputTrackingMPC, one frozen Jacobian
    mpc_ltv_offline          BeamOutputTrackingMPC, one Jacobian per reference
                              sample, precomputed offline

— all share one call signature (``proper_research/analysis/compare_controllers.py``
is the harness that already relies on this):

    step = controller.solve(
        measured_state=z,              # [q1..q6, insertion]  (7,)
        measured_beam_position=p,      # tip xyz, metres      (3,)
        control_index=k,               # progress into the reference
        previous_input=u_prev,         # [qd1..qd6, insertion_rate] (7,)
    )
    step.command                       # -> [qd1..qd6, insertion_rate] (7,)

That is a *joint-space* controller: it measures and commands six UR joint
angles/velocities plus the advancer insertion, never a Cartesian twist.  This
module is the adapter between that protocol and the online stack:

* :class:`OfflineJointControllerAdapter` turns any one of them into the
  ``solve(estimate, dt) -> SolveResult`` callable ``OnlineMPCRunner`` wants,
  tracking progress through the timed reference and the previous command the
  way ``compare_controllers.run_closed_loop`` does in simulation.
* :func:`build_offline_solver` constructs the controller itself (from a
  ``ConfigurationReference``, an MPC config and a Jacobian provider) and wraps
  it, so a hardware entry-point script only has to choose a name.
* Route the joint-velocity command to the arm through
  ``RobotSink(config=RobotSinkConfig(control_mode="joint"))`` — see
  ``robot_sink.py``.  The insertion-rate component (``u0[6]``) still goes to
  ``AdvancerSink.submit_rate`` unchanged; only the first six slots of ``u0``
  change meaning between Cartesian and joint mode.
* :class:`JsonlStepLogger` is a minimal ``on_step`` sink (the old runner had
  ``PublicationRunLogger``; this is its streaming-stack equivalent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .messages import StateEstimate, now_monotonic
from .online_mpc_runner import SolveResult

Array = np.ndarray

__all__ = [
    "OfflineControllerConfig",
    "OfflineJointControllerAdapter",
    "build_offline_solver",
    "JsonlStepLogger",
]

_MPC_KINDS = ("mpc_lti", "mpc_ltv_offline", "mpc_ltv_sqp_online")
_TRIM_KINDS = ("inv_2dof_trim",)
_TRIVIAL_KINDS = ("open_loop_ff",)
_ALL_KINDS = (
    ("naive_inverse_jacobian", "naive_inverse_jacobian_ltv")
    + _MPC_KINDS + _TRIM_KINDS + _TRIVIAL_KINDS
)


# ==========================================================================
# solve-seam adapter
# ==========================================================================
@dataclass
class OfflineControllerConfig:
    """How the adapter tracks progress and recovers from a bad measurement."""

    # "wallclock": control_index = round((t_now - t_first_call) / reference.dt),
    #   clamped to the reference length plus the terminal hold. This is the
    #   honest choice for hardware: the reference is a *timed* trajectory and
    #   control ticks are not guaranteed to land exactly on cfg.dt.
    # "counter": control_index increments by one every call, matching
    #   compare_controllers.run_closed_loop's simulation convention. Only
    #   correct if the runner's control period equals the reference's.
    progress_mode: str = "wallclock"
    terminal_hold_steps: int = 50
    # A controller step reporting success=False produces a zero command
    # rather than whatever it computed (mirrors OnlineMPCRunner's own
    # zero_command_on_infeasible, applied one layer earlier so the SolveResult
    # already carries a safe command if the caller ignores `infeasible`).
    zero_command_on_failure: bool = True
    # If the joint-state getter cannot find a fresh insertion length (e.g. the
    # advancer feedback has not published yet), fall back to the reference's
    # own insertion at the current progress index rather than raising.
    insertion_from_reference_if_missing: bool = True

    def __post_init__(self) -> None:
        if self.progress_mode not in ("wallclock", "counter"):
            raise ValueError(
                f"progress_mode must be 'wallclock' or 'counter'; got {self.progress_mode!r}"
            )
        if self.terminal_hold_steps < 0:
            raise ValueError("terminal_hold_steps must be >= 0")


JointStateGetter = Callable[[StateEstimate], Array]


class OfflineJointControllerAdapter:
    """``solve(estimate, dt) -> SolveResult`` around one joint-space controller.

    Holds exactly the state ``compare_controllers.run_closed_loop`` threads
    through its Python loop by hand: the previous command (for the
    controller's own acceleration clipping) and a progress index into the
    timed reference.  Everything else — the Jacobian, the QP, the clipping —
    stays inside the wrapped controller, unchanged from simulation.
    """

    def __init__(
        self,
        *,
        controller: Any,
        reference: Any,
        config: Optional[OfflineControllerConfig] = None,
        joint_state_getter: Optional[JointStateGetter] = None,
    ) -> None:
        self.controller = controller
        self.reference = reference
        self.config = config or OfflineControllerConfig()
        self._joint_state_getter = joint_state_getter

        self._t0: Optional[float] = None
        self._counter = 0
        self._last_reference_index = 0
        self._prev_input = np.zeros(7, dtype=float)

    def reset(self) -> None:
        """Forget wall-clock origin and the previous command; re-arm at index 0.

        Call before a fresh run with the same adapter instance (mirrors
        ``BaseController.reset()`` in the simulation harness).
        """
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self._t0 = None
        self._counter = 0
        self._last_reference_index = 0
        self._prev_input = np.zeros(7, dtype=float)

    # ------------------------------------------------------------------
    # progress into the timed reference
    # ------------------------------------------------------------------
    def _progress_index(self, estimate: StateEstimate) -> int:
        sample_count = int(self.reference.sample_count)
        ceiling = sample_count - 1 + int(self.config.terminal_hold_steps)
        if self.config.progress_mode == "counter":
            index = self._counter
        else:
            if self._t0 is None:
                self._t0 = estimate.t_monotonic
            elapsed = estimate.t_monotonic - self._t0
            index = int(round(elapsed / float(self.reference.sample_period_s)))
        return int(np.clip(index, 0, ceiling))

    # ------------------------------------------------------------------
    # measured state
    # ------------------------------------------------------------------
    def _joint_state(self, estimate: StateEstimate, reference_index: int) -> Array:
        if self._joint_state_getter is not None:
            return np.asarray(
                self._joint_state_getter(estimate), dtype=float
            ).reshape(7)

        if estimate.robot_joints is None:
            raise RuntimeError(
                "StateEstimate has no robot_joints and no joint_state_getter was "
                "given. Wire CameraSource(robot_joints_getter=robot.get_joints) "
                "(or pass an explicit joint_state_getter to build_offline_solver)."
            )
        q6 = np.asarray(estimate.robot_joints, dtype=float).reshape(6)

        insertion_m = float(estimate.insertion_length_m)
        if not np.isfinite(insertion_m):
            if not self.config.insertion_from_reference_if_missing:
                raise RuntimeError(
                    "StateEstimate.insertion_length_m is not finite and "
                    "insertion_from_reference_if_missing is False."
                )
            clamped = min(reference_index, int(self.reference.sample_count) - 1)
            insertion_m = float(np.asarray(self.reference.state, dtype=float)[clamped, 6])

        return np.concatenate([q6, [insertion_m]])

    # ------------------------------------------------------------------
    # the seam itself
    # ------------------------------------------------------------------
    def __call__(self, estimate: StateEstimate, dt: float) -> SolveResult:
        sample_count = int(self.reference.sample_count)
        progress_index = self._progress_index(estimate)
        reference_index = min(progress_index, sample_count - 1)
        self._last_reference_index = reference_index

        measured_state = self._joint_state(estimate, reference_index)
        measured_beam_position = np.asarray(
            estimate.tip_position_m, dtype=float
        ).reshape(3)

        step = self.controller.solve(
            measured_state=measured_state,
            measured_beam_position=measured_beam_position,
            control_index=reference_index,
            previous_input=self._prev_input,
        )

        command = np.asarray(step.command, dtype=float).reshape(7)
        success = bool(getattr(step, "success", True))
        infeasible = not success
        if infeasible and self.config.zero_command_on_failure:
            command = np.zeros(7, dtype=float)

        self._prev_input = command.copy()
        self._counter += 1

        # 2026-09-11: widened for controller-internals diagnosis (why does one
        # controller track worse than another on the same reference/Jacobian?)
        # -- InverseJacobianStep and BeamOutputMPCStep expose the SAME field
        # set (see their dataclasses), so this is safe for either controller.
        predicted_inputs = getattr(step, "predicted_inputs", None)
        predicted_states = getattr(step, "predicted_states", None)
        predicted_beam_positions = getattr(step, "predicted_beam_positions", None)
        predicted_beam_errors = getattr(step, "predicted_beam_errors", None)
        planned_input = getattr(step, "planned_input", None)
        info = {
            "controller": getattr(self.controller, "name", None)
            or getattr(self.controller, "variant_name", "offline_controller"),
            "progress_index": progress_index,
            "reference_index": reference_index,
            "terminal_hold": bool(progress_index >= sample_count - 1),
            "status": str(getattr(step, "status", "")),
            "success": success,
            "iterations": int(getattr(step, "iterations", 0)),
            "solve_time_s": float(getattr(step, "solve_time_s", 0.0)),
            "objective": float(getattr(step, "objective", np.nan)),
            "primal_residual": float(getattr(step, "primal_residual", np.nan)),
            "dual_residual": float(getattr(step, "dual_residual", np.nan)),
            "first_predicted_beam_error_m": float(
                getattr(step, "first_predicted_beam_error_m", np.nan)
            ),
            "planned_input": (
                None if planned_input is None
                else np.asarray(planned_input, dtype=float).reshape(-1).tolist()
            ),
            # first horizon step only (full horizon is verbose; step-0 is what
            # was actually about to be commanded next, the most diagnostic slice)
            "predicted_input_0": (
                None if predicted_inputs is None
                else np.asarray(predicted_inputs, dtype=float).reshape(-1, 7)[0].tolist()
            ),
            "predicted_state_0": (
                None if predicted_states is None
                else np.asarray(predicted_states, dtype=float).reshape(-1, 7)[0].tolist()
            ),
            "predicted_beam_position_0_m": (
                None if predicted_beam_positions is None
                else np.asarray(predicted_beam_positions, dtype=float).reshape(-1, 3)[0].tolist()
            ),
            "predicted_beam_error_0_m": (
                None if predicted_beam_errors is None
                else np.asarray(predicted_beam_errors, dtype=float).reshape(-1, 3)[0].tolist()
            ),
            "horizon_len": (
                0 if predicted_inputs is None
                else int(np.asarray(predicted_inputs, dtype=float).reshape(-1, 7).shape[0])
            ),
            "measured_joint_state": measured_state,
        }
        return SolveResult(u0=command, infeasible=infeasible, info=info)


@dataclass
class _TrivialFeedforwardStep:
    command: Array
    info: dict
    success: bool = True
    status: str = "open_loop_ff"
    iterations: int = 0
    solve_time_s: float = 0.0
    planned_input: Array | None = None


class TrivialFeedforwardController:
    """``solve()`` that ignores tracking error entirely: ``command = u_ref[index]``.

    2026-09-17: the OPEN-LOOP-FF acceptance test for a new execution seam --
    "does the plan itself, with zero correction, reproduce reasonable tip
    motion through this seam" -- before trusting any closed-loop controller
    comparison run through the same seam. Originally a monkeypatch in a
    one-off script; promoted here once it became a standing part of the
    Stage-A pipeline (see ``hardware/online/rectangle_stage_a/``).
    """

    name = "open_loop_ff"

    def __init__(self, reference: Any) -> None:
        self.reference = reference

    def reset(self) -> None:
        return None

    def solve(
        self, *, measured_state: Any, measured_beam_position: Any,
        control_index: int, previous_input: Any,
    ) -> _TrivialFeedforwardStep:
        idx = int(np.clip(control_index, 0, self.reference.sample_count - 1))
        u_ref = np.asarray(self.reference.input[idx], dtype=float).reshape(7)
        return _TrivialFeedforwardStep(
            command=u_ref, planned_input=u_ref.copy(),
            info={"status": "open_loop_ff", "success": True},
        )


# ==========================================================================
# controller construction
# ==========================================================================
def build_offline_solver(
    kind: str,
    *,
    reference: Any,
    jacobian_provider: Any,
    mpc_config: Any,
    beam_config: Any = None,
    adapter_config: Optional[OfflineControllerConfig] = None,
    schedule: Optional[Array] = None,
    freeze_index: int = 0,
    nominal_reference_positions_m: Any = None,
    position_gain: float = 1.0,
    damping: float = 1.0e-3,
    nullspace_gain: float = 1.0,
    feedforward: bool = True,
    feedforward_full: bool = False,
    allow_undeclared_jacobian: bool = False,
    joint_state_getter: Optional[JointStateGetter] = None,
    selective_damping_gain: float = 0.0,
    selective_damping_floor: float = 0.01,
    relinearise_every: int = 1,
    trim_kp: float = 0.6,
    trim_kn: float = 0.0,
    trim_damping: float = 5.0e-2,
    trim_q_max: float = 0.03,
    trim_max_joint_step_rad: float = 0.010,
    trim_enable_logging: bool = False,
) -> OfflineJointControllerAdapter:
    """Build one of the five offline controllers and wrap it for the runner.

    ``kind``: ``"naive_inverse_jacobian"``, ``"naive_inverse_jacobian_ltv"``,
    ``"mpc_lti"``, ``"mpc_ltv_offline"`` or ``"mpc_ltv_sqp_online"``.

    2026-09-16: ``naive_inverse_jacobian_ltv`` ("inverse-LTV") is the matched
    baseline for ``mpc_ltv_offline``: same precomputed per-sample Jacobian
    schedule (``schedule``/``reference_position_jacobians``), same
    ``mpc_config`` (sample period, box and rate limits), same reference
    progression -- the only thing that differs from ``mpc_ltv_offline`` is
    the control law itself (one-step damped-least-squares resolved-rate vs.
    finite-horizon QP). Exists because the original inv-vs-MPC comparisons
    were confounded: ``naive_inverse_jacobian`` and ``mpc_lti`` turned out to
    share the same frozen ``AnalyticalBeamJacobianProvider`` by accident (see
    `close_loop_logs/nullspace_motion_investigation_2026-09-16.md` §5), so
    "MPC beats inverse" conclusions were never cleanly isolating the control
    law. Needs a ``schedule`` built the same (state-dependent) way
    ``mpc_ltv_offline``'s is -- passing a frozen ``jacobian_provider`` here
    with no external ``schedule`` would silently build a degenerate,
    all-identical "schedule" and defeat the point; this is the same caveat
    the MPC kinds already carry, not a new one.

    2026-09-16: wired in ``mpc_ltv_sqp_online`` (previously only reachable from
    ``mpc_variants.py``'s offline/simulation module) to serve as the "tick-frozen"
    condition in the null-space-motion diagnostic protocol: at each real control
    tick it relinearises at the *measured* state (unlike mpc_lti's run-long freeze
    or mpc_ltv_offline's precomputed per-sample schedule) and holds that single
    Jacobian across the whole prediction horizon (``relinearise_horizon=False``,
    ``inner_iterations=1`` -- one relinearisation per tick, the standard
    real-time SQP). Costs one extra beam-Jacobian evaluation per tick versus the
    other two MPC kinds; not used by any existing comparison unless requested.

    ``jacobian_provider`` is whatever ``beam_jacobian_providers.from_model_bundle``
    (or ``from_controller_pack``) returned — the same provider the simulation
    comparison uses, so the online Jacobian is provably the same model.
    ``mpc_config`` is a ``ConfigurationMPCConfig`` (shared sample period, box
    and rate limits); ``beam_config`` is a ``BeamOutputMPCConfig``, required
    for the two MPC kinds.

    2026-09-16: ``relinearise_every`` throttles ``mpc_ltv_sqp_online``'s
    per-tick relinearisation (see ``mpc_variants.build_sqp_online_mpc`` for
    why: a genuinely state-dependent Jacobian provider can cost far more per
    call off the smooth reference manifold than on it, and calling it every
    tick can blow the real-time budget badly enough to starve the robot of
    commands -- confirmed offline against a nonlinear-plant closed-loop
    simulation to cost ~0 tracking accuracy out to every=20 on the S-curve
    shape, while cutting Jacobian evaluations ~20x). Ignored for the other
    three kinds.
    """
    if kind not in _ALL_KINDS:
        raise ValueError(f"kind must be one of {_ALL_KINDS}; got {kind!r}")

    from proper_research.controllers import mpc_variants

    if kind == "naive_inverse_jacobian":
        from proper_research.controllers.inverse_jacobian_controller import (
            build_inverse_jacobian_controller,
        )

        controller = build_inverse_jacobian_controller(
            reference=reference,
            jacobian_provider=jacobian_provider,
            mpc_config=mpc_config,
            position_gain=position_gain,
            damping=damping,
            nullspace_gain=nullspace_gain,
            feedforward=feedforward,
            feedforward_full=feedforward_full,
            allow_undeclared_jacobian=allow_undeclared_jacobian,
            selective_damping_gain=selective_damping_gain,
            selective_damping_floor=selective_damping_floor,
        )
    elif kind == "naive_inverse_jacobian_ltv":
        from proper_research.controllers.inverse_jacobian_controller import (
            build_inverse_jacobian_controller,
        )

        if schedule is None:
            schedule = mpc_variants.precompute_schedule(
                reference=reference,
                jacobian_provider=jacobian_provider,
                allow_undeclared_jacobian=allow_undeclared_jacobian,
            )
        controller = build_inverse_jacobian_controller(
            reference=reference,
            jacobian_provider=jacobian_provider,
            mpc_config=mpc_config,
            position_gain=position_gain,
            damping=damping,
            nullspace_gain=nullspace_gain,
            feedforward=feedforward,
            feedforward_full=feedforward_full,
            allow_undeclared_jacobian=allow_undeclared_jacobian,
            selective_damping_gain=selective_damping_gain,
            selective_damping_floor=selective_damping_floor,
            reference_position_jacobians=schedule,
        )
    elif kind == "inv_2dof_trim":
        # 2026-09-17: see inverse_jacobian_2dof_trim.py's module docstring
        # for the full architecture and the windup diagnosis that motivated
        # it. Requires the caller's harness to run the accumulator seam
        # (PathFollowConfig.accumulator_seam=True,
        # feedforward_joint_trajectory=False) -- this controller reproduces
        # that seam's own q_cmd update internally so the two stay in
        # lockstep, but does not set the harness config itself.
        from proper_research.controllers.inverse_jacobian_2dof_trim import (
            build_inv_2dof_trim_controller,
        )

        controller = build_inv_2dof_trim_controller(
            reference=reference,
            jacobian_provider=jacobian_provider,
            kp=trim_kp,
            kn=trim_kn,
            damping=trim_damping,
            q_trim_max=trim_q_max,
            dt=float(mpc_config.sample_period_s),
            max_joint_step_rad=trim_max_joint_step_rad,
            enable_logging=trim_enable_logging,
        )
    elif kind == "open_loop_ff":
        controller = TrivialFeedforwardController(reference)
    else:
        if beam_config is None:
            raise ValueError(f"beam_config is required to build {kind!r}.")
        if schedule is None:
            schedule = mpc_variants.precompute_schedule(
                reference=reference,
                jacobian_provider=jacobian_provider,
                allow_undeclared_jacobian=allow_undeclared_jacobian,
            )
        describe = getattr(jacobian_provider, "describe", None)
        provenance = describe() if callable(describe) else None
        shared = dict(
            reference=reference,
            config=mpc_config,
            beam_config=beam_config,
            reference_position_jacobians=schedule,
            jacobian_provenance=provenance,
            nominal_reference_positions_m=nominal_reference_positions_m,
        )
        if kind == "mpc_lti":
            controller = mpc_variants.build_lti_mpc(freeze_index=freeze_index, **shared)
        elif kind == "mpc_ltv_offline":
            controller = mpc_variants.build_ltv_offline_mpc(**shared)
        else:  # mpc_ltv_sqp_online, "tick-frozen": relinearise at the measured
            # state each tick, hold that single Jacobian across the horizon.
            # build_sqp_online_mpc doesn't accept jacobian_provenance directly
            # (it derives provenance itself via declare_provider on
            # jacobian_provider) or reference_position_jacobians as a bare
            # kwarg name under `shared` conflicting with its own signature --
            # unpack `shared` explicitly rather than reusing it blindly.
            controller = mpc_variants.build_sqp_online_mpc(
                reference=reference,
                config=mpc_config,
                beam_config=beam_config,
                reference_position_jacobians=schedule,
                jacobian_provider=jacobian_provider,
                inner_iterations=1,
                relinearise_horizon=False,
                relinearise_every=relinearise_every,
                nominal_reference_positions_m=nominal_reference_positions_m,
                allow_undeclared_jacobian=allow_undeclared_jacobian,
            )

    return OfflineJointControllerAdapter(
        controller=controller,
        reference=reference,
        config=adapter_config,
        joint_state_getter=joint_state_getter,
    )


# ==========================================================================
# logging sink for OnlineMPCRunner(on_step=...)
# ==========================================================================
class JsonlStepLogger:
    """Append one JSON object per control step. The ``on_step`` seam's logger.

    Replaces the old ``PublicationRunLogger``: no CSV schema to keep in sync
    with the loop, just whatever ``OnlineMPCRunner`` and the solver's
    ``SolveResult.info`` hand back, flattened to JSON lines so a run can be
    tailed live (``tail -f run.jsonl``) or loaded with
    ``pandas.read_json(path, lines=True)`` afterwards.
    """

    def __init__(self, path: "str | Path", *, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._flush_every = max(1, int(flush_every))
        self._writes = 0
        self.opened_at = now_monotonic()

    def __call__(self, step: dict) -> None:
        estimate: Optional[StateEstimate] = step.get("estimate")
        info = step.get("solver_info") or {}
        row: dict = {
            "step": step.get("step"),
            "t_monotonic": step.get("t_monotonic"),
            "state_age_s": step.get("state_age_s"),
            "solve_ms": step.get("solve_ms"),
            "infeasible": step.get("infeasible"),
            "u0": _to_list(step.get("u0")),
        }
        if estimate is not None:
            row.update(
                {
                    "frame_index": estimate.frame_index,
                    "tip_position_m": _to_list(estimate.tip_position_m),
                    "tip_tangent": _to_list(estimate.tip_tangent),
                    "insertion_length_m": estimate.insertion_length_m,
                    "vision_beam_length_mm": estimate.vision_beam_length_mm,
                }
            )
        for key, value in info.items():
            if key == "measured_joint_state":
                row[key] = _to_list(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                row[key] = value
            else:
                row[key] = _to_list(value)
        self._fh.write(json.dumps(row) + "\n")
        self._writes += 1
        if self._writes % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "JsonlStepLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [
            item.tolist() if isinstance(item, np.ndarray) else item for item in value
        ]
    return value


# ==========================================================================
# self-test: drives all three offline controllers through the real seam
# ==========================================================================
def run_self_test() -> None:
    """Build all three offline controllers and drive them through fake frames.

    Exercises exactly what a hardware run would do each tick — read a
    ``StateEstimate``, call the adapter, get a ``SolveResult`` — without any
    camera, robot or serial connection. Catches shape/wiring bugs in the seam
    itself; the controller maths is validated separately by each controller's
    own module and by ``compare_controllers.py --self-test``.
    """
    dt = 0.1
    sample_count = 20
    time_values = np.arange(sample_count, dtype=float) * dt
    state_reference = np.zeros((sample_count, 7), dtype=float)
    state_reference[:, 6] = 0.001 * np.arange(sample_count)  # insertion advances
    input_reference = np.zeros_like(state_reference)
    desired_position = np.column_stack(
        [np.zeros(sample_count), np.zeros(sample_count), 5.0 * state_reference[:, 6]]
    )
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (sample_count, 1))

    from proper_research.simulation.simulations import (
        simulate_time_parameterized_configuration_mpc as base_module,
    )
    from proper_research.simulation.simulations import (
        simulate_time_parameterized_beam_output_mpc as beam_module,
    )

    reference = base_module.ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=np.zeros(sample_count),
        state=state_reference,
        input=input_reference,
        acceleration=np.zeros_like(state_reference),
        desired_position_m=desired_position,
        desired_tangent=tangent,
        planned_beam_feasible=np.ones(sample_count, dtype=bool),
        planned_position_error_m=np.zeros(sample_count),
        planned_tangent_error_rad=np.zeros(sample_count),
        sample_period_s=dt,
        source="controller_adapters_self_test",
    )
    mpc_config = base_module.ConfigurationMPCConfig(
        sample_period_s=dt,
        prediction_horizon=5,
        state_min=tuple([-0.5] * 6 + [-0.02]),
        state_max=tuple([0.5] * 6 + [0.05]),
        velocity_limit=tuple([0.4] * 6 + [0.01]),
        acceleration_limit=tuple([2.0] * 6 + [0.1]),
        state_tracking_weight=1.0e-3,
        input_tracking_weight=1.0e-4,
        input_increment_weight=1.0e-4,
        solver_backend="osqp" if base_module.osqp is not None else "scipy",
        solver_absolute_tolerance=1.0e-8,
        solver_relative_tolerance=1.0e-8,
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3),
        position_tracking_weight=1.0,
        use_dare_terminal_cost=True,
    )

    def jacobian_provider(state: Array) -> Array:
        matrix = np.zeros((3, 7), dtype=float)
        matrix[0, 0] = 0.3
        matrix[1, 1] = 0.2
        matrix[2, 6] = 5.0
        return matrix

    def make_estimate(step: int, t: float) -> StateEstimate:
        q6 = 0.001 * step * np.ones(6)
        tip = np.array([0.02 * np.sin(step), 0.0, 5.0 * 0.001 * step])
        return StateEstimate(
            t_monotonic=t,
            x_meas=np.concatenate([tip, [1.0, 0.0, 0.0]]),
            tip_position_m=tip,
            tip_tangent=np.array([1.0, 0.0, 0.0]),
            insertion_length_m=float("nan"),  # forces the reference fallback
            robot_joints=q6,
            frame_index=step,
        )

    for kind in _ALL_KINDS:
        solver = build_offline_solver(
            kind,
            reference=reference,
            jacobian_provider=jacobian_provider,
            mpc_config=mpc_config,
            beam_config=beam_config if kind in _MPC_KINDS else None,
            adapter_config=OfflineControllerConfig(
                progress_mode="wallclock", terminal_hold_steps=3
            ),
            allow_undeclared_jacobian=True,
        )
        t0 = 100.0  # arbitrary monotonic origin, unrelated to wall time
        seen_indices = []
        for step in range(8):
            estimate = make_estimate(step, t0 + step * dt)
            result = solver(estimate, dt)
            assert isinstance(result, SolveResult), f"{kind}: wrong return type"
            u0 = np.asarray(result.u0, dtype=float).reshape(-1)
            assert u0.size == 7, f"{kind}: u0 must be length 7, got {u0.size}"
            assert np.all(np.isfinite(u0)), f"{kind}: non-finite u0 at step {step}"
            seen_indices.append(result.info["reference_index"])
        assert seen_indices == sorted(seen_indices), (
            f"{kind}: reference_index must be non-decreasing, got {seen_indices}"
        )
        assert seen_indices[-1] > seen_indices[0], (
            f"{kind}: progress never advanced across 8 wallclock-spaced calls"
        )
        # A missing insertion_length_m must fall back to the reference, not crash.
        assert np.isfinite(result.info["measured_joint_state"][6]), (
            f"{kind}: insertion fallback did not produce a finite value"
        )

        solver.reset()
        assert solver._counter == 0 and solver._t0 is None, f"{kind}: reset() did not clear progress"
        print(f"[controller_adapters self-test] {kind}: OK, indices={seen_indices}")

    print("[controller_adapters self-test] PASS")


if __name__ == "__main__":
    run_self_test()
