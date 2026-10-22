"""Apply the layer-2 safety/task-space upgrades to the global optimiser.

Every edit is anchored on an exact string from the original file, so this
either applies cleanly or fails loudly. Run it once; it is not idempotent.
"""
from pathlib import Path
import sys

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "global_constrained_configuration_path.py")
text = TARGET.read_text()
applied = []


def swap(name: str, old: str, new: str, count: int = 1) -> None:
    global text
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[{name}] anchor matched {found} times, expected {count}.\n"
            f"--- anchor ---\n{old[:400]}"
        )
    text = text.replace(old, new, count)
    applied.append(name)


# ---------------------------------------------------------------- 1. import
swap(
    "import",
    """try:
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )
except ModuleNotFoundError:  # Allows ``python this_file.py --self-test``.
    from offline_inverse_configuration_head_exclusion import (  # type: ignore
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )
""",
    """try:
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )
except ModuleNotFoundError:  # Allows ``python this_file.py --self-test``.
    from offline_inverse_configuration_head_exclusion import (  # type: ignore
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )

try:
    from proper_research.planning.global_upgrades import (
        NodeExclusionConstraints,
        effective_task_tolerances,
        tip_centring_quadratic,
    )
except ModuleNotFoundError:  # Allows ``python this_file.py --self-test``.
    from global_upgrades import (  # type: ignore
        NodeExclusionConstraints,
        effective_task_tolerances,
        tip_centring_quadratic,
    )
""",
)

# ------------------------------------------------------------ 2. new fields
swap(
    "config fields",
    """    debug: bool = True
    trust_constr_verbose: int = 0

    def validate(self) -> None:
        if self.mode not in {"refine_complete", "recover_partial"}:""",
    '''    debug: bool = True
    trust_constr_verbose: int = 0

    # ---------------------------------------------------------------- safety
    # The source-magnet / lumen keep-out that layer 1 enforces.  Leaving this
    # at None reproduces the previous behaviour exactly: no exclusion
    # constraint.  Set it to the same radius the inverse planner used and the
    # smoother can no longer walk the magnet into the head.
    source_magnet_lumen_exclusion_radius_m: float | None = None
    source_magnet_lumen_constraint_tolerance_m: float = 1.0e-6
    require_analytical_magnet_position_jacobian: bool = False
    magnet_jacobian_joint_step_rad: float = 1.0e-6
    magnet_jacobian_insertion_step_m: float = 1.0e-6

    # ------------------------------------------------------------ task space
    # Fraction of the physical task tolerance the smoother is allowed to
    # spend.  1.0 is the previous behaviour.  Below 1.0 the constraint the
    # optimiser sees is tighter than the tolerance it is reported against, so
    # the remainder survives as margin for the controller instead of being
    # consumed by joint smoothing.
    tolerance_spend_fraction: float = 1.0

    # Optional quadratic pull toward the centre of the tolerance ball,
    # linearised at the seed.  0.0 keeps the objective exactly as it was.
    tip_centring_weight: float = 0.0

    def validate(self) -> None:
        if self.mode not in {"refine_complete", "recover_partial"}:''',
)

swap(
    "config validate",
    """        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
""",
    '''        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        radius = self.source_magnet_lumen_exclusion_radius_m
        if radius is not None and (not np.isfinite(radius) or float(radius) <= 0.0):
            raise ValueError(
                "source_magnet_lumen_exclusion_radius_m must be positive when set."
            )
        for name in (
            "source_magnet_lumen_constraint_tolerance_m",
            "magnet_jacobian_joint_step_rad",
            "magnet_jacobian_insertion_step_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        fraction = float(self.tolerance_spend_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("tolerance_spend_fraction must lie in (0, 1].")
        centring = float(self.tip_centring_weight)
        if not np.isfinite(centring) or centring < 0.0:
            raise ValueError("tip_centring_weight must be finite and non-negative.")
''',
)

# ------------------------------------------- 3. restoration takes exclusion
swap(
    "restoration signature",
    """def _run_restoration(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
) -> tuple[Array, float, Any]:""",
    """def _run_restoration(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
    exclusion: NodeExclusionConstraints | None = None,
) -> tuple[Array, float, Any]:""",
)

# --------------------------------------------- 4. hard solve takes exclusion
swap(
    "hard solve signature",
    """def _run_hard_solve(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
) -> Any:""",
    """def _run_hard_solve(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
    exclusion: NodeExclusionConstraints | None = None,
) -> Any:""",
)

swap(
    "hard solve constraints",
    """    if adjacent is not None:
        constraints.append(adjacent)
    return minimize(
        objective.value,
        x0,
        method="trust-constr",""",
    '''    if adjacent is not None:
        constraints.append(adjacent)
    if exclusion is not None:
        # The keep-out is a physical safety constraint, so it is never given
        # slack and never traded against smoothness.  keep_feasible is set
        # only when the seed already satisfies it, which is the normal case
        # for a layer-1 path that enforced the same inequality.
        initial_exclusion = exclusion.values(x0)
        constraints.append(
            NonlinearConstraint(
                exclusion.values,
                np.zeros(exclusion.node_count),
                np.full(exclusion.node_count, np.inf),
                jac=exclusion.jacobian,
                keep_feasible=bool(np.all(initial_exclusion >= 0.0)),
            )
        )
        if config.debug:
            print(
                "[GLOBAL SEED] "
                f"minimum_exclusion_constraint={np.min(initial_exclusion):.6e} "
                f"jacobian_source={exclusion.jacobian_source}",
                flush=True,
            )
    return minimize(
        objective.value,
        x0,
        method="trust-constr",''',
)

# ------------------------------------------------- 5. violation bookkeeping
swap(
    "violation signature",
    """def _recomputed_constraint_violation(
    *,
    tasks: _TaskConstraints,
    adjacent: LinearConstraint | None,
    bounds: Bounds,
    x: Array,
) -> float:""",
    """def _recomputed_constraint_violation(
    *,
    tasks: _TaskConstraints,
    adjacent: LinearConstraint | None,
    bounds: Bounds,
    x: Array,
    exclusion: NodeExclusionConstraints | None = None,
) -> float:""",
)

swap(
    "violation body",
    """    return float(
        max(
            _task_violation(tasks, x),
            _linear_constraint_violation(adjacent, x),
            _bound_constraint_violation(bounds, x),
        )
    )""",
    """    return float(
        max(
            _task_violation(tasks, x),
            _linear_constraint_violation(adjacent, x),
            _bound_constraint_violation(bounds, x),
            0.0 if exclusion is None else exclusion.violation(x),
        )
    )""",
)

# ----------------------------------- 6. build the upgrades in the solve fn
swap(
    "solve signature",
    """def _solve_global_problem(
    *,
    adapter: Any,
    seed: _SeedPath,
    state_min: Array,
    state_max: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> tuple[_SolveOutcome, _QuadraticPathObjective, _TaskConstraints]:
    node_count = seed.s.size""",
    """def _solve_global_problem(
    *,
    adapter: Any,
    seed: _SeedPath,
    state_min: Array,
    state_max: Array,
    config: GlobalConfigurationOptimizerConfig,
    path: CentrelinePath | None = None,
) -> tuple[_SolveOutcome, _QuadraticPathObjective, _TaskConstraints]:
    node_count = seed.s.size""",
)

swap(
    "task tolerances",
    """    tasks = _TaskConstraints(
        evaluator=evaluator,
        desired_position=seed.desired_position,
        desired_tangent=seed.desired_tangent,
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
    )
    bounds = _decision_bounds(""",
    '''    # The optimiser is constrained against the tolerance it is *allowed to
    # spend*; reporting elsewhere still uses the physical tolerance, so the
    # difference becomes margin rather than budget.
    spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
        spend_fraction=config.tolerance_spend_fraction,
    )
    if config.debug and config.tolerance_spend_fraction < 1.0:
        print(
            "[GLOBAL TOLERANCE] "
            f"spend_fraction={config.tolerance_spend_fraction:.3f} "
            f"position={1.0e3 * spend_position_tolerance:.6f} mm of "
            f"{1.0e3 * config.position_tolerance_m:.6f} mm",
            flush=True,
        )
    tasks = _TaskConstraints(
        evaluator=evaluator,
        desired_position=seed.desired_position,
        desired_tangent=seed.desired_tangent,
        position_tolerance_m=spend_position_tolerance,
        tangent_tolerance_rad=spend_tangent_tolerance,
    )
    exclusion: NodeExclusionConstraints | None = None
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        if path is None:
            raise ValueError(
                "A source-magnet/lumen keep-out radius is configured but no "
                "CentrelinePath was supplied to the global solve. Pass "
                "path=CentrelinePath(lumen_C)."
            )
        exclusion = NodeExclusionConstraints(
            adapter=adapter,
            path=path,
            state_min=state_min,
            state_max=state_max,
            node_count=node_count,
            radius_m=float(config.source_magnet_lumen_exclusion_radius_m),
            require_analytical_jacobian=bool(
                config.require_analytical_magnet_position_jacobian
            ),
            joint_step_rad=float(config.magnet_jacobian_joint_step_rad),
            insertion_step_m=float(config.magnet_jacobian_insertion_step_m),
        )
    if config.tip_centring_weight > 0.0:
        # One extra block-diagonal quadratic in the same QP.  Linearised at
        # the seed, which is also where this round's beam baselines are
        # frozen, so the cost and the constraints see the same model.
        H_tip, f_tip, constant_tip = tip_centring_quadratic(
            evaluator=evaluator,
            seed_states=seed.states,
            desired_position=seed.desired_position,
            position_tolerance_m=config.position_tolerance_m,
            weight=float(config.tip_centring_weight),
        )
        objective.H = csc_matrix(objective.H + H_tip)
        objective.f = np.asarray(objective.f, dtype=float) + f_tip
        objective.constant = float(objective.constant) + constant_tip
        if config.debug:
            print(
                "[GLOBAL TIP COST] "
                f"weight={config.tip_centring_weight:.6g} linearised at seed",
                flush=True,
            )
    bounds = _decision_bounds(''',
)

swap(
    "restoration call",
    """            x0, restoration_slack, restoration_result = _run_restoration(
                x0=x0,
                objective=objective,
                tasks=tasks,
                bounds=bounds,
                adjacent=adjacent,
                config=config,
            )""",
    """            x0, restoration_slack, restoration_result = _run_restoration(
                x0=x0,
                objective=objective,
                tasks=tasks,
                bounds=bounds,
                adjacent=adjacent,
                config=config,
                exclusion=exclusion,
            )""",
)

swap(
    "hard solve call",
    """        solved = _run_hard_solve(
            x0=x0,
            objective=objective,
            tasks=tasks,
            bounds=bounds,
            adjacent=adjacent,
            config=config,
        )""",
    """        solved = _run_hard_solve(
            x0=x0,
            objective=objective,
            tasks=tasks,
            bounds=bounds,
            adjacent=adjacent,
            config=config,
            exclusion=exclusion,
        )""",
)

swap(
    "violation call",
    """        violation = _recomputed_constraint_violation(
            tasks=tasks,
            adjacent=adjacent,
            bounds=bounds,
            x=x,
        )""",
    """        violation = _recomputed_constraint_violation(
            tasks=tasks,
            adjacent=adjacent,
            bounds=bounds,
            x=x,
            exclusion=exclusion,
        )""",
)

swap(
    "solve call site",
    """        outcome, objective, _tasks = _solve_global_problem(
            adapter=adapter,
            seed=seed,
            state_min=state_min,
            state_max=state_max,
            config=config,
        )""",
    """        outcome, objective, _tasks = _solve_global_problem(
            adapter=adapter,
            seed=seed,
            state_min=state_min,
            state_max=state_max,
            config=config,
            path=path,
        )""",
)

# ------------------------------------------------- 7. report the keep-out
swap(
    "summary",
    """    summary = {
        "globally_feasible": globally_feasible,
        "dense_validation_feasible": dense_feasible,""",
    '''    exclusion_summary: dict[str, Any] = {
        "enabled": config.source_magnet_lumen_exclusion_radius_m is not None,
        "radius_m": config.source_magnet_lumen_exclusion_radius_m,
    }
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        _final_exclusion = NodeExclusionConstraints(
            adapter=adapter,
            path=path,
            state_min=state_min,
            state_max=state_max,
            node_count=len(final_nodes),
            radius_m=float(config.source_magnet_lumen_exclusion_radius_m),
            require_analytical_jacobian=bool(
                config.require_analytical_magnet_position_jacobian
            ),
            joint_step_rad=float(config.magnet_jacobian_joint_step_rad),
            insertion_step_m=float(config.magnet_jacobian_insertion_step_m),
        )
        _metrics = _final_exclusion.metrics(final_states.reshape(-1))
        exclusion_summary.update(
            {
                "minimum_distance_m": float(np.min(_metrics["distance_m"])),
                "minimum_margin_m": float(_metrics["minimum_margin_m"]),
                "all_nodes_satisfied": bool(_metrics["all_satisfied"]),
                "violating_node_count": int(
                    np.sum(~np.asarray(_metrics["satisfied"], dtype=bool))
                ),
                "jacobian_source": str(_metrics["jacobian_source"]),
            }
        )
        if config.debug:
            print(
                "[GLOBAL EXCLUSION] "
                f"minimum_margin={1.0e3 * exclusion_summary['minimum_margin_m']:.6f} mm "
                f"satisfied={exclusion_summary['all_nodes_satisfied']}",
                flush=True,
            )

    summary = {
        "globally_feasible": globally_feasible,
        "source_magnet_lumen_exclusion": exclusion_summary,
        "dense_validation_feasible": dense_feasible,''',
)

TARGET.write_text(text)
print("applied:", ", ".join(applied))
