"""Safety and task-space upgrades for the global configuration optimiser.

This module adds three things the global layer did not have, without changing
its mathematics anywhere else:

1. **The source-magnet / lumen keep-out constraint.**  Layer 1 enforces
   ``d(chi)^2 / R^2 - 1 >= 0`` as a hard SLSQP inequality and its minimum
   margin is typically *exactly zero* — the constraint is active and shaping
   the solution.  Layer 2 had no exclusion constraint of any kind, so
   smoothing was free to walk the magnet into the keep-out region.  Nothing
   here re-derives that constraint: it imports the inverse planner's own
   ``_MagnetLumenExclusionConstraint`` and evaluates it once per node, so the
   two layers enforce a bit-identical condition.

2. **A tolerance-spend fraction.**  The global objective has no task-space
   term, so tip error inside the tolerance ball is free and the optimiser
   spends all of it.  Constraining against ``fraction * tolerance`` while
   still *reporting* against the physical tolerance bounds that spend and
   leaves the remainder as margin for the controller.  One number, no new
   mathematics, and it is the cheapest fix for a tip path that wanders.

3. **An optional tip-centring cost.**  When you would rather pay for a
   centred tip than forbid an off-centre one, this adds

       0.5 * w * sum_j || (p_j(chi_j) - p*_j) / tol_p ||^2

   linearised at the round's seed, ``p_j(chi_j) ~ p_j^0 + J_j (chi_j - chi_j^0)``.
   The result is one extra block-diagonal quadratic in the *same* QP: the
   problem stays convex, the Hessian stays sparse, and trust-constr needs no
   new machinery.  Linearising at the seed is consistent with the rest of the
   layer, which already freezes its beam baselines per refinement round.

Use 2 or 3, not usually both.  2 is a hard guarantee with no tuning; 3 is a
soft preference that keeps the whole tolerance available when the geometry
needs it.

4. **A tip-curvature cost.**  1-3 all bound or centre where the tip sits at
   *one* node; none of them couple adjacent nodes in tip space.  The path
   objective's first/second-difference terms (``_first_difference_matrix`` /
   ``_second_difference_matrix`` in ``global_constrained_configuration_path``)
   only act on ``chi`` — so a ``chi`` sequence that is smooth by that objective
   can still map, through the nonlinear (and, near contact, rapidly-varying)
   beam Jacobian, to a tip sequence that zigzags freely inside each node's own
   tolerance ball.  This adds the direct analogue of ``_second_difference_matrix``
   in *tip* space:

       0.5 * w * sum_j || D2 p_j ||^2,   D2 p_j = 2*(p_{j-1}/L + p_{j+1}/R - p_j*(1/L+1/R)) / (L+R)

   with ``p_j(chi_j)`` linearised at the seed exactly as in 3, so each interior
   node contributes one 21x21 (three-node) block to the same sparse QP. This is
   the term that actually makes the *tip* path smooth; 2 and 3 only bound or
   bias how far it can wander.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.sparse import block_diag, csc_matrix, lil_matrix

try:  # the same dual import the global module uses
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        CentrelinePath,
        _MagnetLumenExclusionConstraint,
    )
except ModuleNotFoundError:  # pragma: no cover - self-test path
    from offline_inverse_configuration_head_exclusion import (  # type: ignore
        CentrelinePath,
        _MagnetLumenExclusionConstraint,
    )

Array = np.ndarray


@dataclass(frozen=True)
class _ExclusionShimConfig:
    """The four fields ``_MagnetLumenExclusionConstraint`` actually reads.

    Passing a shim rather than an ``InverseConfigurationPlannerConfig`` keeps
    the global layer from depending on the whole inverse planner config, while
    guaranteeing the constraint sees exactly the values it did at layer 1.
    """

    source_magnet_lumen_exclusion_radius_m: float
    source_magnet_lumen_constraint_tolerance_m: float
    require_analytical_magnet_position_jacobian: bool
    magnet_jacobian_joint_step_rad: float
    magnet_jacobian_insertion_step_m: float


class NodeExclusionConstraints:
    """The layer-1 keep-out, evaluated at every global node.

    ``values(x)`` returns one dimensionless margin per node and ``jacobian(x)``
    the matching block-diagonal derivative — the shape ``trust-constr`` wants
    for a ``NonlinearConstraint`` with ``lb=0``, ``ub=+inf``.

    One constraint object is held per node.  Each caches its last magnet
    position and Jacobian, so a solver that evaluates values and derivatives at
    the same iterate pays for the magnet kinematics once, exactly as the
    per-node inverse solve does.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        path: CentrelinePath,
        state_min: Array,
        state_max: Array,
        node_count: int,
        radius_m: float,
        constraint_tolerance_m: float = 1.0e-6,
        require_analytical_jacobian: bool = False,
        joint_step_rad: float = 1.0e-6,
        insertion_step_m: float = 1.0e-6,
    ) -> None:
        shim = _ExclusionShimConfig(
            source_magnet_lumen_exclusion_radius_m=float(radius_m),
            source_magnet_lumen_constraint_tolerance_m=float(constraint_tolerance_m),
            require_analytical_magnet_position_jacobian=bool(
                require_analytical_jacobian
            ),
            magnet_jacobian_joint_step_rad=float(joint_step_rad),
            magnet_jacobian_insertion_step_m=float(insertion_step_m),
        )
        self.node_count = int(node_count)
        self.radius_m = float(radius_m)
        self.constraints = [
            _MagnetLumenExclusionConstraint(
                adapter=adapter,
                path=path,
                state_min=state_min,
                state_max=state_max,
                config=shim,
            )
            for _ in range(self.node_count)
        ]
        self.jacobian_source = self.constraints[0].jacobian_source

    def _states(self, x: Array) -> Array:
        return np.asarray(x, dtype=float).reshape(self.node_count, 7)

    def values(self, x: Array) -> Array:
        states = self._states(x)
        return np.asarray(
            [
                constraint.value(states[index])
                for index, constraint in enumerate(self.constraints)
            ],
            dtype=float,
        )

    def jacobian(self, x: Array) -> Any:
        states = self._states(x)
        rows = [
            csc_matrix(
                np.asarray(
                    constraint.jacobian(states[index]), dtype=float
                ).reshape(1, 7)
            )
            for index, constraint in enumerate(self.constraints)
        ]
        return block_diag(rows, format="csc")

    def violation(self, x: Array) -> float:
        """Worst dimensionless violation; 0 when every node is outside R."""
        return float(np.max(np.maximum(0.0, -self.values(x))))

    def metrics(self, x: Array) -> dict[str, Any]:
        states = self._states(x)
        distances, margins, segments, satisfied = [], [], [], []
        for index, constraint in enumerate(self.constraints):
            distance, margin, segment, ok = constraint.physical_metrics(
                states[index]
            )
            distances.append(distance)
            margins.append(margin)
            segments.append(segment)
            satisfied.append(bool(ok))
        return {
            "radius_m": self.radius_m,
            "distance_m": np.asarray(distances, dtype=float),
            "margin_m": np.asarray(margins, dtype=float),
            "segment_index": np.asarray(segments, dtype=int),
            "satisfied": np.asarray(satisfied, dtype=bool),
            "minimum_margin_m": float(np.min(margins)) if margins else float("nan"),
            "all_satisfied": bool(np.all(satisfied)) if satisfied else True,
            "jacobian_source": self.jacobian_source,
        }


def tip_centring_quadratic(
    *,
    evaluator: Any,
    seed_states: Array,
    desired_position: Array,
    position_tolerance_m: float,
    weight: float,
) -> tuple[Any, Array, float]:
    """Quadratic pull toward the centre of the tolerance ball.

    Returns ``(H, f, constant)`` for ``0.5 x^T H x + f^T x + constant`` in the
    stacked variable ``x = [chi_0; ...; chi_{N-1}]``.

    The tip map is linearised at the seed:

        p_j(chi_j) ~ p_j^0 + J_j (chi_j - chi_j^0)
        residual_j = (p_j(chi_j) - p*_j) / tol_p
                   = (J_j chi_j + c_j) / tol_p,
        c_j = p_j^0 - J_j chi_j^0 - p*_j

    so each node contributes ``w J_j^T J_j / tol^2`` to the Hessian.  ``H`` is
    block diagonal with 7x7 blocks: the QP's sparsity is unchanged.
    """
    weight = float(weight)
    if weight <= 0.0:
        raise ValueError("tip-centring weight must be positive to build a cost.")
    seed_states = np.asarray(seed_states, dtype=float).reshape(-1, 7)
    desired_position = np.asarray(desired_position, dtype=float).reshape(-1, 3)
    node_count = seed_states.shape[0]
    if desired_position.shape[0] != node_count:
        raise ValueError("desired_position and seed_states disagree on node count.")
    tol = float(position_tolerance_m)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("position_tolerance_m must be finite and positive.")

    scale = weight / (tol * tol)
    blocks, linear, constant = [], np.zeros(7 * node_count), 0.0
    for index in range(node_count):
        state = seed_states[index]
        output = np.asarray(evaluator.output(index, state), dtype=float).reshape(-1)
        jacobian = np.asarray(
            evaluator.jacobian(index, state), dtype=float
        ).reshape(-1, 7)[:3, :]
        offset = output[:3] - jacobian @ state - desired_position[index]
        blocks.append(csc_matrix(2.0 * scale * (jacobian.T @ jacobian)))
        linear[7 * index : 7 * index + 7] = 2.0 * scale * (jacobian.T @ offset)
        constant += scale * float(offset @ offset)
    return block_diag(blocks, format="csc"), linear, float(constant)


def tip_curvature_quadratic(
    *,
    evaluator: Any,
    seed_states: Array,
    s: Array,
    weight: float,
) -> tuple[Any, Array, float]:
    """Penalise curvature of the *achieved* tip path, not just of ``chi``.

    Returns ``(H, f, constant)`` for ``0.5 x^T H x + f^T x + constant`` in the
    stacked variable ``x = [chi_0; ...; chi_{N-1}]``, to be added into the same
    sparse QP the path objective already builds.

    Why this term exists
    ---------------------
    ``_second_difference_matrix`` in ``global_constrained_configuration_path``
    penalises curvature of ``chi`` itself.  The beam-tip map
    ``p_j = p(chi_j)`` is nonlinear — its Jacobian changes fastest exactly
    where contact is engaging or disengaging — so a ``chi`` path with small,
    smooth second differences is *not* the same thing as a tip path with small
    second differences.  Nothing before this term ties adjacent *achieved* tip
    positions together at all: the hard task constraint only keeps each
    ``p_j`` inside its own tolerance ball, and ``tip_centring_quadratic`` only
    pulls each ``p_j`` toward its own static target.  Both leave the
    optimiser free to place ``p_j`` anywhere in its ball independently node to
    node, which is what a jagged tip path with a smooth joint path looks like.

    The math
    --------
    Tip position is linearised at the seed exactly as in
    ``tip_centring_quadratic``: ``p_j(chi_j) ~ J_j chi_j + c_j`` with
    ``c_j = p_j^0 - J_j chi_j^0``.  For interior node ``j`` with neighbour
    spacings ``L = s_j - s_{j-1}``, ``R = s_{j+1} - s_j``:

        D2 p_j = (2 / (L + R)) * (p_{j-1} / L - p_j * (1/L + 1/R) + p_{j+1} / R)

    is linear in ``[chi_{j-1}; chi_j; chi_{j+1}]``, so
    ``0.5 * w * sum_j || D2 p_j ||^2`` contributes one 21x21 block per
    interior node to the stacked Hessian — the tip-space analogue of what
    ``_second_difference_matrix`` already does for ``chi``, using the same
    seed-linearised beam Jacobian ``tip_centring_quadratic`` uses.

    No physical tolerance appears here (unlike ``tip_centring_quadratic``):
    this is a smoothness preference in metres, not a bound, so its own
    ``weight`` sets the trade-off against the ``chi``-space terms directly.
    """
    weight = float(weight)
    if weight <= 0.0:
        raise ValueError("tip-curvature weight must be positive to build a cost.")
    seed_states = np.asarray(seed_states, dtype=float).reshape(-1, 7)
    s = np.asarray(s, dtype=float).reshape(-1)
    node_count = seed_states.shape[0]
    if s.size != node_count:
        raise ValueError("s and seed_states disagree on node count.")
    if node_count < 3:
        return (
            csc_matrix((7 * node_count, 7 * node_count)),
            np.zeros(7 * node_count),
            0.0,
        )
    ds = np.diff(s)
    if np.any(ds <= 0.0):
        raise ValueError("s must be strictly increasing.")

    outputs = np.empty((node_count, 3), dtype=float)
    jacobians = np.empty((node_count, 3, 7), dtype=float)
    for index in range(node_count):
        state = seed_states[index]
        outputs[index] = np.asarray(
            evaluator.output(index, state), dtype=float
        ).reshape(-1)[:3]
        jacobians[index] = np.asarray(
            evaluator.jacobian(index, state), dtype=float
        ).reshape(-1, 7)[:3, :]
    offsets = outputs - np.einsum("nij,nj->ni", jacobians, seed_states)

    H = lil_matrix((7 * node_count, 7 * node_count), dtype=float)
    f = np.zeros(7 * node_count, dtype=float)
    constant = 0.0
    for interior in range(1, node_count - 1):
        left = float(ds[interior - 1])
        right = float(ds[interior])
        common = 2.0 / (left + right)
        coeff_left = common / left
        coeff_centre = -common * (1.0 / left + 1.0 / right)
        coeff_right = common / right

        # A_j: 3 x 21, maps [chi_{j-1}; chi_j; chi_{j+1}] -> D2 p_j.
        A_j = np.hstack(
            (
                coeff_left * jacobians[interior - 1],
                coeff_centre * jacobians[interior],
                coeff_right * jacobians[interior + 1],
            )
        )
        b_j = (
            coeff_left * offsets[interior - 1]
            + coeff_centre * offsets[interior]
            + coeff_right * offsets[interior + 1]
        )

        block = weight * (A_j.T @ A_j)
        linear = weight * (A_j.T @ b_j)
        constant += 0.5 * weight * float(b_j @ b_j)

        columns = np.concatenate(
            (
                np.arange(7 * (interior - 1), 7 * interior),
                np.arange(7 * interior, 7 * (interior + 1)),
                np.arange(7 * (interior + 1), 7 * (interior + 2)),
            )
        )
        H[np.ix_(columns, columns)] += block
        f[columns] += linear

    return csc_matrix(H), f, float(constant)


def effective_task_tolerances(
    *,
    position_tolerance_m: float,
    tangent_tolerance_rad: float,
    spend_fraction: float,
) -> tuple[float, float]:
    """Tolerances the global layer is allowed to spend.

    Reporting still uses the physical tolerances; only the constraint the
    optimiser sees is tightened, so the difference becomes margin the
    downstream controller inherits rather than budget the smoother consumes.
    """
    fraction = float(spend_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("tolerance_spend_fraction must lie in (0, 1].")
    return (
        float(position_tolerance_m) * fraction,
        float(tangent_tolerance_rad) * fraction,
    )


def balanced_second_difference_weights(
    *,
    s: Array,
    states: Array,
    configuration_scale: Sequence[float],
    reference_weight: float = 1.0e-8,
    floor_ratio: float = 1.0e-3,
) -> tuple[float, ...]:
    """Per-coordinate second-difference weights that equalise their cost share.

    With a uniform weight, the coordinate whose scaled second derivative is
    already small contributes least to the objective and is therefore the
    cheapest to sacrifice — which is how insertion ends up rougher after a
    smoothing pass that improved every joint.  Setting
    ``w_k proportional to 1 / rms_k^2`` makes every coordinate contribute the
    same amount at the seed, so the optimiser has no cheap coordinate to
    spend.

    The returned weights are normalised so their mean equals
    ``reference_weight``, keeping the smoothing term's overall magnitude, and
    hence its balance against the seed and centring terms, unchanged.
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    states = np.asarray(states, dtype=float).reshape(-1, 7)
    scale = np.asarray(configuration_scale, dtype=float).reshape(7)
    if s.size < 3:
        return tuple([float(reference_weight)] * 7)
    ds = np.diff(s)
    first = np.diff(states, axis=0) / ds[:, None] / scale[None, :]
    second = 2.0 * np.diff(first, axis=0) / (ds[:-1] + ds[1:])[:, None]
    rms = np.sqrt(np.mean(second * second, axis=0))
    positive = rms[rms > 0.0]
    floor = float(np.max(rms)) * float(floor_ratio) if positive.size else 1.0
    rms = np.maximum(rms, floor)
    weights = 1.0 / (rms * rms)
    weights = weights / float(np.mean(weights)) * float(reference_weight)
    return tuple(float(value) for value in weights)


__all__ = [
    "NodeExclusionConstraints",
    "balanced_second_difference_weights",
    "effective_task_tolerances",
    "tip_centring_quadratic",
    "tip_curvature_quadratic",
]
