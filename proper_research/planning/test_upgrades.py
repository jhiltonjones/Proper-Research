"""Verify the three layer-2 upgrades on a mock adapter, with no project models.

The mock's tip map is p = [q1, q2, L] and its magnet sits at the same point, so
a keep-out radius around a centreline the tip is asked to follow forces a real,
checkable trade: the smoother must hold the magnet off the centreline while the
tip stays inside its tolerance ball.
"""
import numpy as np
import global_constrained_configuration_path as G
import global_upgrades as U

Cfg = G.GlobalConfigurationOptimizerConfig


class Adapter(G._LinearMockAdapter):
    """Tip at [q1,q2,L]; magnet offset from it along +y by joint 5."""

    def magnet_transform(self, state):
        state = np.asarray(state, dtype=float).reshape(7)
        T = np.eye(4)
        T[:3, 3] = [state[0], state[1] + state[4], state[6]]
        return T

    def magnet_position_jacobian(self, state):
        J = np.zeros((3, 7))
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[1, 4] = 1.0
        J[2, 6] = 1.0
        return J


def make_seed(node_count=9, tip_tolerance=2.0e-4, magnet_offset=0.02):
    s = np.linspace(0.0, 0.02, node_count)
    centreline = np.column_stack((np.linspace(0.0, 0.02, 41), np.zeros(41), np.zeros(41)))
    path = G.CentrelinePath(centreline)
    desired_position = np.stack([path.position(v) for v in s])
    desired_tangent = np.stack([path.tangent(v) for v in s])
    states = np.zeros((node_count, 7))
    states[:, 0] = s
    # A deliberately rough seed inside the tolerance ball.
    states[:, 1] = 0.6 * tip_tolerance * np.array([1, -1] * node_count)[:node_count]
    states[:, 4] = magnet_offset
    seed = G._SeedPath(
        s=s, states=states,
        desired_position=desired_position, desired_tangent=desired_tangent,
        trusted_seed_mask=np.ones(node_count, dtype=bool),
        initial_state=states[0].copy(), last_feasible_progress_m=float(s[-1]),
    )
    return seed, path


def solve(config, seed, path, adapter):
    return G._solve_global_problem(
        adapter=adapter, seed=seed,
        state_min=np.array([-1.0] * 6 + [0.0]),
        state_max=np.array([1.0] * 6 + [0.1]),
        config=config, path=path,
    )


def tip_errors(adapter, states, desired):
    return np.array([
        np.linalg.norm(np.asarray(adapter.forward_output(state))[:3] - target)
        for state, target in zip(states, desired)
    ])


BASE = dict(
    mode="refine_complete", position_tolerance_m=2.0e-4, tangent_tolerance_rad=1.2,
    first_difference_weight=(1e-4,) * 7, second_difference_weight=(1e-6,) * 7,
    seed_deviation_weight=(0.0,) * 7, joint_centre_weight=(0.0,) * 7,
    maximum_adjacent_joint_change_rad=(0.25,) * 6, maximum_adjacent_insertion_change_m=3e-3,
    require_contact_model=False, fix_initial_state=False, maximum_iterations=200,
    dense_validation_enabled=False, compute_node_jacobian_diagnostics=False,
    feasibility_restoration_enabled=False, debug=False,
)
adapter = Adapter()
seed, path = make_seed()
failures = []

# ---------------------------------------------------------------- baseline
outcome, _, _ = solve(Cfg(**BASE), seed, path, adapter)
base_states = outcome.x.reshape(-1, 7)
base_tip = tip_errors(adapter, base_states, seed.desired_position)
base_magnet_y = np.abs(base_states[:, 1] + base_states[:, 4])
print(f"baseline          tip p95={1e6*np.percentile(base_tip,95):8.2f} um  "
      f"min |magnet y|={1e3*np.min(base_magnet_y):7.4f} mm  violation={outcome.constraint_violation:.2e}")

# ------------------------------------------------------- 1. the keep-out
RADIUS = 0.025
outcome_x, _, _ = solve(Cfg(**BASE, source_magnet_lumen_exclusion_radius_m=RADIUS),
                        seed, path, adapter)
states_x = outcome_x.x.reshape(-1, 7)
exclusion = U.NodeExclusionConstraints(
    adapter=adapter, path=path, state_min=np.array([-1.0] * 6 + [0.0]),
    state_max=np.array([1.0] * 6 + [0.1]), node_count=states_x.shape[0], radius_m=RADIUS)
metrics = exclusion.metrics(states_x.reshape(-1))
print(f"with keep-out     min margin={1e3*metrics['minimum_margin_m']:+7.4f} mm  "
      f"satisfied={metrics['all_satisfied']}  jac={metrics['jacobian_source']}")
if metrics["minimum_margin_m"] < -1e-9:
    failures.append("keep-out violated after smoothing")
# and confirm the unconstrained solve would have violated it, or the test is vacuous
base_metrics = exclusion.metrics(base_states.reshape(-1))
print(f"  (baseline margin={1e3*base_metrics['minimum_margin_m']:+7.4f} mm -> "
      f"{'constraint was binding' if base_metrics['minimum_margin_m'] < 0 else 'NOT binding: weak test'})")
if base_metrics["minimum_margin_m"] >= 0.0:
    failures.append("test geometry does not make the keep-out binding")

# --------------------------------------------- 2. tolerance spend fraction
for fraction in (1.0, 0.5, 0.25):
    out, _, _ = solve(Cfg(**BASE, tolerance_spend_fraction=fraction), seed, path, adapter)
    tip = tip_errors(adapter, out.x.reshape(-1, 7), seed.desired_position)
    budget = np.max(tip) / BASE["position_tolerance_m"]
    print(f"spend fraction {fraction:4.2f}  tip max={1e6*np.max(tip):8.2f} um  budget={budget:5.3f}x")
    if budget > fraction + 5e-3:
        failures.append(f"spend fraction {fraction} exceeded its budget ({budget:.3f})")

# ------------------------------------------------- 3. tip-centring weight
for weight in (0.0, 1.0e-3, 1.0e-1):
    out, _, _ = solve(Cfg(**BASE, tip_centring_weight=weight), seed, path, adapter)
    tip = tip_errors(adapter, out.x.reshape(-1, 7), seed.desired_position)
    print(f"tip weight {weight:7.0e}  tip rms={1e6*np.sqrt(np.mean(tip**2)):8.2f} um")
    if weight == 0.0:
        zero_rms = np.sqrt(np.mean(tip ** 2))
    else:
        if np.sqrt(np.mean(tip ** 2)) > zero_rms:
            failures.append(f"tip_centring_weight={weight} did not reduce tip error")

print()
if failures:
    raise SystemExit("FAILURES:\n  " + "\n  ".join(failures))
print("[UPGRADE TEST] PASS - keep-out enforced, spend fraction bounds the budget, "
      "tip cost reduces tip error")
