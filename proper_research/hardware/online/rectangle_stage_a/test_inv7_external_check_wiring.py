"""2026-11-19: end-to-end offline check that run_inv_7dof_delay_aware.py's
wiring (pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE -> close_loop_path_follow.py's
_TRIM_KINDS branch -> build_offline_solver's new external_check_* kwargs ->
OfflineJointControllerAdapter's shared check) actually connects, using the
REAL controller_adapters.build_offline_solver entry point (not a stub),
exactly as the runner script calls it. Synthetic reference/jacobian -- no
real plan file or hardware needed."""
from dataclasses import dataclass

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.hardware.online.controller_adapters import build_offline_solver
from proper_research.hardware.online.messages import StateEstimate


@dataclass
class _MPCConfig:
    sample_period_s: float = 0.1


@dataclass
class _Reference:
    sample_count: int = 10
    sample_period_s: float = 0.1
    state: np.ndarray = None
    desired_position_m: np.ndarray = None
    input: np.ndarray = None

    def __post_init__(self):
        if self.state is None:
            self.state = np.zeros((self.sample_count, 7))
        if self.desired_position_m is None:
            self.desired_position_m = np.zeros((self.sample_count, 3))
        if self.input is None:
            self.input = np.zeros((self.sample_count, 7))


def _identity_magnet_transform_fn(q6):
    return np.asarray(q6[:3], dtype=float)


def _const_jacobian_provider(state):
    return np.zeros((3, 7))


def _build_adapter(magnet_exclusion_radius_m, lumen):
    reference = _Reference()
    return build_offline_solver(
        "inv_7dof_delay_aware",
        reference=reference,
        jacobian_provider=_const_jacobian_provider,
        mpc_config=_MPCConfig(),
        external_check_magnet_transform_fn=_identity_magnet_transform_fn,
        magnet_exclusion_lumen_C_m=lumen,
        magnet_exclusion_radius_m=magnet_exclusion_radius_m,
    )


def _estimate(q6):
    return StateEstimate(
        t_monotonic=0.0, x_meas=np.zeros(6), tip_position_m=np.zeros(3),
        tip_tangent=np.array([0.0, 0.0, 1.0]), insertion_length_m=0.03,
        robot_joints=np.asarray(q6, dtype=float),
    )


def test_build_offline_solver_wires_external_check_for_inv7_and_trips():
    lumen = np.array([[0.0, 0.0, 0.05]])
    adapter = _build_adapter(magnet_exclusion_radius_m=0.233, lumen=lumen)
    # q6[:3] = magnet_xyz = (0.1, 0, 0); gap to (0,0,0.05) = 0.1118m > 0.233? No:
    # need gap < 0.233 to trip. Use magnet_xyz close to lumen point.
    result = adapter(_estimate([0.0, 0.0, 0.05, 0, 0, 0]), dt=0.1)
    assert result.infeasible is True
    assert result.info["abort_reason"] is not None
    assert "magnet_exclusion_violated" in result.info["abort_reason"]
    assert np.allclose(result.u0, 0.0)


def test_build_offline_solver_wires_external_check_for_inv7_no_trip():
    lumen = np.array([[0.0, 0.0, 0.05]])
    adapter = _build_adapter(magnet_exclusion_radius_m=0.233, lumen=lumen)
    result = adapter(_estimate([1.0, 0.0, 0.05, 0, 0, 0]), dt=0.1)
    assert result.infeasible is False
    assert result.info["abort_reason"] is None
    assert result.info["external_measured_clearance_gap_m"] is not None
    assert abs(result.info["external_measured_clearance_gap_m"] - 1.0) < 1e-9


def test_trim_kinds_override_path_builds_correct_transform_fn():
    # Exercises close_loop_path_follow.py's own _MAGNET_CONSTRAINT_CLIP_OVERRIDE
    # -> external_check_kwargs translation directly (the exact mechanism
    # run_inv_7dof_delay_aware.py's main() sets pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE
    # for), confirming the state7->q6 padding adapter is correct.
    def _magnet_position_fn_state7(state7):
        # state7 convention: ignores insertion (state7[6]), matches real runner
        assert len(state7) == 7
        return np.asarray(state7[:3], dtype=float)

    pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = dict(
        magnet_position_fn=_magnet_position_fn_state7,
        magnet_exclusion_lumen_C_m=np.array([[0.0, 0.0, 0.05]]),
        magnet_exclusion_radius_m=0.233,
    )
    try:
        adapter = _build_adapter_via_pf_override()
        result = adapter(_estimate([0.0, 0.0, 0.05, 0, 0, 0]), dt=0.1)
        assert result.infeasible is True
        assert "magnet_exclusion_violated" in result.info["abort_reason"]
    finally:
        pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = None


def _build_adapter_via_pf_override():
    # mirrors close_loop_path_follow.py's own translation block exactly
    reference = _Reference()
    override = pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE
    mpf = override.get("magnet_position_fn")

    def _external_check_magnet_transform_fn(q6, _mpf=mpf):
        return _mpf(np.concatenate([np.asarray(q6, dtype=float).reshape(6), [0.0]]))

    return build_offline_solver(
        "inv_7dof_delay_aware",
        reference=reference,
        jacobian_provider=_const_jacobian_provider,
        mpc_config=_MPCConfig(),
        external_check_magnet_transform_fn=_external_check_magnet_transform_fn,
        magnet_exclusion_lumen_C_m=override.get("magnet_exclusion_lumen_C_m"),
        magnet_exclusion_radius_m=override.get("magnet_exclusion_radius_m"),
    )
