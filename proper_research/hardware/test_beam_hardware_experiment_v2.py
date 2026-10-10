"""Hardware-free unit tests for beam_hardware_experiment_v2.py.

Run from the repository root with either:

    python -m unittest -v test_beam_hardware_experiment_v2.py

or pytest.  These tests deliberately do not import or contact robot/camera
hardware.  Project-specific composite factory integration is checked by the
script's runtime validation; the pure configuration and control math are tested
here.
"""

from __future__ import annotations

import unittest

import numpy as np

import proper_research.hardware.beam_hardware_experiment_v2 as experiment


class LinearForwardModel:
    """Known linear p7->tip model with an exact analytic Jacobian."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float).reshape(3, 7)
        self.last_p7 = None
        self.last_jacobian_diag = {
            "hessian_reused": True,
            "difference_scheme": "synthetic_exact",
        }

    def __call__(self, p7, commit=True):
        p7 = np.asarray(p7, dtype=float).reshape(7)
        if commit:
            self.last_p7 = p7.copy()
        return self.matrix @ p7

    def jacobian_tip_actuation_tangent(self, p7, **_kwargs):
        p7 = np.asarray(p7, dtype=float).reshape(7)
        if self.last_p7 is None or not np.allclose(self.last_p7, p7):
            raise RuntimeError("Synthetic stale-cache check failed.")
        return self.matrix.copy()


def identity_p8(xyz=(0.3, -0.2, 0.1), length=0.044):
    # [xyz, quaternion_wxyz, insertion]
    return np.array([*xyz, 1.0, 0.0, 0.0, 0.0, length], dtype=float)


class ExperimentV2Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = experiment.ExperimentConfig()
        self.cfg.workspace_xyz_min_m = (-2.0, -2.0, -2.0)
        self.cfg.workspace_xyz_max_m = (+2.0, +2.0, +2.0)
        self.cfg.arc_center_robot_m = (0.0, 0.0, 0.0)
        self.cfg.arc_sweep_deg = 5.0
        self.cfg.arc_number_of_points = 20
        self.cfg.maximum_magnet_step_mm = 1.0

    def test_default_configuration_and_composite_are_valid(self):
        experiment.validate_config(self.cfg)
        experiment.validate_composite_config(self.cfg.composite)

        invalid = experiment.CompositeBeamConfig(particle_mass_fraction=1.1)
        with self.assertRaises(ValueError):
            experiment.validate_composite_config(invalid)

    def test_p8_to_p7_identity_quaternion(self):
        p8 = identity_p8()
        p7 = experiment.p8_to_p7(p8)
        np.testing.assert_allclose(p7[:3], p8[:3])
        np.testing.assert_allclose(p7[3:6], np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(p7[6], p8[7])

    def test_arc_generation_respects_start_and_step_limit(self):
        start = np.array([0.1, 0.0, 0.0])
        points = experiment.make_arc_points(self.cfg, start)
        self.assertEqual(points.shape, (20, 3))
        np.testing.assert_allclose(points[0], start)
        maximum_step = np.max(np.linalg.norm(np.diff(points, axis=0), axis=1))
        self.assertLessEqual(maximum_step, 0.001 + 1e-12)

    def test_damped_inverse_applies_axis_and_total_saturation(self):
        self.cfg.maximum_axis_steps_mm = (0.6, 0.4)
        self.cfg.maximum_magnet_step_mm = 0.7
        self.cfg.proportional_gain = 1.0
        self.cfg.damped_inverse_lambda = 0.0
        step = experiment.damped_inverse_step(
            np.eye(2),
            np.array([0.010, -0.010]),
            self.cfg,
        )
        self.assertLessEqual(abs(step[0]), 0.0006 + 1e-12)
        self.assertLessEqual(abs(step[1]), 0.0004 + 1e-12)
        self.assertLessEqual(np.linalg.norm(step), 0.0007 + 1e-12)

    def test_analytic_and_fd_jacobians_match_known_linear_model(self):
        matrix = np.array(
            [
                [2.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.1],
                [-0.1, 1.5, 0.0, 0.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        model = LinearForwardModel(matrix)
        adapter = experiment.DirectForwardModelAdapter(model, (0.0, 0.0, 0.0))
        self.cfg.jacobian_source = "compare"
        self.cfg.maximum_relative_jacobian_disagreement = 1e-8
        evaluation, _ = experiment.evaluate_jacobians(
            adapter,
            identity_p8(),
            self.cfg,
        )
        expected = matrix[np.ix_((0, 1), (0, 1))]
        np.testing.assert_allclose(evaluation.analytic, expected, atol=1e-12)
        np.testing.assert_allclose(
            evaluation.finite_difference,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertLess(evaluation.relative_frobenius_difference, 1e-10)
        self.assertTrue(evaluation.hessian_reused)
        # FD perturbations must finish with the nominal p7 cache restored.
        np.testing.assert_allclose(model.last_p7, experiment.p8_to_p7(identity_p8()))

    def test_jacobian_source_selection(self):
        matrix = np.zeros((3, 7))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        adapter = experiment.DirectForwardModelAdapter(
            LinearForwardModel(matrix),
            (0.0, 0.0, 0.0),
        )
        for source in ("analytic", "finite_difference"):
            with self.subTest(source=source):
                self.cfg.jacobian_source = source
                evaluation, _ = experiment.evaluate_jacobians(
                    adapter,
                    identity_p8(),
                    self.cfg,
                )
                np.testing.assert_allclose(evaluation.selected, np.eye(2), atol=1e-10)
                self.assertEqual(evaluation.analytic is not None, source == "analytic")
                self.assertEqual(
                    evaluation.finite_difference is not None,
                    source == "finite_difference",
                )

    def test_predicted_displacement_gate(self):
        evaluation = experiment.JacobianEvaluation(
            selected=np.eye(2),
            analytic=np.eye(2),
            finite_difference=None,
            analytic_time_s=0.0,
            finite_difference_time_s=0.0,
            analytic_condition=1.0,
            finite_difference_condition=np.nan,
            absolute_frobenius_difference=np.nan,
            relative_frobenius_difference=np.nan,
            maximum_absolute_element_error=np.nan,
            column_error_norms=np.full(2, np.nan),
            hessian_reused=False,
            diagnostics={},
        )
        self.cfg.maximum_predicted_tip_displacement_mm = 0.5
        with self.assertRaises(RuntimeError):
            experiment.enforce_predicted_displacement_gate(
                evaluation,
                np.array([0.001, 0.0]),
                self.cfg,
            )

    def test_log_row_contains_every_declared_field(self):
        state = {
            "p8": identity_p8(),
            "measured_tip": np.array([0.01, 0.02, 0.0]),
            "measured_tangent": np.array([1.0, 0.0, 0.0]),
            "model_tip": np.array([0.011, 0.019, 0.0]),
            "model_tangent": np.array([1.0, 0.0, 0.0]),
        }
        evaluation = experiment.JacobianEvaluation(
            selected=np.eye(2),
            analytic=np.eye(2),
            finite_difference=np.eye(2),
            analytic_time_s=0.001,
            finite_difference_time_s=0.002,
            analytic_condition=1.0,
            finite_difference_condition=1.0,
            absolute_frobenius_difference=0.0,
            relative_frobenius_difference=0.0,
            maximum_absolute_element_error=0.0,
            column_error_norms=np.zeros(2),
            hessian_reused=True,
            diagnostics={"hessian_reused": True},
        )
        metadata = {
            "composite_inputs": experiment.asdict(self.cfg.composite),
            "composite_calculated": {"moment_per_length": 1.23},
        }
        row = experiment.make_transition_row(
            sample=1,
            cfg=self.cfg,
            phase="test",
            target_index=0,
            iteration=0,
            pre_state=state,
            post_state=state,
            commanded_p8=identity_p8(),
            target_tip=np.zeros(3),
            model_at_command=np.r_[state["model_tip"], state["model_tangent"]],
            one_step_tip=state["model_tip"],
            evaluation=evaluation,
            active_step=np.array([0.0001, 0.0002]),
            translation_step=np.array([0.0001, 0.0002, 0.0]),
            bias={"position_m": np.zeros(3), "angle_deg": 0.0},
            composite_metadata=metadata,
            command_sent=False,
        )
        self.assertFalse(set(experiment.ROW_FIELDS) - set(row))


if __name__ == "__main__":
    unittest.main()
