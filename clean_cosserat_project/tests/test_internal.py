from __future__ import annotations

import unittest

import numpy as np

from cosserat_clean.analysis import jacobian_metrics
from cosserat_clean.magnetics import dipole_field_from_source
from cosserat_clean.mechanics import effective_lengths, rod_section_stiffness
from cosserat_clean.quaternions import quat_from_small_rotation, quat_mul, skew


class InternalUtilitiesTest(unittest.TestCase):
    def test_effective_lengths_short_insertion(self) -> None:
        model, wire, tip = effective_lengths(
            0.005, full_tip_length_m=0.04, minimum_model_length_m=0.01
        )
        self.assertAlmostEqual(model, 0.01)
        self.assertAlmostEqual(wire, 0.0)
        self.assertAlmostEqual(tip, 0.005)

    def test_effective_lengths_long_insertion(self) -> None:
        model, wire, tip = effective_lengths(
            0.1, full_tip_length_m=0.04, minimum_model_length_m=0.01
        )
        self.assertAlmostEqual(model, 0.1)
        self.assertAlmostEqual(wire, 0.06)
        self.assertAlmostEqual(tip, 0.04)

    def test_quaternion_identity_product(self) -> None:
        q = quat_from_small_rotation(np.array([0.1, -0.2, 0.3]))
        np.testing.assert_allclose(quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), q), q)

    def test_skew_matches_cross_product(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-0.5, 4.0, 2.0])
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))

    def test_dipole_field_on_axis(self) -> None:
        point = np.array([[1.0, 0.0, 0.0]])
        source = np.zeros(3)
        dipole = np.array([1.0, 0.0, 0.0])
        field = dipole_field_from_source(point, source, dipole)[0]
        np.testing.assert_allclose(field, np.array([2e-7, 0.0, 0.0]), rtol=1e-12)

    def test_section_stiffness_positive(self) -> None:
        section = rod_section_stiffness(1e-3, 1e6, 0.3)
        self.assertGreater(section.bending_stiffness_nm2, 0.0)
        self.assertGreater(section.torsional_stiffness_nm2, 0.0)

    def test_jacobian_metrics(self) -> None:
        jacobian = np.zeros((3, 7))
        jacobian[:, :3] = np.eye(3)
        metrics = jacobian_metrics(jacobian)
        self.assertAlmostEqual(metrics["condition_3d"], 1.0)
        self.assertAlmostEqual(metrics["sigma3_3d"], 1.0)


if __name__ == "__main__":
    unittest.main()
