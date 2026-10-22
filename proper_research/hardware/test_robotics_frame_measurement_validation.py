"""Hardware-free tests for robotics_frame_measurement_validation.py."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import robotics_frame_measurement_validation as frames


class FramePipelineTests(unittest.TestCase):
    def test_named_transform_composition_and_inverse(self):
        T_R_B = frames.FrameTransform.from_pose6(
            "R", "B", [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2.0]
        )
        T_B_P = frames.FrameTransform.from_pose6(
            "B", "P", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        T_R_P = T_R_B.compose(T_B_P)
        point_R = T_R_P.apply_points([0.0, 0.0, 0.0])
        np.testing.assert_allclose(point_R, [1.0, 3.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(
            T_R_P.inverse().apply_points(point_R),
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_bad_frame_composition_is_rejected(self):
        T_R_B = frames.FrameTransform.identity("R", "B")
        T_TCP_M = frames.FrameTransform.identity("TCP", "M")
        with self.assertRaises(ValueError):
            T_R_B.compose(T_TCP_M)

    def test_basis_camera_map_matches_legacy_intended_formula(self):
        # Saved ex/ey are image-Cartesian.  Legacy intended coordinates were
        # B.x=-dot([du,-dv],ex), B.y=dot([du,-dv],ey).
        calibration = frames.PlanarPixelCalibration.from_basis_scale(
            origin_px=[100.0, 200.0],
            ex_saved=[1.0, 0.0],
            ey_saved=[0.0, 1.0],
            metres_per_pixel=0.001,
            saved_axis_convention="image_cartesian",
            positive_axis_signs=(-1.0, +1.0),
        )
        pixels = np.array([[90.0, 190.0], [110.0, 210.0]])
        points_B = calibration.pixels_to_beam(pixels)
        np.testing.assert_allclose(
            points_B,
            [[0.010, 0.010, 0.0], [-0.010, -0.010, 0.0]],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            calibration.beam_to_pixels(points_B), pixels, atol=1e-12
        )

    def test_tangent_uses_same_map_as_position(self):
        calibration = frames.PlanarPixelCalibration.from_basis_scale(
            origin_px=[20.0, 30.0],
            ex_saved=[1.0, 0.0],
            ey_saved=[0.0, 1.0],
            metres_per_pixel=0.002,
            saved_axis_convention="image_cartesian",
            positive_axis_signs=(-1.0, +1.0),
        )
        start_B = calibration.pixels_to_beam([20.0, 30.0])
        tip_B = calibration.pixels_to_beam([15.0, 25.0])
        tangent_B = frames._unit(tip_B - start_B, "synthetic tangent")
        np.testing.assert_allclose(
            tangent_B,
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            atol=1e-12,
        )

    def test_homography_round_trip(self):
        H = np.array(
            [
                [0.0010, 0.0001, -0.2],
                [0.0002, -0.0011, 0.3],
                [0.00001, 0.00002, 1.0],
            ]
        )
        calibration = frames.PlanarPixelCalibration.from_homography(H)
        pixels = np.array([[100.0, 120.0], [300.0, 240.0], [50.0, 400.0]])
        np.testing.assert_allclose(
            calibration.beam_to_pixels(calibration.pixels_to_beam(pixels)),
            pixels,
            atol=1e-10,
        )

    def test_tcp_to_magnet_rigid_offset(self):
        cfg = frames.FrameValidationConfig(
            T_tcp_magnet_pose6=(0.0, 0.0, 0.10, 0.0, 0.0, 0.0)
        )
        tcp_pose = np.array([1.0, 2.0, 3.0, 0.0, np.pi / 2.0, 0.0])
        T_R_TCP, T_TCP_M, T_R_M = frames.source_transform_from_tcp(tcp_pose, cfg)
        expected = T_R_TCP.translation + T_R_TCP.rotation @ np.array([0.0, 0.0, 0.10])
        np.testing.assert_allclose(T_R_M.translation, expected, atol=1e-12)
        np.testing.assert_allclose(T_R_M.matrix, T_R_TCP.matrix @ T_TCP_M.matrix)

    def test_missing_tool_transform_fails_early(self):
        cfg = frames.FrameValidationConfig(
            T_tcp_magnet_pose6=None,
            assume_tcp_is_magnet_frame=False,
        )
        with self.assertRaisesRegex(ValueError, "T_tcp_magnet_pose6 is required"):
            frames.validate_configuration(cfg)

    def test_pose_midpoint_and_drift(self):
        a = np.zeros(6)
        b = np.array([0.001, 0.0, 0.0, 0.0, 0.0, np.deg2rad(2.0)])
        midpoint = frames.midpoint_pose6(a, b)
        translation_mm, rotation_deg = frames.pose_drift(a, b)
        np.testing.assert_allclose(midpoint[:3], [0.0005, 0.0, 0.0])
        self.assertAlmostEqual(translation_mm, 1.0)
        self.assertAlmostEqual(rotation_deg, 2.0)
        self.assertAlmostEqual(
            Rot.from_rotvec(midpoint[3:]).magnitude(), np.deg2rad(1.0)
        )


if __name__ == "__main__":
    unittest.main()
