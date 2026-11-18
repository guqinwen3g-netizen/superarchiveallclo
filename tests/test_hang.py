from unittest import TestCase

import numpy as np

from superarchiveallclo.compare_hang import compare_front, compare_ray_depth
from superarchiveallclo.hang_supports import pendulum_forecast, place_for_gravity, rotate_about_support, support_frame


class GravityPoseTest(TestCase):
    def test_equal_support_spacing_places_the_mesh_without_scaling(self):
        garment = np.array([[-0.25, 0, 0.002], [0.25, 0, 0.002], [-0.25, -0.67, 0.002], [0.25, -0.67, 0.002]])
        rest = np.column_stack((-garment[:, 1], garment[:, 0], garment[:, 2]))
        supports = np.array([[0.1, -0.25, 0.8], [0.1, 0.25, 0.8]])
        posed, patches, report = place_for_gravity(garment, rest, rest[:2], supports)
        np.testing.assert_allclose([posed[patch].mean(axis=0) for patch in patches], supports)
        np.testing.assert_allclose(posed[2:, 2], [0.13, 0.13], atol=1e-8)
        self.assertEqual(report["target_support_separation_mm"], 500)

    def test_rejects_an_ill_defined_transverse_support_pair(self):
        with self.assertRaisesRegex(ValueError, "transverse"):
            support_frame(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]), np.array([0.0, 0.0, 1.0]))

    def test_absolute_outline_disagrees_with_a_translated_mesh(self):
        points = np.array([[0.1, y, z] for y in np.linspace(-0.2, 0.2, 121) for z in np.linspace(0.2, 0.7, 151)])
        corners = np.array([[0.1, -0.2, 0.2], [0.1, 0.2, 0.2], [0.1, 0.2, 0.7], [0.1, -0.2, 0.7]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        direct, _ = compare_front(points, corners, faces)
        shifted, _ = compare_front(points, corners + [0, 0.15, 0], faces)
        self.assertGreater(direct, 0.94)
        self.assertLess(shifted, 0.6)

    def test_depth_scores_both_meshes_on_identical_observed_rays(self):
        cloud = np.array([[0, 0, level] for level in (0.94, 0.92, 0.77, 0.75, 0.73)])
        before = np.array([-4, 12, 100, 15, np.nan])
        after = np.array([np.nan, 2, np.nan, -5, 0])
        report = compare_ray_depth(cloud, before, after, 1.0)
        self.assertEqual((report["before_hit_rays"], report["after_hit_rays"], report["common_hit_rays"]), (4, 3, 2))
        self.assertEqual(report["before_on_common_rays"]["absolute_median_mm"], 13.5)
        self.assertEqual(report["after_on_common_rays"]["signed_median_mm"], -1.5)
        first, second = report["bands_by_height"]
        self.assertEqual(first["range_below_highest_support_mm"], [0, 100])
        self.assertEqual(second["after_on_common_rays"]["signed_median_mm"], -5)

    def test_no_common_depth_rays_yields_an_unavailable_score(self):
        cloud = np.array([[0, 0, 0.95], [0, 0, 0.94]])
        report = compare_ray_depth(cloud, np.array([3, np.nan]), np.array([np.nan, -1]), 1)
        self.assertEqual(report["common_hit_rays"], 0)
        self.assertIsNone(report["after_on_common_rays"])

    def test_swing_preserves_supports_and_rotates_the_hanging_surface(self):
        supports = np.array([[0, -0.25, 0], [0, 0.25, 0]])
        points = np.vstack((supports, [0, 0, -0.8]))
        rotated = rotate_about_support(points, supports, -np.pi / 2)
        np.testing.assert_allclose(rotated[:2], supports)
        np.testing.assert_allclose(rotated[2], [0.8, 0, 0], atol=1e-12)
        np.testing.assert_allclose(rotate_about_support(rotated, supports, np.pi / 2), points, atol=1e-12)

    def test_swing_forecast_starts_at_observed_angle(self):
        angles = np.array([-0.2, -0.1, 0.0])
        positions, velocity, period = pendulum_forecast(
            np.array([-0.2, -0.1, 0]), angles, np.array([0, 0.1]), 0.65,
        )
        self.assertEqual(positions[0], angles[-1])
        self.assertGreater(positions[1], positions[0])
        self.assertAlmostEqual(velocity, 1)
        self.assertAlmostEqual(period, 2 * np.pi * np.sqrt(2 * 0.65 / (3 * 9.81)))
