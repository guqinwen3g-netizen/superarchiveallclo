from unittest import TestCase

import numpy as np

from superarchiveallclo.compare_hang import compare_front
from superarchiveallclo.hang_supports import place_for_gravity, support_frame


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
