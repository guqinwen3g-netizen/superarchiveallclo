from unittest import TestCase

import numpy as np

from superarchiveallclo.mesh_relax import projection_area, relax_flat_metric


class FlatMetricTest(TestCase):
    def test_relaxation_recovers_lengths_above_the_support_with_pinned_vertices(self):
        target = np.array([[0, 0, 0], [0.12, 0, 0], [0.12, 0.2, 0.002], [0, 0.2, 0.002]])
        vertices = target.copy()
        vertices[2, :2] = [0.16, 0.18]
        vertices[3, :2] = [0.025, 0.16]
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]])
        a, b = edges.T
        rest = np.linalg.norm(target[b] - target[a], axis=1)
        relax_flat_metric(vertices, edges, rest, [0, 1], faces)
        np.testing.assert_array_equal(vertices[:2], target[:2])
        np.testing.assert_array_equal(vertices[:, 2], target[:, 2])
        np.testing.assert_allclose(np.linalg.norm(vertices[b] - vertices[a], axis=1), rest, atol=0.001)
        self.assertTrue((projection_area(vertices, faces) > 0).all())

    def test_higher_edge_weight_preserves_that_edge_under_conflicting_lengths(self):
        target = np.array([[0, 0, 0], [0.2, 0, 0], [0.1, 0.015, 0]])
        edges = np.array([[0, 1], [0, 2], [1, 2]])
        faces = np.array([[0, 1, 2]])
        lengths = np.array([0.2, 0.08, 0.08])
        uniform, weighted = target.copy(), target.copy()
        relax_flat_metric(uniform, edges, lengths, [0, 1], faces)
        relax_flat_metric(weighted, edges, lengths, [0, 1], faces, np.array([1, 4, 1]))
        self.assertLess(np.linalg.norm(weighted[2] - weighted[0]), np.linalg.norm(uniform[2] - uniform[0]) - 0.005)
        self.assertGreater(projection_area(weighted, faces)[0], 0)
