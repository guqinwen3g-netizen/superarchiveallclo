from unittest import TestCase

import numpy as np

from superarchiveallclo.visual_hull import CameraView, carve, compare_surface, hull_mask


def camera(axis, sign, horizontal):
    origin = np.array([80, 20])
    v, u = np.indices((100, 160))
    coordinate = (u - origin[0]) * sign
    height = origin[1] - v
    mask = (coordinate >= horizontal[0]) & (coordinate <= horizontal[1]) & (height >= -55) & (height <= -10)
    return CameraView(mask, axis, sign, origin, 1.0)


class VisualHullTest(TestCase):
    def test_opposing_views_keep_the_world_axes_and_millimetres(self):
        views = [camera(0, 1, (10, 30)), camera(0, -1, (10, 30)),
                 camera(1, 1, (-25, -5)), camera(1, -1, (-25, -5))]
        volume, low = carve(views, 5)
        complete = np.argwhere(volume) * 5 + low
        np.testing.assert_array_equal(complete.min(axis=0), [10, -25, -55])
        np.testing.assert_array_equal(complete.max(axis=0), [30, -5, -10])
        for view in views:
            overlap = hull_mask(volume, low, 5, view)
            self.assertTrue(np.all(overlap[view.mask]))

    def test_parallel_pictures_cannot_identify_depth(self):
        with self.assertRaisesRegex(ValueError, "side direction"):
            carve([camera(0, 1, (10, 30)), camera(0, -1, (10, 30))], 5)

    def test_only_outside_surface_area_is_counted_as_a_violation(self):
        volume, low = carve([camera(0, 1, (10, 30)), camera(1, 1, (-25, -5))], 5)
        inside = np.array([[16, -15, -40], [20, -15, -40], [18, -15, -30]])
        vertices = np.vstack([inside, inside + [0, 36, 0]])
        report, _ = compare_surface(volume, low, 5, vertices, np.array([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(report["triangle_area_outside_by_more_than_one_voxel_fraction"], 0.5)
        self.assertGreater(report["one_sided_triangle_center_distance_p90_mm"], 25)
