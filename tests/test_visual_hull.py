from unittest import TestCase

import numpy as np

from superarchiveallclo.visual_hull import (
    CameraView, carve, compare_surface, horizontal_spans, hull_mask, silhouette_bands, silhouette_metrics,
)


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

    def test_an_occluded_strip_is_neither_free_space_nor_a_contour(self):
        front = camera(0, 1, (10, 30))
        unobscured = camera(1, 1, (-25, -5))
        side = camera(1, 1, (-25, -5))
        rows, cols = np.indices(side.mask.shape)
        side.occluded = (cols >= 60) & (cols <= 65) & (rows >= 32) & (rows <= 76)
        side.mask = side.mask & ~side.occluded
        expected, original_low = carve([front, unobscured], 5)
        restored, low = carve([front, side], 5)
        np.testing.assert_array_equal(np.argwhere(restored) * 5 + low,
                                      np.argwhere(expected) * 5 + original_low)
        self.assertEqual(silhouette_metrics(side.mask, unobscured.mask, 1.0, side.occluded),
                         {"iou": 1.0, "contour_p90_mm": 0.0})

    def test_only_outside_surface_area_is_counted_as_a_violation(self):
        volume, low = carve([camera(0, 1, (10, 30)), camera(1, 1, (-25, -5))], 5)
        inside = np.array([[16, -15, -40], [20, -15, -40], [18, -15, -30]])
        vertices = np.vstack([inside, inside + [0, 36, 0]])
        report, _ = compare_surface(volume, low, 5, vertices, np.array([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(report["triangle_area_outside_by_more_than_one_voxel_fraction"], 0.5)
        self.assertGreater(report["one_sided_triangle_center_distance_p90_mm"], 25)

    def test_distinct_depths_are_measured_by_height_in_a_shared_frame(self):
        hull = np.array([[-30, -12, -50], [30, 12, -50], [-40, -20, -105], [40, 20, -105]])
        mesh = np.array([[-30, -12, -50], [30, 12, -50], [-39, -33, -125], [39, 37, -125],
                         [41, 0, -220]])
        top, body, below = horizontal_spans(hull, mesh, 5)
        self.assertEqual(top["hull_width_depth_mm"], [65, 29])
        self.assertEqual(body["range_below_datum_mm"], [100, 200])
        self.assertEqual(body["glb_width_depth_mm"], [78, 70])
        self.assertIsNone(below["hull_width_depth_mm"])

    def test_banded_silhouette_counts_include_model_spill_and_exclude_occlusion(self):
        observed = np.zeros((100, 160), dtype=bool)
        observed[20:40, 10:15] = True
        candidate = np.zeros_like(observed)
        candidate[25:45, 10:16] = True
        occluded = np.zeros_like(observed)
        occluded[30:40, 15] = True
        view = CameraView(observed, 0, 1, np.array([80, 20]), 2, occluded)
        first, second, spill = silhouette_bands(observed, candidate, view, band_mm=20)
        self.assertEqual((first["intersection_pixels"], first["reference_only_pixels"], first["model_only_pixels"]),
                         (25, 25, 5))
        self.assertEqual(second["iou"], 1)
        self.assertEqual(spill["range_below_datum_mm"], [40, 60])
        self.assertEqual(spill["model_only_pixels"], 30)
