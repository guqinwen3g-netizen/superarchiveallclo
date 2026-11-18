from unittest import TestCase

import numpy as np

from superarchiveallclo.photo_points import (
    PointCamera, checked_points, nearest_surface_point, pixel_sensitivity, project, projection_matrix, triangulate,
)


class PhotoPointTest(TestCase):
    def test_different_camera_directions_recover_measured_coordinates(self):
        views = {
            "front": PointCamera((120, 100), 0, 1, np.array([61, 17]), 2.0),
            "side": PointCamera((120, 100), 1, 1, np.array([54, 19]), 2.4),
            "back": PointCamera((120, 100), 0, -1, np.array([50, 21]), 2.2),
        }
        measured = np.array([18, -13, -42])
        for focal in (None, np.array([680.0, 702.0])):
            with self.subTest(focal=focal):
                cameras = {name: projection_matrix(view, focal) for name, view in views.items()}
                markers = {name: project(cameras[name], measured) for name in ("front", "side")}
                recovered = triangulate(cameras, views, markers)
                np.testing.assert_allclose(recovered, measured, atol=1e-8)
                np.testing.assert_allclose(project(cameras["back"], recovered), project(cameras["back"], measured))

    def test_nearest_point_includes_triangle_faces_edges_and_vertices(self):
        faces = np.array([[[0.0, 0, 0], [4.0, 0, 0], [0.0, 4, 0]]])
        for position, expected in (
            ([1, 1, 2], [1, 1, 0]),
            ([2, 3, 1], [1.5, 2.5, 0]),
            ([-1, 5, 2], [0, 4, 0]),
        ):
            with self.subTest(position=position):
                np.testing.assert_allclose(nearest_surface_point(position, faces), expected)

    def test_parallel_projection_cannot_resolve_depth(self):
        front = np.array([[1, 0, 0, 30], [0, 0, 1, 40], [0, 0, 0, 1]], dtype=float)
        side = np.array([[0, 1, 0, 30], [0, 0, 1, 40], [0, 0, 0, 1]], dtype=float)
        point = np.array([3, 4, 5])
        self.assertEqual(pixel_sensitivity(point, (front, side)), 1)
        self.assertIsNone(pixel_sensitivity(point, (front, front)))
        near_parallel = front.copy()
        near_parallel[0, 1] = 0.01
        self.assertGreater(pixel_sensitivity(point, (front, near_parallel)), 100)

    def test_point_limits_apply_to_all_fit_and_holdout_observations(self):
        def observation(fit_error, second_check_error):
            return {"region": "sleeve", "example": {
                "fit_residual_proxy_mm": {"carve": fit_error},
                "held_out_residual_proxy_mm": {"first": 2, "second": second_check_error},
                "distance_to_glb_mm": 7, "nearest_glb_minus_reference_xyz_mm": [3, 4, 5],
            }}

        points = {"both": observation(3, 12), "bad_fit": observation(14, 4), "bad_check": observation(3, 15)}
        results = checked_points(points, ("example",), 12)["methods"]["example"]
        self.assertEqual(results["accepted_ids"], ["both"])
        self.assertEqual(results["summary"]["distance_to_glb_median_mm"], 7)
        self.assertEqual(results["failed_views"]["bad_fit"], {"fit_residual_proxy_mm": ["carve"]})
        self.assertEqual(results["failed_views"]["bad_check"], {"held_out_residual_proxy_mm": ["second"]})
        self.assertIsNone(checked_points(points, ("example",), 1)["methods"]["example"]["summary"])
