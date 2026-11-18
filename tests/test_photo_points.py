from unittest import TestCase

import numpy as np

from superarchiveallclo.photo_points import PointCamera, nearest_surface_point, project, projection_matrix, triangulate


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
