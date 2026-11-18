from math import cos, radians, sin
from unittest import TestCase
import json

import numpy as np

from superarchiveallclo.calibrate_cameras import error_pixels, fit_planar_pose, fit_projection, size_from_planar_ruler
from superarchiveallclo.photo_points import PointCamera, project, supplied_matrices, triangulate
from superarchiveallclo.rectify import rectangle


class CalibratedProjectionTest(TestCase):
    def test_oblique_cameras_recover_a_held_out_metric_point(self):
        intrinsic = np.array([[695, 0, 320], [0, 710, 240], [0, 0, 1]])
        target = np.array([27, -63, 49])
        rng = np.random.default_rng(44)
        controls = rng.uniform(-170, 170, (32, 3))
        views, entries, markers = {}, {}, {}
        for name, yaw, position in (("near", 24, [-80, -15, 850]), ("oblique", -39, [90, 20, 725])):
            theta = radians(yaw)
            rotation = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
            camera = intrinsic @ np.column_stack((rotation, position))
            pixels = np.array([project(camera, point) for point in controls])
            fitted = fit_projection(controls[:22], pixels[:22])
            self.assertLess(error_pixels(fitted, controls[22:], pixels[22:])["max_px"], 1e-5)
            views[name] = PointCamera((640, 480), None, None, np.array([320, 240]), 1.4)
            entries[name] = {"photo_sha256": name, "image_from_world": fitted.tolist()}
            markers[name] = project(camera, target)
        manifest = {"schema_version": 1, "item_id": "tee", "world_unit": "mm", "source": "surveyed test fixture",
                    "image_size_px": [640, 480], "views": entries}
        raw = json.dumps(manifest).encode()
        matrices, _ = supplied_matrices(raw, "tee", (640, 480), {"near": "near", "oblique": "oblique"})
        np.testing.assert_allclose(triangulate(matrices, views, markers), target, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "different original photograph"):
            supplied_matrices(raw, "tee", (640, 480), {"near": "wrong", "oblique": "oblique"})

    def test_planar_controls_cannot_define_a_full_projective_camera(self):
        y, x = np.indices((3, 3)).astype(float)
        pixels = np.column_stack((x.ravel(), y.ravel()))
        world = np.column_stack((pixels, np.ones(len(pixels))))
        with self.assertRaisesRegex(ValueError, "do not span enough depth"):
            fit_projection(world, pixels)

    def test_planar_metric_pose_recovers_an_above_plane_control(self):
        theta = radians(27)
        intrinsic = np.array([[695, 0, 320], [0, 710, 240], [0, 0, 1]])
        rotation = np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
        translation = np.array([-75, 25, 880])
        camera = intrinsic @ np.column_stack((rotation, translation))
        ground = rectangle(320, 260)
        corners = np.array([project(camera, [*xy, 0]) for xy in ground])
        ruler = np.array([project(camera, [x, 90, 0]) for x in (-80, 420)])
        size, cosine = size_from_planar_ruler(corners, intrinsic, ruler, 500)
        np.testing.assert_allclose(size, [320, 260], atol=1e-6)
        self.assertAlmostEqual(cosine, 0)
        fitted, center = fit_planar_pose(ground, corners, intrinsic)
        np.testing.assert_allclose(center, -rotation.T @ translation, atol=1e-6)
        above_plane = np.array([196, 172, 210])
        np.testing.assert_allclose(project(fitted, above_plane), project(camera, above_plane), atol=1e-6)
