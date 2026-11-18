from math import cos, radians, sin
from unittest import TestCase
import json

import numpy as np

from superarchiveallclo.calibrate_cameras import error_pixels, fit_projection
from superarchiveallclo.photo_points import PointCamera, project, supplied_matrices, triangulate


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
