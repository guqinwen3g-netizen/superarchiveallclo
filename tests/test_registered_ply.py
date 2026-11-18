from io import BytesIO
from unittest import TestCase

import numpy as np

from superarchiveallclo.registered_ply import registered_depth


class RegisteredPlyTest(TestCase):
    def encoded(self, intrinsics, *, duplicate=False):
        y, x = np.indices((12, 16))
        z = 1.1 + .0015 * x + .0009 * y ** 2
        fx, fy, cx, cy = intrinsics
        points = np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=-1).reshape(-1, 3)
        angle = np.radians(23)
        cosine, sine = np.cos(angle), np.sin(angle)
        world = points @ np.array(((cosine, sine, 0), (-sine, cosine, 0), (0, 0, 1))) + (.2, -.9, .4)
        records = np.column_stack((world, x.ravel(), y.ravel(), z.ravel()))
        if duplicate:
            records[1, 3:5] = records[0, 3:5]
        stream = BytesIO()
        stream.write(("ply\nformat ascii 1.0\nelement vertex 192\n" + "".join(
            f"property float {key}\n" for key in ("x", "y", "z", "cx", "cy", "depth")
        ) + "element face 0\nend_header\n").encode())
        np.savetxt(stream, records[::-1], fmt="%.10f")
        return stream.getvalue(), z

    def test_rescales_rgb_intrinsics_but_preserves_metric_depth(self):
        raw, expected = self.encoded((104, 105, 59.5, 44.5))
        result = registered_depth(raw, (120, 90))
        np.testing.assert_allclose(result.depth, expected, atol=1e-7)
        self.assertEqual(result.diagnostics["serialized_intrinsics_raster"], "rgb")
        self.assertAlmostEqual(result.intrinsics["fx"], 104 / 7.5, places=6)
        self.assertAlmostEqual(result.intrinsics["cx"], 7.5, places=6)
        self.assertAlmostEqual(result.intrinsics["cy"], 5.5, places=6)

    def test_preserves_already_registered_intrinsics_and_rejects_duplicate_rays(self):
        values = (14, 15, 7.5, 5.5)
        raw, _ = self.encoded(values)
        result = registered_depth(raw, (120, 90))
        self.assertEqual(result.diagnostics["serialized_intrinsics_raster"], "depth")
        np.testing.assert_allclose(list(result.intrinsics.values()), values, atol=1e-6)
        repeated, _ = self.encoded(values, duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registered_depth(repeated, (120, 90))
