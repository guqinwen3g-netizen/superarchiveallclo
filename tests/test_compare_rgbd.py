from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from superarchiveallclo.compare_rgbd import reference_rays


class ReferenceDepthTest(TestCase):
    def test_camera_points_and_origin_share_the_declared_rigid_model_frame(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            depth = np.zeros((32, 32), dtype=np.int16)
            depth[16, 17] = 1200
            np.savez_compressed(root / "depth.npz", depth=depth)
            transform = [[0, -1, 0, 0.2], [1, 0, 0, -0.3], [0, 0, 1, 1.0], [0, 0, 0, 1]]
            view = {"depth": "depth.npz", "depth_sha256": sha256((root / "depth.npz").read_bytes()).hexdigest(),
                    "depth_unit": "mm", "intrinsics": {"fx": 30, "fy": 30, "cx": 16, "cy": 16},
                    "model_to_camera_m": transform, "pose_source": "Surveyed test fixture"}
            points, camera, _, raster = reference_rays(root, view)
            self.assertEqual(raster, (32, 32))
            np.testing.assert_allclose(points, [[0.3, 0.16, 0.2]])
            np.testing.assert_allclose(camera, [0.3, 0.2, -1.0])
            view["depth_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "source checksum"):
                reference_rays(root, view)
