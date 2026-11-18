from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from superarchiveallclo.capture import load_capture
from superarchiveallclo.scan import in_world, project, read_pcd, reconstruct
from superarchiveallclo.vision import TeeVision, picture_features

ROOT = Path(__file__).resolve().parents[1]


def garment_points():
    image = Image.new("L", (380, 290))
    ImageDraw.Draw(image).polygon([
        (34, 75), (105, 38), (158, 27), (175, 50), (205, 50), (222, 27), (277, 38),
        (346, 75), (324, 115), (284, 99), (284, 258), (96, 258), (96, 99), (56, 115),
    ], fill=1)
    y, x = np.where(np.asarray(image)[::2, ::2])
    return np.column_stack((x * 0.006, y * 0.006, np.zeros(len(x)))).astype("<f4")


def pcd(points, binary=True):
    header = f"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS {len(points)}\n"
    if binary:
        return header.encode() + b"DATA binary\n" + points.tobytes()
    rows = BytesIO()
    np.savetxt(rows, points)
    return header.encode() + b"DATA ascii\n" + rows.getvalue()


class FlatLayScanTest(TestCase):
    def test_ascii_and_binary_preserve_metres(self):
        points = garment_points()
        np.testing.assert_array_equal(read_pcd(pcd(points, binary=False)), read_pcd(pcd(points)))
        with self.assertRaisesRegex(ValueError, "binary length"):
            read_pcd(pcd(points)[:-4])

    def test_extrinsics_invert_a_rigid_transform(self):
        transform = [[0, -1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 3], [0, 0, 0, 1]]
        np.testing.assert_allclose(in_world(np.array([[1, 5, 5]]), json.dumps(transform)), [[4, 1, 2]])
        transform[2][2] = 2
        with self.assertRaisesRegex(ValueError, "preserve scale"):
            in_world(np.array([[1, 5, 5]]), json.dumps(transform))

    def test_project_retains_scale_and_rejects_nonflat_capture(self):
        points = garment_points()
        image, info = project(points, "x", "y", 2)
        _, evidence = picture_features(image, {"calibration": {"source": "fixture"}, "mm_per_pixel": 2})
        self.assertAlmostEqual(evidence["garment_box_mm"][0], np.ptp(points[:, 0]) * 1000, delta=16)
        self.assertAlmostEqual(evidence["garment_box_mm"][1], 690, delta=14)
        self.assertEqual(info["plane_distance_p95_mm"], 0)
        points[::2, 2] = 0.1
        with self.assertRaisesRegex(ValueError, "flat-lay plane"):
            project(points, "x", "y", 2)

    def test_scan_session_generates_a_candidate_without_a_fabricated_back_view(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "tee.pcd"
            raw.write_bytes(pcd(garment_points()))
            reconstruct(raw, root / "projection", item_id="fixture", source="test coordinates in metres")
            session = root / "projection/session.json"
            weights = ROOT / "models/tee-vision-front.npz"
            manifest, spec = load_capture(session, model_path=weights, blind=True)
            self.assertEqual(list(manifest["views"]), ["front"])
            self.assertEqual(manifest["measurements"]["back_neck_depth_mm"]["source"], "learned_prior")
            self.assertEqual(manifest["measurements"]["flat_chest_mm"]["source"], "scan_silhouette")
            self.assertEqual(manifest["measurements"]["back_length_mm"]["source"], "scan_with_prior")
            front = manifest["inference"]["scale_evidence"]["front"]
            self.assertAlmostEqual(spec.back_length_mm + spec.back_neck_depth_mm, front["shoulder_level_to_hem_mm"])
            with self.assertRaisesRegex(ValueError, "requires paired"):
                TeeVision(weights)
