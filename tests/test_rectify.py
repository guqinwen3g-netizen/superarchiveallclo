from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from superarchiveallclo.capture import capture_observations, capture_photo_hashes
from superarchiveallclo.rectify import prepare, project, rectangle, rectify, rectify_camera_plane


class RectifyTest(TestCase):
    def test_recovers_a_millimetre_grid_from_a_slanted_plane(self):
        y, x = np.indices((230, 300)) + 0.5
        plane_to_source = np.array([[1.3, 0.12, 35], [-0.15, 1.1, 65], [0.0011, -0.0005, 1.0]])
        observed_mm = project(np.linalg.inv(plane_to_source), np.column_stack((x.ravel(), y.ravel())))
        ramp = np.column_stack((observed_mm, observed_mm[:, 0] / 2 + observed_mm[:, 1] / 3))
        photo = Image.fromarray(np.rint(ramp + 40).clip(0, 255).astype(np.uint8).reshape(230, 300, 3))
        corners = project(plane_to_source, rectangle(140, 100))
        rectified, evidence = rectify(photo, {"width_mm": 140, "height_mm": 100, "corners_px": corners.tolist(),
                                               "field_mm": [-10, -6, 160, 120]}, 2)
        self.assertEqual(rectified.size, (85, 63))
        self.assertTrue(evidence["extrapolated_beyond_reference"])
        self.assertEqual(evidence["field_mm"], [-10, -6, 160, 120])
        row, col = np.indices((63, 85))
        np.testing.assert_allclose(np.asarray(rectified)[:, :, 0], 31 + 2 * col, atol=2)
        np.testing.assert_allclose(np.asarray(rectified)[:, :, 1], 35 + 2 * row, atol=2)

    def test_rejects_crossed_corners_and_missing_image_coverage(self):
        image = Image.new("RGB", (140, 120))
        reference = {"width_mm": 100, "height_mm": 80,
                     "corners_px": [[10, 10], [110, 10], [110, 90], [10, 90]]}
        crossed = dict(reference, corners_px=[reference["corners_px"][i] for i in (0, 2, 1, 3)])
        with self.assertRaisesRegex(ValueError, "convex rectangle"):
            rectify(image, crossed, 1)
        with self.assertRaisesRegex(ValueError, "outside the source photograph"):
            rectify(image, dict(reference, field_mm=[-15, 0, 100, 80]), 1)

    def test_measured_camera_plane_can_pad_background_within_a_visible_garment(self):
        image = Image.new("RGB", (120, 100), "gray")
        color = (10, 60, 160)
        ImageDraw.Draw(image).rectangle([20, 30, 65, 63], fill=color)
        mask = np.zeros((100, 120), dtype=bool)
        mask[30:64, 20:66] = True
        camera = {
            "source": "surveyed table", "image_size_px": [120, 100],
            "origin_camera_m": [0, 0, 1], "basis_camera": np.eye(3).tolist(),
            "intrinsics": {"fx": 1000, "fy": 1000, "cx": 60, "cy": 50},
            "field_mm": [-70, -60, 54, 45],
        }
        rectified, evidence = rectify_camera_plane(image, camera, 1, mask)
        self.assertTrue(evidence["padded_outside_source"])
        self.assertEqual(rectified.getpixel((30, 40)), color)
        self.assertEqual(rectified.getpixel((29, 40)), (255, 255, 255))
        self.assertEqual(rectified.getpixel((2, 2)), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "whole garment"):
            rectify_camera_plane(image, dict(camera, field_mm=[-30, -30, 54, 45]), 1, mask)
        with self.assertRaisesRegex(ValueError, "match the upright"):
            rectify_camera_plane(image, dict(camera, image_size_px=[119, 100]), 1, mask)
        with self.assertRaisesRegex(ValueError, "in front of the camera"):
            rectify_camera_plane(image, dict(camera, origin_camera_m=[0, 0, -1]), 1, mask)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image.save(root / "rgb.png")
            Image.fromarray(mask).save(root / "mask.png")
            source = root / "input.json"
            source.write_text(json.dumps({"schema_version": 1, "item_id": "camera-fixture", "source": "test",
                                          "views": {"front": {"rgb": "rgb.png", "mask": "mask.png",
                                                              "camera_plane": camera}}}))
            prepare(source, root / "rectified")
            paths = root / "rectified/session.json"
            capture, _ = capture_observations(paths, allow_partial=True)
            self.assertEqual(capture["views"]["front"]["source_assets"].keys(), {"rgb", "mask"})
            (root / "rectified/evidence/front-mask.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mask source evidence changed"):
                capture_observations(paths, allow_partial=True)

    def test_raw_photo_is_preserved_and_reuse_is_detected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "original.jpg"
            Image.new("RGB", (150, 120), (43, 78, 110)).save(photo)
            reference = {"width_mm": 120, "height_mm": 90,
                         "corners_px": [[10, 10], [130, 10], [137, 100], [12, 104]]}
            view = {"rgb": photo.name, "reference_plane": reference}
            record = {"schema_version": 1, "item_id": "planar-test", "kind": "synthetic_fixture",
                      "source": "fixture", "views": {"front": view, "back": view}}
            path = root / "planar.json"
            path.write_text(json.dumps(record))
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "separate original photographs"):
                prepare(path, output)
            self.assertFalse(output.exists())
            del record["views"]["back"]
            path.write_text(json.dumps(record))
            prepare(path, output)
            archive_photos = root / "photos"
            capture, _ = capture_observations(output / "session.json", archive_photos, allow_partial=True)
            actual = (root / "evidence/front-rgb.jpg").read_bytes()
            self.assertEqual(actual, photo.read_bytes())
            self.assertIn(sha256(actual).hexdigest(), capture_photo_hashes(capture["views"]))
            self.assertEqual(capture["views"]["front"]["calibration"]["origin_type"], "planar_reference")
            (output / "evidence/front-rgb.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "rgb source evidence changed"):
                capture_observations(output / "session.json", allow_partial=True)
