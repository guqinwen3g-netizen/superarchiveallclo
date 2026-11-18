import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from superarchiveallclo.capture import capture_observations, load_capture
from superarchiveallclo.rgbd import back_project, metric_projection, prepare


def oblique_flat_lay():
    camera = {"fx": 480, "fy": 500, "cx": 319.5, "cy": 239.5}
    y, x = np.indices((480, 640))
    tilt = np.radians(24)
    normal = np.array([0, -np.sin(tilt), np.cos(tilt)])
    depth = normal[2] / (normal[1] * (y - camera["cy"]) / camera["fy"] + normal[2])
    horizontal = (x - camera["cx"]) * depth / camera["fx"]
    vertical = ((y - camera["cy"]) * depth / camera["fy"]) / normal[2]
    torso = (np.abs(horizontal) <= 0.25) & (vertical >= -0.29) & (vertical <= 0.34)
    sleeves = (np.abs(horizontal) <= 0.39) & (vertical >= -0.24) & (vertical <= -0.12)
    neck = (np.abs(horizontal) < 0.07) & (vertical < -0.23)
    return camera, depth, (torso | sleeves) & ~neck


class RegisteredDepthTest(TestCase):
    def test_optical_depth_back_projection(self):
        depth = np.array([[1.25, 2.0]])
        points = back_project(depth, np.ones_like(depth, dtype=bool), {"fx": 4, "fy": 5, "cx": 0, "cy": 1})
        np.testing.assert_allclose(points, [[0, -0.25, 1.25], [0.5, -0.4, 2]])

    def test_flat_lay_stays_in_metric_scale_after_removing_camera_tilt(self):
        camera, depth, mask = oblique_flat_lay()
        _, evidence = metric_projection(depth, mask, camera, 2)
        np.testing.assert_allclose(evidence["raw_point_span_p0_1_to_p99_9_mm"], [780, 630], atol=5)
        self.assertLess(evidence["support_plane_residual_p90_mm"], 0.01)
        self.assertEqual(evidence["registered_depth_coverage"], 1)
        with self.assertRaisesRegex(ValueError, "covers less than"):
            metric_projection(np.where(mask, 0, depth), mask, camera, 2)

    def test_copied_archive_evidence_keeps_raw_sensor_data_and_rejects_tampering(self):
        camera, depth, mask = oblique_flat_lay()
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            rgb = np.where(mask, 50, 220).astype(np.uint8)
            Image.fromarray(rgb).save(directory / "rgb.png")
            Image.fromarray(rgb).resize((1280, 960)).save(directory / "detail.png")
            Image.fromarray(np.rint(depth * 1000).astype(np.uint16)).save(directory / "depth.png")
            Image.fromarray(np.where(mask, 2, 1).astype(np.uint8)).save(directory / "mask.png")
            source = directory / "input.json"
            source.write_text(json.dumps({
                "schema_version": 1, "kind": "synthetic_fixture", "item_id": "fixture", "source": "analytical camera fixture",
                "views": {"front": {"rgb": "rgb.png", "depth": "depth.png", "mask": "mask.png",
                                    "mask_label": 2, "depth_unit": "mm", "intrinsics": camera,
                                    "detail_rgb": "detail.png"}},
            }))
            output = directory / "projection"
            prepare(source, output)
            archive = directory / "archive"
            archive.mkdir()
            session = output / "session.json"
            state, _ = capture_observations(session, archive / "photos", allow_partial=True)
            self.assertEqual(state["input_mode"], "front_only")
            depth_asset = state["views"]["front"]["source_assets"]["depth"]["file"]
            self.assertEqual((archive / depth_asset).read_bytes(), (directory / "depth.png").read_bytes())
            detail_asset = state["views"]["front"]["source_assets"]["detail_rgb"]["file"]
            self.assertEqual((archive / detail_asset).read_bytes(), (directory / "detail.png").read_bytes())
            self.assertIn("processed_image_sha256", state["views"]["front"]["calibration"]["registered_color_seams"])
            (output / depth_asset).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "depth source evidence changed"):
                capture_observations(session, allow_partial=True)

    def test_paired_registered_views_can_use_the_paired_photo_model(self):
        camera, depth, front = oblique_flat_lay()
        back = front.copy()
        back[np.flatnonzero(back.any(axis=1))[-5]:] = False
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            np.save(directory / "depth.npy", depth)
            views = {}
            for name, mask in (("front", front), ("back", back)):
                Image.fromarray(np.where(mask, 45, 225).astype(np.uint8)).save(directory / f"{name}.png")
                views[name] = {"rgb": f"{name}.png", "depth": "depth.npy", "depth_unit": "m", "intrinsics": camera}
            source = directory / "input.json"
            source.write_text(json.dumps({"schema_version": 1, "kind": "synthetic_fixture", "item_id": "pair",
                                          "source": "Analytical camera fixture", "views": views}))
            output = directory / "projection"
            prepare(source, output)
            weights = Path(__file__).resolve().parents[1] / "models/tee-vision.npz"
            state, _ = load_capture(output / "session.json", model_path=weights, blind=True)
            self.assertEqual(state["inference"]["method"], "learned_paired_silhouette_ridge")
            self.assertEqual(set(state["views"]), {"front", "back"})

    def test_tiff_npz_and_measured_plane_keep_masked_physical_scale(self):
        camera, depth, mask = oblique_flat_lay()
        angle = np.radians(24)
        c, s = np.cos(angle), np.sin(angle)
        plane = {"source": "independent fixture", "origin_camera_m": [0, 0, 1],
                 "basis_camera": [[1, 0, 0], [0, c, -s], [0, s, c]]}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.where(mask, 40, 230).astype(np.uint8)).save(root / "rgb.tiff")
            masked = np.where(mask, depth, np.nan)
            np.savez_compressed(root / "depth.npz", depth=masked)
            source = root / "capture.json"
            source.write_text(json.dumps({
                "schema_version": 1, "kind": "synthetic_fixture", "item_id": "known-plane", "source": "Test fixture",
                "views": {"front": {"rgb": "rgb.tiff", "depth": "depth.npz", "depth_unit": "m",
                                    "intrinsics": camera, "support_plane": plane}},
            }))
            output = root / "prepared"
            prepare(source, output)
            manifest, _ = capture_observations(output / "session.json", allow_partial=True)
            front = manifest["views"]["front"]
            self.assertEqual(front["calibration"]["support_plane_method"], "measured_plane")
            self.assertIsNone(front["calibration"]["support_plane_residual_p90_mm"])
            np.testing.assert_allclose(front["calibration"]["raw_point_span_p0_1_to_p99_9_mm"], [780, 630], atol=5)
            for kind, original in (("rgb", "rgb.tiff"), ("depth", "depth.npz")):
                path = output / front["source_assets"][kind]["file"]
                self.assertEqual(path.read_bytes(), (root / original).read_bytes())
        with self.assertRaisesRegex(ValueError, "visible tabletop"):
            metric_projection(masked, mask, camera, 2)

    def test_organized_point_cloud_preserves_metric_scale_and_pixel_indices(self):
        camera, depth, mask = oblique_flat_lay()
        points = back_project(depth, np.ones_like(mask), camera).reshape(*mask.shape, 3)
        points[0, 0] = np.nan
        height, width = mask.shape
        header = (
            f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            f"WIDTH {width}\nHEIGHT {height}\nPOINTS {width * height}\nDATA binary\n"
        ).encode()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "xyz.pcd").write_bytes(header + (points * 1000).astype("<f4").tobytes())
            Image.fromarray(np.where(mask, 45, 225).astype(np.uint8)).save(root / "rgb.png")
            Image.fromarray(mask.astype(np.uint8)).save(root / "mask.png")
            source = root / "input.json"
            source.write_text(json.dumps({
                "schema_version": 1, "kind": "synthetic_fixture", "item_id": "registered-grid",
                "source": "Analytical registered grid",
                "views": {"front": {"rgb": "rgb.png", "pcd": "xyz.pcd", "pcd_unit": "mm", "mask": "mask.png"}},
            }))
            output = root / "projection"
            prepare(source, output)
            copied = root / "archive"
            copied.mkdir()
            capture, _ = capture_observations(output / "session.json", copied / "photos", allow_partial=True)
            view = capture["views"]["front"]
            self.assertEqual(view["calibration"]["origin_type"], "registered_pcd")
            np.testing.assert_allclose(view["calibration"]["raw_point_span_p0_1_to_p99_9_mm"], [780, 630], atol=5)
            self.assertEqual((copied / view["source_assets"]["pcd"]["file"]).read_bytes(), (root / "xyz.pcd").read_bytes())
