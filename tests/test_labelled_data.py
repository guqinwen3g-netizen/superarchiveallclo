from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from superarchiveallclo.capture import load_capture
from superarchiveallclo.dataset import arrays, evaluate, load_dataset
from superarchiveallclo.train_vision import synthetic_view
from superarchiveallclo.tee import TeeSpec
from superarchiveallclo.vision import FIELDS, TeeVision

ROOT = Path(__file__).resolve().parents[1]


class LabelledDatasetTest(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def entry(self, item, group, split, *, complete=False, photo_from=None, blank=False):
        spec = TeeSpec(garment_shoulder_width_mm=495)
        folder = self.root / item
        folder.mkdir()
        views = {}
        for view in ("front", "back"):
            if photo_from:
                photo = self.root / photo_from / f"{view}.png"
            else:
                seed = len(list(self.root.iterdir())) * 19 + (view == "front")
                image = synthetic_view(spec, view == "front", np.random.default_rng(seed))
                if blank:
                    gray = 230 + (view == "front")
                    image = Image.new("RGB", image.size, (gray, gray, gray))
                photo = folder / f"{view}.png"
                image.save(photo)
            views[view] = {"file": str(photo)}
        measurements = {
            field: {"value": getattr(spec, field)}
            for field in (FIELDS if complete else ("flat_chest_mm", "back_length_mm"))
        }
        (folder / "session.json").write_text(json.dumps({
            "schema_version": 1, "kind": "synthetic_fixture", "garment": "crew_neck_tee",
            "item_id": item, "views": views, "measurements": measurements,
        }), encoding="utf8")
        return {
            "session": f"{item}/session.json", "group": group, "split": split,
            "rights": "Procedurally generated integration-test fixture",
        }

    def manifest(self, *entries):
        file = self.root / "dataset.json"
        file.write_text(json.dumps({"schema_version": 1, "samples": entries}), encoding="utf8")
        return file

    def test_partial_test_labels_and_rejected_captures_are_counted(self):
        manifest = self.manifest(
            self.entry("labelled", "style-a", "train", complete=True),
            self.entry("heldout", "style-b", "test"),
            self.entry("bad_photo", "style-c", "test", blank=True),
        )
        samples, checksum = load_dataset(manifest)
        self.assertEqual(checksum, sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(arrays(samples, "train")[1].shape, (1, len(FIELDS)))
        self.assertIsNone(arrays(samples, "test"))
        metrics = evaluate(samples, TeeVision(ROOT / "models/tee-vision.npz"))
        self.assertEqual(list(metrics), ["synthetic_fixture"])
        holdout = metrics["synthetic_fixture"]
        self.assertEqual(holdout["cases"], 2)
        self.assertEqual(holdout["status_counts"], {"pattern_passed": 1, "capture_rejected": 1})
        self.assertEqual(holdout["measurement_error_mm"]["flat_chest_mm"]["count"], 1)
        self.assertNotIn("neck_width_mm", holdout["measurement_error_mm"])

    def test_style_group_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            load_dataset(self.manifest(
                self.entry("original", "same-style", "train"),
                self.entry("rephoto", "same-style", "test"),
            ))

    def test_duplicate_photo_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "duplicate photo"):
            load_dataset(self.manifest(
                self.entry("first", "style-a", "train"),
                self.entry("duplicate", "style-b", "test", photo_from="first"),
            ))

    def test_existing_photo_only_inference_and_archival_copy(self):
        capture, _ = load_capture(ROOT / "examples/vision-demo/session.json", self.root / "photos",
                                  ROOT / "models/tee-vision.npz")
        self.assertEqual(sum(d["source"] == "vision_model" for d in capture["measurements"].values()), 11)
        for data in capture["views"].values():
            self.assertEqual(sha256((self.root / data["file"]).read_bytes()).hexdigest(), data["sha256"])
