from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from superarchiveallclo.capture import load_capture
from superarchiveallclo.dataset import arrays, evaluate, load_dataset
from superarchiveallclo.review import verify_prediction
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

    def test_a_catalogue_dimension_used_for_photo_scale_is_held_out_of_scoring(self):
        entry = self.entry("museum-garment", "accession-1", "test")
        session = self.root / entry["session"]
        data = json.loads(session.read_text())
        data["views"]["back"]["scale"] = {"mm": 680, "points": [[200, 52], [200, 264]],
                                            "basis_field": "back_length_mm"}
        data["measurements"]["flat_chest_mm"].update(method="catalogue_measurement", source="Museum catalogue")
        session.write_text(json.dumps(data))
        samples, _ = load_dataset(self.manifest(entry))
        self.assertEqual(list(samples[0].truth), ["flat_chest_mm"])
        self.assertEqual(samples[0].sources["flat_chest_mm"], "catalogue_measurement")

    def test_reprojected_depth_cannot_cross_splits(self):
        training = self.entry("first", "style-a", "train")
        testing = self.entry("reprojection", "style-b", "test")
        for entry in (training, testing):
            session = self.root / entry["session"]
            data = json.loads(session.read_text())
            evidence = session.parent / "evidence"
            evidence.mkdir()
            rgb = (session.parent / "front.png").read_bytes()
            assets = {}
            for key, name, payload in (("rgb", "front-rgb.png", rgb), ("depth", "front-depth.npy", b"same depth")):
                (evidence / name).write_bytes(payload)
                assets[key] = {"file": f"evidence/{name}", "sha256": sha256(payload).hexdigest()}
            data["views"]["front"]["source_assets"] = assets
            session.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "duplicate photo or raw depth"):
            load_dataset(self.manifest(training, testing))

    def test_existing_photo_only_inference_and_archival_copy(self):
        capture, _ = load_capture(ROOT / "examples/vision-demo/session.json", self.root / "photos",
                                  ROOT / "models/tee-vision.npz")
        self.assertEqual(sum(d["source"] == "vision_model" for d in capture["measurements"].values()), 11)
        for data in capture["views"].values():
            self.assertEqual(sha256((self.root / data["file"]).read_bytes()).hexdigest(), data["sha256"])

    def test_blind_mode_does_not_feed_reference_labels_into_reconstruction(self):
        entry = self.entry("blind", "style-d", "test")
        session = self.root / entry["session"]
        source = json.loads(session.read_text())
        source["measurements"]["flat_chest_mm"] = {"value": 432}
        session.write_text(json.dumps(source), encoding="utf8")
        model = ROOT / "models/tee-vision.npz"
        manual, _ = load_capture(session, model_path=model)
        inferred, _ = load_capture(session, model_path=model, blind=True)
        self.assertEqual(manual["measurements"]["flat_chest_mm"]["value"], 432)
        self.assertEqual(inferred["measurements"]["flat_chest_mm"]["source"], "vision_model")
        self.assertTrue(inferred["inference"]["reference_measurements_withheld"])
        self.assertTrue(all(m["source"] == "vision_model" for name, m in inferred["measurements"].items()
                            if name in FIELDS))

    def test_review_rejects_a_mismatched_photo_or_prediction(self):
        samples, _ = load_dataset(self.manifest(self.entry("scored", "style-e", "test")))
        sample = samples[0]
        weights = ROOT / "models/tee-vision.npz"
        vision = TeeVision(weights)
        result = evaluate(samples, vision)[sample.kind]["items"][0]
        archive, _ = load_capture(sample.session, model_path=weights, blind=True)
        verify_prediction(archive, sample, vision, result)
        old_hash = archive["views"]["front"]["sha256"]
        archive["views"]["front"]["sha256"] = "different photograph"
        with self.assertRaisesRegex(ValueError, "photographs changed"):
            verify_prediction(archive, sample, vision, result)
        archive["views"]["front"]["sha256"] = old_hash
        archive["measurements"]["flat_chest_mm"]["value"] += 10
        with self.assertRaisesRegex(ValueError, "differs from the scored prediction"):
            verify_prediction(archive, sample, vision, result)

    def test_paired_body_profile_scores_the_same_hem_and_chest_that_it_archives(self):
        entry = self.entry("body-profile", "style-f", "test")
        session = self.root / entry["session"]
        source = json.loads(session.read_text())
        source["body_profile"] = "paired_silhouette"
        source["measurements"]["flat_hem_mm"] = {"value": 560}
        session.write_text(json.dumps(source), encoding="utf8")
        samples, _ = load_dataset(self.manifest(entry))
        sample = samples[0]
        weights = ROOT / "models/tee-vision.npz"
        vision = TeeVision(weights)
        result = evaluate(samples, vision)[sample.kind]["items"][0]
        self.assertEqual(result["body_profile"]["status"], "applied")
        self.assertLess(result["absolute_error_mm"]["flat_hem_mm"], 15)
        capture, spec = load_capture(sample.session, model_path=weights, blind=True)
        self.assertEqual(capture["measurements"]["flat_hem_mm"]["source"], "paired_silhouette")
        self.assertEqual(spec.flat_hem_mm, capture["measurements"]["flat_hem_mm"]["value"])
        verify_prediction(capture, sample, vision, result)

    def test_test_item_cannot_reuse_a_photo_recorded_in_weights(self):
        samples, _ = load_dataset(self.manifest(self.entry("cross-manifest", "holdout-group", "test")))
        weights = self.root / "previous.npz"
        with np.load(ROOT / "models/tee-vision.npz", allow_pickle=False) as original:
            updated = dict(original)
        updated["labelled_training_groups"] = np.array(["other-group"])
        updated["labelled_training_photos"] = np.array([samples[0].photo_hashes["back"]])
        np.savez_compressed(weights, **updated)
        result = evaluate(samples, TeeVision(weights))["synthetic_fixture"]
        self.assertEqual(result["status_counts"], {"training_overlap": 1})
        self.assertEqual(result["measurement_error_mm"], {})
