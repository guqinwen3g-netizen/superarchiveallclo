from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from superarchiveallclo.compare_flat import sleeve_angle_from_cuffs, sleeve_angles_from_cuffs
from superarchiveallclo.sleeve_span import refine
from superarchiveallclo.tee import TeeSpec, draft


CUFFS = {
    "image_left": {"outer_mm": [np.float64(4), 130], "inner_mm": [51, 260]},
    "image_right": {"outer_mm": [810, 190], "inner_mm": [718, 302]},
}


class SleeveSpanTest(TestCase):
    def test_retains_both_directions_for_the_asymmetric_pose(self):
        left, right = sleeve_angles_from_cuffs(CUFFS)
        self.assertLess(left, right)
        self.assertAlmostEqual(sleeve_angle_from_cuffs(CUFFS), (left + right) / 2)
        self.assertIsNone(sleeve_angles_from_cuffs({"image_left": CUFFS["image_left"]}))

    def test_adapts_only_inferred_sleeve_length_from_input_span(self):
        spec = TeeSpec()
        pattern = draft(spec)
        measurements = {"sleeve_length_mm": {"source": "vision_model", "value": spec.sleeve_length_mm},
                        "flat_chest_mm": {"source": "vision_model", "value": spec.flat_chest_mm}}
        manifest = {"measurements": measurements, "inference": {"scale_evidence": {"front": {
            "flat_landmarks": {"cuffs": CUFFS}, "processed_mm_per_pixel": 2,
        }}}}
        with TemporaryDirectory() as root, patch("superarchiveallclo.sleeve_span.rendered_span") as render:
            render.side_effect = lambda candidate, *_: 350 + 2 * candidate["parameters"]["sleeve_length_mm"]
            fitted, evidence = refine(spec, pattern, manifest, Path(root) / "work", "blender")
            self.assertEqual(evidence["status"], "applied")
            self.assertEqual(evidence["input_cuff_span_mm"], 806)
            self.assertLessEqual(abs(fitted["parameters"]["sleeve_length_mm"] - 228), 2)
            self.assertEqual(measurements["sleeve_length_mm"]["source"], "calibrated_cuff_span_fit")
            self.assertEqual(measurements["flat_chest_mm"]["value"], spec.flat_chest_mm)
            self.assertEqual(pattern["parameters"]["sleeve_length_mm"], spec.sleeve_length_mm)
            self.assertEqual(len(fitted["seams"]), len(pattern["seams"]))

    def test_never_replaces_a_supplied_sleeve_length(self):
        spec = TeeSpec()
        measurements = {"sleeve_length_mm": {"source": "tape", "value": spec.sleeve_length_mm,
                                            "observation": {"value": spec.sleeve_length_mm}}}
        original = draft(spec)
        with patch("superarchiveallclo.sleeve_span.rendered_span") as render:
            pattern, report = refine(spec, original, {"measurements": measurements}, None, "blender")
        self.assertIs(pattern, original)
        self.assertEqual(report["status"], "retained_supplied_measurement")
        render.assert_not_called()
