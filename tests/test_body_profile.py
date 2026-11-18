from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from superarchiveallclo.body_profile import observe
from superarchiveallclo.tee import TeeSpec, draft


class BodyProfileTest(TestCase):
    def test_recovers_different_chest_and_hem_widths(self):
        for hem_width in (111, 136, 161):
            with self.subTest(hem_width=hem_width):
                image = Image.new("1", (350, 380))
                hem_left = 175 - hem_width // 2
                hem_right = hem_left + hem_width - 1
                ImageDraw.Draw(image).polygon([
                    (110, 34), (149, 19), (159, 26), (190, 26), (199, 19), (241, 34),
                    (309, 73), (292, 131), (274, 130), (242, 117), (hem_right, 340),
                    (hem_left, 340), (107, 117), (74, 131), (55, 131), (41, 73),
                ], fill=1)
                result = observe(np.asarray(image.crop(image.getbbox())), 2.5)
                self.assertAlmostEqual(result["chest_width_mm"], 340, delta=3)
                self.assertAlmostEqual(result["hem_width_mm"], 2.5 * hem_width, delta=3)

    def test_rejects_a_torso_without_visible_sleeves(self):
        self.assertIsNone(observe(np.ones((260, 180), dtype=bool), 2.5))

    def test_rejects_a_scale_smaller_than_a_sleeve_trace(self):
        self.assertIsNone(observe(np.ones((260, 180), dtype=bool), 1e-12))

    def test_drafted_hem_changes_the_side_seams_and_preserves_chest(self):
        spec = TeeSpec(flat_hem_mm=620)
        pattern = draft(spec)
        for panel in pattern["panels"][:2]:
            edges = {edge["id"]: edge for edge in panel["edges"]}
            self.assertEqual(edges["side_right"]["start"][0] * 2, spec.flat_chest_mm)
            self.assertEqual(edges["side_right"]["end"][0] * 2, spec.flat_hem_mm)
            self.assertEqual(edges["hem"]["start"][0] - edges["hem"]["end"][0], spec.flat_hem_mm)
        self.assertNotIn("flat_hem_mm", draft(TeeSpec())["parameters"])

    def test_shoulder_may_overhang_the_underarm(self):
        pattern = draft(TeeSpec(garment_shoulder_width_mm=500, flat_chest_mm=490, flat_hem_mm=540))
        self.assertEqual(pattern["measurements"]["across_shoulders_mm"], 500)
        self.assertEqual(len(pattern["seams"]), 15)
