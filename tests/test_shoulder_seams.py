from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from superarchiveallclo import body_profile, shoulder_seams
from superarchiveallclo.tee import TeeSpec, draft


class ShoulderSeamsTest(TestCase):
    def test_finds_two_curved_joins_and_rejects_an_unmarked_view(self):
        outline = Image.new("1", (350, 380))
        ImageDraw.Draw(outline).polygon([
            (110, 34), (149, 19), (159, 26), (190, 26), (199, 19), (241, 34),
            (309, 73), (292, 131), (274, 130), (242, 117), (248, 340),
            (102, 340), (107, 117), (74, 131), (55, 131), (41, 73),
        ], fill=1)
        bounds = outline.getbbox()
        mask = np.asarray(outline.crop(bounds))
        step = 2.5
        torso = body_profile.observe(mask, step)
        photo = Image.fromarray(np.where(outline, 180, 30).astype(np.uint8))
        draw = ImageDraw.Draw(photo)
        draw.rectangle((140, 65, 175, 100), fill=40)
        t = np.linspace(0, 1, 120)
        for apex, underarm, bend in ((110, 107, 8), (241, 242, -8)):
            x = (1 - t) * apex + t * underarm + 4 * t * (1 - t) * bend
            y = (1 - t) * 34 + t * 117
            draw.line(list(zip(x, y, strict=True)), fill=130, width=2)
        measured = shoulder_seams.observe(np.asarray(photo.crop(bounds), dtype=float), mask, step, torso)
        self.assertAlmostEqual(measured["projected_width_mm"], (241 - 110) * step, delta=6)
        self.assertIsNone(shoulder_seams.observe(np.where(mask, 180.0, 30.0), mask, step, torso))

    def test_averages_agreeing_views_and_preserves_disagreement(self):
        views = {"front": {"projected_shoulder_seams": {"projected_width_mm": 505}},
                 "back": {"projected_shoulder_seams": {"projected_width_mm": 520}}}
        result = shoulder_seams.paired_evidence(views, 530)
        self.assertEqual(result["pattern_minus_projection_mm"], 17.5)
        views["back"]["projected_shoulder_seams"]["projected_width_mm"] = 540
        self.assertEqual(shoulder_seams.paired_evidence(views, 530)["status"], "projections_disagree")

    def test_calibrated_color_changes_only_the_inferred_shoulder_width(self):
        spec = TeeSpec()
        trace = {"projected_width_mm": 510.0, "near_tie_width_range_mm": [505, 513]}
        detail = {"registered_color_seams": {"projected_shoulder_seams": trace}}
        field = "garment_shoulder_width_mm"
        capture = {"views": {"front": {"calibration": detail}},
                   "measurements": {field: {"source": "vision_model", "value": spec.shoulder_width_mm}}}
        original = draft(spec)
        revised, pattern, report = shoulder_seams.refine_width(spec, original, capture)
        self.assertEqual(report["status"], "applied")
        self.assertEqual(revised.shoulder_width_mm, 510.0)
        self.assertEqual(pattern["parameters"]["sleeve_length_mm"], spec.sleeve_length_mm)
        self.assertEqual(capture["measurements"][field]["source"], "projected_shoulder_seams")
        trace["near_tie_width_range_mm"] = [456, 554]
        _, pattern, report = shoulder_seams.refine_width(spec, original, capture)
        self.assertEqual(report["status"], "uncertain_trace")
        self.assertIs(pattern, original)
