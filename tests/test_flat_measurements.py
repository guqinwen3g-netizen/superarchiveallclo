from math import cos, radians, sin
from unittest import TestCase

import numpy as np

from examples.evaluate_flat_landmarks import fixture
from superarchiveallclo.flat_measurements import (
    compatible_photo_sleeve_width, compatible_scan_sleeve_length, draft_constraints, flat_landmarks,
    paired_cuff_span, sleeve_layout,
)
from superarchiveallclo.tee import TeeSpec, body_panel, cap_dimensions, draft
from superarchiveallclo.vision import FIELDS


class FlatLandmarkTest(TestCase):
    def test_hidden_photo_width_can_be_constrained_without_changing_measured_cuffs(self):
        spec = TeeSpec(flat_chest_mm=511, garment_shoulder_width_mm=465, shoulder_slope_mm=51,
                       armhole_depth_mm=193, sleeve_bicep_flat_mm=368, sleeve_opening_flat_mm=337,
                       sleeve_length_mm=187)
        prior = {name: getattr(spec, name) for name in FIELDS}
        with self.assertRaisesRegex(ValueError, "incompatible with the armholes"):
            draft(spec)
        updates = compatible_photo_sleeve_width(prior, {})
        self.assertAlmostEqual(updates["sleeve_opening_flat_mm"] / updates["sleeve_bicep_flat_mm"], 337 / 368)
        self.assertEqual(len(draft(TeeSpec(**{**prior, **updates}))["panels"]), 5)
        self.assertEqual(compatible_photo_sleeve_width(prior, {"sleeve_opening_flat_mm": 280}), {})

    def test_join_projection_recovers_a_known_folded_sleeve_in_different_poses(self):
        part = TeeSpec(flat_chest_mm=610, garment_shoulder_width_mm=526, shoulder_slope_mm=43,
                       armhole_depth_mm=303, sleeve_bicep_flat_mm=458, sleeve_opening_flat_mm=365,
                       sleeve_length_mm=267)
        front, back = (body_panel(part, side).edge("armhole_right").length() for side in (True, False))
        _, cap = cap_dimensions(part.sleeve_bicep_flat_mm, part.sleeve_length_mm - 25, front, back)
        center, top = 500, 13
        cuffs = {}
        for side, key, degrees in ((-1, "image_left", 16), (1, "image_right", 29)):
            angle = radians(degrees)
            across = np.array([-side * sin(angle), cos(angle)])
            outward = np.array([side * cos(angle), sin(angle)])
            underarm = np.array([center + side * part.flat_chest_mm / 2, top + part.armhole_depth_mm])
            outer = underarm + outward * (part.sleeve_length_mm - cap) - across * part.sleeve_bicep_flat_mm / 2
            cuffs[key] = {"outer_mm": outer.tolist(),
                          "inner_mm": (outer + across * part.sleeve_opening_flat_mm / 2).tolist(),
                          "folded_opening_mm": part.sleeve_opening_flat_mm / 2,
                          "underarm_join_y_mm": float(underarm[1])}
        evidence = {"lower_torso_width_mm": part.flat_chest_mm, "shoulder_line_y_mm": top,
                    "shoulder_level_to_hem_mm": part.back_length_mm + part.back_neck_depth_mm,
                    "flat_landmarks": {"cuffs": cuffs, "torso_center_x_mm": center,
                                       "sleeve_layout": sleeve_layout(cuffs, center, part.flat_chest_mm)}}
        prior = {name: getattr(part, name) for name in FIELDS}
        prior.update(sleeve_bicep_flat_mm=384, sleeve_length_mm=208, armhole_depth_mm=255)
        estimates, sources, reason = draft_constraints(prior, evidence)
        self.assertIsNone(reason)
        for name in ("sleeve_bicep_flat_mm", "sleeve_length_mm", "armhole_depth_mm"):
            self.assertEqual(sources[name], "scan_projected_joins")
            self.assertAlmostEqual(estimates[name], getattr(part, name), delta=0.15)
        draft(TeeSpec(**{**prior, **estimates}))

    def test_cuff_span_recovers_sleeve_reach_with_one_hidden_underarm(self):
        part = TeeSpec(flat_chest_mm=550, garment_shoulder_width_mm=480, shoulder_slope_mm=46,
                       armhole_depth_mm=250, sleeve_bicep_flat_mm=385, sleeve_opening_flat_mm=330,
                       sleeve_length_mm=247)
        front, back = (body_panel(part, side).edge("armhole_right").length() for side in (True, False))
        _, cap = cap_dimensions(part.sleeve_bicep_flat_mm, min(front, back), front, back)
        cuffs = {}
        for side, key, degrees in ((-1, "image_left", 8), (1, "image_right", 18)):
            angle = radians(degrees)
            across = np.array([-side * sin(angle), cos(angle)])
            outward = np.array([side * cos(angle), sin(angle)])
            underarm = np.array([500 + side * part.flat_chest_mm / 2, part.armhole_depth_mm])
            outer = underarm + outward * (part.sleeve_length_mm - cap) - across * part.sleeve_bicep_flat_mm / 2
            # An off-center torso reading makes the two shoulder-based runs unreliable.
            cuffs[key] = {"outer_mm": outer.tolist(),
                         "inner_mm": (outer + across * part.sleeve_opening_flat_mm / 2).tolist(),
                         "folded_opening_mm": part.sleeve_opening_flat_mm / 2,
                         "underarm_join_y_mm": float(underarm[1]) if side == 1 else None}
        evidence = {"lower_torso_width_mm": part.flat_chest_mm, "shoulder_line_y_mm": 0,
                    "shoulder_level_to_hem_mm": part.back_length_mm + part.back_neck_depth_mm,
                    "flat_landmarks": {"cuffs": cuffs, "torso_center_x_mm": 576}}
        prior = {field: getattr(part, field) for field in FIELDS}
        prior["sleeve_length_mm"] = 180
        estimates, sources, reason = draft_constraints(prior, evidence)
        self.assertIsNone(reason)
        self.assertEqual(sources["sleeve_length_mm"], "scan_cuff_span")
        self.assertAlmostEqual(estimates["sleeve_length_mm"], part.sleeve_length_mm, delta=0.15)

        photo = {"cuff_span_landmarks": {"cuffs": cuffs}}
        estimates, report = paired_cuff_span(prior, {"front": photo, "back": photo})
        self.assertEqual(report["sources"], {"sleeve_length_mm": "paired_cuff_span"})
        self.assertAlmostEqual(estimates["sleeve_length_mm"], part.sleeve_length_mm, delta=0.15)
        different = dict(cuffs)
        right = cuffs["image_right"]
        different["image_right"] = {**right, **{name: [right[name][0] + 65, right[name][1]]
                                                for name in ("outer_mm", "inner_mm")}}
        estimates, report = paired_cuff_span(prior, {"front": photo, "back": {"cuff_span_landmarks": {"cuffs": different}}})
        self.assertEqual(estimates, {})
        self.assertIn("disagree", report["reason"])

    def test_folded_cuffs_preserve_scale_with_rotation_and_different_sleeve_poses(self):
        for seed in (1, 2, 6):
            with self.subTest(seed=seed):
                mask, mm, width, cuff_mm, _ = fixture(np.random.default_rng(seed))
                landmarks = flat_landmarks(mask, mm, width)
                for cuff in landmarks["cuffs"].values():
                    self.assertAlmostEqual(cuff["folded_opening_mm"], cuff_mm, delta=2)

    def test_unsupported_landmarks_leave_sleeve_and_neck_predictions_in_place(self):
        landmarks = flat_landmarks(np.ones((220, 175), dtype=bool), 3, 525)
        self.assertNotIn("cuffs", landmarks)
        self.assertNotIn("visible_neck", landmarks)
        learned = TeeSpec(garment_shoulder_width_mm=495)
        measures = {field: getattr(learned, field) for field in FIELDS}
        evidence = {"lower_torso_width_mm": 550, "shoulder_level_to_hem_mm": 716,
                    "shoulder_line_y_mm": 0, "flat_landmarks": landmarks}
        updates, sources, _ = draft_constraints(measures, evidence)
        self.assertEqual(set(sources), {"flat_chest_mm", "back_length_mm"})
        draft(TeeSpec(**{**measures, **updates}))
        self.assertIsNone(compatible_scan_sleeve_length(measures, updates))

    def test_hidden_sleeve_can_be_adjusted_to_match_the_armholes(self):
        prior = TeeSpec(armhole_depth_mm=223, sleeve_bicep_flat_mm=248, sleeve_opening_flat_mm=244,
                        sleeve_length_mm=120, shoulder_slope_mm=51, garment_shoulder_width_mm=420)
        measures = {field: getattr(prior, field) for field in FIELDS}
        scan = {"flat_chest_mm": 467}
        with self.assertRaisesRegex(ValueError, "incompatible with the armholes"):
            draft(TeeSpec(**{**measures, **scan}))
        corrected = compatible_scan_sleeve_length(measures, scan)
        self.assertGreater(corrected, prior.sleeve_length_mm)
        draft(TeeSpec(**{**measures, **scan, "sleeve_length_mm": corrected}))
        self.assertIsNone(compatible_scan_sleeve_length(measures, {**scan, "sleeve_length_mm": 155}))

    def test_incompatible_cuff_is_reported_without_overriding_a_valid_draft(self):
        learned = TeeSpec(garment_shoulder_width_mm=495)
        measures = {field: getattr(learned, field) for field in FIELDS}
        cuffs = {
            side: {"outer_mm": [x, 120], "inner_mm": [x, 330], "folded_opening_mm": 210,
                   "underarm_join_y_mm": 240}
            for side, x in (("image_left", 0), ("image_right", 940))
        }
        evidence = {"lower_torso_width_mm": 560, "shoulder_level_to_hem_mm": 704, "shoulder_line_y_mm": 0,
                    "flat_landmarks": {"cuffs": cuffs, "torso_center_x_mm": 470}}
        updates, _, reason = draft_constraints(measures, evidence)
        self.assertIn("armhole", reason)
        self.assertNotIn("sleeve_opening_flat_mm", updates)
        draft(TeeSpec(**{**measures, **updates}))
