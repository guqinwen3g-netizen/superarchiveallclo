from math import cos, radians, sin
from unittest import TestCase

import numpy as np

from examples.evaluate_flat_landmarks import fixture
from superarchiveallclo.flat_measurements import draft_constraints, flat_landmarks, sleeve_layout
from superarchiveallclo.tee import TeeSpec, body_panel, cap_dimensions, draft
from superarchiveallclo.vision import FIELDS


class FlatLandmarkTest(TestCase):
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
