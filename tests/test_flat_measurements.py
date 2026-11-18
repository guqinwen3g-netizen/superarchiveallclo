from unittest import TestCase

import numpy as np

from examples.evaluate_flat_landmarks import fixture
from superarchiveallclo.flat_measurements import draft_constraints, flat_landmarks
from superarchiveallclo.tee import TeeSpec, draft
from superarchiveallclo.vision import FIELDS


class FlatLandmarkTest(TestCase):
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
