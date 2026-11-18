from unittest import TestCase

import numpy as np

from superarchiveallclo.tee import TeeSpec, draft
from superarchiveallclo.vision import ARMHOLE, BICEP, FIELDS, OPENING, decode_sleeve_ratios, encode_sleeve_ratios


class SleeveRatioTest(TestCase):
    def test_width_and_opening_follow_armhole_drop(self):
        spec = TeeSpec(flat_chest_mm=401, garment_shoulder_width_mm=355, shoulder_slope_mm=38, armhole_depth_mm=192,
                       sleeve_bicep_flat_mm=267, sleeve_opening_flat_mm=220, sleeve_length_mm=169)
        labels = np.array([[getattr(spec, field) for field in FIELDS]] * 2)
        encoded = encode_sleeve_ratios(labels)
        np.testing.assert_allclose(decode_sleeve_ratios(encoded), labels)
        self.assertLess(encoded[0, OPENING], 1)
        encoded[0, ARMHOLE] -= 26
        decoded = decode_sleeve_ratios(encoded)[0]
        self.assertLess(decoded[BICEP], labels[0, BICEP])
        self.assertAlmostEqual(decoded[OPENING] / decoded[BICEP], 220 / 267)
        draft(TeeSpec(**dict(zip(FIELDS, map(float, decoded), strict=True))))

    def test_invalid_opening_cannot_become_a_training_label(self):
        measurements = [getattr(TeeSpec(garment_shoulder_width_mm=490), field) for field in FIELDS]
        measurements[OPENING] = measurements[BICEP] + 1
        with self.assertRaisesRegex(ValueError, "cuff no wider"):
            encode_sleeve_ratios(measurements)
