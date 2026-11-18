from unittest import TestCase

import numpy as np

from superarchiveallclo.reference_form import arm_angle, display_neck, initial_sleeve_angle, orient_display_neck, upper_arm


class UpperArmSupportTest(TestCase):
    def test_rotation_keeps_arm_extent_and_proximal_pivot(self):
        for side in (-1, 1):
            base, base_extent, base_vector = upper_arm(side, 1.16, 1.05, 1.52, 45.0)
            base_unit = np.asarray(base_vector) / np.linalg.norm(base_vector)
            base_pivot = np.array(base) - 0.7 * base_extent[2] * base_unit
            for angle in (0.0, 16.0, 90.0):
                center, extent, direction = upper_arm(side, 1.16, 1.05, 1.52, angle)
                self.assertEqual(base_extent, extent)
                np.testing.assert_allclose(np.array(center) - 0.7 * extent[2] * np.array(direction), base_pivot)

    def test_angle_validation_preserves_legacy_and_arms_free_forms(self):
        self.assertEqual(arm_angle("torso_with_arms", None), 45.0)
        self.assertIsNone(arm_angle("display_torso", None))
        with self.assertRaisesRegex(ValueError, "requires the torso_with_arms"):
            arm_angle("display_torso", 16)
        for invalid in (-1, 91, float("inf"), True):
            with self.assertRaisesRegex(ValueError, "between 0 and 90"):
                arm_angle("torso_with_arms", invalid)


class DisplayNeckSupportTest(TestCase):
    def test_forward_pitch_preserves_top_datum_and_lower_torso(self):
        top = 1.62
        self.assertEqual(orient_display_neck((0, 0, top), top, 22), (0, 0, top))
        self.assertEqual(orient_display_neck((.1, -.1, 1.35), top, 22), (.1, -.1, 1.35))
        x, y, z = orient_display_neck((.1, 0, 1.56), top, 22)
        self.assertEqual(x, .1)
        self.assertGreater(y, 0)
        self.assertGreater(z, 1.56)

    def test_display_profile_requires_observed_provenance_and_units(self):
        profile = {"schema_version": 1, "pitch_forward_deg": 22, "lateral_width_mm": 101,
                   "source": "Opposing calibrated stand photographs"}
        self.assertEqual(display_neck(profile), profile)
        for invalid in ({"pitch_forward_deg": float("nan")}, {"lateral_width_mm": .101}, {"source": " "}):
            with self.assertRaises(ValueError):
                display_neck({**profile, **invalid})

    def test_vertical_initialization_is_explicit_and_restricted_to_the_display_form(self):
        self.assertEqual(initial_sleeve_angle("torso_only", None), 45)
        self.assertEqual(initial_sleeve_angle("display_torso", None), 20)
        self.assertEqual(initial_sleeve_angle("display_torso", 0), 0)
        with self.assertRaisesRegex(ValueError, "requires the display_torso"):
            initial_sleeve_angle("torso_with_arms", 0)
