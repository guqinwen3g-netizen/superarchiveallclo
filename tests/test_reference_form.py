from unittest import TestCase

import numpy as np

from superarchiveallclo.reference_form import arm_angle, upper_arm


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
