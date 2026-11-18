import unittest

import numpy as np

from superarchiveallclo.appearance import bake_body, photo_coordinates


PARAMETERS = {
    "armhole_depth_mm": 250, "back_length_mm": 625, "back_neck_depth_mm": 25,
    "flat_chest_mm": 500, "flat_hem_mm": 580,
}
PROFILE = {
    "chest_points_crop_px": [[35, 40], [85, 40]],
    "hem_points_crop_px": [[31, 90], [89, 90]],
}


class AppearanceTests(unittest.TestCase):
    def test_mapping_uses_both_widths_and_reverses_rear_camera(self):
        paper_x = np.array([-250, 250, -290, 290])
        paper_y = np.array([250, 250, 650, 650])
        front_x, front_y = photo_coordinates(paper_x, paper_y, PARAMETERS, PROFILE, 10, "front")
        back_x, back_y = photo_coordinates(paper_x, paper_y, PARAMETERS, PROFILE, 10, "back")
        np.testing.assert_allclose(front_x, [35, 85, 31, 89])
        np.testing.assert_allclose(back_x, [85, 35, 89, 31])
        np.testing.assert_allclose(front_y, [40, 40, 90, 90])
        np.testing.assert_allclose(front_y, back_y)

    def test_bake_preserves_print_and_rejects_background(self):
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[8:92, 30:92] = True
        fabric = np.array([205, 183, 95], dtype=np.uint8)
        print_color = np.array([30, 50, 92], dtype=np.uint8)
        rgb[mask] = fabric
        rgb[35:53, 55:67] = print_color
        tile, evidence = bake_body(
            rgb, mask, 10, (-450, 0, 450, 650), (121, 99), PROFILE, fabric, "front", PARAMETERS,
        )
        pixels = np.asarray(tile)
        self.assertTrue(np.any(np.all(pixels == print_color, axis=-1)))
        self.assertFalse(np.any(np.all(pixels == 0, axis=-1)))
        self.assertEqual(evidence["mapping"], "shoulder_underarm_hem")

        tile, evidence = bake_body(
            rgb, mask, 10, (-450, 0, 450, 650), (121, 99), None, fabric, "front", PARAMETERS,
        )
        self.assertEqual(tile.getpixel((60, 50)), tuple(fabric))
        self.assertEqual(evidence["mapping"], "observed_solid_color")


if __name__ == "__main__":
    unittest.main()
