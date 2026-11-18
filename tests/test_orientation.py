from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from superarchiveallclo.orientation import upright_angle


class TeeOrientationTest(TestCase):
    def test_resolves_both_rotation_and_collar_direction(self):
        pattern = Image.new("1", (180, 160))
        ImageDraw.Draw(pattern).polygon([
            (74, 10), (90, 22), (106, 10), (123, 18), (172, 39), (160, 65),
            (139, 54), (142, 148), (38, 148), (41, 54), (20, 65), (8, 39), (56, 18),
        ], fill=1)
        silhouette = np.asarray(pattern)
        for rotation in (-87, 23, 153):
            with self.subTest(rotation=rotation):
                observed = ndimage.rotate(silhouette, rotation, order=0)
                predicted, score = upright_angle(observed)
                self.assertLess(abs((predicted + rotation + 180) % 360 - 180), 3)
                self.assertGreater(score, 0.04)

    def test_does_not_infer_an_axis_from_a_folded_rectangle(self):
        with self.assertRaisesRegex(ValueError, "sleeve and hem"):
            upright_angle(np.ones((80, 120), dtype=bool))
