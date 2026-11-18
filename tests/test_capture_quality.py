from math import cos, radians, sin
from unittest import TestCase

from PIL import Image, ImageDraw

from superarchiveallclo.vision import paired_features, picture_features


def photo(*, turn=0, card_height=40, body_width=140, circle=False, shirt_color=43):
    image = Image.new("RGB", (340, 335), (227, 229, 223))
    paint = ImageDraw.Draw(image)
    left, right = 185 - body_width / 2, 185 + body_width / 2
    paint.polygon((
        (left + 10, 62), (151, 55), (164, 55), (172, 69), (198, 69),
        (207, 55), (220, 55), (right - 10, 62), (right + 34, 106),
        (right + 20, 132), (right, 119), (right, 268), (left, 268),
        (left, 119), (left - 20, 132), (left - 34, 106),
    ), fill=(shirt_color, 61, 68))
    marker = (215, 39, 157)
    if circle:
        paint.ellipse((27, 218, 67, 258), fill=marker)
    else:
        angle = radians(turn)
        w, h = 20, card_height / 2
        paint.polygon([
            (47 + x * cos(angle) - y * sin(angle), 238 + x * sin(angle) + y * cos(angle))
            for x, y in ((-w, -h), (w, -h), (w, h), (-w, h))
        ], fill=marker)
    return image


class CaptureQualityTest(TestCase):
    def test_square_turned_on_table_preserves_scale(self):
        _, standard = picture_features(photo())
        _, turned = picture_features(photo(turn=27))
        self.assertLess(turned["card_aspect_ratio"], 1.05)
        ratio = turned["processed_mm_per_pixel"] / standard["processed_mm_per_pixel"]
        self.assertAlmostEqual(ratio, 1.0, delta=0.03)

    def test_distorted_or_non_square_marker_is_rejected(self):
        for frame in (photo(card_height=34), photo(card_height=29, turn=25), photo(circle=True)):
            with self.assertRaisesRegex(ValueError, "marker looks distorted"):
                picture_features(frame)

    def test_same_frame_cannot_be_used_for_both_views(self):
        front = photo()
        with self.assertRaisesRegex(ValueError, "two distinct photographs"):
            paired_features(front, front.copy())

    def test_larger_torso_is_rejected_even_at_same_length(self):
        with self.assertRaisesRegex(ValueError, "disagree on garment length or torso width"):
            paired_features(photo(), photo(body_width=173, shirt_color=58))

    def test_small_repositioning_of_the_garment_is_accepted(self):
        _, pair = paired_features(photo(), photo(body_width=145, turn=13, shirt_color=58))
        self.assertLess(pair["front"]["lower_torso_width_mm"], pair["back"]["lower_torso_width_mm"])
