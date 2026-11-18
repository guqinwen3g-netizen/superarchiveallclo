from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from superarchiveallclo.compare_shape import GRID_MM, compare_masks, project, reference_mask


class ShapeComparisonTest(TestCase):
    def test_identical_silhouettes(self):
        target = np.ones((110, 80), dtype=bool)
        result, _ = compare_masks(target, target)
        self.assertEqual(result["intersection_over_union"], 1)
        self.assertEqual(result["contour_distance_mm"]["p90"], 0)

    def test_scale_error_survives_translation_only_alignment(self):
        target = np.ones((100, 100), dtype=bool)
        estimate = np.ones((120, 120), dtype=bool)
        result, overlay = compare_masks(target, estimate)
        self.assertAlmostEqual(result["intersection_over_union"], 1 / 1.2 ** 2, places=3)
        self.assertGreater(result["contour_distance_mm"]["p90"], 10)
        self.assertTrue(((overlay == (205, 86, 41)).all(axis=2)).any())

    def test_projected_garment_preserves_metric_spans(self):
        vertices = np.array([[x, y, z] for x in (0, 0.4) for y in (0, 0.12) for z in (0, 0.6)])
        triangles = np.array([(a, b, c) for a, b, c in (
            (0, 2, 4), (2, 4, 6), (1, 3, 5), (3, 5, 7), (0, 1, 2), (1, 2, 3),
            (4, 5, 6), (5, 6, 7), (0, 1, 4), (1, 4, 5), (2, 3, 6), (3, 6, 7),
        )])
        _, front = project(vertices, triangles, "front")
        _, left = project(vertices, triangles, "left")
        np.testing.assert_allclose(front, [400, 600])
        np.testing.assert_allclose(left, [120, 600])

    def test_reference_scale_and_cropped_mask(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "mask.png"
            image = np.zeros((202, 102), dtype=np.uint8)
            image[1:-1, 1:-1] = 255
            Image.fromarray(image).save(target)
            raster, span, _ = reference_mask(target, GRID_MM * 0.5)
            self.assertEqual(raster.shape, (100, 50))
            np.testing.assert_allclose(span, [100, 200])
            image[0, 5] = 255
            Image.fromarray(image).save(target)
            with self.assertRaisesRegex(ValueError, "black margin"):
                reference_mask(target, 1)
