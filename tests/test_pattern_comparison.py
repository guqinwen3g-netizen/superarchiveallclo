from unittest import TestCase

import numpy as np

from superarchiveallclo.compare_pattern import chart_faces, read_obj, scale_from_surface, shared_pairs


class UVReferenceTest(TestCase):
    def test_split_uvs_retain_a_welded_panel_connection(self):
        vertices, uv, faces = read_obj(b"""v 0 0 0
v 0.4 0 0
v 0.4 0.2 0
v 0 0.2 0
vt 0 0
vt 1 0
vt 1 0.5
vt 2 1
vt 3 1.5
vt 2 1.5
f 1/1 2/2 3/3
f 1/4 3/5 4/6
""")
        first, second = chart_faces(faces, len(uv))
        self.assertEqual(shared_pairs({"front": first, "back": second}), {("back", "front")})
        scale = scale_from_surface(vertices, uv, [np.tile(first, (7, 1, 1)), np.tile(second, (7, 1, 1))])
        np.testing.assert_allclose(scale, [400, 400, 400])
