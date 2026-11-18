from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from superarchiveallclo.mesh_openings import boundary_loops
from superarchiveallclo.scan_openings import measure_scan


class MeshOpeningsTest(TestCase):
    def test_trace_a_free_loop_without_counting_the_interior_diagonal(self):
        self.assertEqual(boundary_loops([(0, 1, 2), (0, 2, 3)]), [[0, 1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "more than two triangles"):
            boundary_loops([(0, 1, 2), (0, 1, 3), (0, 1, 4)])

    def test_measure_registered_parts_and_explicit_units(self):
        # A connected mesh with three holes supplies four known boundary lengths.
        size = 10
        xyz = np.array([(x / 1000, y / 1000, 0) for y in range(size) for x in range(size)])
        holes = {(2, 2): "collar", (6, 3): "sleeves", (5, 6): "sleeves"}
        triangles = []
        labels = {"collar": [], "sleeves": [], "body": []}
        for y in range(size):
            for x in range(size):
                if x in (0, size - 1) or y in (0, size - 1):
                    labels["body"].append(y * size + x)
        for y in range(size - 1):
            for x in range(size - 1):
                a = y * size + x
                b, c, d = a + 1, a + size + 1, a + size
                if (x, y) in holes:
                    labels[holes[x, y]].extend((a, b, c, d))
                else:
                    triangles.extend(((a, b, c), (a, c, d)))
        with TemporaryDirectory() as temp:
            p = Path(temp)
            obj = p / "scan.obj"
            obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in xyz)
                           + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in triangles))
            parts = {name: p / f"{name}.asc" for name in labels}
            for name, path in parts.items():
                np.savetxt(path, xyz[labels[name]])
            report = measure_scan(obj, parts, unit="m")
            self.assertEqual(report["reported_unit"], "mm")
            self.assertEqual([(o["part"], o["perimeter"]) for o in report["openings"]],
                             [("cuff", 4), ("cuff", 4), ("hem", 36), ("neck", 4)])
            np.savetxt(parts["collar"], xyz[labels["collar"]] + 1)
            with self.assertRaisesRegex(ValueError, "collar: part points do not align"):
                measure_scan(obj, parts)
