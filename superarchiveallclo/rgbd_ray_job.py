"""Blender job to compare an exact GLB surface with independent depth rays."""

from pathlib import Path
import sys

from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superarchiveallclo.gravity_pose import ray_depth_errors
from superarchiveallclo.read_garment_glb import mesh_geometry


def main():
    mesh, measurements, output = sys.argv[sys.argv.index("--") + 1:]
    vertices, triangles = mesh_geometry(mesh)
    surface = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    with np.load(measurements, allow_pickle=False) as input_data:
        errors = {name: ray_depth_errors(surface, input_data[name], input_data[f"camera-{name}"])
                  for name in input_data.files if not name.startswith("camera-")}
    np.savez_compressed(output, **errors)


if __name__ == "__main__":
    main()
