"""Read the exact exported mesh with Blender's glTF importer."""

from pathlib import Path
import sys

import bpy
import numpy as np


def mesh_geometry(source):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(source))
    vertices, triangles = [], []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        mesh.calc_loop_triangles()
        start = len(vertices)
        vertices.extend(tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices)
        triangles.extend(tuple(start + i for i in face.vertices) for face in mesh.loop_triangles)
    if not vertices or not triangles:
        raise ValueError("Garment GLB contains no triangle mesh")
    return np.asarray(vertices), np.asarray(triangles)


def main():
    source, target = map(Path, sys.argv[sys.argv.index("--") + 1:])
    vertices, triangles = mesh_geometry(source)
    np.savez_compressed(target, vertices=vertices, triangles=triangles)


if __name__ == "__main__":
    main()
