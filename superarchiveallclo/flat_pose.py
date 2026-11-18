"""Blender entry point for a sewn tee in a supported flat-lay pose."""

from functools import partial
import json
from math import cos, radians, sin
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superarchiveallclo.pattern import Edge
from superarchiveallclo.read_garment_glb import mesh_geometry
from superarchiveallclo.simulate import build_assembly, sewn_mesh

FRONT_LEVEL_M = 0.002


def body_point(name, x, y):
    return Vector((x / 1000, -y / 1000, FRONT_LEVEL_M if name == "front" else 0))


def body_border(pattern, name, edge, fraction):
    x, y = edge.point_at(fraction)[0]
    point = body_point(name, x, y)
    if edge.id.startswith(("shoulder", "side")):
        point.z = FRONT_LEVEL_M / 2
    return point


def place_flat(pattern, name, edges, point, angle):
    x, y = point
    if name in ("front", "back"):
        return body_point(name, x, y)
    if name == "collar":
        attachments = [edge for edge in edges if edge.id.startswith("attach_")]
        edge = next((entry for entry in attachments if x <= entry.end[0]), attachments[-1])
        fraction = min(1, max(0, (x - edge.start[0]) / edge.length()))
        seam = next(record for record in pattern["seams"] if record["id"] == edge.id)
        body_name, edge_id = seam["a"]
        panel = next(record for record in pattern["panels"] if record["id"] == body_name)
        neckline = Edge(**next(record for record in panel["edges"] if record["id"] == edge_id))
        anchor = body_border(pattern, body_name, neckline, 1 - fraction if seam["reversed"] else fraction)
        toward_neck = Vector((-anchor.x, -anchor.y, 0)).normalized()
        return anchor + toward_neck * (y / 1000)

    side = 1 if name == "sleeve_right" else -1
    shoulder = pattern["measurements"]["across_shoulders_mm"] / 2
    drop = pattern["parameters"]["shoulder_slope_mm"]
    along = Vector((side * cos(angle), -sin(angle), 0))
    across = Vector((-side * sin(angle), -cos(angle), 0))
    anchor = body_point("front" if side * x >= 0 else "back", side * shoulder, drop)
    return anchor + (along * y + across * abs(x)) / 1000


def main():
    filename, runtime, angle = sys.argv[sys.argv.index("--") + 1:]
    target = Path(runtime)
    pattern = json.loads(Path(filename).read_bytes())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    position = partial(place_flat, angle=radians(float(angle)))
    points, welded, faces, uv, _, edges, lengths, panels = build_assembly(
        pattern, position, body_border, lock_height=True,
    )
    model, openings = sewn_mesh(points, welded, faces, uv, panels)
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    mesh_path = target / "flat-pose.glb"
    bpy.ops.export_scene.gltf(filepath=str(mesh_path), use_selection=True, export_apply=True)

    vertices, triangles = mesh_geometry(mesh_path)
    surface = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    strain = np.abs(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1) / lengths - 1)
    results = {"vertices": vertices, "triangles": triangles, "openings": openings,
               "mesh_edge_strain_p95_percent": np.quantile(strain, 0.95) * 100}
    with np.load(target / "reference-points.npz", allow_pickle=False) as refs:
        for key in refs.files:
            cloud = refs[key]
            # Align the crop's upper edge and transverse midpoint once. The median
            # measured height provides a translation datum for the thin upper layer.
            low, high = np.quantile(cloud[:, :2], [0.001, 0.999], axis=0)
            shift = np.array([
                (vertices[:, 0].min() + vertices[:, 0].max() - low[0] - high[0]) / 2,
                vertices[:, 1].max() - high[1],
                FRONT_LEVEL_M - np.median(cloud[:, 2]),
            ])
            distances = np.array([surface.find_nearest(point.tolist())[3] for point in cloud + shift]) * 1000
            results[f"{key}_distance_mm"] = distances
            results[f"{key}_translation_m"] = shift
    np.savez_compressed(target / "metrics.npz", **results)


if __name__ == "__main__":
    main()
