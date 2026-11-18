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
from superarchiveallclo.mesh_relax import relax_flat_metric
from superarchiveallclo.simulate import build_assembly, sewn_mesh, triangle_edges

FRONT_LEVEL_M = 0.002
SLEEVE_UNDERARM_WEIGHT = 2.0


def body_point(name, x, y):
    return Vector((x / 1000, -y / 1000, FRONT_LEVEL_M if name == "front" else 0))


def body_border(pattern, name, edge, fraction):
    x, y = edge.point_at(fraction)[0]
    point = body_point(name, x, y)
    if edge.id.startswith(("shoulder", "side")):
        point.z = FRONT_LEVEL_M / 2
    return point


def place_flat(pattern, name, edges, point, angle, sleeve_fold=True):
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
    p = pattern["parameters"]
    along = Vector((side * cos(angle), -sin(angle), 0))
    across = Vector((-side * sin(angle), -cos(angle), 0))
    upper = side * x >= 0
    layer = "front" if upper else "back"
    cap = next(edge for edge in edges if edge.id == "cap_front")
    cap_height = max(cap.start[1], cap.end[1])
    underarm = body_point(layer, side * p["flat_chest_mm"] / 2, p["armhole_depth_mm"])
    shoulder_point = body_point(layer, side * shoulder, p["shoulder_slope_mm"])
    origin = underarm - (along * cap_height + across * p["sleeve_bicep_flat_mm"] / 2) / 1000
    # Place the lower sleeve from its underarm seam. Fade the corresponding
    # shoulder correction through the cap so the cuff keeps its metric run.
    shoulder_offset = (shoulder_point - origin) * max(0, 1 - y / cap_height)
    point = origin + (along * y + across * abs(x)) / 1000 + shoulder_offset
    if sleeve_fold and abs(x) < 0.001:
        point.z = FRONT_LEVEL_M / 2
    return point


def main():
    filename, runtime, angle, solver = sys.argv[sys.argv.index("--") + 1:]
    target = Path(runtime)
    pattern = json.loads(Path(filename).read_bytes())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    sleeve_fold = solver == "relaxed"
    position = partial(place_flat, angle=radians(float(angle)), sleeve_fold=sleeve_fold)
    seam_edges = {}

    def relax(points, edges, lengths, pins, faces):
        weights = np.ones(len(edges))
        for name, (a_edges, b_edges) in seam_edges.items():
            if name.startswith("sleeve_underarm_"):
                weights[a_edges + b_edges] = SLEEVE_UNDERARM_WEIGHT
        relax_flat_metric(points, edges, lengths, pins, faces, weights)

    points, welded, faces, uv, _, edges, lengths, panels = build_assembly(
        pattern, position, body_border, lock_height=True, sleeve_fold=sleeve_fold,
        relax=relax if sleeve_fold else None, seam_edge_indices=seam_edges,
    )
    model, openings = sewn_mesh(points, welded, faces, uv, panels)
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    mesh_path = target / "flat-pose.glb"
    bpy.ops.export_scene.gltf(filepath=str(mesh_path), use_selection=True, export_apply=True)

    vertices, triangles = mesh_geometry(mesh_path)
    surface = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    actual = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    strain = np.abs(actual / lengths - 1)
    results = {"vertices": vertices, "triangles": triangles, "openings": openings,
               "sleeve_underarm_solver_weight": SLEEVE_UNDERARM_WEIGHT if sleeve_fold else 1.0,
               "mesh_edge_strain_quantiles_percent": np.quantile(strain, [0.5, 0.95, 0.99]) * 100}
    source_edges = triangle_edges(faces)
    for name, panel in panels.items():
        start = panel["vertex_start"]
        belongs = (source_edges[:, 0] >= start) & (source_edges[:, 0] < start + panel["vertex_count"])
        results[f"{name}_strain_quantiles_percent"] = np.quantile(strain[belongs], [0.5, 0.95, 0.99]) * 100
        if name.startswith("sleeve_"):
            x = np.asarray(uv)[source_edges[belongs], 0]
            results[f"{name}_edges_crossing_fold"] = ((x.min(axis=1) < -1e-6) & (x.max(axis=1) > 1e-6)).sum()
    results["seam_strain_json"] = json.dumps({
        seam["id"]: {
            key: {
                "pattern_edge": seam[key],
                "rest_mm": round(float(lengths[indices].sum() * 1000), 2),
                "posed_mm": round(float(actual[indices].sum() * 1000), 2),
                "edge_strain_p95_percent": round(float(np.quantile(strain[indices], 0.95) * 100), 2),
            }
            for key, indices in zip(("a", "b"), seam_edges[seam["id"]], strict=True)
        }
        for seam in pattern["seams"]
    })
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
