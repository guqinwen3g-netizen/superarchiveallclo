"""Blender entry point for sewing and draping a captured tee on a reference form."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from functools import partial
from hashlib import sha256
import json
from math import ceil, cos, floor, pi, radians, sin, sqrt
from pathlib import Path
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import delaunay_2d_cdt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superarchiveallclo.pattern import Edge

STEP_MM = 13
NECK_Z = 1.52
FRAME = 38


def panel_edges(panel):
    edges = [Edge(**edge) for edge in panel["edges"]]
    if panel["id"] != "collar":
        return edges

    # Simulate the folded neckband as a single effective layer.
    attachments = [edge for edge in edges if edge.id.startswith("attach_")]
    width = attachments[-1].end[0]
    height = panel["foldline"][0][1]
    return [
        *attachments, Edge("join_right", (width, 0), (width, height)),
        Edge("fold", (width, height), (0, height), role="fold"),
        Edge("join_left", (0, height), (0, 0)),
    ]


def flat_mesh(edges, steps, *, sleeve_fold=False):
    perimeter = []
    refs = {}
    for edge in edges:
        count = steps[edge.id]
        points = [edge.point_at(i / count)[0] for i in range(count)]
        if sleeve_fold and edge.role == "hem":
            middle = (0.0, edge.start[1])
            index = next(i for i, point in enumerate([*points, edge.end]) if point[0] * edge.start[0] <= 0)
            if 0 < index < len(points) and abs(points[index][0]) < 0.001:
                points[index] = middle
            else:
                points.insert(index, middle)
        refs[edge.id] = list(range(len(perimeter), len(perimeter) + len(points) + 1))
        perimeter.extend(points)
    for indices in refs.values():
        indices[-1] %= len(perimeter)
    x, y = zip(*perimeter)
    row_height = STEP_MM * sqrt(3) / 2
    candidates = np.array([
        ((i + 0.5 * (j % 2)) * STEP_MM, j * row_height)
        for j in range(floor(min(y) / row_height), ceil(max(y) / row_height) + 1)
        for i in range(floor(min(x) / STEP_MM) - 1, ceil(max(x) / STEP_MM) + 1)
    ])
    boundary = np.array(perimeter)
    border = np.roll(boundary, -1, axis=0) - boundary
    squared_lengths = (border ** 2).sum(axis=1)
    interior = []
    for start in range(0, len(candidates), 256):
        batch = candidates[start:start + 256]
        delta = batch[:, None, :] - boundary[None, :, :]
        projection = (delta * border).sum(axis=2) / squared_lengths
        separation = delta - np.clip(projection, 0, 1)[..., None] * border
        inside = (separation ** 2).sum(axis=2).min(axis=1) > (STEP_MM * 0.38) ** 2
        interior.extend(batch[inside].tolist())
    constraints = []
    if sleeve_fold:
        seam = [i for i, (x, _) in enumerate(perimeter) if abs(x) < 0.001]
        apex = min(seam, key=lambda i: perimeter[i][1])
        cuff = max(seam, key=lambda i: perimeter[i][1])
        top, bottom = perimeter[apex][1], perimeter[cuff][1]
        interior = [point for point in interior if abs(point[0]) >= 0.001]
        fold_y = np.linspace(top, bottom, ceil((bottom - top) / STEP_MM) + 1)[1:-1]
        start = len(perimeter) + len(interior)
        fold = [apex, *range(start, start + len(fold_y)), cuff]
        interior.extend((0.0, y) for y in fold_y)
        constraints = list(zip(fold, fold[1:]))
    verts, _, faces, origins, _, _ = delaunay_2d_cdt(
        [Vector(p) for p in perimeter + interior], constraints, [list(range(len(perimeter)))], 1, 0.001,
    )
    retained = sorted({i for face in faces for i in face})
    compact = {i: j for j, i in enumerate(retained)}
    originals = {i: compact[j] for j in retained for i in origins[j]}
    return (
        [verts[i] for i in retained], [tuple(compact[i] for i in face) for face in faces],
        {name: [originals[i] for i in indices] for name, indices in refs.items()},
    )


def body_half_width_at(parameters, height):
    chest = parameters["flat_chest_mm"]
    hem = parameters.get("flat_hem_mm", chest)
    underarm = parameters["armhole_depth_mm"]
    length = parameters["back_length_mm"] + parameters["back_neck_depth_mm"]
    fraction = max(0, min(1, (height - underarm) / (length - underarm)))
    return (chest + fraction * (hem - chest)) / 2


def place(pattern, panel, edges, point, *, sleeve_angle_from_vertical=45):
    x, y = point
    p = pattern["parameters"]
    half_width = p["flat_chest_mm"] / 2
    rx, ry = half_width * 0.69 / 1000, half_width * 0.52 / 1000
    if panel in ("front", "back"):
        local_width = body_half_width_at(p, y)
        projected_x = min(local_width, max(-local_width, x))
        angle = projected_x / local_width * pi / 2
        radius_ratio = local_width / half_width
        sign = -1 if panel == "front" else 1
        return Vector((rx * radius_ratio * sin(angle) + (x - projected_x) / 1000,
                       sign * (ry * radius_ratio * cos(angle) + 0.002),
                       NECK_Z - y / 1000))
    if panel == "collar":
        band = [edge for edge in edges if edge.id.startswith("attach_")]
        edge = next((edge for edge in band if x <= edge.end[0]), band[-1])
        t = min(1, max(0, (x - edge.start[0]) / edge.length()))
        seam = next(seam for seam in pattern["seams"] if seam["id"] == edge.id)
        name, neck_id = seam["a"]
        body = next(piece for piece in pattern["panels"] if piece["id"] == name)
        neck = Edge(**next(item for item in body["edges"] if item["id"] == neck_id))
        anchor = body_border(pattern, name, neck, 1 - t if seam["reversed"] else t)
        inward = Vector((-anchor.x, -anchor.y, 0)).normalized()
        fold = inward * 0.85 + Vector((0, 0, 0.53))
        return anchor + fold * ((y + 1.5) / 1000)

    by_id = {edge.id: edge for edge in edges}
    side = 1 if panel == "sleeve_right" else -1
    cap = by_id["cap_front"]
    point_at_underarm = max((cap.start, cap.end), key=lambda xy: xy[1])
    cap_depth = point_at_underarm[1]
    front_span = abs(point_at_underarm[0])
    back_span = p["sleeve_bicep_flat_mm"] - front_span
    sleeve_length = p["sleeve_length_mm"]
    taper = (p["sleeve_bicep_flat_mm"] - p["sleeve_opening_flat_mm"]) / 2
    taper *= max(0, y - cap_depth) / (sleeve_length - cap_depth)
    front_span, back_span = front_span - taper, back_span - taper
    transverse = side * x
    angle = pi * transverse / (front_span if transverse >= 0 else back_span)
    radius = (front_span + back_span) / (2 * pi * 1000)
    normal = Vector((side, 0, 1)).normalized()
    direction = Vector((side, 0, -1)).normalized()
    shoulder_ratio = pattern["measurements"]["across_shoulders_mm"] / (2 * half_width)
    shoulder_x = rx * sin(min(shoulder_ratio, 1) * pi / 2) + max(0, shoulder_ratio - 1) * half_width / 1000
    shoulder = Vector((side * shoulder_x, 0, NECK_Z - p["shoulder_slope_mm"] / 1000))
    posed = (
        shoulder - normal * radius + direction * (y / 1000)
        + normal * (radius * cos(angle)) + Vector((0, -radius * sin(angle), 0))
    )
    if sleeve_angle_from_vertical == 45:
        return posed
    rotation = radians(side * (45 - sleeve_angle_from_vertical))
    x, y, z = posed - shoulder
    return shoulder + Vector((cos(rotation) * x + sin(rotation) * z, y,
                              -sin(rotation) * x + cos(rotation) * z))


def body_border(pattern, name, edge, fraction):
    """Place the body seam curve onto the neutral form before fitting its metric."""
    p = pattern["parameters"]
    x, y = edge.point_at(fraction)[0]
    sign = -1 if name == "front" else 1
    half_width = p["flat_chest_mm"] / 2000
    neck = p["neck_width_mm"] / 2000
    shoulder = pattern["measurements"]["across_shoulders_mm"] / 2000
    side = -1 if "_left" in edge.id else 1
    x_shoulder = shoulder * 0.87
    x_side = half_width * 0.72
    if edge.id.startswith("neck"):
        phase = min(1, abs(x) / (neck * 1000))
        depth = (0.092 if name == "front" else 0.069) * half_width / 0.28
        return Vector((x * 0.93 / 1000, sign * depth * (1 - phase ** 2) ** 0.65, NECK_Z - y / 1000))
    if edge.id.startswith("shoulder"):
        along = (abs(x) / 1000 - neck) / (shoulder - neck)
        return Vector((side * (0.93 * neck + along * (x_shoulder - 0.93 * neck)), 0, NECK_Z - y / 1000))
    if edge.id.startswith("armhole"):
        along = 1 - fraction if side == -1 else fraction
        top = p["shoulder_slope_mm"] / 1000
        drop = (p["armhole_depth_mm"] - p["shoulder_slope_mm"]) / 1000
        depth = (0.345 if name == "front" else 0.29) * drop
        return Vector((
            side * (x_shoulder + along * (x_side - x_shoulder)),
            sign * depth * sin(pi * along),
            NECK_Z - top - 0.84 * drop * along,
        ))
    hem = p["back_length_mm"] + p["back_neck_depth_mm"]
    recovery = (hem - y) / (hem - p["armhole_depth_mm"])
    lift = 0.00016 * (p["armhole_depth_mm"] - p["shoulder_slope_mm"])
    local_x_side = body_half_width_at(p, y) / 1000 * 0.72
    return Vector((side * local_x_side, 0, NECK_Z - y / 1000 + recovery * lift))


def weighted_sum(values, indices, weights, n):
    return np.column_stack([
        np.bincount(indices, weights=values[:, axis] * weights, minlength=n)
        for axis in range(3)
    ])


def spread_border_adjustments(positions, rest, faces, anchors):
    """Extend seam displacement into each panel, retaining its initial curvature."""
    original = np.asarray(positions)
    neighbors = sorted({(a, b) for f in faces for a in f for b in f if a != b})
    a, b = np.array(neighbors).T
    rest = np.asarray(rest)
    weights = 1 / np.maximum(np.linalg.norm(rest[a] - rest[b], axis=1), 3)
    n = len(rest)
    scale = np.bincount(a, weights=weights, minlength=n) * 1.018
    fixed = np.array(list(anchors))
    delta = np.zeros_like(original)
    boundary = np.asarray(list(anchors.values())) - original[fixed]
    for _ in range(110):
        delta = weighted_sum(delta[b], a, weights, n) / scale[:, None]
        delta[fixed] = boundary
    return original + delta


def triangle_edges(faces):
    return np.array(sorted({tuple(sorted((a, b))) for f in faces for a, b in zip(f, (*f[1:], f[0]))}))


def build_assembly(pattern, place_vertex=place, place_body_border=body_border, *, lock_height=False,
                   sleeve_fold=False, relax=None, seam_edge_indices=None):
    edges = {panel["id"]: panel_edges(panel) for panel in pattern["panels"]}
    steps = {
        (name, edge.id): max(1, ceil(edge.length() / STEP_MM))
        for name, panel in edges.items() for edge in panel
    }
    for seam in pattern["seams"]:
        a, b = tuple(seam["a"]), tuple(seam["b"])
        steps[a] = steps[b] = max(steps[a], steps[b])

    positions, faces, refs, uv = [], [], {}, []
    panels = {}
    candidates = defaultdict(list)
    shoulder_points = set()
    for name, panel in edges.items():
        verts, triangles, local_refs = flat_mesh(panel, {edge.id: steps[name, edge.id] for edge in panel},
                                                 sleeve_fold=sleeve_fold and name.startswith("sleeve_"))
        offset = len(positions)
        positions.extend(place_vertex(pattern, name, panel, v) for v in verts)
        uv.extend(tuple(v / 1000) for v in verts)
        faces.extend(tuple(i + offset for i in face) for face in triangles)
        refs.update({(name, edge): [i + offset for i in values] for edge, values in local_refs.items()})
        panels[name] = {"vertex_start": offset, "vertex_count": len(verts)}
        if name in ("front", "back"):
            for edge in panel:
                if edge.role == "hem":
                    continue
                indices = refs[name, edge.id]
                for i, index in enumerate(indices):
                    fraction = i / (len(indices) - 1)
                    candidates[index].append(place_body_border(pattern, name, edge, fraction))
                    if edge.id.startswith("shoulder") and 0.18 < fraction < 0.82:
                        shoulder_points.add(index)

    parent = list(range(len(positions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    stitches = set()
    for seam in pattern["seams"]:
        a, b = refs[tuple(seam["a"])], refs[tuple(seam["b"])]
        for i, j in zip(a, reversed(b) if seam["reversed"] else b):
            parent[root(j)] = root(i)
            stitches.update((i, j))

    groups = defaultdict(list)
    for i in range(len(positions)):
        groups[root(i)].append(i)
    anchors = {}
    for group in groups.values():
        if group[0] not in stitches:
            continue
        targets = [p for i in group for p in candidates[i]]
        if not targets:
            targets = [positions[i] for i in group]
        target = sum(targets, Vector()) / len(targets)
        anchors.update(dict.fromkeys(group, target))

    placed = spread_border_adjustments(positions, np.asarray(uv) * 1000, faces, anchors)
    mapping = {key: i for i, key in enumerate(groups)}
    transform = np.asarray([mapping[root(i)] for i in range(len(positions))])
    vertices = np.array([placed[group].mean(axis=0) for group in groups.values()])
    welded_faces = [tuple(transform[i] for i in f) for f in faces]
    pins = sorted({int(transform[i]) for i in shoulder_points})
    source_edges = triangle_edges(faces)
    basis = np.asarray(uv)
    target_lengths = np.linalg.norm(basis[source_edges[:, 0]] - basis[source_edges[:, 1]], axis=1)
    if seam_edge_indices is not None:
        edge_lookup = {tuple(indices): i for i, indices in enumerate(source_edges)}
        for seam in pattern["seams"]:
            groups = []
            for key in ("a", "b"):
                curve = refs[tuple(seam[key])]
                groups.append([edge_lookup[tuple(sorted((a, b)))] for a, b in zip(curve, curve[1:])])
            seam_edge_indices[seam["id"]] = groups
    if relax is None:
        fit_metric(vertices, transform[source_edges], target_lengths, pins, lock_height=lock_height)
    else:
        relax(vertices, transform[source_edges], target_lengths, pins, welded_faces)
    return vertices, welded_faces, faces, uv, pins, transform[source_edges], target_lengths, panels


def fit_metric(vertices, edges, target_lengths, pins, *, lock_height=False):
    """Fit stitched rest geometry to the flat edge lengths without removing ease."""
    a, b = edges.T
    n = len(vertices)
    degree = np.bincount(np.r_[a, b], minlength=n)
    original = vertices.copy()
    weight = np.ones(len(edges))
    for _ in range(200):
        along = vertices[b] - vertices[a]
        length = np.linalg.norm(along, axis=1)
        correction = along * ((length - target_lengths) / np.maximum(length, 1e-9))[:, None]
        if lock_height:
            correction[:, 2] = 0
        gradient = weighted_sum(correction, a, weight, n) - weighted_sum(correction, b, weight, n)
        vertices += 0.68 * gradient / degree[:, None]
        vertices += 0.008 * (original - vertices)
        vertices[pins] = original[pins]


def configure_cloth(model, pins, *, frame_end=FRAME):
    group = model.vertex_groups.new(name="Mannequin shoulder support")
    if len(pins):
        group.add(pins, 1, "REPLACE")
    cloth = model.modifiers.new("Cloth drape", "CLOTH")
    cloth.settings.quality = 10
    cloth.settings.mass = 0.15
    cloth.settings.tension_stiffness = 32
    cloth.settings.compression_stiffness = 32
    cloth.settings.shear_stiffness = 19
    cloth.settings.bending_stiffness = 1.45
    cloth.settings.bending_damping = 0.8
    cloth.settings.vertex_group_mass = group.name
    cloth.settings.pin_stiffness = 1
    cloth.settings.effector_weights.gravity = 0.62
    cloth.collision_settings.distance_min = 0.0018
    cloth.collision_settings.use_self_collision = True
    cloth.collision_settings.self_distance_min = 0.0015
    cloth.point_cache.frame_end = frame_end


def evaluated_positions(garment):
    evaluated = garment.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return np.asarray([vertex.co[:] for vertex in evaluated.data.vertices])


def displacement_summary(earlier, later, start_frame, end_frame):
    displacement = 1000 * np.linalg.norm(later - earlier, axis=1)
    return {
        "frames": [start_frame, end_frame],
        "median": round(float(np.median(displacement)), 2),
        "p90": round(float(np.quantile(displacement, 0.9)), 2),
    }


def collider(name, location, scale, direction=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    form = bpy.context.object
    form.name = name
    form.scale = scale
    if direction:
        form.rotation_euler = Vector(direction).to_track_quat("Z", "Y").to_euler()
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    form.modifiers.new("Cloth collision", "COLLISION")
    form.collision.thickness_outer = 0.002
    form.hide_render = True
    return form


def add_form(pattern, *, with_arms=True):
    flat_chest = pattern["measurements"]["flat_chest_mm"] / 1000
    length = pattern["measurements"]["back_center_length_mm"] / 1000
    # Neck, shoulders, chest, waist and hip heights on a neutral fitting form.
    rings = (
        (0.10, 0.060, 0.053), (0.019, 0.061, 0.054), (-0.016, 0.122, 0.066),
        (-0.062, 0.178, 0.085), (-0.13, 0.167, 0.103), (-0.23, 0.158, 0.115),
        (-0.39, 0.144, 0.106), (-0.52, 0.141, 0.103), (-0.64, 0.166, 0.112),
        (-0.76, 0.172, 0.111), (-0.91, 0.170, 0.107),
    )
    segments = 48
    scale_x = flat_chest / 0.56
    vertices = [
        (rx * scale_x * cos(2 * pi * i / segments), ry * scale_x * sin(2 * pi * i / segments),
         NECK_Z + level * length / 0.68)
        for level, rx, ry in rings for i in range(segments)
    ]
    faces = [
        (j * segments + i, (j + 1) * segments + i,
         (j + 1) * segments + (i + 1) % segments, j * segments + (i + 1) % segments)
        for j in range(len(rings) - 1) for i in range(segments)
    ]
    faces.extend((
        tuple(range(segments)),
        tuple(reversed(range((len(rings) - 1) * segments, len(rings) * segments))),
    ))
    mesh = bpy.data.meshes.new("Reference torso")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    torso = bpy.data.objects.new("Reference torso", mesh)
    bpy.context.collection.objects.link(torso)
    torso.modifiers.new("Cloth collision", "COLLISION")
    torso.collision.thickness_outer = 0.002
    torso.hide_render = True

    forms = [torso]
    for side in ((-1, 1) if with_arms else ()):
        forms.append(collider(
            "Reference upper arm", (side * 0.235 * scale_x, 0, NECK_Z - 0.148 * length / 0.68),
            (0.041 * scale_x, 0.047 * scale_x, 0.15 * length / 0.68), (side, 0, -1),
        ))
    return forms


def contact_surfaces(forms):
    graph = bpy.context.evaluated_depsgraph_get()
    return [
        (BVHTree.FromObject(form, graph), form.matrix_world.copy(), form.matrix_world.inverted())
        for form in forms
    ]


def stabilize_contact(surfaces, vertices, clearance):
    """Project deeply overlapping vertices and report numerical contact depth."""
    result = vertices.copy()
    depth = 0.0
    for tree, world, inverse in surfaces:
        for i, point in enumerate(result):
            local = inverse @ Vector(point)
            nearest, normal, _, _ = tree.find_nearest(local)
            if nearest is None:
                continue
            overlap = (nearest - local).dot(normal)
            depth = max(depth, overlap)
            if clearance is not None and overlap > -clearance:
                result[i] = world @ (nearest + clearance * normal)
    return result, round(depth * 1000, 2)


def sewn_mesh(vertices, welded_faces, faces, uv, panels):
    boundary = Counter()
    for face in welded_faces:
        if len(set(face)) != 3:
            raise ValueError("Degenerate triangle after joining seams")
        for a, b in zip(face, (*face[1:], face[0])):
            boundary[tuple(sorted((a, b)))] += 1
    if any(count > 2 for count in boundary.values()):
        raise ValueError("Non-manifold sewing junction after joining seams")
    neighbors = defaultdict(set)
    for (a, b), count in boundary.items():
        if count == 1:
            neighbors[a].add(b)
            neighbors[b].add(a)
    if any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("Expected closed boundary curves at neck, cuffs and hem")
    openings = 0
    unseen = set(neighbors)
    while unseen:
        openings += 1
        pending = [unseen.pop()]
        for point in pending:
            for adjacent in neighbors[point] & unseen:
                unseen.remove(adjacent)
                pending.append(adjacent)
    if openings != 4:
        raise ValueError(f"Expected four garment openings, found {openings}")
    mesh = bpy.data.meshes.new("Sewn surface")
    mesh.from_pydata(vertices, [], welded_faces)
    mesh.update()
    layer = mesh.uv_layers.new(name="Flat panels")
    mesh.materials.append(None)
    mesh.materials.append(None)
    collar_start = panels["collar"]["vertex_start"]
    for poly, face in zip(mesh.polygons, faces):
        poly.use_smooth = True
        poly.material_index = int(face[0] >= collar_start)
        for loop, index in zip(poly.loop_indices, face):
            layer.data[loop].uv = uv[index]
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    model = bpy.data.objects.new("Sewn tee", mesh)
    bpy.context.collection.objects.link(model)
    return model, openings


def freeze_drape(source, positions):
    mesh = source.data.copy()
    for vertex, point in zip(mesh.vertices, positions):
        vertex.co = point
    displayed = bpy.data.objects.new("Sewn tee preview", mesh)
    bpy.context.collection.objects.link(displayed)
    subdivision = displayed.modifiers.new("Subdivided textile", "SUBSURF")
    subdivision.levels = subdivision.render_levels = 1
    thickness = displayed.modifiers.new("Textile thickness", "SOLIDIFY")
    thickness.thickness = 0.001
    thickness.offset = -1
    source.hide_render = True
    source.hide_set(True)
    return displayed


def textile(name, color, rib=False):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    connect = material.node_tree.links.new
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = material.diffuse_color
    shader.inputs["Roughness"].default_value = 0.87
    shader.inputs["Specular IOR Level"].default_value = 0.23
    shader.inputs["Sheen Weight"].default_value = 0.32

    coordinates = nodes.new("ShaderNodeTexCoord")
    texture = nodes.new("ShaderNodeTexWave" if rib else "ShaderNodeTexNoise")
    texture.inputs["Scale"].default_value = 150 if rib else 1650
    if rib:
        texture.bands_direction = "X"
        texture.inputs["Distortion"].default_value = 0
    connect(coordinates.outputs["UV"], texture.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.32 if rib else 0.22
    bump.inputs["Distance"].default_value = 0.0003 if rib else 0.00015
    connect(texture.outputs["Color" if rib else "Fac"], bump.inputs["Height"])
    connect(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def render(model, output, pattern):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 40
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.world.color = (0.48, 0.48, 0.48)
    scene.world.use_nodes = True
    world = scene.world.node_tree
    nodes = world.nodes
    background = nodes.get("Background")
    background.inputs["Color"].default_value = (0.4, 0.43, 0.46, 1)
    background.inputs["Strength"].default_value = 0.55
    camera_background = nodes.new("ShaderNodeBackground")
    camera_background.inputs["Color"].default_value = (0.74, 0.71, 0.65, 1)
    mix = nodes.new("ShaderNodeMixShader")
    camera_ray = nodes.new("ShaderNodeLightPath")
    world.links.new(camera_ray.outputs["Is Camera Ray"], mix.inputs[0])
    world.links.new(background.outputs[0], mix.inputs[1])
    world.links.new(camera_background.outputs[0], mix.inputs[2])
    world.links.new(mix.outputs[0], nodes.get("World Output").inputs["Surface"])
    model.data.materials[0] = textile("Reference cotton jersey", (0.082, 0.092, 0.094))
    model.data.materials[1] = textile("Reference neck rib", (0.065, 0.074, 0.077), rib=True)
    target = Vector((0, 0, NECK_Z - pattern["measurements"]["back_center_length_mm"] / 2000))
    for location, energy, size in (
        ((-2, -3, 3), 350, 3), ((2, 2, 3), 250, 3), ((1, 1, -0.4), 35, 2),
    ):
        light = bpy.data.lights.new("Studio softbox", "AREA")
        light.energy = energy
        light.shape, light.size = "DISK", size
        obj = bpy.data.objects.new(light.name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.cameras.new("Inspection camera")
    obj = bpy.data.objects.new(camera.name, camera)
    bpy.context.collection.objects.link(obj)
    scene.camera = obj
    camera.type, camera.ortho_scale = "ORTHO", max(0.95, pattern["measurements"]["back_center_length_mm"] / 760)
    for name, x, y in (("front", 0, -2.7), ("back", 0, 2.7), ("three-quarter", 1.6, -2.2)):
        obj.location = (x, y, target.z + 0.04)
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(output / (name + ".png"))
        bpy.ops.render.render(write_still=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("form", nargs="?", default="torso_with_arms",
                        choices=("torso_with_arms", "torso_only", "display_torso"))
    parser.add_argument("--frames", type=int, default=FRAME)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    source, output, form = args.source, args.output, args.form
    if args.frames < 2:
        parser.error("Drape requires at least two frames")
    pattern = json.loads(source.read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.frame_end = args.frames
    forms = add_form(pattern, with_arms=form == "torso_with_arms")
    contact_only = form == "display_torso"
    sleeve_angle = 20 if contact_only else 45
    placement = partial(place, sleeve_angle_from_vertical=sleeve_angle)
    points, welded_faces, faces, uv, pins, edges, rest, panels = build_assembly(pattern, place_vertex=placement)
    garment, openings = sewn_mesh(points, welded_faces, faces, uv, panels)
    configure_cloth(garment, [] if contact_only else pins, frame_end=args.frames)
    earlier_frame = max(1, args.frames - 5)
    for frame in range(1, args.frames + 1):
        scene.frame_set(frame)
        if frame == earlier_frame:
            earlier = evaluated_positions(garment)
        if frame in (1, 20, args.frames):
            print(f"Draping frame {frame}/{args.frames}", flush=True)
    positions = evaluated_positions(garment)
    if not np.isfinite(positions).all() or not np.isfinite(earlier).all():
        raise ValueError("Cloth solve produced non-finite coordinates")
    motion = displacement_summary(earlier, positions, earlier_frame, args.frames)
    contacts = contact_surfaces(forms)
    _, raw_penetration = stabilize_contact(contacts, positions, None)
    for _ in range(3):
        positions, _ = stabilize_contact(contacts, positions, 0.0022)
    model = freeze_drape(garment, positions)
    bpy.context.view_layer.update()
    drawn = model.evaluated_get(bpy.context.evaluated_depsgraph_get())
    _, visible_penetration = stabilize_contact(contacts, np.asarray([v.co[:] for v in drawn.data.vertices]), None)
    a, b = edges.T
    actual = np.linalg.norm(positions[a] - positions[b], axis=1)
    strain = np.abs(actual / rest - 1)
    report = {
        "solver": "Flat edge-length fit followed by Blender Cloth on a sewn surface",
        "reference_form": form,
        "cloth_support": "contact_only" if contact_only else "fixed_shoulders_and_contact",
        "initial_sleeve_angle_from_vertical_deg": sleeve_angle,
        "seam_connection": "Shared vertices throughout draping",
        "evaluated_frame": args.frames,
        "frames_per_second": scene.render.fps / scene.render.fps_base,
        "last_interval_vertex_displacement_mm": motion,
        "flat_metric_absolute_strain_percent": {
            "median": round(float(np.median(strain)) * 100, 2),
            "p95": round(float(np.quantile(strain, 0.95)) * 100, 2),
        },
        "welded_opening_count": openings,
        "max_reference_form_vertex_penetration_mm": {
            "cloth_solve": raw_penetration,
            "exported_surface": visible_penetration,
        },
        "pattern_sha256": sha256(source.read_bytes()).hexdigest(),
        "preview_surface": "Subdivision and 1 mm reference material thickness",
        "review": "Reference form and estimated knit settings, subject to physical comparison",
        "panel_vertex_ranges_before_welding": panels,
    }
    (output / "simulation.json").write_text(json.dumps(report, indent=2) + "\n")
    for form in forms:
        form.hide_set(True)
    bpy.ops.object.select_all(action="DESELECT")
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    render(model, output, pattern)
    bpy.ops.export_scene.gltf(filepath=str(output / "garment.glb"), use_selection=True, export_apply=True)
    scene.render.filepath = "//"
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "garment.blend"))


if __name__ == "__main__":
    main()
