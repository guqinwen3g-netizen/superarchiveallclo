"""Blender entry point for sewing and draping a captured tee on a reference form."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import ceil, cos, pi, sin
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superarchiveallclo.pattern import Edge

STEP_MM = 19
NECK_Z = 1.52
FRAME = 48


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


def flat_mesh(edges, steps):
    perimeter = []
    refs = {}
    for edge in edges:
        count = steps[edge.id]
        refs[edge.id] = list(range(len(perimeter), len(perimeter) + count + 1))
        perimeter.extend(edge.point_at(i / count)[0] for i in range(count))
    for indices in refs.values():
        indices[-1] %= len(perimeter)
    x, y = zip(*perimeter)
    interior = [
        (i * STEP_MM + STEP_MM * 0.37, j * STEP_MM + STEP_MM * 0.41)
        for i in range(int(min(x) / STEP_MM) - 1, int(max(x) / STEP_MM) + 1)
        for j in range(int(min(y) / STEP_MM) - 1, int(max(y) / STEP_MM) + 1)
    ]
    verts, _, faces, origins, _, _ = delaunay_2d_cdt(
        [Vector(p) for p in perimeter + interior], [], [list(range(len(perimeter)))], 1, 0.001,
    )
    retained = sorted({i for face in faces for i in face})
    compact = {i: j for j, i in enumerate(retained)}
    originals = {i: compact[j] for j in retained for i in origins[j]}
    return (
        [verts[i] for i in retained], [tuple(compact[i] for i in face) for face in faces],
        {name: [originals[i] for i in indices] for name, indices in refs.items()},
    )


def place(pattern, panel, edges, point):
    x, y = point
    p = pattern["parameters"]
    half_width = p["flat_chest_mm"] / 2
    rx, ry = half_width * 0.69 / 1000, half_width * 0.52 / 1000
    if panel in ("front", "back"):
        angle = x / half_width * pi / 2
        sign = -1 if panel == "front" else 1
        return Vector((rx * sin(angle), sign * (ry * cos(angle) + 0.002), NECK_Z - y / 1000))
    if panel == "collar":
        band = [edge for edge in edges if edge.id.startswith("attach_")]
        edge = next((edge for edge in band if x <= edge.end[0]), band[-1])
        t = min(1, max(0, (x - edge.start[0]) / edge.length()))
        seam = next(seam for seam in pattern["seams"] if seam["id"] == edge.id)
        name, neck_id = seam["a"]
        body = next(piece for piece in pattern["panels"] if piece["id"] == name)
        neck = Edge(**next(item for item in body["edges"] if item["id"] == neck_id))
        anchor = place(pattern, name, (), neck.point_at(1 - t if seam["reversed"] else t)[0])
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
    normal = Vector((0.54 * side, 0, 0.84))
    direction = Vector((0.84 * side, 0, -0.54))
    shoulder_x = rx * sin(pattern["measurements"]["across_shoulders_mm"] / (2 * half_width) * pi / 2)
    shoulder = Vector((side * shoulder_x, 0, NECK_Z - p["shoulder_slope_mm"] / 1000))
    return (
        shoulder - normal * radius + direction * (y / 1000)
        + normal * (radius * cos(angle)) + Vector((0, -radius * sin(angle), 0))
    )


def build_assembly(pattern):
    edges = {panel["id"]: panel_edges(panel) for panel in pattern["panels"]}
    steps = {
        (name, edge.id): max(1, ceil(edge.length() / STEP_MM))
        for name, panel in edges.items() for edge in panel
    }
    for seam in pattern["seams"]:
        a, b = tuple(seam["a"]), tuple(seam["b"])
        steps[a] = steps[b] = max(steps[a], steps[b])

    positions, rest, faces, refs, uv = [], [], [], {}, []
    panels = {}
    for name, panel in edges.items():
        verts, triangles, local_refs = flat_mesh(panel, {edge.id: steps[name, edge.id] for edge in panel})
        offset = len(positions)
        positions.extend(place(pattern, name, panel, v) for v in verts)
        rest.extend(Vector((v.x / 1000, 0, -v.y / 1000)) for v in verts)
        uv.extend(tuple(v / 1000) for v in verts)
        # Front panel is oriented toward -Y.
        faces.extend(tuple(i + offset for i in (face[::-1] if name == "front" else face)) for face in triangles)
        refs.update({(name, edge): [i + offset for i in indices] for edge, indices in local_refs.items()})
        panels[name] = {"vertex_start": offset, "vertex_count": len(verts)}

    links = {}
    neighbors = defaultdict(set)
    pins = set()
    for seam in pattern["seams"]:
        a, b = refs[tuple(seam["a"])], refs[tuple(seam["b"])]
        if seam["reversed"]:
            b = b[::-1]
        pairs = list(zip(a, b))
        links[seam["id"]] = pairs
        if seam["id"].startswith("shoulder"):
            pins.add(a[len(a) // 2])
        for i, j in pairs:
            neighbors[i].add(j)
            neighbors[j].add(i)

    # Pull neighboring edge vertices close before the solver starts.
    seeded = positions.copy()
    visited = set()
    for start in neighbors:
        if start in visited:
            continue
        group, pending = set(), [start]
        for point in pending:
            if point not in group:
                group.add(point)
                pending.extend(neighbors[point] - group)
        center = sum((positions[i] for i in group), Vector((0, 0, 0))) / len(group)
        for i in group:
            seeded[i] = positions[i].lerp(center, 0.94)
        visited.update(group)

    stitches = sorted({tuple(sorted(pair)) for pairs in links.values() for pair in pairs})
    mesh = bpy.data.meshes.new("Sewing panels")
    mesh.from_pydata(seeded, stitches, faces)
    mesh.update()
    garment = bpy.data.objects.new("Archived tee", mesh)
    bpy.context.collection.objects.link(garment)
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)

    garment.shape_key_add(name="Assembly")
    flat = garment.shape_key_add(name="Flat sewing rest")
    for i, co in enumerate(rest):
        flat.data[i].co = co
    group = garment.vertex_groups.new(name="Shoulder support")
    group.add(list(pins), 1, "REPLACE")
    cloth = garment.modifiers.new("Cloth sewing", "CLOTH")
    cloth.settings.rest_shape_key = flat
    cloth.settings.use_sewing_springs = True
    cloth.settings.sewing_force_max = 0
    cloth.settings.quality = 12
    cloth.settings.mass = 0.18
    cloth.settings.tension_stiffness = 19
    cloth.settings.compression_stiffness = 19
    cloth.settings.shear_stiffness = 12
    cloth.settings.bending_stiffness = 0.45
    cloth.settings.vertex_group_mass = group.name
    cloth.settings.pin_stiffness = 0.9
    cloth.collision_settings.use_self_collision = True
    cloth.collision_settings.self_distance_min = 0.0015
    cloth.point_cache.frame_end = FRAME
    return garment, links, panels, uv, faces


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


def add_form(pattern):
    flat_chest = pattern["measurements"]["flat_chest_mm"] / 1000
    length = pattern["measurements"]["back_center_length_mm"] / 1000
    rx, ry = 0.285 * flat_chest, 0.209 * flat_chest
    forms = [collider("Reference torso", (0, 0, NECK_Z - length * 0.43), (rx, ry, length * 0.52))]
    for side in (-1, 1):
        forms.append(collider(
            "Reference upper arm", (side * rx * 1.17, 0, NECK_Z - 0.15),
            (0.033, 0.045, 0.145), (side * 0.84, 0, -0.54),
        ))
    return forms


def baked_garment(garment, links, positions, faces, uv):
    # Merge only the sewn vertex pairs. Preserve panel-specific UVs per face corner.
    parent = list(range(len(positions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pairs in links.values():
        for a, b in pairs:
            parent[root(b)] = root(a)
    groups = defaultdict(list)
    for i in range(len(positions)):
        groups[root(i)].append(i)
    mapping = {k: i for i, k in enumerate(groups)}
    welded = [sum((positions[i] for i in indices), Vector((0, 0, 0))) / len(indices) for indices in groups.values()]
    welded_faces = [tuple(mapping[root(v)] for v in f) for f in faces]
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
    mesh.from_pydata(welded, [], welded_faces)
    mesh.update()
    layer = mesh.uv_layers.new(name="Flat panels")
    for poly, face in zip(mesh.polygons, faces):
        poly.use_smooth = True
        for loop, index in zip(poly.loop_indices, face):
            layer.data[loop].uv = uv[index]
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    model = bpy.data.objects.new("Sewn tee", mesh)
    bpy.context.collection.objects.link(model)
    smooth = model.modifiers.new("Relax sewn junctions", "SMOOTH")
    smooth.factor = 0.48
    smooth.iterations = 4
    subdivision = model.modifiers.new("Preview surface", "SUBSURF")
    subdivision.levels = subdivision.render_levels = 1
    garment.hide_render = True
    garment.hide_set(True)
    return model, openings


def render(model, output, pattern):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.world.color = (0.52, 0.52, 0.52)
    material = bpy.data.materials.new("Reference charcoal knit")
    material.diffuse_color = (0.155, 0.183, 0.169, 1)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = material.diffuse_color
    shader.inputs["Roughness"].default_value = 0.82
    shader.inputs["Sheen Weight"].default_value = 0.25
    model.data.materials.append(material)
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
    source, output = map(Path, sys.argv[sys.argv.index("--") + 1:])
    pattern = json.loads(source.read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.frame_end = FRAME
    forms = add_form(pattern)
    garment, links, panels, uv, faces = build_assembly(pattern)
    for frame in range(1, FRAME + 1):
        scene.frame_set(frame)
        if frame in (1, 24, FRAME):
            print(f"Sewing frame {frame}/{FRAME}", flush=True)
    evaluated = garment.evaluated_get(bpy.context.evaluated_depsgraph_get())
    positions = [v.co.copy() for v in evaluated.data.vertices]
    gaps = {
        name: round(max((positions[a] - positions[b]).length for a, b in pairs) * 1000, 3)
        for name, pairs in links.items()
    }
    model, openings = baked_garment(garment, links, positions, faces, uv)
    report = {
        "solver": "Blender Cloth with rest lengths from flat sewing panels",
        "evaluated_frame": FRAME,
        "pre_weld_max_seam_gap_mm": gaps,
        "welded_opening_count": openings,
        "pattern_sha256": sha256(source.read_bytes()).hexdigest(),
        "preview_surface": "Four smoothing passes and one subdivision level applied after sewing",
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
