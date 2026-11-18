"""Blender job for the static two-grip cloth experiment."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superarchiveallclo.flat_pose import flat_assembly
from superarchiveallclo.hang_supports import place_for_gravity, rotate_about_support
from superarchiveallclo.read_garment_glb import mesh_geometry
from superarchiveallclo.simulate import bending_settings, configure_cloth, displacement_summary, evaluated_positions, sewn_mesh

FRAME = 55
MOTION_INTERVAL_FRAMES = 8


def ray_depth_errors(surface, cloud, camera):
    """Signed distance to the first mesh hit along each observed camera ray."""
    errors = np.full(len(cloud), np.nan)
    origin = Vector(camera)
    directions = cloud - camera
    ranges = np.linalg.norm(directions, axis=1)
    for i in np.flatnonzero(ranges > 1e-9):
        direction = Vector(directions[i] / ranges[i])
        _, _, _, distance = surface.ray_cast(origin, direction)
        if distance is not None:
            errors[i] = (distance - ranges[i]) * 1000
    return errors


def save_pose(source, positions, filename):
    mesh = source.data.copy()
    for vertex, position in zip(mesh.vertices, positions, strict=True):
        vertex.co = position
    model = bpy.data.objects.new(filename.stem, mesh)
    bpy.context.collection.objects.link(model)
    bpy.ops.object.select_all(action="DESELECT")
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.export_scene.gltf(filepath=str(filename), use_selection=True)
    bpy.data.objects.remove(model, do_unlink=True)


def main():
    filename, runtime, angle = sys.argv[sys.argv.index("--") + 1:]
    directory = Path(runtime)
    pattern = json.loads(Path(filename).read_bytes())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.frame_end = FRAME
    scene.render.fps = 24
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    flat, triangles, faces, uv, _, edges, rest, panels = flat_assembly(pattern, float(angle), {})
    with np.load(directory / "hang-input.npz", allow_pickle=False) as job:
        posed, patches, history = place_for_gravity(flat, job["flat"], job["rest_contacts"], job["supports"])
        scans = {key: job[key] for key in job.files if key.startswith("scan-")}
        supports = job["supports"]
        camera = job["camera_world_m"] if "camera_world_m" in job else None
        forecast = job["forecast_rotation_radians"] if "forecast_rotation_radians" in job else None
        mass = float(job["garment_mass_kg"]) if "garment_mass_kg" in job else None
        bending = float(job["blender_bending_stiffness"]) if "blender_bending_stiffness" in job else None
    garment, loops = sewn_mesh(posed, triangles, faces, uv, panels)
    save_pose(garment, posed, directory / "before-gravity.glb")
    mass_settings = configure_cloth(garment, np.concatenate(patches).tolist(), frame_end=FRAME, garment_mass_kg=mass,
                                    bending_stiffness=bending)
    cloth = garment.modifiers["Cloth drape"]
    cloth.settings.effector_weights.gravity = 1.0
    cloth.settings.time_scale = 0.8
    for frame in range(1, FRAME + 1):
        scene.frame_set(frame)
        if frame == FRAME - MOTION_INTERVAL_FRAMES:
            earlier = evaluated_positions(garment)
        if frame in (1, 20, 40, FRAME):
            print(f"Suspended cloth frame {frame}/{FRAME}", flush=True)
    final = evaluated_positions(garment)
    if not np.isfinite(final).all() or not np.isfinite(earlier).all():
        raise ValueError("Suspended cloth solve produced non-finite coordinates")
    save_pose(garment, final, directory / "gravity-pose.glb")
    support_drift = [float(np.linalg.norm(final[patch].mean(axis=0) - goal) * 1000)
                     for patch, goal in zip(patches, supports, strict=True)]
    strain = np.abs(np.linalg.norm(final[edges[:, 0]] - final[edges[:, 1]], axis=1) / rest - 1)
    (directory / "support-metrics.json").write_text(json.dumps({
        **history, "pinned_patch_centroid_drift_mm": np.round(support_drift, 4).tolist(),
        "edge_strain_p95_percent": round(float(np.quantile(strain, 0.95) * 100), 2),
        "frame": FRAME, "fps": scene.render.fps, "time_scale": 0.8,
        "last_interval_vertex_displacement_mm": displacement_summary(
            earlier, final, FRAME - MOTION_INTERVAL_FRAMES, FRAME,
        ),
        "openings": len(loops), "cloth_mass_setting": mass_settings["vertex_mass_kg"], "mass_settings": mass_settings,
        "bending_settings": bending_settings(garment),
    }, indent=2) + "\n")
    metrics = {}
    for variant, name in (("before", "before-gravity.glb"), ("after", "gravity-pose.glb")):
        vertices, triangles = mesh_geometry(directory / name)
        metrics[f"{variant}_vertices"] = vertices
        metrics[f"{variant}_triangles"] = triangles
        surface = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
        for number, (key, cloud) in enumerate(scans.items()):
            metrics[f"{variant}_{key}"] = 1000 * np.array([surface.find_nearest(point.tolist())[3] for point in cloud])
            if camera is not None:
                metrics[f"{variant}_{key}_ray_depth_mm"] = ray_depth_errors(surface, cloud, camera)
            if variant == "after" and forecast is not None:
                # Invert the rigid swing on the rays instead of rebuilding the BVH for every frame.
                angle = -forecast[number]
                metrics[f"forecast_{key}_ray_depth_mm"] = ray_depth_errors(
                    surface, rotate_about_support(cloud, supports, angle), rotate_about_support(camera, supports, angle),
                )
    np.savez_compressed(directory / "hang-metrics.npz", **metrics)


if __name__ == "__main__":
    main()
