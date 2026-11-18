"""Calibrate two held-out museum views from the physical stand and ruler."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from itertools import pairwise
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from examples.evaluate_auckland_drape import SOURCE as PHOTOS
from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.calibrate_cameras import error_pixels, fit_planar_pose, size_from_planar_ruler
from superarchiveallclo.compare_shape import read_mesh
from superarchiveallclo.photo_points import (
    compare, exif_focal_pixels, pixel_sensitivity, projection_matrix, read_point_view, triangulate,
)
from superarchiveallclo.rectify import project, rectangle
from superarchiveallclo.visual_hull import projected_mesh_mask, silhouette_metrics


GUIDES = Path(__file__).with_name("auckland-stand.json")
LANDMARKS = Path(__file__).with_name("auckland-landmarks.json")
FRONT = ("photo-01", "photo-20")


def plate_edge(gray, guide, factor):
    start, end = np.asarray(guide)
    along = end - start
    normal = along[::-1] * [-1, 1] / np.linalg.norm(along)
    offsets = np.arange(-9, 9.1, .25)
    fractions = np.linspace(0, 1, 48)
    edge = start + fractions[:, None] * along
    probes = (edge[:, None, :] + offsets[None, :, None] * normal) * factor
    intensities = map_coordinates(gray, [probes[..., 1], probes[..., 0]], order=1)
    gradient = abs(np.diff(intensities, axis=1))
    best = gradient.argmax(axis=1)
    scores = gradient[np.arange(len(best)), best]
    observed = (offsets[best] + offsets[best + 1]) / 2
    valid = scores > max(3.5, .75 * np.quantile(scores, .35))
    if valid.sum() < 12:
        raise ValueError("The original photograph does not show enough of a straight stand edge")
    valid &= abs(observed - np.median(observed[valid])) < 5
    fit = np.polyfit(fractions[valid], observed[valid], 1)
    valid &= abs(observed - np.polyval(fit, fractions)) < 2
    if valid.sum() < 12:
        raise ValueError("The plate edge has too little straight line support")
    slope, offset = np.polyfit(fractions[valid], observed[valid], 1)
    delta = abs(observed[valid] - (slope * fractions[valid] + offset))
    line = np.cross(np.r_[start + offset * normal, 1], np.r_[end + (slope + offset) * normal, 1])
    return line, {"accepted_samples": int(valid.sum()), "distance_p90_px": round(float(np.quantile(delta, .9)), 4)}


def plate_corners(encoded, guides, size):
    focal = exif_focal_pixels(encoded, size, "landscape_sensor")
    camera = np.diag([*focal, 1.0])
    camera[:2, 2] = (np.asarray(size) - 1) / 2
    with Image.open(BytesIO(encoded)) as photo:
        factor = photo.width / size[0]
        gray = gaussian_filter(np.asarray(photo.convert("L"), dtype=float), .35 * factor)
    lines, edges = {}, {}
    for name, guide in guides.items():
        lines[name], edges[name] = plate_edge(gray, guide, factor)
    sides = ("left", "top", "right", "bottom", "left")
    corners = np.array([np.cross(lines[a], lines[b]) for a, b in pairwise(sides)])
    return corners[:, :2] / corners[:, 2:], camera, edges


def height_at_photo_rows(matrix, anchor, rows):
    q = matrix @ np.r_[anchor, 0, 1]
    return (q[1] - rows * q[2]) / (rows * matrix[2, 2] - matrix[1, 2])


def shaft_holdout(encoded, roi, size, camera, anchor):
    """Measure the black shaft independently of the four base edges."""
    with Image.open(BytesIO(encoded)) as photo:
        factor = photo.width / size[0]
        left, top, right, bottom = (round(v * factor) for v in roi)
        tile = np.asarray(photo.crop((left, top, right, bottom)).convert("L"), dtype=float)
    gradient = gaussian_filter1d(tile, 1.25, order=1, axis=1)
    middle = tile.shape[1] // 2
    l = gradient[:, :middle].argmin(axis=1)
    r = middle + gradient[:, middle:].argmax(axis=1)
    rows = np.arange(len(l))
    accepted = (gradient[rows, l] < -6) & (gradient[rows, r] > 6)
    if not accepted.any():
        raise ValueError("Could not independently observe both sides of the shaft")
    actual_x = (left + (l[accepted] + r[accepted]) / 2) / factor
    width = (r[accepted] - l[accepted]) / factor
    height = height_at_photo_rows(camera, anchor, (top + rows[accepted]) / factor)
    world = np.column_stack((np.broadcast_to(anchor, (len(height), 2)), height, np.ones(len(height))))
    physical = world @ camera.T
    predicted_x = physical[:, 0] / physical[:, 2]
    lateral_gradient = abs((camera[0, 0] - predicted_x * camera[2, 0]) / physical[:, 2])
    residual = abs(predicted_x - actual_x)
    return {"pixel_rows": int(accepted.sum()), "horizontal_reprojection_median_px": round(float(np.median(residual)), 3),
            "horizontal_reprojection_p90_px": round(float(np.quantile(residual, .90)), 3),
            "shaft_diameter_estimate_mm": round(float(np.median(width / lateral_gradient)), 3)}


def tested_silhouettes(root, metadata, views, cameras, glb, blender):
    vertices, faces = read_mesh(glb, blender)
    points = vertices * 1000 - metadata["model_origin_mm"]
    result = {}
    for name in FRONT:
        with Image.open(root / metadata["views"][name]["mask"]) as encoded:
            observed = np.asarray(encoded.convert("1"))
        methods = {}
        for method, camera in (("orthographic", projection_matrix(views[name])), ("metric_front", cameras[name])):
            raster = projected_mesh_mask(points, faces, camera, observed.shape)
            methods[method] = silhouette_metrics(observed, raster, views[name].mm_per_pixel)
        result[name] = methods
    return result


def evaluate(study, output, blender):
    root = study / "reference"
    reference = root / "hull-reference.json"
    metadata = json.loads(reference.read_bytes())
    guides = json.loads(GUIDES.read_bytes())
    if metadata["item_id"] != guides["item_id"] or not set(FRONT) <= metadata["views"].keys():
        raise ValueError("Run the complete Auckland netball experiment with both held-out front photographs")
    raw = {name: (root / view["photo"]).read_bytes() for name, view in metadata["views"].items()}
    size = tuple(guides["annotation_size_px"])
    views = {name: read_point_view(raw[name], view, size) for name, view in metadata["views"].items()}
    originals = json.loads(PHOTOS.read_bytes())
    plates = {name: plate_corners(raw[name], guides["plate_edge_guides_px"][name], size) for name in FRONT}
    ruler = originals["calibration"]["rulers"][guides["ruler_index"]]
    ruler_pixels = np.array(ruler["points_original_px"]) * size[0] / originals["photo_size_px"][0]
    scale_name = guides["original_scale_photo"]
    width_depth, cosine = size_from_planar_ruler(*plates[scale_name][:2], ruler_pixels, ruler["length_mm"])
    world_xy = rectangle(*width_depth) * [1, -1]
    fits = {name: fit_planar_pose(world_xy, *plates[name][:2]) for name in FRONT}
    anchor_xy = project(np.linalg.inv(fits["photo-01"][0][:, [0, 1, 3]]),
                        np.array([guides["stand_axis_foot_photo_01_px"]]))[0]
    independently_observed_heights = {name: float(height_at_photo_rows(fits[name][0], anchor_xy, views[name].origin[1]))
                                      for name in FRONT}
    height = float(np.mean(list(independently_observed_heights.values())))
    origin = np.r_[anchor_xy, height, 1]
    cameras = {name: projection_matrix(view) for name, view in views.items()}
    audits = {}
    for name, (matrix, center) in fits.items():
        translated = matrix.copy()
        translated[:, 3] = matrix @ origin
        cameras[name] = translated
        asset = originals["views"][str(int(name[-2:]))]
        audits[name] = {
            "base_plate_corners_px": plates[name][0].round(4).tolist(),
            "base_edge_diagnostics": plates[name][2],
            "metric_camera_center_mm": center.round(3).tolist(),
            "base_corners_fit": error_pixels(matrix, np.column_stack((world_xy, np.zeros(4))), plates[name][0]),
            "shaft_holdout": shaft_holdout(raw[name], asset["pole_roi"], size, matrix, anchor_xy),
        }

    output.mkdir(parents=True)
    solution = write_json(output / "cameras.json", {
        "schema_version": 1, "world_unit": "mm", "item_id": metadata["item_id"], "image_size_px": list(size),
        "source": "Two front poses from rectangular stand, 500 mm ruler and EXIF. Close-up camera geometry remains orthographic.",
        "views": {name: {"photo_sha256": metadata["views"][name]["photo_sha256"], "image_from_world": camera.tolist()}
                  for name, camera in cameras.items()},
    })
    archive = study / "review/archive"
    baseline = compare(archive, reference, LANDMARKS, output / "orthographic", blender=blender, reprojection_limit_mm=25)
    revised = compare(archive, reference, LANDMARKS, output / "metric-front", blender=blender, camera_solution=solution,
                     reprojection_limit_mm=25)
    first, second = (fits[name][1] for name in FRONT)
    measured_correspondence = json.loads(LANDMARKS.read_bytes())["points"]
    sensitivities = {}
    for label, entry in measured_correspondence.items():
        xyz = triangulate(cameras, views, entry["held_out"])
        sensitivities[label] = pixel_sensitivity(xyz, [cameras[name] for name in FRONT])
    silhouettes = tested_silhouettes(root, metadata, views, cameras, archive / "model/garment.glb", blender)
    return write_json(output / "study.json", {
        "schema_version": 1, "item_id": metadata["item_id"], "pattern_sha256": revised["pattern_sha256"],
        "archive_audit": revised["archive_audit"], "garment_glb_sha256": revised["garment_glb_sha256"],
        "fixture_guide_sha256": sha256(GUIDES.read_bytes()).hexdigest(), "camera_solution_sha256": sha256(solution.read_bytes()).hexdigest(),
        "reference_sha256": revised["reference_sha256"], "reference_photo_sha256": revised["reference_photo_sha256"],
        "metric_scale": {"photo": scale_name, "name": ruler["name"], "length_mm": ruler["length_mm"],
                         "assumption": "The ruler marks lie on the same plane as the top edges of the stand",
                         "ruler_marks_px": ruler_pixels.tolist(), "stand_width_depth_mm": width_depth.round(3).tolist(),
                         "recovered_plane_axes_cosine": round(cosine, 5)},
        "fixture": {"vertical_shaft_foot_mm": anchor_xy.round(3).tolist(),
                    "rim_level_observations_mm": independently_observed_heights,
                    "mean_rim_datum_mm": round(height, 3), "front_camera_baseline_mm": round(float(np.linalg.norm(first-second)), 3),
                    "front_only_worst_direction_mm_per_pixel": sensitivities, "per_view": audits},
        "baseline": {"fit_summary": baseline["summary"]["orthographic"],
                     "checked": baseline["reprojection_check"]["methods"]["orthographic"]},
        "metric_front": {"fit_summary": revised["summary"]["supplied_projection"],
                         "checked": revised["reprojection_check"]["methods"]["supplied_projection"]},
        "held_out_silhouettes": silhouettes,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Output from examples.evaluate_auckland_archive")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(evaluate(args.study, args.output, args.blender))


if __name__ == "__main__":
    main()
