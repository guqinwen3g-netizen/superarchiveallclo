"""Test a display form measured from visible neck edges on the Auckland archive."""

from argparse import ArgumentParser
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from examples.evaluate_auckland_tees import PAIRED_WEIGHTS, write_json
from superarchiveallclo.capture import archive
from superarchiveallclo.compare_shape import read_mesh
from superarchiveallclo.photo_points import compare as compare_points, supplied_matrices
from superarchiveallclo.visual_hull import projected_mesh_mask, read_view, save_overlay, silhouette_bands, silhouette_metrics

LANDMARKS = Path(__file__).with_name("auckland-landmarks.json")
NECK_EDGES = {
    "photo-02": ((216, 250, 81, 111, 1), (320, 345, 81, 111, -1)),
    "photo-03": ((316, 354, 72, 108, -1),),
    "photo-05": ((164, 204, 68, 96, 1),),
}


def trace_edge(chroma, guide):
    left, right, top, bottom, direction = guide
    rows = np.arange(top, bottom)
    strength = direction * gaussian_filter1d(chroma[top:bottom, left:right], 1, order=1, axis=1)
    columns = strength.argmax(axis=1)
    valid = ((columns > 1) & (columns < right - left - 2)
             & (strength[np.arange(len(rows)), columns] > 1.2))
    columns += left
    if valid.sum() < .8 * len(rows):
        raise ValueError("The neck edge has too little contrast against the background")
    estimate = np.polyfit(rows[valid], columns[valid], 1)
    valid &= np.abs(np.polyval(estimate, rows) - columns) <= 2
    if valid.sum() < .8 * len(rows):
        raise ValueError("The fit window does not contain a straight exposed neck edge")
    line = np.polyfit(rows[valid], columns[valid], 1)
    error = np.abs(np.polyval(line, rows[valid]) - columns[valid])
    return {"guide": list(guide), "x_from_y": line.tolist(), "sample_count": int(valid.sum()),
            "residual_p90_px": round(float(np.quantile(error, .9)), 3)}


def measure_neck(reference, views):
    edges = {}
    for name, guides in NECK_EDGES.items():
        view, entry = views[name]
        if entry["role"] != "carve":
            raise ValueError("Use only calibration photographs to measure the display neck")
        with Image.open(BytesIO((reference.parent / entry["photo"]).read_bytes())) as photo:
            rgb = np.asarray(photo.convert("RGB").resize(view.mask.shape[::-1]), dtype=float)
        chroma = rgb[:, :, :2].mean(axis=2) - rgb[:, :, 2]
        edges[name] = {"photo_sha256": entry["photo_sha256"], "edges": [trace_edge(chroma, guide) for guide in guides]}
    front, left, right = NECK_EDGES
    slopes = [edges[name]["edges"][0]["x_from_y"][0] for name in (left, right)]
    pitch = np.degrees(np.arctan(np.array(slopes) * [views[left][0].sign, views[right][0].sign]))
    if abs(pitch[0] - pitch[1]) > 6:
        raise ValueError("The two side views disagree on the exposed neck direction")
    front_left, front_right = (np.polyval(edge["x_from_y"], 90) for edge in edges[front]["edges"])
    return {"schema_version": 1, "lateral_width_mm": round(float((front_right - front_left) * views[front][0].mm_per_pixel), 3),
            "pitch_forward_deg": round(float(pitch.mean()), 3),
            "source": "Auckland Museum 2017.66.22, exposed beige display-form neck edges on photo-02, photo-03 and photo-05.",
            "evidence": {"front_width_sample_row_px": 90, "side_pitch_deg": pitch.round(3).tolist(), "per_view": edges}}


def score(candidate, reference, views, matrices, cameras, output, blender):
    output.mkdir()
    points = compare_points(candidate, reference, LANDMARKS, output / "points", blender=blender,
                            camera_solution=cameras, reprojection_limit_mm=25)
    glb = candidate / "model/garment.glb"
    vertices, faces = read_mesh(glb, blender)
    vertices = vertices * 1000 - json.loads(reference.read_bytes())["model_origin_mm"]
    projections = {}
    for name, (view, entry) in views.items():
        image = projected_mesh_mask(vertices, faces, matrices[name], view.mask.shape)
        save_overlay(output / f"{name}.png", view.mask, image, view.occluded)
        projections[name] = {"role": entry["role"], **silhouette_metrics(view.mask, image, view.mm_per_pixel, view.occluded),
                             "candidate_reaches_image_edge": bool(image[[0, -1]].any() or image[:, [0, -1]].any()),
                             "height_bands": silhouette_bands(view.mask, image, view)}
    simulation = json.loads((candidate / "model/simulation.json").read_bytes())
    return {"projection": projections, "archive_audit": points["archive_audit"],
            "garment_glb_sha256": points["garment_glb_sha256"],
            "display_neck": simulation.get("display_neck"),
            "initial_sleeve_angle_from_vertical_deg": simulation["initial_sleeve_angle_from_vertical_deg"],
            "max_reference_form_vertex_penetration_mm": simulation["max_reference_form_vertex_penetration_mm"],
            "flat_metric_absolute_strain_percent": simulation["flat_metric_absolute_strain_percent"],
            "local_3d": points["summary"]["supplied_projection"]}


def evaluate(study, cameras, output, blender, *, sleeve_angle=None):
    reference = study / "reference/hull-reference.json"
    record = json.loads(reference.read_bytes())
    if record["item_id"] != "auckland-2017.66.22":
        raise ValueError("Use the complete netball study for Auckland Museum 2017.66.22")
    views = {name: read_view(reference.parent, entry) for name, entry in record["views"].items()}
    camera_bytes = cameras.read_bytes()
    matrices, calibration = supplied_matrices(camera_bytes, record["item_id"],
        views["photo-02"][0].mask.shape[::-1], {name: entry["photo_sha256"] for name, (_, entry) in views.items()})
    output.mkdir(parents=True)
    profile = write_json(output / "display-neck.json", measure_neck(reference, views))
    control, test = study / "review/archive", output / "archive"
    archive(study / "input/paired.json", test, blender, Path(__file__).resolve().parents[1] / PAIRED_WEIGHTS,
            blind=True, form="display_torso", photo_appearance=True, display_neck_path=profile,
            initial_sleeve_angle_deg=sleeve_angle)
    if (control / "pattern/pattern.json").read_bytes() != (test / "pattern/pattern.json").read_bytes():
        raise ValueError("Both simulations must use the same sewing pattern")
    return write_json(output / "study.json", {
        "schema_version": 1, "item_id": record["item_id"], "camera_calibration": calibration,
        "profile": json.loads(profile.read_bytes()),
        "scope": "Only the exposed support neck is measured. The three measurement photos supply no garment target to the simulation.",
        "control": score(control, reference, views, matrices, cameras, output / "control", blender),
        "measured_neck": score(test, reference, views, matrices, cameras, output / "measured-neck", blender),
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--cameras", required=True, type=Path, help="Camera JSON from examples.evaluate_auckland_stand")
    parser.add_argument("--initial-sleeve-angle", type=float)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(evaluate(args.study, args.cameras, args.output, args.blender, sleeve_angle=args.initial_sleeve_angle))


if __name__ == "__main__":
    main()
