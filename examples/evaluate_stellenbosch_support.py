"""Isolate the effect of the upper-arm support on a measured museum tee."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import atan2, degrees
from pathlib import Path
from statistics import median

import numpy as np

from examples.evaluate_auckland_tees import PAIRED_WEIGHTS, write_json
from superarchiveallclo import photo_points, visual_hull
from superarchiveallclo.capture import archive
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.reference_form import DEFAULT_ARM_ANGLE_DEG

LANDMARKS = Path(__file__).with_name("stellenbosch-landmarks.json")

# The stitch line is visible on the support just below each cloth cuff.
# Points are in the same 576 x 864 portrait raster as the reference masks.
ARM_STITCH_LINES = {"image_left": [[104, 426], [90, 480]], "image_right": [[474, 432], [490, 485]]}


def observed_angle(points):
    top, bottom = points
    return degrees(atan2(abs(bottom[0] - top[0]), bottom[1] - top[1]))


def summarize(candidate, reference, output, blender):
    simulation = json.loads((candidate / "model/simulation.json").read_bytes())
    shapes = visual_hull.compare(candidate, reference, output / "hull", blender=blender)
    points = photo_points.compare(candidate, reference, LANDMARKS, output / "landmarks", blender=blender)
    if shapes["garment_glb_sha256"] != points["garment_glb_sha256"]:
        raise ValueError("Both measurements must refer to the same delivered GLB")
    spans = []
    for section in shapes["horizontal_spans_by_height"]:
        reference_span, model_span = section["hull_width_depth_mm"], section["glb_width_depth_mm"]
        if reference_span is not None and model_span is not None:
            spans.append({**section, "glb_minus_hull_width_depth_mm":
                          (np.array(model_span) - reference_span).round(2).tolist()})
    return {
        "pattern_sha256": shapes["pattern_sha256"], "garment_glb_sha256": shapes["garment_glb_sha256"],
        "arm_angle_from_vertical_deg": simulation.get("reference_upper_arm_angle_from_vertical_deg", DEFAULT_ARM_ANGLE_DEG),
        "held_out_contours": {name: view["glb"] for name, view in shapes["views"].items() if view["role"] == "held_out"},
        "surface_to_hull": shapes["candidate_surface_vs_hull"], "horizontal_spans": spans,
        "local_points": points["summary"],
        "strain_percent": simulation["flat_metric_absolute_strain_percent"],
        "last_interval_motion_mm": simulation["last_interval_vertex_displacement_mm"],
        "penetration_mm": simulation["max_reference_form_vertex_penetration_mm"],
    }


def evaluate(study, output, blender):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    baseline = study / "review/archive"
    reference = study / "reference/hull-reference.json"
    baseline_audit = check_archive(baseline)
    base_simulation = json.loads((baseline / "model/simulation.json").read_bytes())
    base_capture = json.loads((baseline / "capture.json").read_bytes())
    reference_source = json.loads(reference.read_bytes())
    form = base_simulation["reference_form"]
    if (base_capture["item_id"] != "auckland-2017.66.37" or form != "torso_with_arms"
            or base_simulation.get("reference_upper_arm_angle_from_vertical_deg", DEFAULT_ARM_ANGLE_DEG) != DEFAULT_ARM_ANGLE_DEG
            or not base_capture["inference"]["reference_measurements_withheld"]):
        raise ValueError("Supply the default, blind Stellenbosch archive from the full photo comparison")
    if sha256(PAIRED_WEIGHTS.read_bytes()).hexdigest() != base_capture["inference"]["weights_sha256"]:
        raise ValueError("Paired model weights differ from the baseline archive")
    source_frame = reference_source["views"]["photo-02"]
    angle_by_arm = {side: round(observed_angle(points), 3) for side, points in ARM_STITCH_LINES.items()}
    angle = round(median(angle_by_arm.values()), 3)
    output.mkdir(parents=True)
    adjusted = output / "adjusted/archive"
    archive(study / "input/paired.json", adjusted, blender, PAIRED_WEIGHTS, blind=True, form=form,
            drape_frames=base_simulation["evaluated_frame"], photo_appearance="embedded_photo_atlases" in baseline_audit["glb"],
            arm_angle_deg=angle)
    adjusted_audit = check_archive(adjusted)
    if adjusted_audit["pattern_sha256"] != baseline_audit["pattern_sha256"]:
        raise ValueError("Arm support experiment changed the source paper pattern")
    adjusted_simulation = json.loads((adjusted / "model/simulation.json").read_bytes())
    for key in ("initial_sleeve_angle_from_vertical_deg", "evaluated_frame", "cloth_support"):
        if base_simulation[key] != adjusted_simulation[key]:
            raise ValueError(f"Support experiment changed {key}")
    if base_simulation.get("appearance_atlas_sha256") != adjusted_simulation.get("appearance_atlas_sha256"):
        raise ValueError("Support experiment changed the photo appearance atlas")
    baseline_report = summarize(baseline, reference, output / "baseline", blender)
    adjusted_report = summarize(adjusted, reference, output / "adjusted", blender)
    return write_json(output / "study.json", {
        "item_id": base_capture["item_id"], "study": "Same pattern, cloth parameters, initial pose and frame count",
        "scope": "The visible lower part of each upper-arm seam estimates the frontal angle. Depth or anatomical pivot is unmeasured.",
        "support_reference": {
            "file": source_frame["photo"], "sha256": source_frame["photo_sha256"], "raster_px": [576, 864],
            "stitch_line_points_px": ARM_STITCH_LINES, "angle_by_arm_deg": angle_by_arm,
        },
        "reference_manifest_sha256": sha256(reference.read_bytes()).hexdigest(),
        "baseline": baseline_report, "observed_support": adjusted_report,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.study, args.output, args.blender)}")


if __name__ == "__main__":
    main()
