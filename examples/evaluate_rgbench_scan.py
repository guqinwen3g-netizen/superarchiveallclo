"""Reproduce the real single-view flat-lay comparison on RGBench green_tshirt."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import dist
from pathlib import Path
from statistics import median
from urllib.request import urlopen

from superarchiveallclo.capture import load_capture
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat import compare_flat
from superarchiveallclo.compare_pattern import compare_pattern
from superarchiveallclo.scan import reconstruct
from superarchiveallclo.tee import draft

DATASET = "https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1"
REVISION = "136c00dc5f96b6b3d20427e93875a1c00d7a7cc9"
CAPTURES = {
    "18-21-20": ("1753698080.867096", "1753698081.589540", "1753698086.646549"),
    "18-22-41": ("1753698161.836080", "1753698162.557688", "1753698168.089555"),
    "18-29-24": ("1753698564.743875", "1753698565.465436", "1753698571.260411"),
}
FRAME_IDS = ("start", "start-2", "middle")
INPUT = "18-21-20-start"
REFERENCE_OBJ = "meshes/Green_Tshirt/green_tshirt_5k.obj"
REFERENCE_LANDMARKS = "green_tshirt/green_tshirt_fold_2025-07-28-18-21-20/calibration/landmarks.json"


def download(directory, filename, remote):
    path = directory / filename
    if not path.is_file():
        with urlopen(f"{DATASET}/resolve/{REVISION}/{remote}", timeout=45) as response:
            path.write_bytes(response.read())
    return path


def summarize(rows, dimensions):
    comparison = {}
    for name, value in dimensions.items():
        values = [row["observed_mm"][name] for row in rows]
        comparison[name] = {
            "reference_range_mm": [min(values), max(values)],
            "candidate_mm": value,
            "median_absolute_difference_mm": round(median(abs(sample - value) for sample in values), 2),
        }
    return comparison


def cuff_reference(directory, input_row):
    encoded = download(directory, "reference-landmarks.json", REFERENCE_LANDMARKS).read_bytes()
    points = json.loads(encoded)["scan_points"]
    measured = input_row["flat_landmarks"].get("cuffs", {})
    pairs = {"image_left": (1, 2), "image_right": (5, 6)}
    comparisons = {}
    for side, (a, b) in pairs.items():
        reference = dist(points[a - 1], points[b - 1]) * 1000
        observed = measured.get(side, {}).get("folded_opening_mm")
        comparisons[side] = {
            "source_point_indices_1_based": [a, b], "source_chord_mm": round(reference, 2),
            "automated_folded_cuff_mm": observed,
            "absolute_delta_mm": round(abs(reference - observed), 2) if observed is not None else None,
        }
    return {"dataset_path": REFERENCE_LANDMARKS, "sha256": sha256(encoded).hexdigest(),
            "scope": "Cuff endpoint pairs identified by location in the authors' 3D calibration picks. Compare with the projected automatic chord.",
            "cuffs": comparisons}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/tee-vision-front.npz"))
    parser.add_argument("--archive", action="store_true", help="Also generate the input frame's DXF and GLB")
    parser.add_argument("--compare-references", action="store_true", help="Generate the archive and compare net panels and the flat-pose GLB")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    generate_archive = args.archive or args.compare_references
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "frames").exists():
        parser.error("Choose a fresh output directory to reproduce all frames")
    rows, variants = [], {}
    for run, timestamps in CAPTURES.items():
        folder = f"green_tshirt/green_tshirt_fold_2025-07-28-{run}"
        transform = download(args.output, f"{run}-transform.json", f"{folder}/calibration/world_to_camera_transform.json")
        for frame, timestamp in zip(FRAME_IDS, timestamps, strict=True):
            key = f"{run}-{frame}"
            remote = f"{folder}/segment_pcds/pointcloud_{timestamp}_segmented.pcd"
            cloud = download(args.output, f"{key}.pcd", remote)
            output = args.output / "frames" / key
            reconstruct(cloud, output, source=DATASET, item_id="rgbench-green-tshirt", transform=transform,
                        right="y", down="x", weights=args.model if generate_archive and key == INPUT else None,
                        blender=args.blender)
            metadata = json.loads((output / "projection.json").read_text())
            rows.append({
                "frame": key,
                "data_path": remote,
                "cloud_sha256": metadata["cloud_sha256"],
                "world_to_camera_sha256": metadata["world_to_camera_sha256"],
                "plane_distance_p95_mm": metadata["plane_distance_p95_mm"],
                "flat_landmarks": metadata["flat_landmarks"],
                "observed_mm": {name: metadata[name] for name in ("lower_torso_width_mm", "shoulder_level_to_hem_mm")},
            })
            if key == INPUT:
                manifest, spec = load_capture(output / "session.json", model_path=args.model, blind=True)
                draft(spec)
                inference = manifest["inference"]
                weights_sha256 = inference["weights_sha256"]
                prior = inference["regression_before_scan_constraints_mm"]
                variants = {
                    "regression_only": {
                        "lower_torso_width_mm": prior["flat_chest_mm"],
                        "shoulder_level_to_hem_mm": round(prior["back_length_mm"] + spec.back_neck_depth_mm, 3),
                    },
                    "scan_constrained": {
                        "lower_torso_width_mm": spec.flat_chest_mm,
                        "shoulder_level_to_hem_mm": round(spec.back_length_mm + spec.back_neck_depth_mm, 3),
                    },
                }
    holdout = [row for row in rows if row["frame"] != INPUT]
    report = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "license": "CC BY 4.0",
        "weights_sha256": weights_sha256,
        "input_frame": INPUT,
        "scope": "One physical garment, three capture sequences. Differences compare visible dimensions against other scans; they are not tape or pattern accuracy.",
        "comparison_mm": {name: summarize(holdout, values) for name, values in variants.items()},
        "frames": rows,
    }
    if generate_archive:
        candidate = args.output / "frames" / INPUT / "archive"
        report["archive"] = check_archive(candidate)
    if args.compare_references:
        if len({row["world_to_camera_sha256"] for row in rows}) != 1:
            raise ValueError("The reproduction expects the same calibration across these scans")
        compare_flat(candidate, [args.output / f'{row["frame"]}.pcd' for row in holdout],
                     args.output / "flat-comparison", item_id="rgbench-green-tshirt",
                     setup="RGBench green_tshirt, three RealSense D455 recordings, CC BY 4.0, flat visible side",
                     transform=args.output / "18-21-20-transform.json", right="y", down="x", blender=args.blender)
        reference = download(args.output, "green_tshirt_5k.obj", REFERENCE_OBJ)
        compare_pattern(candidate, reference, args.output / "pattern-comparison", item_id="rgbench-green-tshirt",
                        source="RGBench green_tshirt_5k Style3D simulation mesh, CC BY 4.0, Sim2Real-v1",
                        order=("front", "back", "sleeve_right", "sleeve_left", "collar"))
        report["physical_cuff_reference"] = cuff_reference(args.output, next(row for row in rows if row["frame"] == INPUT))
    (args.output / "study.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison_mm"], indent=2))


if __name__ == "__main__":
    main()
