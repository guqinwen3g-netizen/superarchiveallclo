"""Evaluate an archived garment under measured two-point suspension."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from .check_archive import check_archive, glb_inventory
from .compare_flat import capture_sleeve_angle
from .compare_shape import check_frame
from .scan import in_world, read_pcd

GRID_M = 0.004


def contacts(row):
    points = np.asarray(row, dtype=float)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("Provide two finite XYZ contacts in metres")
    return points


def compare_front(cloud, vertices, triangles):
    scan = cloud[:, (1, 2)] * [1, -1]
    mesh = vertices[:, (1, 2)] * [1, -1]
    origin = np.minimum(scan.min(axis=0), mesh.min(axis=0)) - 2 * GRID_M
    high = np.maximum(scan.max(axis=0), mesh.max(axis=0)) + 2 * GRID_M
    width, height = np.ceil((high - origin) / GRID_M).astype(int) + 1
    check_frame(width, height)
    pix = np.rint((scan - origin) / GRID_M).astype(int)
    marked = np.zeros((height, width), dtype=bool)
    marked[pix[:, 1], pix[:, 0]] = True
    observed = ndimage.distance_transform_edt(~marked) <= 1.5
    observed = ndimage.binary_closing(observed, iterations=2)
    render = Image.new("1", (width, height))
    draw = ImageDraw.Draw(render)
    for face in ((mesh - origin) / GRID_M)[triangles]:
        draw.polygon([tuple(point) for point in face], fill=1)
    model = np.asarray(render, dtype=bool)
    intersection = observed & model
    union = observed | model
    map_colors = np.full((height, width, 3), (250, 248, 243), dtype=np.uint8)
    map_colors[observed & ~model] = (22, 119, 167)
    map_colors[model & ~observed] = (205, 86, 41)
    map_colors[intersection] = (49, 61, 60)
    return round(float(intersection.sum() / union.sum()), 4), map_colors


def reference_motion(scans):
    first_id, first = next(iter(scans.items()))
    first_tree = cKDTree(first)
    rows = []
    for key, cloud in list(scans.items())[1:]:
        forward = first_tree.query(cloud)[0]
        reverse = cKDTree(cloud).query(first)[0]
        distance = 1000 * np.concatenate((forward, reverse))
        rows.append({"reference": first_id, "scan": key,
                     "median_mm": round(float(np.median(distance)), 2),
                     "p90_mm": round(float(np.quantile(distance, 0.9)), 2)})
    return rows


def compare_hang(archive, manifest, output, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    integrity = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_text())
    encoded = manifest.read_bytes()
    source = json.loads(encoded)
    if (not isinstance(source, dict) or source.get("schema_version") != 1
            or capture["kind"] != "physical" or source.get("item_id") != capture["item_id"]):
        raise ValueError("Provide a schema_version 1 manifest for the same physical garment")
    names = source.get("reference_scans")
    provenance = source.get("source")
    if (not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names)
            or not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError("Provide independent suspended scans with source provenance")
    extrinsic = (manifest.parent / source["world_to_camera"]).read_bytes()
    rest_file = (manifest.parent / source["rest_scan"]).read_bytes()
    support_file = (manifest.parent / source["support_calibration_scan"]).read_bytes()
    evidence_hashes = {sha256(asset).hexdigest() for asset in (rest_file, support_file)}
    evidence_hashes.update(view.get("calibration", {}).get("cloud_sha256") for view in capture["views"].values())
    if len({sha256(asset).hexdigest() for asset in (rest_file, support_file)}) != 2:
        raise ValueError("Provide separate rest and support calibration scans")
    rest = in_world(read_pcd(rest_file), extrinsic)
    if np.quantile(np.abs(rest[:, 2] - np.median(rest[:, 2])), 0.95) > 0.035:
        raise ValueError("The resting registration scan must be flat and metric")
    job = {"flat": rest, "rest_contacts": contacts(source["rest_contacts_world_m"]),
           "supports": contacts(source["supports_world_m"])}
    angle, angle_source = capture_sleeve_angle(capture)
    report = {
        "schema_version": 1, "item_id": source["item_id"], "source": source["source"],
        "manifest_sha256": sha256(encoded).hexdigest(), "world_to_camera_sha256": sha256(extrinsic).hexdigest(),
        "rest_scan_sha256": sha256(rest_file).hexdigest(), "support_calibration_sha256": sha256(support_file).hexdigest(),
        "pattern_sha256": integrity["pattern_sha256"], "source_sleeve_angle_deg": angle,
        "source_sleeve_angle_basis": angle_source,
        "rest_contacts_world_m": job["rest_contacts"].tolist(), "supports_world_m": job["supports"].tolist(),
        "scope": "Fixed two-point suspension; absolute comparison in calibrated world coordinates",
        "reference_scans": [],
    }
    for number, name in enumerate(names, 1):
        encoded_cloud = (manifest.parent / name).read_bytes()
        digest = sha256(encoded_cloud).hexdigest()
        if digest in evidence_hashes:
            raise ValueError(f"Select a held-out suspended scan for {name}")
        evidence_hashes.add(digest)
        key = f"scan-{number:02d}"
        job[key] = in_world(read_pcd(encoded_cloud), extrinsic)
        report["reference_scans"].append({"id": key, "file": name, "sha256": digest, "points": len(job[key])})
    report["reference_motion_from_first_scan"] = reference_motion(
        {row["id"]: job[row["id"]] for row in report["reference_scans"]}
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name("gravity_pose.py")
    with TemporaryDirectory(dir=output.parent) as temp:
        workspace = Path(temp)
        bundle = workspace / "suspension"
        bundle.mkdir()
        snapshot = workspace / "pattern.json"
        pattern = (archive / "pattern/pattern.json").read_bytes()
        if sha256(pattern).hexdigest() != integrity["pattern_sha256"]:
            raise ValueError("The pattern changed during verification")
        snapshot.write_bytes(pattern)
        np.savez_compressed(workspace / "hang-input.npz", **job)
        result = run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
                      "--", str(snapshot), str(workspace), str(angle)], capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError("Suspension solve failed: " + (result.stdout + result.stderr)[-2800:])
        report["support_metrics"] = json.loads((workspace / "support-metrics.json").read_text())
        with np.load(workspace / "hang-metrics.npz", allow_pickle=False) as data:
            for variant, filename in (("before", "before-gravity.glb"), ("after", "gravity-pose.glb")):
                model_path = workspace / filename
                report[variant] = {**glb_inventory(model_path), "sha256": sha256(model_path.read_bytes()).hexdigest()}
                for row in report["reference_scans"]:
                    key = row["id"]
                    distance = data[f"{variant}_{key}"]
                    iou, overlay = compare_front(job[key], data[f"{variant}_vertices"], data[f"{variant}_triangles"])
                    image = f"{key}-{variant}.png"
                    Image.fromarray(overlay).save(bundle / image)
                    median, p90 = np.quantile(distance, [0.5, 0.9]).round(2).tolist()
                    row[variant] = {"point_to_surface_mm": {"median": median, "p90": p90},
                                    "front_iou": iou, "overlay": image}
                model_path.rename(bundle / filename)
        (bundle / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        report = compare_hang(args.archive, args.manifest, args.output, args.blender)
    except (ValueError, TypeError, KeyError, OSError, DXFError) as error:
        parser.error(str(error))
    for row in report["reference_scans"]:
        print(row["file"], row["before"]["point_to_surface_mm"], row["after"]["point_to_surface_mm"])


if __name__ == "__main__":
    main()
