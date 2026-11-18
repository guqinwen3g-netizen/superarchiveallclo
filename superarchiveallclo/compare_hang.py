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
from .cloth_material import (blender_bending, mass_observation, material_bending, material_mass,
                             verify_bending_settings, verify_mass_settings)
from .compare_flat import capture_sleeve_angle
from .compare_shape import check_frame
from .hang_supports import observed_pitch, pendulum_forecast, rotate_about_support
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


def depth_error_stats(errors):
    if not len(errors):
        return None
    absolute = np.abs(errors)
    return {
        "signed_median_mm": round(float(np.median(errors)), 2),
        "absolute_median_mm": round(float(np.median(absolute)), 2),
        "absolute_p90_mm": round(float(np.quantile(absolute, 0.9)), 2),
    }


def compare_ray_depth(cloud, before, after, support_z_m, *, labels=("before", "after")):
    before_hits, after_hits = np.isfinite(before), np.isfinite(after)
    common = before_hits & after_hits

    def metrics(selection):
        paired = selection & common
        first, second = labels
        return {
            "measured_rays": int(selection.sum()),
            f"{first}_hit_rays": int((selection & before_hits).sum()),
            f"{second}_hit_rays": int((selection & after_hits).sum()),
            "common_hit_rays": int(paired.sum()),
            f"{first}_on_common_rays": depth_error_stats(before[paired]),
            f"{second}_on_common_rays": depth_error_stats(after[paired]),
        }

    band_mm = 100
    band_ids = np.floor((support_z_m - cloud[:, 2]) * 1000 / band_mm).astype(int)
    result = metrics(np.ones(len(cloud), dtype=bool))
    result["bands_by_height"] = [
        {"range_below_highest_support_mm": [int(band * band_mm), int((band + 1) * band_mm)],
         **metrics(band_ids == band)}
        for band in np.unique(band_ids)
    ]
    return result


def prepare_swing(source, folder, extrinsic, support_file, evidence_hashes, job):
    history = source["swing_history"]
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("Provide at least two pre-calibration scans for the swing history")
    observations, inputs = [], []
    for entry in history:
        name = entry["file"]
        raw = (folder / name).read_bytes()
        digest = sha256(raw).hexdigest()
        if digest in evidence_hashes:
            raise ValueError("Use distinct scans for swing calibration and scoring")
        evidence_hashes.add(digest)
        observations.append(raw)
        inputs.append({"file": name, "timestamp": float(entry["timestamp"]), "sha256": digest})
    observations.append(support_file)
    inputs.append({"file": source["support_calibration_scan"], "timestamp": float(source["support_timestamp"]),
                   "sha256": sha256(support_file).hexdigest()})
    time = np.asarray([sample["timestamp"] for sample in inputs])
    test_times = np.asarray(source["reference_timestamps"], dtype=float)
    if (test_times.shape != (len(source["reference_scans"]),) or not np.isfinite(np.r_[time, test_times]).all()
            or np.any(np.diff(time) <= 0) or np.any(np.diff(test_times) <= 0) or test_times[0] <= time[-1]):
        raise ValueError("List strictly increasing swing calibration and subsequent scoring timestamps")
    supports, camera = job["supports"], job["camera_world_m"]
    measured = [observed_pitch(in_world(read_pcd(raw), extrinsic), supports, camera) for raw in observations]
    angles, lengths, signs = np.asarray(measured).T
    forecast, velocity, period = pendulum_forecast(time - time[-1], angles, test_times - time[-1], lengths[-1])
    job["forecast_rotation_radians"] = -signs[-1] * forecast
    return {
        "method": "Short uniform-strip pendulum forecast about the measured support axis",
        "input_scans": [{**row, "pitch_toward_camera_deg": round(float(np.degrees(angle)), 4)}
                        for row, angle in zip(inputs, angles, strict=True)],
        "observed_length_mm": round(float(lengths[-1]) * 1000, 2),
        "period_seconds": round(period, 5), "initial_angular_velocity_deg_per_second": round(float(np.degrees(velocity)), 4),
        "scoring_timestamps": test_times.tolist(),
        "predicted_pitch_toward_camera_deg": np.degrees(forecast).round(4).tolist(),
    }


def compare_hang(archive, manifest, output, blender="blender", *, save_ray_errors=False):
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
    camera = in_world(np.zeros((1, 3)), extrinsic)[0]
    job = {"flat": rest, "camera_world_m": camera, "rest_contacts": contacts(source["rest_contacts_world_m"]),
           "supports": contacts(source["supports_world_m"])}
    mass = mass_observation(source["garment_mass"]) if "garment_mass" in source else material_mass(capture.get("material"))
    if mass is not None:
        job["garment_mass_kg"] = np.asarray(mass["kg"])
    bending = blender_bending(source["blender_bending"]) if "blender_bending" in source else material_bending(capture.get("material"))
    if bending is not None:
        job["blender_bending_stiffness"] = np.asarray(bending["stiffness"])
    angle, angle_source = capture_sleeve_angle(capture)
    report = {
        "schema_version": 1, "item_id": source["item_id"], "source": source["source"],
        "manifest_sha256": sha256(encoded).hexdigest(), "world_to_camera_sha256": sha256(extrinsic).hexdigest(),
        "rest_scan_sha256": sha256(rest_file).hexdigest(), "support_calibration_sha256": sha256(support_file).hexdigest(),
        "pattern_sha256": integrity["pattern_sha256"], "source_sleeve_angle_deg": angle,
        "source_sleeve_angle_basis": angle_source,
        "rest_contacts_world_m": job["rest_contacts"].tolist(), "supports_world_m": job["supports"].tolist(),
        "scope": "Fixed two-point suspension; absolute comparison in calibrated world coordinates",
        "camera_ray_depth": {
            "camera_world_m": camera.tolist(),
            "sign": "First mesh hit minus measured range from camera; positive values are farther from the camera",
            "scoring": "Both models are scored only on measured rays that hit both meshes. All hit counts are reported.",
        },
        "reference_scans": [],
    }
    if mass is not None:
        report["garment_mass"] = mass
    if bending is not None:
        report["blender_bending"] = bending
    if "swing_history" in source:
        report["swing_forecast"] = prepare_swing(source, manifest.parent, extrinsic, support_file, evidence_hashes, job)
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
        if mass is not None:
            verify_mass_settings(mass, report["support_metrics"].get("mass_settings"))
        if bending is not None:
            verify_bending_settings(bending, report["support_metrics"].get("bending_settings"))
        with np.load(workspace / "hang-metrics.npz", allow_pickle=False) as data:
            if save_ray_errors:
                ray_errors = bundle / "gravity-ray-errors.npz"
                keys = [row["id"] for row in report["reference_scans"]]
                np.savez_compressed(ray_errors, **{key: data[f"after_{key}_ray_depth_mm"] for key in keys})
                report["gravity_ray_errors"] = {"file": ray_errors.name, "sha256": sha256(ray_errors.read_bytes()).hexdigest()}
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
            support_height = float(job["supports"][:, 2].max())
            for number, row in enumerate(report["reference_scans"]):
                key = row["id"]
                cloud = job[key]
                row["camera_ray_depth"] = compare_ray_depth(
                    cloud, data[f"before_{key}_ray_depth_mm"], data[f"after_{key}_ray_depth_mm"], support_height,
                )
                if "swing_forecast" not in report:
                    continue
                rotation = job["forecast_rotation_radians"][number]
                vertices = rotate_about_support(data["after_vertices"], job["supports"], rotation)
                iou, overlay = compare_front(cloud, vertices, data["after_triangles"])
                image = f"{key}-forecast.png"
                Image.fromarray(overlay).save(bundle / image)
                try:
                    observed, _, _ = observed_pitch(cloud, job["supports"], camera)
                except ValueError:
                    measured_pitch = None
                else:
                    measured_pitch = round(float(np.degrees(observed)), 4)
                row["swing_forecast"] = {
                    "world_rotation_about_support_radians": float(rotation),
                    "measured_pitch_toward_camera_deg": measured_pitch,
                    "front_iou": iou, "overlay": image,
                    "ray_depth": compare_ray_depth(
                        cloud, data[f"after_{key}_ray_depth_mm"], data[f"forecast_{key}_ray_depth_mm"],
                        support_height, labels=("fixed", "forecast"),
                    ),
                }
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
