"""Measure an archived tee against independent public iPad LiDAR captures."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.capture import archive
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat import compare_flat
from superarchiveallclo.compare_rgbd import camera_from_flat_pose, compare as compare_rays
from superarchiveallclo.registered_ply import registered_depth
from superarchiveallclo.rgbd import back_project, color_seam_evidence, metric_projection, prepare
from superarchiveallclo.vision import picture_features

ITEM = "hifiai-exp1-green-crew-tee"
SOURCE = "https://doi.org/10.3390/app12105286"
MODEL = Path("models/tee-vision-front-compact.npz")
INPUT = "000031"
HOLDOUT = ("000034", "000040")
RESERVED = ("000037", "000039")
ASSETS = {
    "000031.jpg": ("1veWoE51cZd-ncLeWdHDz6vgsQgoDii6x", "305441ab13239bb60c4c2e51496264e3d5a483de9a9a29c104fa5ce2af0e484b"),
    "000031.ply": ("1iYqVg7goHPmkWwhBxLlMO4ZXeo6wwYWM", "910921a4c3d0d39fa715941ad8f48dcd9892717526c6889fc68e9ab644761f2e"),
    "000034.jpg": ("1EWN9FfiSFjNNFu4PdyaYOR5w4cdubuWY", "a8b8d84aae4575b0047066ab74871603394424beeaf679f7bd103ee706042723"),
    "000034.ply": ("1nT8Zmj3LOo_uqUCNlrwqIYIJ490B_teg", "f5af24f9dda8d986778bfa85911c9d31cb7232626fcee5278e8996c932dfd3e8"),
    "000037.jpg": ("1N5mO4vSabTjiLxvvpOujR_vfU9hx4xp9", "c907687e73c4782250ef0de4d77b191726f614d51682f69c4d70d8b8b1e34c4c"),
    "000037.ply": ("1ImI6AavgSlGskQBbodqH57wf6JuV_Sh0", "5a00cee9b9e590ec7ba5e60f60da3d1f85576df0159f7f5fd3a9096e770dc2e1"),
    "000039.jpg": ("1FHlKJZj498Qaemcpyb2sDMWjBbn0YP6g", "72e4a8d36a1ea69e4d1d5a2c86618096b8393fb1fe707def585dee539f9b3c49"),
    "000039.ply": ("1pCRrPuEHtIxHWgGn0pUiZpGKyMtl4Cz2", "5839c0b1c81f994137c87398fa20f33be91a4548ce7b6f175676b899fbccc230"),
    "000040.jpg": ("1tyWfpmmfY-Ob0aMLInovvGQmjzR0Vf_F", "650139408c6049d24475a8f411a87109b36eb41df45c35734a0e3b7fb58d825e"),
    "000040.ply": ("17sF22YGwF0okrNi41ut42Es6SX2pThZN", "4e3a404eced33c538ce65947174f9b4426f4e0c274905d8f60daa82b48dce281"),
}


def pinned_assets(directory, source, identifiers):
    selected = {name: asset for name, asset in ASSETS.items() if Path(name).stem in identifiers}
    for filename, (drive_id, digest) in selected.items():
        local = (source or directory) / filename
        if local.is_file():
            raw = local.read_bytes()
        elif source is None:
            request = Request(f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t",
                              headers={"User-Agent": "SuperArchiveAllClone research"})
            with urlopen(request, timeout=35) as response:
                raw = response.read(4_000_001)
        else:
            raise ValueError(f"The supplied source folder needs the original {filename}")
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"{filename} differs from the published experiment 1 file")
        target = directory / filename
        if not target.is_file():
            target.write_bytes(raw)
    return {name: asset[1] for name, asset in selected.items()}


def prepare_frame(root, frame):
    encoded = (root / f"{frame}.ply").read_bytes()
    with Image.open(root / f"{frame}.jpg") as photo:
        rgb = ImageOps.exif_transpose(photo).convert("RGB")
    depth = registered_depth(encoded, rgb.size)
    photo = rgb.resize(depth.depth.shape[::-1], Image.Resampling.BOX)
    colors = np.asarray(photo, np.int16)
    # The green sample has chroma separation from the white support and
    # dark equipment at the edge. This separates the visible reference cloth.
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(colors[:, :, 1] - colors[:, :, 0] > 17, iterations=2))
    labels, _ = ndimage.label(mask)
    counts = np.bincount(labels[mask])
    if mask.mean() < .12 or mask.mean() > .55 or counts.max() < .97 * mask.sum():
        raise ValueError("Could not isolate a single complete green garment in the reference photo")
    mask = labels == counts.argmax()
    img_path, mask_path, depth_path = (root / f"{frame}-{suffix}" for suffix in ("rgb.png", "mask.png", "depth.npz"))
    photo.save(img_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    np.savez_compressed(depth_path, depth=depth.depth)
    visible = mask & np.isfinite(depth.depth)
    image, calibration = metric_projection(depth.depth, mask, depth.intrinsics, 2, auto_upright=True)
    _, silhouette = picture_features(image, {"calibration": {**calibration, "source": SOURCE},
                                             "mm_per_pixel": 2}, measure_body=True)
    origin = np.array(calibration["support_origin_camera_m"])
    basis = np.array(calibration["support_basis_camera"])
    points = (back_project(depth.depth, visible, depth.intrinsics) - origin) @ basis
    profile = silhouette.get("torso_profile")
    measured = {
        "outline_mm": silhouette["garment_box_mm"], "recognized_underarm_and_hem": profile,
        "color_seam_evidence": color_seam_evidence(rgb, mask, depth.intrinsics, origin, basis, points),
        "surface_height_relative_to_support_p05_p50_p95_mm": (-np.quantile(points[:, 2], [.95, .50, .05]) * 1000).round(3).tolist(),
        "support_plane_residual_p90_mm": calibration["support_plane_residual_p90_mm"],
        "foreground_points": len(points), "foreground_mask_sha256": sha256(mask_path.read_bytes()).hexdigest(),
        "geometry": depth.diagnostics,
    }
    return {
        "capture": {"rgb": img_path.name, "depth": depth_path.name, "depth_unit": "m", "mask": mask_path.name,
                    "intrinsics": depth.intrinsics},
        "calibration": calibration, "measurements": measured,
        "points": points, "original_sha256": sha256(encoded).hexdigest(),
    }


def write_cloud(path, points):
    with path.open("wb") as stream:
        stream.write((f"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                      f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n").encode())
        stream.write(points.astype("<f4").tobytes())
    return path


def evaluate(output, source, blender, *, fit_sleeve_span=False, fit_shoulder_seams=False, fit_armhole=False, holdout=HOLDOUT):
    data = output / "source"
    data.mkdir(parents=True)
    identifiers = (INPUT, *holdout)
    hashes = pinned_assets(data, source, identifiers)
    input_frame = prepare_frame(data, INPUT)
    capture = write_json(data / "registered.json", {
        "schema_version": 1, "item_id": ITEM, "source": f"HifiAI experiment 1, frame {INPUT}, metric optical depth in PLY; {SOURCE}",
        "views": {"front": {**input_frame["capture"], "detail_rgb": f"{INPUT}.jpg"}},
    })
    prepared = output / "registered"
    prepare(capture, prepared, auto_upright=True)
    session = prepared / "session.json"
    target = output / "archive"
    archive(session, target, blender, MODEL, blind=True, form="display_torso", fit_sleeve_span=fit_sleeve_span,
            fit_shoulder_seams=fit_shoulder_seams, fit_armhole=fit_armhole)
    frames = {INPUT: input_frame, **{number: prepare_frame(data, number) for number in holdout}}
    pattern = json.loads((target / "pattern/pattern.json").read_bytes())["parameters"]
    capture_state = json.loads((target / "capture.json").read_bytes())
    clouds = [write_cloud(data / f"held-out-{number}.pcd", frames[number]["points"]) for number in holdout]
    flat_path = output / "flat-comparison"
    flat = compare_flat(target, clouds, flat_path, item_id=ITEM, blender=blender,
                        setup="Separate iPad captures of the same green tee, white support, tabletop aligned")
    rays = {}
    for number, row in zip(holdout, flat["references"], strict=True):
        entry = frames[number]
        view = {key: value for key, value in entry["capture"].items() if key != "rgb"}
        calibration = entry["calibration"]
        rays[f"frame-{number}"] = {
            **view, "depth_sha256": sha256((data / view["depth"]).read_bytes()).hexdigest(),
            "mask_sha256": sha256((data / view["mask"]).read_bytes()).hexdigest(),
            "source_pcd_sha256": entry["original_sha256"],
            "model_to_camera_m": camera_from_flat_pose(calibration["support_origin_camera_m"],
                calibration["support_basis_camera"], row["reference_translation_mm"]),
            "pose_source": "Actual support plane and bounded translation at the measured scale",
        }
    independent = write_json(data / "reference.json", {"schema_version": 1, "item_id": ITEM, "views": rays})
    depth = compare_rays(target, independent, output / "depth-comparison", flat_comparison=flat_path, blender=blender)
    chest_checks = {}
    for number in holdout:
        profile = frames[number]["measurements"]["recognized_underarm_and_hem"]
        if profile is not None:
            observed = profile["chest_width_mm"]
            chest_checks[number] = {"measured_scan_projection_mm": observed,
                                    "pattern_minus_scan_mm": round(pattern["flat_chest_mm"] - observed, 3)}
    return write_json(output / "study.json", {
        "source": SOURCE, "item_id": ITEM, "input_frame": INPUT, "independent_frames": list(holdout),
        "input_model_sha256": sha256(MODEL.read_bytes()).hexdigest(),
        "pixel_and_metric_evidence": {key: value["measurements"] for key, value in frames.items()},
        "source_sha256": hashes,
        "pattern_candidate_mm": {field: pattern[field] for field in (
            "flat_chest_mm", "back_length_mm", "garment_shoulder_width_mm", "armhole_depth_mm", "sleeve_length_mm",
        )},
        **{key: capture_state[key] for key in ("sleeve_span_fit", "shoulder_seam_fit", "armhole_silhouette_fit")
           if key in capture_state},
        "held_out_flat_chest_projection": chest_checks,
        "archive": check_archive(target), "flat_mesh": flat, "depth_rays": depth,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Optional local HifiAI experiment 1 source folder")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--fit-sleeve-span", action="store_true", help="Fit both input cuff tips before comparing held-out scans")
    parser.add_argument("--fit-shoulder-seams", action="store_true", help="Use the high-resolution calibrated seam evidence")
    parser.add_argument("--fit-armhole", action="store_true", help="Compare sewable armholes with the input silhouette")
    parser.add_argument("--reserved-frames", action="store_true", help="Evaluate frames 000037 and 000039")
    args = parser.parse_args()
    print(evaluate(args.output, args.source, args.blender, fit_sleeve_span=args.fit_sleeve_span,
                   fit_shoulder_seams=args.fit_shoulder_seams, fit_armhole=args.fit_armhole,
                   holdout=RESERVED if args.reserved_frames else HOLDOUT))


if __name__ == "__main__":
    main()
