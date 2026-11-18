"""Verify that registered garment depth stays metric when tabletop pixels are masked."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image

from examples.evaluate_auckland_tees import write_json
from examples.evaluate_iri_tee import FRAMES, ITEM, WEIGHTS, checked, download, level_reference
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat import compare_flat
from superarchiveallclo.compare_rgbd import camera_from_flat_pose, compare as compare_depth
from superarchiveallclo.rgbd import plane_basis, prepare, registered_regions
from superarchiveallclo.scan import read_pcd


def frame_data(root, frame):
    cloud_hash = FRAMES[frame]["pcd"][2]
    rgb_hash = FRAMES[frame]["img"][2]
    pcd = checked((root / f"{frame}.pcd").read_bytes(), cloud_hash)
    checked((root / f"{frame}.png").read_bytes(), rgb_hash)
    cloud = read_pcd(pcd, organized=True)
    with Image.open(root / f"{frame}-mask.png") as image:
        mask = np.asarray(image.convert("L")) > 127
    valid = np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)
    return cloud, mask, valid


def ray_reference(data, frame, cloud, mask, valid, evidence, flat, intrinsics):
    depth = data / "held-out-raw-depth.npz"
    np.savez_compressed(depth, depth=np.where(mask & valid, cloud[:, :, 2], np.nan))
    transform = camera_from_flat_pose(evidence["leveled_origin_camera_m"], evidence["leveled_basis_camera"],
                                     flat["references"][0]["reference_translation_mm"])
    return write_json(data / "independent-depth.json", {"schema_version": 1, "item_id": ITEM, "views": {frame: {
        "depth": depth.name, "depth_unit": "m", "depth_sha256": sha256(depth.read_bytes()).hexdigest(),
        "source_pcd_sha256": FRAMES[frame]["pcd"][2], "intrinsics": intrinsics, "model_to_camera_m": transform,
        "pose_source": "Table and hem rotation from the reference plus translation aligned to the flat view. Scale unchanged.",
    }}})


def evaluate(output, source, blender):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    frames = "n00700", "n00764"
    captured, checked_frame = frames
    data = output / "source"
    data.mkdir(parents=True)
    if source is None:
        download(data, frames)
    else:
        for frame in frames:
            for suffix in (".png", ".pcd", "-mask.png"):
                name = frame + suffix
                copyfile(source / name, data / name)

    front, mask, valid = frame_data(data, captured)
    calibration, other_mask, other_valid = frame_data(data, checked_frame)
    _, ring, _ = registered_regions(other_mask, other_valid)
    origin, basis, residual = plane_basis(calibration[ring])
    # The dataset's organized XYZ uses the pinhole registration recovered
    # independently in evaluate_iri_camera. Store Z from those real points.
    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}
    camera = {"source": f"Lateral tabletop pixels in the separate {checked_frame} Kinect capture",
              "origin_camera_m": origin.tolist(), "basis_camera": basis.tolist()}
    shared = {"rgb": f"{captured}.png", "mask": f"{captured}-mask.png", "intrinsics": intrinsics, "depth_unit": "m"}
    record = {"schema_version": 1, "item_id": ITEM,
              "source": "IRI UPC-CSIC Clothing Part Dataset metric Kinect coordinates"}
    np.savez_compressed(data / "unmasked.npz", depth=np.where(valid, front[:, :, 2], np.nan))
    np.savez_compressed(data / "foreground-only.npz", depth=np.where(valid & mask, front[:, :, 2], np.nan))
    before = write_json(data / "depth-with-table.json", {**record, "views": {"front": {**shared, "depth": "unmasked.npz"}}})
    after = write_json(data / "depth-without-table.json", {**record, "views": {"front": {
        **shared, "depth": "foreground-only.npz", "support_plane": camera,
    }}})
    full = prepare(before, output / "automatic-plane")
    masked = prepare(after, output / "provided-plane", weights=WEIGHTS, blender=blender)
    archive = output / "provided-plane/archive"
    scan, scan_evidence = level_reference(data, checked_frame)
    flat = output / "flat-comparison"
    score = compare_flat(archive, [scan], flat, blender=blender, item_id=ITEM,
                         setup="Independent foreground of IRI n00764. Only its surrounding table calibrated the input plane.")
    rays = ray_reference(data, checked_frame, calibration, other_mask, other_valid, scan_evidence, score, intrinsics)
    depth = compare_depth(archive, rays, output / "ray-comparison", flat_comparison=flat, blender=blender)
    return write_json(output / "study.json", {
        "source_xyz_sha256": {frame: FRAMES[frame]["pcd"][2] for frame in frames},
        "scope": "A controlled foreground-only crop of measured input. The independent scan supplies a table plane and withheld cloth.",
        "fixture_plane": camera, "fixture_points": int(ring.sum()), "fixture_residual_p90_mm": round(residual * 1000, 3),
        "automatic_plane": full, "provided_plane": masked, "archive_audit": check_archive(archive),
        "held_out_foreground": scan_evidence, "same_pattern_flat_comparison": score,
        "same_pattern_ray_comparison": depth,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Use the already downloaded n00700 and n00764 assets")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.output, args.source, args.blender)}")


if __name__ == "__main__":
    main()
