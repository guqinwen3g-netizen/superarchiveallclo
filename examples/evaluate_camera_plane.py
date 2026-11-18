"""Archive a real RGB-only capture using a separately measured camera-to-table plane."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
from scipy import ndimage

from examples.evaluate_auckland_tees import write_json
from examples.evaluate_iri_tee import FRAMES, ITEM, WEIGHTS, checked, download, level_reference
from examples.evaluate_measured_support import frame_data, ray_view
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat import compare_flat, sleeve_angle_from_cuffs
from superarchiveallclo.compare_rgbd import compare as compare_depth
from superarchiveallclo.compare_shape import GRID_MM, compare_masks, crop_surface, metric_mask
from superarchiveallclo.flat_measurements import flat_landmarks
from superarchiveallclo.rectify import prepare
from superarchiveallclo.rgbd import plane_basis, registered_regions
from superarchiveallclo.scan import project
from superarchiveallclo.vision import picture_silhouette


def blue_foreground(rgb):
    """Isolate the blue shirt on this neutral table using only RGB."""
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    blue = ndimage.binary_opening((b - r > 34) & (b - g > 18) & (b > 85), iterations=2)
    blue = ndimage.binary_fill_holes(ndimage.binary_closing(blue, iterations=2))
    labels, _ = ndimage.label(blue)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if not counts.max():
        raise ValueError("The blue shirt is not visible in this RGB frame")
    return labels == counts.argmax()


def evaluate(output, source, blender):
    captured, calibration_frame = "n00700", "n00764"
    data = output / "source"
    data.mkdir(parents=True)
    frames = captured, calibration_frame
    if source is None:
        download(data, frames)
    else:
        for frame in frames:
            for suffix in (".png", ".pcd", "-mask.png"):
                name = frame + suffix
                copyfile(source / name, data / name)

    calibration, calibration_mask, calibration_valid = frame_data(data, calibration_frame)
    _, ring, _ = registered_regions(calibration_mask, calibration_valid)
    origin, axes, residual = plane_basis(calibration[ring])
    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}
    camera = {
        "source": f"Tabletop XYZ in IRI UPC-CSIC {calibration_frame}, SHA-256 {FRAMES[calibration_frame]['pcd'][2]}",
        "image_size_px": [640, 480], "intrinsics": intrinsics,
        "origin_camera_m": origin.tolist(), "basis_camera": axes.tolist(),
        "field_mm": [-450, -390, 540, 404],
    }
    picture = data / f"{captured}.png"
    checked(picture.read_bytes(), FRAMES[captured]["img"][2])
    with Image.open(picture) as photo:
        mask = blue_foreground(np.asarray(photo.convert("RGB")))
    matte = data / "front-from-rgb.png"
    Image.fromarray(mask).save(matte)
    input_file = write_json(data / "photograph.json", {
        "schema_version": 1, "item_id": ITEM,
        "source": f"IRI UPC-CSIC Clothing Part Dataset, {captured} RGB-only garment input",
        "views": {"front": {"rgb": picture.name, "mask": matte.name, "camera_plane": camera}},
    })
    rectified = output / "rectified"
    session = prepare(input_file, rectified, weights=WEIGHTS, blender=blender, mm_per_pixel=GRID_MM)
    archive = rectified / "archive"

    # These garment XYZ observations are first read as a reference after archiving.
    observed, annotated, valid = frame_data(data, captured)
    held = (observed[annotated & valid] - origin) @ axes
    scan_image, _ = project(held, "x", "y", GRID_MM)
    raw_scan_mask = crop_surface(np.asarray(scan_image) < 128)
    view = session["views"]["front"]
    with Image.open(rectified / view["file"]) as photo:
        _, cutout, _, step, _ = picture_silhouette(photo, {**view, "mm_per_pixel": GRID_MM}, outer_only=True)
    photo_mask, photo_span = metric_mask(cutout, step)
    observed_span = np.array(raw_scan_mask.shape[::-1]) * GRID_MM
    projection, overlay = compare_masks(raw_scan_mask, photo_mask)
    Image.fromarray(overlay).save(output / "photo-vs-xyz.png")
    torso = cutout[round(0.64 * len(cutout)):round(0.87 * len(cutout))]
    cuffs = flat_landmarks(cutout, step, float(np.median(torso.sum(axis=1)) * step)).get("cuffs", {})
    angle = sleeve_angle_from_cuffs(cuffs)

    levels = [level_reference(data, frame) for frame in frames]
    flat = compare_flat(archive, [path for path, _ in levels], output / "flat-comparison", item_id=ITEM,
                        setup="Same-frame withheld cloth XYZ plus a repeat front capture, each with its own leveled plane",
                        sleeve_angle=angle, blender=blender)
    depth_frames = {captured: (observed, annotated, valid), calibration_frame: (calibration, calibration_mask, calibration_valid)}
    ray_views = {}
    for frame, (_, setup), row in zip(frames, levels, flat["references"], strict=True):
        ray_views[frame] = ray_view(data / f"{frame}-withheld-depth.npz", frame, *depth_frames[frame], setup, row, intrinsics)
    ray_manifest = write_json(data / "independent-depth.json", {"schema_version": 1, "item_id": ITEM, "views": ray_views})
    depth = compare_depth(archive, ray_manifest, output / "ray-comparison", flat_comparison=output / "flat-comparison",
                          blender=blender)
    return write_json(output / "study.json", {
        "camera_plane": camera, "fixture_points": int(ring.sum()), "fixture_residual_p90_mm": round(residual * 1000, 3),
        "colour_mask_vs_source_annotation_iou": round(float((mask & annotated).sum() / (mask | annotated).sum()), 5),
        "rgb_plane_vs_withheld_xyz_outline": {**projection, "rgb_width_height_mm": photo_span.tolist(),
                                               "xyz_width_height_mm": observed_span.tolist(),
                                               "rgb_minus_xyz_width_height_mm": (photo_span - observed_span).tolist()},
        "camera_pose_note": "The plane and registered intrinsics are measured separately. Cloth thickness is ignored in RGB rectification.",
        "archive_audit": check_archive(archive), "flat_photo_sleeve_angle_deg": angle,
        "same_pattern_flat_comparison": flat, "same_pattern_ray_comparison": depth,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Use already downloaded IRI n00700 and n00764 captures")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.output, args.source, args.blender)}")


if __name__ == "__main__":
    main()
