"""Reproduce the 2017.66.22 museum mannequin visual-hull experiment."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import dist
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

from examples.evaluate_auckland_tees import source_photo, write_json
from superarchiveallclo.visual_hull import compare

SOURCE = Path(__file__).with_name("auckland-drape.json")


def pole_measurement(photo, bounds, resize_factor):
    x, y, right, bottom = (value * resize_factor for value in bounds)
    on_pole = np.asarray(photo.convert("L"), dtype=float)[y:bottom, x:right]
    edges = gaussian_filter1d(on_pole, 1.25, order=1, axis=1)
    middle = edges.shape[1] // 2
    left = np.argmin(edges[:, :middle], axis=1)
    right = middle + np.argmax(edges[:, middle:], axis=1)
    row = np.arange(len(left))
    reliable = (edges[row, left] < -6) & (edges[row, right] > 6)
    if not reliable.any():
        raise ValueError("Could not detect both edges of the common stand pole")
    width = (right - left)[reliable]
    center = (right + left)[reliable] / 2 + x
    return {"width_original_px": float(np.median(width)), "center_original_px": float(np.median(center)),
            "width_p10_p90_original_px": np.quantile(width, [0.10, 0.90]).tolist(),
            "sample_rows": int(reliable.sum())}


def load_photo(directory, source, photo_size):
    path = directory / source["file"]
    source_photo(path, source)
    photo = Image.open(path)
    if photo.size != tuple(photo_size):
        photo.close()
        raise ValueError(f"{path.name}: source dimensions differ from the annotation")
    return photo


def prepare(data, archive, source):
    photo_size, mask_size = source["photo_size_px"], source["annotation_size_px"]
    resize_factor = photo_size[0] // mask_size[0]
    if [size * resize_factor for size in mask_size] != photo_size:
        raise ValueError("The annotation must preserve the camera frame aspect ratio")
    record = source["calibration"]
    with load_photo(data, record, photo_size) as photo:
        baseline_pole = pole_measurement(photo, record["pole_roi"], resize_factor)
        fixed_distance_focal = float(photo.getexif().get_ifd(34665)[37386])
    rulers = [{**ruler, "mm_per_original_pixel": ruler["length_mm"] / dist(*ruler["points_original_px"])}
              for ruler in record["rulers"]]
    base_scale = median(row["mm_per_original_pixel"] for row in rulers)
    pole_mm = baseline_pole["width_original_px"] * base_scale
    details = {"photo": record["file"], "photo_sha256": record["sha256"], "rulers": rulers,
               "scale_mm_per_original_pixel": base_scale, "pole": baseline_pole,
               "pole_diameter_estimate_mm": pole_mm, "views": {}}
    views = {}
    for frame, view in source["views"].items():
        with load_photo(data, view, photo_size) as photo:
            pole = pole_measurement(photo, view["pole_roi"], resize_factor)
            focal = float(photo.getexif().get_ifd(34665)[37386])
        scale = pole_mm * resize_factor / pole["width_original_px"]
        name = f"photo-{int(frame):02d}"
        mask_path = f"{name}-mask.png"
        image = Image.new("1", tuple(mask_size))
        ImageDraw.Draw(image).polygon(view["silhouette_px"], fill=1)
        views[name] = {"photo": view["file"], "photo_sha256": view["sha256"], "mask": mask_path,
                       "role": view["role"], "axis": view["axis"], "mm_per_pixel": scale,
                       "origin_px": [pole["center_original_px"] / resize_factor, view["neck_rim_y_px"]]}
        if "occlusion_px" in view:
            hidden = Image.new("1", tuple(mask_size))
            ImageDraw.Draw(hidden).polygon(view["occlusion_px"], fill=1)
            occlusion_path = f"{name}-occluded.png"
            hidden.save(data / occlusion_path)
            image.paste(0, (0, 0), hidden)
            views[name]["occlusion_mask"] = occlusion_path
        image.save(data / mask_path)
        details["views"][name] = {"pole": pole, "exif_focal_length_mm": focal,
                                   "mm_per_mask_pixel": scale, "same_distance_exif_only_mm_per_mask_pixel":
                                       base_scale * resize_factor * fixed_distance_focal / focal}
    pattern = json.loads((archive / "pattern/pattern.json").read_bytes())
    length = pattern["measurements"]["back_center_length_mm"]
    # Existing preview form's top neck rim, defined in simulate.add_form.
    model_origin = [0, 0, 1520 + 100 * length / 680]
    manifest = write_json(data / "hull-reference.json", {
        "schema_version": 1, "kind": "physical", "item_id": source["item_id"],
        "setup": source.get("setup", "Near-cardinal photographs of the museum's mannequin. Scale transfers from two rulers "
                            "through the common stand pole. Orthographic cameras use the pole center and neck rim height. "
                            "The GLB uses the generic archive form; its neck rim is registered by translation."),
        "model_origin_mm": model_origin, "views": views,
    })
    return manifest, details


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="The paired Auckland 2017.66.22 archive")
    parser.add_argument("--torso-only-archive", type=Path, help="Optional resimulation of the same pattern")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    pattern = Path("pattern/pattern.json")
    if (args.torso_only_archive
            and (args.archive / pattern).read_bytes() != (args.torso_only_archive / pattern).read_bytes()):
        parser.error("Both simulated forms must use the same paper pattern")
    data = args.output / "data"
    data.mkdir(parents=True, exist_ok=True)
    configuration = SOURCE.read_bytes()
    source = json.loads(configuration)
    manifest, calibration = prepare(data, args.archive, source)
    baseline = compare(args.archive, manifest, args.output / "volume", blender=args.blender)
    report = {
        "schema_version": 1,
        "source_configuration_sha256": sha256(configuration).hexdigest(),
        "calibration": calibration,
        "comparison": baseline,
    }
    if args.torso_only_archive:
        report["torso_only_comparison"] = compare(
            args.torso_only_archive, manifest, args.output / "torso-only-volume", blender=args.blender,
        )
    write_json(args.output / "study.json", report)
    print(f"Wrote {args.output / 'study.json'}")


if __name__ == "__main__":
    main()
