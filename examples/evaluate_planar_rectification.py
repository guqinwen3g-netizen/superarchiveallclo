"""Test simulated camera tilt on two pinned, ruler-scaled museum photographs."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import json

import numpy as np
from PIL import Image, ImageOps

from examples.evaluate_auckland_tees import PAIRED_WEIGHTS, SOURCE, source_photo, write_json
from superarchiveallclo.capture import capture_observations, view_calibration
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.rectify import homography, prepare, project, rectangle
from superarchiveallclo.vision import picture_features


QUADS = {
    "front": [[120, 275], [1730, 92], [1930, 1615], [160, 1710]],
    "back": [[95, 105], [1900, 315], [1760, 1690], [180, 1560]],
}


def measured_geometry(session):
    capture, _ = capture_observations(session, model_path=PAIRED_WEIGHTS, blind=True)
    return capture["inference"]["scale_evidence"], {
        name: reading["value"] for name, reading in capture["measurements"].items() if reading["source"] != "assumed"
    }


def run(output, photos, blender):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    published = json.loads(SOURCE.read_bytes())
    case = published["cases"]["peace"]
    data = output / "input"
    data.mkdir(parents=True)
    original = deepcopy(case["session"])
    tilted = deepcopy(original)
    tilted["kind"] = "synthetic_fixture"
    plane = {"schema_version": 1, "item_id": original["item_id"], "kind": "synthetic_fixture",
             "source": "Virtual corners projected from the original taped museum images; artificial camera tilt",
             "views": {}}
    raw_boxes = {}
    for name, view in original["views"].items():
        filename = view["file"]
        target = data / filename
        if photos is not None:
            copyfile(photos / filename, target)
        source_photo(target, case["assets"][filename])
        with Image.open(target) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        known = view_calibration(view, image.size, name)
        corners = np.asarray(QUADS[name], dtype=float)
        to_tilted = homography(rectangle(*image.size), corners)
        to_original = np.linalg.inv(to_tilted)
        to_original /= to_original[2, 2]
        distorted = image.transform((2010, 1820), Image.Transform.PERSPECTIVE, to_original.ravel()[:8],
                                    Image.Resampling.BICUBIC, fillcolor=image.getpixel((10, 10)))
        filename = f"{name}-simulated-tilt.png"
        distorted.save(data / filename)
        tilted_scale = {**view["scale"], "points": project(to_tilted, np.asarray(view["scale"]["points"])).tolist()}
        tilted["views"][name] = {"file": filename, "scale": tilted_scale}
        plane["views"][name] = {"rgb": filename, "reference_plane": {
            "corners_px": QUADS[name], "width_mm": image.width * known["mm_per_pixel"],
            "height_mm": image.height * known["mm_per_pixel"],
        }}
        try:
            _, readings = picture_features(distorted, view_calibration(tilted["views"][name], distorted.size, name))
        except ValueError as error:
            raw_boxes[name] = {"capture_rejected": str(error)}
        else:
            raw_boxes[name] = {"garment_box_mm": readings["garment_box_mm"]}

    original_features, original_mm = measured_geometry(write_json(data / "original.json", original))
    tilted_session = write_json(data / "tilted-single-ruler.json", tilted)
    try:
        _, tilted_mm = measured_geometry(tilted_session)
        baseline = {"delta_from_original_model_mm": {name: round(value - original_mm[name], 3)
                                                     for name, value in tilted_mm.items()}}
    except ValueError as error:
        baseline = {"capture_rejected": str(error)}
    fixed = output / "rectified"
    prepare(write_json(data / "planar.json", plane), fixed, weights=PAIRED_WEIGHTS, blender=blender,
            photo_appearance=True)
    archive = fixed / "archive"
    audited = check_archive(archive)
    fixed_features, fixed_mm = measured_geometry(fixed / "session.json")
    correction = {name: round(value - original_mm[name], 3) for name, value in fixed_mm.items()}
    report = {
        "scope": "Artificial camera tilt of real photos; agreement with the source model, not physical accuracy",
        "licensed_source": {"attribution": published["photo_attribution"], "license": published["license"],
                            "license_url": published["license_url"], "assets": case["assets"]},
        "camera_control_quads_px": QUADS,
        "reference_image_scale_source": "Actual 20-to-30 inch tape marks in the input photographs",
        "original_model_mm": original_mm,
        "original_garment_boxes_mm": {name: row["garment_box_mm"] for name, row in original_features.items()},
        "tilted_with_one_scale_segment": {"view_results": raw_boxes, **baseline},
        "rectified_with_virtual_plane": {
            "garment_boxes_mm": {name: row["garment_box_mm"] for name, row in fixed_features.items()},
            "model_mm": fixed_mm, "delta_from_original_model_mm": correction,
            "median_absolute_delta_from_original_mm": round(float(np.median(np.abs(list(correction.values())))), 3),
            "max_absolute_delta_from_original_mm": round(max(map(abs, correction.values())), 3),
        },
        "final_archive_audit": audited,
    }
    return write_json(output / "study.json", report)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--photos", type=Path, help="Use local copies of the pinned museum originals")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(run(args.output, args.photos, args.blender))


if __name__ == "__main__":
    main()
