"""Reconstruct real RISE T-shirt pairs on their independently sized photography table."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from struct import unpack_from
from urllib.request import Request, urlopen
from zlib import decompress

import json
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.capture import archive, load_capture
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat_photos import compare
from superarchiveallclo.rectify import homography, prepare, project, rectangle


SOURCE = Path(__file__).with_name("secondhand-grid.json")
ARCHIVE_SIZE = 26_812_054_849
DOWNLOAD = "https://zenodo.org/records/13788681/files/clothing_v3.zip?download=1"
CELL_MM = 100
MODEL = Path("models/tee-vision-varied.npz")
FRONT_MODEL = Path("models/tee-vision-front-compact.npz")


def download_member(asset, output):
    cached = output.is_file()
    if cached:
        data = output.read_bytes()
    else:
        name = asset["member"]
        start = asset["byte_offset"]
        end = start + 30 + len(name.encode()) + asset["compressed_size"] + 300
        header = {"Range": f"bytes={start}-{end}", "User-Agent": "SuperArchiveAllClone research"}
        with urlopen(Request(f"{DOWNLOAD}&part={start}-{end}", headers=header), timeout=25) as response:
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
                raise ValueError("Repository did not return the pinned ZIP byte range")
            encoded = response.read(end - start + 2)
        if len(encoded) != end - start + 1 or encoded[:4] != b"PK\x03\x04":
            raise ValueError("Unexpected ZIP member header")
        method = unpack_from("<H", encoded, 8)[0]
        name_length, extra_length = unpack_from("<HH", encoded, 26)
        if encoded[30:30 + name_length].decode() != name or method not in (0, 8):
            raise ValueError("ZIP member does not match the pinned filename and compression method")
        offset = 30 + name_length + extra_length
        packed = encoded[offset:offset + asset["compressed_size"]]
        if len(packed) != asset["compressed_size"]:
            raise ValueError("Incomplete ZIP member")
        data = decompress(packed, -15) if method == 8 else packed
    if sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"{output.name} differs from the publisher's pinned release")
    if not cached:
        output.write_bytes(data)


def fit_line(ridge, anchor, probes, anchor_at):
    centers = anchor[0] + (np.asarray(probes) - anchor_at[0]) * (anchor[1] - anchor[0]) / np.ptp(anchor_at)
    found = []
    for scan, center in zip(probes, centers, strict=True):
        first = round(center) - 7
        signal = np.median(ridge[scan - 3:scan + 4, first:first + 15], axis=0)
        k = int(signal.argmax())
        if 0 < k < len(signal) - 1 and signal[k] >= 16:
            a, b, c = signal[k - 1:k + 2]
            delta = 0.5 * (a - c) / (a + c - 2 * b) if a + c < 2 * b else 0.0
            found.append([scan, first + k + delta, float(signal[k])])
    if len(found) < 5:
        raise ValueError("Insufficient visible support for a grid line")
    points = np.array(found)
    slope, offset = np.polyfit(points[:, 0], points[:, 1], 1)
    if np.max(np.abs(points[:, 1] - slope * points[:, 0] - offset)) > 4:
        raise ValueError("Grid line intersections move too far from a single straight line")
    return slope, offset


def intersection(vertical, horizontal):
    a, b = vertical
    c, d = horizontal
    x, y, z = np.cross((1, -a, -b), (-c, 1, -d))
    return [float(x / z), float(y / z)]


def grid_geometry(image, setup):
    gray = np.asarray(image.convert("L"), dtype=float)
    ridge = ndimage.gaussian_filter(gray, 2.5) - gray
    vertical = {int(key): fit_line(ridge, marks, (75, 160, 255, 350, 445, 535, 615), (70, 620))
                for key, marks in setup["vertical"].items()}
    horizontal = {int(key): fit_line(ridge.T, marks, (110, 150, 220, 275, 1000, 1060, 1115, 1180), (150, 1115))
                  for key, marks in setup["horizontal"].items()}
    end_x = max(vertical)
    end_y = max(horizontal)
    # For this station the neckline points toward raster right. Start its
    # upright rectangle at the original top right grid corner.
    vertices = [(end_x, 0), (end_x, end_y), (0, end_y), (0, 0)]
    corners = [intersection(vertical[x], horizontal[y]) for x, y in vertices]
    width, height = CELL_MM * end_y, CELL_MM * end_x
    inverse = homography(np.asarray(corners), rectangle(width, height))
    withheld = [(x, y) for x in vertical for y in horizontal if (x, y) not in vertices]
    points_px = np.array([intersection(vertical[x], horizontal[y]) for x, y in withheld])
    truth = CELL_MM * np.array([(y, end_x - x) for x, y in withheld])
    errors = np.linalg.norm(project(inverse, points_px) - truth, axis=1)
    return {"width_mm": width, "height_mm": height, "corners_px": corners}, inverse, {
        "reference_nodes": len(vertices), "withheld_nodes": len(withheld),
        "withheld_intersection_median_mm": round(float(np.median(errors)), 3),
        "withheld_intersection_p90_mm": round(float(np.quantile(errors, .90)), 3),
        "withheld_nodes_px": points_px.round(3).tolist(), "withheld_nodes_mm": truth.tolist(),
    }


def garment_mask(image, inverse, width, height):
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    gray = rgb.mean(axis=2)
    chroma = np.ptp(rgb, axis=2)
    # Saturated or dark opaque fabric can be isolated from this gray table.
    # Opening removes the table's thin dark lines before keeping the garment.
    candidate = (gray < 61) | (chroma > 42)
    candidate = ndimage.binary_opening(candidate, structure=np.ones((7, 7), dtype=bool))
    y, x = np.where(candidate)
    xy_mm = project(inverse, np.column_stack((x, y)))
    inside = ((xy_mm > 8) & (xy_mm < np.array([width, height]) - 8)).all(axis=1)
    groups, count = ndimage.label(candidate)
    areas = np.bincount(groups[y[inside], x[inside]], minlength=count + 1)[1:]
    if not areas.size or areas.max() < image.width * image.height * .05:
        raise ValueError("No sufficiently large garment separated from the gray grid")
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(groups == areas.argmax() + 1, iterations=4))
    if areas.max() < areas.sum() * .90:
        raise ValueError("Garment foreground is divided into incompatible components")
    return mask


def reference_view(path, setup, output):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if list(image.size) != setup["raster_px"]:
        raise ValueError("Original picture size does not match this documented table setup")
    reference, inverse, evidence = grid_geometry(image, setup)
    width, height = reference["width_mm"], reference["height_mm"]
    mask = garment_mask(image, inverse, width, height)
    y, x = np.where(mask & ~ndimage.binary_erosion(mask))
    points = project(inverse, np.column_stack((x, y)))
    lower, upper = points.min(axis=0), points.max(axis=0)
    to_photo = np.linalg.inv(inverse)
    for margin in (35, 25, 15, 8):
        field = np.r_[np.floor(lower - margin), np.ceil(upper + margin)]
        footprint = project(to_photo, rectangle(*(field[2:] - field[:2])) + field[:2])
        if (footprint >= 0).all() and (footprint <= image.size).all():
            reference["field_mm"] = field.tolist()
            break
    else:
        raise ValueError("The projected field cannot include the complete garment and remain inside the photo")
    Image.fromarray(mask.astype(np.uint8) * 255).save(output)
    return {"rgb": path.name, "mask": output.name, "reference_plane": reference}, evidence


def run_case(case, setup, output, weights, blender):
    raw = output / "input"
    raw.mkdir(parents=True, exist_ok=True)
    views, geometry = {}, {}
    for name, member in case["assets"].items():
        label = name == "labels"
        path = raw / (name + (".json" if label else ".jpg"))
        download_member(member, path)
        if label:
            continue
        views[name], geometry[name] = reference_view(path, setup, raw / f"{name}-mask.png")
    labels = json.loads((raw / "labels.json").read_text(encoding="utf-8-sig"))
    source = write_json(raw / "session.json", {
        "schema_version": 1, "item_id": case["item_id"], "kind": "physical",
        "source": "RISE Second-Hand Fashion v3 station 1. Publisher documents the 100 mm square table grid.",
        "body_profile": "paired_silhouette", "views": views,
    })
    output_rectified = output / "rectified"
    prepared = prepare(source, output_rectified, mm_per_pixel=1.0)
    session = output_rectified / "session.json"
    observations, spec = load_capture(session, model_path=weights, blind=True)
    metrics = {}
    for name, view in observations["inference"]["scale_evidence"].items():
        silhouette = view["garment_box_mm"]
        profile = view["torso_profile"] or {}
        metrics[name] = {
            "silhouette_width_mm": silhouette[0], "silhouette_height_mm": silhouette[1],
            "flat_chest_projection_mm": profile.get("chest_width_mm"),
            "hem_projection_mm": profile.get("hem_width_mm"),
        }
    target = output / "archive"
    archive(session, target, blender, weights, blind=True, form="display_torso", photo_appearance=True)
    reference = write_json(output_rectified / "reference.json", {
        "schema_version": 1, "item_id": case["item_id"], "kind": "physical",
        "setup": "Separate front/back exposures on the RISE measured 100 mm square grid",
        "views": prepared["views"],
    })
    views_only = {key: value for key, value in prepared.items() if key != "body_profile"}
    front_only = write_json(output_rectified / "front-only.json", {
        **views_only, "input_mode": "front_only", "views": {"front": prepared["views"]["front"]},
    })
    _, front_spec = load_capture(front_only, model_path=FRONT_MODEL, blind=True)
    reference_chest = metrics["back"]["flat_chest_projection_mm"]
    flat_report = compare(target, reference, output / "flat-comparison", blender)
    shared = [key for key in metrics["front"] if all(row[key] is not None for row in metrics.values())]
    return write_json(output / "evidence.json", {
        "item_id": case["item_id"], "asset_sha256": {k: a["sha256"] for k, a in case["assets"].items()},
        "source": "https://zenodo.org/records/13788681", "license": "CC BY 4.0",
        "paired_weights_sha256": sha256(weights.read_bytes()).hexdigest(),
        "publisher_metadata": {key: labels.get(key) for key in ("type", "cut", "category", "material")},
        "calibration": geometry, "view_measurements_mm": metrics,
        "front_back_projection_delta_mm": {key: round(abs(metrics["front"][key] - metrics["back"][key]), 3)
                                           for key in shared},
        "front_only_check": {
            "weights_sha256": sha256(FRONT_MODEL.read_bytes()).hexdigest(),
            "inferred_chest_mm": front_spec.flat_chest_mm, "withheld_back_chest_projection_mm": reference_chest,
            "candidate_minus_back_projection_mm": (
                round(front_spec.flat_chest_mm - reference_chest, 3) if reference_chest is not None else None
            ),
        },
        "pattern_candidate_mm": {key: getattr(spec, key) for key in (
            "flat_chest_mm", "flat_hem_mm", "back_length_mm", "garment_shoulder_width_mm", "sleeve_length_mm",
        )},
        "archive": check_archive(target), "flat_comparison": flat_report,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--blender", default="blender")
    cases = json.loads(SOURCE.read_bytes())
    parser.add_argument("--case", choices=cases["cases"])
    args = parser.parse_args()
    for name, value in cases["cases"].items():
        if args.case is None or args.case == name:
            print(run_case(value, cases["setups"][value["setup"]], args.output / name, args.model, args.blender))


if __name__ == "__main__":
    main()
