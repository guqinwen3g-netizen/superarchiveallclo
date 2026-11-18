"""Reproduce the Auckland Museum photo-pair experiments from pinned originals."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from math import dist
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageOps

from superarchiveallclo.capture import archive, load_capture
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat_photos import compare
from superarchiveallclo.review import review

SOURCE = Path(__file__).with_name("auckland-tees.json")
USER_AGENT = "SuperArchiveAllClone Research (https://github.com/guqinwen3g-netizen/superarchiveallclo)"
PAIRED_WEIGHTS = Path("models/tee-vision.npz")
FRONT_WEIGHTS = Path("models/tee-vision-front.npz")


def write_json(path, body):
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return path


def source_photo(path, photo):
    data = path.read_bytes() if path.is_file() else None
    if data is None:
        with urlopen(Request(photo["url"], headers={"User-Agent": USER_AGENT}), timeout=25) as response:
            data = response.read()
    if sha256(data).hexdigest() != photo["sha256"]:
        raise ValueError(f"{path.name}: photo does not match the pinned source")
    if not path.is_file():
        path.write_bytes(data)


def tape_intervals(samples):
    report = {}
    for view, marks in samples.items():
        rows = []
        for start, end in zip((15, 20, 30), (20, 30, 32), strict=True):
            length = (end - start) * 25.4
            spacing = dist(marks[str(start)], marks[str(end)])
            rows.append({"inch_marks": [start, end], "pixels": round(spacing, 3),
                         "mm_per_pixel": round(length / spacing, 5)})
        report[view] = rows
    return report


def neckline_sensitivity(case, data, model):
    session = deepcopy(case["session"])
    for view, point in case["alternate_neck_points_px"].items():
        session["views"][view]["scale"]["points"][0] = point
    source = write_json(data / "alternate-neck-edge.json", session)
    capture, _ = load_capture(source, model_path=model, blind=True)
    basis = {view.get("scale", {}).get("basis_field") for view in capture["views"].values()}
    return {
        "mm_per_pixel": {name: view["mm_per_pixel"] for name, view in capture["views"].items() if "scale" in view},
        "held_out_measurements_mm": {
            name: {"predicted_mm": capture["measurements"][name]["value"], "catalogue_mm": row["value"],
                   "absolute_error_mm": round(abs(capture["measurements"][name]["value"] - row["value"]), 3)}
            for name, row in session["measurements"].items() if name not in basis
        },
    }


def shoulder_overlay(archive_root, output):
    capture = json.loads((archive_root / "capture.json").read_bytes())
    figures = []
    for name, view in capture["inference"]["scale_evidence"].items():
        seams = view.get("projected_shoulder_seams")
        if seams is None:
            continue
        with Image.open(archive_root / capture["views"][name]["file"]) as photo:
            image = ImageOps.exif_transpose(photo).convert("RGB")
            image.thumbnail((720, 720), Image.Resampling.BOX)
        if sha256(image.tobytes()).hexdigest() != view["processed_image_sha256"]:
            raise ValueError(f"Processed {name} photo changed since shoulder measurement")
        left, top, right, _ = view["garment_crop_px"]
        lower = max(y for side in ("image_left", "image_right") for _, y in seams[side]["trace_crop_px"])
        figure = image.crop((left, max(0, top - 6), right + 1, top + int(lower) + 12))
        offset_y = min(6, top)
        draw = ImageDraw.Draw(figure)
        for side in ("image_left", "image_right"):
            trace = seams[side]
            draw.line([(x, y + offset_y) for x, y in trace["trace_crop_px"]], fill="#e41842", width=3)
            x, y = trace["shoulder_crop_px"]
            y += offset_y
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#e41842")
        figures.append((name, seams["projected_width_mm"], figure))
    if not figures:
        return None
    sheet = Image.new("RGB", (max(figure.width for _, _, figure in figures),
                              sum(figure.height + 23 for _, _, figure in figures)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for name, width, figure in figures:
        draw.text((6, y + 4), f"{name}   {width:.2f} mm", fill="black")
        sheet.paste(figure, (0, y + 23))
        y += figure.height + 23
    sheet.save(output)
    return output.name


def evaluate_case(case, output, blender, rights, *, model=PAIRED_WEIGHTS, measure_body=False, measure_sleeve=False):
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    for filename, photo in case["assets"].items():
        source_photo(data / filename, photo)
    session = dict(case["session"])
    if measure_body:
        session["body_profile"] = "paired_silhouette"
    if measure_sleeve:
        session["sleeve_profile"] = "paired_cuff_span"
    reference = write_json(data / "reference.json", {
        "schema_version": 1, "item_id": session["item_id"], "kind": session["kind"], "setup": case["setup"],
        "views": {name: session["views"][name] for name in ("front", "back")},
    })
    paired = write_json(data / "paired.json", session)
    if session["measurements"]:
        dataset = write_json(data / "dataset.json", {"schema_version": 1, "samples": [{
            "session": paired.name, "group": session["item_id"], "split": "test", "rights": rights,
        }]})
        evidence = review(dataset, session["item_id"], model, output / "review", blender)
        result = {"review": evidence}
        candidate = output / "review/archive"
    else:
        candidate = output / "paired"
        archive(paired, candidate, blender, model, blind=True)
        result = {"archive": check_archive(candidate)}
    if candidate.is_dir():
        result["flat_layout"] = compare(candidate, reference, output / "flat-layout", blender)
        if measure_body:
            result["shoulder_seam_overlay"] = shoulder_overlay(candidate, output / "shoulder-seams.png")
    if "alternate_neck_points_px" in case:
        result["alternate_neck_edge"] = neckline_sensitivity({**case, "session": session}, data, model)

    if "tape_checks" in case:
        single_session = {
            **session, "input_mode": "front_only", "views": {"front": session["views"]["front"]},
        }
        single_session.pop("body_profile", None)
        single_session.pop("sleeve_profile", None)
        front = write_json(data / "front-only.json", single_session)
        single = output / "single-front"
        archive(front, single, blender, FRONT_WEIGHTS, blind=True)
        result["single_front_archive"] = check_archive(single)
        result["single_front_flat_layout"] = compare(single, reference, output / "single-front-flat-layout", blender)
        result["tape_scale_checks"] = tape_intervals(case["tape_checks"])
    write_json(output / "study.json", result)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    document = json.loads(SOURCE.read_bytes())
    parser.add_argument("--case", choices=document["cases"], help="Run only the specified physical garment")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--model", type=Path, default=PAIRED_WEIGHTS, help="Paired photo model weights")
    parser.add_argument("--body-profile", action="store_true", help="Measure chest, hem and projected shoulder seams")
    parser.add_argument("--sleeve-profile", action="store_true", help="Constrain sleeve length using both cuff spans")
    args = parser.parse_args()
    rights = f'{document["photo_attribution"]}, {document["license"]}, {document["license_url"]}'
    for name, case in document["cases"].items():
        if args.case is None or args.case == name:
            evaluate_case(case, args.output / name, args.blender, rights, model=args.model, measure_body=args.body_profile,
                          measure_sleeve=args.sleeve_profile)
            print(f"Wrote {args.output / name / 'study.json'}")


if __name__ == "__main__":
    main()
