"""Check the reference cameras using mannequin landmarks apart from the garment."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo import photo_points

FIXTURE = Path(__file__).with_name("stellenbosch-fixture.json")
LANDMARKS = Path(__file__).with_name("stellenbosch-landmarks.json")


def similarity(reference, observed):
    x, y = reference.T
    one, zero = np.ones(len(x)), np.zeros(len(x))
    design = np.vstack((np.column_stack((x, -y, one, zero)), np.column_stack((y, x, zero, one))))
    values, _, rank, _ = np.linalg.lstsq(design, observed.T.ravel(), rcond=None)
    if rank != 4:
        raise ValueError("Frontal registration needs distinct support points")
    a, b, tx, ty = values
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])


def fit_fixture(orthographic, fixture):
    origin = fixture["reference_view"]
    anchor = orthographic[origin]
    marks = fixture["markers"]
    fronts = dict(orthographic)
    diagnostics = {"front": {}, "side": {}}
    for name in fixture["front_views"]:
        ref = np.array([marks[key][origin] for key in fixture["front_fit"]])
        observed = np.array([marks[key][name] for key in fixture["front_fit"]])
        registered = similarity(ref, observed)
        fronts[name] = registered @ anchor
        original = orthographic[name] @ np.linalg.pinv(anchor)
        errors = {}
        for key in fixture["front_checks"]:
            point = np.r_[marks[key][origin], 1]
            measured = np.array(marks[key][name])
            errors[key] = {
                "prior_px": round(float(np.linalg.norm((original @ point)[:2] - measured)), 3),
                "registered_px": round(float(np.linalg.norm((registered @ point)[:2] - measured)), 3),
            }
        diagnostics["front"][name] = {"image_transform": registered.tolist(), "holdout": errors}

    side = dict(fronts)
    for name, roles in fixture["side_views"].items():
        base = [marks[key][origin][1] for key in roles["fit"]]
        values = [marks[key][name][1] for key in roles["fit"]]
        gain, offset = np.linalg.solve(np.column_stack((base, np.ones(2))), values)
        original = orthographic[name][1, 2] / anchor[1, 2]
        original_offset = orthographic[name][1, 3] - original * anchor[1, 3]
        side[name] = orthographic[name].copy()
        side[name][1] = gain * anchor[1] + offset * anchor[2]
        check = marks[roles["check"]]
        start, expected = check[origin][1], check[name][1]
        diagnostics["side"][name] = {"row_gain": float(gain), "row_offset": float(offset),
                                    "holdout": {"prior_px": round(abs(original * start + original_offset - expected), 3),
                                                "registered_px": round(abs(gain * start + offset - expected), 3)}}
    return fronts, side, diagnostics


def evaluate(baseline, revised, output, blender):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    reference = baseline / "reference/hull-reference.json"
    source = json.loads(reference.read_bytes())
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    if fixture["item_id"] != source["item_id"]:
        raise ValueError("Fixture landmarks refer to a different garment")
    metadata, size = source["views"], tuple(fixture["image_size_px"])
    orthographic = {}
    for name, view in metadata.items():
        photo = (reference.parent / view["photo"]).read_bytes()
        camera = photo_points.read_point_view(photo, view, size)
        orthographic[name] = photo_points.projection_matrix(camera)
    fronts, sides, diagnostics = fit_fixture(orthographic, fixture)
    output.mkdir(parents=True)
    results = {}
    for method, matrices, description in (
        ("front-fixture", fronts, "Frontal similarity transfer from mannequin landmarks, with original side cameras"),
        ("front-and-side-fixture", sides, "Frontal transfer and vertical side registration using the articulated arms"),
    ):
        camera = write_json(output / f"{method}.json", {
            "schema_version": 1, "item_id": source["item_id"], "world_unit": "mm", "image_size_px": list(size),
            "source": f"{description}. Ruler and pole still provide the original absolute scale.",
            "views": {name: {"photo_sha256": metadata[name]["photo_sha256"], "image_from_world": matrix.tolist()}
                      for name, matrix in matrices.items()},
        })
        results[method] = {}
        for case_name, case in (("before", baseline), ("revised-sleeve", revised)):
            comparison = photo_points.compare(case / "review/archive", reference, LANDMARKS,
                                              output / method / case_name, blender=blender, camera_solution=camera)
            results[method][case_name] = {
                "pattern_sha256": comparison["pattern_sha256"], "glb_sha256": comparison["garment_glb_sha256"],
                "summary": comparison["summary"]["supplied_projection"],
            }
    return write_json(output / "study.json", {
        "item_id": source["item_id"], "fixture_sha256": sha256(fixture_bytes).hexdigest(),
        "reference_sha256": sha256(reference.read_bytes()).hexdigest(), "fixture_holdouts": diagnostics,
        "garment_holdouts": results,
        "scope": "Independent support observations improve relative cameras; the physical depth and absolute scale retain the pole-transfer assumption.",
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("revised", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.baseline, args.revised, args.output, args.blender)}")


if __name__ == "__main__":
    main()
