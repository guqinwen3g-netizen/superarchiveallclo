"""Reproduce the end-to-end archive review for the 2017.66.22 museum garment."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from examples import evaluate_auckland_drape as drape
from examples import evaluate_auckland_tees as tees
from superarchiveallclo import photo_points, visual_hull
from superarchiveallclo.review import review

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = Path(__file__).with_name("auckland-landmarks.json")


def evaluate(output, blender):
    data = output / "input"
    data.mkdir(parents=True)
    sources = json.loads(tees.SOURCE.read_bytes())
    case = sources["cases"]["netball"]
    session = {**case["session"], "body_profile": "paired_silhouette"}
    item = session["item_id"]
    for filename, asset in case["assets"].items():
        tees.source_photo(data / filename, asset)
    capture = tees.write_json(data / "paired.json", session)
    rights = f'{sources["photo_attribution"]}, {sources["license"]}, {sources["license_url"]}'
    dataset = tees.write_json(data / "dataset.json", {"schema_version": 1, "samples": [{
        "session": capture.name, "group": item, "split": "test", "rights": rights,
    }]})
    checked = review(dataset, item, ROOT / tees.PAIRED_WEIGHTS, output / "review", blender,
                     form="display_torso", photo_appearance=True)
    if checked["archive"]["status"] != "traceability_verified":
        raise ValueError("The held-out physical review did not produce a verified archive")
    garment = output / "review/archive"
    references = output / "reference"
    references.mkdir()
    manifest, calibration = drape.prepare(references, garment, json.loads(drape.SOURCE.read_bytes()))
    tees.write_json(references / "calibration.json", calibration)
    local = photo_points.compare(garment, manifest, LANDMARKS, output / "landmarks", blender=blender)
    hull = visual_hull.compare(garment, manifest, output / "hull", blender=blender)
    if local["garment_glb_sha256"] != hull["garment_glb_sha256"]:
        raise ValueError("Local points and global silhouette must score the same delivered GLB")
    simulation = json.loads((garment / "model/simulation.json").read_bytes())
    structure = {}
    for view in ("sleeve_outer", "sleeve_inner"):
        name = session["views"][view]["file"]
        source = case["assets"][name]
        structure[view] = {"photo": name, "sha256": source["sha256"], "source_url": source["catalogue_photo"]}
    return tees.write_json(output / "evidence.json", {
        "schema_version": 1, "item_id": item, "rights": rights,
        "weights_sha256": checked["weights_sha256"], "pattern_sha256": local["pattern_sha256"],
        "garment_glb_sha256": local["garment_glb_sha256"],
        "dimensions": checked["dimensions"], "archive_audit": local["archive_audit"],
        "construction": {
            "scope": "Physical photos show the cuff fold and the interior seam. Seam allowances and original cuts are unmeasured.",
            "photos": structure, "simulated_openings_mm": simulation["opening_perimeters_mm"],
        },
        "photo_reference": {
            "calibration": "reference/calibration.json", "hull": "hull/comparison.json",
            "local_points": "landmarks/comparison.json", "landmarks_sha256": sha256(LANDMARKS.read_bytes()).hexdigest(),
            "holdout_projection": hull["views"]["photo-01"]["glb"],
            "local_point_summary": local["summary"],
            "scope": "Actual mannequin photographs and sparse triangulated points under approximate camera poses.",
        },
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.output, args.blender)}")


if __name__ == "__main__":
    main()
