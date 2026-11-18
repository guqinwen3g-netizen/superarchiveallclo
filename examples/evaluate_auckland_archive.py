"""Reproduce the full archive and independent photo comparisons for museum tees."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from examples import evaluate_auckland_drape as drape
from examples import evaluate_auckland_tees as tees
from superarchiveallclo import photo_points, visual_hull
from superarchiveallclo.review import review

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "netball": {
        "reference": "auckland", "form": "display_torso", "holdout": "photo-01",
        "structure": ("sleeve_outer", "sleeve_inner"),
        "structure_scope": "Physical photos show the cuff fold and the interior seam. Seam allowances and original cuts are unmeasured.",
    },
    "stellenbosch": {
        "reference": "stellenbosch", "form": "torso_with_arms", "holdout": "photo-01",
        "structure": ("collar_inner", "material_label"),
        "structure_scope": "Physical photos show the inner neck rib join and tag. Fiber composition is documented; mechanical fabric properties are unmeasured.",
    },
}


def evaluate(output, blender, *, case_name="netball"):
    config = CASES[case_name]
    data = output / "input"
    data.mkdir(parents=True, exist_ok=True)
    sources = json.loads(tees.SOURCE.read_bytes())
    case = sources["cases"][case_name]
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
                     form=config["form"], photo_appearance=True)
    if checked["archive"]["status"] != "traceability_verified":
        raise ValueError("The held-out physical review did not produce a verified archive")
    garment = output / "review/archive"
    references = output / "reference"
    references.mkdir(exist_ok=True)
    drape_source = Path(__file__).with_name(f'{config["reference"]}-drape.json')
    markings = Path(__file__).with_name(f'{config["reference"]}-landmarks.json')
    manifest, calibration = drape.prepare(references, garment, json.loads(drape_source.read_bytes()))
    tees.write_json(references / "calibration.json", calibration)
    local = photo_points.compare(garment, manifest, markings, output / "landmarks", blender=blender)
    hull = visual_hull.compare(garment, manifest, output / "hull", blender=blender)
    if local["garment_glb_sha256"] != hull["garment_glb_sha256"]:
        raise ValueError("Local points and global silhouette must score the same delivered GLB")
    simulation = json.loads((garment / "model/simulation.json").read_bytes())
    structure = {}
    for view in config["structure"]:
        name = session["views"][view]["file"]
        source = case["assets"][name]
        structure[view] = {"photo": name, "sha256": source["sha256"], "source_url": source["catalogue_photo"]}
    return tees.write_json(output / "evidence.json", {
        "schema_version": 1, "item_id": item, "rights": rights,
        "weights_sha256": checked["weights_sha256"], "pattern_sha256": local["pattern_sha256"],
        "garment_glb_sha256": local["garment_glb_sha256"],
        "dimensions": checked["dimensions"], "archive_audit": local["archive_audit"],
        "construction": {
            "scope": config["structure_scope"],
            "photos": structure, "simulated_openings_mm": simulation["opening_perimeters_mm"],
        },
        "photo_reference": {
            "calibration": "reference/calibration.json", "hull": "hull/comparison.json",
            "local_points": "landmarks/comparison.json", "landmarks_sha256": sha256(markings.read_bytes()).hexdigest(),
            "holdout_projection": hull["views"][config["holdout"]]["glb"],
            "holdout_projections": {name: view["glb"] for name, view in hull["views"].items() if view["role"] == "held_out"},
            "local_point_summary": local["summary"],
            "local_point_regions": local["by_region"],
            "scope": "Actual mannequin photographs and sparse triangulated points under approximate camera poses.",
        },
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--case", choices=CASES, default="netball")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.output, args.blender, case_name=args.case)}")


if __name__ == "__main__":
    main()
