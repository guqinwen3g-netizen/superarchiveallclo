"""Run and cross-check a single held-out garment review."""

from argparse import ArgumentParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import ezdxf

from .capture import archive
from .check_archive import check_archive
from .compare_shape import compare
from .dataset import evaluate, load_dataset, original_sensor_hashes
from .vision import TeeVision


def verify_prediction(capture, sample, model, result):
    if capture["item_id"] != sample.item_id or capture["kind"] != sample.kind:
        raise ValueError("Archive item does not match the labelled test garment")
    if {name: view["sha256"] for name, view in capture["views"].items()} != sample.photo_hashes:
        raise ValueError("Source photographs changed between scoring and reconstruction")
    if original_sensor_hashes(capture["views"]) != sample.source_hashes:
        raise ValueError("The raw RGB-D evidence changed between scoring and reconstruction")
    inference = capture["inference"]
    if inference["weights_sha256"] != model.checksum or not inference["reference_measurements_withheld"]:
        raise ValueError("Archive used different weights or received the reference measurements")
    sources = {}
    for profile in ("body_profile", "sleeve_profile"):
        if inference.get(profile) != result.get(profile):
            raise ValueError(f"Archived {profile} differs from the scored profile")
        sources.update(result.get(profile, {}).get("sources", {}))
    for name, value in result["inferred_mm"].items():
        actual = capture["measurements"][name]
        expected = sources.get(name, "vision_model")
        if actual["source"] != expected or actual["value"] != value:
            raise ValueError(f"{name}: archived prediction differs from the scored prediction")


def review(dataset, item_id, weights, output, blender, shape_reference=None, *,
           form="torso_with_arms", photo_appearance=False):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    samples, dataset_checksum = load_dataset(dataset)
    sample = next((item for item in samples if item.item_id == item_id), None)
    if sample is None or sample.split != "test":
        raise ValueError(f"{item_id}: select an item from the independent test split")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        working = Path(temp)
        bundle = working / "review"
        bundle.mkdir()
        # Reuse a single weight snapshot for scoring and the reconstruction process.
        snapshot = working / "model.npz"
        snapshot.write_bytes(weights.read_bytes())
        model = TeeVision(snapshot)
        result = evaluate([sample], model)[sample.kind]["items"][0]
        report = {
            "schema_version": 1,
            "dataset_sha256": dataset_checksum,
            "weights_sha256": model.checksum,
            "capture_kind": sample.kind,
            "dimensions": result,
            "archive": {"status": "not_generated"},
            "shape": {"status": "awaiting_reference"},
            "structure": {"status": "awaiting_manual_review"},
            "scope": "Measurements and silhouette evidence for review, subject to physical structure and material verification.",
        }
        if result["status"] == "pattern_passed":
            candidate = bundle / "archive"
            archive(sample.session, candidate, blender, snapshot, blind=True, form=form, photo_appearance=photo_appearance)
            capture = json.loads((candidate / "capture.json").read_text(encoding="utf8"))
            verify_prediction(capture, sample, model, result)
            report["archive"] = {
                "status": "traceability_verified", "directory": "archive",
                **check_archive(candidate),
            }
            report["structure"]["additional_capture_views"] = [
                name for name in capture["views"] if name not in ("front", "back")
            ]
            if shape_reference is not None:
                try:
                    shape_result = compare(candidate, shape_reference, bundle / "shape", blender)
                except (ValueError, KeyError, TypeError, OSError) as reason:
                    report["shape"] = {"status": "comparison_failed", "reason": str(reason)}
                else:
                    report["shape"] = {
                        "status": "measured", "report": "shape/comparison.json",
                        "views": {name: {
                            "intersection_over_union": view["intersection_over_union"],
                            "model_minus_reference_width_height_mm": view["model_minus_reference_width_height_mm"],
                            "contour_distance_mm": view["contour_distance_mm"],
                        } for name, view in shape_result["views"].items()},
                    }
        else:
            report["shape"]["status"] = "not_compared"

        (bundle / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf8")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--item", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--shape-reference", type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        report = review(args.dataset, args.item, args.model, args.output, args.blender, args.shape_reference)
    except (ValueError, KeyError, TypeError, OSError, ezdxf.DXFError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output / 'review.json'}")
    print(f"Archive {report['archive']['status']}, shape {report['shape']['status']}")


if __name__ == "__main__":
    main()
