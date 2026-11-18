"""Compare the sewn flat GLB with scaled front and back photographs."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageOps

from .capture import capture_photo_hashes, view_calibration
from .check_archive import check_archive, glb_inventory
from .compare_flat import run_flat_pose, sleeve_angle_from_cuffs
from .compare_shape import compare_masks, metric_mask, project
from .flat_measurements import flat_landmarks
from .vision import picture_silhouette


def source_sleeve_angle(archive, view):
    with Image.open(archive / view["file"]) as photo:
        _, mask, _, scale, _ = picture_silhouette(photo, view)
    torso = mask[round(0.64 * len(mask)):round(0.87 * len(mask))]
    width = float(np.median(torso.sum(axis=1)) * scale)
    cuffs = flat_landmarks(mask, scale, width).get("cuffs", {})
    angle = sleeve_angle_from_cuffs(cuffs)
    if angle is not None:
        return angle, "Mean of cuff directions from the reconstruction front photo"
    return 23.0, "Standard flat-lay placement"


def compare(archive, reference, output, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    integrity = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_bytes())
    encoded = reference.read_bytes()
    record = json.loads(encoded)
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or not record.get("item_id") or record["item_id"] != capture["item_id"]
            or record.get("kind") != capture["kind"]):
        raise ValueError("Provide a schema_version 1 reference for the same garment and capture kind")
    setup, views = record.get("setup"), record.get("views")
    if not isinstance(setup, str) or not setup.strip():
        raise ValueError("Identify the layout setup and scale source")
    if not isinstance(views, dict) or not views or not views.keys() <= {"front", "back"}:
        raise ValueError("Provide front and/or back flat reference photos")
    angle, angle_basis = source_sleeve_angle(archive, capture["views"]["front"])
    pattern = (archive / "pattern/pattern.json").read_bytes()
    if sha256(pattern).hexdigest() != integrity["pattern_sha256"]:
        raise ValueError("Pattern changed during archive verification")
    raw_inputs = capture_photo_hashes(capture["views"])
    processed_inputs = {view["processed_image_sha256"]
                        for view in capture.get("inference", {}).get("scale_evidence", {}).values()}
    references = {}
    for name, view in views.items():
        raw = (reference.parent / view["file"]).read_bytes()
        digest = sha256(raw).hexdigest()
        with Image.open(BytesIO(raw)) as photo:
            calibration = view_calibration(view, ImageOps.exif_transpose(photo).size, name)
            processed, silhouette, _, scale, _ = picture_silhouette(photo, calibration, outer_only=True)
        was_input = digest in raw_inputs or sha256(processed.tobytes()).hexdigest() in processed_inputs
        observed, span = metric_mask(silhouette, scale)
        references[name] = (observed, span, {
            "photo_sha256": digest, "role": "reconstruction_input" if was_input else "separate_photo",
            "calibration": calibration, "processed_mm_per_pixel": round(scale, 6),
        })

    report = {
        "schema_version": 1, "item_id": record["item_id"], "reference_setup": setup,
        "reference_manifest_sha256": sha256(encoded).hexdigest(), "pattern_sha256": integrity["pattern_sha256"],
        "scope": "Outer contours of a sewn flat GLB compared with photographs; no depth reference",
        "alignment": "Translate the top and horizontal center of silhouette bounds; no fitted scale or rotation",
        "sleeve_angle_below_horizontal_deg": angle, "sleeve_angle_source": angle_basis, "views": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        working = Path(temp)
        bundle = working / "comparison"
        bundle.mkdir()
        model, result = run_flat_pose(pattern, working, {}, angle, blender)
        report["flat_glb"] = {**glb_inventory(model), "sha256": sha256(model.read_bytes()).hexdigest()}
        with np.load(result, allow_pickle=False) as mesh:
            grid, span = project(mesh["vertices"], mesh["triangles"], "top")
        for name, (observed, observed_span, details) in references.items():
            metrics, overlay = compare_masks(observed, grid if name == "front" else grid[:, ::-1])
            image = f"{name}-overlay.png"
            Image.fromarray(overlay).save(bundle / image)
            report["views"][name] = {
                **details, **metrics, "overlay": image, "reference_width_height_mm": observed_span.round(2).tolist(),
                "model_width_height_mm": span.round(2).tolist(),
                "model_minus_reference_width_height_mm": (span - observed_span).round(2).tolist(),
            }
        model.rename(bundle / model.name)
        (bundle / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        result = compare(args.archive, args.reference, args.output, args.blender)
    except (ValueError, KeyError, TypeError, OSError, DXFError) as error:
        parser.error(str(error))
    print(json.dumps(result["views"], indent=2))


if __name__ == "__main__":
    main()
