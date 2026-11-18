"""Compare metric orthographic GLB silhouettes with a garment reference."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from math import ceil, isfinite
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import ezdxf
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .check_archive import check_archive

GRID_MM = 2.0
VIEWS = {"front": (0, 1), "back": (0, -1), "left": (1, 1), "right": (1, -1)}
# Keep accidental misreported units from allocating an unbounded image.
MAX_PIXELS = 16_000_000


def read_mesh(path, blender):
    script = Path(__file__).resolve().with_name("read_garment_glb.py")
    with TemporaryDirectory() as temp:
        output = Path(temp) / "mesh.npz"
        process = run([
            blender, "-b", "--factory-startup", "--python-exit-code", "1",
            "--python", str(script), "--", str(path.resolve()), str(output),
        ], capture_output=True, text=True)
        if process.returncode or not output.is_file():
            raise ValueError("Could not import garment GLB: " + (process.stdout + process.stderr)[-1600:])
        with np.load(output, allow_pickle=False) as archive:
            vertices, triangles = archive["vertices"], archive["triangles"]
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite GLB garment vertex")
    return vertices, triangles


def check_frame(width, height):
    if width < 2 or height < 2 or width * height > MAX_PIXELS:
        raise ValueError("Invalid silhouette size. Check the input scale and dimensions")


def crop_surface(mask):
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError("Empty silhouette")
    return mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]


def project(vertices, triangles, view):
    axis, direction = VIEWS[view]
    plane = np.column_stack((vertices[:, axis] * direction, -vertices[:, 2])) * 1000
    span = np.ptp(plane, axis=0)
    width, height = (ceil(axis_span / GRID_MM) + 3 for axis_span in span)
    check_frame(width, height)
    grid = (plane - plane.min(axis=0)) / GRID_MM + 1
    mask = Image.new("1", (width, height))
    paint = ImageDraw.Draw(mask)
    for triangle in grid[triangles]:
        paint.polygon([tuple(p) for p in triangle], fill=1)
    return crop_surface(np.asarray(mask, dtype=bool)), span


def reference_mask(path, mm_per_pixel):
    if type(mm_per_pixel) not in (int, float) or not isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("Reference mm_per_pixel must be positive and finite")
    encoded = path.read_bytes()
    with Image.open(BytesIO(encoded)) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Reference garment mask must have a black margin on all four sides")
    cropped = crop_surface(mask)
    height, width = cropped.shape
    span = np.asarray((width, height)) * mm_per_pixel
    target_width, target_height = [round(s / GRID_MM) for s in span]
    check_frame(target_width, target_height)
    grid = Image.fromarray(cropped.astype(np.uint8) * 255).resize(
        (target_width, target_height), Image.Resampling.BOX,
    )
    return crop_surface(np.asarray(grid) > 127), span, sha256(encoded).hexdigest()


def compare_masks(observed, projected):
    h, w = (max(a, b) + 8 for a, b in zip(observed.shape, projected.shape, strict=True))
    check_frame(w, h)

    def align(mask):
        canvas = np.zeros((h, w), dtype=bool)
        height, width = mask.shape
        left = (w - width) // 2
        canvas[4:4 + height, left:left + width] = mask
        return canvas

    observed, projected = align(observed), align(projected)
    overlay = np.full((h, w, 3), (250, 248, 243), dtype=np.uint8)
    overlay[observed & ~projected] = (22, 119, 167)
    overlay[projected & ~observed] = (205, 86, 41)
    overlay[observed & projected] = (49, 61, 60)
    obs_edge = observed & ~ndimage.binary_erosion(observed)
    proj_edge = projected & ~ndimage.binary_erosion(projected)
    offsets = np.r_[
        ndimage.distance_transform_edt(~obs_edge)[proj_edge],
        ndimage.distance_transform_edt(~proj_edge)[obs_edge],
    ] * GRID_MM
    return {
        "intersection_over_union": round(float((observed & projected).sum() / (observed | projected).sum()), 4),
        "contour_distance_mm": {
            "median": round(float(np.median(offsets)), 2),
            "p90": round(float(np.quantile(offsets, 0.9)), 2),
        },
    }, overlay


def compare(archive, reference, output, blender):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    archive_check = check_archive(archive)
    captured = json.loads((archive / "capture.json").read_text(encoding="utf8"))
    encoded = reference.read_bytes()
    source = json.loads(encoded)
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError("Expected reference schema_version 1")
    if source.get("kind") != captured["kind"]:
        raise ValueError("Reference and archive must declare the same capture kind")
    item_id = source.get("item_id")
    if not isinstance(item_id, str) or not item_id.strip() or item_id != captured["item_id"]:
        raise ValueError("Reference must identify the same garment as the archive")
    if not isinstance(source.get("setup"), str) or not source["setup"].strip():
        raise ValueError("Record the pose and camera calibration setup used for the reference")
    views = source.get("views")
    if not isinstance(views, dict) or not views or not views.keys() <= VIEWS.keys():
        raise ValueError("Provide one or more front, back, left or right reference views")

    model = archive / "model/garment.glb"
    model_hash = sha256(model.read_bytes()).hexdigest()
    vertices, triangles = read_mesh(model, blender)
    if sha256(model.read_bytes()).hexdigest() != model_hash:
        raise ValueError("Garment GLB changed during comparison")
    report = {
        "schema_version": 1,
        "item_id": item_id, "capture_kind": captured["kind"],
        "reference_setup": source["setup"],
        "reference_manifest_sha256": sha256(encoded).hexdigest(),
        "pattern_sha256": archive_check["pattern_sha256"], "garment_glb_sha256": model_hash,
        "alignment": "Top and horizontal center of silhouette bounds; translation only",
        "grid_mm_per_pixel": GRID_MM,
        "scope": "Orthographic silhouette metrics. Does not measure unseen surface or material fidelity.",
        "overlay_colors": {"blue": "reference_only", "orange": "model_only", "charcoal": "overlap"},
        "views": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        bundle = Path(temp)
        for view, entry in views.items():
            reference_grid, reference_span, digest = reference_mask(
                reference.parent / entry["mask"], entry["mm_per_pixel"],
            )
            model_grid, model_span = project(vertices, triangles, view)
            metrics, overlay = compare_masks(reference_grid, model_grid)
            artifacts = {"overlay": f"{view}-overlay.png"}
            Image.fromarray(overlay).save(bundle / artifacts["overlay"])
            report["views"][view] = {
                **metrics, **artifacts,
                "reference_mask_sha256": digest,
                "reference_mm_per_pixel": entry["mm_per_pixel"],
                "reference_width_height_mm": reference_span.round(2).tolist(),
                "model_width_height_mm": model_span.round(2).tolist(),
                "model_minus_reference_width_height_mm": (model_span - reference_span).round(2).tolist(),
            }
        (bundle / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf8")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        compare(args.archive, args.reference, args.output, args.blender)
    except (ValueError, KeyError, TypeError, OSError, ezdxf.DXFError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
