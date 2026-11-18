"""Compare a verified garment GLB with registered depths at a declared metric pose."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image

from .capture import capture_photo_hashes
from .check_archive import check_archive
from .compare_hang import depth_error_stats
from .compare_shape import check_frame
from .rgbd import back_project, read_depth
from .scan import in_world


def reference_rays(base, record):
    path = base / record["depth"]
    raw = path.read_bytes()
    checksum = sha256(raw).hexdigest()
    if checksum != record["depth_sha256"]:
        raise ValueError("Reference depth differs from its source checksum")
    unit = record["depth_unit"]
    if unit not in ("m", "mm"):
        raise ValueError("Reference depth unit must be m or mm")
    depth = read_depth(raw, path.suffix.lower()).astype(float) * (0.001 if unit == "mm" else 1)
    if depth.ndim != 2:
        raise ValueError("Provide one two-dimensional registered depth raster per view")
    check_frame(*depth.shape[::-1])
    visible = np.isfinite(depth) & (depth > 0)
    if "mask" in record:
        mask = (base / record["mask"]).read_bytes()
        if sha256(mask).hexdigest() != record["mask_sha256"]:
            raise ValueError("Reference mask differs from its source checksum")
        with Image.open(BytesIO(mask)) as image:
            if image.size != depth.shape[::-1]:
                raise ValueError("Reference mask and depth must share the same pixel grid")
            visible &= np.asarray(image.convert("L")) > 127
    if not visible.any():
        raise ValueError("The reference contains no visible, finite garment depth")
    if not isinstance(record["pose_source"], str) or not record["pose_source"].strip():
        raise ValueError("Identify how the rigid model-to-camera placement was measured")
    pose = json.dumps(record["model_to_camera_m"]).encode()
    points = back_project(depth, visible, record["intrinsics"])
    return in_world(points, pose), in_world(np.zeros((1, 3)), pose)[0], checksum, depth.shape[::-1]


def compare(archive, reference, output, *, flat_comparison=None, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    audit = check_archive(archive)
    captured = json.loads((archive / "capture.json").read_bytes())
    encoded = reference.read_bytes()
    record = json.loads(encoded)
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or captured["kind"] != "physical"
            or record.get("item_id") != captured["item_id"]):
        raise ValueError("Identify the same physical garment in the depth reference")
    views = record.get("views")
    if not isinstance(views, dict) or not views:
        raise ValueError("Provide at least one independent registered depth view")
    mesh = archive / "model/garment.glb"
    if flat_comparison is not None:
        mesh = flat_comparison / "flat-pose.glb"
    mesh_hash = sha256(mesh.read_bytes()).hexdigest()
    if flat_comparison is not None:
        report = json.loads((flat_comparison / "comparison.json").read_bytes())
        if report["pattern_sha256"] != audit["pattern_sha256"] or report["flat_glb"]["sha256"] != mesh_hash:
            raise ValueError("The flat-pose mesh must come from this archive's verified pattern")
    used = capture_photo_hashes(captured["views"])
    used.update(asset["sha256"] for view in captured["views"].values() for asset in view.get("source_assets", {}).values())
    used.update(view.get("calibration", {}).get("cloud_sha256") for view in captured["views"].values()
                if "cloud_sha256" in view.get("calibration", {}))
    job, details = {}, {}
    for name, view in views.items():
        if not name.isascii() or not name.replace("-", "").isalnum() or name.startswith("camera-"):
            raise ValueError("Use alphanumeric view names with optional hyphens")
        points, camera, digest, size = reference_rays(reference.parent, view)
        lineage = view.get("source_pcd_sha256")
        if digest in used or lineage in used:
            raise ValueError(f"{name}: independent depth must differ from all input captures and other reference frames")
        used.add(digest)
        job[name], job[f"camera-{name}"] = points, camera
        details[name] = {"depth_sha256": digest, "image_size_px": list(size), "pose_source": view["pose_source"],
                         "measured_rays": len(points)}
        if lineage is not None:
            details[name]["declared_source_pcd_sha256"] = lineage

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary)
        measured, errors = bundle / "rays.npz", bundle / "errors.npz"
        np.savez_compressed(measured, **job)
        script = Path(__file__).resolve().with_name("rgbd_ray_job.py")
        run_result = run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
                          "--", str(mesh), str(measured), str(errors)], capture_output=True, text=True, check=False)
        if run_result.returncode or not errors.is_file():
            raise ValueError(f"Blender depth comparison failed:\n{(run_result.stdout + run_result.stderr)[-2200:]}")
        if sha256(mesh.read_bytes()).hexdigest() != mesh_hash:
            raise ValueError("The GLB changed during depth evaluation")
        with np.load(errors, allow_pickle=False) as result:
            for name, detail in details.items():
                difference = result[name]
                hit = np.isfinite(difference)
                detail.update(hit_rays=int(hit.sum()), ray_hit_fraction=round(float(hit.mean()), 5),
                              depth_error=depth_error_stats(difference[hit]))
        measured.unlink()
        report = {"schema_version": 1, "item_id": record["item_id"], "pattern_sha256": audit["pattern_sha256"],
                  "garment_glb_sha256": mesh_hash, "reference_sha256": sha256(encoded).hexdigest(),
                  "scope": "Signed distance to the first GLB hit on each observed metric depth ray. Positive is farther.",
                  "pose": "provided rigid transformations, scale preserved", "views": details}
        (bundle / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--flat-comparison", type=Path, help="Optional verified same-pattern flat GLB")
    args = parser.parse_args()
    try:
        compare(args.archive, args.reference, args.output, flat_comparison=args.flat_comparison, blender=args.blender)
    except (ValueError, KeyError, TypeError, OSError, DXFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
