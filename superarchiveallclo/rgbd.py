"""Turn registered metric RGB-D captures into scale-preserving flat-lay inputs."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .capture import archive
from .orientation import orient_flat_points
from .scan import project, read_pcd
from .vision import garment_crop, picture_features


def back_project(depth, coordinates, intrinsics):
    fx, fy, cx, cy = [float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")]
    if not np.isfinite((fx, fy, cx, cy)).all() or min(fx, fy) <= 0:
        raise ValueError("Provide finite camera intrinsics with positive focal lengths in pixels")
    y, x = np.where(coordinates)
    z = depth[y, x]
    return np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))


def plane_basis(support):
    if len(support) < 100:
        raise ValueError("The depth image needs visible tabletop around the garment")
    support = support[::max(1, len(support) // 20_000)]
    active = np.ones(len(support), dtype=bool)
    for _ in range(4):
        origin = np.mean(support[active], axis=0)
        _, span, basis = np.linalg.svd(support[active] - origin, full_matrices=False)
        if span[1] / np.sqrt(active.sum()) < 0.025:
            raise ValueError("Table evidence must span both directions around the garment")
        normal = basis[2] if basis[2, 2] > 0 else -basis[2]
        signed = (support - origin) @ normal
        active = np.abs(signed) <= max(0.006, 3 * np.median(np.abs(signed)))
        if active.mean() < 0.7:
            raise ValueError("The support ring does not resolve to one tabletop plane")
    residual = float(np.quantile(np.abs(signed), 0.9))
    if residual > 0.015 or normal[2] < 0.25:
        raise ValueError("The measured tabletop is not reliably visible in the registered depth")
    right = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    right /= np.linalg.norm(right)
    return origin, np.column_stack((right, np.cross(normal, right), normal)), residual


def registered_regions(mask, valid):
    if mask.sum() < 100:
        raise ValueError("The foreground mask needs at least 100 pixels")
    garment = valid & mask
    coverage = garment.sum() / mask.sum()
    if coverage < 0.75:
        raise ValueError("Registered depth covers less than 75% of the garment mask")
    distance = ndimage.distance_transform_edt(~mask)
    margin = max(4.0, min(mask.shape) * 0.01)
    ring = valid & (distance >= margin) & (distance <= 4 * margin)
    return garment, ring, coverage


def project_surface(points, support, coverage, mm_per_pixel, *, auto_upright=False):
    origin, basis, residual = plane_basis(support)
    tabletop = (points - origin) @ basis
    orientation = {}
    if auto_upright:
        tabletop, rotation, orientation = orient_flat_points(tabletop)
        basis = basis @ rotation
    image, evidence = project(tabletop, "x", "y", mm_per_pixel)
    return image, {
        **evidence,
        "registered_depth_coverage": round(float(coverage), 4),
        "support_points": len(support), "support_plane_residual_p90_mm": round(residual * 1000, 3),
        "surface_distance_to_support_p95_mm": round(float(np.quantile(np.abs(tabletop[:, 2]), 0.95) * 1000), 3),
        "support_origin_camera_m": origin.tolist(), "support_basis_camera": basis.tolist(),
        **orientation,
    }


def metric_projection(depth, mask, intrinsics, mm_per_pixel, *, auto_upright=False):
    if depth.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("The depth and foreground mask must share one registered pixel grid")
    garment, ring, coverage = registered_regions(mask, np.isfinite(depth) & (depth > 0))
    return project_surface(back_project(depth, garment, intrinsics),
                           back_project(depth, ring, intrinsics), coverage, mm_per_pixel, auto_upright=auto_upright)


def read_depth(raw, extension):
    if extension == ".npy":
        depth = np.load(BytesIO(raw), allow_pickle=False)
        if not isinstance(depth, np.ndarray) or depth.dtype.kind not in "fui":
            raise ValueError("Depth NPY must contain a numeric array")
        return depth
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode not in ("I", "I;16", "I;16B", "I;16L"):
            raise ValueError("Depth must be a 16-bit PNG or a numeric NPY array")
        return np.asarray(image)


def prepare_view(root, view, name, directory, provenance, mm_per_pixel, *, auto_upright=False):
    rgb_bytes = (root / view["rgb"]).read_bytes()
    with Image.open(BytesIO(rgb_bytes)) as encoded:
        extension = {"JPEG": ".jpg", "PNG": ".png"}.get(encoded.format)
        if extension is None:
            raise ValueError("Registered colour inputs must be JPEG or PNG")
        rgb = np.asarray(ImageOps.exif_transpose(encoded).convert("RGB"))
    assets = {"rgb": (extension, rgb_bytes)}
    sources = {"depth", "pcd"} & view.keys()
    if len(sources) != 1:
        raise ValueError(f"{name}: supply one registered geometry source, depth or PCD")
    geometry = sources.pop()
    path = root / view[geometry]
    unit = view[f"{geometry}_unit"]
    if unit not in ("m", "mm"):
        raise ValueError(f"{name}: {geometry} unit must be m or mm")
    metres = 0.001 if unit == "mm" else 1.0
    data = path.read_bytes()
    assets[geometry] = (path.suffix.lower(), data)
    mask_label = view.get("mask_label")
    if "mask" in view:
        mask_bytes = (root / view["mask"]).read_bytes()
        with Image.open(BytesIO(mask_bytes)) as encoded:
            if encoded.format != "PNG" or encoded.size != (rgb.shape[1], rgb.shape[0]):
                raise ValueError(f"{name}: mask must be a PNG matching the upright RGB image")
            labels = np.asarray(encoded)
            if labels.ndim != 2:
                raise ValueError(f"{name}: use a single-channel or indexed PNG mask")
            if mask_label is not None and (type(mask_label) is not int or mask_label < 0):
                raise ValueError("Mask label must be a nonnegative integer")
            mask = labels != 0 if mask_label is None else labels == mask_label
        assets["mask"] = (".png", mask_bytes)
    else:
        if mask_label is not None:
            raise ValueError("A mask label requires a source mask")
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        crop, (left, top, right, bottom) = garment_crop(rgb.astype(np.int16), mask)
        mask[top:bottom + 1, left:right + 1] = crop
    if auto_upright and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
        raise ValueError(f"{name}: automatic orientation needs the full garment within the RGB frame")
    if geometry == "depth":
        depth = read_depth(data, path.suffix.lower()).astype(float) * metres
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"{name}: depth must align with the upright RGB image dimensions")
        image, readings = metric_projection(depth, mask, view["intrinsics"], mm_per_pixel, auto_upright=auto_upright)
        sensor = {"origin_type": "registered_rgbd", "depth_unit": unit, "camera_intrinsics_px": view["intrinsics"]}
    else:
        points = read_pcd(data, organized=True) * metres
        if points.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"{name}: PCD must align with the upright RGB image dimensions")
        valid = np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
        garment, ring, coverage = registered_regions(mask, valid)
        image, readings = project_surface(points[garment], points[ring], coverage, mm_per_pixel, auto_upright=auto_upright)
        sensor = {"origin_type": "registered_pcd", "pcd_unit": unit, "cloud_sha256": sha256(data).hexdigest()}
    calibration = {
        **readings, "mm_per_pixel": mm_per_pixel, "source": provenance,
        **sensor,
        "mask_method": "source_labels" if "mask" in view else "rgb_segmentation",
    }
    if mask_label is not None:
        calibration["mask_label"] = mask_label
    evidence = {}
    for key, (suffix, raw) in assets.items():
        filename = f"evidence/{name}-{key}{suffix}"
        (directory / filename).write_bytes(raw)
        evidence[key] = {"file": filename, "sha256": sha256(raw).hexdigest()}
    name = f"{name}-orthographic.png"
    image.save(directory / name)
    _, measurement = picture_features(image, {"calibration": calibration, "mm_per_pixel": mm_per_pixel})
    return {"file": name, "calibration": calibration, "source_assets": evidence}, measurement


def prepare(source, output, *, weights=None, blender="blender", mm_per_pixel=2.0, auto_upright=False):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    record = json.loads(source.read_bytes())
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Provide a registered RGB-D schema_version 1 capture")
    item_id, provenance = [record.get(key) for key in ("item_id", "source")]
    if not all(isinstance(value, str) and value.strip() for value in (item_id, provenance)):
        raise ValueError("Identify the physical garment and the depth calibration source")
    views = record["views"]
    if not isinstance(views, dict) or set(views) not in ({"front"}, {"front", "back"}):
        raise ValueError("Provide an upright front capture and, when available, a separate back capture")
    kind = record.get("kind", "physical")
    if kind not in ("physical", "synthetic_fixture"):
        raise ValueError("Capture kind must be physical or synthetic_fixture")
    session = {"schema_version": 1, "kind": kind, "garment": "crew_neck_tee", "item_id": item_id,
               "input_mode": "paired" if "back" in views else "front_only", "views": {}}
    if "material" in record:
        session["material"] = record["material"]
    result = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "rgbd"
        (bundle / "evidence").mkdir(parents=True)
        for name, view in views.items():
            session["views"][name], result[name] = prepare_view(
                source.parent, view, name, bundle, provenance, mm_per_pixel, auto_upright=auto_upright,
            )
        if "back" in views:
            hashes = [view["source_assets"]["rgb"]["sha256"] for view in session["views"].values()]
            if len(set(hashes)) != 2:
                raise ValueError("Front and back must be separate RGB-D captures")
        (bundle / "session.json").write_text(json.dumps(session, indent=2) + "\n")
        (bundle / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
        if weights is not None:
            archive(bundle / "session.json", bundle / "archive", blender, weights, blind=True)
        bundle.rename(output)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Generate an archive with matching front-only or paired weights")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--mm-per-pixel", type=float, default=2.0)
    parser.add_argument("--auto-upright", action="store_true", help="Recover the in-plane axis of a full flat-lay tee")
    args = parser.parse_args()
    try:
        report = prepare(args.session, args.output, weights=args.model, blender=args.blender, mm_per_pixel=args.mm_per_pixel,
                         auto_upright=args.auto_upright)
    except (KeyError, TypeError, ValueError, OSError, DXFError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
