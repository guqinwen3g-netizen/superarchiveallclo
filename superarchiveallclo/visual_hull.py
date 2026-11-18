"""Compare a garment GLB with a metric visual hull from cardinal photographs."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from .check_archive import check_archive
from .compare_shape import check_frame, read_mesh

AXES = {"front": (0, 1), "back": (0, -1), "left": (1, 1), "right": (1, -1)}
MAX_VOXELS = 12_000_000


@dataclass
class CameraView:
    mask: np.ndarray
    axis: int
    sign: int
    origin: np.ndarray
    mm_per_pixel: float


def photograph_indices(view, horizontal, height):
    u = np.rint(view.origin[0] + view.sign * horizontal / view.mm_per_pixel).astype(int)
    v = np.rint(view.origin[1] - height / view.mm_per_pixel).astype(int)
    return u, v


def carve(views, step):
    if not 2 <= step <= 15:
        raise ValueError("Visual hull grid spacing must be between 2 and 15 mm")
    if {view.axis for view in views} != {0, 1}:
        raise ValueError("Provide both a front/back direction and a side direction")
    extent = [[], [], []]
    for view in views:
        v, u = np.where(view.mask)
        horizontal = (np.array([u.min(), u.max()]) - view.origin[0]) * view.sign * view.mm_per_pixel
        height = (view.origin[1] - np.array([v.min(), v.max()])) * view.mm_per_pixel
        extent[view.axis].extend(horizontal)
        extent[2].extend(height)
    low = np.floor(np.array(list(map(min, extent))) / step).astype(int) - 2
    high = np.ceil(np.array(list(map(max, extent))) / step).astype(int) + 3
    shape = high - low
    if any(n > 1000 for n in shape) or np.prod(shape) > MAX_VOXELS:
        raise ValueError("Visual hull grid is too large. Check camera units and pixel scales")
    axes = [np.arange(a, b) * step for a, b in zip(low, high, strict=True)]
    volume = np.ones(shape, dtype=bool)
    for view in views:
        u, v = photograph_indices(view, axes[view.axis], axes[2])
        height, width = view.mask.shape
        contained = (u >= 0)[:, None] & (u < width)[:, None] & (v >= 0) & (v < height)
        silhouette = contained & view.mask[v.clip(0, height - 1)[None, :], u.clip(0, width - 1)[:, None]]
        volume &= silhouette[:, None, :] if view.axis == 0 else silhouette[None, :, :]
    if not volume.any():
        raise ValueError("The calibrated silhouettes have no common 3D volume")
    return volume, low * step


def hull_mask(volume, low, step, view):
    height, width = view.mask.shape
    horizontal = (np.arange(width) - view.origin[0]) * view.mm_per_pixel * view.sign
    vertical = (view.origin[1] - np.arange(height)) * view.mm_per_pixel
    u = np.rint((horizontal - low[view.axis]) / step).astype(int)
    v = np.rint((vertical - low[2]) / step).astype(int)
    grid = volume.any(axis=1 - view.axis).T
    rows, columns = grid.shape
    contained = (v >= 0)[:, None] & (v < rows)[:, None] & (u >= 0) & (u < columns)
    return contained & grid[v.clip(0, rows - 1)[:, None], u.clip(0, columns - 1)[None, :]]


def mesh_mask(vertices_mm, triangles, view):
    u = view.origin[0] + view.sign * vertices_mm[:, view.axis] / view.mm_per_pixel
    v = view.origin[1] - vertices_mm[:, 2] / view.mm_per_pixel
    image = Image.new("1", view.mask.shape[::-1])
    draw = ImageDraw.Draw(image)
    for triangle in np.column_stack((u, v))[triangles]:
        draw.polygon([tuple(p) for p in triangle], fill=1)
    return np.asarray(image, dtype=bool)


def silhouette_metrics(observed, candidate, mm_per_pixel):
    if not candidate.any():
        return {"iou": 0, "contour_p90_mm": None}
    a = observed & ~ndimage.binary_erosion(observed)
    b = candidate & ~ndimage.binary_erosion(candidate)
    distances = np.r_[ndimage.distance_transform_edt(~a)[b], ndimage.distance_transform_edt(~b)[a]] * mm_per_pixel
    return {"iou": round(float((observed & candidate).sum() / (observed | candidate).sum()), 4),
            "contour_p90_mm": round(float(np.quantile(distances, 0.9)), 2)}


def save_overlay(path, observed, candidate):
    image = np.full((*observed.shape, 3), (250, 248, 243), dtype=np.uint8)
    image[observed & ~candidate] = (22, 119, 167)
    image[candidate & ~observed] = (205, 86, 41)
    image[candidate & observed] = (49, 61, 60)
    Image.fromarray(image).save(path)


def compare_surface(volume, low, step, vertices, triangles):
    points = vertices[triangles]
    area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
    centers = points.mean(axis=1)
    index = np.rint((centers - low) / step).astype(int)
    contained = np.all((index >= 0) & (index < np.array(volume.shape)), axis=1)
    inside = np.zeros(len(points), dtype=bool)
    inside[contained] = volume[tuple(index[contained].T)]
    boundary = np.argwhere(volume & ~ndimage.binary_erosion(volume)) * step + low
    outside_distances = np.zeros(len(points))
    outside_distances[~inside] = cKDTree(boundary).query(centers[~inside])[0]
    ordered = np.argsort(outside_distances)
    p90 = outside_distances[ordered[np.searchsorted(np.cumsum(area[ordered]), 0.9 * area.sum())]]
    return {
        "triangle_area_outside_by_more_than_one_voxel_fraction": round(float(area[outside_distances > step].sum() / area.sum()), 4),
        "one_sided_triangle_center_distance_p90_mm": round(float(p90), 2),
    }, boundary


def write_surface(path, boundary):
    vertices = np.asarray(boundary / 1000, dtype="<f4")
    with path.open("wb") as stream:
        stream.write((f"ply\nformat binary_little_endian 1.0\ncomment visual hull boundary voxels in metres\n"
                      f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n").encode())
        stream.write(vertices.tobytes())


def read_view(base, entry):
    raw = (base / entry["mask"]).read_bytes()
    with Image.open(BytesIO(raw)) as image:
        check_frame(*image.size)
        mask = np.asarray(image.convert("L")) > 127
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Each visual hull mask must contain a garment with an unclipped border")
    origin = np.asarray(entry["origin_px"], dtype=float)
    scale = float(entry["mm_per_pixel"])
    if (origin.shape != (2,) or not np.isfinite(origin).all() or np.any(origin < 0)
            or np.any(origin >= mask.shape[::-1]) or not isfinite(scale) or not 0.05 <= scale <= 20):
        raise ValueError("Provide an in-frame origin and a finite camera scale in millimetres per pixel")
    role = entry["role"]
    if role not in ("carve", "held_out"):
        raise ValueError("Use carve or held_out for the view role")
    photo = sha256((base / entry["photo"]).read_bytes()).hexdigest()
    if photo != entry["photo_sha256"]:
        raise ValueError(f'{entry["photo"]}: photograph does not match the reference checksum')
    axis, sign = AXES[entry["axis"]]
    return CameraView(mask, axis, sign, origin, scale), {**entry, "mask_sha256": sha256(raw).hexdigest()}


def compare(archive, reference, output, *, blender="blender", step=5.0):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    audit = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_bytes())
    simulation = json.loads((archive / "model/simulation.json").read_bytes())
    raw = reference.read_bytes()
    source = json.loads(raw)
    if (not isinstance(source, dict) or source.get("schema_version") != 1
            or not source.get("item_id") or source["item_id"] != capture["item_id"]
            or source.get("kind") != capture["kind"]):
        raise ValueError("The visual hull reference must identify the archived garment and capture kind")
    setup = source["setup"]
    placement = np.asarray(source["model_origin_mm"], dtype=float)
    if not isinstance(setup, str) or not setup.strip() or placement.shape != (3,) or not np.isfinite(placement).all():
        raise ValueError("Record the common datum and the camera calibration setup")
    views = {name: read_view(reference.parent, entry) for name, entry in source["views"].items()}
    if any(not name.isascii() or not name.replace("-", "").isalnum() for name in views):
        raise ValueError("Use alphanumeric view IDs with optional hyphens")
    hashes = [entry["photo_sha256"] for _, entry in views.values()]
    inputs = {view["sha256"] for view in capture["views"].values()}
    if len(hashes) != len(set(hashes)) or inputs.intersection(hashes):
        raise ValueError("Provide distinct reference photographs held out from garment reconstruction")
    volume, low = carve([view for view, entry in views.values() if entry["role"] == "carve"], step)
    path = archive / "model/garment.glb"
    model_hash = sha256(path.read_bytes()).hexdigest()
    vertices, triangles = read_mesh(path, blender)
    if sha256(path.read_bytes()).hexdigest() != model_hash:
        raise ValueError("Garment GLB changed during mesh evaluation")
    vertices = vertices * 1000 - placement
    support, boundary = compare_surface(volume, low, step, vertices, triangles)
    report = {
        "schema_version": 1, "item_id": capture["item_id"], "setup": setup,
        "scope": "Photo-derived outer volume and projected contours under the declared camera assumptions",
        "model_origin_mm": placement.tolist(), "reference_manifest_sha256": sha256(raw).hexdigest(),
        "pattern_sha256": audit["pattern_sha256"], "garment_glb_sha256": model_hash,
        "reference_form": simulation.get("reference_form", "torso_with_arms"),
        "archive_audit": audit,
        "voxel_step_mm": step, "hull_bounds_span_mm": (np.ptp(boundary, axis=0) + step).round(2).tolist(),
        "glb_bounds_span_mm": np.ptp(vertices, axis=0).round(2).tolist(),
        "hull_volume_litres": round(float(volume.sum() * step ** 3 / 1_000_000), 3),
        "candidate_surface_vs_hull": support, "views": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        bundle = Path(temp) / "volume"
        bundle.mkdir()
        write_surface(bundle / "hull-boundary.ply", boundary)
        np.savez_compressed(bundle / "hull.npz", occupancy=volume, low_mm=low, step_mm=step)
        for name, (view, entry) in views.items():
            hull = hull_mask(volume, low, step, view)
            model = mesh_mask(vertices, triangles, view)
            save_overlay(bundle / f"{name}-hull.png", view.mask, hull)
            save_overlay(bundle / f"{name}-glb.png", view.mask, model)
            report["views"][name] = {**entry, "hull": silhouette_metrics(view.mask, hull, view.mm_per_pixel),
                                     "glb": silhouette_metrics(view.mask, model, view.mm_per_pixel)}
        (bundle / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--step-mm", type=float, default=5)
    args = parser.parse_args()
    try:
        compare(args.archive, args.reference, args.output, blender=args.blender, step=args.step_mm)
    except (ValueError, KeyError, TypeError, OSError, DXFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
