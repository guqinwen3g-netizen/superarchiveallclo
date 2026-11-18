"""Compare local cloth surface points reconstructed from separate reference photographs."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageDraw

from .capture import capture_photo_hashes
from .check_archive import check_archive
from .compare_shape import check_frame, read_mesh
from .visual_hull import AXES


@dataclass
class PointCamera:
    size: tuple[int, int]
    axis: int
    sign: int
    origin: np.ndarray
    mm_per_pixel: float


def read_point_view(raw, entry, size):
    if sha256(raw).hexdigest() != entry["photo_sha256"]:
        raise ValueError(f'{entry["photo"]}: photograph does not match the reference checksum')
    with Image.open(BytesIO(raw)) as photo:
        if photo.width * size[1] != photo.height * size[0]:
            raise ValueError("Resizing to the landmark raster must preserve the original photograph aspect")
    origin = np.asarray(entry["origin_px"], dtype=float)
    scale = float(entry["mm_per_pixel"])
    if (origin.shape != (2,) or not np.isfinite(origin).all() or np.any(origin < 0)
            or np.any(origin >= size) or not 0.05 <= scale <= 20):
        raise ValueError("Provide an in-frame camera datum and a finite millimetre scale")
    axis, sign = AXES[entry["axis"]]
    return PointCamera(size, axis, sign, origin, scale)


def projection_matrix(camera, focal_pixels=None):
    unit = np.eye(3)
    right = camera.sign * unit[camera.axis]
    down = -unit[2]
    if focal_pixels is None:
        return np.vstack((np.column_stack((np.stack((right, down)) / camera.mm_per_pixel, camera.origin)),
                          [0, 0, 0, 1]))
    # The observed pole fixes scale in the datum plane. EXIF supplies focal
    # length, while the nearly cardinal viewing directions stay fixed.
    forward = camera.sign * unit[1] if camera.axis == 0 else -camera.sign * unit[0]
    center = (np.array(camera.size) - 1) / 2
    distance = focal_pixels[0] * camera.mm_per_pixel
    plane = np.stack((right, down)) * focal_pixels[:, None] + center[:, None] * forward
    return np.vstack((np.column_stack((plane, camera.origin * distance)), np.r_[forward, distance]))


def exif_focal_pixels(raw, raster_size, axes):
    with Image.open(BytesIO(raw)) as photo:
        exif = photo.getexif().get_ifd(34665)
        physical_length = float(exif[37386])
        unit_mm = {2: 25.4, 3: 10.0}[int(exif[41488])]
        density = np.array([float(exif[41486]), float(exif[41487])])
        if axes == "landscape_sensor" and photo.width < photo.height:
            density = density[::-1]
        focal = density * physical_length / unit_mm * np.asarray(raster_size) / photo.size
    if not np.isfinite(focal).all() or not (focal > 0).all():
        raise ValueError("EXIF focal length and pixel density must be positive and finite")
    return focal


def project(camera, point):
    q = camera @ np.r_[point, 1]
    if q[2] <= 0:
        raise ValueError("A reference point projects behind a camera")
    return q[:2] / q[2]


def triangulate(matrices, views, observations):
    rows = np.vstack([matrices[name][:2] - np.asarray(uv)[:, None] * matrices[name][2]
                      for name, uv in observations.items()])
    # Give comparable millimetre residuals to near and distant photographs.
    rows /= np.linalg.norm(rows[:, :3], axis=1)[:, None]
    initial, _, rank, _ = np.linalg.lstsq(rows[:, :3], -rows[:, 3], rcond=None)
    if rank != 3:
        raise ValueError("The reference views do not resolve all three coordinates")
    # The linear equations are exact for orthographic references. For a
    # pinhole camera use the same equations with projected depth weights.
    for _ in range(2):
        weights = np.repeat([views[n].mm_per_pixel * np.linalg.norm(matrices[n][0, :3])
                             / (matrices[n][2] @ np.r_[initial, 1]) for n in observations], 2)
        initial = np.linalg.lstsq(rows[:, :3] * weights[:, None], -rows[:, 3] * weights, rcond=None)[0]
    return initial


def nearest_surface_point(point, triangles):
    """Return the exact nearest point on the triangles, including their edges."""
    a, b, c = triangles.transpose(1, 0, 2)
    first, second, relative = b - a, c - a, point - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    aa, ab, bb = dot(first, first), dot(first, second), dot(second, second)
    ap, bp = dot(first, relative), dot(second, relative)
    denominator = aa * bb - ab * ab
    valid = denominator > 0
    u = np.divide(bb * ap - ab * bp, denominator, out=np.zeros_like(denominator), where=valid)
    v = np.divide(aa * bp - ab * ap, denominator, out=np.zeros_like(denominator), where=valid)
    interior = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    on_plane = a + u[:, None] * first + v[:, None] * second
    candidates = [on_plane[interior]]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        span = dot(edge, edge)
        fraction = np.divide(dot(point - start, edge), span, out=np.zeros_like(span), where=span > 0)
        candidates.append(start + np.clip(fraction, 0, 1)[:, None] * edge)
    possible = np.concatenate(candidates)
    return possible[np.argmin(np.sum((possible - point) ** 2, axis=1))]


def observed_markers(entry, views, metadata):
    fit, hold = entry["fit"], entry["held_out"]
    if {views[name].axis for name in fit} != {0, 1} or not hold or set(fit) & set(hold):
        raise ValueError("Each point needs a front/back and side fit, plus a separate held-out photograph")
    if any(metadata[name]["role"] != "carve" for name in fit) or any(metadata[name]["role"] != "held_out" for name in hold):
        raise ValueError("Landmark fit and holdout must follow the declared reference roles")
    markers = {}
    for name, xy in (fit | hold).items():
        point = np.asarray(xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= views[name].size):
            raise ValueError("Landmark pixel must fall inside its reference image")
        markers[name] = point
    return {name: markers[name] for name in fit}, {name: markers[name] for name in hold}


def projected_errors(point, markers, matrices, views):
    return {name: round(float(np.linalg.norm(project(matrices[name], point) - pixel) * views[name].mm_per_pixel), 3)
            for name, pixel in markers.items()}


def score(matrix_sets, views, metadata, annotations, triangles):
    points = {}
    for name, entry in annotations.items():
        fit, hold = observed_markers(entry, views, metadata)
        measurements = {}
        for method, matrices in matrix_sets.items():
            xyz = triangulate(matrices, views, fit)
            closest = nearest_surface_point(xyz, triangles)
            measurements[method] = {
                "position_mm": xyz.round(3).tolist(),
                "fit_residual_proxy_mm": projected_errors(xyz, fit, matrices, views),
                "held_out_residual_proxy_mm": projected_errors(xyz, hold, matrices, views),
                "nearest_glb_point_mm": closest.round(3).tolist(),
                "nearest_glb_minus_reference_xyz_mm": (closest - xyz).round(3).tolist(),
                "distance_to_glb_mm": round(float(np.linalg.norm(closest - xyz)), 3),
            }
        points[name] = {"label": entry["label"], "region": entry["region"], **measurements}
        if "exif_pinhole" in measurements:
            primary = np.asarray(measurements["orthographic"]["position_mm"])
            sensitivity = np.linalg.norm(primary - measurements["exif_pinhole"]["position_mm"])
            points[name]["camera_assumption_displacement_mm"] = round(float(sensitivity), 3)
    return points


def summarize_points(points, methods):
    summary, by_region = {}, {}
    for method in methods:
        local = [point[method]["distance_to_glb_mm"] for point in points.values()]
        reprojections = {}
        for point in points.values():
            for view, error in point[method]["held_out_residual_proxy_mm"].items():
                reprojections.setdefault(view, []).append(error)
        heldout = [error for errors in reprojections.values() for error in errors]
        summary[method] = {
            "distance_to_glb_median_mm": round(float(np.median(local)), 3),
            "distance_to_glb_p90_mm": round(float(np.quantile(local, 0.9)), 3),
            "held_out_residual_median_proxy_mm": round(float(np.median(heldout)), 3),
            "held_out_residual_max_proxy_mm": max(heldout),
            "held_out_residual_median_by_view_proxy_mm": {
                name: round(float(np.median(errors)), 3) for name, errors in reprojections.items()
            },
        }
    for region in sorted({point["region"] for point in points.values()}):
        selected = [point for point in points.values() if point["region"] == region]
        by_region[region] = {"points": len(selected)}
        for method in methods:
            distances = [point[method]["distance_to_glb_mm"] for point in selected]
            offsets = [point[method]["nearest_glb_minus_reference_xyz_mm"] for point in selected]
            by_region[region][method] = {
                "distance_to_glb_median_mm": round(float(np.median(distances)), 3),
                "median_nearest_glb_minus_reference_xyz_mm": np.median(offsets, axis=0).round(3).tolist(),
            }
    return summary, by_region


def draw_held_out(directory, photos, views, annotations, points, matrix_sets):
    names = sorted({view for entry in annotations.values() for view in entry["held_out"]})
    colors = {"orthographic": (235, 108, 40), "exif_pinhole": (147, 82, 193)}
    for name in names:
        with Image.open(BytesIO(photos[name])) as raw:
            photo = raw.convert("RGB").resize(views[name].size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(photo)
        for key, entry in annotations.items():
            if name not in entry["held_out"]:
                continue
            x, y = entry["held_out"][name]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(50, 176, 211), width=2)
            for method in matrix_sets:
                color = colors[method]
                a, b = project(matrix_sets[method][name], points[key][method]["position_mm"])
                draw.line((x, y, a, b), fill=color, width=1)
                draw.line((a - 3, b, a + 3, b), fill=color, width=2)
                draw.line((a, b - 3, a, b + 3), fill=color, width=2)
        photo.save(directory / f"{name}-landmarks.png")


def compare(archive, reference, observations, output, *, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    audit = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_bytes())
    raw, marked = reference.read_bytes(), observations.read_bytes()
    source, labels = json.loads(raw), json.loads(marked)
    if (source.get("schema_version") != 1 or labels.get("schema_version") != 1
            or capture["kind"] != "physical" or source["kind"] != "physical"
            or capture["item_id"] != source["item_id"] or source["item_id"] != labels["item_id"]):
        raise ValueError("Reference photographs and points must identify the same physical garment")
    if any(not name.isascii() or not name.replace("-", "").isalnum() for name in source["views"]):
        raise ValueError("Use alphanumeric view IDs with optional hyphens")
    size = tuple(labels["annotation_size_px"])
    if len(size) != 2 or any(type(value) is not int for value in size):
        raise ValueError("Provide the width and height of the landmark raster")
    check_frame(*size)
    metadata = source["views"]
    photos = {name: (reference.parent / entry["photo"]).read_bytes() for name, entry in metadata.items()}
    views = {name: read_point_view(photos[name], entry, size) for name, entry in metadata.items()}
    hashes = [entry["photo_sha256"] for entry in metadata.values()]
    if len(set(hashes)) != len(hashes) or set(hashes) & capture_photo_hashes(capture["views"]):
        raise ValueError("Reference photographs must be distinct and withheld from the paper reconstruction")
    axes = labels.get("exif_focal_plane_axes")
    if axes not in (None, "image", "landscape_sensor"):
        raise ValueError("Declare how the EXIF focal plane axes correspond to the image")
    matrix_sets = {"orthographic": {name: projection_matrix(view) for name, view in views.items()}}
    focal = {}
    if axes is not None:
        pinhole = {}
        for name, view in views.items():
            pixels = exif_focal_pixels(photos[name], view.size, axes)
            focal[name] = pixels.round(3).tolist()
            pinhole[name] = projection_matrix(view, pixels)
        matrix_sets["exif_pinhole"] = pinhole
    if not labels["points"]:
        raise ValueError("Provide at least one photographed reference point")
    placement = np.asarray(source["model_origin_mm"], dtype=float)
    if placement.shape != (3,) or not np.isfinite(placement).all():
        raise ValueError("Provide a finite three-dimensional model datum in millimetres")
    model = archive / "model/garment.glb"
    checksum = sha256(model.read_bytes()).hexdigest()
    vertices, triangles = read_mesh(model, blender)
    if sha256(model.read_bytes()).hexdigest() != checksum:
        raise ValueError("GLB changed during landmark evaluation")
    points = score(matrix_sets, views, metadata, labels["points"], (vertices * 1000 - placement)[triangles])
    summary, by_region = summarize_points(points, matrix_sets)
    report = {
        "schema_version": 1, "item_id": source["item_id"], "setup": source["setup"],
        "scope": "Sparse physical points under declared camera approximations. Reprojection residuals use datum-plane scales.",
        "reference_sha256": sha256(raw).hexdigest(), "observations_sha256": sha256(marked).hexdigest(),
        "pattern_sha256": audit["pattern_sha256"], "garment_glb_sha256": checksum, "archive_audit": audit,
        "exif_focal_pixels": focal, "reference_photo_sha256": dict(zip(views, hashes, strict=True)),
        "model_origin_mm": placement.tolist(), "summary": summary, "by_region": by_region, "points": points,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        bundle = Path(tmp) / "landmarks"
        bundle.mkdir()
        draw_held_out(bundle, photos, views, labels["points"], points, matrix_sets)
        (bundle / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        compare(args.archive, args.reference, args.observations, args.output, blender=args.blender)
    except (KeyError, TypeError, ValueError, OSError, DXFError, np.linalg.LinAlgError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
