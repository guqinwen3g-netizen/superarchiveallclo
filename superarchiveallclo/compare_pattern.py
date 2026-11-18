"""Compare candidate sewing outlines with a welded OBJ's flat UV charts."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .check_archive import check_archive
from .compare_shape import GRID_MM, check_frame, compare_masks, crop_surface
from .pattern import Edge


def read_obj(encoded):
    vertices, uvs, faces = [], [], []
    for line in encoded.decode("utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        tag, *values = fields
        if tag == "v":
            vertices.append([float(value) for value in values[:3]])
        elif tag == "vt":
            uvs.append([float(value) for value in values[:2]])
        elif tag == "f":
            if len(values) != 3:
                raise ValueError("Reference OBJ faces must be triangles")
            indices = [tuple(int(value) - 1 for value in item.split("/")[:2]) for item in values]
            if any(len(index) != 2 for index in indices):
                raise ValueError("Every reference face vertex must include a UV index")
            faces.append(indices)
    vertices, uvs, faces = np.asarray(vertices), np.asarray(uvs), np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or uvs.ndim != 2 or uvs.shape[1] != 2 or not len(faces):
        raise ValueError("Reference OBJ needs XYZ vertices, 2D UVs and indexed triangles")
    if not np.isfinite(vertices).all() or not np.isfinite(uvs).all():
        raise ValueError("Reference geometry has non-finite coordinates")
    if faces.min() < 0 or faces[:, :, 0].max() >= len(vertices) or faces[:, :, 1].max() >= len(uvs):
        raise ValueError("Reference OBJ needs positive, in-range vertex and UV indices")
    return vertices, uvs, faces


def chart_faces(faces, uv_count):
    indices = faces[:, :, 1]
    left, right = indices.ravel(), np.roll(indices, -1, axis=1).ravel()
    links = coo_matrix((np.ones(len(left), dtype=bool), (left, right)), shape=(uv_count, uv_count))
    _, components = connected_components(links, directed=False)
    order = {}
    for i, component in enumerate(components[indices[:, 0]]):
        order.setdefault(int(component), []).append(i)
    return [faces[members] for members in order.values()]


def scale_from_surface(vertices, uvs, charts):
    faces = np.concatenate(charts)
    edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=2).reshape(-1, 2, 2)
    vertex_pairs, uv_pairs = edges[:, :, 0], edges[:, :, 1]
    surface_mm = np.linalg.norm(vertices[vertex_pairs[:, 0]] - vertices[vertex_pairs[:, 1]], axis=1) * 1000
    uv_length = np.linalg.norm(uvs[uv_pairs[:, 0]] - uvs[uv_pairs[:, 1]], axis=1)
    usable = (uv_length > 1e-7) & (surface_mm > 0.05)
    if usable.sum() < 20:
        raise ValueError("Reference mesh has too few usable edges to establish the UV scale")
    return np.quantile(surface_mm[usable] / uv_length[usable], [0.05, 0.5, 0.95])


def shared_pairs(charts):
    members = defaultdict(set)
    for name, faces in charts.items():
        vertices = faces[:, :, 0]
        for a, b in zip(vertices.ravel(), np.roll(vertices, -1, axis=1).ravel(), strict=True):
            members[tuple(sorted((int(a), int(b))))].add(name)
    return {tuple(sorted(names)) for names in members.values() if len(names) == 2}


def polygon_mask(polygons):
    points = np.concatenate(polygons)
    span = np.ptp(points, axis=0)
    width, height = (ceil(axis / GRID_MM) + 3 for axis in span)
    check_frame(width, height)
    origin = points.min(axis=0)
    image = Image.new("1", (width, height))
    paint = ImageDraw.Draw(image)
    for polygon in polygons:
        pixels = np.rint((polygon - origin) / GRID_MM).astype(int) + 1
        paint.polygon([tuple(p) for p in pixels], fill=1)
    return crop_surface(np.asarray(image, dtype=bool)), span


def sewing_outline(panel):
    if panel["id"] == "collar":
        left, right = panel["foldline"]
        return np.array([(left[0], 0), (right[0], 0), right, left])
    return np.array([point for record in panel["edges"] for point in Edge(**record).samples()[:-1]])


def compare_pattern(archive, reference, output, *, item_id, source, order, mm_per_uv=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not source.strip():
        raise ValueError("Record the source and garment identity of the reference OBJ")
    audit = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_text())
    if capture["kind"] != "physical" or not item_id.strip() or item_id != capture["item_id"]:
        raise ValueError("Reference and candidate must identify the same physical garment")
    encoded_pattern = (archive / "pattern/pattern.json").read_bytes()
    if sha256(encoded_pattern).hexdigest() != audit["pattern_sha256"]:
        raise ValueError("Pattern changed during archive verification")
    pattern = json.loads(encoded_pattern)
    panels = {panel["id"]: panel for panel in pattern["panels"]}
    encoded = reference.read_bytes()
    vertices, uvs, faces = read_obj(encoded)
    charts = dict(zip(order, chart_faces(faces, len(uvs)), strict=True))
    if len(order) != len(panels) or set(charts) != panels.keys():
        raise ValueError("Name every UV chart once using the candidate's panel IDs in OBJ face order")
    scale_range = scale_from_surface(vertices, uvs, [chart for name, chart in charts.items() if name != "collar"])
    if mm_per_uv is None:
        if scale_range[2] / scale_range[0] > 1.10:
            raise ValueError("UV scale varies across the reference geometry. Supply an independently calibrated scale")
        mm_per_uv = float(scale_range[1])
        calibration = "Median 3D edge length in millimetres divided by the paired UV edge length, excluding the collar"
    else:
        if not isfinite(mm_per_uv) or mm_per_uv <= 0:
            raise ValueError("UV scale must be a positive finite value")
        calibration = "User-supplied independent mm per UV unit"
    reference_pairs = shared_pairs(charts)
    planned_pairs = {tuple(sorted((seam["a"][0], seam["b"][0]))) for seam in pattern["seams"]
                     if seam["a"][0] != seam["b"][0]}
    report = {
        "schema_version": 1, "item_id": item_id, "reference_source": source,
        "reference_obj_sha256": sha256(encoded).hexdigest(), "pattern_sha256": audit["pattern_sha256"],
        "reference_mm_per_uv": round(mm_per_uv, 5), "scale_method": calibration,
        "reference_edge_mm_per_uv_p05_p50_p95": scale_range.round(5).tolist(),
        "scope": "Net sewing geometry and cross-panel adjacency from the digital reference. The collar uses the candidate folded strip. Seam allowances and self-seams need independent evidence.",
        "alignment": "Flip OBJ V for image coordinates, then translate chart top and horizontal center; no rescaling to fit",
        "adjacency": {"shared": sorted(reference_pairs & planned_pairs),
                      "reference_only": sorted(reference_pairs - planned_pairs),
                      "candidate_only": sorted(planned_pairs - reference_pairs)},
        "panels": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        bundle = Path(temp)
        for name, chart in charts.items():
            reference_grid, reference_span = polygon_mask(uvs[chart[:, :, 1]] * [mm_per_uv, -mm_per_uv])
            candidate_grid, candidate_span = polygon_mask([sewing_outline(panels[name])])
            metrics, overlay = compare_masks(reference_grid, candidate_grid)
            filename = f"{name}-overlay.png"
            Image.fromarray(overlay).save(bundle / filename)
            report["panels"][name] = {
                **metrics, "reference_width_height_mm": reference_span.round(2).tolist(),
                "candidate_width_height_mm": candidate_span.round(2).tolist(),
                "candidate_minus_reference_width_height_mm": (candidate_span - reference_span).round(2).tolist(),
                "overlay": filename,
            }
        (bundle / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path, help="Triangulated OBJ with flat UV charts")
    parser.add_argument("--panel-order", nargs="+", required=True, help="Panel names in OBJ first-face order")
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mm-per-uv", type=float)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_pattern(args.archive, args.reference, args.output, item_id=args.item_id, source=args.source,
                                 order=args.panel_order, mm_per_uv=args.mm_per_uv)
    except (ValueError, KeyError, TypeError, OSError, DXFError) as error:
        parser.error(str(error))
    for name, panel in report["panels"].items():
        print(f'{name} · IoU {panel["intersection_over_union"]:.3f} · '
              f'width/height differences {panel["candidate_minus_reference_width_height_mm"]} mm')


if __name__ == "__main__":
    main()
