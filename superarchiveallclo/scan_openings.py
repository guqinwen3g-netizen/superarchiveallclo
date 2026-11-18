"""Measure physical garment openings on a triangulated scan with registered part labels."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from ezdxf import DXFError
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .check_archive import check_archive
from .mesh_openings import boundary_loops, chord_length, loop_edges

PARTS = {"collar": "neck", "sleeves": "cuff", "body": "hem"}


def read_obj(raw):
    vertices, triangles = [], []
    for line in raw.splitlines():
        fields = line.split(b"#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == b"v":
            if len(fields) < 4:
                raise ValueError("An OBJ vertex needs three coordinates")
            vertices.append([float(value) for value in fields[1:4]])
        if fields[0] == b"f":
            if len(fields) != 4:
                raise ValueError("Opening measurement requires a triangulated OBJ")
            face = [int(value.split(b"/", 1)[0]) for value in fields[1:]]
            if 0 in face:
                raise ValueError("OBJ vertex indices are one-based")
            triangles.append([index - 1 if index > 0 else len(vertices) + index for index in face])
    vertices, triangles = np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)
    if not len(vertices) or not np.isfinite(vertices).all() or not len(triangles):
        raise ValueError("OBJ needs finite vertices and triangle faces")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("OBJ triangle refers to a missing vertex")
    return vertices, triangles


def measure_scan(source, part_files, *, unit="native"):
    raw = source.read_bytes()
    vertices, triangles = read_obj(raw)
    used = np.unique(triangles)
    indexed = np.searchsorted(used, triangles)
    graph = coo_matrix(
        (np.ones(indexed.size), (indexed.ravel(), np.roll(indexed, 1, axis=1).ravel())),
        shape=(len(used), len(used)),
    )
    if connected_components(graph.tocsr(), directed=False, return_labels=False) != 1:
        raise ValueError("Scan faces must form one connected garment surface")
    loops = boundary_loops(triangles)
    if len(loops) != 4:
        raise ValueError(f"Expected four open boundaries in the tee scan, found {len(loops)}")
    registration = {}
    surface_vertices = vertices[used]
    bounds = np.ptp(surface_vertices, axis=0)
    tolerance = 0.02 * np.linalg.norm(bounds)
    vertex_tree = cKDTree(surface_vertices)
    trees = []
    for part in PARTS:
        points_raw = part_files[part].read_bytes()
        points = np.loadtxt(BytesIO(points_raw), usecols=(0, 1, 2), ndmin=2)
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"{part}: expected finite registered XYZ points")
        nearest, _ = vertex_tree.query(points)
        assigned = nearest <= tolerance
        if assigned.mean() < 0.85:
            raise ValueError(f"{part}: part points do not align to the scanned mesh")
        registration[part] = {
            "sha256": sha256(points_raw).hexdigest(), "points": len(points),
            "registered_points": int(assigned.sum()),
            "part_to_scan_vertex_p90_native": round(float(np.quantile(nearest, 0.9)), 5),
        }
        trees.append(cKDTree(points[assigned]))
    scale = {"native": 1, "mm": 1, "m": 1000}[unit]
    openings = []
    for loop in loops:
        start, end, lengths = loop_edges(vertices, loop)
        length = float(lengths.sum())
        if length <= 0:
            raise ValueError("Scan has a zero-length opening")
        distances = np.array([tree.query((start + end) / 2)[0] for tree in trees])
        support = np.bincount(distances.argmin(axis=0), weights=lengths, minlength=len(trees)) / length
        dominant = int(support.argmax())
        if support[dominant] < 0.7:
            raise ValueError("Registered part labels do not identify the entire opening")
        openings.append({
            "part": tuple(PARTS.values())[dominant], "vertices": len(loop),
            "perimeter": round(length * scale, 3),
            "perimeter_64_chords": round(chord_length(start, end, lengths) * scale, 3),
            "part_support": round(float(support[dominant]), 5),
            "center_native": np.average((start + end) / 2, axis=0, weights=lengths).round(4).tolist(),
        })
    if Counter(loop["part"] for loop in openings) != {"neck": 1, "cuff": 2, "hem": 1}:
        raise ValueError("Scan boundaries must identify one neck, two cuffs and one hem")
    return {
        "scan_sha256": sha256(raw).hexdigest(), "source_unit": unit,
        "reported_unit": "native" if unit == "native" else "mm",
        "vertices": len(vertices), "triangles": len(triangles),
        "bounds_span_native": bounds.round(3).tolist(),
        "part_registration_tolerance_native": round(float(tolerance), 5),
        "part_registration": registration,
        "openings": sorted(openings, key=lambda loop: (loop["part"], loop["perimeter"])),
    }


def compare_archive_openings(report, archive, item_id):
    if not item_id or report["reported_unit"] != "mm":
        raise ValueError("Comparing an archive requires an item ID and known scan units")
    check_archive(archive)
    captured = json.loads((archive / "capture.json").read_bytes())
    if captured["kind"] != "physical" or captured["item_id"] != item_id:
        raise ValueError("Scan and archive must identify the same physical garment")
    simulation = json.loads((archive / "model/simulation.json").read_bytes())
    if "opening_perimeters_mm" not in simulation:
        raise ValueError("Regenerate the archive to record simulated opening perimeters")
    simulated = simulation["opening_perimeters_mm"]
    length = "perimeter_64_chords"
    pairs = {
        "cuff": sorted((simulated["sleeve_left"], simulated["sleeve_right"]), key=lambda loop: loop[length]),
        "hem": [simulated["hem"]], "neck": [simulated["neck"]],
    }
    report["comparison"] = {
        "item_id": item_id, "pattern_sha256": simulation["pattern_sha256"],
        "method": "64 equally spaced chords around each opening; cuffs paired by length rank",
        "errors_mm": {},
    }
    for part, values in pairs.items():
        references = sorted(loop[length] for loop in report["openings"] if loop["part"] == part)
        report["comparison"]["errors_mm"][part] = [
            {"scan": scanned, "simulated": model[length], "simulated_minus_scan": round(model[length] - scanned, 3)}
            for scanned, model in zip(references, values, strict=True)
        ]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    for part in PARTS:
        parser.add_argument(f"--{part}", required=True, type=Path)
    parser.add_argument("--unit", choices=("native", "mm", "m"), default="native")
    parser.add_argument("--archive", type=Path, help="Optional same-item simulated archive")
    parser.add_argument("--item-id", help="Stable ID that identifies the same scanned garment")
    args = parser.parse_args()
    try:
        report = measure_scan(args.scan, {part: getattr(args, part) for part in PARTS}, unit=args.unit)
        if args.archive is not None:
            compare_archive_openings(report, args.archive, args.item_id)
    except (KeyError, TypeError, ValueError, OSError, DXFError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
