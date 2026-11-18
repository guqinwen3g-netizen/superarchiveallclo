"""Audit the traceability and format integrity of an exported garment archive."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import dist, isfinite
from pathlib import Path
from struct import unpack

import ezdxf
from ezdxf import units


def dxf_inventory(path, pattern):
    source = ezdxf.readfile(path)
    if source.header.get("$INSUNITS") != units.MM or source.audit().has_errors:
        raise ValueError("DXF must reopen as a valid millimetre drawing")
    blocks = [entry.dxf.name for entry in source.modelspace().query("INSERT")]
    if sorted(blocks) != sorted(panel["id"].upper() for panel in pattern["panels"]):
        raise ValueError("DXF block names do not match the pattern panel inventory")
    for panel in pattern["panels"]:
        block = source.blocks[panel["id"].upper()]
        cuts = list(block.query('LWPOLYLINE[layer=="CUT"]'))
        if len(cuts) != 1 or not cuts[0].closed:
            raise ValueError(f"{panel['id']}: expected one closed DXF cut outline")
        original = panel["cut_outline"]
        exported = [(float(x), -float(y)) for x, y in cuts[0].get_points(format="xy")]
        if len(original) != len(exported):
            raise ValueError(f"{panel['id']}: DXF cut vertex count changed")
        for point, actual in zip(original, exported, strict=True):
            reference = tuple(val - origin for val, origin in zip(point, original[0], strict=True))
            relative = tuple(val - origin for val, origin in zip(actual, exported[0], strict=True))
            if dist(reference, relative) > 0.001:
                raise ValueError(f"{panel['id']}: DXF cut geometry changed after export")
    return len(blocks)


def glb_inventory(path):
    with path.open("rb") as source:
        header = source.read(20)
        if len(header) != 20:
            raise ValueError("Truncated GLB header")
        magic, version, total, json_size, kind = unpack("<4sIII4s", header)
        if (magic, version, kind) != (b"glTF", 2, b"JSON") or total != path.stat().st_size:
            raise ValueError("Invalid GLB 2.0 container")
        if not 0 < json_size <= total - 20:
            raise ValueError("GLB metadata length exceeds container")
        document = json.loads(source.read(json_size))
        remaining = total - 20 - json_size
        chunk = source.read(8)
        if remaining < 8 or len(chunk) != 8:
            raise ValueError("Garment GLB has no geometry buffer")
        binary_size, binary_kind = unpack("<I4s", chunk)
        if binary_kind != b"BIN\x00" or remaining != binary_size + 8 or binary_size == 0:
            raise ValueError("Invalid GLB geometry buffer")
    positions = [
        document["accessors"][primitive["attributes"]["POSITION"]]
        for mesh in document["meshes"] for primitive in mesh["primitives"]
    ]
    if not positions or any(a["type"] != "VEC3" or a["count"] < 3 for a in positions):
        raise ValueError("GLB has no garment vertices")
    lower = [min(a["min"][axis] for a in positions) for axis in range(3)]
    upper = [max(a["max"][axis] for a in positions) for axis in range(3)]
    span = [b - a for a, b in zip(lower, upper, strict=True)]
    if any(not isfinite(value) or value <= 0 for value in span):
        raise ValueError("GLB garment bounds are invalid")
    return {"meshes": len(document["meshes"]), "bounds_span_m": [round(value, 4) for value in span]}


def check_archive(root: Path):
    capture = json.loads((root / "capture.json").read_text(encoding="utf8"))
    photos = (root / "photos").resolve()
    evidence = (root / "evidence").resolve()
    for name, view in capture["views"].items():
        original = (root / view["file"]).resolve()
        if not original.is_relative_to(photos) or sha256(original.read_bytes()).hexdigest() != view["sha256"]:
            raise ValueError(f"{name}: archived photo does not match its source checksum")
        for key, asset in view.get("source_assets", {}).items():
            path = (root / asset["file"]).resolve()
            if not path.is_relative_to(evidence) or sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"{name}: archived {key} does not match its source checksum")
    pattern_bytes = (root / "pattern/pattern.json").read_bytes()
    pattern = json.loads(pattern_bytes)
    simulation = json.loads((root / "model/simulation.json").read_text(encoding="utf8"))
    checksum = sha256(pattern_bytes).hexdigest()
    if checksum != simulation["pattern_sha256"]:
        raise ValueError("3D simulation refers to a different source pattern")
    if pattern["units"] != "mm":
        raise ValueError("Pattern must declare millimetres")
    if not (root / "model/garment.blend").stat().st_size:
        raise ValueError("Native Blender scene is empty")
    return {
        "capture_kind": capture["kind"],
        "pattern_sha256": checksum,
        "source_photos": len(capture["views"]),
        "vision_estimates": sum(m["source"] == "vision_model" for m in capture["measurements"].values()),
        "dxf_panels": dxf_inventory(root / "pattern/pattern.dxf", pattern),
        "seams": len(pattern["seams"]),
        "garment_openings": simulation["welded_opening_count"],
        "glb": glb_inventory(root / "model/garment.glb"),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = check_archive(args.archive)
    except (KeyError, TypeError, ValueError, OSError, ezdxf.DXFError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
