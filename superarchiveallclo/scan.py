"""Project a segmented metric flat-lay scan and optionally reconstruct a tee."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image
from scipy import ndimage

from .capture import archive
from .vision import picture_features

AXES = ("x", "y", "z", "-x", "-y", "-z")
PCD_TYPES = {"F": "f", "U": "u", "I": "i"}


def read_pcd(encoded):
    with BytesIO(encoded) as stream:
        header = {}
        for _ in range(32):
            fields = stream.readline(8192).decode("ascii").split()
            if not fields or fields[0].startswith("#"):
                continue
            key, *values = fields
            header[key] = values
            if key == "DATA":
                break
        columns = header.get("FIELDS", [])
        if len(columns) != len(set(columns)) or not {"x", "y", "z"}.issubset(columns):
            raise ValueError("PCD requires unique fields including x, y and z")
        count = int(header["POINTS"][0])
        components = tuple(map(int, header.get("COUNT", ["1"] * len(columns))))
        if not 100 <= count <= 5_000_000 or any(not 1 <= n <= 32 for n in components):
            raise ValueError("PCD has an unsupported point or component count")
        positions = [columns.index(axis) for axis in "xyz"]
        if any(components[i] != 1 for i in positions):
            raise ValueError("Each PCD coordinate must have one component")
        if header["DATA"] == ["binary"]:
            dtype = np.dtype([
                (name, f"<{PCD_TYPES[kind]}{size}", (n,))
                for name, kind, size, n in zip(columns, header["TYPE"], header["SIZE"], components, strict=True)
            ])
            payload = stream.read()
            if len(payload) != count * dtype.itemsize:
                raise ValueError("PCD binary length differs from its declared point count")
            records = np.frombuffer(payload, dtype=dtype)
            points = np.column_stack([records[axis][:, 0] for axis in "xyz"])
        elif header["DATA"] == ["ascii"]:
            offsets = np.cumsum((0,) + components[:-1])
            points = np.loadtxt(stream, usecols=tuple(offsets[i] for i in positions), ndmin=2)
            if len(points) != count:
                raise ValueError("PCD text length differs from its declared point count")
        else:
            raise ValueError("PCD must use ASCII or uncompressed binary data")
    points = points[np.isfinite(points).all(axis=1)].astype(float)
    if len(points) < 100:
        raise ValueError("PCD needs at least 100 finite XYZ points")
    return points


def in_world(points, encoded):
    transform = np.asarray(json.loads(encoded), dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("World-to-camera transform must be a finite 4×4 matrix")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("World-to-camera transform must be rigid and preserve scale")
    return (points - transform[:3, 3]) @ rotation


def project(points_m, right, down, mm_per_pixel):
    x_axis, y_axis = "xyz".index(right[-1]), "xyz".index(down[-1])
    if x_axis == y_axis:
        raise ValueError("Projection right and down need distinct axes")
    if not 1 <= mm_per_pixel <= 4:
        raise ValueError("Projection resolution must be between 1 and 4 mm per pixel")
    points = points_m * 1000
    projected = points[:, (x_axis, y_axis)] * [-1 if axis.startswith("-") else 1 for axis in (right, down)]
    normal = points[:, 3 - x_axis - y_axis]
    plane_distance = float(np.quantile(np.abs(normal - np.median(normal)), 0.95))
    if plane_distance > 35:
        raise ValueError("Scan departs more than 35 mm from the flat-lay plane at P95")

    low, high = np.quantile(projected, [0.001, 0.999], axis=0)
    span = high - low
    if np.any(span < 100) or np.any(span > 2500):
        raise ValueError("Scan dimensions are outside the flat-lay range. Check that XYZ units are metres")
    keep = np.all((projected >= low) & (projected <= high), axis=1)
    padding = 12
    pixels = np.rint((projected[keep] - low) / mm_per_pixel).astype(int) + padding
    width, height = np.ceil(span / mm_per_pixel).astype(int) + 2 * padding + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[pixels[:, 1], pixels[:, 0]] = True
    # Bridge the gaps between neighbouring returns without filling enclosed cut-outs.
    mask = ndimage.distance_transform_edt(~mask) <= 5.0 / mm_per_pixel
    mask = ndimage.binary_closing(mask, iterations=2)
    image = Image.fromarray(np.where(mask, 40, 232).astype(np.uint8))
    return image, {
        "type": "metric_orthographic_point_cloud",
        "right_axis": right,
        "down_axis": down,
        "point_count": len(points),
        "plane_distance_p95_mm": round(plane_distance, 3),
        "raw_point_span_p0_1_to_p99_9_mm": span.round(3).tolist(),
    }


def reconstruct(cloud, output, *, source, item_id, transform=None, right="x", down="y",
                mm_per_pixel=2.0, weights=None, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not source.strip():
        raise ValueError("Identify the scan source")
    raw = cloud.read_bytes()
    points = read_pcd(raw)
    world_transform = transform.read_bytes() if transform else None
    if world_transform is not None:
        points = in_world(points, world_transform)
    image, evidence = project(points, right, down, mm_per_pixel)
    calibration = {
        "source": source,
        "mm_per_pixel": mm_per_pixel,
        "cloud_sha256": sha256(raw).hexdigest(),
        **evidence,
    }
    if world_transform is not None:
        calibration["world_to_camera_sha256"] = sha256(world_transform).hexdigest()
    _, silhouette = picture_features(image, {"calibration": calibration, "mm_per_pixel": mm_per_pixel})
    session = {
        "schema_version": 1,
        "kind": "physical",
        "garment": "crew_neck_tee",
        "input_mode": "front_only",
        "item_id": item_id,
        "views": {"front": {"file": "projection.png", "calibration": calibration}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "scan"
        evidence_dir = bundle / "evidence"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "source.pcd").write_bytes(raw)
        if world_transform is not None:
            (evidence_dir / "world-to-camera.json").write_bytes(world_transform)
        image.save(bundle / "projection.png")
        (bundle / "projection.json").write_text(json.dumps({**calibration, **silhouette}, indent=2) + "\n")
        (bundle / "session.json").write_text(json.dumps(session, indent=2) + "\n")
        if weights is not None:
            archive(bundle / "session.json", bundle / "archive", blender, weights, blind=True)
        bundle.rename(output)
    return silhouette


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cloud", type=Path, help="Segmented PCD with XYZ coordinates in metres")
    parser.add_argument("--world-to-camera", type=Path, help="Optional JSON 4×4 rigid transform")
    parser.add_argument("--right", choices=AXES, default="x")
    parser.add_argument("--down", choices=AXES, default="y")
    parser.add_argument("--source", required=True, help="Provenance of the metric scale and capture")
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--mm-per-pixel", type=float, default=2)
    parser.add_argument("--model", type=Path, help="Optional single-front-view weights for candidate reconstruction")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = reconstruct(args.cloud, args.output, source=args.source, item_id=args.item_id,
                             transform=args.world_to_camera, right=args.right, down=args.down,
                             mm_per_pixel=args.mm_per_pixel, weights=args.model, blender=args.blender)
    except (KeyError, TypeError, ValueError, OSError, DXFError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
