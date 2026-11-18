"""Fit metric projective cameras from surveyed 3D control points and image observations."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .compare_shape import check_frame
from .photo_points import supplied_matrices


def normalize(points):
    center = np.mean(points, axis=0)
    distance = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
    if distance <= 1e-9:
        raise ValueError("Calibration points need a measurable spatial spread")
    width = points.shape[1]
    scale = np.sqrt(width) / distance
    transform = np.eye(width + 1)
    transform[:width, :width] *= scale
    transform[:width, width] = -scale * center
    return (points - center) * scale, transform


def fit_projection(world_mm, image_px):
    """Solve a normalized DLT camera with millimetre coordinates."""
    if (world_mm.ndim != 2 or world_mm.shape[1] != 3 or image_px.shape != (len(world_mm), 2)
            or len(world_mm) < 6 or not np.isfinite(world_mm).all() or not np.isfinite(image_px).all()):
        raise ValueError("Provide at least six finite 3D points with matched 2D image positions")
    world, world_normalizer = normalize(world_mm)
    image, image_normalizer = normalize(image_px)
    point = np.column_stack((world, np.ones(len(world))))
    zero = np.zeros_like(point)
    rows = np.vstack((np.column_stack((point, zero, -image[:, 0, None] * point)),
                      np.column_stack((zero, point, -image[:, 1, None] * point))))
    _, singular, vectors = np.linalg.svd(rows, full_matrices=False)
    if singular[-2] <= singular[0] * 1e-5:
        raise ValueError("The 3D control points do not span enough depth and image area to estimate a camera")
    matrix = np.linalg.solve(image_normalizer, vectors[-1].reshape(3, 4) @ world_normalizer)
    if not np.isfinite(matrix).all() or abs(matrix[2, 3]) <= 1e-9 * np.linalg.norm(matrix[2]):
        raise ValueError("Use a metric datum in the reference scene, away from the camera plane")
    return matrix / matrix[2, 3]


def error_pixels(matrix, world, image):
    projected = np.column_stack((world, np.ones(len(world)))) @ matrix.T
    if np.any(projected[:, 2] <= 0):
        raise ValueError("Observed control points must share the camera half-space of the world datum")
    error = np.linalg.norm(projected[:, :2] / projected[:, 2:] - image, axis=1)
    return {"points": len(error), "median_px": round(float(np.median(error)), 6),
            "p90_px": round(float(np.quantile(error, 0.9)), 6), "max_px": round(float(error.max()), 6)}


def controls(rows, size):
    world = np.asarray([row["world_mm"] for row in rows], dtype=float).reshape(-1, 3)
    pixels = np.asarray([row["pixel"] for row in rows], dtype=float).reshape(-1, 2)
    if not np.isfinite(world).all() or not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels >= size):
        raise ValueError("Provide finite metric control points with observations inside the annotation raster")
    return world, pixels


def calibrate(source):
    raw = source.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("world_unit") != "mm":
        raise ValueError("Provide a schema_version 1 set of millimetre control observations")
    size = tuple(record["image_size_px"])
    if len(size) != 2 or any(type(value) is not int for value in size):
        raise ValueError("Provide the width and height of the annotation raster")
    check_frame(*size)
    item, provenance = record.get("item_id"), record.get("source")
    if not all(isinstance(value, str) and value.strip() for value in (item, provenance)):
        raise ValueError("Identify the garment and the independent source of the control geometry")
    views = record["views"]
    if not isinstance(views, dict) or not views:
        raise ValueError("Provide at least one photographed view")
    calibrated, hashes = {}, {}
    for name, view in views.items():
        photo = (source.parent / view["photo"]).read_bytes()
        digest = sha256(photo).hexdigest()
        if digest != view["photo_sha256"]:
            raise ValueError(f"{name}: photograph differs from the surveyed control observations")
        with Image.open(BytesIO(photo)) as image:
            if image.width * size[1] != image.height * size[0]:
                raise ValueError(f"{name}: the annotation raster must preserve the photograph aspect ratio")
        world, pixels = controls(view["fit"], size)
        reserved, reserved_px = controls(view.get("holdout", []), size)
        if {tuple(row) for row in world} & {tuple(row) for row in reserved}:
            raise ValueError(f"{name}: keep check points out of the camera fit")
        camera = fit_projection(world, pixels)
        calibrated[name] = {
            "photo_sha256": digest, "image_from_world": camera.tolist(), "fit": error_pixels(camera, world, pixels),
            "holdout": error_pixels(camera, reserved, reserved_px) if len(reserved) else {"points": 0},
        }
        hashes[name] = digest
    result = {"schema_version": 1, "item_id": item, "source": provenance, "world_unit": "mm",
              "image_size_px": list(size), "input_sha256": sha256(raw).hexdigest(), "views": calibrated}
    supplied_matrices(json.dumps(result).encode(), item, size, hashes)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = calibrate(args.controls)
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (KeyError, TypeError, ValueError, OSError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
