"""Prepare tilted flat-lay photographs with an independently measured plane reference."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image, ImageOps

from .capture import archive, number


def rectangle(width, height):
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)


def homography(source, target):
    """Fit the projective map between four planar control points."""
    def normalize(points):
        x, y = points.mean(axis=0)
        span = np.sqrt(np.mean(np.sum((points - (x, y)) ** 2, axis=1)))
        return np.array([[1 / span, 0, -x / span], [0, 1 / span, -y / span], [0, 0, 1]])

    before, after = normalize(source), normalize(target)
    x, y = project(before, source).T
    u, v = project(after, target).T
    one, zero = np.ones(4), np.zeros(4)
    equations = np.vstack((
        np.stack((x, y, one, zero, zero, zero, -x * u, -y * u, -u), axis=1),
        np.stack((zero, zero, zero, x, y, one, -x * v, -y * v, -v), axis=1),
    ))
    _, singular, basis = np.linalg.svd(equations)
    if singular[-1] < singular[0] * 1e-5:
        raise ValueError("The four reference corners are too foreshortened for a stable plane")
    return np.linalg.solve(after, basis[-1].reshape(3, 3) @ before)


def project(matrix, points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def rectify(image, reference, mm_per_pixel):
    image = ImageOps.exif_transpose(image).convert("RGB")
    mm_per_pixel = number(mm_per_pixel, "mm_per_pixel")
    width = number(reference["width_mm"], "reference width")
    height = number(reference["height_mm"], "reference height")
    corners = np.asarray(reference["corners_px"], dtype=float)
    if (corners.shape != (4, 2) or not np.isfinite(corners).all()
            or np.any(corners < 0) or np.any(corners >= image.size)):
        raise ValueError("Mark all four reference corners inside the upright source photograph")
    edges = np.roll(corners, -1, axis=0) - corners
    following = np.roll(edges, -1, axis=0)
    if np.any(edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0] <= 0):
        raise ValueError("Mark a convex rectangle clockwise, starting at its top-left in garment coordinates")

    field = np.array(reference.get("field_mm", [0, 0, width, height]), dtype=float)
    if field.shape != (4,) or not np.isfinite(field).all() or np.any(field[2:] <= field[:2]):
        raise ValueError("Provide field_mm as left, top, right and bottom in millimetres")
    size = np.floor((field[2:] - field[:2]) / mm_per_pixel + 1e-9)
    if not np.isfinite(size).all() or np.any(size < 1) or np.prod(size) > 16_000_000:
        raise ValueError("Choose a sampling scale that yields a nonempty output of at most 16 megapixels")
    output_size = tuple(size.astype(int).tolist())
    millimetres_to_source = homography(rectangle(width, height), corners)
    output_to_mm = np.array([[mm_per_pixel, 0, field[0]], [0, mm_per_pixel, field[1]], [0, 0, 1]])
    mapping = millimetres_to_source @ output_to_mm
    output_corners = rectangle(*output_size)
    denominators = np.column_stack((output_corners, np.ones(4))) @ mapping[2]
    if (min(abs(denominators)) < max(abs(denominators)) * 1e-6
            or np.any(np.sign(denominators) != np.sign(denominators[0]))):
        raise ValueError("The requested field crosses the camera's plane horizon")
    footprint = project(mapping, output_corners)
    if np.any(footprint < -1e-6) or np.any(footprint > np.array(image.size) + 1e-6):
        raise ValueError("The requested field extends outside the source photograph")
    mapping /= mapping[2, 2]
    result = image.transform(output_size, Image.Transform.PERSPECTIVE, mapping.ravel()[:8],
                             Image.Resampling.BICUBIC)
    return result, {
        "origin_type": "planar_reference", "mm_per_pixel": mm_per_pixel,
        "reference_plane": {"width_mm": width, "height_mm": height, "corners_px": corners.tolist()},
        "requested_field_mm": field.tolist(),
        "field_mm": np.r_[field[:2], field[:2] + size * mm_per_pixel].tolist(),
        "source_upright_px": list(image.size), "output_to_source_px": mapping.tolist(),
        "extrapolated_beyond_reference": bool(np.any(field[:2] < 0) or np.any(field[2:] > [width, height])),
    }


def prepare(source, output, *, weights=None, blender="blender", mm_per_pixel=1.0, photo_appearance=False):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if photo_appearance and weights is None:
        raise ValueError("Photo appearance requires --model")
    record = json.loads(source.read_bytes())
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Provide a planar photograph schema_version 1 capture")
    item_id, provenance = (record.get(key) for key in ("item_id", "source"))
    if not all(isinstance(value, str) and value.strip() for value in (item_id, provenance)):
        raise ValueError("Identify the garment and the independently measured plane reference")
    views = record["views"]
    if not isinstance(views, dict) or set(views) not in ({"front"}, {"front", "back"}):
        raise ValueError("Provide a front photograph and, when available, a separate back photograph")
    if photo_appearance and "back" not in views:
        raise ValueError("Photo appearance requires separate front and back photographs")
    kind = record.get("kind", "physical")
    if kind not in ("physical", "synthetic_fixture"):
        raise ValueError("Capture kind must be physical or synthetic_fixture")
    session = {"schema_version": 1, "kind": kind, "garment": "crew_neck_tee", "item_id": item_id,
               "input_mode": "paired" if "back" in views else "front_only", "views": {}}
    for key in ("body_profile", "material"):
        if key in record:
            session[key] = record[key]

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "rectified"
        (bundle / "evidence").mkdir(parents=True)
        hashes = set()
        for name, view in views.items():
            raw = (source.parent / view["rgb"]).read_bytes()
            digest = sha256(raw).hexdigest()
            if digest in hashes:
                raise ValueError("Front and back must be separate original photographs")
            hashes.add(digest)
            with Image.open(BytesIO(raw)) as photo:
                suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(photo.format)
                if suffix is None:
                    raise ValueError("Use JPEG or PNG original photographs")
                image, calibration = rectify(photo, view["reference_plane"], mm_per_pixel)
            evidence = f"evidence/{name}-rgb{suffix}"
            (bundle / evidence).write_bytes(raw)
            filename = f"{name}-rectified.png"
            image.save(bundle / filename)
            session["views"][name] = {"file": filename, "calibration": {**calibration, "source": provenance},
                                      "source_assets": {"rgb": {"file": evidence, "sha256": digest}}}
        (bundle / "session.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf8")
        if weights is not None:
            archive(bundle / "session.json", bundle / "archive", blender, weights, blind=True,
                    photo_appearance=photo_appearance)
        bundle.rename(output)
    return session


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Generate a blind archive using matching front-only or paired weights")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--mm-per-pixel", type=float, default=1.0)
    parser.add_argument("--photo-appearance", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.session, args.output, weights=args.model, blender=args.blender, mm_per_pixel=args.mm_per_pixel,
                photo_appearance=args.photo_appearance)
    except (KeyError, TypeError, ValueError, OSError, DXFError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    print(f"Prepared {args.output / 'session.json'}")


if __name__ == "__main__":
    main()
