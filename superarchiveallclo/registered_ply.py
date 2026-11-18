"""Decode metric depth and pixel registration from an organized LiDAR PLY export."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np


@dataclass(frozen=True)
class RegisteredDepth:
    depth: np.ndarray
    intrinsics: dict
    diagnostics: dict


def point_records(raw):
    with BytesIO(raw) as stream:
        if stream.readline(100).strip() != b"ply" or stream.readline(100).strip() != b"format ascii 1.0":
            raise ValueError("Expected an ASCII PLY 1.0 export")
        vertex, count, fields = False, 0, []
        for _ in range(48):
            line = stream.readline(2048).decode("ascii").split()
            if not line:
                raise ValueError("Truncated PLY header")
            if line == ["end_header"]:
                break
            if line[0] == "element":
                if len(line) != 3 or not line[2].isdigit():
                    raise ValueError("Expected a labeled PLY element with a nonnegative count")
                vertex = line[1] == "vertex"
                if vertex:
                    if count:
                        raise ValueError("Expected a single PLY vertex element")
                    count = int(line[2])
                elif int(line[2]) != 0:
                    raise ValueError("Expected an organized point PLY without faces")
            if line[0] == "property" and vertex:
                if len(line) != 3:
                    raise ValueError("Expected scalar vertex properties")
                fields.append(line[-1])
        else:
            raise ValueError("PLY header is too long")
        required = ("x", "y", "z", "cx", "cy", "depth")
        if len(fields) != len(set(fields)) or not set(required).issubset(fields) or not 100 <= count <= 2_000_000:
            raise ValueError("Provide XYZ, integral cx/cy and metric depth for each PLY vertex")
        points = np.loadtxt(stream, usecols=[fields.index(name) for name in required], ndmin=2)
    if len(points) != count or not np.isfinite(points).all():
        raise ValueError("PLY point count or coordinates differ from the declared vertex grid")
    return points


def registered_depth(raw, rgb_size):
    """Use vendor depth values and infer the intrinsics' raster from the encoded rays."""
    rows = point_records(raw)
    world, xy, z = rows[:, :3], rows[:, 3:5], rows[:, 5]
    if np.any(xy < 0) or np.any(xy >= 8192) or not np.array_equal(xy, xy.round()):
        raise ValueError("Expected integral, nonnegative pixel addresses in cx and cy")
    pixels = xy.astype(int)
    size = pixels.max(axis=0) + 1
    rgb = np.asarray(rgb_size)
    if np.prod(size) != len(rows) or min(size) < 10 or rgb.shape != (2,) or np.any(rgb < size):
        raise ValueError("Provide a complete depth grid and the corresponding upright RGB raster size")
    x, y = pixels.T
    if len(np.unique(y * size[0] + x)) != len(rows):
        raise ValueError("Organized PLY includes duplicate depth pixels")
    valid = (z > 0) & (z < 25)
    if valid.sum() < 100:
        raise ValueError("Organized PLY needs supported depth returns in metres")

    # Recover camera translation, principal ray, and the two ray increments.
    # One fifth of the vertices remain outside this fit.
    design = np.column_stack((np.ones(len(z)), z, xy * z[:, None]))
    held = valid & (np.arange(len(z)) % 5 == 0)
    fit = valid & ~held
    if held.sum() < 10 or fit.sum() < 80:
        raise ValueError("Depth returns do not support both calibration and verification")
    coeff, _, rank, _ = np.linalg.lstsq(design[fit], world[fit], rcond=None)
    errors = np.linalg.norm(design[held] @ coeff - world[held], axis=1)
    residual = float(np.quantile(errors, .90))
    increments = coeff[2:]
    step = np.linalg.norm(increments, axis=1)
    directions = np.divide(increments, step[:, None], out=np.zeros_like(increments), where=step[:, None] > 0)
    forward = coeff[1] - (directions @ coeff[1]) @ directions
    if (rank != 4 or residual > 0.0005 or np.any(step < 1e-8)
            or not np.allclose(directions @ directions.T, np.eye(2), atol=.01)
            or not np.isclose(np.linalg.norm(forward), 1, atol=.02)):
        raise ValueError("PLY XYZ and recorded optical depth do not recover a consistent pinhole projection")
    focal = 1 / step
    principal = -(directions @ coeff[1]) / step
    # A central principal point identifies which raster the exporter used.
    # Some exports apply RGB-sized intrinsics directly to the depth indices.
    candidates = (("depth", size),) if np.array_equal(size, rgb) else (("depth", size), ("rgb", rgb))
    matches = [(name, raster) for name, raster in candidates
               if np.all(np.abs((principal + .5) / raster - .5) <= .16)
               and np.all((focal / raster >= .30) & (focal / raster <= 3.0))]
    if len(matches) != 1:
        raise ValueError("PLY projection does not identify an unambiguous depth or colour intrinsics raster")
    name, intrinsic_size = matches[0]
    factor = size / intrinsic_size
    if not np.isclose(factor[0], factor[1], rtol=0.01):
        raise ValueError("Provide the matching RGB field of view with the same aspect as the depth grid")
    depth = np.full(tuple(size[::-1]), np.nan, dtype=np.float32)
    depth[y[valid], x[valid]] = z[valid]
    f = focal * factor
    c = (principal + .5) * factor - .5
    return RegisteredDepth(depth, dict(zip(("fx", "fy", "cx", "cy"), np.r_[f, c].tolist(), strict=True)), {
        "depth_size_px": size.tolist(), "rgb_size_px": rgb.tolist(),
        "serialized_intrinsics_raster": name, "serialized_focal_px": focal.tolist(),
        "serialized_principal_px": principal.tolist(), "serialized_fit_held_out_vertices": int(held.sum()),
        "serialized_fit_held_out_p90_mm": round(residual * 1000, 6),
        "has_registered_depth_fraction": round(float(valid.mean()), 6),
    })
