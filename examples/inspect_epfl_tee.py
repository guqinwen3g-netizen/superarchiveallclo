"""Inspect paired-view EPFL T-shirt depth samples from the publisher's archive."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import re

import numpy as np
from PIL import Image

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.rgbd import back_project
from superarchiveallclo.scan import FLAT_PLANE_DISTANCE_P95_MM

PREFIX = "textureless_deformable_surfaces"
INTRINSICS_SHA = "17f4e9aa779f5458cc00cf1b2cdfe0c14fa81904fc8b591ce8c7506fad6056f7"
FRAMES = {
    ("Ld_front", "0047"): ("a9177117bab4611c6e40c247c9831686a155b1dc6c2b4ebe554f5676757f1037",
                            "c24b727db2de948c100e90ea4e5ec599ab8cf57988b211832ba6c6d377919afa"),
    ("Ld_back", "0054"): ("8ed78ac335e9c52a1be6ce0e12d6f7e11bbf524cb714057fa7fdb89658875ac7",
                           "ee38b0a086331c762ee859e49536ad35157b2a089adc7c535f6b5f3bd2054e6c"),
    ("Ll_front", "0423"): ("42ade81bf7b8ba2848ba960e071f5ba469d39461e00d3a0a8600b896b69c5149",
                            "2fe915ca629d54fd72e73589f81069c3a5d74cd4f958cf45a3369fc5a4a5fcdc"),
    ("Ll_back", "0388"): ("d646245e7c34d38d1f68a124842cf82ddb10e498eefc15bda711466713aae569",
                           "a05dd89639d45d625aac10dea9ee2970f0fc0cb708ca59be606b2eada29fb910"),
}


def checked(read, name, expected):
    data = read(name)
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected bytes for the published EPFL member {name}")
    return data


def observed_surface(raw, intrinsics):
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        depth = archive["depth"]
    valid = (depth > 0) & np.isfinite(depth)
    surface = back_project(depth, valid, intrinsics) * 1000
    if len(surface) < 3:
        raise ValueError("Insufficient depth points for a plane")
    centered = surface - surface.mean(axis=0)
    _, span, basis = np.linalg.svd(centered, full_matrices=False)
    if span[1] < 1e-6:
        raise ValueError("Depth points do not span a plane")
    signed = centered @ basis[-1]
    return depth.shape[::-1], surface, signed


def screen_flatness(read, entries, intrinsics):
    views = {}
    pattern = re.compile(r"tshirt/depth_maps/(L[cdlr]_(?:front(?:_big)?|back))/depth_(\d+)\.npz")
    for name in sorted(entries):
        found = pattern.fullmatch(name)
        if found is None:
            continue
        sequence, frame = found.groups()
        raw = read(name)
        row = {"frame": frame, "depth_sha256": sha256(raw).hexdigest()}
        try:
            _, points, signed = observed_surface(raw, intrinsics)
        except ValueError as reason:
            row["rejected"] = str(reason)
        else:
            row.update(visible_points=len(points),
                       centered_plane_p95_mm=round(float(np.quantile(np.abs(signed - np.median(signed)), 0.95)), 3))
        views.setdefault(sequence, []).append(row)
    return {"limit_mm": FLAT_PLANE_DISTANCE_P95_MM, "sample_count": sum(map(len, views.values())),
            "sequences": {name: sorted(frames, key=lambda row: row.get("centered_plane_p95_mm", float("inf")))
                          for name, frames in views.items()},
            "scope": "Depth-only plane screening before RGB correspondence and paired garment checks."}


def inspect_members(read, entries=None):
    calibration = np.loadtxt(BytesIO(checked(read, "camera_intrinsics.txt", INTRINSICS_SHA)))
    intrinsics = {"fx": calibration[0, 0], "fy": calibration[1, 1],
                  "cx": calibration[0, 2], "cy": calibration[1, 2]}
    summary = {}
    for (sequence, frame), (photo_hash, depth_hash) in FRAMES.items():
        photo_name = f"tshirt/images/{sequence}/rgb_{frame}.tiff"
        depth_name = f"tshirt/depth_maps/{sequence}/depth_{frame}.npz"
        with Image.open(BytesIO(checked(read, photo_name, photo_hash))) as photo:
            size = photo.size
        depth_size, surface, signed = observed_surface(checked(read, depth_name, depth_hash), intrinsics)
        if depth_size != size:
            raise ValueError("Published RGB and depth must share the same raster")
        summary[f"{sequence}/{frame}"] = {
            "photo": photo_name, "photo_sha256": photo_hash, "depth": depth_name, "depth_sha256": depth_hash,
            "image_size_px": list(size), "visible_points": len(surface),
            "camera_xyz_span_mm": np.ptp(surface, axis=0).round(1).tolist(),
            "best_plane_residual_p90_mm": round(float(np.quantile(np.abs(signed), 0.9)), 1),
        }
    report = {"source_url": "https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/", "camera_intrinsics": calibration.tolist(),
              "intrinsics_sha256": INTRINSICS_SHA, "depth_unit_assumption": "m",
              "scope": "Visible cloth of separate held and deformed poses, measured on the registered 224 px grid.",
              "frames": summary}
    if entries is not None:
        report["screening"] = screen_flatness(read, entries, intrinsics)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Published ZIP or directory containing the extracted camera file and tshirt directory")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--screen", action="store_true", help="Rank the available front and back depth frames by planarity")
    args = parser.parse_args()
    if args.source.is_dir():
        entries = (path.relative_to(args.source).as_posix() for path in args.source.glob("tshirt/depth_maps/*/*.npz"))
        result = inspect_members(lambda name: (args.source / name).read_bytes(), entries if args.screen else None)
    else:
        with ZipFile(args.source) as archive:
            entries = (name.removeprefix(f"{PREFIX}/") for name in archive.namelist()) if args.screen else None
            result = inspect_members(lambda name: archive.read(f"{PREFIX}/{name}"), entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {write_json(args.output, result)}")


if __name__ == "__main__":
    main()
