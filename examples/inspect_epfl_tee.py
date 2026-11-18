"""Inspect paired-view EPFL T-shirt depth samples from the publisher's archive."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.rgbd import back_project

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


def inspect_members(read):
    calibration = np.loadtxt(BytesIO(checked(read, "camera_intrinsics.txt", INTRINSICS_SHA)))
    intrinsics = {"fx": calibration[0, 0], "fy": calibration[1, 1],
                  "cx": calibration[0, 2], "cy": calibration[1, 2]}
    summary = {}
    for (sequence, frame), (photo_hash, depth_hash) in FRAMES.items():
        photo_name = f"tshirt/images/{sequence}/rgb_{frame}.tiff"
        depth_name = f"tshirt/depth_maps/{sequence}/depth_{frame}.npz"
        with Image.open(BytesIO(checked(read, photo_name, photo_hash))) as photo:
            size = photo.size
        with np.load(BytesIO(checked(read, depth_name, depth_hash)), allow_pickle=False) as archive:
            depth = archive["depth"]
        if depth.shape != size[::-1]:
            raise ValueError("Published RGB and depth must share the same raster")
        valid = (depth > 0) & np.isfinite(depth)
        surface = back_project(depth, valid, intrinsics) * 1000
        centered = surface - surface.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        plane_residual = np.abs(centered @ basis[-1])
        summary[f"{sequence}/{frame}"] = {
            "photo": photo_name, "photo_sha256": photo_hash, "depth": depth_name, "depth_sha256": depth_hash,
            "image_size_px": list(size), "visible_points": len(surface),
            "camera_xyz_span_mm": np.ptp(surface, axis=0).round(1).tolist(),
            "best_plane_residual_p90_mm": round(float(np.quantile(plane_residual, 0.9)), 1),
        }
    return {"source_url": "https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/", "camera_intrinsics": calibration.tolist(),
            "intrinsics_sha256": INTRINSICS_SHA, "depth_unit_assumption": "m",
            "scope": "Visible cloth of separate held and deformed poses, measured on the registered 224 px grid.",
            "frames": summary}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Published ZIP or directory containing the extracted camera file and tshirt directory")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.is_dir():
        result = inspect_members(lambda name: (args.source / name).read_bytes())
    else:
        with ZipFile(args.source) as archive:
            result = inspect_members(lambda name: archive.read(f"{PREFIX}/{name}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {write_json(args.output, result)}")


if __name__ == "__main__":
    main()
