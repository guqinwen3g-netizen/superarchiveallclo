"""Probe the paired tee model on near-planar registered EPFL front and back frames."""

from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
from PIL import Image

from examples.evaluate_auckland_tees import PAIRED_WEIGHTS, write_json
from examples.inspect_epfl_tee import INTRINSICS_SHA, PREFIX, checked
from superarchiveallclo.capture import archive, capture_observations
from superarchiveallclo.rgbd import back_project, prepare, read_depth

FRAMES = {
    "front": ("Ld_front", "0582", "faf49bcbda3b22f68cdbe27898a882836f24cf2a169af9786e596abe4eae0547",
              "fe46f11549e0d5b97ea2613b85fdcc7a7fea9320fd26f1e05ceb35e86cfc1a46"),
    "back": ("Ld_back", "0216", "a7b008ab9966398a1ac1d08ae4f0adb508bac347d37d27676103eb673f42bad2",
             "baa8be385c911b1ff9fc89eada7bce02ec540a33923d2eaac007abca82b326ed"),
}


def prepare_pair(read, output, model, blender):
    data = output / "data"
    data.mkdir(parents=True)
    k = np.loadtxt(BytesIO(checked(read, "camera_intrinsics.txt", INTRINSICS_SHA)))
    camera = {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}
    session, sources = {}, {}
    for name, (series, frame, rgb_hash, depth_hash) in FRAMES.items():
        rgb = f"tshirt/images/{series}/rgb_{frame}.tiff"
        depth = f"tshirt/depth_maps/{series}/depth_{frame}.npz"
        rgb_bytes, depth_bytes = checked(read, rgb, rgb_hash), checked(read, depth, depth_hash)
        (data / f"{name}.tiff").write_bytes(rgb_bytes)
        (data / f"{name}.npz").write_bytes(depth_bytes)
        z = read_depth(depth_bytes, ".npz")
        valid = np.isfinite(z) & (z > 0)
        Image.fromarray(valid.astype(np.uint8) * 255).save(data / f"{name}-mask.png")
        points = back_project(z, valid, camera)
        origin = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
        normal = axes[-1] * np.sign(axes[-1, 2])
        right = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        right /= np.linalg.norm(right)
        basis = np.column_stack((right, np.cross(normal, right), normal))
        session[name] = {"rgb": f"{name}.tiff", "depth": f"{name}.npz", "depth_unit": "m",
                         "mask": f"{name}-mask.png", "intrinsics": camera, "support_plane": {
                             "source": "Experimental best-fit plane of the camera's measured visible cloth",
                             "origin_camera_m": origin.tolist(), "basis_camera": basis.tolist(),
                         }}
        sources[name] = {"rgb": rgb, "depth": depth, "rgb_sha256": rgb_hash, "depth_sha256": depth_hash}
    source = write_json(data / "session.json", {"schema_version": 1, "item_id": "epfl-tds-tee",
                                              "source": "EPFL Texture-less Deformable Surfaces, registered depth and published focal lengths",
                                              "views": session})
    projected = output / "prepared"
    views = prepare(source, projected)
    try:
        capture, _ = capture_observations(projected / "session.json", model_path=model, blind=True)
    except ValueError as reason:
        result = {"status": "model_rejected", "reason": str(reason)}
    else:
        archive(projected / "session.json", output / "archive", blender, model, blind=True)
        result = {"status": "candidate_archive", "inference": capture["inference"]}
    return write_json(output / "study.json", {"physical_sources": sources, "observed_views": views, "result": result,
        "scope": "Exploratory leveling of separately manipulated front and back views. No independent cut dimensions."})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PAIRED_WEIGHTS)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    if args.source.is_dir():
        report = prepare_pair(lambda name: (args.source / name).read_bytes(), args.output, args.model, args.blender)
    else:
        with ZipFile(args.source) as pack:
            report = prepare_pair(lambda name: pack.read(f"{PREFIX}/{name}"), args.output, args.model, args.blender)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
