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
from superarchiveallclo.check_archive import check_archive
from superarchiveallclo.compare_flat import compare_flat
from superarchiveallclo.compare_rgbd import camera_from_flat_pose, compare as compare_depth
from superarchiveallclo.rgbd import back_project, prepare, read_depth

FRAMES = {
    "front": ("Ld_front", "0582", "faf49bcbda3b22f68cdbe27898a882836f24cf2a169af9786e596abe4eae0547",
              "fe46f11549e0d5b97ea2613b85fdcc7a7fea9320fd26f1e05ceb35e86cfc1a46"),
    "back": ("Ld_back", "0216", "a7b008ab9966398a1ac1d08ae4f0adb508bac347d37d27676103eb673f42bad2",
             "baa8be385c911b1ff9fc89eada7bce02ec540a33923d2eaac007abca82b326ed"),
}
HOLDOUT = {
    "0576": "f11d524e463888a602d6e6f9e8906f562a745bee5d21c6e04369a68fe9a8738f",
    "0577": "8a174e65e2073ffbf62295134e81834775d0ed63aa31ee7378df5dda80d2b35b",
}
ITEM = "epfl-tds-tee"


def leveled_depth(raw, camera):
    z = read_depth(raw, ".npz")
    valid = np.isfinite(z) & (z > 0)
    points = back_project(z, valid, camera)
    origin = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
    normal = axes[-1] * np.sign(axes[-1, 2])
    right = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    right /= np.linalg.norm(right)
    basis = np.column_stack((right, np.cross(normal, right), normal))
    return valid, origin, basis, (points - origin) @ basis


def lower_body_chords(points):
    """Measure visible lower-body widths while retaining the observed depth variation."""
    low, high = np.quantile(points[:, 1], [0.001, 0.999])
    widths = []
    for position in (0.86, 0.89, 0.92, 0.95):
        row = points[np.abs(points[:, 1] - (low + position * (high - low))) <= 0.008]
        if len(row) < 12:
            continue
        order = row[row[:, 0].argsort()]
        edges = max(2, len(order) // 25)
        left, right = np.median(order[:edges], axis=0), np.median(order[-edges:], axis=0)
        widths.append({"position_down_garment": position,
                       "visible_3d_chord_mm": round(float(np.linalg.norm(right - left) * 1000), 2)})
    return widths


def check_depth(read, output, camera, garment, blender):
    data = output / "depth-reference"
    data.mkdir()
    clouds, frames, chords = [], {}, {}
    for number, digest in HOLDOUT.items():
        name = f"depth-{number}.npz"
        raw = checked(read, f"tshirt/depth_maps/Ld_front/depth_{number}.npz", digest)
        (data / name).write_bytes(raw)
        _, origin, basis, points = leveled_depth(raw, camera)
        cloud = data / f"visible-{number}.pcd"
        with cloud.open("wb") as stream:
            stream.write((f"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                          f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n").encode())
            stream.write(points.astype("<f4").tobytes())
        clouds.append(cloud)
        frames[number] = (name, digest, origin, basis)
        chords[number] = lower_body_chords(points)
    comparison = output / "flat-comparison"
    flat = compare_flat(garment, clouds, comparison, item_id=ITEM, blender=blender,
                        setup="Withheld EPFL foreground depths of separately manipulated views, best-fit plane per camera.")
    views = {}
    for (number, (name, digest, origin, basis)), measurement in zip(frames.items(), flat["references"], strict=True):
        views[f"frame-{number}"] = {"depth": name, "depth_sha256": digest, "depth_unit": "m", "intrinsics": camera,
                                   "model_to_camera_m": camera_from_flat_pose(origin, basis, measurement["reference_translation_mm"]),
                                   "pose_source": "Cloth-plane orientation and same-scale translation to the observed scan before ray scoring."}
    reference = write_json(data / "independent-depth.json", {"schema_version": 1, "item_id": ITEM, "views": views})
    rays = compare_depth(garment, reference, output / "ray-comparison", flat_comparison=comparison, blender=blender)
    return {"flat": flat, "rays": rays, "visible_lower_body_chords": chords}


def prepare_pair(read, output, model, blender, *, compare_heldout=False):
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
        valid, origin, basis, _ = leveled_depth(depth_bytes, camera)
        Image.fromarray(valid.astype(np.uint8) * 255).save(data / f"{name}-mask.png")
        session[name] = {"rgb": f"{name}.tiff", "depth": f"{name}.npz", "depth_unit": "m",
                         "mask": f"{name}-mask.png", "intrinsics": camera, "support_plane": {
                             "source": "Experimental best-fit plane of the camera's measured visible cloth",
                             "origin_camera_m": origin.tolist(), "basis_camera": basis.tolist(),
                         }}
        sources[name] = {"rgb": rgb, "depth": depth, "rgb_sha256": rgb_hash, "depth_sha256": depth_hash}
    source = write_json(data / "session.json", {"schema_version": 1, "item_id": ITEM,
                                              "source": "EPFL Texture-less Deformable Surfaces, registered depth and published focal lengths",
                                              "views": session})
    projected = output / "prepared"
    views = prepare(source, projected)
    try:
        capture, _ = capture_observations(projected / "session.json", model_path=model, blind=True)
    except ValueError as reason:
        result = {"status": "model_rejected", "reason": str(reason)}
    else:
        garment = output / "archive"
        archive(projected / "session.json", garment, blender, model, blind=True)
        result = {"status": "candidate_archive", "inference": capture["inference"], "audit": check_archive(garment)}
        if compare_heldout:
            result["heldout_depth"] = check_depth(read, output, camera, garment, blender)
    return write_json(output / "study.json", {"physical_sources": sources, "observed_views": views, "result": result,
        "scope": "Exploratory leveling of separately manipulated front and back views. No independent cut dimensions."})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PAIRED_WEIGHTS)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--check-depth", action="store_true", help="Check two additional held-out raw depth frames")
    args = parser.parse_args()
    if args.source.is_dir():
        report = prepare_pair(lambda name: (args.source / name).read_bytes(), args.output, args.model, args.blender,
                              compare_heldout=args.check_depth)
    else:
        with ZipFile(args.source) as pack:
            report = prepare_pair(lambda name: pack.read(f"{PREFIX}/{name}"), args.output, args.model, args.blender,
                                  compare_heldout=args.check_depth)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
