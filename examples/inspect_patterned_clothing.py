"""Measure the publisher's printed pattern mask and verify one real registered shirt frame."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage


VERSION = "3650cb5c2842a3b4af144bab8b24863af45eb240"
ROOT = f"https://raw.githubusercontent.com/facebookresearch/PatternedClothing/{VERSION}/shirt/model"
SERIES = ("https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/"
          "PatternedCloth/shirt/subject_00.zip")
FRAME = "subject_00/003577.ply"
ARCHIVE_SIZE = 8_109_304_731
HASHES = {
    "mask.png": "61a3269120e99ae20fde1553011277f1cda1668461b41e36a1a6adb5f8d035cf",
    "template.ply": "797b5db802b1b955e06e9cc915939a7eeb2bfbaa82d9e7ed413d80ee151cae6e",
    FRAME: "c6b7e240b66db1231a88b4e9c8b9209bd2e058dbd2ec4827654c899b5abe6907",
}
PRINTED_CELL_MM = 2.7


def download_range(start, end):
    request = Request(f"{SERIES}?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=30) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
            raise ValueError("The publisher archive did not return the expected byte range")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("The publisher archive returned an incomplete byte range")
    return data


def selected_frame():
    tail_start = ARCHIVE_SIZE - 1_000_000
    with TemporaryFile() as sparse_archive:
        sparse_archive.seek(tail_start)
        sparse_archive.write(download_range(tail_start, ARCHIVE_SIZE - 1))
        with ZipFile(sparse_archive) as archive:
            frame = archive.getinfo(FRAME)
            if frame.header_offset != 69 or frame.compress_size != 2_052_739:
                raise ValueError("The publisher frame moved within the pinned ZIP")
            sparse_archive.seek(0)
            sparse_archive.write(download_range(0, 2_052_886))
            return archive.read(frame), len(archive.infolist()) - 1


def pattern_cells(encoded):
    mask = np.asarray(Image.open(BytesIO(encoded)).convert("L")) > 0
    labels, _ = ndimage.label(mask)
    boxes = ndimage.find_objects(labels)
    order = ("collar", "front", "back", "sleeve_upper_on_board", "sleeve_lower_on_board")
    return {
        name: {
            "bounds_xyxy_cell_exclusive": [x.start, y.start, x.stop, y.stop],
            "board_width_height_mm": [round((axis.stop - axis.start) * PRINTED_CELL_MM, 2) for axis in (x, y)],
            "visible_cells": int((labels[y, x] == index).sum()),
        }
        for name, (index, (y, x)) in zip(order, enumerate(boxes, 1), strict=True)
    }


def mesh_metrics(encoded):
    header, body = encoded.split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    if lines[:2] != ["ply", "format binary_big_endian 1.0"]:
        raise ValueError("Unexpected publisher PLY encoding")
    n = int(next(line.split()[-1] for line in lines if line.startswith("element vertex ")))
    m = int(next(line.split()[-1] for line in lines if line.startswith("element face ")))
    points = np.frombuffer(body, dtype=">f4", count=3 * n).reshape(n, 3)
    properties = [("count", "u1"), ("vertices", ">i4", (3,))]
    uvs = "property list uchar float texcoord" in lines
    if uvs:
        properties.extend([("uv_count", "u1"), ("uv", ">f4", (6,))])
    triangles = np.frombuffer(body, dtype=properties, offset=12 * n)
    indices = triangles["vertices"]
    if (len(triangles) != m or np.any(triangles["count"] != 3) or not np.isfinite(points).all()
            or indices.min() < 0 or indices.max() >= n or (uvs and np.any(triangles["uv_count"] != 6))):
        raise ValueError("Invalid triangle or vertex in publisher PLY")
    edges = np.linalg.norm(points[indices] - points[np.roll(indices, 1, axis=1)], axis=2)
    return {
        "vertices": n, "triangles": m, "face_indices_sha256": sha256(indices.tobytes()).hexdigest(),
        "vertex_axis_spans_native": np.ptp(points.astype(float), axis=0).round(2).tolist(),
        "triangle_edge_native_p05_p50_p95": np.quantile(edges, [0.05, 0.5, 0.95]).round(4).tolist(),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Use local mask.png, template.ply and optionally 003577.ply")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    assets = {}
    for name in ("mask.png", "template.ply"):
        if args.source:
            assets[name] = (args.source / name).read_bytes()
        else:
            with urlopen(f"{ROOT}/{name}", timeout=30) as response:
                assets[name] = response.read()
    if args.source and (args.source / Path(FRAME).name).is_file():
        assets[FRAME] = (args.source / Path(FRAME).name).read_bytes()
        frame_count = None
    else:
        assets[FRAME], frame_count = selected_frame()
    if any(sha256(data).hexdigest() != HASHES[name] for name, data in assets.items()):
        raise ValueError("Publisher asset does not match the pinned SHA-256")

    template, sample = (mesh_metrics(assets[name]) for name in ("template.ply", FRAME))
    report = {
        "publisher_commit": VERSION, "license": "CC BY-NC 4.0", "asset_sha256": HASHES,
        "printed_cell_mm": PRINTED_CELL_MM, "printed_visible_panel_bounds": pattern_cells(assets["mask.png"]),
        "registered_frame": FRAME, "number_of_registered_frames": frame_count,
        "template_mesh": template, "registered_mesh": sample,
        "same_indexed_triangles": template["face_indices_sha256"] == sample["face_indices_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
