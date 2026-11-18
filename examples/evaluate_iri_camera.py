"""Verify metric camera registration using the real IRI Kinect frame n00764."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image

from examples.evaluate_auckland_tees import write_json
from examples.evaluate_iri_tee import FRAMES, ITEM, checked, download
from superarchiveallclo.calibrate_cameras import calibrate
from superarchiveallclo.scan import read_pcd

FRAME = "n00764"
MASK_SHA256 = "754a4b32b9379bc968033947789aa1f535dcfb53e96232f0a23547614a1447a6"


def evaluate(output, source):
    data = output / "source"
    data.mkdir(parents=True)
    if source is None:
        download(data, (FRAME,))
    else:
        for suffix in (".png", ".pcd", "-mask.png"):
            name = FRAME + suffix
            copyfile(source / name, data / name)
    photo = data / f"{FRAME}.png"
    photo_hash = FRAMES[FRAME]["img"][2]
    cloud_hash = FRAMES[FRAME]["pcd"][2]
    checked(photo.read_bytes(), photo_hash)
    geometry = checked((data / f"{FRAME}.pcd").read_bytes(), cloud_hash)
    mask = data / f"{FRAME}-mask.png"
    checked(mask.read_bytes(), MASK_SHA256)
    with Image.open(mask) as image:
        garment = np.asarray(image.convert("L")) > 127
    # The known metric datum is one metre in front of the Kinect. The raw
    # coordinate origin is at the camera, where a projection is singular.
    datum = [0, 0, 1000]
    world = read_pcd(geometry, organized=True) * 1000 - datum
    if garment.shape != world.shape[:2]:
        raise ValueError("The point cloud and mask must share the registered raster")
    y, x = np.indices(garment.shape)
    valid = np.isfinite(world).all(axis=2) & (world[:, :, 2] > -datum[2])
    outside = valid & ~garment & (x % 23 == 7) & (y % 23 == 6)
    withheld = valid & garment & (x % 23 == 11) & (y % 23 == 12)
    pixels = np.stack((x, y), axis=-1)
    rows = {}
    for name, selected in (("fit", outside), ("holdout", withheld)):
        rows[name] = [{"world_mm": xyz.tolist(), "pixel": uv.tolist()}
                      for xyz, uv in zip(world[selected], pixels[selected], strict=True)]
    observations = write_json(output / "controls.json", {
        "schema_version": 1, "item_id": ITEM, "world_unit": "mm", "image_size_px": list(garment.shape[::-1]),
        "source": "IRI UPC-CSIC Clothing Part Dataset Kinect registered XYZ, frame n00764. Background fit and cloth holdout.",
        "camera_frame_datum_mm": datum, "source_pcd_sha256": cloud_hash, "source_mask_sha256": MASK_SHA256,
        "views": {FRAME: {"photo": f"source/{photo.name}", "photo_sha256": photo_hash, **rows}},
    })
    return write_json(output / "camera.json", calibrate(observations))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Use previously downloaded n00764 assets")
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.output, args.source)}")


if __name__ == "__main__":
    main()
