"""Reproduce the registered-XYZ tee archive and the independent IRI flat scan comparison."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from struct import unpack_from
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
from zlib import decompress

import numpy as np
from PIL import Image
from scipy import ndimage

from superarchiveallclo.compare_flat import compare_flat
from superarchiveallclo.compare_shape import compare_masks, crop_surface
from superarchiveallclo.rgbd import plane_basis, prepare
from superarchiveallclo.scan import project, read_pcd

DATA = "http://www.iri.upc.edu/groups/perception/clothingDataset/data"
ITEM = "iri-upc-blue-vespa-tee"
WEIGHTS = Path(__file__).resolve().parents[1] / "models/tee-vision-front-compact.npz"
FRAMES = {
    "n00634": {
        "img": (179293747, 400925, "a1cf0ee50553ca1e7c2ed412d9bb106543d68038f14cb22d375577ceae9e4314"),
        "pcd": (1275932464, 1284288, "91e4ea9b2996b7a7fbadf4c532899222a6bd883143ab13fbe95b1860e501af58"),
    },
    "n00647": {
        "img": (77440358, 389883, "89fce2ffd9875aa383bf25d2b0a8a5a50da226c4264286fc8cc00aedfe41aa4c"),
        "pcd": (887218809, 1281316, "3f4cc60ee4175eb6610915c8ea481258e759034938b0203653091ee380bb2563"),
    },
    "n00700": {
        "img": (153633002, 399855, "49fca407f6a467be9c0cd2aed7aefd84a35781596737a9606c02f470401a3097"),
        "pcd": (826201866, 1275403, "5370d63f0356f80bac392849ce2502ffc7cd89b8e58f42e60c01045409bcd3e6"),
    },
    "n00764": {
        "img": (122487079, 395096, "52117afb1c6f6d921f4ff48579b31d35e3f1fbcd88b3943fd0d1a9cea091a808"),
        "pcd": (358604811, 1287950, "9525ff64c9cd211cff87f0101c29ddf33da74d561d8f92a4f2e69b00536ca15e"),
    },
}
MASKS_SHA256 = "28f963652c15edaf416f4e7c06a36439a33a249a0ef304bf26cfd1a4290f7430"


def checked(data, expected):
    if sha256(data).hexdigest() != expected:
        raise ValueError("IRI source asset differs from the pinned research release")
    return data


def download_member(archive, member, name, entry):
    # Retrieve one pinned local ZIP entry, avoiding whole-image and point-cloud archives.
    start, size, checksum = entry
    end = start + size + 511
    for attempt in range(3):
        url = f"{DATA}/{archive}.zip?entry={name}"
        if attempt:
            url += f"&retry={attempt}"
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urlopen(request, timeout=35) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("IRI server did not return the pinned ZIP byte range")
                raw = response.read(size + 513)
            break
        except (URLError, TimeoutError) as error:
            if attempt == 2 or isinstance(error, HTTPError) and error.code < 500:
                raise
    signature, _, _, method, _, _, _, _, _, length, extra = unpack_from("<4s5H3L2H", raw)
    if signature != b"PK\x03\x04" or method != 8 or raw[30:30 + length].decode() != member:
        raise ValueError("IRI source ZIP entry changed")
    start = 30 + length + extra
    return checked(decompress(raw[start:start + size], -15), checksum)


def download(root, frames):
    with urlopen(f"{DATA}/masks.zip?archive_download=1", timeout=35) as response:
        masks = checked(response.read(2_900_001), MASKS_SHA256)
    with ZipFile(BytesIO(masks)) as archive:
        for frame in frames:
            assets = FRAMES[frame]
            (root / f"{frame}-mask.png").write_bytes(archive.read(f"masks/{frame}.png"))
            (root / f"{frame}.png").write_bytes(download_member("img", f"png/{frame}.png", frame, assets["img"]))
            (root / f"{frame}.pcd").write_bytes(download_member("pcd", f"pcd/{frame}.pcd", frame, assets["pcd"]))


def level_reference(root, frame):
    points = read_pcd((root / f"{frame}.pcd").read_bytes(), organized=True)
    with Image.open(root / f"{frame}-mask.png") as image:
        mask = np.asarray(image.convert("L")) > 127
    valid = np.isfinite(points).all(axis=2)
    border = ndimage.distance_transform_edt(~mask)
    origin, axes, residual = plane_basis(points[valid & (border >= 12) & (border <= 35)])
    cloth = (points[mask & valid] - origin) @ axes

    # Fit the independently observed hem, retaining every metric scale and every depth value.
    x, y = cloth[:, :2].T
    bins = np.linspace(*np.quantile(x, [0.26, 0.74]), 21)
    hem = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        column = (x >= left) & (x < right)
        if column.sum() > 40:
            hem.append((x[column].mean(), np.quantile(y[column], 0.99)))
    slope, _ = np.polyfit(*np.asarray(hem).T, 1)
    angle = np.arctan(slope)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    path = root / f"{frame}-flat.pcd"
    count = len(cloth)
    with path.open("w") as output:
        output.write(f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                     f"WIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA ascii\n")
        np.savetxt(output, cloth @ rotation, fmt="%.8f")
    return path, {
        "frame": frame, "support_plane_residual_p90_mm": round(residual * 1000, 3),
        "hem_rotation_deg": round(float(np.degrees(angle)), 4),
        "foreground_coverage": round(float(valid[mask].mean()), 6),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--input-frame", choices=("n00700", "n00634", "n00647"), default="n00700")
    parser.add_argument("--auto-upright", action="store_true")
    args = parser.parse_args()
    root = args.output / "source"
    root.mkdir(parents=True)
    capture = args.input_frame
    download(root, (capture, "n00764"))
    source = root / "registered.json"
    source.write_text(json.dumps({
        "schema_version": 1, "item_id": ITEM,
        "source": f"IRI UPC-CSIC Clothing Part Dataset, frame {capture}, Kinect RGB-registered XYZ",
        "views": {"front": {"rgb": f"{capture}.png", "pcd": f"{capture}.pcd", "pcd_unit": "m",
                            "mask": f"{capture}-mask.png"}},
    }, indent=2) + "\n")
    prepare(source, args.output / "registered", weights=WEIGHTS, auto_upright=args.auto_upright)
    reference, evidence = level_reference(root, "n00764")
    compare_flat(args.output / "registered/archive", [reference], args.output / "comparison", item_id=ITEM,
                 setup="IRI n00764 independent front capture; measured tabletop leveled and hem aligned")
    (args.output / "reference.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if capture == "n00700":
        capture_cloud, _ = level_reference(root, capture)
        masks = []
        for cloud in (reference, capture_cloud):
            image, _ = project(read_pcd(cloud.read_bytes()), "x", "y", 2)
            masks.append(crop_surface(np.asarray(image) < 128))
        outline, overlay = compare_masks(*masks)
        (args.output / "repeat-capture.json").write_text(json.dumps(outline, indent=2) + "\n")
        Image.fromarray(overlay).save(args.output / "repeat-capture.png")


if __name__ == "__main__":
    main()
