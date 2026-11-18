"""Reproduce the scan-opening measurements using GarmentLab's public benchmark."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

from superarchiveallclo.scan_openings import measure_scan

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/garmentlab-opening-study.json"
DOWNLOAD = "https://drive.usercontent.google.com/download"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True)
    record = json.loads(RECORD.read_bytes())
    for name, sample in record["samples"].items():
        root = output / name
        root.mkdir()
        assets = {}
        for part, file in sample["assets"].items():
            path = root / file["name"]
            url = f"{DOWNLOAD}?id={file['drive_id']}&export=download&confirm=t"
            with urlopen(url, timeout=45) as response:
                raw = response.read(5_000_001)
            if sha256(raw).hexdigest() != file["sha256"]:
                raise ValueError(f"Published source changed: {file['name']}")
            path.write_bytes(raw)
            assets[part] = path
        report = measure_scan(assets.pop("scan"), assets)
        (root / "openings.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
