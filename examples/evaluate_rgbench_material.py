"""Compare the same sewn pattern with and without the publisher's recorded garment mass."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.compare_hang import compare_hang, depth_error_stats


CONFIG_REVISION = "1631f595ecad273c9a56104d183a95b0d9318a9c"
CONFIG_URL = f"https://raw.githubusercontent.com/hwk0809/RGBench/{CONFIG_REVISION}/configs/cloth_params/green_tshirt_5k.yaml"
CONFIG_SHA256 = "e36bbe0ed8903eb8b3de9bf9f89f639f8be9eccc44a9b654781749db760a288d"


def publisher_mass(config):
    if config is None:
        with urlopen(CONFIG_URL, timeout=20) as response:
            encoded = response.read(16384)
    else:
        encoded = config.read_bytes()
    if sha256(encoded).hexdigest() != CONFIG_SHA256:
        raise ValueError("RGBench mass configuration does not match the pinned publisher revision")
    line = re.search(rb"^mass: ([0-9.]+) # kg$", encoded, re.MULTILINE)
    if line is None:
        raise ValueError("Pinned RGBench configuration must record garment mass in kg")
    return {"kg": float(line[1]), "source": CONFIG_URL}


def compare_variants(baseline, measured, baseline_dir, measured_dir, *, labels=("baseline", "publisher_mass")):
    if (baseline["pattern_sha256"] != measured["pattern_sha256"]
            or baseline["before"]["sha256"] != measured["before"]["sha256"]):
        raise ValueError("The mass comparison must start from the same pattern and mesh")
    arrays = []
    for folder, report in ((baseline_dir, baseline), (measured_dir, measured)):
        archive = report["gravity_ray_errors"]
        source = folder / archive["file"]
        if sha256(source.read_bytes()).hexdigest() != archive["sha256"]:
            raise ValueError(f"{folder.name}: ray evidence changed during comparison")
        with np.load(source, allow_pickle=False) as data:
            arrays.append({key: data[key] for key in data.files})
    cases = []
    for earlier, later in zip(baseline["reference_scans"], measured["reference_scans"], strict=True):
        key = earlier["id"]
        if (key, earlier["sha256"]) != (later["id"], later["sha256"]):
            raise ValueError("Both simulations must be scored on the same observed rays")
        old, new = (array[key] for array in arrays)
        old_hits, new_hits = np.isfinite(old), np.isfinite(new)
        common = old_hits & new_hits
        left, right = labels
        cases.append({
            "scan": key, "sha256": earlier["sha256"], "measured_rays": len(old), "common_hits": int(common.sum()),
            f"{left}_hit_rays": int(old_hits.sum()), f"{right}_hit_rays": int(new_hits.sum()),
            f"{left}_depth_mm": depth_error_stats(old[common]), f"{right}_depth_mm": depth_error_stats(new[common]),
            f"{left}_front_iou": earlier["after"]["front_iou"], f"{right}_front_iou": later["after"]["front_iou"],
        })
    return cases


def summarize_pairs(rows, labels):
    summary = {}
    for name in labels:
        depths = [row[f"{name}_depth_mm"] for row in rows]
        summary[name] = {
            "front_iou": round(float(np.median([row[f"{name}_front_iou"] for row in rows])), 5),
            "absolute_depth_median_mm": round(float(np.median([row["absolute_median_mm"] for row in depths])), 3)
            if all(depths) else None,
            "absolute_depth_p90_mm": round(float(np.median([row["absolute_p90_mm"] for row in depths])), 3)
            if all(depths) else None,
        }
    return summary


def evaluate(archive, suspension, config, output, blender):
    mass = publisher_mass(config)
    source = json.loads(suspension.read_bytes())
    if source["item_id"] != "rgbench-green-tshirt":
        raise ValueError("RGBench's mass describes the green T-shirt in this experiment")
    for name in ("rest_scan", "world_to_camera", "support_calibration_scan"):
        source[name] = str((suspension.parent / source[name]).resolve())
    source["reference_scans"] = [str((suspension.parent / path).resolve()) for path in source["reference_scans"]]
    source.pop("swing_history", None)
    output.mkdir(parents=True)
    experiments = {"baseline": None, "publisher-mass": mass}

    def run(case):
        name, setting = case
        manifest = write_json(output / f"{name}.json", {**source, "garment_mass": setting})
        return compare_hang(archive, manifest, output / name, blender, save_ray_errors=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline, measured = pool.map(run, experiments.items())
    rows = compare_variants(baseline, measured, output / "baseline", output / "publisher-mass")
    summary = summarize_pairs(rows, ("baseline", "publisher_mass"))
    return write_json(output / "study.json", {
        "item_id": source["item_id"], "pattern_sha256": baseline["pattern_sha256"],
        "initial_glb_sha256": baseline["before"]["sha256"],
        "baseline_glb_sha256": baseline["after"]["sha256"],
        "publisher_mass_glb_sha256": measured["after"]["sha256"],
        "publisher_mass": mass, "publisher_config_sha256": CONFIG_SHA256,
        "baseline_solver": baseline["support_metrics"], "publisher_mass_solver": measured["support_metrics"],
        "scope": "No scoring frame changes garment mass, initial placement, stiffness, supports or solver settings.",
        "summary": summary, "frames": rows,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("suspension", type=Path, help="RGBench same-item two-support reference manifest")
    parser.add_argument("--config", type=Path, help="Use the already downloaded pinned RGBench green_tshirt_5k.yaml")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.archive, args.suspension, args.config, args.output, args.blender)}")


if __name__ == "__main__":
    main()
