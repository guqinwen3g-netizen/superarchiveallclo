"""Fit one effective Blender bending setting on early scans and score separately held-out scans."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from examples.evaluate_auckland_tees import write_json
from examples.evaluate_rgbench_material import CONFIG_SHA256, compare_variants, publisher_mass, summarize_pairs
from superarchiveallclo.cloth_material import REFERENCE_BENDING_STIFFNESS
from superarchiveallclo.compare_hang import compare_hang

CANDIDATES = (REFERENCE_BENDING_STIFFNESS, 0.145, 0.0145, 0.00145, 0.000145)
MIN_COMMON_RAY_FRACTION = 0.95
FIT_DEPTH_TOLERANCE_MM = 4.0


def split_reference(suspension):
    encoded = suspension.read_bytes()
    fixture = json.loads(encoded)
    if fixture["item_id"] != "rgbench-green-tshirt" or len(fixture["reference_scans"]) != 6:
        raise ValueError("Use the six-frame RGBench green T-shirt suspension manifest")
    paths = [suspension.parent / name for name in fixture.pop("reference_scans")]
    # Only fit-scan bytes are opened during selection. Compare timestamps and
    # paths here, and check content hashes when the held-out scans are read.
    times = [float(value) for value in fixture.pop("reference_timestamps")]
    if (len(times) != len(paths) or not all(map(isfinite, times))
            or any(after <= before for before, after in zip(times, times[1:]))):
        raise ValueError("Provide strictly time-ordered observations")
    distinct = [path.resolve() for path in paths]
    if len(set(distinct)) != len(paths):
        raise ValueError("Use separate calibration and held-out scans")
    fixture.pop("swing_history", None)
    for name in ("rest_scan", "world_to_camera", "support_calibration_scan"):
        fixture[name] = str((suspension.parent / fixture[name]).resolve())
    return fixture, distinct[:2], distinct[2:], sha256(encoded).hexdigest()


def quality(rows):
    summary = summarize_pairs(rows, ("control", "candidate"))
    coverage = min(row["common_hits"] / row["measured_rays"] for row in rows)
    control = summary["control"]["absolute_depth_p90_mm"]
    candidate = summary["candidate"]["absolute_depth_p90_mm"]
    feasible = coverage >= MIN_COMMON_RAY_FRACTION and candidate is not None and candidate <= control + FIT_DEPTH_TOLERANCE_MM
    return {"summary": summary, "minimum_common_ray_fraction": round(coverage, 6), "eligible": feasible, "frames": rows}


def evaluate(archive, suspension, config, output, blender):
    mass = publisher_mass(config)
    fixture, fit, holdout, source_hash = split_reference(suspension)
    output.mkdir(parents=True)
    def run_case(path, scans, bending):
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = write_json(path.with_name(f"{path.name}.json"), {
            **fixture, "reference_scans": [str(path) for path in scans],
            "garment_mass": mass, "blender_bending": bending,
        })
        return compare_hang(archive, manifest, path, blender, save_ray_errors=True)

    fit_dir = output / "fit"
    def run_candidate(value):
        location = fit_dir / f"bend-{value:g}"
        setting = {"stiffness": value, "source": "Predeclared Blender candidate, two RGBench fit scans"}
        try:
            result = run_case(location, fit, setting)
        except ValueError as error:
            if value == REFERENCE_BENDING_STIFFNESS:
                raise
            return value, (location, {"error": str(error)})
        return value, (location, result)

    with ThreadPoolExecutor(max_workers=2) as executor:
        candidates = dict(executor.map(run_candidate, CANDIDATES))
    control_path, control = candidates[REFERENCE_BENDING_STIFFNESS]
    fitting = {}
    for value, (path, comparison) in candidates.items():
        if "error" in comparison:
            fitting[value] = {"eligible": False, "error": comparison["error"]}
        else:
            rows = compare_variants(control, comparison, control_path, path, labels=("control", "candidate"))
            fitting[value] = quality(rows)
    eligible = [value for value, row in fitting.items() if row["eligible"]]
    if not eligible:
        raise ValueError("No fitting result covered at least 95 percent of observed rays")

    def rank(value):
        score = fitting[value]["summary"]["candidate"]
        return score["front_iou"], -score["absolute_depth_p90_mm"], value

    selected = max(eligible, key=rank)
    profile = {
        "garment_mass": mass,
        "blender_bending": {"stiffness": selected, "source": "Blender fit on the first two RGBench suspension observations"},
    }
    fit_report = write_json(output / "fit-selection.json", {
        "pattern_sha256": control["pattern_sha256"], "source_manifest_sha256": source_hash,
        "publisher_config_sha256": CONFIG_SHA256, "fit_scan_names": [path.name for path in fit],
        "test_scan_names": [path.name for path in holdout],
        "minimum_common_ray_fraction_per_fit_scan": MIN_COMMON_RAY_FRACTION,
        "allowed_fit_depth_p90_increase_mm": FIT_DEPTH_TOLERANCE_MM,
        "selection_rule": "Maximize fit silhouette IoU after the depth and coverage gates; break ties by depth and then stiffness.",
        "candidates": fitting, "material": profile,
    })
    write_json(output / "fitted-material.json", profile)

    holdout_hashes = {sha256(path.read_bytes()).hexdigest() for path in holdout}
    fit_hashes = {row["sha256"] for row in control["reference_scans"]}
    if len(holdout_hashes) != len(holdout) or holdout_hashes & fit_hashes:
        raise ValueError("Held-out scan content overlaps the calibration set")
    paths = output / "held-out" / "control", output / "held-out" / "fitted"
    with ThreadPoolExecutor(max_workers=2) as executor:
        before = executor.submit(run_case, paths[0], holdout, None)
        after = executor.submit(run_case, paths[1], holdout, profile["blender_bending"])
        reference, fitted = before.result(), after.result()
    rows = compare_variants(reference, fitted, *paths, labels=("control", "fitted"))
    return write_json(output / "study.json", {
        "item_id": fixture["item_id"], "fit_selection_sha256": sha256(fit_report.read_bytes()).hexdigest(),
        "pattern_sha256": fitted["pattern_sha256"], "control_glb_sha256": reference["after"]["sha256"],
        "fitted_glb_sha256": fitted["after"]["sha256"], "material": profile,
        "control_solver": reference["support_metrics"], "fitted_solver": fitted["support_metrics"],
        "held_out": summarize_pairs(rows, ("control", "fitted")), "frames": rows,
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("suspension", type=Path, help="Six timestamp-ordered same-item observations")
    parser.add_argument("--config", type=Path, help="Use the already downloaded pinned RGBench green_tshirt_5k.yaml")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Wrote {evaluate(args.archive, args.suspension, args.config, args.output, args.blender)}")


if __name__ == "__main__":
    main()
