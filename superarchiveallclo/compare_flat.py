"""Compare a sewn flat-lay GLB against independent metric scans of the same tee."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from ezdxf import DXFError
import numpy as np
from PIL import Image

from .check_archive import check_archive, glb_inventory
from .compare_shape import GRID_MM, compare_masks, crop_surface, project as project_mesh
from .scan import AXES, in_world, project as project_scan, read_pcd


def flat_coordinates(points, right, down):
    axes = np.eye(3)
    horizontal = axes["xyz".index(right[-1])] * (-1 if right.startswith("-") else 1)
    vertical = axes["xyz".index(down[-1])] * (1 if down.startswith("-") else -1)
    normal = np.cross(horizontal, vertical)
    if not np.any(normal):
        raise ValueError("Projection right and down need distinct axes")
    return points @ np.column_stack((horizontal, vertical, normal))


def compare_flat(archive, clouds, output, *, item_id, setup, transform=None, right="x", down="y",
                 sleeve_angle=23.0, blender="blender"):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not 0 <= sleeve_angle <= 50:
        raise ValueError("Flat sleeve angle must be between 0 and 50 degrees below horizontal")
    if not setup.strip() or not clouds:
        raise ValueError("Provide reference scans and record the flat-lay capture setup")
    integrity = check_archive(archive)
    capture = json.loads((archive / "capture.json").read_text())
    if capture["kind"] != "physical" or not item_id.strip() or item_id != capture["item_id"]:
        raise ValueError("Reference scans must identify the same physical garment as the archive")
    original = {view.get("calibration", {}).get("cloud_sha256") for view in capture["views"].values()}
    pattern = (archive / "pattern/pattern.json").read_bytes()
    if sha256(pattern).hexdigest() != integrity["pattern_sha256"]:
        raise ValueError("Pattern changed during archive verification")
    calibration = transform.read_bytes() if transform is not None else None
    report = {
        "schema_version": 1,
        "item_id": item_id,
        "reference_setup": setup,
        "pattern_sha256": integrity["pattern_sha256"],
        "capture_clouds_sha256": sorted(digest for digest in original if digest),
        "model_pose": "Sewn panels supported flat with a 2 mm layer separation and a prescribed sleeve direction",
        "sleeve_angle_below_horizontal_deg": sleeve_angle,
        "alignment": "Translation of the top and horizontal midpoint, with median scan height at the upper layer",
        "scope": "Overhead silhouette and one-way observed-point-to-mesh distances in flat pose",
        "reference_axes": {"right": right, "down": down},
        "references": [],
    }
    if calibration is not None:
        report["world_to_camera_sha256"] = sha256(calibration).hexdigest()

    used = set()
    scans, points = {}, {}
    for index, path in enumerate(clouds, 1):
        key = f"scan-{index:02d}"
        encoded = path.read_bytes()
        digest = sha256(encoded).hexdigest()
        if digest in original or digest in used:
            raise ValueError(f"Use a distinct reference scan for {path.name}")
        used.add(digest)
        cloud = read_pcd(encoded)
        if calibration is not None:
            cloud = in_world(cloud, calibration)
        image, evidence = project_scan(cloud, right, down, GRID_MM)
        points[key] = flat_coordinates(cloud, right, down)
        scans[key] = crop_surface(np.asarray(image) < 128)
        report["references"].append({
            "id": key, "file": path.name, "cloud_sha256": digest,
            "point_count": len(cloud), "plane_distance_p95_mm": evidence["plane_distance_p95_mm"],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name("flat_pose.py")
    with TemporaryDirectory(dir=output.parent) as temp:
        working = Path(temp)
        bundle = working / "comparison"
        bundle.mkdir()
        snapshot = working / "pattern.json"
        snapshot.write_bytes(pattern)
        np.savez_compressed(working / "reference-points.npz", **points)
        process = run([
            blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
            "--", str(snapshot), str(working), str(sleeve_angle),
        ], capture_output=True, text=True, check=False)
        result = working / "metrics.npz"
        if process.returncode or not result.is_file():
            raise ValueError("Blender could not compare a sewn flat pose: " + (process.stdout + process.stderr)[-2300:])

        model = working / "flat-pose.glb"
        report["flat_glb"] = {**glb_inventory(model), "sha256": sha256(model.read_bytes()).hexdigest()}
        with np.load(result, allow_pickle=False) as values:
            model_grid, model_span = project_mesh(values["vertices"], values["triangles"], "top")
            report["flat_glb"]["welded_openings"] = int(values["openings"])
            report["flat_glb"]["mesh_edge_strain_p95_percent"] = round(float(values["mesh_edge_strain_p95_percent"]), 2)
            for entry in report["references"]:
                key = entry["id"]
                observed = scans[key]
                outline, overlay = compare_masks(observed, model_grid)
                reference_span = np.array(observed.shape[::-1]) * GRID_MM
                distances = values[f"{key}_distance_mm"]
                entry.update({
                    **outline,
                    "model_minus_reference_width_height_mm": (model_span - reference_span).round(2).tolist(),
                    "reference_to_glb_surface_distance_mm": {
                        "median": round(float(np.median(distances)), 2),
                        "p90": round(float(np.quantile(distances, 0.90)), 2),
                    },
                    "reference_translation_mm": (1000 * values[f"{key}_translation_m"]).round(2).tolist(),
                    "overlay": f"{key}-overlay.png",
                })
                Image.fromarray(overlay).save(bundle / entry["overlay"])
        model.rename(bundle / model.name)
        (bundle / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        bundle.rename(output)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("clouds", type=Path, nargs="+", help="Independent segmented PCDs, with XYZ in metres")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--setup", required=True, help="Source and flat garment pose of the reference scans")
    parser.add_argument("--world-to-camera", type=Path)
    parser.add_argument("--right", choices=AXES, default="x")
    parser.add_argument("--down", choices=AXES, default="y")
    parser.add_argument("--sleeve-angle-deg", type=float, default=23)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    try:
        report = compare_flat(args.archive, args.clouds, args.output, item_id=args.item_id, setup=args.setup,
                              transform=args.world_to_camera, right=args.right, down=args.down,
                              sleeve_angle=args.sleeve_angle_deg, blender=args.blender)
    except (ValueError, KeyError, TypeError, OSError, DXFError) as error:
        parser.error(str(error))
    for entry in report["references"]:
        print(f'{entry["file"]} · IoU {entry["intersection_over_union"]:.3f} · '
              f'3D P90 {entry["reference_to_glb_surface_distance_mm"]["p90"]:.2f} mm')


if __name__ == "__main__":
    main()
