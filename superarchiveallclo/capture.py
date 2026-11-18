"""Archive calibrated photos and reconstruct a supported garment."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import dist, isfinite
from pathlib import Path
import re
from shutil import copyfile
from subprocess import run
from tempfile import TemporaryDirectory

from ezdxf import DXFError
from PIL import Image, ImageOps, UnidentifiedImageError

from .appearance import build_appearance
from .cad import export_dxf
from .check_archive import check_archive
from .export import write_exports
from .reference_form import arm_angle
from .tee import TeeSpec, draft
from .vision import FIELDS, TeeVision

ASSUMPTIONS = {
    "collar_finished_width_mm": TeeSpec.collar_finished_width_mm,
    "collar_length_ratio": TeeSpec.collar_length_ratio,
}
OPTIONAL_MEASUREMENTS = ("flat_hem_mm",)


def number(value, name):
    if type(value) not in (float, int) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name}: expected a positive finite value")
    return float(value)


def segment_length(segment, size, axis=None):
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError("Expected two pixel points")
    for p in segment:
        if not isinstance(p, list) or len(p) != 2 or not all(
            type(v) in (int, float) and isfinite(v) and 0 <= v < size[i] for i, v in enumerate(p)
        ):
            raise ValueError(f"Pixel point {p!r} is outside the upright image {size}")
    a, b = segment
    if axis == "x":
        return abs(a[0] - b[0])
    if axis == "y":
        return abs(a[1] - b[1])
    if axis is not None:
        raise ValueError("Axis must be x, y, or omitted")
    return dist(a, b)


def view_calibration(view, size, name):
    if "scale" in view and "calibration" in view:
        raise ValueError(f"{name}: choose either a scale segment or metric calibration")
    if "scale" in view:
        scale = view["scale"]
        span = segment_length(scale["points"], size)
        if "basis_field" in scale and scale["basis_field"] not in FIELDS + OPTIONAL_MEASUREMENTS:
            raise ValueError(f"{name}: scale basis_field must be a supported garment dimension")
        return {"scale": scale, "mm_per_pixel": number(scale["mm"], name) / number(span, f"{name} scale pixels")}
    if "calibration" in view:
        metric = view["calibration"]
        if not isinstance(metric, dict) or not isinstance(metric.get("source"), str) or not metric["source"].strip():
            raise ValueError(f"{name}: identify the source of metric calibration")
        return {"calibration": metric, "mm_per_pixel": number(metric.get("mm_per_pixel"), f"{name} mm_per_pixel")}
    return {}


def source_evidence(source, view, name, photo_dir):
    planar = view.get("calibration", {}).get("origin_type") == "planar_reference"
    if "source_assets" not in view and not planar:
        return {}
    assets = view.get("source_assets")
    error = f"{name}: expected original planar RGB, RGB-D or registered PCD evidence"
    if not isinstance(assets, dict):
        raise ValueError(error)
    accepted = ({"rgb"}, {"rgb", "mask"}) if planar else ({"rgb", "depth"}, {"rgb", "pcd"},
                                           {"rgb", "depth", "mask"}, {"rgb", "pcd", "mask"})
    if assets.keys() not in accepted:
        raise ValueError(error)
    source_root = (source.parent / "evidence").resolve()
    for key, asset in assets.items():
        original = (source.parent / asset["file"]).resolve()
        filename = f"{name}-{key}{original.suffix}"
        if original != source_root / filename or asset["file"] != f"evidence/{filename}":
            raise ValueError(f"{name}: invalid source evidence path for {key}")
        raw = original.read_bytes()
        if sha256(raw).hexdigest() != asset["sha256"]:
            raise ValueError(f"{name}: {key} source evidence changed")
        if photo_dir is not None:
            target = photo_dir.parent / "evidence" / original.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw)
    return {"source_assets": assets}


def capture_photo_hashes(views):
    hashes = {view["sha256"] for view in views.values()}
    hashes.update(view["source_assets"]["rgb"]["sha256"] for view in views.values() if "source_assets" in view)
    return hashes


def capture_observations(source, photo_dir=None, model_path=None, *, allow_partial=False, blind=False):
    if blind and model_path is None:
        raise ValueError("Blind reconstruction requires trained --model weights")
    capture = json.loads(source.read_text(encoding="utf8"))
    if not isinstance(capture, dict) or capture.get("schema_version") != 1:
        raise ValueError("Expected capture schema_version 1")
    if capture.get("kind") not in ("physical", "synthetic_fixture"):
        raise ValueError("Capture kind must be physical or synthetic_fixture")
    if capture.get("garment") != "crew_neck_tee":
        raise ValueError("This reconstruction supports crew_neck_tee")
    mode = capture.get("input_mode", "paired")
    if mode not in ("paired", "front_only"):
        raise ValueError("Unknown capture input mode")
    measure_body = "body_profile" in capture
    if measure_body and (capture["body_profile"] != "paired_silhouette" or mode != "paired"):
        raise ValueError("The paired_silhouette body profile requires front and back views")
    measure_sleeve = "sleeve_profile" in capture
    if measure_sleeve and (capture["sleeve_profile"] != "paired_cuff_span" or mode != "paired"):
        raise ValueError("The paired_cuff_span sleeve profile requires front and back views")
    required = {"front"} if mode == "front_only" else {"front", "back"}
    views = capture["views"]
    if not isinstance(views, dict) or not required.issubset(views):
        raise ValueError("Include a front view" if mode == "front_only" else "Include front and back views")
    if mode == "front_only" and "back" in views:
        raise ValueError("Use paired capture for an available back view")
    archived, paths = {}, {}
    if photo_dir is not None:
        photo_dir.mkdir()
    for name, view in views.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid view ID {name!r}")
        if not isinstance(view, dict) or not isinstance(view.get("file"), str) or not view["file"]:
            raise ValueError(f"{name}: expected a photo file path")
        original = source.parent / view["file"]
        with Image.open(original) as image:
            size = ImageOps.exif_transpose(image).size
            extension = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format)
        if extension is None:
            raise ValueError(f"{name}: photo must be JPEG or PNG")
        target = original
        output_name = view["file"]
        if photo_dir is not None:
            target = photo_dir / (name + extension)
            copyfile(original, target)
            output_name = f"photos/{target.name}"
        paths[name] = target
        info = {
            "file": output_name, "upright_px": list(size),
            "sha256": sha256(target.read_bytes()).hexdigest(),
            **view_calibration(view, size, name),
            **source_evidence(source, view, name, photo_dir),
        }
        archived[name] = info

    predicted, inference = {}, None
    if model_path is not None:
        with Image.open(paths["front"]) as front:
            if mode == "front_only":
                predicted, inference = TeeVision(model_path, "front").estimate(front, views=archived)
            else:
                with Image.open(paths["back"]) as back:
                    predicted, inference = TeeVision(model_path).estimate(
                        front, back, archived, measure_body=measure_body, measure_sleeve=measure_sleeve,
                    )

    values, provenance = {}, {}
    supplied = {} if blind else capture.get("measurements", {})
    if not isinstance(supplied, dict):
        raise ValueError("Measurements must be an object")
    for name in FIELDS + tuple(ASSUMPTIONS) + OPTIONAL_MEASUREMENTS:
        if name not in supplied:
            if name in ASSUMPTIONS:
                values[name], method = ASSUMPTIONS[name], "assumed"
            elif name in predicted:
                values[name] = predicted[name]
                if name in inference.get("scan_constraint_sources", {}):
                    method = inference["scan_constraint_sources"][name]
                elif name in inference.get("rectified_photo_constraint_sources", {}):
                    method = inference["rectified_photo_constraint_sources"][name]
                elif name in inference.get("body_profile", {}).get("sources", {}):
                    method = inference["body_profile"]["sources"][name]
                elif name in inference.get("sleeve_profile", {}).get("sources", {}):
                    method = inference["sleeve_profile"]["sources"][name]
                elif mode == "front_only" and name == "back_neck_depth_mm":
                    method = "learned_prior"
                else:
                    method = "vision_model"
            elif allow_partial or name in OPTIONAL_MEASUREMENTS:
                continue
            else:
                raise ValueError(f"{name}: provide a measurement or a trained --model")
            provenance[name] = {"source": method, "value": values[name]}
            continue
        observation = supplied[name]
        if not isinstance(observation, dict):
            raise ValueError(f"{name}: expected a measurement record")
        if "value" in observation:
            measured = observation["value"]
            method = "provided_ratio" if name == "collar_length_ratio" else "tape"
            if method == "tape" and observation.get("method") == "catalogue_measurement":
                if not isinstance(observation.get("source"), str) or not observation["source"].strip():
                    raise ValueError(f"{name}: identify the catalogue for the published measurement")
                method = "catalogue_measurement"
        else:
            if name == "collar_length_ratio":
                raise ValueError("Collar ratio is unitless and requires a directly supplied value")
            view = archived[observation["view"]]
            if "mm_per_pixel" not in view:
                raise ValueError(f"{name}: provide a scale on its source photo")
            measured = view["mm_per_pixel"] * segment_length(
                observation["points"], view["upright_px"], observation.get("axis"),
            )
            method = "photo_landmarks"
        values[name] = round(number(measured, name), 4)
        provenance[name] = {"observation": observation, "source": method, "value": values[name]}

    manifest = {
        "schema_version": 1, "kind": capture["kind"], "garment": capture["garment"],
        "item_id": capture.get("item_id"), "views": archived, "measurements": provenance,
        "material": capture.get("material"),
        "geometry_method": "Candidate symmetric tee with paired seam curves from captured or inferred dimensions",
        "review": "Unverified against a physical original",
    }
    if inference is not None:
        inference["reference_measurements_withheld"] = blind
        manifest["inference"] = inference
    if measure_body:
        manifest["body_profile"] = capture["body_profile"]
    if measure_sleeve:
        manifest["sleeve_profile"] = capture["sleeve_profile"]
    if mode == "front_only":
        manifest["input_mode"] = mode
        manifest["review"] = "Single-sided candidate. Back geometry and hidden construction await independent evidence."
    return manifest, values


def load_capture(source, photo_dir=None, model_path=None, *, blind=False):
    manifest, values = capture_observations(source, photo_dir, model_path, blind=blind)
    spec = TeeSpec(body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None, **values)
    return manifest, spec


def archive(source, output, blender, model_path=None, *, blind=False, form="torso_with_arms", drape_frames=None,
            photo_appearance=False, arm_angle_deg=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if drape_frames is not None and (type(drape_frames) is not int or drape_frames < 2):
        raise ValueError("Drape requires at least two frames")
    angle = arm_angle(form, arm_angle_deg)
    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name("simulate.py")
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        manifest, spec = load_capture(source, bundle / "photos", model_path, blind=blind)
        (bundle / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        pattern = draft(spec)
        write_exports(pattern, bundle / "pattern")
        export_dxf(pattern, bundle / "pattern" / "pattern.dxf")
        model = bundle / "model"
        model.mkdir()
        if photo_appearance:
            build_appearance(bundle, manifest, pattern)
        with (model / "blender.log").open("w", encoding="utf8") as log:
            frame_options = [] if drape_frames is None else ["--frames", str(drape_frames)]
            photo_options = ["--photo-appearance"] if photo_appearance else []
            arm_options = [] if arm_angle_deg is None else ["--arm-angle", str(angle)]
            result = run([
                blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
                "--", str(bundle / "pattern" / "pattern.json"), str(model), form, *frame_options, *photo_options,
                *arm_options,
            ], stdout=log, stderr=log, check=False)
        if result.returncode or not all((model / name).is_file() for name in (
            "garment.glb", "garment.blend", "simulation.json", "front.png", "back.png", "three-quarter.png",
        )):
            log = (model / "blender.log").read_text(encoding="utf8", errors="replace")
            raise ValueError(f"Blender did not finish:\n{log[-2500:]}")
        check_archive(bundle)
        bundle.rename(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument("--model", type=Path, help="Use trained weights to infer dimensions from the photos")
    parser.add_argument("--blind", action="store_true", help="Withhold all recorded measurements from reconstruction")
    parser.add_argument("--form", choices=("torso_with_arms", "torso_only", "display_torso"), default="torso_with_arms",
                        help="Collision support used for the draped GLB and Blender scene")
    parser.add_argument("--drape-frames", type=int, help="End frame for Blender Cloth; default 38")
    parser.add_argument("--arm-angle", type=float, help="Symmetric upper-arm support angle from vertical in degrees")
    parser.add_argument("--photo-appearance", action="store_true", help="Transfer visible fabric from paired photographs")
    args = parser.parse_args()
    try:
        archive(args.session, args.output, args.blender, args.model, blind=args.blind, form=args.form,
                drape_frames=args.drape_frames, photo_appearance=args.photo_appearance, arm_angle_deg=args.arm_angle)
    except (ValueError, TypeError, KeyError, OSError, UnidentifiedImageError, DXFError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
