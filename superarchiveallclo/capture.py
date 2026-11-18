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

from .cad import export_dxf
from .check_archive import check_archive
from .export import write_exports
from .tee import TeeSpec, draft
from .vision import FIELDS, TeeVision

ASSUMPTIONS = {
    "collar_finished_width_mm": 18,
    "collar_length_ratio": 0.85,
}


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
        return {"scale": scale, "mm_per_pixel": number(scale["mm"], name) / number(span, f"{name} scale pixels")}
    if "calibration" in view:
        metric = view["calibration"]
        if not isinstance(metric, dict) or not isinstance(metric.get("source"), str) or not metric["source"].strip():
            raise ValueError(f"{name}: identify the source of metric calibration")
        return {"calibration": metric, "mm_per_pixel": number(metric.get("mm_per_pixel"), f"{name} mm_per_pixel")}
    return {}


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
        }
        archived[name] = info

    predicted, inference = {}, None
    if model_path is not None:
        with Image.open(paths["front"]) as front:
            if mode == "front_only":
                predicted, inference = TeeVision(model_path, "front").estimate(front, views=archived)
            else:
                with Image.open(paths["back"]) as back:
                    predicted, inference = TeeVision(model_path).estimate(front, back, archived)

    values, provenance = {}, {}
    supplied = {} if blind else capture.get("measurements", {})
    if not isinstance(supplied, dict):
        raise ValueError("Measurements must be an object")
    for name in FIELDS + tuple(ASSUMPTIONS):
        if name not in supplied:
            if name in ASSUMPTIONS:
                values[name], method = ASSUMPTIONS[name], "assumed"
            elif name in predicted:
                values[name] = predicted[name]
                if name in inference.get("regression_before_scan_constraints_mm", {}):
                    method = "scan_with_prior" if name == "back_length_mm" else "scan_silhouette"
                elif mode == "front_only" and name == "back_neck_depth_mm":
                    method = "learned_prior"
                else:
                    method = "vision_model"
            elif allow_partial:
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
    if mode == "front_only":
        manifest["input_mode"] = mode
        manifest["review"] = "Single-sided candidate. Back geometry and hidden construction await independent evidence."
    return manifest, values


def load_capture(source, photo_dir=None, model_path=None, *, blind=False):
    manifest, values = capture_observations(source, photo_dir, model_path, blind=blind)
    spec = TeeSpec(body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None, **values)
    return manifest, spec


def archive(source, output, blender, model_path=None, *, blind=False):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
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
        with (model / "blender.log").open("w", encoding="utf8") as log:
            result = run([
                blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
                "--", str(bundle / "pattern" / "pattern.json"), str(model),
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
    args = parser.parse_args()
    try:
        archive(args.session, args.output, args.blender, args.model, blind=args.blind)
    except (ValueError, TypeError, KeyError, OSError, UnidentifiedImageError, DXFError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
