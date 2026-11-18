"""Train and evaluate a reproducible photo-to-measurement tee model."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from io import BytesIO
import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from .dataset import arrays, evaluate, load_dataset, usable_labels
from .flat_measurements import compatible_scan_sleeve_length
from .tee import TeeSpec, body_panel, draft
from .vision import (CARD_MM, FIELDS, MODEL_VERSION, SLEEVE_RATIOS, TeeVision, decode_sleeve_ratios,
                     encode_sleeve_ratios, paired_features)

SEED = 481516


def sample_spec(rng, *, compact=False):
    if compact:
        chest = rng.uniform(425, 560)
        shoulder = chest - rng.uniform(42, 83)
        slope = rng.uniform(19, 47)
        armhole = rng.uniform(174, 225)
        bicep = min(rng.uniform(262, 336), 2 * (armhole - slope) - 12)
        return TeeSpec(
            body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None,
            flat_chest_mm=chest, back_length_mm=rng.uniform(560, 730),
            garment_shoulder_width_mm=shoulder, shoulder_slope_mm=slope, armhole_depth_mm=armhole,
            sleeve_length_mm=rng.uniform(120, 197), sleeve_bicep_flat_mm=bicep,
            sleeve_opening_flat_mm=bicep - rng.uniform(20, 58),
            neck_width_mm=rng.uniform(146, 190), front_neck_depth_mm=rng.uniform(60, 104),
            back_neck_depth_mm=rng.uniform(15, 32),
        )
    chest = rng.uniform(485, 635)
    shoulder = chest - rng.uniform(45, 92)
    slope = rng.uniform(22, 51)
    armhole = rng.uniform(211, 265)
    bicep = min(rng.uniform(310, 380), 2 * (armhole - slope) - 12)
    return TeeSpec(
        body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None,
        flat_chest_mm=chest, back_length_mm=rng.uniform(605, 750),
        garment_shoulder_width_mm=shoulder, shoulder_slope_mm=slope, armhole_depth_mm=armhole,
        sleeve_length_mm=rng.uniform(174, 249), sleeve_bicep_flat_mm=bicep,
        sleeve_opening_flat_mm=bicep - rng.uniform(22, 66),
        neck_width_mm=rng.uniform(160, 200), front_neck_depth_mm=rng.uniform(73, 105),
        back_neck_depth_mm=rng.uniform(18, 32),
    )


def narrow_spec(rng):
    chest = rng.uniform(355, 465)
    shoulder = chest - rng.uniform(20, 70)
    slope = rng.uniform(19, 43)
    armhole = rng.uniform(177, 226)
    bicep = min(rng.uniform(239, 317), 2 * (armhole - slope) - 12)
    return TeeSpec(
        body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None,
        flat_chest_mm=chest, back_length_mm=rng.uniform(535, 725),
        garment_shoulder_width_mm=shoulder, shoulder_slope_mm=slope, armhole_depth_mm=armhole,
        sleeve_length_mm=rng.uniform(119, 192), sleeve_bicep_flat_mm=bicep,
        sleeve_opening_flat_mm=bicep - rng.uniform(20, 60),
        neck_width_mm=rng.uniform(132, 176), front_neck_depth_mm=rng.uniform(54, 97),
        back_neck_depth_mm=rng.uniform(13, 29),
    )


def varied_spec(rng):
    style = rng.random()
    spec = narrow_spec(rng) if style < 0.32 else sample_spec(rng, compact=style < 0.57)
    return replace(spec, flat_hem_mm=spec.flat_chest_mm * rng.uniform(0.90, 1.18))


def synthetic_view(spec, front, rng, *, varied_layouts=False):
    """Project a stylized flat lay from template geometry with capture variation."""
    width, height = 412, 352
    pixels_per_mm = rng.uniform(0.29, 0.35)
    angle = radians(rng.uniform(-3, 3))
    origin = (width / 2 + rng.uniform(-7, 7), rng.uniform(32, 43))

    def pixel(point):
        x, y = point
        return (
            origin[0] + pixels_per_mm * (x * cos(angle) - y * sin(angle)),
            origin[1] + pixels_per_mm * (x * sin(angle) + y * cos(angle)),
        )

    light_background = bool(rng.integers(0, 2))
    bg = rng.uniform(196, 243) if light_background else rng.uniform(36, 78)
    fg = rng.uniform(34, 110) if light_background else rng.uniform(166, 224)
    backdrop = tuple(int(c) for c in bg + rng.uniform(-13, 13, 3))
    fabric = tuple(int(c) for c in fg + rng.uniform(-12, 12, 3))
    image = Image.new("RGB", (width, height), backdrop)
    paint = ImageDraw.Draw(image)
    body = body_panel(spec, front)
    paint.polygon([pixel(p) for p in body.outline()], fill=fabric)
    size = spec.sleeve_length_mm
    upper = spec.sleeve_bicep_flat_mm / 2
    opening = spec.sleeve_opening_flat_mm / 2
    shoulder = spec.shoulder_width_mm / 2
    underarm = spec.flat_chest_mm / 2
    limits = (9, 47) if varied_layouts else (19, 27)
    for side in (-1, 1):
        pose = radians(rng.uniform(*limits))
        c, s = cos(pose), sin(pose)
        visible_length = size * rng.uniform(0.88, 1) if varied_layouts else size
        sill_x, sill_y = shoulder + visible_length * c, spec.shoulder_slope_mm + visible_length * s
        sleeve = (
            (shoulder - 11, spec.shoulder_slope_mm - 3),
            (sill_x, sill_y),
            (sill_x - opening * s, sill_y + opening * c),
            (shoulder + 0.37 * visible_length * c - upper * s, spec.shoulder_slope_mm + 0.37 * visible_length * s + upper * c),
            (underarm - 12, spec.armhole_depth_mm),
        )
        paint.polygon([pixel((side * x, y)) for x, y in sleeve], fill=fabric)
    card_px = round(pixels_per_mm * CARD_MM)
    card_x, card_y = int(rng.integers(15, 29)), int(rng.integers(215, 260))
    paint.rectangle((card_x, card_y, card_x + card_px - 1, card_y + card_px - 1), fill=(215, 39, 157))
    gradients = np.linspace(-5, 5, width, dtype=np.float32)[None, :, None]
    noise = rng.normal(0, 2.1, (height, width, 1)).astype(np.float32)
    rgb = np.clip(np.asarray(image) + gradients + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def sample_set(size, seed, front_only=False, compact_sleeves=False, varied_layouts=False):
    rng = np.random.default_rng(seed)
    features, targets, boxes = [], [], []
    for _ in range(size):
        spec = varied_spec(rng) if varied_layouts else sample_spec(rng, compact=compact_sleeves and rng.random() < 0.45)
        if compact_sleeves or varied_layouts:
            length = compatible_scan_sleeve_length({field: getattr(spec, field) for field in FIELDS}, {})
            if length is not None:
                spec = replace(spec, sleeve_length_mm=length)
        front = synthetic_view(spec, True, rng, varied_layouts=varied_layouts)
        back = synthetic_view(spec, False, rng, varied_layouts=varied_layouts)
        if rng.random() < 0.75:
            front = jpeg_view(front, int(rng.integers(83, 97)))
            back = jpeg_view(back, int(rng.integers(83, 97)))
        vector, evidence = paired_features(front, back)
        features.append(vector[:len(vector) // 2] if front_only else vector)
        targets.append([getattr(spec, field) for field in FIELDS])
        boxes.append([evidence[name]["garment_box_mm"] for name in (("front",) if front_only else ("front", "back"))])
    return np.array(features), np.array(targets), np.array(boxes)


def jpeg_view(image, quality):
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    with Image.open(stream) as encoded:
        return encoded.copy()


def fit_ridge(train, validate, front_only=False, *, sleeve_ratios=False):
    x, y, boxes = train
    x_val, y_val, _ = validate
    if sleeve_ratios:
        y, y_val = encode_sleeve_ratios(y), encode_sleeve_ratios(y_val)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.08)
    labels = y.mean(axis=0)
    normalized = (x - center) / scale
    cov = normalized.T @ normalized
    rhs = normalized.T @ (y - labels)
    if front_only:
        rhs[:, FIELDS.index("back_neck_depth_mm")] = 0
    scores = []
    for penalty in (12.0, 40.0, 140.0):
        coefficients = np.linalg.solve(cov + penalty * np.eye(cov.shape[0]), rhs)
        error = np.abs(((x_val - center) / scale) @ coefficients + labels - y_val)
        scores.append(((error / y.std(axis=0)).mean(), penalty, coefficients))
    _, penalty, coefficients = min(scores, key=lambda candidate: candidate[0])
    return dict(
        version=np.array(MODEL_VERSION), fields=np.array(FIELDS),
        center=center, scale=scale, intercept=labels, coefficients=coefficients,
        target_min=y.min(axis=0), target_max=y.max(axis=0),
        box_min=boxes.min(axis=0), box_max=boxes.max(axis=0),
    ), penalty


def score_geometry(model, data):
    """Apply the production capture range, regressor and sewing checks to labelled held-out layouts."""
    vectors, truths, boxes = data
    names = ("front",) if model.view_mode == "front" else ("front", "back")
    chest = truths[:, FIELDS.index("flat_chest_mm")]
    report = {}
    for group, subset in (("chest_under_425_mm", chest < 425), ("chest_425_to_484_mm", (chest >= 425) & (chest < 485)),
                          ("chest_485_mm_and_above", chest >= 485)):
        counts = Counter()
        errors = []
        for features, target, reads in zip(vectors[subset], truths[subset], boxes[subset], strict=True):
            evidence = {name: {"garment_box_mm": bounds} for name, bounds in zip(names, reads, strict=True)}
            try:
                fitted, _ = model.estimate_features(features, evidence)
            except ValueError:
                counts["capture_rejected"] += 1
                continue
            try:
                draft(TeeSpec(body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None, **fitted))
            except ValueError:
                counts["pattern_rejected"] += 1
            else:
                counts["pattern_passed"] += 1
            predicted = np.array([fitted[field] for field in FIELDS])
            errors.append(np.abs(predicted - target))
        row = {"samples": int(subset.sum()), "outcomes": dict(counts)}
        if errors:
            med, p90 = np.quantile(np.asarray(errors), [0.5, 0.90], axis=0)
            row["accepted_measurement_error_mm"] = {
                field: {"median": round(float(a), 2), "p90": round(float(b), 2)}
                for field, a, b in zip(FIELDS, med, p90, strict=True)
            }
        report[group] = row
    return report


def save_card(path):
    canvas = Canvas(str(path), invariant=True)
    canvas.setFillColorRGB(215 / 255, 39 / 255, 157 / 255)
    canvas.rect(22 * mm, 147 * mm, CARD_MM * mm, CARD_MM * mm, fill=1, stroke=0)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(22 * mm, 135 * mm, "100 mm square - print at Actual size, then verify with a ruler.")
    canvas.drawString(22 * mm, 128 * mm, "Cut around the colored square. Place it flat beside the garment.")
    canvas.save()


def write_demo(path, model):
    rng = np.random.default_rng(SEED + 8)
    # This seed is separate from training, validation and test.
    spec = sample_spec(rng)
    path.mkdir(parents=True)
    views = {}
    images = []
    for name, front in (("front", True), ("back", False)):
        image = synthetic_view(spec, front, rng)
        image.save(path / f"{name}.jpg", quality=91)
        views[name] = {"file": f"{name}.jpg"}
        images.append(Image.open(path / f"{name}.jpg"))
    try:
        measured, _ = model.estimate(*images)
    finally:
        for image in images:
            image.close()
    truth = {f: round(float(getattr(spec, f)), 3) for f in FIELDS}
    (path / "session.json").write_text(json.dumps({
        "schema_version": 1, "kind": "synthetic_fixture", "garment": "crew_neck_tee",
        "item_id": "model-demo", "views": views,
    }, indent=2) + "\n")
    (path / "reference.json").write_text(json.dumps({
        "kind": "synthetic_reference", "truth_mm": truth, "inferred_mm": measured,
    }, indent=2) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--demo", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--dataset", type=Path, help="Optional manifest of independently labelled photo pairs")
    parser.add_argument("--compare-model", type=Path, help="Score incumbent weights on the same labelled test rows")
    parser.add_argument("--front-only", action="store_true", help="Train on front views with a prior for back neck depth")
    parser.add_argument("--compact-sleeves", action="store_true", help="Include narrower, shorter-sleeved tees in the synthetic set")
    parser.add_argument("--varied-layouts", action="store_true", help="Train on narrow and tapered tees with a wider range of sleeve placement")
    args = parser.parse_args()
    if args.weights.suffix != ".npz":
        parser.error("--weights must have an .npz extension")
    if args.compare_model is not None and args.dataset is None and not args.varied_layouts:
        parser.error("--compare-model requires --dataset or --varied-layouts")
    if args.front_only and (args.dataset or args.demo):
        parser.error("--dataset and --demo currently support paired weights")
    try:
        labelled, dataset_checksum = load_dataset(args.dataset) if args.dataset else ([], None)
        incumbent = TeeVision(args.compare_model, "front" if args.front_only else "paired") if args.compare_model else None
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))

    sizes = (960, 160, 200)
    train, validation, test = (
        sample_set(size, SEED + split, args.front_only, args.compact_sleeves, args.varied_layouts)
        for split, size in enumerate(sizes)
    )
    extra_train = arrays(labelled, "train")
    extra_validation = arrays(labelled, "validation")
    if extra_train is not None:
        train = tuple(np.concatenate((a, b)) for a, b in zip(train, extra_train, strict=True))
    if extra_validation is not None:
        validation = extra_validation
    weights, penalty = fit_ridge(train, validation, args.front_only, sleeve_ratios=args.varied_layouts)
    if args.front_only:
        weights["view_mode"] = np.array("front")
    if args.varied_layouts:
        weights["sleeve_target_encoding"] = np.array(SLEEVE_RATIOS)
    used_labels = usable_labels(labelled, "train") + usable_labels(labelled, "validation")
    weights["labelled_training_groups"] = np.array(sorted({item.group for item in used_labels}), dtype=str)
    weights["labelled_training_photos"] = np.array(sorted({
        digest for item in used_labels for digest in item.fingerprints
    }), dtype=str)
    if extra_validation is not None:
        weights["validation_domain"] = np.array("labelled validation split; see evaluation report")
    elif args.varied_layouts:
        weights["validation_domain"] = np.array("synthetic tees with narrower and tapered torsos and varied sleeve orientations")
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.weights, **weights)
    x, y, _ = test
    prediction = np.clip(
        (x - weights["center"]) / weights["scale"] @ weights["coefficients"] + weights["intercept"],
        weights["target_min"], weights["target_max"],
    )
    if args.varied_layouts:
        prediction = decode_sleeve_ratios(prediction)
    errors = np.abs(prediction - y)
    naive = np.abs(np.median(train[1], axis=0) - y)
    geometry_checks = []
    for row in prediction[np.linspace(0, len(y) - 1, 24).round().astype(int)]:
        try:
            draft(TeeSpec(
                body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None,
                **dict(zip(FIELDS, map(float, row), strict=True)),
            ))
            geometry_checks.append(True)
        except ValueError:
            geometry_checks.append(False)
    report = {
        "training_domain": "synthetic flat-lay silhouettes with varied framing, lighting, sleeve pose and JPEG compression",
        "seeds_train_validation_test": [SEED, SEED + 1, SEED + 2],
        "sample_counts_train_validation_test": sizes,
        "ridge_penalty": penalty,
        "test_valid_pattern_subset": {
            "passed": sum(geometry_checks), "checked": len(geometry_checks),
        },
        "test_metrics_mm": {
            name: {
                "median_absolute_error": round(float(np.median(errors[:, i])), 2),
                "p90_absolute_error": round(float(np.quantile(errors[:, i], 0.9)), 2),
                "typical_size_baseline_p90": round(float(np.quantile(naive[:, i], 0.9)), 2),
            }
            for i, name in enumerate(FIELDS)
        },
    }
    if args.front_only:
        report["model_input"] = "front"
        report["prior_fields"] = ["back_neck_depth_mm"]
    if args.compact_sleeves:
        report["synthetic_sampling"] = {"compact_probability": 0.45, "compact_chest_mm": [425, 560],
                                        "compact_sleeve_length_prior_mm": [120, 197],
                                        "sleeve_length_adjusted_to_allow_a_sewable_cap": True}
    model = TeeVision(args.weights, "front" if args.front_only else "paired")
    if args.varied_layouts:
        report["synthetic_sampling"] = {"narrow_probability": 0.32, "compact_probability": 0.25,
                                        "narrow_chest_mm": [355, 465], "sleeve_angle_deg": [9, 47],
                                        "visible_sleeve_length_fraction": [0.88, 1], "hem_chest_ratio": [0.90, 1.18],
                                        "sleeve_target_encoding": SLEEVE_RATIOS}
        report["held_out_geometry"] = {"weights_sha256": model.checksum, "by_chest_range": score_geometry(model, test)}
        if incumbent is not None:
            report["held_out_geometry"]["incumbent"] = {
                "weights_sha256": incumbent.checksum, "by_chest_range": score_geometry(incumbent, test),
            }
    if args.dataset:
        report["labelled_dataset"] = {
            "manifest_sha256": dataset_checksum,
            "items_by_split": dict(Counter(item.split for item in labelled)),
            "rejected_captures_by_split": dict(Counter(
                item.split for item in labelled if item.capture_error is not None
            )),
            "partial_references_by_split": dict(Counter(
                item.split for item in labelled if len(item.truth) < len(FIELDS)
            )),
            "used_for_training": 0 if extra_train is None else len(extra_train[0]),
            "used_for_regularization_choice": 0 if extra_validation is None else len(extra_validation[0]),
            "weights_sha256": model.checksum,
            "test": evaluate(labelled, model),
        }
        if incumbent is not None:
            report["labelled_dataset"]["incumbent"] = {
                "weights_sha256": incumbent.checksum,
                "test": evaluate(labelled, incumbent),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.demo:
        write_demo(args.demo, model)
    if args.card:
        save_card(args.card)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
