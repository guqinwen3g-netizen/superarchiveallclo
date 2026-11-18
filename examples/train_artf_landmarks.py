"""Train and evaluate a landmark model using the publisher's three aRTF splits."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from examples.evaluate_auckland_tees import write_json
from superarchiveallclo.landmark_vision import (
    MODEL_VERSION, NAMES, Silhouette, TeeLandmarkVision, canonical_labels, infer_orientation, kernel,
)

SPLITS = ("train", "val", "test")
ALIGNMENT_NEIGHBORS = (1, 5, 12)
PENALTIES = (0.04, 0.16, 0.64)
GEOMETRIC_JITTER_DEG = (-8, 0, 8)
URL = "https://zenodo.org/records/14501499"
ANNOTATION_HASHES = {
    "train": "ee3e61b55bd93a9c6b5913bcf15cf417eff3cac66c6647dd30d756cd62c9c5b0",
    "val": "f5c99a1e647a745e5bb886cbda97ecea918c11845b67ec08dc9f770230dac6ca",
    "test": "2feeb9124f95ee5fac4e37d43fd08cb27590f7e3d315cef5b1ab22cdc3c54589",
}


def input_masks(archive, name):
    encoded = archive.read(f"artf-{name}_resized_512x256/annotations.json")
    if sha256(encoded).hexdigest() != ANNOTATION_HASHES[name]:
        raise ValueError(f"{name}: annotation file does not match the pinned aRTF release")
    record = json.loads(encoded)
    category = next(label for label in record["categories"] if label["name"] == "tshirt")
    if tuple(category["keypoints"]) != NAMES:
        raise ValueError("Unexpected aRTF landmark order")
    images = {image["id"]: image for image in record["images"]}
    samples, rejections = [], []
    for item in record["annotations"]:
        if item["category_id"] != category["id"]:
            continue
        image = images[item["image_id"]]
        points = np.asarray(item["keypoints"]).reshape(len(NAMES), 3)
        angle, points = canonical_labels(points)
        if points is None:
            rejections.append(image["file_name"])
            continue
        mask = Image.new("1", (image["width"], image["height"]))
        paint = ImageDraw.Draw(mask)
        for polygon in item["segmentation"]:
            paint.polygon(list(zip(polygon[::2], polygon[1::2], strict=True)), fill=1)
        try:
            frame = Silhouette(np.asarray(mask))
        except ValueError:
            rejections.append(image["file_name"])
        else:
            samples.append((image["file_name"], frame, angle, points))
    return samples, rejections


def prepare_regression(train):
    support, labels, visible = [], [], []
    for _, frame, angle, points in train:
        for jitter in GEOMETRIC_JITTER_DEG:
            support.append(frame.features(angle + jitter))
            labels.append(frame.normalize(points[:, :2], angle + jitter))
            visible.append(points[:, 2] == 2)
    return np.stack(support), np.stack(labels), np.stack(visible)


def fit_kernel(gram, coordinates, visible, ridge):
    count, points, _ = coordinates.shape
    coefficients = np.zeros((count, points, 2), dtype=np.float32)
    mean = []
    for name in range(points):
        observed = visible[:, name]
        targets = coordinates[observed, name]
        center = targets.mean(axis=0)
        local = gram[np.ix_(observed, observed)].copy()
        local.flat[::len(local) + 1] += ridge
        coefficients[observed, name] = np.linalg.solve(local, targets - center)
        mean.append(center)
    return coefficients.reshape(count, -1), np.array(mean)


def observed_orientations(samples, support, neighbors):
    return [infer_orientation(frame, support, neighbors) for _, frame, _, _ in samples]


def angular_errors(samples, detected):
    return np.array([abs((prediction[0] - item[2] + 180) % 360 - 180)
                     for item, prediction in zip(samples, detected, strict=True)])


def score(samples, orientation, shape):
    mistakes = []
    for (_, frame, _, points), (estimated, _, _), normalized in zip(samples, orientation, shape, strict=True):
        errors = np.linalg.norm(frame.pixels(normalized, estimated) - points[:, :2], axis=1)
        errors[points[:, 2] != 2] = np.nan
        mistakes.append(errors)
    angles = angular_errors(samples, orientation)
    errors = np.stack(mistakes)
    normalized = errors / np.array([value[1].diagonal for value in samples])[:, None]

    def summarize(px, fraction):
        use = np.isfinite(px)
        return {
            "visible_points": int(use.sum()), "median_px": round(float(np.nanmedian(px)), 3),
            "p90_px": round(float(np.nanquantile(px, 0.90)), 3),
            "pck_5pct_bbox_diagonal": round(float((fraction[use] <= 0.05).mean()), 5),
        }
    return {
        **summarize(errors, normalized),
        "orientation_error_median_deg": round(float(np.median(angles)), 3),
        "orientation_error_p90_deg": round(float(np.quantile(angles, 0.9)), 3),
        "orientations_within_22_5deg": round(float((angles <= 22.5).mean()), 5),
        "by_keypoint": {name: summarize(errors[:, index], normalized[:, index]) for index, name in enumerate(NAMES)},
    }


def train_model(source, weights, report):
    with ZipFile(source) as file:
        groups, skips = {}, {}
        for split in SPLITS:
            groups[split], skips[split] = input_masks(file, split)
    names = [[Path(row[0]).name for row in groups[split]] + [Path(p).name for p in skips[split]]
             for split in SPLITS]
    if sum(map(len, names)) != len(set().union(*names)):
        raise ValueError("Photograph names cross the aRTF data splits")
    train, val, test = (groups[split] for split in SPLITS)
    support, coordinates, visible = prepare_regression(train)
    gram = kernel(support, support)
    alignments = {k: observed_orientations(val, support, k) for k in ALIGNMENT_NEIGHBORS}
    orientation_error = {k: float(np.median(angular_errors(val, rows))) for k, rows in alignments.items()}
    neighbors = min(ALIGNMENT_NEIGHBORS, key=orientation_error.get)
    validation = alignments[neighbors]
    lookup = kernel(np.stack([row[1] for row in validation]), support)
    choices, learned = {}, {}
    for ridge in PENALTIES:
        coefficients, mean = fit_kernel(gram, coordinates, visible, ridge)
        learned[ridge] = coefficients, mean
        shape = mean + (lookup @ coefficients).reshape(-1, len(NAMES), 2)
        choices[ridge] = score(val, validation, shape)
    selected = max(PENALTIES, key=lambda ridge: choices[ridge]["pck_5pct_bbox_diagonal"])
    coefficients, mean = learned[selected]
    weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights, version=MODEL_VERSION, names=NAMES, neighbors=neighbors, support=support,
                        coefficients=coefficients, mean=mean, data_source=URL,
                        labelled_annotation_sha256=np.array([ANNOTATION_HASHES[s] for s in ("train", "val")]))
    model = TeeLandmarkVision(weights)
    detections = observed_orientations(test, model.support, model.neighbors)
    features = np.stack([estimated[1] for estimated in detections])
    predictions = model.mean + (kernel(features, model.support) @ model.coefficients).reshape(-1, len(NAMES), 2)
    train_locations = {row[0].split("/")[1] for row in train}
    unseen_locations = [i for i, row in enumerate(test) if row[0].split("/")[1] not in train_locations]
    report.parent.mkdir(parents=True, exist_ok=True)
    return write_json(report, {
        "source": URL, "license": "CC BY 4.0", "annotation_sha256": ANNOTATION_HASHES,
        "scope": "Tee image coordinates with publisher silhouette masks; every valid mask receives a prediction. No metric scale or side labels.",
        "weights_sha256": model.checksum, "ridge": selected, "alignment_neighbors": neighbors,
        "validation_median_orientation_error_by_neighbors_deg": {k: round(v, 3) for k, v in orientation_error.items()},
        "cases": {name: {"evaluated": len(groups[name]), "unusable_annotations": skips[name]} for name in SPLITS},
        "validation": {key: {"pck_5pct_bbox_diagonal": row["pck_5pct_bbox_diagonal"], "median_px": row["median_px"]}
                       for key, row in choices.items()},
        "test_mean_shape_baseline": score(test, detections, np.broadcast_to(mean, predictions.shape)),
        "test": score(test, detections, predictions),
        "test_unseen_locations": {
            "images": len(unseen_locations),
            **score([test[i] for i in unseen_locations], [detections[i] for i in unseen_locations],
                    predictions[unseen_locations]),
        },
    })


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Published artf-all-categories-resized_512x256.zip")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(f"Wrote {train_model(args.source, args.weights, args.report)}")


if __name__ == "__main__":
    main()
