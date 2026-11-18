"""Load split-safe labelled clothing captures and evaluate saved vision weights."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .capture import capture_observations
from .tee import TeeSpec, draft
from .vision import FIELDS, TeeVision, paired_features

SPLITS = ("train", "validation", "test")


@dataclass
class Sample:
    session: Path
    item_id: str
    group: str
    split: str
    kind: str
    rights: str
    truth: dict
    sources: dict
    photo_hashes: dict
    source_hashes: frozenset
    features: np.ndarray | None
    evidence: dict | None
    capture_error: str | None = None

    @property
    def fingerprints(self):
        return self.source_hashes.union(self.photo_hashes.values())


def text_field(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty text")
    return value.strip()


def original_sensor_hashes(views):
    return frozenset(asset["sha256"] for view in views.values()
                     for kind, asset in view.get("source_assets", {}).items() if kind in ("rgb", "depth"))


def load_dataset(path: Path):
    """Validate provenance and split isolation before extracting any features."""
    encoded = path.read_bytes()
    data = json.loads(encoded)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected labelled dataset schema_version 1")
    entries = data.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Provide a nonempty samples list")

    inputs = []
    groups, photos, ids = {}, {}, set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each dataset sample must be an object")
        session = path.parent / text_field(entry.get("session"), "session")
        try:
            capture, _ = capture_observations(session, allow_partial=True)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"{session}: {error}") from error
        item_id = text_field(capture["item_id"], f"{session} item_id")
        if item_id in ids:
            raise ValueError(f"Duplicate labelled item {item_id}")
        ids.add(item_id)
        group = text_field(entry.get("group"), f"{item_id} group")
        split = entry.get("split")
        if split not in SPLITS:
            raise ValueError(f"{item_id}: split must be train, validation, or test")
        rights = text_field(entry.get("rights"), f"{item_id} rights")
        scale_basis_fields = {view.get("scale", {}).get("basis_field") for view in capture["views"].values()}
        known = {name: m for name, m in capture["measurements"].items()
                 if name in FIELDS and (split != "test" or name not in scale_basis_fields)}
        if not known:
            raise ValueError(f"{item_id}: provide at least one independent reference measurement")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Garment group {group!r} crosses dataset splits")
        hashes = {name: photo["sha256"] for name, photo in capture["views"].items()}
        if capture.get("input_mode") == "front_only":
            raise ValueError(f"{item_id}: paired evaluation requires a front and back capture")
        if hashes["front"] == hashes["back"]:
            raise ValueError(f"{item_id}: front and back must be different captures")
        raw = original_sensor_hashes(capture["views"])
        for digest in raw.union(hashes.values()):
            if photos.setdefault(digest, split) != split:
                raise ValueError(f"{item_id}: duplicate photo or raw depth capture across dataset splits")
        inputs.append((session, capture, item_id, group, split, rights, hashes, raw, known))

    samples = []
    for session, capture, item_id, group, split, rights, hashes, raw, known in inputs:
        photographs = []
        for name in ("front", "back"):
            photo = (session.parent / capture["views"][name]["file"]).read_bytes()
            if sha256(photo).hexdigest() != hashes[name]:
                raise ValueError(f"{item_id}: {name} photo changed during dataset loading")
            photographs.append(photo)
        try:
            with Image.open(BytesIO(photographs[0])) as front, Image.open(BytesIO(photographs[1])) as back:
                features, evidence = paired_features(front, back, capture["views"])
        except (ValueError, OSError) as reason:
            features, evidence, error = None, None, str(reason)
        else:
            error = None
        truth = {name: observation["value"] for name, observation in known.items()}
        sources = {name: observation["source"] for name, observation in known.items()}
        samples.append(Sample(session, item_id, group, split, capture["kind"], rights,
                              truth, sources, hashes, raw, features, evidence, error))
    return samples, sha256(encoded).hexdigest()


def usable_labels(samples, split):
    return [item for item in samples if item.split == split and item.features is not None
            and len(item.truth) == len(FIELDS)]


def arrays(samples, split):
    usable = usable_labels(samples, split)
    if not usable:
        return None
    return (
        np.stack([item.features for item in usable]),
        np.array([[item.truth[name] for name in FIELDS] for item in usable]),
        np.array([[item.evidence[name]["garment_box_mm"] for name in ("front", "back")] for item in usable]),
    )


def evaluate(samples, vision):
    results = defaultdict(list)
    for item in samples:
        if item.split != "test":
            continue
        result = {
            "item_id": item.item_id, "group": item.group, "rights": item.rights,
            "reference_mm": item.truth, "reference_sources": item.sources,
            "photo_sha256": item.photo_hashes,
        }
        if item.source_hashes:
            result["source_asset_sha256"] = sorted(item.source_hashes)
        results[item.kind].append(result)
        groups, photos = vision.labelled_training_groups, vision.labelled_training_photos
        if groups is not None and photos is not None:
            if item.group in groups or photos.intersection(item.fingerprints):
                result.update(
                    status="training_overlap",
                    reason="Garment group or photo was used to train or tune these weights",
                )
                continue
            result["training_overlap_check"] = "no_declared_overlap"
        else:
            result["training_overlap_check"] = "training_fingerprints_unavailable"
        if item.capture_error is not None:
            result.update(status="capture_rejected", reason=item.capture_error)
            continue
        try:
            estimated, bounded = vision.estimate_features(item.features, item.evidence)
        except ValueError as error:
            result.update(status="model_rejected", reason=str(error))
            continue
        result.update(
            inferred_mm=estimated, bounded_fields=bounded, scale_evidence=item.evidence,
            absolute_error_mm={name: round(abs(estimated[name] - value), 3)
                               for name, value in item.truth.items()},
        )
        try:
            pattern = draft(TeeSpec(
                body_chest_mm=None, body_shoulder_width_mm=None, shoulder_extension_mm=None, **estimated,
            ))
        except ValueError as error:
            result.update(status="pattern_rejected", reason=str(error))
        else:
            result.update(status="pattern_passed", panels=len(pattern["panels"]), seams=len(pattern["seams"]))

    report = {}
    for kind, entries in results.items():
        errors = [sample["absolute_error_mm"] for sample in entries if "absolute_error_mm" in sample]
        row = {"cases": len(entries), "status_counts": dict(Counter(s["status"] for s in entries)), "items": entries}
        metrics = {}
        for name in FIELDS:
            observations = [case[name] for case in errors if name in case]
            if observations:
                metrics[name] = {
                    "count": len(observations),
                    "median": round(float(np.median(observations)), 3),
                    "p90": round(float(np.quantile(observations, 0.9)), 3),
                }
        row["measurement_error_mm"] = metrics
        report[kind] = row
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        samples, checksum = load_dataset(args.dataset)
        vision = TeeVision(args.model)
        results = evaluate(samples, vision)
        if not results:
            raise ValueError("Evaluation needs a labelled test split")
        report = {
            "dataset_sha256": checksum,
            "weights_sha256": vision.checksum,
            "test": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
