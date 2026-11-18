"""Calibrated silhouette features and a learned tee measurement regressor."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

# The card is a 100 mm square of solid magenta printed at actual size.
CARD_MM = 100.0
MODEL_VERSION = 1
FIELDS = (
    "flat_chest_mm", "back_length_mm", "garment_shoulder_width_mm", "shoulder_slope_mm",
    "armhole_depth_mm", "sleeve_length_mm", "sleeve_bicep_flat_mm", "sleeve_opening_flat_mm",
    "neck_width_mm", "front_neck_depth_mm", "back_neck_depth_mm",
)


def regions(mask):
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels, sizes


def card_shape(x, y):
    """Measure the marker along its projected edges rather than image axes."""
    coordinates = np.column_stack((x, y)).astype(float)
    try:
        hull = coordinates[ConvexHull(coordinates).vertices]
    except QhullError:
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)
    c, s = np.cos(angles), np.sin(angles)
    widths = np.ptp(hull[:, 0, None] * c + hull[:, 1, None] * s, axis=0) + 1
    heights = np.ptp(hull[:, 1, None] * c - hull[:, 0, None] * s, axis=0) + 1
    index = np.argmin(widths * heights)
    a, b = sorted((widths[index], heights[index]))
    return float(b / a), float(len(x) / (a * b))


def calibration(rgb):
    height, width = rgb.shape[:2]
    r, g, b = rgb.transpose(2, 0, 1)
    card = (r - g > 48) & (b - g > 42) & (r > 130) & (b > 115)
    labels, sizes = regions(card)
    candidates, possible_cards = 0, []
    for index in np.flatnonzero(sizes >= 100):
        yy, xx = np.where(labels == index)
        span_x, span_y = np.ptp(xx) + 1, np.ptp(yy) + 1
        area = sizes[index]
        if (
            0.60 <= span_x / span_y <= 1.67 and area >= 0.46 * span_x * span_y
            and min(xx.min(), yy.min(), width - 1 - xx.max(), height - 1 - yy.max()) > 1
        ):
            candidates += 1
            shape = card_shape(xx, yy)
            if shape is not None and shape[0] <= 1.12 and shape[1] >= 0.84:
                possible_cards.append((int(index), yy, xx, *shape))
    if not possible_cards and candidates:
        raise ValueError("Calibration marker looks distorted. Place it flat and shoot directly overhead")
    if len(possible_cards) != 1:
        raise ValueError("Show one full 100 mm magenta calibration card, separate from the garment")
    card_id, yy, xx, aspect, fill = possible_cards[0]
    # JPEG spreads color across the card edge. Integrate the local chroma
    # rather than counting all pixels whose color passed the locating threshold.
    signal = (r + b) * 0.5 - g
    window = signal[max(0, yy.min() - 4):yy.max() + 5, max(0, xx.min() - 4):xx.max() + 5]
    rim = np.concatenate((window[0], window[-1], window[:, 0], window[:, -1]))
    local_background = float(np.median(rim))
    chroma = float(np.quantile(signal[labels == card_id], 0.8)) - local_background
    area = (window - local_background).sum() / max(chroma, 1)
    if not 0.68 * sizes[card_id] <= area <= 1.23 * sizes[card_id]:
        raise ValueError("Keep the calibration card clear of garment edges and shadows")
    exclusion = ndimage.binary_dilation(labels == card_id, iterations=3)
    return float(CARD_MM / np.sqrt(area)), exclusion, {
        "method": "magenta_card_chroma_area",
        "card_mm": CARD_MM,
        "card_aspect_ratio": round(aspect, 4),
        "card_rectangular_fill": round(fill, 4),
    }


def picture_features(image):
    """Return metric dimensions and crop-relative silhouette descriptors."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((720, 720), Image.Resampling.BOX)
    rgb = np.asarray(image, dtype=np.int16)
    height, width = rgb.shape[:2]
    mm_per_px, exclusion, scale = calibration(rgb)
    border = np.concatenate((rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
                             rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)))
    backdrop = np.median(border, axis=0)
    difference = np.max(np.abs(rgb - backdrop), axis=2)
    # The backdrop is expected to be matte and approximately uniform.
    threshold = max(38.0, float(np.quantile(np.max(np.abs(border - backdrop), axis=1), 0.92)) + 15)
    foreground = ndimage.binary_closing((difference > threshold) & ~exclusion)
    objects, areas = regions(foreground)
    piece = int(areas.argmax())
    if areas[piece] < 0.08 * width * height:
        raise ValueError("Garment silhouette could not be isolated on the backdrop")
    other = np.delete(areas, piece)
    if other.size and other.max() > 0.2 * areas[piece]:
        raise ValueError("Frame the garment alone on a plain backdrop")
    mask = objects == piece
    yy, xx = np.where(mask)
    top, bottom, left, right = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
    if min(top, left, height - 1 - bottom, width - 1 - right) < 3:
        raise ValueError("Keep the entire garment, including the sleeves, away from image edges")
    crop = mask[top:bottom + 1, left:right + 1]
    h, w = crop.shape
    actual_width, actual_height = w * mm_per_px, h * mm_per_px
    bands = np.linspace(0, h - 1, 48).round().astype(int)
    row_widths = crop.sum(axis=1)[bands] * mm_per_px
    columns = np.linspace(0, w - 1, 48).round().astype(int)
    top_edges = crop.argmax(axis=0)[columns] * mm_per_px
    bottom_edges = crop[::-1].argmax(axis=0)[columns] * mm_per_px
    small = Image.fromarray(crop.astype(np.uint8) * 255).resize((18, 18), Image.Resampling.BOX)
    descriptor = np.concatenate((
        [actual_width, actual_height],
        row_widths, top_edges, bottom_edges, np.asarray(small).ravel() / 255.0,
    ))
    torso = float(np.median(row_widths[30:42]))
    return descriptor, {
        **scale,
        "processed_mm_per_pixel": round(mm_per_px, 5),
        "processed_image_px": [width, height],
        "processed_image_sha256": sha256(image.tobytes()).hexdigest(),
        "garment_crop_px": [left, top, right, bottom],
        "garment_box_mm": [round(actual_width, 2), round(actual_height, 2)],
        "lower_torso_width_mm": round(torso, 2),
    }


def paired_features(front, back):
    front_features, front_scale = picture_features(front)
    back_features, back_scale = picture_features(back)
    if front_scale["processed_image_sha256"] == back_scale["processed_image_sha256"]:
        raise ValueError("Provide two distinct photographs, one front and one back")
    if not have_consistent_dimensions(front_scale, back_scale):
        raise ValueError("Front and back photographs disagree on garment length or torso width")
    return np.r_[front_features, back_features], {"front": front_scale, "back": back_scale}


def have_consistent_dimensions(front, back):
    for a, b in (
        (front["garment_box_mm"][1], back["garment_box_mm"][1]),
        (front["lower_torso_width_mm"], back["lower_torso_width_mm"]),
    ):
        if min(a, b) <= 0 or max(a, b) / min(a, b) > 1.12:
            return False
    return True


class TeeVision:
    """A multi-target ridge model with portable, non-executable weights."""

    def __init__(self, path: Path):
        encoded = path.read_bytes()
        self.checksum = sha256(encoded).hexdigest()
        with np.load(BytesIO(encoded), allow_pickle=False) as data:
            if int(data["version"]) != MODEL_VERSION or tuple(data["fields"]) != FIELDS:
                raise ValueError("Unsupported tee model format")
            self.center = data["center"]
            self.scale = data["scale"]
            self.intercept = data["intercept"]
            self.coefficients = data["coefficients"]
            self.target_min = data["target_min"]
            self.target_max = data["target_max"]
            self.box_min = data["box_min"]
            self.box_max = data["box_max"]
            self.validation_domain = str(data["validation_domain"]) if "validation_domain" in data else (
                "synthetic flat-lay views"
            )
            self.labelled_training_groups = (
                frozenset(data["labelled_training_groups"]) if "labelled_training_groups" in data else None
            )
            self.labelled_training_photos = (
                frozenset(data["labelled_training_photos"]) if "labelled_training_photos" in data else None
            )

    def estimate(self, front, back):
        features, scale = paired_features(front, back)
        measures, bounded = self.estimate_features(features, scale)
        return measures, {
            "method": "learned_paired_silhouette_ridge",
            "weights_sha256": self.checksum,
            "scale_evidence": scale,
            "bounded_fields": bounded,
            "validation_domain": self.validation_domain,
        }

    def estimate_features(self, features, scale):
        normalized = (features - self.center) / self.scale
        values = normalized @ self.coefficients + self.intercept
        boxes = np.array([scale[name]["garment_box_mm"] for name in ("front", "back")])
        if np.any(boxes < self.box_min * 0.89) or np.any(boxes > self.box_max * 1.11):
            raise ValueError("Garment dimensions fall outside the current tee model's capture range")
        clipped = np.clip(values, self.target_min, self.target_max)
        measures = {key: round(float(value), 3) for key, value in zip(FIELDS, clipped, strict=True)}
        return measures, [field for field, raw, used in zip(FIELDS, values, clipped) if raw != used]


def main():
    parser = ArgumentParser(description="Check a calibrated front/back photo pair before creating a 3D archive.")
    parser.add_argument("front", type=Path)
    parser.add_argument("back", type=Path)
    args = parser.parse_args()
    try:
        with Image.open(args.front) as front, Image.open(args.back) as back:
            _, evidence = paired_features(front, back)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
