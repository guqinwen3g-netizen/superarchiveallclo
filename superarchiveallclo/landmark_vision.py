"""Predict labeled T-shirt image landmarks from an upright or rotated silhouette."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .vision import garment_crop, picture_silhouette


MODEL_VERSION = 1
NAMES = (
    "shoulder_left", "neck_left", "neck_right", "shoulder_right", "sleeve_right_top", "sleeve_right_bottom",
    "armpit_right", "waist_right", "waist_left", "armpit_left", "sleeve_left_bottom", "sleeve_left_top",
)
REFLECT = [3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4]
SIDE = 28
grid = (np.arange(SIDE) + 0.5) / SIDE - 0.5
U, V = np.meshgrid(grid, grid)
ANGLES = np.arange(-180, 180, 10)
CHORDS = {
    "shoulder_span": (0, 3), "neck_opening_span": (1, 2), "underarm_span": (6, 9),
    "left_cuff_span": (10, 11), "right_cuff_span": (4, 5), "hem_span": (7, 8),
}
SCOPE = "Image landmarks are named left/right with the neckline up. Chords are in-plane projections."


def axes(degrees):
    cosine, sine = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[cosine, sine], [-sine, cosine]])


class Silhouette:
    def __init__(self, mask):
        y, x = np.where(mask)
        if len(x) < 100 or min(np.ptp(x), np.ptp(y)) < 16:
            raise ValueError("Keep the garment large enough to resolve its outline")
        self.center = np.array([x.mean(), y.mean()])
        self.scale = float(2.06 * np.hypot(x - self.center[0], y - self.center[1]).max())
        self.diagonal = float(np.hypot(np.ptp(x), np.ptp(y)))
        self.mask = mask.astype(np.float32)

    def features(self, angle):
        matrix = axes(angle) * self.scale
        x = self.center[0] + U * matrix[0, 0] + V * matrix[0, 1]
        y = self.center[1] + U * matrix[1, 0] + V * matrix[1, 1]
        return ndimage.map_coordinates(self.mask, [y, x], order=1, prefilter=False).ravel()

    def normalize(self, xy, angle):
        return (xy - self.center) @ axes(angle) / self.scale

    def pixels(self, xy, angle):
        return xy @ axes(angle).T * self.scale + self.center


def canonical_labels(points):
    """Return labels in image-left/right order after making the neckline upright."""
    ends = points[[1, 2, 7, 8]]
    if np.any(ends[:, 2] == 0):
        return None, None
    direction = ends[2:, :2].mean(axis=0) - ends[:2, :2].mean(axis=0)
    if np.linalg.norm(direction) < 8:
        return None, None
    angle = float(np.degrees(np.arctan2(*direction)))
    side = axes(angle)[:, 0]
    if (points[7, :2] - points[8, :2]) @ side < 0:
        points = points[REFLECT]
    return angle, points


def kernel(features, support):
    norms = np.sum(features * features, axis=1)[:, None] + np.sum(support * support, axis=1)
    distances = np.maximum(0, norms - 2 * features @ support.T) / features.shape[1]
    return np.exp(-distances / 0.045)


def alignment_scores(features, support, neighbors):
    matches = kernel(features, support)
    return np.partition(matches, -neighbors, axis=1)[:, -neighbors:].mean(axis=1)


def infer_orientation(silhouette, support, neighbors):
    features = np.stack([silhouette.features(angle) for angle in ANGLES])
    scores = alignment_scores(features, support, neighbors)
    index = int(scores.argmax())
    coarse = ANGLES[index]
    local = coarse + np.arange(-8, 9, 2)
    refined = np.stack([silhouette.features(angle) for angle in local])
    refined_scores = alignment_scores(refined, support, neighbors)
    best = int(refined_scores.argmax())
    angle = float((local[best] + 180) % 360 - 180)
    distinct = abs((ANGLES - angle + 180) % 360 - 180) >= 45
    margin = float(refined_scores[best] - scores[distinct].max())
    return angle, refined[best], margin


class TeeLandmarkVision:
    def __init__(self, path):
        encoded = path.read_bytes()
        self.checksum = sha256(encoded).hexdigest()
        with np.load(BytesIO(encoded), allow_pickle=False) as weights:
            if int(weights["version"]) != MODEL_VERSION or tuple(weights["names"]) != NAMES:
                raise ValueError("Unsupported tee landmark weights")
            self.neighbors = int(weights["neighbors"])
            self.support = weights["support"]
            self.coefficients = weights["coefficients"]
            self.mean = weights["mean"]

    def predict(self, silhouette):
        angle, features, margin = infer_orientation(silhouette, self.support, self.neighbors)
        influence = kernel(features[None, :], self.support)[0]
        points = self.mean + (influence @ self.coefficients).reshape(-1, 2)
        return angle, silhouette.pixels(points, angle), margin

    def photo_evidence(self, photo, view):
        processed, crop, bounds, mm_per_pixel, _ = picture_silhouette(photo, view, outer_only=True)
        angle, points, margin = self.predict(Silhouette(crop))
        size = view["upright_px"]
        factor = np.array(size) / np.array(processed.size)
        return {
            "status": "candidate",
            "photo_sha256": view["sha256"],
            **image_record(size, (points + bounds[:2] + 0.5) * factor - 0.5, angle, margin),
            "photo_projected_chords_mm": {
                name: round(float(np.linalg.norm(points[a] - points[b]) * mm_per_pixel), 2)
                for name, (a, b) in CHORDS.items()
            },
        }


def image_record(size, points, angle, margin):
    return {
        "source_upright_px": list(size), "down_axis_deg_from_image_vertical": angle,
        "orientation_score_margin": round(margin, 4),
        "keypoints_px": dict(zip(NAMES, points.round(2).tolist(), strict=True)),
    }


def read_silhouette(photo, mask_path):
    size = np.array(photo.size)
    photo.thumbnail((720, 720), Image.Resampling.BOX)
    if mask_path is not None:
        with Image.open(mask_path) as encoded:
            if tuple(size) != encoded.size or encoded.format != "PNG" or np.asarray(encoded).ndim != 2:
                raise ValueError("Use a single-channel PNG mask matching the upright-orientation photo raster")
            mask = np.asarray(encoded.resize(photo.size, Image.Resampling.NEAREST)) > 0
    else:
        rgb = np.asarray(photo, dtype=np.int16)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        crop, (left, top, right, bottom) = garment_crop(rgb, mask, fill_interior=True)
        mask[top:bottom + 1, left:right + 1] = crop
    return Silhouette(mask), size / np.array(photo.size)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--mask", type=Path, help="Optional aligned silhouette PNG for a textured background")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    try:
        with Image.open(args.photo) as encoded:
            image = ImageOps.exif_transpose(encoded).convert("RGB")
        model = TeeLandmarkVision(args.model)
        silhouette, factor = read_silhouette(image.copy(), args.mask)
        angle, xy, margin = model.predict(silhouette)
        source_points = (xy + 0.5) * factor - 0.5
        print(json.dumps({
            "scope": SCOPE, "model_sha256": model.checksum,
            **image_record(image.size, source_points, angle, margin),
        }, indent=2))
        if args.overlay is not None:
            draw = ImageDraw.Draw(image)
            radius = max(3, round(max(image.size) / 180))
            for x, y in source_points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#35d9ee", width=2)
            image.save(args.overlay)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
