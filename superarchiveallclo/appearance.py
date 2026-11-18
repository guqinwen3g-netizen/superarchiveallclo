"""Bake observed fabric color and front/back photographs into a paper-space atlas."""

from hashlib import sha256
import json

import numpy as np
from PIL import Image
from scipy import ndimage

from .body_profile import observe
from .pattern import Edge
from .vision import picture_silhouette

ATLAS_SIZE = (2048, 1024)
PANELS = {
    "front": [0, 0, 1024, 896],
    "back": [1024, 0, 1024, 896],
    "sleeve_right": [0, 896, 640, 128],
    "sleeve_left": [640, 896, 640, 128],
    "collar": [1280, 896, 768, 128],
}


def sample_fabric(rgb, mask):
    height, width = mask.shape
    area = np.s_[round(0.57 * height):round(0.88 * height), round(0.37 * width):round(0.63 * width)]
    pixels = rgb[area][mask[area]]
    if len(pixels) < 16:
        pixels = rgb[mask]
    return np.rint(np.median(pixels, axis=0)).astype(np.uint8)


def shoulder_row(mask, center, width):
    x = np.arange(mask.shape[1])
    offsets = np.abs(x - center)
    near_neck = (offsets >= 0.15 * width) & (offsets <= 0.23 * width) & mask.any(axis=0)
    if near_neck.any():
        return float(np.median(mask[:, near_neck].argmax(axis=0)))
    return 0.0


def paper_bounds(panel):
    xy = np.asarray([point for edge in panel["edges"] for point in Edge(**edge).samples()])
    return [*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist()]


def photo_coordinates(paper_x, paper_y, parameters, profile, neck_y, view):
    """Map flat body coordinates to the photographed clothing, including the rear view."""
    chest_y = parameters["armhole_depth_mm"]
    hem_y = parameters["back_length_mm"] + parameters["back_neck_depth_mm"]
    chest = np.array(profile["chest_points_crop_px"])
    hem = np.array(profile["hem_points_crop_px"])
    chest_center, hem_center = chest.mean(axis=0), hem.mean(axis=0)
    chest_px, hem_px = abs(chest[1, 0] - chest[0, 0]), abs(hem[1, 0] - hem[0, 0])
    width = parameters["flat_chest_mm"]
    hem_width = parameters.get("flat_hem_mm", width)
    section = np.clip((paper_y - chest_y) / (hem_y - chest_y), 0, 1)
    centers = chest_center[0] + section * (hem_center[0] - chest_center[0])
    photo_widths = chest_px + section * (hem_px - chest_px)
    paper_widths = width + section * (hem_width - width)
    reach = paper_x if view == "front" else -paper_x
    return (centers + reach * photo_widths / paper_widths,
            np.interp(paper_y, [0, chest_y, hem_y], [neck_y, chest_center[1], hem_center[1]]))


def bake_body(rgb, mask, scale, panel_bounds, tile_size, profile, fabric, view, parameters):
    width, height = tile_size
    if profile is None:
        return Image.new("RGB", tile_size, tuple(fabric)), {"mapping": "observed_solid_color"}
    x0, y0, x1, y1 = panel_bounds
    paper_x = np.linspace(x0, x1, width)[None, :]
    paper_y = np.linspace(y0, y1, height)[:, None]
    chest = np.asarray(profile["chest_points_crop_px"])
    neck_y = shoulder_row(mask, chest[:, 0].mean(), abs(chest[1, 0] - chest[0, 0]))
    x, y = photo_coordinates(paper_x, paper_y, parameters, profile, neck_y, view)
    y = np.broadcast_to(y, x.shape)
    # Use measured interiors and extend them narrowly around the cut line.
    # Atlas samples away from reliable cloth receive an observed solid color.
    inside = ndimage.binary_erosion(mask)
    if not inside.any():
        return Image.new("RGB", tile_size, tuple(fabric)), {"mapping": "observed_solid_color"}
    distance, nearest = ndimage.distance_transform_edt(~inside, return_indices=True)
    rows = np.rint(y).astype(int).clip(0, mask.shape[0] - 1)
    cols = np.rint(x).astype(int).clip(0, mask.shape[1] - 1)
    trusted = inside[rows, cols]
    supported = distance[rows, cols] <= 12 / scale
    supported &= (x >= -12 / scale) & (x < mask.shape[1] + 12 / scale)
    supported &= (y >= 0) & (y < mask.shape[0])
    coords = (np.where(trusted, y, nearest[0, rows, cols]), np.where(trusted, x, nearest[1, rows, cols]))
    baked = np.stack([ndimage.map_coordinates(rgb[:, :, i], coords, order=1, mode="nearest") for i in range(3)], axis=-1)
    baked[~supported] = fabric
    return Image.fromarray(baked), {
        "mapping": "shoulder_underarm_hem",
        "shoulder_row_crop_px": round(neck_y, 2),
        "underarm_crop_px": profile["chest_points_crop_px"],
        "hem_crop_px": profile["hem_points_crop_px"],
    }


def build_appearance(bundle, manifest, pattern):
    if not {"front", "back"} <= manifest["views"].keys():
        raise ValueError("Photographic appearance requires distinct front and back views")
    front, back = (manifest["views"][name] for name in ("front", "back"))
    if front["sha256"] == back["sha256"]:
        raise ValueError("Photographic appearance requires distinct front and back views")
    atlas = Image.new("RGB", ATLAS_SIZE)
    panels = {}
    source = {}
    colors = []
    for panel in pattern["panels"]:
        name = panel["id"]
        rectangle = PANELS[name]
        panels[name] = {"rect_px": rectangle}
        if name not in ("front", "back"):
            continue
        view = manifest["views"][name]
        with Image.open(bundle / view["file"]) as original:
            photo, mask, crop, scale, _ = picture_silhouette(original, view, outer_only=True)
        left, top, right, bottom = crop
        rgb = np.asarray(photo)[top:bottom + 1, left:right + 1]
        fabric = sample_fabric(rgb, mask)
        colors.append(fabric)
        profile = observe(mask, scale)
        bounds = paper_bounds(panel)
        panels[name]["bounds_mm"] = bounds
        baked, evidence = bake_body(rgb, mask, scale, bounds, rectangle[2:], profile, fabric,
                                    name, pattern["parameters"])
        atlas.paste(baked, tuple(rectangle[:2]))
        source[name] = {
            "sha256": view["sha256"], "processed_crop_px": crop,
            "fabric_srgb": fabric.tolist(), **evidence,
        }
    fabric = np.rint(np.mean(colors, axis=0)).astype(np.uint8).tolist()
    for name in ("sleeve_left", "sleeve_right", "collar"):
        x, y, width, height = PANELS[name]
        atlas.paste(tuple(fabric), (x, y, x + width, y + height))
    path = bundle / "model/appearance.png"
    atlas.save(path)
    record = {
        "schema_version": 1, "appearance": "paired_flat_photographs",
        "atlas": path.name, "atlas_px": list(ATLAS_SIZE), "atlas_sha256": sha256(path.read_bytes()).hexdigest(),
        "panels": panels, "photograph_evidence": source,
        "sampled_sleeve_and_collar_color_srgb": fabric,
    }
    (bundle / "model/appearance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
