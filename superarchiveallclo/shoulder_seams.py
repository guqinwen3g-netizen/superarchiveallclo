"""Find narrow, sustained sleeve seam traces on calibrated flat-lay photos."""

from dataclasses import replace
from math import isfinite

import numpy as np
from scipy import ndimage

from .tee import draft


def trace_side(ridge, interior, top, underarm, chest, step, noise):
    height, width = ridge.shape
    root_x, root_y = underarm
    if not 0 <= root_y < height:
        return None
    # A set-in seam runs from the upper silhouette to the side seam. Allow a
    # different shoulder extension on each side and a curved sleeve join.
    apex = np.arange(max(1, round(root_x - 0.11 * chest / step)),
                     min(width - 1, round(root_x + 0.15 * chest / step)), max(1, round(2.25 / step)))
    ends = np.arange(root_x - 14 / step, root_x + 15 / step, 3.5 / step)
    bends = np.arange(-0.12 * chest / step, 0.12 * chest / step, 5 / step)
    if not apex.size or not bends.size:
        return None

    apex_x, end_x, bend = (part.ravel() for part in np.meshgrid(apex, ends, bends, indexing="ij"))
    apex_y = top[apex_x]
    t = np.arange(17, 87, 2) / 100
    x = (1 - t) * apex_x[:, None] + t * end_x[:, None] + 4 * t * (1 - t) * bend[:, None]
    y = (1 - t) * apex_y[:, None] + t * root_y
    inside = ndimage.map_coordinates(interior, [y, x], order=1)
    contrast = ndimage.map_coordinates(ridge, [y, x], order=1)
    signal = np.clip(contrast / noise, -4, 4)
    score = signal.mean(axis=1) - 0.07 * np.abs(bend * step) / (0.12 * chest)
    supports = (signal > 0.5).reshape(len(score), 5, -1).mean(axis=2)
    usable = ((inside > 0.9).all(axis=1) & ((root_y - apex_y) * step >= 65)
              & (supports.mean(axis=1) >= 0.70) & (supports.min(axis=1) >= 0.10)
              & (score >= 1.10))
    if not usable.any():
        return None

    best = int(np.argmax(np.where(usable, score, -np.inf)))
    near_ties = usable & (score >= 0.88 * score[best])
    return {
        "shoulder_crop_px": [int(apex_x[best]), int(apex_y[best])],
        "trace_crop_px": np.column_stack((x[best], y[best])).round(2).tolist(),
        "supported_apex_x_crop_px": [int(apex_x[near_ties].min()), int(apex_x[near_ties].max())],
        "contrast_score": round(float(score[best]), 3),
        "section_support": supports[best].round(3).tolist(),
    }


def observe(gray, mask, step, torso):
    """Return bilateral, image-only shoulder evidence when both seams are visible."""
    chest = torso["chest_width_mm"]
    blurred = ndimage.gaussian_filter(gray, (max(0.7, 2 / step), max(0.7, 1.25 / step)))
    ridge = ndimage.gaussian_filter1d(blurred, max(1, 3.5 / step), axis=1) - blurred
    interior = ndimage.distance_transform_edt(mask) * step >= 6
    if not interior.any():
        return None
    noise = max(0.6, float(np.median(np.abs(ridge[interior]))))
    interior = interior.astype(float)
    top = ndimage.median_filter(mask.argmax(axis=0), size=5)
    border = mask.shape[1] - 1
    left_root, right_root = torso["chest_points_crop_px"]
    left = trace_side(ridge, interior, top, left_root, chest, step, noise)
    if left is None:
        return None
    right = trace_side(ridge[:, ::-1], interior[:, ::-1], top[::-1],
                       (border - right_root[0], right_root[1]), chest, step, noise)
    if right is None:
        return None
    right["shoulder_crop_px"][0] = border - right["shoulder_crop_px"][0]
    right["trace_crop_px"] = [[round(border - x, 2), y] for x, y in right["trace_crop_px"]]
    right["supported_apex_x_crop_px"] = [border - x for x in reversed(right["supported_apex_x_crop_px"])]
    distance = (right["shoulder_crop_px"][0] - left["shoulder_crop_px"][0]) * step
    left_min, left_max = left["supported_apex_x_crop_px"]
    right_min, right_max = right["supported_apex_x_crop_px"]
    return {
        "projected_width_mm": round(float(distance), 2),
        "near_tie_width_range_mm": [round((right_min - left_max) * step, 2),
                                    round((right_max - left_min) * step, 2)],
        "image_left": left, "image_right": right,
    }


def paired_evidence(views, predicted_width):
    front, back = (views[name].get("projected_shoulder_seams") for name in ("front", "back"))
    if front is None or back is None:
        return {"status": "bilateral_seams_unavailable"}
    a, b = (view["projected_width_mm"] for view in (front, back))
    result = {"view_width_delta_mm": round(abs(a - b), 2)}
    if abs(a - b) > max(20, 0.05 * min(a, b)):
        return {"status": "projections_disagree", **result}
    average = (a + b) / 2
    return {"status": "measured_projection", **result, "projected_width_mm": round(average, 2),
            "pattern_minus_projection_mm": round(predicted_width - average, 2)}


def refine_width(spec, pattern, capture):
    """Use a stable projected shoulder width while retaining a sewable candidate."""
    field = "garment_shoulder_width_mm"
    measurement = capture["measurements"][field]
    if "observation" in measurement:
        return spec, pattern, {"status": "retained_supplied_measurement"}
    observed = {}
    scale = capture.get("inference", {}).get("scale_evidence", {})
    for name in ("front", "back"):
        if name not in capture["views"]:
            continue
        view = capture["views"][name]
        detail = view.get("calibration", {}).get("registered_color_seams", {})
        traces = detail.get("projected_shoulder_seams") or scale.get(name, {}).get("projected_shoulder_seams")
        if traces is None:
            return spec, pattern, {"status": "unavailable", "reason": f"No bilateral seam traces in the {name} view"}
        width = float(traces["projected_width_mm"])
        low, high = map(float, traces["near_tie_width_range_mm"])
        if (not all(map(isfinite, (low, width, high))) or not 0 < low <= width <= high
                or high - low > max(20, 0.05 * width)):
            return spec, pattern, {"status": "uncertain_trace", "view": name,
                                   "near_tie_width_range_mm": [low, high]}
        observed[name] = {"projected_shoulder_seams": traces}
    proposed = float(observed["front"]["projected_shoulder_seams"]["projected_width_mm"])
    result = {"regression_before_mm": spec.shoulder_width_mm, "widths_by_view_mm": {
        name: values["projected_shoulder_seams"]["projected_width_mm"] for name, values in observed.items()}}
    if "back" in observed:
        paired = paired_evidence(observed, spec.shoulder_width_mm)
        if paired["status"] != "measured_projection":
            return spec, pattern, {**result, **paired}
        proposed = paired["projected_width_mm"]
    selected = replace(spec, garment_shoulder_width_mm=proposed)
    try:
        revised = draft(selected)
    except ValueError as error:
        return spec, pattern, {**result, "status": "pattern_incompatible", "reason": str(error)}
    capture["measurements"][field] = {
        "source": "projected_shoulder_seams", "value": proposed, "before_fit_mm": spec.shoulder_width_mm,
    }
    return selected, revised, {**result, "status": "applied", "selected_mm": proposed}
