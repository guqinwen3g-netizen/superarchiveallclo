"""Recover the in-plane axis of a fully visible, flat crew-neck tee."""

import numpy as np
from scipy import ndimage

from .scan import project


def tee_orientation_score(mask):
    height, width = mask.shape
    rows = np.arange(round(0.54 * height), round(0.85 * height))
    if len(rows) < 10 or not mask[rows].any(axis=1).all():
        return float("-inf")

    left = mask.argmax(axis=1)
    right = width - 1 - mask[:, ::-1].argmax(axis=1)
    torso_width = np.median((right - left)[rows])
    if not max(16, 0.42 * height) <= torso_width <= 1.17 * height:
        return float("-inf")

    center = (right + left) / 2
    torso_center = np.median(center[rows])
    sleeve_rows = slice(round(0.10 * height), round(0.45 * height))
    left_extension = (np.median(left[rows]) - np.quantile(left[sleeve_rows], 0.08)) / torso_width
    right_extension = (np.quantile(right[sleeve_rows], 0.92) - np.median(right[rows])) / torso_width
    if min(left_extension, right_extension) < 0.06:
        return float("-inf")

    # Judge the hem only within the center of the torso to avoid folded corners.
    inner = np.arange(round(torso_center - 0.33 * torso_width), round(torso_center + 0.33 * torso_width))
    inner = inner[(0 <= inner) & (inner < width)]
    if not len(inner) or not mask[:, inner].any(axis=0).all():
        return float("-inf")
    hem = height - 1 - mask[::-1, inner].argmax(axis=0)
    hem_roughness = (np.quantile(hem, 0.9) - np.quantile(hem, 0.1)) / height
    hem_inset = (height - 1 - np.median(hem)) / height
    drift = np.ptp(center[rows]) / torso_width
    side_error = max(
        np.quantile(abs(border[rows] - np.polyval(np.polyfit(rows, border[rows], 1), rows)), 0.9) / torso_width
        for border in (left, right)
    )
    top_width = np.median((right - left)[round(0.025 * height):max(2, round(0.08 * height))]) / torso_width
    sleeve_support = min(left_extension, right_extension, 0.35)
    return float(sleeve_support - 2.5 * hem_roughness - 1.5 * hem_inset - 2 * side_error - 0.55 * drift
                 - 0.5 * max(0, top_width - 0.70) - 0.30 * abs(left_extension - right_extension))


def upright_angle(mask):
    def score(angle):
        rotated = ndimage.rotate(mask, angle, order=0)
        y, x = np.where(rotated)
        if not len(y):
            return float("-inf")
        return tee_orientation_score(rotated[y.min():y.max() + 1, x.min():x.max() + 1])

    candidates = [(score(angle), angle) for angle in range(-180, 180, 5)]
    coarse_score, coarse = max(candidates)
    if not np.isfinite(coarse_score):
        raise ValueError("Could not resolve an upright tee from the visible sleeve and hem outlines")
    best, angle = max((score(coarse + offset), coarse + offset) for offset in np.arange(-4.5, 5, 0.5))
    competing = max(value for value, other in candidates if abs((other - angle + 180) % 360 - 180) > 35)
    if best < 0.04 or best - competing < 0.08:
        raise ValueError("Could not resolve an upright tee from the visible sleeve and hem outlines")
    return float((angle + 180) % 360 - 180), best


def orient_flat_points(points):
    preview, _ = project(points, "x", "y", 4)
    angle, score = upright_angle(np.asarray(preview) < 128)
    # Preserve a capture already upright to within the landmark placement tolerance.
    applied = 0.0 if abs(angle) < 3 else angle
    rad = np.radians(applied)
    cosine, sine = np.cos(rad), np.sin(rad)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return points @ rotation, rotation, {
        "upright_estimated_rotation_deg": angle,
        "upright_rotation_deg": applied,
        "upright_outline_score": round(score, 5),
    }
