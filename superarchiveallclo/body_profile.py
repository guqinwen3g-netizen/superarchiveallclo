"""Observe chest and hem widths on the same calibrated flat layout."""

import numpy as np

from .tee import TeeSpec, draft

FIELDS = ("flat_chest_mm", "flat_hem_mm")


def fit_side(y, edge):
    fit = np.polyfit(y, edge, 1)
    residual = np.abs(np.polyval(fit, y) - edge)
    inliers = residual <= max(2, 3 * np.median(residual))
    if inliers.sum() < 0.70 * len(y):
        return None
    fit = np.polyfit(y[inliers], edge[inliers], 1)
    return fit, float(np.quantile(np.abs(np.polyval(fit, y) - edge), 0.90))


def underarm_at(mask, side_line, sign, step):
    # Follow narrow strips outside the side seam. The end of each sustained
    # sleeve run samples the underside of the sleeve near the underarm.
    offsets = np.arange(8, 44, 2)
    height, width = mask.shape
    rows = np.arange(height)
    run = max(3, round(14 / step))
    search = slice(round(0.18 * height), round(0.58 * height))
    roots = np.full(len(offsets), np.nan)
    for i, offset in enumerate(offsets):
        columns = np.rint(side_line + sign * offset / step).astype(int)
        in_frame = (columns >= 0) & (columns < width)
        visible = in_frame & mask[rows, columns.clip(0, width - 1)]
        supported = np.convolve(visible, np.ones(run, dtype=int), mode="valid") == run
        ends = np.flatnonzero(supported) + run - 1
        ends = ends[(ends >= search.start) & (ends < search.stop)]
        if ends.size:
            roots[i] = ends[-1] * step
    # A pleat or shadow can shift the innermost strip. Choose the closest run
    # of mutually consistent offsets and extend that short trace to the seam.
    for start in range(len(offsets) - 6):
        segment = slice(start, start + 7)
        values = roots[segment]
        if not np.isfinite(values).all():
            continue
        slope, join = np.polyfit(offsets[segment], values, 1)
        error = np.max(np.abs(values - (slope * offsets[segment] + join)))
        if -0.65 <= slope <= 2.15 and error <= max(4, 1.6 * step) and search.start <= join / step < search.stop:
            return join / step
    return None


def observe(mask, step):
    """Estimate straight side seams independently of the learned dimensions."""
    height, width = mask.shape
    if min(height, width) < 48 or not np.isfinite(step) or height * step < 28:
        return None
    rows = np.arange(height)
    fit_band = slice(round(0.58 * height), round(0.85 * height))
    left = mask.argmax(axis=1)
    right = width - 1 - mask[:, ::-1].argmax(axis=1)
    sides = [fit_side(rows[fit_band], edge[fit_band]) for edge in (left, right)]
    if any(side is None for side in sides):
        return None
    lines = [np.polyval(side[0], rows) for side in sides]
    torso = float(np.median((lines[1] - lines[0])[fit_band]) * step)
    residuals = [p90 * step for _, p90 in sides]
    if torso <= 0 or max(residuals) > 0.025 * torso:
        return None
    center = round(float(np.median(((lines[0] + lines[1]) / 2)[fit_band])))
    reach = round(0.22 * torso / step)
    if reach < 2 or center - reach < 0 or center + reach >= width:
        return None
    hem = float(np.median(height - 1 - mask[::-1, center - reach:center + reach + 1].argmax(axis=0)))
    joins = [underarm_at(mask, line, sign, step) for line, sign in zip(lines, (-1, 1), strict=True)]
    if (any(join is None for join in joins) or hem < 0.91 * height
            or abs(joins[0] - joins[1]) * step > max(38, 0.12 * height * step)):
        return None

    chest_points = [[float(np.polyval(fit, y)), y] for (fit, _), y in zip(sides, joins, strict=True)]
    hem_points = [[float(np.polyval(fit, hem)), hem] for fit, _ in sides]
    chest, hem_width = (float((points[1][0] - points[0][0] + 1) * step) for points in (chest_points, hem_points))
    if not (0.82 * torso <= chest <= 1.18 * torso and 0.76 * chest <= hem_width <= 1.30 * chest):
        return None
    return {
        "chest_width_mm": round(chest, 2), "hem_width_mm": round(hem_width, 2),
        "chest_points_crop_px": np.round(chest_points, 2).tolist(),
        "hem_points_crop_px": np.round(hem_points, 2).tolist(),
        "side_fit_p90_mm": np.round(residuals, 2).tolist(),
    }


def constrain(measures, scale):
    profiles = [scale[view].get("torso_profile") for view in ("front", "back")]
    if any(profile is None for profile in profiles):
        return {}, {"status": "paired_landmarks_unavailable"}
    estimates = {}
    for field, key in zip(FIELDS, ("chest_width_mm", "hem_width_mm"), strict=True):
        a, b = (profile[key] for profile in profiles)
        if abs(a - b) > max(16, 0.045 * min(a, b)):
            return {}, {"status": "paired_widths_disagree", "field": field}
        estimates[field] = round((a + b) / 2, 3)
    try:
        draft(TeeSpec(**{**measures, **estimates}))
    except ValueError as error:
        return {}, {"status": "pattern_incompatible", "reason": str(error)}
    return estimates, {
        "status": "applied",
        "regression_before_mm": {field: measures[field] for field in estimates if field in measures},
        "sources": dict.fromkeys(estimates, "paired_silhouette"),
    }
