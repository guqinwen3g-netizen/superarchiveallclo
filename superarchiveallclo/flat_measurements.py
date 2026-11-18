"""Visible landmarks of a metric flat-lay silhouette, before estimating seams."""

import numpy as np
from scipy import ndimage

from .tee import MIN_SLEEVE_BELOW_CAP_MM, TeeSpec, body_panel, cap_dimensions, half_cap, sleeve_panels


def sleeve_cuff(outline, raw, torso_x, torso_width, mm_per_pixel):
    limit = round(0.62 * len(outline))
    edge = outline[:limit]
    extent = edge.max() - torso_x
    if extent < 0.075 * torso_width:
        return None
    start = int(np.flatnonzero(edge >= edge.max() - 3)[0])
    # The folded cuff is roughly vertical in an upright flat lay. End its
    # trace at the sustained inward turn toward the underarm seam.
    rolling = max(2, round(14 / mm_per_pixel))
    inward_turn = np.gradient(edge, mm_per_pixel) < -0.90
    turn_support = np.convolve(inward_turn, np.ones(rolling), mode="valid")
    ends = np.flatnonzero(turn_support >= rolling)
    ends = ends[ends >= start + round(0.13 * torso_width / mm_per_pixel)]
    if not ends.size:
        return None
    end = int(ends[0])
    # Refine clean edges against the original pixel trace. The smoothing pass
    # finds turns on wrinkled cloth but moves the corners of a straight cuff.
    inset = max(1, round((end - start) * 0.18))
    core = np.arange(start + inset, end - inset + 1)
    if len(core) >= 4:
        slope, intercept = np.polyfit(core, raw[core], 1)
        error = np.abs(raw - (slope * np.arange(len(raw)) + intercept))
        if np.quantile(error[core], 0.9) <= 3:
            margin = round(16 / mm_per_pixel)
            before = np.flatnonzero(error[max(0, start - margin):core[0]] > 4)
            after = np.flatnonzero(error[core[-1] + 1:min(limit, end + margin + 1)] > 4)
            if before.size and after.size:
                start = max(0, start - margin) + int(before[-1]) + 1
                end = int(core[-1] + after[0])
                edge = raw[:limit]
    delta_y = (end - start) * mm_per_pixel
    delta_x = edge[end] - edge[start]
    length = float(np.hypot(delta_x, delta_y))
    chord = np.linspace(edge[start], edge[end], end - start + 1)
    precision = float(np.quantile(np.abs(chord - edge[start:end + 1]), 0.9))
    if not 0.14 * torso_width <= length <= 0.42 * torso_width or abs(delta_x) > delta_y or precision > 0.13 * length:
        return None
    torso_rows = np.flatnonzero(outline[end + 1:limit] <= torso_x + max(7, 0.06 * extent))
    return {
        "outer_mm": [float(edge[start]), start * mm_per_pixel],
        "inner_mm": [float(edge[end]), end * mm_per_pixel],
        "folded_opening_mm": round(length, 2),
        "edge_fit_p90_mm": round(precision, 2),
        "underarm_join_y_mm": (end + 1 + int(torso_rows[0])) * mm_per_pixel if torso_rows.size else None,
    }


def visible_neck(mask, center_x, torso_width, mm_per_pixel):
    width = mask.shape[1]
    raw = mask.argmax(axis=0).astype(float) * mm_per_pixel
    top = ndimage.gaussian_filter1d(raw, max(0.65, 6 / mm_per_pixel))
    x = np.arange(width) * mm_per_pixel
    offset = x - center_x
    # Keep the search on the neck-facing ends of the shoulders.
    left, right = [np.flatnonzero((side * offset >= 0.07 * torso_width) & (side * offset <= 0.28 * torso_width))
                   for side in (-1, 1)]
    central = np.flatnonzero(np.abs(offset) <= 0.065 * torso_width)
    if not len(left) or not len(right) or not len(central):
        return None
    a = left[top[left].argmin()]
    b = right[top[right].argmin()]
    opening = x[b] - x[a]
    baseline = (top[a] + top[b]) / 2
    gap_center = (x >= x[a] + 0.35 * opening) & (x <= x[b] - 0.35 * opening)
    depth = np.median(top[gap_center]) - baseline
    mid_offset = abs((x[a] + x[b]) / 2 - center_x)
    if (not 0.19 * torso_width <= opening <= 0.45 * torso_width or not 12 <= depth <= 0.24 * torso_width
            or abs(top[a] - top[b]) > 18 or mid_offset > opening * 0.19):
        return None
    # Verify that the shoulder extrema bound the central notch rather than
    # spanning over a second fold at the rim. Use its nearest inner crossings.
    peak = central[top[central].argmax()]
    threshold = baseline + 0.25 * (top[peak] - baseline)
    near_rim = np.flatnonzero(top <= threshold)
    inner_left = near_rim[near_rim < peak]
    inner_right = near_rim[near_rim > peak]
    confirmed = bool(len(inner_left) and len(inner_right)
                     and x[inner_right[0]] - x[inner_left[-1]] >= 0.72 * opening)
    if confirmed:
        refined = ndimage.median_filter(raw, size=3)
        radius = round(12 / mm_per_pixel)
        a, b = (int(window[refined[window].argmin()]) for window in (
            left[abs(left - a) <= radius], right[abs(right - b) <= radius],
        ))
        opening = x[b] - x[a]
    return {
        "opening_width_mm": round(float(opening), 2),
        "silhouette_depth_mm": round(float(depth), 2),
        "corners_mm": [[round(float(x[i]), 2), round(float(top[i]), 2)] for i in (a, b)],
        "confirmed_by_inner_notch": confirmed,
    }


def cuff_axes(cuff, side):
    outer = np.asarray(cuff["outer_mm"])
    chord = np.asarray(cuff["inner_mm"]) - outer
    across = chord / np.linalg.norm(chord)
    outward = np.array([side * across[1], -side * across[0]])
    return outer, across, outward


def sleeve_layout(cuffs, center, torso_width):
    """Resolve the visible underarm join along and across each sleeve."""
    layout = {}
    for side, tag in ((-1, "image_left"), (1, "image_right")):
        cuff = cuffs[tag]
        height = cuff["underarm_join_y_mm"]
        if height is None:
            return None
        underarm = np.array([center + side * torso_width / 2, height])
        tip, across, outward = cuff_axes(cuff, side)
        layout[tag] = {
            "underarm_mm": underarm.round(2).tolist(),
            "folded_width_at_join_mm": round(float((underarm - tip) @ across), 2),
            "join_to_cuff_run_mm": round(float((tip - underarm) @ outward), 2),
        }
    return layout


def flat_landmarks(mask, mm_per_pixel, torso_width):
    """Return only bilateral landmarks supported by this projected view."""
    height, width = mask.shape
    rows = slice(round(0.58 * height), round(0.86 * height))
    reflect_x = (width - 1) * mm_per_pixel
    sides = {}
    body_edges = []
    for side, image in (("image_right", mask), ("image_left", mask[:, ::-1])):
        x = (width - 1 - image[:, ::-1].argmax(axis=1)) * mm_per_pixel
        contour = ndimage.gaussian_filter1d(x.astype(float), max(0.65, 5 / mm_per_pixel))
        center_edge = float(np.median(contour[rows]))
        is_left = side == "image_left"
        body_edges.append(reflect_x - center_edge if is_left else center_edge)
        cuff = sleeve_cuff(contour, x, center_edge, torso_width, mm_per_pixel)
        if cuff is None:
            continue
        for key in ("outer_mm", "inner_mm"):
            x, y = cuff[key]
            cuff[key] = [round(reflect_x - x if is_left else x, 2), round(y, 2)]
        sides[side] = cuff
    center = sum(body_edges) / 2
    evidence = {"torso_center_x_mm": round(center, 2)}
    if len(sides) == 2:
        left, right = (sides[side]["folded_opening_mm"] for side in ("image_left", "image_right"))
        if max(left, right) <= 1.18 * min(left, right):
            evidence["cuffs"] = sides
            layout = sleeve_layout(sides, center, torso_width)
            if layout is not None:
                evidence["sleeve_layout"] = layout
    neckline = visible_neck(mask, center, torso_width, mm_per_pixel)
    if neckline is not None:
        evidence["visible_neck"] = neckline
    return evidence


def projected_sleeve(measures, updates, opening, layout, top):
    """Use both underarms and the folded cuff directions to constrain the sleeve cap."""
    rows = tuple(layout.values())
    widths = [row["folded_width_at_join_mm"] for row in rows]
    runs = [row["join_to_cuff_run_mm"] for row in rows]
    depths = [row["underarm_mm"][1] - top for row in rows]
    chest = updates["flat_chest_mm"]
    bicep = sum(widths)
    depth = sum(depths) / 2
    if (min(runs) < 25 or max(runs) > 1.25 * min(runs) or max(widths) > 1.18 * min(widths)
            or max(depths) - min(depths) > 0.065 * chest or not opening <= bicep <= 0.94 * chest
            or depth <= measures["shoulder_slope_mm"] + 40 or depth > 0.65 * chest):
        return None
    proposed = {"sleeve_bicep_flat_mm": bicep, "sleeve_opening_flat_mm": opening, "armhole_depth_mm": depth}
    spec = TeeSpec(**{**measures, **updates, **proposed})
    front, back = (body_panel(spec, side).edge("armhole_right").length() for side in (True, False))
    try:
        _, cap = cap_dimensions(bicep, min(front, back), front, back)
    except ValueError:
        return None
    length = cap + sum(runs) / 2
    if not 0.17 * chest <= length <= 0.60 * chest:
        return None
    return {**proposed, "sleeve_length_mm": length}


def sleeve_length_from_span(spec, cuffs):
    """Recover the cuff reach from both sleeve ends when only one underarm is visible."""
    left, _, left_vector = cuff_axes(cuffs["image_left"], -1)
    right, _, right_vector = cuff_axes(cuffs["image_right"], 1)
    directions = np.array([left_vector, right_vector])
    if not np.all((0 <= directions[:, 1]) & (directions[:, 1] <= 0.65)):
        return None
    span = float(right[0] - left[0])
    # The outer tips lie on the folded sleeve centerlines. Their transverse
    # reach already includes the rotation about each underarm join.
    width = spec.sleeve_bicep_flat_mm
    taper = (spec.sleeve_bicep_flat_mm - spec.sleeve_opening_flat_mm) / 2
    reach = (span - spec.flat_chest_mm - width / 2 * directions[:, 1].sum()) / abs(directions[:, 0]).sum()
    if not MIN_SLEEVE_BELOW_CAP_MM + 2 <= reach <= 0.48 * spec.flat_chest_mm:
        return None
    front, back = (body_panel(spec, side).edge("armhole_right").length() for side in (True, False))
    try:
        front_span, cap = cap_dimensions(width, min(front, back), front, back)
    except ValueError:
        return None
    if cap < 10 or taper >= min(front_span, width - front_span):
        return None
    length = float(cap + reach)
    return length if 0.17 * spec.flat_chest_mm <= length <= 0.60 * spec.flat_chest_mm else None


def paired_cuff_span(measures, views):
    """Change sleeve length only when the two flat photographs corroborate the span."""
    field = "sleeve_length_mm"
    spec = TeeSpec(**measures)
    readings = {}
    for view, evidence in views.items():
        cuffs = evidence.get("cuff_span_landmarks", {}).get("cuffs")
        length = sleeve_length_from_span(spec, cuffs) if cuffs else None
        readings[view] = round(length, 3) if length is not None else None
    report = {"status": "kept_prior", "regression_before_mm": measures[field], "length_by_view_mm": readings}
    if readings.get("front") is None or readings.get("back") is None:
        return {}, {**report, "reason": "Both views need valid bilateral cuff geometry"}
    short, long = sorted((readings["front"], readings["back"]))
    if long - short > max(10, 0.06 * short):
        return {}, {**report, "reason": "The two views disagree on sleeve length"}
    length = round((short + long) / 2, 3)
    updated = TeeSpec(**{**measures, field: length})
    try:
        updated.check()
        sleeve_panels(updated, body_panel(updated, True), body_panel(updated, False))
    except ValueError as error:
        return {}, {**report, "reason": str(error)}
    return {field: length}, {**report, "status": "applied", "sources": {field: "paired_cuff_span"}}


def draft_constraints(measures, evidence):
    """Constrain observed lengths while retaining priors for hidden seam geometry."""
    updates = {
        "flat_chest_mm": evidence["lower_torso_width_mm"],
        "back_length_mm": evidence["shoulder_level_to_hem_mm"] - measures["back_neck_depth_mm"],
    }
    methods = {"flat_chest_mm": "scan_silhouette", "back_length_mm": "scan_with_prior"}
    landmarks = evidence["flat_landmarks"]
    neckline = landmarks.get("visible_neck")
    if neckline and neckline["confirmed_by_inner_notch"]:
        # The inner opening is visible in silhouette. Approximate the width of
        # the attachment seam by adding the nominal folded rib at both sides.
        width = neckline["opening_width_mm"] + 2 * TeeSpec.collar_finished_width_mm
        if width + 30 < measures["garment_shoulder_width_mm"]:
            updates["neck_width_mm"] = width
            methods["neck_width_mm"] = "scan_with_prior"

    if "cuffs" in landmarks:
        cuffs = landmarks["cuffs"]
        opening = sum(cuff["folded_opening_mm"] for cuff in cuffs.values())
        top = evidence["shoulder_line_y_mm"]
        if "sleeve_layout" in landmarks:
            geometry = projected_sleeve(measures, updates, opening, landmarks["sleeve_layout"], top)
            if geometry is not None:
                updates.update(geometry)
                methods.update(dict.fromkeys(geometry, "scan_projected_joins"))
                methods["sleeve_opening_flat_mm"] = "scan_with_prior"
                return updates, methods, None

        taper = max(0, measures["sleeve_bicep_flat_mm"] - measures["sleeve_opening_flat_mm"])
        bicep = opening + taper
        shoulder_slope = measures["shoulder_slope_mm"]
        armhole = max(measures["armhole_depth_mm"], shoulder_slope + bicep / 2 + 6)
        proposed = {"sleeve_opening_flat_mm": opening, "sleeve_bicep_flat_mm": bicep, "armhole_depth_mm": armhole}
        center = landmarks["torso_center_x_mm"]
        lengths = []
        for side, tag in ((-1, "image_left"), (1, "image_right")):
            cuff = cuffs[tag]
            outer, _, direction = cuff_axes(cuff, side)
            shoulder = np.array([center + side * measures["garment_shoulder_width_mm"] / 2, top + shoulder_slope])
            lengths.append(float((outer - shoulder) @ direction))
        short, long = sorted(lengths)
        chest = updates["flat_chest_mm"]
        if 0.17 * chest <= short and long <= min(1.19 * short, 0.60 * chest):
            proposed["sleeve_length_mm"] = sum(lengths) / 2

        underarms = [cuff["underarm_join_y_mm"] - top for cuff in cuffs.values() if cuff["underarm_join_y_mm"] is not None]
        if underarms and armhole > min(underarms):
            return updates, methods, "Required armhole would extend past the visible sleeve-to-body join"
        span_length = None
        if len(underarms) == 1 and "sleeve_length_mm" not in proposed:
            span_length = sleeve_length_from_span(TeeSpec(**{**measures, **updates, **proposed}), cuffs)
            if span_length is not None:
                proposed["sleeve_length_mm"] = span_length
        spec = TeeSpec(**{**measures, **updates, **proposed})
        try:
            spec.check()
            sleeve_panels(spec, body_panel(spec, True), body_panel(spec, False))
        except ValueError as error:
            return updates, methods, str(error)
        updates.update(proposed)
        methods.update(dict.fromkeys(proposed, "scan_with_prior"))
        if span_length is not None:
            methods["sleeve_length_mm"] = "scan_cuff_span"
    return updates, methods, None


def compatible_scan_sleeve_length(measures, updates):
    """Extend a regressed sleeve only when its cap cannot meet the drafted armholes."""
    if "sleeve_length_mm" in updates:
        return None
    spec = TeeSpec(**{**measures, **updates})
    try:
        spec.check()
        front, back = (body_panel(spec, side).edge("armhole_right").length() for side in (True, False))
        available = spec.sleeve_length_mm - MIN_SLEEVE_BELOW_CAP_MM
        if available >= min(front, back):
            return None
        # A proportional split is enough to establish feasibility without
        # solving the entire cap if each curve could already span its armhole.
        if available > 0 and all(
            half_cap("cap", spec.sleeve_bicep_flat_mm * length / (front + back), available).length() > length
            for length in (front, back)
        ):
            return None
        _, height = cap_dimensions(spec.sleeve_bicep_flat_mm, min(front, back), front, back)
    except ValueError:
        return None
    minimum = height + MIN_SLEEVE_BELOW_CAP_MM
    if spec.sleeve_length_mm <= minimum:
        return minimum + 1
    return None
