"""Fit a candidate sleeve length to the observed cuff-to-cuff span in a calibrated flat view."""

from dataclasses import replace
import json
from math import cos, isfinite, radians

import numpy as np

from .compare_flat import run_flat_pose, sleeve_angles_from_cuffs
from .compare_shape import compare_masks, metric_mask, project
from .flat_measurements import compatible_scan_sleeve_length
from .tee import draft
from .vision import picture_silhouette


def rendered_span(pattern, working, angles, blender):
    _, mesh = run_flat_pose(json.dumps(pattern).encode(), working, {}, angles, blender)
    with np.load(mesh, allow_pickle=False) as data:
        return float(1000 * np.ptp(data["vertices"][:, 0]))


def refine(spec, pattern, manifest, working, blender):
    measurement = manifest["measurements"]["sleeve_length_mm"]
    if "observation" in measurement:
        return pattern, {"status": "retained_supplied_measurement"}
    evidence = manifest.get("inference", {}).get("scale_evidence", {}).get("front", {})
    cuffs = evidence.get("flat_landmarks", {}).get("cuffs", {})
    angles = sleeve_angles_from_cuffs(cuffs)
    if angles is None:
        return pattern, {"status": "unavailable", "reason": "Both cuff directions are required on the calibrated front view"}
    target = float(cuffs["image_right"]["outer_mm"][0] - cuffs["image_left"]["outer_mm"][0])
    if not isfinite(target) or target <= spec.flat_chest_mm:
        return pattern, {"status": "unavailable", "reason": "Visible cuff ends must span the body"}
    working.mkdir()
    length = spec.sleeve_length_mm
    span = rendered_span(pattern, working, angles, blender)
    history = [{"length_mm": length, "mesh_span_mm": round(span, 3)}]
    recorded_lengths = {length}
    best = (abs(target - span), length, pattern)
    slope = sum(cos(radians(angle)) for angle in angles)
    tolerance = max(4.0, 2 * float(evidence["processed_mm_per_pixel"]))
    lower, upper = .17 * spec.flat_chest_mm, .60 * spec.flat_chest_mm
    for _ in range(3):
        if abs(target - span) <= tolerance:
            break
        next_length = round(max(lower, min(upper, length + (target - span) / slope)), 3)
        if next_length in recorded_lengths:
            break
        recorded_lengths.add(next_length)
        try:
            candidate = draft(replace(spec, sleeve_length_mm=next_length))
        except ValueError as error:
            history.append({"length_mm": next_length, "draft_error": str(error)})
            break
        next_span = rendered_span(candidate, working, angles, blender)
        history.append({"length_mm": next_length, "mesh_span_mm": round(next_span, 3)})
        error = abs(target - next_span)
        if error < best[0]:
            best = (error, next_length, candidate)
        slope = (next_span - span) / (next_length - length)
        if not 0.25 <= slope <= 4:
            break
        length, span = next_length, next_span
    _, best_length, best_pattern = best
    report = {
        "status": "applied" if best_length != spec.sleeve_length_mm else "retained_initial_candidate",
        "input_cuff_span_mm": round(target, 3),
        "cuff_angles_below_horizontal_deg": dict(zip(("image_left", "image_right"), angles, strict=True)),
        "tolerance_mm": tolerance, "evaluations": history, "selected_sleeve_length_mm": best_length,
        "scope": "Candidate matched to the input's cuff-tip projection at native scale",
    }
    if best_length != spec.sleeve_length_mm:
        manifest["measurements"]["sleeve_length_mm"] = {
            "source": "calibrated_cuff_span_fit", "value": best_length, "before_fit_mm": spec.sleeve_length_mm,
        }
    return best_pattern, report


def refine_armhole(spec, pattern, capture, photo, working, blender):
    """Choose a compatible armhole depth by comparing candidate meshes to the input silhouette."""
    field = "armhole_depth_mm"
    if "observation" in capture["measurements"][field]:
        return pattern, {"status": "retained_supplied_measurement"}
    front = capture["views"]["front"]
    color = front.get("calibration", {}).get("registered_color_seams", {})
    traces, torso = color.get("projected_shoulder_seams"), color.get("torso_profile")
    outline = capture.get("inference", {}).get("scale_evidence", {}).get("front", {})
    angles = sleeve_angles_from_cuffs(outline.get("flat_landmarks", {}).get("cuffs", {}))
    if angles is None or traces is None or torso is None:
        return pattern, {"status": "unavailable", "reason": "This fit uses visible cuff directions and calibrated color seam traces"}

    upper = np.mean([traces[key]["shoulder_crop_px"][1] for key in ("image_left", "image_right")])
    lower = np.mean([point[1] for point in torso["chest_points_crop_px"]])
    inferred = spec.shoulder_slope_mm + (lower - upper) * color["processed_mm_per_pixel"]
    limit = min(max(spec.shoulder_slope_mm + 42, float(inferred)),
                spec.back_length_mm + spec.back_neck_depth_mm - 52)
    candidates = tuple(dict.fromkeys(float(value) for value in np.linspace(spec.armhole_depth_mm, limit, 4).round(3)))
    _, mask, _, step, _ = picture_silhouette(photo, front, outer_only=True)
    observed, _ = metric_mask(mask, step)
    working.mkdir()
    history = []
    best = None
    sleeve = "sleeve_length_mm"
    supplied_sleeve = "observation" in capture["measurements"][sleeve]
    for index, depth in enumerate(candidates):
        selected = replace(spec, armhole_depth_mm=depth)
        if not supplied_sleeve:
            compatible = compatible_scan_sleeve_length(vars(selected), {})
            if compatible is not None:
                selected = replace(selected, sleeve_length_mm=float(compatible))
        trial = {**capture, "measurements": capture["measurements"].copy()}
        try:
            proposed = draft(selected)
        except ValueError as error:
            history.append({"armhole_depth_mm": depth, "draft_error": str(error)})
            continue
        fitted, cuff = refine(selected, proposed, trial, working / f"cuffs-{index}", blender)
        pose = working / f"pose-{index}"
        pose.mkdir()
        _, mesh = run_flat_pose(json.dumps(fitted).encode(), pose, {}, angles, blender)
        with np.load(mesh, allow_pickle=False) as data:
            silhouette, _ = project(data["vertices"], data["triangles"], "top")
            strain = round(float(data["mesh_edge_strain_quantiles_percent"][1]), 3)
        metrics, _ = compare_masks(observed, silhouette)
        row = {"armhole_depth_mm": depth, "sleeve_length_mm": fitted["parameters"][sleeve],
               "mesh_edge_strain_p95_percent": strain, **metrics}
        history.append(row)
        score = (row["intersection_over_union"], -row["contour_distance_mm"]["p90"])
        if best is None or score > best[0]:
            best = (score, fitted, row, trial["measurements"][sleeve], cuff, selected.sleeve_length_mm)
    if best is None:
        return pattern, {"status": "pattern_incompatible", "trials": history}

    _, fitted, row, measure, cuff, seed_length = best
    depth = row["armhole_depth_mm"]
    if depth != spec.armhole_depth_mm:
        capture["measurements"][field] = {"source": "calibrated_armhole_silhouette", "value": depth,
                                          "before_fit_mm": spec.armhole_depth_mm}
    sleeve_length = row["sleeve_length_mm"]
    if not supplied_sleeve and sleeve_length != spec.sleeve_length_mm:
        source = measure["source"] if sleeve_length != seed_length else "geometry_constrained_prior"
        capture["measurements"][sleeve] = {"source": source, "value": sleeve_length,
                                           "before_fit_mm": spec.sleeve_length_mm}
    capture["sleeve_span_fit"] = cuff
    return fitted, {
        "status": "applied" if depth != spec.armhole_depth_mm else "retained_initial_depth",
        "initial_armhole_depth_mm": spec.armhole_depth_mm, "projected_armhole_depth_limit_mm": round(limit, 3),
        "selected_armhole_depth_mm": depth, "trials": history,
        "cuff_angles_below_horizontal_deg": dict(zip(("image_left", "image_right"), angles, strict=True)),
        "selection": "Highest front input silhouette IoU, then shortest contour-distance P90",
    }
