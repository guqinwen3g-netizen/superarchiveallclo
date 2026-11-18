"""Fit a candidate sleeve length to the observed cuff-to-cuff span in a calibrated flat view."""

from dataclasses import replace
import json
from math import cos, isfinite, radians

import numpy as np

from .compare_flat import run_flat_pose, sleeve_angles_from_cuffs
from .tee import draft


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
