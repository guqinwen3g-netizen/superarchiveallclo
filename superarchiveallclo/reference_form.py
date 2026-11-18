"""Shared dimensions and settings for the default simulation support."""

from math import cos, isfinite, radians, sin, sqrt

DEFAULT_ARM_ANGLE_DEG = 45.0


def display_neck(record):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Use a version 1 measured display-neck profile")
    pitch, width, source = record.get("pitch_forward_deg"), record.get("lateral_width_mm"), record.get("source")
    if type(pitch) not in (int, float) or not isfinite(pitch) or not -35 <= pitch <= 35:
        raise ValueError("Display neck pitch needs a finite angle in degrees between -35 and 35")
    if type(width) not in (int, float) or not isfinite(width) or not 65 <= width <= 180:
        raise ValueError("Display neck lateral width needs a measured value between 65 and 180 mm")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Identify the source used to measure the display neck")
    return {"schema_version": 1, "pitch_forward_deg": float(pitch), "lateral_width_mm": float(width), "source": source}


def orient_display_neck(point, datum_z, pitch_deg):
    """Apply an observed neck pitch, blending into the unseen generic upper torso."""
    x, y, z = point
    drop = datum_z - z
    influence = max(0, min(1, 1 - drop / 0.22))
    pitch = radians(pitch_deg)
    return (x, y + (cos(pitch) * y + sin(pitch) * drop - y) * influence,
            z + (sin(pitch) * y - cos(pitch) * drop + drop) * influence)


def initial_sleeve_angle(form, angle):
    if angle is None:
        return 20 if form == "display_torso" else 45
    if form != "display_torso" or type(angle) not in (int, float) or not isfinite(angle) or not 0 <= angle <= 45:
        raise ValueError("An initial sleeve angle of 0 to 45 degrees requires the display_torso form")
    return float(angle)


def arm_angle(form, angle):
    if angle is None:
        return DEFAULT_ARM_ANGLE_DEG if form == "torso_with_arms" else None
    if form != "torso_with_arms":
        raise ValueError("An arm angle requires the torso_with_arms reference form")
    if type(angle) not in (float, int) or not isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError("Arm angle from vertical must be between 0 and 90 degrees")
    return float(angle)


def upper_arm(side, chest_scale, length_scale, neck_z, angle):
    """Rotate the unchanged upper-arm ellipsoid about its proximal support point."""
    base_x = side * 0.235 * chest_scale
    base_z = neck_z - 0.148 * length_scale
    extent = (0.041 * chest_scale, 0.047 * chest_scale, 0.15 * length_scale)
    if angle == DEFAULT_ARM_ANGLE_DEG:
        return (base_x, 0, base_z), extent, (side, 0, -1)
    phase = radians(angle)
    direction = (side * sin(phase), 0, -cos(phase))
    pivot_to_center = 0.70 * extent[2]
    diagonal = sqrt(0.5)
    center = (base_x + pivot_to_center * (direction[0] - side * diagonal),
              0, base_z + pivot_to_center * (direction[2] + diagonal))
    return center, extent, direction
