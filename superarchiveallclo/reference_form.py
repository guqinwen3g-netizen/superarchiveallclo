"""Shared dimensions and settings for the default simulation support."""

from math import cos, isfinite, radians, sin, sqrt

DEFAULT_ARM_ANGLE_DEG = 45.0


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
