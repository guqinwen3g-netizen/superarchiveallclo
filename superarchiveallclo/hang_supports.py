"""Rigid registration and grip patches for a two-point suspension experiment."""

import numpy as np


def support_frame(contacts, up):
    right = contacts[1] - contacts[0]
    span = np.linalg.norm(right)
    if span < 0.04:
        raise ValueError("The suspension requires two separate support points")
    right /= span
    vertical = up - np.dot(up, right) * right
    norm = np.linalg.norm(vertical)
    if norm < 0.2:
        raise ValueError("The support points must define a transverse line")
    vertical /= norm
    return np.column_stack((right, vertical, np.cross(right, vertical)))


def observed_pitch(points, supports, camera):
    """Estimate the central hanging cloth's pitch toward the camera."""
    pivot = supports.mean(axis=0)
    basis = support_frame(supports, np.array([0.0, 0.0, 1.0]))
    toward_camera = 1 if (camera - pivot) @ basis[:, 2] >= 0 else -1
    local = (points - pivot) @ basis
    separation = np.linalg.norm(supports[1] - supports[0])
    torso = local[np.abs(local[:, 0]) < separation / 4]
    if len(torso) < 100:
        raise ValueError("The calibration scan needs central cloth points below the supports")
    down = -torso[:, 1]
    normal = toward_camera * torso[:, 2]
    length = float(np.quantile(down, 0.99))
    if length <= 0:
        raise ValueError("The observed cloth must extend below the supports")
    edges = np.linspace(0.15 * length, 0.90 * length, 6)
    bands = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        chosen = (down >= low) & (down < high)
        if chosen.sum() >= 20:
            bands.append([np.median(down[chosen]), np.median(normal[chosen])])
    if len(bands) < 4:
        raise ValueError("Too few observed cloth height bands to estimate pitch")
    slope, _ = np.polyfit(*np.asarray(bands).T, 1)
    return float(np.arctan(slope)), length, toward_camera


def pendulum_forecast(offsets, angles, times, length_m):
    """Predict a short free swing about a fixed transverse support axis."""
    # A uniform strip pivoted about one edge has equivalent length 2/3 L.
    period = 2 * np.pi * np.sqrt((2 * length_m / 3) / 9.81)
    omega = 2 * np.pi / period
    velocity = np.polyfit(offsets, angles, 1)[0]
    phases = omega * times
    return angles[-1] * np.cos(phases) + velocity / omega * np.sin(phases), float(velocity), float(period)


def rotate_about_support(points, supports, angle):
    pivot = supports.mean(axis=0)
    vector = supports[1] - supports[0]
    axis = vector / np.linalg.norm(vector)
    delta = points - pivot
    c, s = np.cos(angle), np.sin(angle)
    return pivot + c * delta + s * np.cross(axis, delta) + (1 - c) * (delta @ axis)[..., None] * axis


def place_for_gravity(vertices, flat_reference, rest_contacts, supports):
    """Register the rest surface, then suspend at two known absolute positions."""
    proposed_world = np.column_stack((-vertices[:, 1], vertices[:, 0], vertices[:, 2]))
    low, high = np.quantile(flat_reference, [0.001, 0.999], axis=0)
    translation = np.array([
        low[0] - proposed_world[:, 0].min(),
        (low[1] + high[1] - proposed_world[:, 1].min() - proposed_world[:, 1].max()) / 2,
        np.median(flat_reference[:, 2]) - np.median(proposed_world[:, 2]),
    ])
    proposed_world += translation
    patches, gaps = [], []
    for point in rest_contacts:
        distances = np.linalg.norm(proposed_world[:, :2] - point[:2], axis=1)
        nearest = int(distances.argmin())
        gaps.append(float(distances[nearest]))
        patch = np.flatnonzero(np.linalg.norm(proposed_world[:, :2] - proposed_world[nearest, :2], axis=1) <= 0.012)
        patches.append(patch)
    if np.intersect1d(*patches).size or max(gaps) > 0.06:
        raise ValueError("The resting support locations do not resolve to two distinct garment patches")

    start = np.array([vertices[patch].mean(axis=0) for patch in patches])
    source_frame = support_frame(start, np.array([0.0, 1.0, 0.0]))
    target_frame = support_frame(supports, np.array([0.0, 0.0, 1.0]))
    posed = (vertices - start.mean(axis=0)) @ source_frame @ target_frame.T + supports.mean(axis=0)
    for patch, center, support in zip(patches, start, supports, strict=True):
        shift = support - posed[patch].mean(axis=0)
        distance = np.linalg.norm(vertices[:, :2] - center[:2], axis=1)
        feather = np.clip((0.055 - distance) / 0.043, 0, 1)
        feather[patch] = 1
        posed += feather[:, None] * shift
    return posed, patches, {
        "flat_translation_world_m": translation.tolist(),
        "rest_registration_gaps_mm": (np.array(gaps) * 1000).round(2).tolist(),
        "pinned_vertices_per_patch": [len(patch) for patch in patches],
        "natural_patch_centroid_separation_mm": round(float(np.linalg.norm(start[1] - start[0]) * 1000), 2),
        "target_support_separation_mm": round(float(np.linalg.norm(supports[1] - supports[0]) * 1000), 2),
    }
