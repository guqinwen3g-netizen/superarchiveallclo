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
