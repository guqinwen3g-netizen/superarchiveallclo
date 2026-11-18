"""Relax a folded garment in its support plane while preserving seam topology."""

import numpy as np

POSE_WEIGHT = 0.0015


def projection_area(points, triangles):
    corners = points[triangles]
    a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def relax_flat_metric(vertices, edges, lengths, pins, faces, edge_weights=None):
    """Alternate edge projections with a global, softly pose-anchored solve."""
    a, b = edges.T
    n = len(vertices)
    pins = np.asarray(pins, dtype=int)
    triangles = np.asarray(faces, dtype=int)
    before = vertices[:, :2].copy()
    expected = projection_area(before, triangles)
    protected = np.abs(expected) > 1e-7
    weights = np.ones(len(edges)) if edge_weights is None else edge_weights
    degree = np.bincount(np.r_[a, b], weights=np.r_[weights, weights], minlength=n)
    pose = POSE_WEIGHT * degree
    diagonal = degree + pose
    height_delta = vertices[b, 2] - vertices[a, 2]
    planar_lengths = np.sqrt(np.maximum(lengths ** 2 - height_delta ** 2, 0))

    def gather(values, indices):
        return np.column_stack([np.bincount(indices, weights=values[:, axis], minlength=n) for axis in range(2)])

    def matrix(values):
        left = gather(weights[:, None] * values[b], a)
        right = gather(weights[:, None] * values[a], b)
        return diagonal[:, None] * values - left - right

    fixed_term = pose[:, None] * before - matrix(before)
    displacement = np.zeros_like(before)
    for _ in range(80):
        previous = displacement.copy()
        xy = before + previous
        along = xy[b] - xy[a]
        norm = np.linalg.norm(along, axis=1)
        wanted = along * (weights * planar_lengths / np.maximum(norm, 1e-9))[:, None]
        rhs = gather(wanted, b) - gather(wanted, a) + fixed_term
        residual = rhs - matrix(displacement)
        residual[pins] = 0
        direction = residual / diagonal[:, None]
        dot = np.sum(residual * direction)
        # Solve both in-plane coordinates on the sewn graph. Fixed positions
        # have zero displacement, so their residual and search direction stay zero.
        for _ in range(45):
            if dot < 1e-15:
                break
            response = matrix(direction)
            response[pins] = 0
            step = dot / np.sum(direction * response)
            displacement += step * direction
            residual -= step * response
            preconditioned = residual / diagonal[:, None]
            next_dot = np.sum(residual * preconditioned)
            direction = preconditioned + (next_dot / dot) * direction
            dot = next_dot

        # Preserve the visible orientation of the supported sheet.
        for _ in range(6):
            area = projection_area(before + displacement, triangles)
            if not np.any((area * expected < 0) & protected):
                break
            displacement = (previous + displacement) / 2
        else:
            displacement = previous
            break
        if np.max(np.abs(displacement - previous)) < 1e-5:
            break
    vertices[:, :2] = before + displacement
