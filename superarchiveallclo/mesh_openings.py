"""Trace the actual free boundaries of a triangulated cloth mid-surface."""

from collections import Counter, defaultdict

import numpy as np


def boundary_loops(triangles):
    edges = Counter()
    for face in triangles:
        if len(set(face)) != 3:
            raise ValueError("Degenerate triangle in cloth surface")
        a, b, c = face
        edges.update(tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
    if max(edges.values(), default=0) > 2:
        raise ValueError("Cloth surface has an edge shared by more than two triangles")
    neighbors = defaultdict(list)
    for (a, b), count in edges.items():
        if count == 1:
            neighbors[a].append(b)
            neighbors[b].append(a)
    if any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("Free garment edges do not form disjoint closed loops")

    unseen = set(neighbors)
    loops = []
    while unseen:
        start = min(unseen)
        loop = [start]
        previous, point = start, min(neighbors[start])
        while point != start:
            loop.append(point)
            next_point = next(adjacent for adjacent in neighbors[point] if adjacent != previous)
            previous, point = point, next_point
        unseen.difference_update(loop)
        loops.append(loop)
    return loops


def loop_edges(vertices, loop):
    start = vertices[loop]
    end = np.roll(start, -1, axis=0)
    return start, end, np.linalg.norm(end - start, axis=1)


def chord_length(start, end, lengths, samples=64):
    path = np.r_[0, np.cumsum(lengths)]
    targets = np.arange(samples) * path[-1] / samples
    segments = np.searchsorted(path[1:], targets, side="right")
    fractions = (targets - path[segments]) / lengths[segments]
    sampled = start[segments] + fractions[:, None] * (end[segments] - start[segments])
    return float(np.linalg.norm(sampled - np.roll(sampled, -1, axis=0), axis=1).sum())


def sewn_opening_perimeters(vertices, loops, welded_faces, faces, panels):
    panel_labels = np.empty(sum(panel["vertex_count"] for panel in panels.values()), dtype=np.int8)
    names = ("hem", "neck", "sleeve_left", "sleeve_right")
    codes = dict(front=0, back=0, collar=1, sleeve_left=2, sleeve_right=3)
    for name, panel in panels.items():
        start = panel["vertex_start"]
        panel_labels[start:start + panel["vertex_count"]] = codes[name]
    welded = np.asarray(welded_faces).ravel()
    labels = panel_labels[np.asarray(faces).ravel()]
    perimeters = {}
    for loop in loops:
        label = np.unique(labels[np.isin(welded, loop)])
        if len(label) != 1 or names[label[0]] in perimeters:
            raise ValueError("Each free edge must belong to one hem, collar or cuff")
        start, end, lengths = loop_edges(vertices, loop)
        perimeters[names[label[0]]] = {
            "perimeter": round(float(lengths.sum() * 1000), 3),
            "perimeter_64_chords": round(chord_length(start, end, lengths) * 1000, 3),
        }
    return perimeters
