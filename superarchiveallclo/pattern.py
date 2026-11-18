"""Two-dimensional sewing pattern geometry. Coordinates are millimetres, y down."""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import dist, hypot
from typing import Iterable

Point = tuple[float, float]
EPSILON = 1e-7


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def subtract(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def scale(p: Point, factor: float) -> Point:
    return p[0] * factor, p[1] * factor


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def intersection(a: Point, direction: Point, b: Point, other_direction: Point) -> Point:
    offset = cross(subtract(b, a), other_direction) / cross(direction, other_direction)
    return add(a, scale(direction, offset))


def outward(vector: Point) -> Point:
    length = hypot(*vector)
    return vector[1] / length, -vector[0] / length


@dataclass(frozen=True)
class Edge:
    id: str
    start: Point
    end: Point
    allowance_mm: float = 10
    role: str = "seam"
    controls: tuple[Point, Point] | None = None

    def samples(self) -> tuple[Point, ...]:
        if self.controls is None:
            return self.start, self.end
        start, control_a, control_b, end = self.start, *self.controls, self.end
        return tuple(
            tuple(
                (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b
                + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                for a, b, c, d in zip(start, control_a, control_b, end)
            )
            for t in (i / 40 for i in range(41))
        )

    def length(self) -> float:
        points = self.samples()
        return sum(dist(a, b) for a, b in zip(points, points[1:]))

    def point_at(self, fraction: float) -> tuple[Point, Point]:
        points = self.samples()
        remaining = self.length() * fraction
        for a, b in zip(points, points[1:]):
            length = dist(a, b)
            if remaining <= length + EPSILON:
                tangent = subtract(b, a)
                return add(a, scale(tangent, remaining / length)), outward(tangent)
            remaining -= length
        raise ValueError(f"Invalid notch position on {self.id}")

    def mirrored(self) -> "Edge":
        def mirror(p: Point) -> Point:
            return -p[0], p[1]

        controls = tuple(map(mirror, self.controls[::-1])) if self.controls else None
        return replace(self, start=mirror(self.end), end=mirror(self.start), controls=controls)


@dataclass(frozen=True)
class Panel:
    id: str
    material: str
    edges: tuple[Edge, ...]
    grainline: tuple[Point, Point]
    foldline: tuple[Point, Point] | None = None

    def edge(self, edge_id: str) -> Edge:
        return next(edge for edge in self.edges if edge.id == edge_id)

    def outline(self) -> tuple[Point, ...]:
        return tuple(point for edge in self.edges for point in edge.samples()[:-1])


@dataclass(frozen=True)
class Seam:
    id: str
    a: tuple[str, str]
    b: tuple[str, str]
    reversed: bool = False
    b_to_a_ratio: float = 1.0
    notch_fractions: tuple[float, ...] = (0.5,)


def closed_segments(points: tuple[Point, ...]) -> Iterable[tuple[Point, Point]]:
    return zip(points, (*points[1:], points[0]))


def validate_outline(points: tuple[Point, ...], label: str) -> None:
    segments = tuple(closed_segments(points))
    if sum(cross(a, b) for a, b in segments) <= EPSILON:
        raise ValueError(f"{label}: expected a clockwise, positive-area outline")
    for i, (a, b) in enumerate(segments):
        if dist(a, b) < EPSILON:
            raise ValueError(f"{label}: zero-length border")
        for j in range(i + 2, len(segments)):
            if i == 0 and j == len(segments) - 1:
                continue
            c, d = segments[j]
            if any(
                max(a[k], b[k]) + EPSILON < min(c[k], d[k])
                or max(c[k], d[k]) + EPSILON < min(a[k], b[k])
                for k in (0, 1)
            ):
                continue
            ab, cd = subtract(b, a), subtract(d, c)
            values = (
                cross(ab, subtract(c, a)), cross(ab, subtract(d, a)),
                cross(cd, subtract(a, c)), cross(cd, subtract(b, c)),
            )
            if (
                min(values[:2]) <= EPSILON and max(values[:2]) >= -EPSILON
                and min(values[2:]) <= EPSILON and max(values[2:]) >= -EPSILON
            ):
                raise ValueError(f"{label}: self-intersecting outline")


def cutting_outline(panel: Panel) -> tuple[Point, ...]:
    points = panel.outline()
    segments = tuple(edge for edge in panel.edges for _ in edge.samples()[:-1])
    result = []
    for i, point in enumerate(points):
        incoming = subtract(point, points[i - 1])
        outgoing = subtract(points[(i + 1) % len(points)], point)
        previous, following = segments[i - 1], segments[i]
        before = add(point, scale(outward(incoming), previous.allowance_mm))
        after = add(point, scale(outward(outgoing), following.allowance_mm))
        starts_hem = following.role == "hem" and previous.role != "hem"
        ends_hem = previous.role == "hem" and following.role != "hem"
        if starts_hem or ends_hem:
            hem, side = (outgoing, incoming) if starts_hem else (incoming, outgoing)
            side_offset, hem_offset = (before, after) if starts_hem else (after, before)
            normal = outward(hem)
            projection = side[0] * normal[0] + side[1] * normal[1]
            reflected = subtract(side, scale(normal, 2 * projection))
            turn = intersection(side_offset, side, point, hem)
            tip = intersection(turn, reflected, hem_offset, hem)
            result.extend((turn, tip) if starts_hem else (tip, turn))
            continue
        denominator = cross(incoming, outgoing)
        if abs(denominator) > EPSILON:
            join = intersection(before, incoming, after, outgoing)
            if dist(point, join) <= 4 * max(previous.allowance_mm, following.allowance_mm):
                result.append(join)
                continue
        result.append(before)
        if dist(before, after) > EPSILON:
            result.append(after)
    return tuple(result)


def notch_mark(edge: Edge, fraction: float, cut: tuple[Point, ...]) -> tuple[Point, Point]:
    point, normal = edge.point_at(fraction)
    distances = []
    for a, b in closed_segments(cut):
        segment = subtract(b, a)
        denominator = cross(normal, segment)
        if abs(denominator) < EPSILON:
            continue
        delta = subtract(a, point)
        radius = cross(delta, segment) / denominator
        along = cross(delta, normal) / denominator
        if radius > EPSILON and -EPSILON <= along <= 1 + EPSILON:
            distances.append(radius)
    if not distances:
        raise ValueError(f"{edge.id}: notch missed cutting outline")
    radius = min(distances)
    return add(point, scale(normal, radius)), add(point, scale(normal, radius - 3))


def validate_pattern(panels: tuple[Panel, ...], seams: tuple[Seam, ...]) -> tuple[dict, dict]:
    edges = {}
    cuts = {}
    for panel in panels:
        ends = tuple(edge.end for edge in panel.edges)
        starts = tuple(edge.start for edge in (*panel.edges[1:], panel.edges[0]))
        if any(dist(a, b) > EPSILON for a, b in zip(ends, starts)):
            raise ValueError(f"{panel.id}: open sewing outline")
        validate_outline(panel.outline(), panel.id)
        cuts[panel.id] = cutting_outline(panel)
        validate_outline(cuts[panel.id], f"{panel.id} cut line")
        for edge in panel.edges:
            key = panel.id, edge.id
            if key in edges:
                raise ValueError(f"Duplicate edge {key}")
            edges[key] = edge

    coverage = Counter(ref for seam in seams for ref in (seam.a, seam.b))
    expected = Counter(key for key, edge in edges.items() if edge.role == "seam")
    if coverage != expected:
        raise ValueError("Seam graph must match every sewable edge exactly once")

    report, notches = {}, defaultdict(list)
    for seam in seams:
        length_a = edges[seam.a].length()
        length_b = edges[seam.b].length()
        residual = length_b - length_a * seam.b_to_a_ratio
        if abs(residual) > 2:
            raise ValueError(f"{seam.id}: seam length residual {residual:.2f} mm exceeds 2 mm")
        report[seam.id] = dict(
            length_a_mm=round(length_a, 3), length_b_mm=round(length_b, 3),
            residual_mm=round(residual, 3),
        )
        for fraction in seam.notch_fractions:
            fraction_b = 1 - fraction if seam.reversed else fraction
            for ref, position in ((seam.a, fraction), (seam.b, fraction_b)):
                notches[ref[0]].append(notch_mark(edges[ref], position, cuts[ref[0]]))
    return {
        panel.id: {"cut_outline": cuts[panel.id], "notches": notches[panel.id]}
        for panel in panels
    }, report
