"""Draft a drop-shoulder crew-neck tee from garment and reference body measures."""

from dataclasses import asdict, dataclass, fields, replace
from math import isfinite

from .pattern import Edge, Panel, Seam, validate_pattern

MIN_SLEEVE_BELOW_CAP_MM = 25


@dataclass(frozen=True)
class TeeSpec:
    body_chest_mm: float | None = 960
    body_shoulder_width_mm: float | None = 440
    flat_chest_mm: float = 560
    back_length_mm: float = 680
    shoulder_extension_mm: float | None = 27.5
    shoulder_slope_mm: float = 37
    armhole_depth_mm: float = 230
    sleeve_length_mm: float = 215
    sleeve_bicep_flat_mm: float = 350
    sleeve_opening_flat_mm: float = 310
    neck_width_mm: float = 178
    front_neck_depth_mm: float = 91
    back_neck_depth_mm: float = 24
    collar_finished_width_mm: float = 18
    collar_length_ratio: float = 0.85
    garment_shoulder_width_mm: float | None = None

    def check(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None and field.name in {
                "body_chest_mm", "body_shoulder_width_mm", "shoulder_extension_mm",
                "garment_shoulder_width_mm",
            }:
                continue
            if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 3000:
                raise ValueError(f"{field.name} must be a finite, positive number no greater than 3000")
        if not 0.65 <= self.collar_length_ratio <= 1:
            raise ValueError("collar_length_ratio must be between 0.65 and 1")
        if not self.back_neck_depth_mm < self.front_neck_depth_mm < self.armhole_depth_mm:
            raise ValueError("Neck depths must increase from back to front and fall above the underarm")
        if not self.neck_width_mm + 30 < self.shoulder_width_mm < self.flat_chest_mm:
            raise ValueError("Shoulder width must leave room between neckline and chest")
        if self.armhole_depth_mm <= self.shoulder_slope_mm + 40:
            raise ValueError("Armhole depth must leave room below the shoulder")
        if self.back_length_mm + self.back_neck_depth_mm < self.armhole_depth_mm + 50:
            raise ValueError("Body length must leave room for the hem below the underarm")
        if not 0 < self.sleeve_opening_flat_mm <= self.sleeve_bicep_flat_mm:
            raise ValueError("Sleeve opening cannot exceed the bicep width")

    @property
    def shoulder_width_mm(self) -> float:
        if self.garment_shoulder_width_mm is not None:
            return self.garment_shoulder_width_mm
        if self.body_shoulder_width_mm is None or self.shoulder_extension_mm is None:
            raise ValueError("Provide garment shoulder width or both body shoulder measurements")
        return self.body_shoulder_width_mm + 2 * self.shoulder_extension_mm


def body_panel(spec: TeeSpec, front: bool) -> Panel:
    width = spec.flat_chest_mm / 2
    neck = spec.neck_width_mm / 2
    shoulder = spec.shoulder_width_mm / 2
    drop = spec.shoulder_slope_mm
    depth = spec.armhole_depth_mm
    hem = spec.back_length_mm + spec.back_neck_depth_mm
    neck_depth = spec.front_neck_depth_mm if front else spec.back_neck_depth_mm
    arm_height = depth - drop

    neck_edge = Edge("neck_right", (0, neck_depth), (neck, 0), 8, controls=(
        (neck * 0.61, neck_depth), (neck * 0.93, neck_depth * 0.53),
    ))
    shoulder_edge = Edge("shoulder_right", (neck, 0), (shoulder, drop))
    arm_edge = Edge("armhole_right", (shoulder, drop), (width, depth), controls=(
        (shoulder - (12 if front else 3), drop + arm_height * 0.32),
        (shoulder - (25 if front else 13), depth - arm_height * 0.06),
    ))
    side = Edge("side_right", (width, depth), (width, hem))
    right = neck_edge, shoulder_edge, arm_edge, side
    left = tuple(replace(edge.mirrored(), id=edge.id.replace("_right", "_left")) for edge in reversed(right))
    return Panel("front" if front else "back", "jersey", (
        *right, Edge("hem", (width, hem), (-width, hem), 25, "hem"), *left,
    ), ((0, depth + 30), (0, hem - 35)))


def half_cap(name: str, span: float, height: float) -> Edge:
    return Edge(name, (0, 0), (span, height), controls=(
        (span * 0.58, 0), (span * 0.58, height),
    ))


def cap_dimensions(width: float, max_height: float, front: float, back: float) -> tuple[float, float]:
    """Solve shared cap height and front span while meeting both armhole lengths."""
    def span_at(height: float, length: float) -> float:
        low, high = 0.0, length
        for _ in range(22):
            middle = (low + high) / 2
            if half_cap("cap", middle, height).length() < length:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    low, high = 0.0, min(front, back, max_height)
    if width >= front + back or span_at(high, front) + span_at(high, back) >= width:
        raise ValueError("Sleeve bicep width or sleeve length is incompatible with the armholes")
    for _ in range(22):
        height = (low + high) / 2
        if span_at(height, front) + span_at(height, back) > width:
            low = height
        else:
            high = height
    height = (low + high) / 2
    return span_at(height, front), height


def sleeve_panels(spec: TeeSpec, front: Panel, back: Panel) -> tuple[Panel, Panel]:
    width = spec.sleeve_bicep_flat_mm
    length = spec.sleeve_length_mm
    front_span, height = cap_dimensions(
        width, length - MIN_SLEEVE_BELOW_CAP_MM, front.edge("armhole_right").length(), back.edge("armhole_right").length(),
    )
    back_span = width - front_span
    taper = (width - spec.sleeve_opening_flat_mm) / 2
    if taper >= min(front_span, back_span) or height < 10:
        raise ValueError("Sleeve dimensions leave no usable cap or cuff")
    cap_front = half_cap("cap_front", front_span, height)
    cap_back = half_cap("cap_back", back_span, height).mirrored()
    cuff_front = front_span - taper, length
    cuff_back = -back_span + taper, length
    edges = (
        cap_front, Edge("underarm_front", cap_front.end, cuff_front),
        Edge("hem", cuff_front, cuff_back, 20, "hem"),
        Edge("underarm_back", cuff_back, cap_back.start), cap_back,
    )
    grain = ((0, height + 6), (0, length - 6))
    return (
        Panel("sleeve_right", "jersey", edges, grain),
        Panel("sleeve_left", "jersey", tuple(edge.mirrored() for edge in reversed(edges)), grain),
    )


def collar_panel(spec: TeeSpec, front: Panel, back: Panel) -> tuple[Panel, tuple[Seam, ...]]:
    references = (
        (back, "neck_right", False), (front, "neck_right", True),
        (front, "neck_left", True), (back, "neck_left", False),
    )
    height = spec.collar_finished_width_mm
    start = (0.0, 0.0)
    edges, seams = [], []
    for panel, neck_id, reverse in references:
        edge_id = f"attach_{panel.id}_{neck_id}"
        end = (start[0] + panel.edge(neck_id).length() * spec.collar_length_ratio, 0)
        edges.append(Edge(edge_id, start, end, 8))
        seams.append(Seam(
            edge_id, (panel.id, neck_id), ("collar", edge_id), reverse, spec.collar_length_ratio,
        ))
        start = end

    width = start[0]
    edges.extend((
        Edge("join_right", (width, 0), (width, 2 * height), 8),
        Edge("folded_raw_edge", (width, 2 * height), (0, 2 * height), 8, "fold_overlay"),
        Edge("join_left", (0, 2 * height), (0, 0), 8),
    ))
    seams.append(Seam("collar_join", ("collar", "join_right"), ("collar", "join_left"), True, notch_fractions=()))
    return Panel("collar", "rib", tuple(edges), (
        (width / 2, height / 4), (width / 2, height * 1.75),
    ), ((0, height), (width, height))), tuple(seams)


def draft(spec: TeeSpec) -> dict:
    spec.check()
    front = body_panel(spec, True)
    back = body_panel(spec, False)
    sleeves = sleeve_panels(spec, front, back)
    collar, neck_seams = collar_panel(spec, front, back)
    panels = front, back, *sleeves, collar
    seams = list(neck_seams)
    for side in ("right", "left"):
        sleeve = f"sleeve_{side}"
        for edge in ("shoulder", "side"):
            name = f"{edge}_{side}"
            seams.append(Seam(name, ("front", name), ("back", name), notch_fractions=()))
        seams.extend((
            Seam(f"front_armhole_{side}", ("front", f"armhole_{side}"), (sleeve, "cap_front")),
            Seam(f"back_armhole_{side}", ("back", f"armhole_{side}"), (sleeve, "cap_back"),
                 reversed=True, notch_fractions=(0.49, 0.51)),
            Seam(f"sleeve_underarm_{side}", (sleeve, "underarm_front"), (sleeve, "underarm_back"),
                 reversed=True, notch_fractions=()),
        ))
    outlines, measurements = validate_pattern(panels, tuple(seams))
    return {
        "schema_version": 1,
        "style": "Drop-shoulder crew-neck tee",
        "status": "prototype",
        "units": "mm",
        "coordinate_system": "x right, y down; edge IDs use garment left/right",
        "parameters": asdict(spec),
        "materials": {"jersey": "Reference knit assumption", "rib": "Reference neck rib assumption"},
        "measurements": {
            "flat_chest_mm": spec.flat_chest_mm,
            "garment_chest_mm": 2 * spec.flat_chest_mm,
            "chest_ease_mm": (
                2 * spec.flat_chest_mm - spec.body_chest_mm if spec.body_chest_mm is not None else None
            ),
            "across_shoulders_mm": spec.shoulder_width_mm,
            "back_center_length_mm": spec.back_length_mm,
            "sleeve_length_apex_to_hem_mm": spec.sleeve_length_mm,
            "neck_seam_circumference_mm": round(sum(
                panel.edge(edge).length()
                for panel in (front, back) for edge in ("neck_left", "neck_right")
            ), 3),
        },
        "assembly": {"collar": "Join short ends, fold along foldline and sew both raw edges to the neckline together."},
        "panels": [
            {**asdict(panel), "cut_quantity": 1, **outlines[panel.id]} for panel in panels
        ],
        "seams": [{**asdict(seam), **measurements[seam.id]} for seam in seams],
    }
