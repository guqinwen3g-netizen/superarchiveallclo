"""Millimetre DXF exchange drawing generated from the canonical pattern."""

import ezdxf
from ezdxf import units

from .export import bounds, outlines


def export_dxf(pattern, target):
    drawing = ezdxf.new("R2010", units=units.MM)
    for name, color in (("CUT", 7), ("SEW", 5), ("FOLD", 30), ("MARK", 2), ("LABEL", 7)):
        drawing.layers.new(name, dxfattribs={"color": color})
    space = drawing.modelspace()
    y = 0
    for panel in pattern["panels"]:
        left, top, _, bottom = bounds(panel)
        piece = drawing.blocks.new(panel["id"].upper())
        piece.add_text(
            f"{panel['id'].upper()} / CUT 1 / {panel['material'].upper()}",
            dxfattribs={"insert": (0, 5), "height": 5, "layer": "LABEL"},
        )
        for style, commands in outlines(panel):
            if style == "sew":
                for edge in panel["edges"]:
                    a = edge["start"][0] - left, top - edge["start"][1]
                    b = edge["end"][0] - left, top - edge["end"][1]
                    if edge["controls"]:
                        control = [(x - left, top - p_y) for x, p_y in edge["controls"]]
                        piece.add_open_spline([a, *control, b], degree=3, dxfattribs={"layer": "SEW"})
                    else:
                        piece.add_line(a, b, dxfattribs={"layer": "SEW"})
            else:
                points = [(xy[0] - left, top - xy[1]) for op, xy in commands if op != "Z"]
                piece.add_lwpolyline(points, close=commands[-1][0] == "Z", dxfattribs={"layer": style.upper()})
        space.add_blockref(piece.name, (0, -y))
        y += bottom - top + 20
    drawing.saveas(target)
