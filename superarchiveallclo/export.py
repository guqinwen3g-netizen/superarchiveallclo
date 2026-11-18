"""Export the same pattern geometry to SVG, A4 tiles and tabular reports."""

from csv import DictWriter
from json import dumps
from math import ceil
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas


STYLES = {
    "cut": ("#161716", 0.32, ()),
    "sew": ("#235b91", 0.24, (3, 2)),
    "fold": ("#b06a2a", 0.25, (6, 2, 1, 2)),
    "mark": ("#161716", 0.32, ()),
}
TILE_LEFT, TILE_TOP, TILE_WIDTH, TILE_HEIGHT = 12, 22, 186, 258
TILE_OVERLAP = 12
PIECE_PADDING = 4


def bounds(panel):
    x, y = zip(*panel["cut_outline"])
    return min(x) - PIECE_PADDING, min(y) - PIECE_PADDING, max(x) + PIECE_PADDING, max(y) + PIECE_PADDING


def polyline(points, close=False):
    return [
        ("M" if i == 0 else "L", point) for i, point in enumerate(points)
    ] + ([("Z", ())] if close else [])


def outlines(panel):
    yield "cut", polyline(panel["cut_outline"], close=True)
    edges = panel["edges"]
    sewing = [("M", edges[0]["start"])]
    for edge in edges:
        if edge["controls"]:
            sewing.append(("C", (*edge["controls"][0], *edge["controls"][1], *edge["end"])))
        else:
            sewing.append(("L", edge["end"]))
    yield "sew", [*sewing, ("Z", ())]
    if panel["foldline"]:
        yield "fold", polyline(panel["foldline"])
    for edge in edges:
        if edge["role"] == "hem":
            # The finished hem is also the turn line.
            yield "fold", polyline((edge["start"], edge["end"]))
    for start, end in panel["notches"]:
        yield "mark", polyline((start, end))
    (x, start), (_, end) = panel["grainline"]
    yield "mark", polyline(((x, start), (x, end)))
    for y, direction in ((start, 1), (end, -1)):
        yield "mark", polyline(((x - 2, y + 4 * direction), (x, y), (x + 2, y + 4 * direction)))


def piece_label(panel):
    return f"{panel['id'].replace('_', ' ').upper()}  /  CUT {panel['cut_quantity']} {panel['material'].upper()}"


def export_svg(pattern, target):
    panels = pattern["panels"]
    sizes = [bounds(panel) for panel in panels]
    widest = max(right - left for left, _, right, _ in sizes)
    total_height = sum(bottom - top + 16 for _, top, _, bottom in sizes) + 12
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "width": f"{widest:g}mm", "height": f"{total_height:g}mm",
        "viewBox": f"0 0 {widest:g} {total_height:g}",
        "font-family": "sans-serif", "font-size": "4",
    })
    ET.SubElement(svg, "title").text = pattern["style"]
    ET.SubElement(svg, "text", {"x": "8", "y": "8"}).text = "A4 print: use the tiled PDF at Actual size"
    y = 12
    for panel, (left, top, _, bottom) in zip(panels, sizes):
        ET.SubElement(svg, "text", {"x": "8", "y": str(y + 7)}).text = piece_label(panel)
        group = ET.SubElement(svg, "g", {
            "id": panel["id"], "transform": f"translate({-left:.4f} {y + 12 - top:.4f})", "fill": "none",
        })
        for style, commands in outlines(panel):
            color, width, dash = STYLES[style]
            data = " ".join(f"{op} " + " ".join(f"{v:.4f}" for v in coords) for op, coords in commands)
            attributes = {"d": data, "stroke": color, "stroke-width": str(width), "class": style}
            if dash:
                attributes["stroke-dasharray"] = " ".join(map(str, dash))
            ET.SubElement(group, "path", attributes)
        y += bottom - top + 16
    ET.indent(svg, space="  ")
    ET.ElementTree(svg).write(target, encoding="utf-8", xml_declaration=True)


def pdf_text(canvas, x, y, text, size=10, bold=False):
    canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    canvas.drawString(x * mm, (297 - y) * mm, text)


def pdf_paths(canvas, art, origin_x, origin_y):
    for style, commands in art:
        color, width, dash = STYLES[style]
        canvas.setStrokeColor(color)
        canvas.setLineWidth(width * mm)
        canvas.setDash([interval * mm for interval in dash])
        path = canvas.beginPath()
        for operation, coords in commands:
            points = tuple(
                (coordinate + origin_x) * mm if i % 2 == 0
                else (297 - coordinate - origin_y) * mm
                for i, coordinate in enumerate(coords)
            )
            if operation == "M":
                path.moveTo(*points)
            elif operation == "L":
                path.lineTo(*points)
            elif operation == "C":
                path.curveTo(*points)
            else:
                path.close()
        canvas.drawPath(path)


def grid(panel):
    left, top, right, bottom = bounds(panel)
    cols = max(1, ceil((right - left - TILE_OVERLAP) / (TILE_WIDTH - TILE_OVERLAP)))
    rows = max(1, ceil((bottom - top - TILE_OVERLAP) / (TILE_HEIGHT - TILE_OVERLAP)))
    return left, top, cols, rows


def export_pdf(pattern, target):
    canvas = Canvas(str(target), pagesize=A4, invariant=True)
    canvas.setTitle(pattern["style"])
    pdf_text(canvas, 12, 20, pattern["style"], 18, bold=True)
    pdf_text(canvas, 12, 30, "Prototype sewing pattern  /  A4  /  millimetres")
    pdf_text(canvas, 12, 42, "Print Actual size / 100%. Disable Fit to page and measure the square first.")
    pdf_text(canvas, 12, 48, "Tape tiles of the same piece in row/column order. Match the crosses.")
    pdf_text(canvas, 12, 54, "Solid = cut. Blue dashed = stitch. Long dash/dot = fold. Arrows = grain.")
    pdf_text(canvas, 12, 60, "Cut one of each labelled piece. Use mirrored sleeve templates as printed.")
    pdf_text(canvas, 12, 66, "Join the collar ends. Fold lengthwise and attach both raw edges together.")
    canvas.setLineWidth(0.3 * mm)
    canvas.rect(12 * mm, (297 - 75 - 100) * mm, 100 * mm, 100 * mm)
    pdf_text(canvas, 21, 128, "100 mm x 100 mm", 13)
    pdf_text(canvas, 12, 185, "TILE INDEX", 11, bold=True)
    for i, panel in enumerate(pattern["panels"]):
        _, _, cols, rows = grid(panel)
        pdf_text(canvas, 12, 193 + i * 6, f"{piece_label(panel)}  /  {rows} rows x {cols} columns")
    canvas.showPage()

    page = 2
    for panel in pattern["panels"]:
        left, top, cols, rows = grid(panel)
        art = tuple(outlines(panel))
        for row in range(rows):
            for col in range(cols):
                title = f"{piece_label(panel)}   |   row {row + 1}/{rows}   col {col + 1}/{cols}"
                pdf_text(canvas, 12, 16, title, 10, bold=True)
                pdf_text(canvas, 12, 288, f"A4  /  Actual size  /  12 mm overlap                                 Page {page}", 9)
                canvas.saveState()
                clip = canvas.beginPath()
                clip.rect(TILE_LEFT * mm, (297 - TILE_TOP - TILE_HEIGHT) * mm, TILE_WIDTH * mm, TILE_HEIGHT * mm)
                canvas.clipPath(clip, stroke=0, fill=0)
                offset_x = TILE_LEFT - left - col * (TILE_WIDTH - TILE_OVERLAP)
                offset_y = TILE_TOP - top - row * (TILE_HEIGHT - TILE_OVERLAP)
                pdf_paths(canvas, art, offset_x, offset_y)
                canvas.restoreState()
                canvas.setStrokeColor("#b0b3b5")
                canvas.setLineWidth(0.16 * mm)
                canvas.rect(TILE_LEFT * mm, (297 - TILE_TOP - TILE_HEIGHT) * mm, TILE_WIDTH * mm, TILE_HEIGHT * mm)
                # Images share the same world coordinates at these crosses on adjoining tiles.
                half = TILE_OVERLAP / 2
                for x in (TILE_LEFT + half, TILE_LEFT + TILE_WIDTH - half):
                    for y in (297 - TILE_TOP - half, 297 - TILE_TOP - TILE_HEIGHT + half):
                        canvas.line((x - 2) * mm, y * mm, (x + 2) * mm, y * mm)
                        canvas.line(x * mm, (y - 2) * mm, x * mm, (y + 2) * mm)
                canvas.showPage()
                page += 1
    canvas.save()


def write_exports(pattern, target: Path):
    target.mkdir()
    (target / "pattern.json").write_text(dumps(pattern, indent=2) + "\n", encoding="utf-8")
    export_svg(pattern, target / "pattern.svg")
    export_pdf(pattern, target / "pattern-a4.pdf")
    with (target / "measurements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=("measurement", "value_mm"))
        writer.writeheader()
        for name, value in pattern["measurements"].items():
            writer.writerow({"measurement": name, "value_mm": value})
    with (target / "seams.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=(
            "id", "panel_a", "edge_a", "panel_b", "edge_b", "reversed", "b_to_a_ratio",
            "length_a_mm", "length_b_mm", "residual_mm",
        ))
        writer.writeheader()
        for seam in pattern["seams"]:
            row = {key: seam[key] for key in writer.fieldnames if key in seam}
            row["panel_a"], row["edge_a"] = seam["a"]
            row["panel_b"], row["edge_b"] = seam["b"]
            writer.writerow(row)
