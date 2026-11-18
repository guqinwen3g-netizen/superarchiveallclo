"""Generate a labelled calibration fixture. The resulting images are synthetic."""

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw

from superarchiveallclo.tee import TeeSpec, draft

parser = ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True)

spec = TeeSpec()
pattern = draft(spec)
pixels_per_mm = 1.3


def pixel(x, y):
    return [501 + x * pixels_per_mm, 106 + y * pixels_per_mm]


views = {}
for panel in pattern["panels"]:
    view = panel["id"]
    if view not in ("front", "back"):
        continue
    image = Image.new("RGB", (1024, 1090), "#eee9db")
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(pixel(x, y)) for x, y in panel["cut_outline"]], fill="#535750")
    draw.text((40, 38), f"SYNTHETIC CALIBRATION FIXTURE / {view.upper()}", fill="#202421", font_size=23)
    ruler = [[70, 1030], [70 + 200 * pixels_per_mm, 1030]]
    draw.line([tuple(p) for p in ruler], fill="#202421", width=3)
    for x, y in ruler:
        draw.line((x, y - 8, x, y + 8), fill="#202421", width=3)
    draw.text((70, 1049), "200 mm", fill="#202421", font_size=16)
    image.save(args.output / f"{view}.png")
    views[view] = {"file": f"{view}.png", "scale": {"mm": 200, "points": ruler}}


def observation(view, start, end, axis):
    return {"view": view, "points": [pixel(*start), pixel(*end)], "axis": axis}


chest_edge = spec.flat_chest_mm / 2
shoulder_edge = spec.shoulder_width_mm / 2
hem = spec.back_length_mm + spec.back_neck_depth_mm
measurements = {
    "flat_chest_mm": observation("front", (-chest_edge, spec.armhole_depth_mm),
                                (chest_edge, spec.armhole_depth_mm), "x"),
    "back_length_mm": observation("back", (0, spec.back_neck_depth_mm), (0, hem), "y"),
    "garment_shoulder_width_mm": observation("front", (-shoulder_edge, spec.shoulder_slope_mm),
                                           (shoulder_edge, spec.shoulder_slope_mm), "x"),
    "front_neck_depth_mm": observation("front", (0, 0), (0, spec.front_neck_depth_mm), "y"),
}
for name in (
    "shoulder_slope_mm", "armhole_depth_mm", "sleeve_length_mm", "sleeve_bicep_flat_mm",
    "sleeve_opening_flat_mm", "neck_width_mm", "back_neck_depth_mm",
):
    measurements[name] = {"value": getattr(spec, name)}
session = {
    "schema_version": 1,
    "kind": "synthetic_fixture",
    "garment": "crew_neck_tee",
    "item_id": "synthetic_calibration_tee",
    "views": views,
    "measurements": measurements,
}
(args.output / "session.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf8")
