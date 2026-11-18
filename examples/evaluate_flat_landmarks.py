"""Measure flat-landmark accuracy on independent synthetic garment silhouettes."""

from argparse import ArgumentParser
from math import cos, radians, sin
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from superarchiveallclo.flat_measurements import flat_landmarks
from superarchiveallclo.tee import TeeSpec, body_panel

SEED = 842067
CASES = 160


def fixture(rng):
    width = float(rng.uniform(460, 645))
    spec = TeeSpec(
        flat_chest_mm=width,
        back_length_mm=float(rng.uniform(600, 755)),
        garment_shoulder_width_mm=width - float(rng.uniform(48, 94)),
        shoulder_slope_mm=float(rng.uniform(25, 48)),
        armhole_depth_mm=float(rng.uniform(220, 285)),
        neck_width_mm=float(rng.uniform(135, 211)),
        front_neck_depth_mm=float(rng.uniform(28, 66)),
    )
    cuff = float(rng.uniform(0.22, 0.36)) * width
    sleeve = float(rng.uniform(175, 252))
    mm = float(rng.uniform(1.5, 3))
    tilt = radians(float(rng.uniform(-3, 3)))
    c, s = cos(tilt), sin(tilt)

    def pixel(xy):
        x, y = xy
        return (600 + x * c - y * s) / mm, (110 + x * s + y * c) / mm

    image = Image.new("1", (round(1200 / mm), round(1000 / mm)))
    paint = ImageDraw.Draw(image)
    paint.polygon([pixel(p) for p in body_panel(spec, True).outline()], fill=1)
    for side in (-1, 1):
        pose = radians(float(rng.uniform(12, 34)))
        along, across = cos(pose), sin(pose)
        shoulder = spec.shoulder_width_mm / 2
        slope = spec.shoulder_slope_mm
        x = shoulder + sleeve * along
        y = slope + sleeve * across
        points = (
            (shoulder - 12, slope - 2), (x, y), (x - cuff * across, y + cuff * along),
            (width / 2 - 16, spec.armhole_depth_mm),
        )
        paint.polygon([pixel((side * a, b)) for a, b in points], fill=1)
    mask = np.asarray(image).copy()
    yy, xx = np.where(mask)
    cropped = mask[yy.min(): yy.max() + 1, xx.min(): xx.max() + 1]
    return cropped, mm, width, cuff, spec.neck_width_mm


def stats(errors):
    if not errors:
        return {"cases_measured": 0, "median_absolute_error_mm": None, "p90_absolute_error_mm": None}
    return {"cases_measured": len(errors), "median_absolute_error_mm": round(float(np.median(errors)), 2),
            "p90_absolute_error_mm": round(float(np.quantile(errors, 0.9)), 2)}


def evaluate():
    rng = np.random.default_rng(SEED)
    cuffs, necks = [], []
    for _ in range(CASES):
        mask, mm, width, cuff_mm, neck_mm = fixture(rng)
        landmarks = flat_landmarks(mask, mm, width)
        if "cuffs" in landmarks:
            measured = sum(row["folded_opening_mm"] for row in landmarks["cuffs"].values()) / 2
            cuffs.append(abs(measured - cuff_mm))
        neck = landmarks.get("visible_neck")
        if neck and neck["confirmed_by_inner_notch"]:
            necks.append(abs(neck["opening_width_mm"] - neck_mm))
    return {
        "data": "Analytic folded cuffs and visible openings. Vary sleeve direction, size, rotation and scan resolution.",
        "seed": SEED, "cases": CASES,
        "folded_cuff": stats(cuffs), "visible_neck_opening": stats(necks),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(evaluate(), indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
