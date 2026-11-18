"""Reproduce the two-support suspension comparison on RGBench green_tshirt."""

from argparse import ArgumentParser
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from examples.evaluate_rgbench_scan import DATASET, REVISION, download
from superarchiveallclo.compare_hang import compare_hang
from superarchiveallclo.scan import in_world, read_pcd

RUN = "green_tshirt/green_tshirt_grasp_2025-07-19-19-14-03"
REST = "1752923646.955944"
SUPPORT = "1752923650.819356"
HISTORY = ("1752923650.578554", "1752923650.698956")
HOLD = (
    "1752923650.939758", "1752923651.060162", "1752923651.180568",
    "1752923651.300973", "1752923651.421379", "1752923651.541785",
)


def wrist_poses(path, side):
    with path.open() as stream:
        records = csv.DictReader(stream)
        poses = np.array([[float(row[key]) for key in ("time", "pos_x", "pos_y", "pos_z", "pos_gripper")]
                          for row in records])
    poses[:, 2] += -0.25 if side == "left" else 0.25
    return poses


def interpolate(poses, time):
    return np.array([np.interp(time, poses[:, 0], poses[:, i]) for i in range(1, 4)])


def grip_calibration(rest, raised, poses):
    closed = poses[1:, 4] < 0.006
    closes = np.flatnonzero(closed & (poses[:-1, 4] >= 0.006))
    if not closes.size:
        raise ValueError("No recorded gripper closure below 6 mm")
    closure = poses[1 + closes[0]]
    if not float(REST) <= closure[0] < float(SUPPORT):
        raise ValueError("The grippers must close between the rest and support calibration frames")
    distance = np.linalg.norm(rest[:, :2] - closure[None, 1:3], axis=1)
    nearest = rest[distance.argmin()]
    rest_patch = rest[np.linalg.norm(rest[:, :2] - nearest[:2], axis=1) < 0.01]
    rest_anchor = np.median(rest_patch, axis=0)
    wrist = interpolate(poses, float(SUPPORT))
    side = raised[np.abs(raised[:, 1] - wrist[1]) < 0.057]
    if len(side) < 50:
        raise ValueError("Not enough measured cloth near the held support")
    upper = side[side[:, 2] >= np.quantile(side[:, 2], 0.97)]
    support = np.median(upper, axis=0)
    travel = [float(np.linalg.norm(interpolate(poses, float(time)) - wrist) * 1000) for time in HOLD]
    return rest_anchor, support, {
        "closure_timestamp": float(closure[0]), "rest_patch_points": len(rest_patch),
        "rest_wrist_to_visible_xy_mm": round(float(np.linalg.norm(closure[1:3] - rest_anchor[:2])) * 1000, 2),
        "support_region_points": len(upper), "wrist_travel_across_test_frames_mm": round(max(travel), 2),
    }


def prepare(directory, *, forecast_swing=False):
    calibration = download(directory, "world-to-camera.json", f"{RUN}/calibration/world_to_camera_transform.json")
    def frame(name, time):
        return download(directory, name, f"{RUN}/segment_pcds/pointcloud_{time}_segmented.pcd")
    rest_file = frame("rest.pcd", REST)
    support_file = frame("support-calibration.pcd", SUPPORT)
    refs = [frame(f"hold-{i:02d}.pcd", time) for i, time in enumerate(HOLD, 1)]
    extrinsic = calibration.read_bytes()
    rest, raised = (in_world(read_pcd(file.read_bytes()), extrinsic) for file in (rest_file, support_file))
    observations, supports, calibrated = [], [], {}
    for side in ("left", "right"):
        joints = download(directory, f"{side}-arm.csv", f"{RUN}/joints/{side}_arm_joint_states_and_end_pose.csv")
        flat, held, evidence = grip_calibration(rest, raised, wrist_poses(joints, side))
        observations.append(flat.tolist())
        supports.append(held.tolist())
        calibrated[side] = {**evidence, "trajectory_sha256": sha256(joints.read_bytes()).hexdigest()}
    manifest = directory / "suspension.json"
    record = {
        "schema_version": 1, "item_id": "rgbench-green-tshirt",
        "source": f"{DATASET} at {REVISION}; {RUN}; CC BY 4.0",
        "world_to_camera": calibration.name, "rest_scan": rest_file.name, "support_calibration_scan": support_file.name,
        "rest_contacts_world_m": observations, "supports_world_m": supports,
        "reference_scans": [ref.name for ref in refs], "reference_timestamps": HOLD,
        "rest_timestamp": REST, "support_timestamp": SUPPORT,
        "supports_source": "Visible cloth near each wrist at the top of one designated calibration scan",
        "grip_calibration": calibrated,
    }
    if forecast_swing:
        record["swing_history"] = [
            {"file": frame(f"swing-{i:02d}.pcd", time).name, "timestamp": time}
            for i, time in enumerate(HISTORY, 1)
        ]
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Archive reconstructed from the RGBench flat scan")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--forecast-swing", action="store_true", help="Forecast short motion from earlier scans")
    args = parser.parse_args()
    if (args.output / "suspension").exists():
        parser.error("Choose a fresh output directory for a new simulation")
    data = args.output / "data"
    data.mkdir(parents=True, exist_ok=True)
    source = prepare(data, forecast_swing=args.forecast_swing)
    report = compare_hang(args.archive, source, args.output / "suspension", args.blender)
    print(json.dumps({"support_metrics": report["support_metrics"], "reference_scans": report["reference_scans"]}, indent=2))


if __name__ == "__main__":
    main()
