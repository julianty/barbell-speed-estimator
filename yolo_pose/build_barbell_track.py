# build_barbell_track.py

import os
import json

LEFT_WRIST  = 9
RIGHT_WRIST = 10


def _select_by_largest_bbox(boxes):
    """Return index of the detection with the greatest bounding box area."""
    best, best_area = 0, -1
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area, best = area, i
    return best


def _select_by_wrist_proximity(keypoints, last_center):
    """Return index of the person whose wrist midpoint is closest to last_center."""
    lx, ly = last_center
    best, best_dist = 0, float("inf")
    for i, kp in enumerate(keypoints):
        if len(kp) <= RIGHT_WRIST:
            continue
        wx = (kp[LEFT_WRIST][0] + kp[RIGHT_WRIST][0]) / 2
        wy = (kp[LEFT_WRIST][1] + kp[RIGHT_WRIST][1]) / 2
        d  = (wx - lx) ** 2 + (wy - ly) ** 2
        if d < best_dist:
            best_dist, best = d, i
    return best


def compute_barbell_center(keypoints, boxes=None, last_center=None):
    if not keypoints:
        return None

    if len(keypoints) == 1:
        idx = 0
    elif last_center is not None:
        idx = _select_by_wrist_proximity(keypoints, last_center)
    elif boxes:
        idx = _select_by_largest_bbox(boxes)
    else:
        idx = 0

    kp = keypoints[idx]
    if len(kp) <= RIGHT_WRIST:
        return None

    lx, ly = kp[LEFT_WRIST]
    rx, ry = kp[RIGHT_WRIST]
    return [(lx + rx) / 2, (ly + ry) / 2]


def build_barbell_track_from_keypoints(keypoints_dir, output_path):
    frames = sorted(f for f in os.listdir(keypoints_dir) if f.endswith(".json"))
    track = []
    last_center = None

    for frame_file in frames:
        with open(os.path.join(keypoints_dir, frame_file), "r") as f:
            data = json.load(f)

        boxes  = data.get("boxes", [])
        center = compute_barbell_center(data["keypoints"], boxes=boxes, last_center=last_center)
        if center is not None:
            last_center = center

        track.append({"frame": frame_file, "center": center})

    with open(output_path, "w") as f:
        json.dump(track, f, indent=2)

    print(f"[INFO] Saved barbell track to {output_path}")
