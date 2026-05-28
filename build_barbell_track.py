# build_barbell_track.py

import os
import json

LEFT_WRIST = 9
RIGHT_WRIST = 10


def select_lifter_idx(keypoints, frame_w, frame_h):
    """Pick the person whose average keypoint position is closest to the frame center."""
    cx, cy = frame_w / 2, frame_h / 2
    best_idx = 0
    best_dist = float("inf")

    for i, person_kp in enumerate(keypoints):
        pts = [p for p in person_kp if p[0] > 0 or p[1] > 0]
        if not pts:
            continue
        px = sum(p[0] for p in pts) / len(pts)
        py = sum(p[1] for p in pts) / len(pts)
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def compute_barbell_center(keypoints, frame_size=None):
    if not keypoints:
        return None

    if frame_size and len(keypoints) > 1:
        idx = select_lifter_idx(keypoints, *frame_size)
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

    for frame_file in frames:
        with open(os.path.join(keypoints_dir, frame_file), "r") as f:
            data = json.load(f)

        frame_size = data.get("frame_size")
        center = compute_barbell_center(data["keypoints"], frame_size=frame_size)
        track.append({
            "frame": frame_file,
            "center": center
        })

    with open(output_path, "w") as f:
        json.dump(track, f, indent=2)

    print(f"[INFO] Saved barbell track to {output_path}")
