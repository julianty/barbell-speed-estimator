# build_barbell_track.py

import os
import json

LEFT_WRIST = 9
RIGHT_WRIST = 10

def compute_barbell_center(keypoints):
    if not keypoints or len(keypoints[0]) <= RIGHT_WRIST:
        return None

    kp = keypoints[0]
    lx, ly = kp[LEFT_WRIST]
    rx, ry = kp[RIGHT_WRIST]

    return [(lx + rx) / 2, (ly + ry) / 2]

def build_barbell_track_from_keypoints(keypoints_dir, output_path):
    frames = sorted(f for f in os.listdir(keypoints_dir) if f.endswith(".json"))
    track = []

    for frame_file in frames:
        with open(os.path.join(keypoints_dir, frame_file), "r") as f:
            data = json.load(f)

        center = compute_barbell_center(data["keypoints"])
        track.append({
            "frame": frame_file,
            "center": center
        })

    with open(output_path, "w") as f:
        json.dump(track, f, indent=2)

    print(f"[INFO] Saved barbell track to {output_path}")
