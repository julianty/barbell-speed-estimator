# build_barbell_track.py

import os
import json

LEFT_WRIST  = 9
RIGHT_WRIST = 10

# Max fraction of the frame diagonal a wrist-midpoint may move between consecutive
# frames before a match is rejected as a probable track swap (e.g. onto a person in
# the background) rather than the same person's barbell-proxy motion.
MAX_JUMP_FRACTION = 0.20


def _select_by_largest_bbox(boxes):
    """Return index of the detection with the greatest bounding box area."""
    best, best_area = 0, -1
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area, best = area, i
    return best


def _wrist_midpoint(kp):
    if len(kp) <= RIGHT_WRIST:
        return None
    lx, ly = kp[LEFT_WRIST]
    rx, ry = kp[RIGHT_WRIST]
    return [(lx + rx) / 2, (ly + ry) / 2]


def _select_by_wrist_proximity(keypoints, last_center):
    """Return index of the person whose wrist midpoint is closest to last_center."""
    lx, ly = last_center
    best, best_dist = None, float("inf")
    for i, kp in enumerate(keypoints):
        mid = _wrist_midpoint(kp)
        if mid is None:
            continue
        d = (mid[0] - lx) ** 2 + (mid[1] - ly) ** 2
        if d < best_dist:
            best_dist, best = d, i
    return best


def compute_barbell_center(keypoints, boxes=None, last_center=None, max_jump_dist=None):
    """Returns (center, jump_rejected). jump_rejected is True only when a candidate was
    found but discarded for moving farther than max_jump_dist from last_center."""
    if not keypoints:
        return None, False

    if len(keypoints) == 1:
        idx = 0
    elif last_center is not None:
        idx = _select_by_wrist_proximity(keypoints, last_center)
        if idx is None:
            return None, False
    elif boxes:
        idx = _select_by_largest_bbox(boxes)
    else:
        idx = 0

    mid = _wrist_midpoint(keypoints[idx])
    if mid is None:
        return None, False

    if last_center is not None and max_jump_dist is not None:
        lx, ly = last_center
        dist = ((mid[0] - lx) ** 2 + (mid[1] - ly) ** 2) ** 0.5
        if dist > max_jump_dist:
            return None, True

    return mid, False


def build_barbell_track_from_keypoints(keypoints_dir, output_path):
    frames = sorted(f for f in os.listdir(keypoints_dir) if f.endswith(".json"))
    track = []
    last_center = None
    rejected_jumps = 0

    for frame_file in frames:
        with open(os.path.join(keypoints_dir, frame_file), "r") as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        frame_size = data.get("frame_size")
        max_jump_dist = None
        if frame_size:
            w, h = frame_size
            max_jump_dist = MAX_JUMP_FRACTION * ((w ** 2 + h ** 2) ** 0.5)

        center, jump_rejected = compute_barbell_center(
            data["keypoints"], boxes=boxes, last_center=last_center, max_jump_dist=max_jump_dist
        )
        if jump_rejected:
            rejected_jumps += 1
        if center is not None:
            last_center = center

        track.append({"frame": frame_file, "center": center})

    with open(output_path, "w") as f:
        json.dump(track, f, indent=2)

    if rejected_jumps:
        print(f"[INFO] Rejected {rejected_jumps} frame(s) whose nearest wrist match exceeded "
              f"the {MAX_JUMP_FRACTION:.0%}-of-diagonal jump guard (treated as no detection).")
    print(f"[INFO] Saved barbell track to {output_path}")
