"""mediapipe_pose.py — run MediaPipe Pose Landmarker (BlazePose GHUM) over a frames dir.

Part of the pose-backend comparison (tools/compare_pose_backends.py). Writes the
same JSON shape as tools/vision_pose.swift. BlazePose's 33 landmarks are
remapped to COCO-17 order, and each joint's confidence is MediaPipe's
`visibility`.

Runs in VIDEO mode, as an app would: the landmarker tracks the person between
frames rather than re-detecting cold. Up to --num-poses people per frame, so
the app's torso selection still decides who the lifter is.

Runs in the separate Python 3.12 env with `mediapipe` installed:

    <env>/bin/python tools/mediapipe_pose.py --model pose_landmarker_full.task <frames_dir> <out.json>

Model: pose_landmarker_{lite,full,heavy}.task from
storage.googleapis.com/mediapipe-models. Apache 2.0.
"""

import argparse
import json
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# BlazePose landmark index for each COCO-17 joint.
COCO_FROM_BLAZEPOSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-poses", type=int, default=3)
    p.add_argument("--fps", type=float, default=30.0, help="for VIDEO-mode timestamps")
    p.add_argument("frames_dir")
    p.add_argument("out")
    a = p.parse_args()

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=a.num_poses,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    files = sorted(f for f in os.listdir(a.frames_dir) if f.endswith(".jpg"))
    frames, model_s, size = [], 0.0, [0, 0]
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i, name in enumerate(files):
            bgr = cv2.imread(os.path.join(a.frames_dir, name))
            h, w = bgr.shape[:2]
            size = [w, h]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            t = time.perf_counter()
            res = lm.detect_for_video(image, int(round(i * 1000 / a.fps)))
            model_s += time.perf_counter() - t
            people = []
            for lms in res.pose_landmarks:
                kp = [[lms[j].x * w, lms[j].y * h, float(lms[j].visibility)] for j in COCO_FROM_BLAZEPOSE]
                xs = [q.x * w for q in lms]
                ys = [q.y * h for q in lms]
                people.append({"kp": kp, "box": [min(xs), min(ys), max(xs), max(ys)],
                               "score": float(sum(q.presence for q in lms) / len(lms))})
            frames.append({"frame": name, "people": people})
    with open(a.out, "w") as f:
        json.dump({"backend": "mediapipe", "frame_size": size, "model_s": model_s,
                   "frames": frames}, f)
    print(f"mediapipe: {len(files)} frames, {1000 * model_s / max(len(files), 1):.2f} ms/frame")


if __name__ == "__main__":
    main()
