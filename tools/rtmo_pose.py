"""rtmo_pose.py — run RTMO (OpenMMLab, one-stage multi-person pose) over a frames dir.

Part of the pose-backend comparison (tools/compare_pose_backends.py). Writes the
same JSON shape as tools/vision_pose.swift: COCO-17 keypoints [x, y, score]
per person, plus a box. RTMO already outputs COCO-17 order.

Runs in the separate Python 3.12 env (see movenet_pose.py), with rtmlib and
onnxruntime installed:

    <env>/bin/python tools/rtmo_pose.py --model <dir>/end2end.onnx <frames_dir> <out.json>

The ONNX model is OpenMMLab's published RTMO export, e.g.
rtmo-s_8xb32-600e_body7-640x640 from download.openmmlab.com. The code is
Apache 2.0, but the body7 checkpoints are trained partly on datasets with
research-only terms. Before shipping, use a COCO-only checkpoint or check
the terms.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from rtmlib import RTMO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="person score threshold (rtmlib default 0.7 drops partly visible lifters)")
    p.add_argument("frames_dir")
    p.add_argument("out")
    a = p.parse_args()

    model = RTMO(a.model, model_input_size=(640, 640), score_thr=a.score_thr,
                 backend="onnxruntime", device="cpu")
    files = sorted(f for f in os.listdir(a.frames_dir) if f.endswith(".jpg"))
    frames, model_s, size = [], 0.0, [0, 0]
    for name in files:
        img = cv2.imread(os.path.join(a.frames_dir, name))
        size = [img.shape[1], img.shape[0]]
        t = time.perf_counter()
        kps, scores = model(img)
        model_s += time.perf_counter() - t
        people = []
        for k, s in zip(np.asarray(kps).reshape(-1, 17, 2), np.asarray(scores).reshape(-1, 17)):
            # rtmlib returns raw keypoint scores (often above 1); squash to 0..1
            # so the app's confidence threshold has a comparable meaning.
            conf = 1 / (1 + np.exp(-np.asarray(s, float))) if np.max(s) > 1 else np.asarray(s, float)
            pts = k[conf >= 0.3]
            box = ([float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())] if len(pts) else [0, 0, 0, 0])
            people.append({"kp": [[float(x), float(y), float(c)] for (x, y), c in zip(k, conf)],
                           "box": box, "score": float(np.mean(conf))})
        frames.append({"frame": name, "people": people})
    with open(a.out, "w") as f:
        json.dump({"backend": "rtmo", "frame_size": size, "model_s": model_s, "frames": frames}, f)
    print(f"rtmo: {len(files)} frames, {1000 * model_s / max(len(files), 1):.2f} ms/frame")


if __name__ == "__main__":
    main()
