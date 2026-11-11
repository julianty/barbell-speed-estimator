"""movenet_pose.py — run MoveNet single-pose (TFLite) over a directory of frames.

Part of the pose-backend comparison (tools/compare_pose_backends.py). Writes the
same JSON shape as tools/vision_pose.swift. MoveNet already outputs COCO-17
order, so wrists are 9 and 10 with no remapping.

Needs ai-edge-litert, which has no wheel for the project venv's Python 3.13, so
it runs in its own Python 3.12 env:

    uv venv -p 3.12 <env> && VIRTUAL_ENV=<env> uv pip install ai-edge-litert numpy opencv-python-headless
    <env>/bin/python tools/movenet_pose.py --model movenet_thunder.tflite <frames_dir> <out.json>

Cropping follows the reference video loop in TensorFlow's MoveNet tutorial
(`init_crop_region` / `determine_crop_region`): the first frame is the whole
image padded to a square, and each later frame is a square crop around the
previous frame's confident joints. Without it, a 1080x1920 portrait frame
squeezed to 256x256 leaves the lifter about 140 px tall. `--no-crop` turns it
off.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MIN_CROP_KEYPOINT_SCORE = 0.2
# COCO indices: 5/6 shoulders, 11/12 hips.
TORSO = [5, 6, 11, 12]


def init_crop_region(h, w):
    """Whole image, padded to a square, in normalized coords."""
    if w > h:
        box_h = w / h
        return {"ymin": (h / 2 - w / 2) / h, "xmin": 0.0, "h": box_h, "w": 1.0}
    box_w = h / w
    return {"ymin": 0.0, "xmin": (w / 2 - h / 2) / w, "h": 1.0, "w": box_w}


def torso_visible(kp):
    return ((kp[11, 2] > MIN_CROP_KEYPOINT_SCORE or kp[12, 2] > MIN_CROP_KEYPOINT_SCORE)
            and (kp[5, 2] > MIN_CROP_KEYPOINT_SCORE or kp[6, 2] > MIN_CROP_KEYPOINT_SCORE))


def determine_crop_region(kp_px, h, w):
    """kp_px: 17x3 of (y_px, x_px, score) in full-image pixels."""
    if not torso_visible(kp_px):
        return init_crop_region(h, w)
    cy = (kp_px[11, 0] + kp_px[12, 0]) / 2
    cx = (kp_px[11, 1] + kp_px[12, 1]) / 2
    max_torso_y = max(abs(cy - kp_px[j, 0]) for j in TORSO)
    max_torso_x = max(abs(cx - kp_px[j, 1]) for j in TORSO)
    conf = kp_px[:, 2] > MIN_CROP_KEYPOINT_SCORE
    max_body_y = np.max(np.abs(cy - kp_px[conf, 0]))
    max_body_x = np.max(np.abs(cx - kp_px[conf, 1]))
    half = max(max_torso_x * 1.9, max_torso_y * 1.9, max_body_y * 1.2, max_body_x * 1.2)
    half = min(half, max(cx, w - cx, cy, h - cy))
    if half > max(w, h) / 2:
        return init_crop_region(h, w)
    side = 2 * half
    return {"ymin": (cy - half) / h, "xmin": (cx - half) / w, "h": side / h, "w": side / w}


def crop_and_resize(img, region, size):
    """Crop a (possibly out-of-bounds) normalized region, zero-padding, then resize."""
    h, w = img.shape[:2]
    y0 = int(round(region["ymin"] * h))
    x0 = int(round(region["xmin"] * w))
    y1 = int(round((region["ymin"] + region["h"]) * h))
    x1 = int(round((region["xmin"] + region["w"]) * w))
    canvas = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y1, h), min(x1, w)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("frames_dir")
    p.add_argument("out")
    a = p.parse_args()

    interp = Interpreter(model_path=a.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    size = int(inp["shape"][1])

    files = sorted(f for f in os.listdir(a.frames_dir) if f.endswith(".jpg"))
    frames, model_s, region, frame_size = [], 0.0, None, [0, 0]
    for name in files:
        bgr = cv2.imread(os.path.join(a.frames_dir, name))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        frame_size = [w, h]
        if region is None or a.no_crop:
            region = init_crop_region(h, w)
        x = crop_and_resize(img, region, size)[None].astype(np.uint8)

        t = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0, 0]  # 17 x (y, x, score), crop-normalized
        model_s += time.perf_counter() - t

        kp_px = np.empty_like(raw)
        kp_px[:, 0] = (region["ymin"] + raw[:, 0] * region["h"]) * h
        kp_px[:, 1] = (region["xmin"] + raw[:, 1] * region["w"]) * w
        kp_px[:, 2] = raw[:, 2]

        kp = [[float(c[1]), float(c[0]), float(c[2])] for c in kp_px]  # -> (x, y, score)
        good = kp_px[kp_px[:, 2] > MIN_CROP_KEYPOINT_SCORE]
        box = ([float(good[:, 1].min()), float(good[:, 0].min()),
                float(good[:, 1].max()), float(good[:, 0].max())] if len(good) else [0, 0, 0, 0])
        frames.append({"frame": name, "people": [{"kp": kp, "box": box,
                                                   "score": float(kp_px[:, 2].mean())}]})
        if not a.no_crop:
            region = determine_crop_region(kp_px, h, w)

    with open(a.out, "w") as f:
        json.dump({"backend": os.path.basename(a.model), "frame_size": frame_size,
                   "model_s": model_s, "frames": frames}, f)
    print(f"movenet: {len(files)} frames, {1000 * model_s / max(len(files), 1):.2f} ms/frame")


if __name__ == "__main__":
    main()
