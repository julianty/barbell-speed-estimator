"""Stage-0 profiler for the resolution-downscaling investigation.

Measures per-frame cost of each pipeline stage as a function of the *source*
frame resolution, to answer: how much of the per-frame budget is actually
resolution-bound, and how much is fixed by the model's 640x640 input?

Both deployed models (yolo26s-pose.mlpackage, yolov8s-pose.onnx) declare a
fixed 640x640 input with dynamic=False, so anything above 640 on the long edge
is discarded before the network sees it. This script measures what that
discarded detail costs us per frame.

Stages timed, per resolution:
  decode    - cv2 frame read from the source video (always full-res; the cost
              you cannot avoid without re-encoding)
  resize    - full-res -> target resolution (the cost downscaling *adds*)
  png       - PNG encode at the target resolution. Proxy for the iOS backend's
              per-frame dart:ui PNG encode (ios_yolo_backend.dart:37). Skia vs
              libpng differ in constant factor; the scaling law is the point.
  letterbox - RGBA -> (1,3,640,640) float tensor. Proxy for the web backend's
              preprocessLetterbox (yolo_decoder.dart:57). numpy here vs a Dart
              scalar loop there: treat as a lower bound, not an estimate.
  model     - ultralytics predict on the target-resolution frame.

Usage:
  python tools/profile_stages.py --video "videos/26-6-30 225lb bench press 1.mov"
  python tools/profile_stages.py --frames 40 --json out.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import pick_video_file  # noqa: E402

# Long-edge targets to profile. 1920 = native (no resize); 640 = the models'
# native input size; below that the model is upscaling padding, not detail.
DEFAULT_LONG_EDGES = [1920, 1280, 960, 640, 480, 360]

MODEL_INPUT = 640


def resize_long_edge(frame, long_edge):
    h, w = frame.shape[:2]
    if max(h, w) == long_edge:
        return frame
    scale = long_edge / max(h, w)
    return cv2.resize(
        frame,
        (round(w * scale), round(h * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )


def letterbox_tensor(frame, size=MODEL_INPUT):
    """Mirror of yolo_decoder.dart preprocessLetterbox: aspect-preserving
    resize into a 114-gray square, then NCHW float32 in 0..1."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = round((size - nh) / 2 - 0.1)
    left = round((size - nw) / 2 - 0.1)
    canvas[top:top + nh, left:left + nw] = resized
    return np.ascontiguousarray(
        canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )


def summarize(samples_ms):
    """Median, not mean: the first few frames carry lazy-init and thermal
    noise that a mean would smear across the whole measurement."""
    if not samples_ms:
        return None
    return {
        "median_ms": round(statistics.median(samples_ms), 2),
        "p90_ms": round(sorted(samples_ms)[int(0.9 * (len(samples_ms) - 1))], 2),
        "min_ms": round(min(samples_ms), 2),
        "n": len(samples_ms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=None,
                        help="source video (file picker if omitted)")
    parser.add_argument("--frames", type=int, default=30,
                        help="frames to time per resolution (default: 30)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="untimed frames before each resolution (default: 3)")
    parser.add_argument("--model", default="yolo_pose/models/yolo26s-pose.pt",
                        help="ultralytics weights; default matches the app's "
                             "deployed model scale (s), not the large weights "
                             "the Python pipeline usually runs")
    parser.add_argument("--long-edges", type=int, nargs="+",
                        default=DEFAULT_LONG_EDGES)
    parser.add_argument("--json", default=None, help="write results as JSON")
    parser.add_argument("--skip-model", action="store_true",
                        help="time only the CPU stages (no weights needed)")
    args = parser.parse_args()

    video = args.video or pick_video_file("Select video to profile")
    if not video:
        parser.error("no video selected")

    model = None
    if not args.skip_model:
        from ultralytics import YOLO
        model = YOLO(args.model)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        parser.error(f"could not open {video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    # Decode once into memory so every resolution is timed against identical
    # frames (and decode cost isn't re-paid per resolution).
    need = args.frames + args.warmup
    frames = []
    while len(frames) < need:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        parser.error(f"only {len(frames)} frames decoded from {video}")

    # Decode cost, measured separately on a fresh reader.
    cap = cv2.VideoCapture(video)
    decode_ms = []
    for _ in range(need):
        t = time.perf_counter()
        ok, _f = cap.read()
        decode_ms.append((time.perf_counter() - t) * 1000)
        if not ok:
            break
    cap.release()
    decode_ms = decode_ms[args.warmup:]

    print(f"video      : {os.path.basename(video)}")
    print(f"source     : {src_w}x{src_h} @ {src_fps:.3f} fps")
    print(f"model      : {'(skipped)' if model is None else args.model}")
    print(f"frames     : {len(frames) - args.warmup} timed "
          f"({args.warmup} warmup) per resolution")
    print(f"decode     : {summarize(decode_ms)['median_ms']} ms/frame "
          f"(resolution-independent; source is {src_w}x{src_h})")
    print()

    results = {
        "video": video,
        "source": {"width": src_w, "height": src_h, "fps": src_fps},
        "model": None if model is None else args.model,
        "frames_timed": len(frames) - args.warmup,
        "decode": summarize(decode_ms),
        "by_long_edge": {},
    }

    header = f"{'long edge':>10} {'size':>11} {'resize':>9} {'png':>9} {'letterbox':>11} {'model':>9} {'total*':>9}"
    print(header)
    print("-" * len(header))

    for long_edge in args.long_edges:
        scaled = [resize_long_edge(f, long_edge) for f in frames]
        h, w = scaled[0].shape[:2]

        resize_ms, png_ms, lb_ms, model_ms = [], [], [], []

        for i, (full, small) in enumerate(zip(frames, scaled)):
            timed = i >= args.warmup

            t = time.perf_counter()
            r = resize_long_edge(full, long_edge)
            dt = (time.perf_counter() - t) * 1000
            if timed:
                resize_ms.append(dt)

            t = time.perf_counter()
            cv2.imencode(".png", small)
            dt = (time.perf_counter() - t) * 1000
            if timed:
                png_ms.append(dt)

            t = time.perf_counter()
            letterbox_tensor(small)
            dt = (time.perf_counter() - t) * 1000
            if timed:
                lb_ms.append(dt)

            if model is not None:
                t = time.perf_counter()
                model.predict(small, verbose=False)
                dt = (time.perf_counter() - t) * 1000
                if timed:
                    model_ms.append(dt)
            del r

        entry = {
            "width": w,
            "height": h,
            "megapixels": round(w * h / 1e6, 2),
            "resize": summarize(resize_ms),
            "png": summarize(png_ms),
            "letterbox": summarize(lb_ms),
            "model": summarize(model_ms),
        }
        results["by_long_edge"][str(long_edge)] = entry

        def med(stage):
            s = entry[stage]
            return s["median_ms"] if s else 0.0

        # total* = the iOS-shaped path (resize + png + model). letterbox is the
        # web-shaped alternative to png, not an addition, so it's excluded.
        total = med("resize") + med("png") + med("model")
        print(f"{long_edge:>10} {w}x{h:<6} {med('resize'):>9.2f} {med('png'):>9.2f} "
              f"{med('letterbox'):>11.2f} {med('model'):>9.2f} {total:>9.2f}")

    print()
    print("* total = resize + png + model (the iOS-shaped path). 'letterbox' is")
    print("  the web path's alternative to png, not an addition to it.")
    print("  Decode is paid once at source resolution regardless, and on web is")
    print("  further floored by realtime playback (requestVideoFrameCallback).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
