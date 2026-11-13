"""bench_rtmo_onnx.py — time an RTMO ONNX model on CPU and Core ML, and check it.

Runs <model> under ONNX Runtime's CPU provider and its Core ML provider with
each compute-unit setting, and compares the top person's keypoints after NMS
(IoU 0.45, score 0.3, as rtmlib) with <reference> on the CPU. Preprocessing is
rtmlib's: scale the long side to 640, paste top-left on a 640x640 canvas of
114 grey, BGR, CHW float32 with no normalisation.

    ffmpeg -i videos/deadlift4.mov -vf "fps=30,scale=-2:640" -frames:v 60 <dir>/f_%03d.jpg
    .venv312/bin/python tools/bench_rtmo_onnx.py <dir> models/rtmo/coco/static_k20.onnx \\
        models/rtmo/coco/end2end.onnx
"""

import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib.tools.object_detection.post_processings import multiclass_nms


def prep(img):
    c = np.full((640, 640, 3), 114, np.uint8)
    r = min(640 / img.shape[0], 640 / img.shape[1])
    im = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    c[:im.shape[0], :im.shape[1]] = im
    return c.transpose(2, 0, 1)[None].astype(np.float32)


def top_person(outputs):
    d, k = outputs
    _, keep = multiclass_nms(d[0, :, :4], d[0, :, 4:5], nms_thr=0.45, score_thr=0.3)
    return None if keep is None else k[0][keep][0]


def run(model, providers, xs):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=providers)
    name = s.get_inputs()[0].name
    for x in xs[:3]:
        s.run(None, {name: x})
    t = time.perf_counter()
    outs = [s.run(None, {name: x}) for x in xs]
    return 1000 * (time.perf_counter() - t) / len(xs), [top_person(o) for o in outs]


def coreml(units):
    return [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": units}),
            "CPUExecutionProvider"]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    frames_dir, model, reference = sys.argv[1:4]
    xs = [prep(cv2.imread(os.path.join(frames_dir, f)))
          for f in sorted(os.listdir(frames_dir)) if f.endswith(".jpg")]
    _, ref = run(reference, ["CPUExecutionProvider"], xs)
    for label, prov in [("CPU", ["CPUExecutionProvider"]), ("CoreML ALL", coreml("ALL")),
                        ("CoreML CPUAndGPU", coreml("CPUAndGPU")),
                        ("CoreML CPUAndNeuralEngine", coreml("CPUAndNeuralEngine"))]:
        ms, kp = run(model, prov, xs)
        pairs = [(a, b) for a, b in zip(kp, ref) if a is not None and b is not None]
        d = [np.median(np.linalg.norm(a[:, :2] - b[:, :2], axis=1)) for a, b in pairs]
        print(f"{label:28} {ms:6.1f} ms/frame  vs reference: kp median {np.median(d):.2f} px, "
              f"max {np.max(d):.2f}; person found {sum(a is not None for a in kp)}/{len(kp)}")


if __name__ == "__main__":
    main()
