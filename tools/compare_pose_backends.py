"""compare_pose_backends.py — score replacement pose models against YOLO.

Launch-checklist item 1 in the app's LEDGER.md: find a non-AGPL pose model to
replace yolo26s-pose. For each clip this extracts frames at 30 fps (the app's
shipped rate, D2), runs each backend, builds the wrist-midpoint bar track the
same way the YOLO sweep does, and runs the same analysis. The reference is the
existing YOLO run at `sweep/<clip>/fps30/`, produced by tools/sweep_configs.py
with yolo26s-pose, the iOS model's architecture and scale.

YOLO is the reference, not the truth: the analysis is golden-exact only for
YOLO's wrists, so agreement with YOLO is what tells us how much changes. Where
a rep was adjudicated by hand (sweep/_ground_truth_lifts/disputed_reps.json),
the error against the truth is reported too.

Backends:
  vision            Apple Vision VNDetectHumanBodyPoseRequest (tools/vision_pose.swift)
  movenet_thunder   MoveNet SinglePose Thunder, float16 TFLite (tools/movenet_pose.py)
  movenet_lightning MoveNet SinglePose Lightning, float16 TFLite
  yolo26s           yolo26s-pose via ultralytics, in-process. Useful with
                    --max-edge 640: it is the app's actual configuration, so it
                    shows how much the downscale alone moves YOLO's reps.

--max-edge 640 downscales every frame to a 640 long edge before inference, as
the app's decode does (FrameExtractor.scaledDecodeSize, D4), then scales the
keypoints back to source pixels, as `scaleDetectionsToSource` does, so the
pixel-unit analysis thresholds mean the same thing. Run dirs get a `_<edge>`
suffix.

    python tools/compare_pose_backends.py --movenet-python <env>/bin/python \\
        --movenet-dir <dir with movenet_thunder.tflite, movenet_lightning.tflite>

Writes sweep/<clip>/fps30_<backend>/{keypoints.json, barbell_track.json,
meta.json, timing.json, <arm>/{reps.json, signals.json}} and
sweep/_backends/{REPORT.md, reps.csv}. ms/frame here is on the Mac, which says
nothing about the iPhone; device speed is a separate measurement.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "yolo_pose"))

from utils import extract_frames, get_video_fps  # noqa: E402
from build_barbell_track import compute_barbell_center, MAX_JUMP_FRACTION  # noqa: E402
from sweep_configs import analyse, ARMS  # noqa: E402

FPS = 30.0
BACKENDS = ["vision", "movenet_thunder", "movenet_lightning", "yolo26s"]
DEFAULT_VIDEOS = [f"deadlift{i}" for i in range(1, 7)] + ["squat2", "squat3"]
LW, RW = 9, 10


def run_yolo(frames_dir, out_json):
    import time
    import cv2
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "yolo_pose", "models", "yolo26s-pose.pt"))
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    frames, model_s, size = [], 0.0, [0, 0]
    for name in files:
        img = cv2.imread(os.path.join(frames_dir, name))
        size = [img.shape[1], img.shape[0]]
        t = time.perf_counter()
        r = model.predict(img, verbose=False)[0]
        model_s += time.perf_counter() - t
        people = []
        if r.keypoints is not None and r.keypoints.conf is not None:
            for xy, cf, box in zip(r.keypoints.xy.tolist(), r.keypoints.conf.tolist(),
                                   r.boxes.xyxy.tolist()):
                people.append({"kp": [[x, y, c] for (x, y), c in zip(xy, cf)], "box": box})
        frames.append({"frame": name, "people": people})
    with open(out_json, "w") as f:
        json.dump({"backend": "yolo26s", "frame_size": size, "model_s": model_s,
                   "frames": frames}, f)
    print(f"yolo26s: {len(files)} frames, {1000 * model_s / max(len(files), 1):.2f} ms/frame")


def run_backend(backend, frames_dir, out_json, args):
    if backend == "yolo26s":
        return run_yolo(frames_dir, out_json)
    if backend == "vision":
        cmd = ["swift", os.path.join(ROOT, "tools", "vision_pose.swift"), frames_dir, out_json]
    else:
        model = os.path.join(args.movenet_dir, f"{backend}.tflite")
        cmd = [args.movenet_python, "-I", os.path.join(ROOT, "tools", "movenet_pose.py"),
               "--model", model, frames_dir, out_json]
    subprocess.run(cmd, check=True)


def build_track(kp_json, min_conf, source_size):
    """Wrist midpoint per frame, as build_barbell_track does for YOLO, with one
    difference YOLO never needed: these models report a per-joint confidence and
    can lose a wrist. A wrist under `min_conf` is replaced by the other one; a
    person with neither is dropped from that frame. Keypoints are scaled from
    the inference frame to `source_size`, as the app does."""
    w, h = source_size
    sx, sy = w / kp_json["frame_size"][0], h / kp_json["frame_size"][1]
    max_jump = MAX_JUMP_FRACTION * (w ** 2 + h ** 2) ** 0.5
    track, last, one_wrist, no_wrist, rejected = [], None, 0, 0, 0
    for fr in kp_json["frames"]:
        kps, boxes = [], []
        for p in fr["people"]:
            kp = [list(j) for j in p["kp"]]
            lc, rc = kp[LW][2], kp[RW][2]
            if lc < min_conf and rc < min_conf:
                continue
            if lc < min_conf:
                kp[LW] = kp[RW]
            elif rc < min_conf:
                kp[RW] = kp[LW]
            kps.append([[j[0] * sx, j[1] * sy] for j in kp])
            b = p["box"]
            boxes.append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
            if (lc < min_conf) != (rc < min_conf):
                one_wrist += 1
        if not kps:
            no_wrist += 1
        center, jumped = compute_barbell_center(kps, boxes=boxes, last_center=last,
                                                max_jump_dist=max_jump)
        rejected += jumped
        if center is not None:
            last = center
        track.append({"frame": fr["frame"], "center": center})
    stats = {"one_wrist_frames": one_wrist, "no_wrist_frames": no_wrist,
             "rejected_jumps": rejected}
    return track, (w, h), stats


def downscale(frames_dir, max_edge):
    """Resize frames in place so the long edge is max_edge, short edge rounded
    to even, as FrameExtractor.scaledDecodeSize does. Never upscales."""
    import cv2
    for f in os.listdir(frames_dir):
        if not f.endswith(".jpg"):
            continue
        path = os.path.join(frames_dir, f)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        long_edge = max(w, h)
        if long_edge <= max_edge:
            continue
        s = max_edge / long_edge
        size = (max_edge, 2 * round(h * s / 2)) if w >= h else (2 * round(w * s / 2), max_edge)
        cv2.imwrite(path, cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def jitter(track, frame_h):
    """Std of (raw y - Savitzky-Golay y) on detected frames, in frame heights x 1000.
    A model-independent measure of frame-to-frame wrist noise; lower is steadier."""
    y = np.array([np.nan if e["center"] is None else e["center"][1] for e in track], float)
    ok = ~np.isnan(y)
    if ok.sum() < 15:
        return float("nan")
    yy = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    resid = (yy - savgol_filter(yy, 9, 3))[ok]
    return float(np.std(resid) / frame_h * 1000)


def run_clip(name, backend, args):
    suffix = f"_{args.max_edge}" if args.max_edge else ""
    run_dir = os.path.join(args.sweep, name, f"fps30_{backend}{suffix}")
    kp_path = os.path.join(run_dir, "keypoints.json")
    os.makedirs(run_dir, exist_ok=True)
    if args.force or not os.path.exists(kp_path):
        frames_dir = os.path.join(args.sweep, "_frames", f"{name}_fps30{suffix}")
        if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
            extract_frames(os.path.join(ROOT, "videos", f"{name}.mov"), frames_dir, fps=FPS)
            if args.max_edge:
                downscale(frames_dir, args.max_edge)
        run_backend(backend, frames_dir, kp_path, args)

    with open(kp_path) as f:
        kp_json = json.load(f)
    source_fps = get_video_fps(os.path.join(ROOT, "videos", f"{name}.mov"))
    # Source size in the rotated, upright orientation the frames come out in.
    src = json.load(open(os.path.join(args.sweep, name, "fps30", "meta.json")))["frame_size"]
    track, (w, h), stats = build_track(kp_json, args.min_conf, src)
    n = len(track)
    with open(os.path.join(run_dir, "barbell_track.json"), "w") as f:
        json.dump(track, f)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"fps": FPS, "source_fps": source_fps, "video": f"videos/{name}.mov",
                   "frame_size": [w, h], "baseline": False, "backend": backend}, f, indent=2)
    timing = {"backend": backend, "frames": n, "model_s": round(kp_json["model_s"], 3),
              "model_ms_per_frame": round(1000 * kp_json["model_s"] / n, 2), **stats}
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    for arm in ARMS:
        reps, signals = analyse(track, FPS, source_fps, h, scale_constants=(arm == "normalized"))
        os.makedirs(os.path.join(run_dir, arm), exist_ok=True)
        with open(os.path.join(run_dir, arm, "reps.json"), "w") as f:
            json.dump(reps, f, indent=2)
        with open(os.path.join(run_dir, arm, "signals.json"), "w") as f:
            json.dump(signals, f)
    return run_dir


def load_reps(run_dir, arm):
    with open(os.path.join(run_dir, arm, "reps.json")) as f:
        return json.load(f)


def match(ref, cand):
    """Pair each reference rep with the candidate rep it overlaps most in time."""
    pairs, used = [], set()
    for r in ref:
        best, best_ov = None, 0.0
        for i, c in enumerate(cand):
            if i in used:
                continue
            ov = min(r["end_s"], c["end_s"]) - max(r["start_s"], c["start_s"])
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None:
            used.add(best)
        pairs.append((r, None if best is None else cand[best]))
    extra = [c for i, c in enumerate(cand) if i not in used]
    return pairs, extra


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", action="append", help=f"clip name; default {DEFAULT_VIDEOS}")
    p.add_argument("--backend", action="append", choices=BACKENDS)
    p.add_argument("--movenet-python", help="python with ai-edge-litert (3.12 env)")
    p.add_argument("--movenet-dir", help="directory holding movenet_*.tflite")
    p.add_argument("--min-conf", type=float, default=0.1,
                   help="wrist confidence below which a wrist counts as lost (default 0.1)")
    p.add_argument("--arm", default="normalized", choices=ARMS,
                   help="analysis arm to score; normalized is what the app ships (R3)")
    p.add_argument("--max-edge", type=int, default=None,
                   help="downscale frames to this long edge before inference (app: 640)")
    p.add_argument("--ref", default="fps30",
                   help="reference run dir under sweep/<clip>/ (default fps30, the "
                        "full-res YOLO sweep)")
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--force", action="store_true", help="re-run inference")
    args = p.parse_args()
    videos = args.video or DEFAULT_VIDEOS
    backends = args.backend or [b for b in BACKENDS if b != "yolo26s"]
    if any(b.startswith("movenet") for b in backends) and not (args.movenet_python and args.movenet_dir):
        p.error("movenet backends need --movenet-python and --movenet-dir")

    with open(os.path.join(args.sweep, "_ground_truth_lifts", "disputed_reps.json")) as f:
        truth = [t for t in json.load(f) if t["truth_end_s"] is not None]

    rows, summary = [], {b: {"ms": [], "jit": [], "det": [], "missed": 0, "extra": 0,
                             "d_start": [], "d_end": [], "d_dur": [], "n_ref": 0}
                         for b in ["ref"] + backends}
    truth_rows = []
    for name in videos:
        yolo_dir = os.path.join(args.sweep, name, args.ref)
        ref = load_reps(yolo_dir, args.arm)
        with open(os.path.join(yolo_dir, "barbell_track.json")) as f:
            ytrack = json.load(f)
        with open(os.path.join(yolo_dir, "timing.json")) as f:
            ytime = json.load(f)
        h = json.load(open(os.path.join(yolo_dir, "meta.json")))["frame_size"][1]
        s = summary["ref"]
        s["ms"].append(ytime["model_ms_per_frame"])
        s["jit"].append(jitter(ytrack, h))
        s["det"].append(np.mean([e["center"] is not None for e in ytrack]))
        s["n_ref"] += len(ref)
        runs = {"ref": yolo_dir}

        for b in backends:
            print(f"--- {name} / {b}")
            run_dir = run_clip(name, b, args)
            runs[b] = run_dir
            cand = load_reps(run_dir, args.arm)
            track = json.load(open(os.path.join(run_dir, "barbell_track.json")))
            t = json.load(open(os.path.join(run_dir, "timing.json")))
            s = summary[b]
            s["ms"].append(t["model_ms_per_frame"])
            s["jit"].append(jitter(track, h))
            s["det"].append(np.mean([e["center"] is not None for e in track]))
            s["n_ref"] += len(ref)
            pairs, extra = match(ref, cand)
            s["extra"] += len(extra)
            for r, c in pairs:
                row = {"video": name, "backend": b, "rep": r["rep"],
                       "yolo_start_s": r["start_s"], "yolo_end_s": r["end_s"],
                       "yolo_dur_s": r["duration_s"]}
                if c is None:
                    s["missed"] += 1
                else:
                    ds, de = c["start_s"] - r["start_s"], c["end_s"] - r["end_s"]
                    s["d_start"].append(ds)
                    s["d_end"].append(de)
                    s["d_dur"].append(c["duration_s"] - r["duration_s"])
                    row.update({"start_s": c["start_s"], "end_s": c["end_s"],
                                "dur_s": c["duration_s"], "d_start_s": ds, "d_end_s": de,
                                "d_dur_s": c["duration_s"] - r["duration_s"]})
                rows.append(row)
            for c in extra:
                rows.append({"video": name, "backend": b, "rep": None, "start_s": c["start_s"],
                             "end_s": c["end_s"], "dur_s": c["duration_s"], "extra": True})

        for tr in (t for t in truth if t["video"] == name):
            for b, rd in runs.items():
                reps = load_reps(rd, args.arm)
                # The rep whose window best overlaps the adjudicated one.
                lo = tr["truth_start_s"] if tr["truth_start_s"] is not None else tr["truth_end_s"] - 1.0
                best = max(reps, key=lambda c: min(c["end_s"], tr["truth_end_s"]) - max(c["start_s"], lo),
                           default=None)
                truth_rows.append({"video": name, "rep": tr["rep"], "backend": b,
                                   "start_err_ms": None if best is None or tr["truth_start_s"] is None
                                   else 1000 * (best["start_s"] - tr["truth_start_s"]),
                                   "end_err_ms": None if best is None
                                   else 1000 * (best["end_s"] - tr["truth_end_s"])})

    out_dir = os.path.join(args.sweep, "_backends" + (f"_{args.max_edge}" if args.max_edge else ""))
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(out_dir, "reps.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    def ms(xs, q):
        return f"{1000 * np.percentile(np.abs(xs), q):.0f}" if xs else "–"

    edge = f"{args.max_edge}px long edge" if args.max_edge else "full resolution"
    lines = [f"# Pose backends vs `{args.ref}` @ 30 fps, {edge} ({args.arm} arm)", "",
             f"Clips: {', '.join(videos)}. Wrist min confidence {args.min_conf}. "
             "Timing errors are against YOLO's rep boundaries, in ms (|median| / |p90| / |max|).", "",
             "| backend | Mac ms/frame | detection | jitter (‰ frame h) | YOLO reps matched | missed | extra "
             "| start err | end err | duration err |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b, s in summary.items():
        if b == "ref":
            lines.append(f"| {args.ref} (ref) | {np.mean(s['ms']):.1f} | {np.mean(s['det']):.1%} "
                         f"| {np.nanmean(s['jit']):.2f} | {s['n_ref']} | – | – | – | – | – |")
            continue
        within = sum(abs(a) <= 0.1 and abs(e) <= 0.1 for a, e in zip(s["d_start"], s["d_end"]))
        lines.append(
            f"| {b} | {np.mean(s['ms']):.1f} | {np.mean(s['det']):.1%} | {np.nanmean(s['jit']):.2f} "
            f"| {s['n_ref'] - s['missed']}/{s['n_ref']} ({within} within 100 ms both ends) "
            f"| {s['missed']} | {s['extra']} "
            f"| {ms(s['d_start'], 50)} / {ms(s['d_start'], 90)} / {ms(s['d_start'], 100)} "
            f"| {ms(s['d_end'], 50)} / {ms(s['d_end'], 90)} / {ms(s['d_end'], 100)} "
            f"| {ms(s['d_dur'], 50)} / {ms(s['d_dur'], 90)} / {ms(s['d_dur'], 100)} |")
    lines += ["", "## Against hand-adjudicated reps (ms, + = late)", "",
              "| clip | rep | " + " | ".join(summary) + " |", "|---|---|" + "---|" * len(summary)]
    for (v, rep) in dict.fromkeys((r["video"], r["rep"]) for r in truth_rows):
        cells = []
        for b in summary:
            r = next(x for x in truth_rows if x["video"] == v and x["rep"] == rep and x["backend"] == b)
            st = "–" if r["start_err_ms"] is None else f"{r['start_err_ms']:+.0f}"
            en = "–" if r["end_err_ms"] is None else f"{r['end_err_ms']:+.0f}"
            cells.append(f"{st} / {en}")
        lines.append(f"| {v} | {rep} | " + " | ".join(cells) + " |")
    lines += ["", "Cells are start / end error. Per-rep detail: reps.csv."]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    shutil.rmtree(os.path.join(args.sweep, "_frames"), ignore_errors=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
