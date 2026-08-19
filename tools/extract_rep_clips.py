"""Cut annotated clips for the reps where 60 fps and 30 fps disagree.

Stage 1b left five reps (of 40) where the two frame rates report materially
different concentric durations, always with 30 fps reporting the longer rep and
always on the fastest, early-set reps. Neither run is known to be correct: the
dataset has no ground-truth rep duration, and "agrees with 60 fps" is not a
valid referee when 60 fps is one of the two runs on trial.

This renders each disputed rep as a slow-motion clip showing BOTH runs' start
and lockout marks against the actual barbell motion, so a human can judge which
boundary is right and produce the missing ground truth.

Each frame carries:
  - the source frame with the wrist-midpoint marker the track is built from
  - a position-vs-time trace for the window, with both runs' boundaries drawn
  - live badges showing which run currently considers the lifter mid-rep

Playback is slowed (default 15 fps out from 60 fps in, so 4x slow) because the
disagreements are a few frames wide and are invisible at normal speed.

Frames are extracted with the same full-clip ffmpeg pass the pipeline uses, and
indexed directly, rather than seeking to the window: seek rounding would offset
the overlay from the track by a frame or two, which is the same order as the
thing being judged.

Usage:
  python tools/extract_rep_clips.py
  python tools/extract_rep_clips.py --threshold 0.10 --out sweep/_ground_truth
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "yolo_pose"))

from utils import remove_outliers, smooth_positions  # noqa: E402
import analysis as ana  # noqa: E402

# These clips are portrait (1080x1920). Stacking a wide trace panel under a
# portrait frame gives a ~2000px-tall video that no one can actually watch, so
# portrait sources get the panel beside them instead and landscape below.
VIDEO_H = 760
PANEL_H = 260
PANEL_W_SIDE = 620

C_BASE = (120, 255, 120)  # BGR, green  - the native (60 fps) run
C_CFG = (60, 170, 255)    # BGR, orange - the config under test
C_CUR = (255, 255, 255)


def smoothed(track, fps, source_fps):
    s = fps / source_fps
    raw = [None if e["center"] is None else -e["center"][1] for e in track]
    clean = remove_outliers(raw, window=ana._frames(ana.OUTLIER_WINDOW_REF, s), threshold=50)
    poly = 3
    w = ana._odd_savgol_window(ana.SMOOTH_WINDOW_REF, s, len(clean))
    if w <= poly:
        poly = max(0, w - 1)
    return np.asarray(smooth_positions(clean, window=w, poly=poly))


def draw_panel(pos, lo, hi, cur, marks, width, height):
    """Position trace over [lo,hi) with boundary marks and a playback cursor."""
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    seg = pos[lo:hi]
    if len(seg) < 2:
        return panel
    y0, y1 = float(seg.min()), float(seg.max())
    rng = (y1 - y0) or 1.0
    pad = 30

    def px(i):
        return int((i - lo) / max(1, hi - lo - 1) * (width - 1))

    def py(v):
        return int(height - pad - (v - y0) / rng * (height - 2 * pad))

    # Labels are staggered down the panel: the 60 and 30 marks can land within
    # a few pixels of each other, which is exactly the case worth reading.
    for row, (label, frame, color, dash) in enumerate(marks):
        x = px(frame)
        if not (0 <= x < width):
            continue
        if dash:
            for y in range(0, height, 10):
                cv2.line(panel, (x, y), (x, min(height, y + 5)), color, 2)
        else:
            cv2.line(panel, (x, 0), (x, height), color, 2)
        ly = 18 + row * 17
        lx = x + 4 if x < width - 70 else x - 66
        cv2.putText(panel, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    pts = np.array([[px(lo + i), py(v)] for i, v in enumerate(seg)], dtype=np.int32)
    cv2.polylines(panel, [pts], False, (235, 235, 235), 2, cv2.LINE_AA)

    x = px(cur)
    cv2.line(panel, (x, 0), (x, height), C_CUR, 1)
    cv2.circle(panel, (x, py(pos[cur])), 5, C_CUR, -1)
    cv2.putText(panel, "bar height (up = higher)", (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    return panel


def render(video, track, pos, source_fps, rep_no, b60, b30, out_path, frames_dir,
           pad_s, out_fps, lab_base="60fps", lab_cfg="30fps"):
    s60, e60 = b60
    s30, e30 = b30
    lo = max(0, int(round(min(s60, s30) - pad_s * source_fps)))
    hi = min(len(pos), int(round(max(e60, e30) + pad_s * source_fps)))

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    first = cv2.imread(os.path.join(frames_dir, files[lo]))
    fh, fw = first.shape[:2]
    portrait = fh > fw
    scale = VIDEO_H / fh
    vw, vh = int(round(fw * scale)), VIDEO_H
    if portrait:
        panel_w, panel_h = PANEL_W_SIDE, vh
        out_size = (vw + panel_w, vh)
    else:
        panel_w, panel_h = vw, PANEL_H
        out_size = (vw, vh + panel_h)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, out_size)
    n_base = lab_base.replace("fps", "")
    n_cfg = lab_cfg.replace("fps", "")
    marks = [(f"{n_base} start", s60, C_BASE, False), (f"{n_base} lock", e60, C_BASE, False),
             (f"{n_cfg} start", s30, C_CFG, True), (f"{n_cfg} lock", e30, C_CFG, True)]

    for i in range(lo, hi):
        if i >= len(files):
            break
        frame = cv2.imread(os.path.join(frames_dir, files[i]))
        if frame is None:
            continue
        frame = cv2.resize(frame, (vw, vh))

        c = track[i]["center"]
        if c is not None:
            cv2.circle(frame, (int(c[0] * scale), int(c[1] * scale)), 9, (0, 215, 255), 2)
            cv2.drawMarker(frame, (int(c[0] * scale), int(c[1] * scale)),
                           (0, 215, 255), cv2.MARKER_CROSS, 22, 1)

        # Gym footage is bright and cluttered; plain text over it is unreadable.
        banner = frame[0:96, 0:min(vw, 430)].copy()
        frame[0:96, 0:min(vw, 430)] = cv2.addWeighted(
            banner, 0.25, np.zeros_like(banner), 0.75, 0)

        t = i / source_fps
        cv2.putText(frame, f"rep {rep_no}   frame {i}   t={t:.3f}s", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        for row, (name, (a, b), color) in enumerate(
                ((lab_base, (s60, e60), C_BASE), (lab_cfg, (s30, e30), C_CFG))):
            inside = a <= i <= b
            txt = (f"{name}: CONCENTRIC  ({(b - a) / source_fps:.2f}s)" if inside
                   else f"{name}: -")
            cv2.putText(frame, txt, (12, 54 + row * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color if inside else (110, 110, 110), 2, cv2.LINE_AA)

        panel = draw_panel(pos, lo, hi, i, marks, panel_w, panel_h)
        writer.write(np.hstack([frame, panel]) if portrait
                     else np.vstack([frame, panel]))

    writer.release()
    return lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="sweep")
    p.add_argument("--out", default=None, help="default: <sweep>/_ground_truth")
    p.add_argument("--config", default="fps30", help="run to compare against native")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="relative duration disagreement that makes a rep disputed")
    p.add_argument("--pad-s", type=float, default=0.6, help="context each side")
    p.add_argument("--out-fps", type=float, default=15.0,
                   help="playback fps; 15 from a 60 fps source is 4x slow")
    args = p.parse_args()

    sweep = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    out_dir = args.out or os.path.join(sweep, "_ground_truth")
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, "_frames")

    videos = sorted(d for d in os.listdir(sweep)
                    if not d.startswith("_") and os.path.isdir(os.path.join(sweep, d)))

    disputed = []
    for v in videos:
        n = json.load(open(f"{sweep}/{v}/native/normalized/reps.json"))
        c = json.load(open(f"{sweep}/{v}/{args.config}/normalized/reps.json"))
        if len(n) != len(c):
            print(f"[WARN] {v}: rep counts differ ({len(n)} vs {len(c)}) — skipped")
            continue
        for i, (a, b) in enumerate(zip(n, c)):
            err = abs(b["duration_s"] - a["duration_s"]) / a["duration_s"]
            if err > args.threshold:
                disputed.append((v, i + 1, a, b, err))

    if not disputed:
        sys.exit(f"no reps disagree by more than {args.threshold:.0%}")

    print(f"{len(disputed)} disputed rep(s) at >{args.threshold:.0%}:\n")
    index = []
    by_video = {}
    for d in disputed:
        by_video.setdefault(d[0], []).append(d)

    for v, items in by_video.items():
        meta = json.load(open(f"{sweep}/{v}/native/meta.json"))
        sf = meta["source_fps"]
        video = meta["video"]
        video = video if os.path.isabs(video) else os.path.join(ROOT, video)
        track = json.load(open(f"{sweep}/{v}/native/barbell_track.json"))
        pos = smoothed(track, sf, sf)

        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-qscale:v", "2",
                        os.path.join(scratch, "frame_%06d.jpg")], check=True)

        for _v, rep_no, a, b, err in items:
            b60 = (int(round(a["start_s"] * sf)), int(round(a["end_s"] * sf)))
            b30 = (int(round(b["start_s"] * sf)), int(round(b["end_s"] * sf)))
            name = f"{v} rep{rep_no}.mp4".replace(" ", "_")
            out_path = os.path.join(out_dir, name)
            cfg_label = args.config.replace("fps", "") + "fps"
            lo, hi = render(video, track, pos, sf, rep_no, b60, b30, out_path,
                            scratch, args.pad_s, args.out_fps,
                            lab_base="60fps", lab_cfg=cfg_label)
            print(f"  {v} rep {rep_no}: 60fps {a['duration_s']:.2f}s vs "
                  f"{cfg_label} {b['duration_s']:.2f}s ({err:+.0%})  -> {name}")
            index.append({
                "video": v, "rep": rep_no, "clip": name, "config": args.config,
                "window_frames": [lo, hi], "source_fps": sf,
                "fps60": {"start_s": a["start_s"], "end_s": a["end_s"],
                          "duration_s": a["duration_s"]},
                "fps30": {"start_s": b["start_s"], "end_s": b["end_s"],
                          "duration_s": b["duration_s"]},
                "rel_error": err,
                "truth_start_s": None, "truth_end_s": None,  # fill in by hand
            })

    shutil.rmtree(scratch, ignore_errors=True)
    idx = os.path.join(out_dir, "disputed_reps.json")
    with open(idx, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {len(index)} clip(s) + {idx}")
    print("Fill in truth_start_s / truth_end_s by eye, then both runs can be "
          "scored against a real reference instead of against each other.")


if __name__ == "__main__":
    main()
