"""render_wrist_review.py — clips for judging bar-midpoint candidates by eye.

Follow-on to tools/compare_wrist_rules.py (run it with --save-tracks first).
Vision's peak velocities run about twice YOLO's on some reps under every
wrist rule, and timing labels cannot say which trace is right. A person can:
the bar sits between the hands even when a wrist is hidden.

Each clip is a window around one rep's peak velocity, at 30 fps (the frames
the models saw), played 3x slow. Drawn on each frame:
  - Vision's wrists: filled dots, with confidence; "L lost" / "R lost" when
    under --min-conf
  - three candidate midpoints, BLINDED as letters A/B/C, shuffled per clip:
    YOLO (yolo26s@640, confidence ignored), Vision fallback, Vision offset
  - a short trail for each candidate, so a jump is visible as a jump
The letter mapping is written to review.json.

    python tools/compare_wrist_rules.py --save-tracks
    python tools/render_wrist_review.py
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "yolo_pose")]
from sweep_configs import analyse  # noqa: E402
from utils import extract_frames  # noqa: E402

CANDS = [("yolo", "fps30_yolo26s_640", "ignore"),
         ("vision_fallback", "fps30_vision_640", "fallback"),
         ("vision_offset", "fps30_vision_640", "offset")]
# Distinct BGR colours and marker shapes, so overlapping candidates stay readable.
STYLE = [((0, 220, 255), cv2.MARKER_CROSS), ((255, 120, 60), cv2.MARKER_TILTED_CROSS),
         ((90, 255, 90), cv2.MARKER_DIAMOND)]
# (clip, rep) windows: the reps whose Vision peaks diverge from YOLO's most.
DEFAULT_REPS = ["deadlift4:2", "deadlift4:3", "squat2:2", "squat2:5", "squat2:1", "squat3:4"]
VIDEO_H = 1000


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--reps", nargs="+", default=DEFAULT_REPS, metavar="CLIP:REP")
    p.add_argument("--half-window", type=int, default=22, help="frames each side of the peak")
    p.add_argument("--min-conf", type=float, default=0.1)
    p.add_argument("--out-fps", type=float, default=10.0, help="10 = 3x slow from 30 fps")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cands", nargs=3, default=None, metavar="NAME=RUN:RULE",
                   help="three candidates, e.g. yolo=fps30_yolo26s_640:ignore "
                        "offset05=fps30_vision_640:offset_c0.5 (rule = file under <run>/rules/)")
    p.add_argument("--out", default="_wrist_review", help="output dir under sweep/")
    p.add_argument("--wrist-conf", type=float, default=None,
                   help="threshold for drawing Vision wrists as lost (default --min-conf)")
    a = p.parse_args()
    global CANDS
    if a.cands:
        CANDS = [(c.split("=")[0], c.split("=")[1].split(":")[0], c.split("=")[1].split(":")[1])
                 for c in a.cands]
    if a.wrist_conf is not None:
        a.min_conf = a.wrist_conf
    out_dir = os.path.join(a.sweep, a.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(a.seed)

    index, clips = [], []
    for n, spec in enumerate(a.reps, 1):
        v, rep_no = spec.split(":")[0], int(spec.split(":")[1])
        meta = json.load(open(os.path.join(a.sweep, v, "fps30", "meta.json")))
        tracks = {name: json.load(open(os.path.join(a.sweep, v, run, "rules", f"{rule}.json")))
                  for name, run, rule in CANDS}
        # Window: peak velocity of this rep under Vision offset.
        reps, sig = analyse(tracks[CANDS[-1][0]], 30.0, meta["source_fps"],
                            meta["frame_size"][1], scale_constants=True)
        rep = reps[rep_no - 1]
        vel = np.array(sig["v_h_s"])
        peak = rep["start_frame"] + int(np.argmax(vel[rep["start_frame"]:rep["end_frame"] + 1]))
        kp = json.load(open(os.path.join(a.sweep, v, "fps30_vision_640", "keypoints.json")))
        src_w, src_h = meta["frame_size"]
        sx, sy = src_w / kp["frame_size"][0], src_h / kp["frame_size"][1]

        frames_dir = os.path.join(out_dir, "_frames")
        shutil.rmtree(frames_dir, ignore_errors=True)
        extract_frames(os.path.join(ROOT, "videos", f"{v}.mov"), frames_dir, fps=30)
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        lo, hi = max(0, peak - a.half_window), min(len(files), peak + a.half_window + 1)

        order = list(range(len(CANDS)))
        rng.shuffle(order)
        letters = {CANDS[k][0]: "ABC"[i] for i, k in enumerate(order)}
        style = {CANDS[k][0]: STYLE[i] for i, k in enumerate(order)}

        scale = VIDEO_H / src_h
        size = (int(src_w * scale) // 2 * 2, VIDEO_H)
        raw = os.path.join(out_dir, f"_{v}_rep{rep_no}.mp4")
        name = f"{n:02d}_{v}_rep{rep_no}.mp4"
        wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps, size)
        for i in range(lo, hi):
            img = cv2.resize(cv2.imread(os.path.join(frames_dir, files[i])), size)
            band = img[0:130].copy()
            img[0:130] = cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.75, 0)
            cv2.putText(img, f"({n}) {v} rep {rep_no}   frame {i}", (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            # Selected Vision person = the one whose visible wrists sit nearest
            # the offset candidate.
            c = tracks[CANDS[-1][0]][i]["center"]
            best = None
            for person in kp["frames"][i]["people"]:
                ws = [(person["kp"][j][0] * sx, person["kp"][j][1] * sy, person["kp"][j][2]) for j in (9, 10)]
                vis = [w for w in ws if w[2] >= a.min_conf]
                if not vis or c is None:
                    continue
                d = min((w[0] - c[0]) ** 2 + (w[1] - c[1]) ** 2 for w in vis)
                if best is None or d < best[0]:
                    best = (d, ws)
            status = []
            if best is not None:
                for side, w in zip("LR", best[1]):
                    if w[2] >= a.min_conf:
                        pt = (int(w[0] * scale), int(w[1] * scale))
                        cv2.circle(img, pt, 9, (255, 255, 255), -1)
                        cv2.circle(img, pt, 9, (0, 0, 0), 2)
                        cv2.putText(img, f"{side} {w[2]:.2f}", (pt[0] + 12, pt[1] + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        status.append(f"{side} lost")
            cv2.putText(img, "Vision wrists: " + (", ".join(status) if status else "both seen"),
                        (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (80, 80, 255) if status else (200, 200, 200), 2, cv2.LINE_AA)
            legend = "   ".join(f"{letters[k]}" for k in sorted(letters, key=letters.get))
            cv2.putText(img, f"midpoints: {legend}", (14, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2, cv2.LINE_AA)
            for cname, _, _ in CANDS:
                col, mk = style[cname]
                trail = [tracks[cname][j]["center"] for j in range(max(lo, i - 6), i + 1)]
                trail = [(int(t[0] * scale), int(t[1] * scale)) for t in trail if t is not None]
                for p0, p1 in zip(trail, trail[1:]):
                    cv2.line(img, p0, p1, col, 2, cv2.LINE_AA)
                cur = tracks[cname][i]["center"]
                if cur is not None:
                    pt = (int(cur[0] * scale), int(cur[1] * scale))
                    cv2.drawMarker(img, pt, col, mk, 26, 3, cv2.LINE_AA)
                    cv2.putText(img, letters[cname], (pt[0] - 30 + 30 * "ABC".index(letters[cname]), pt[1] - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)
            wr.write(img)
        wr.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-r", str(a.out_fps), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                        os.path.join(out_dir, name)], check=True)
        os.remove(raw)
        clips.append(name)
        index.append({"n": n, "video": v, "rep": rep_no, "clip": name, "frames": [lo, hi],
                      "peak_frame": peak, "letters": letters, "judgment": None})
        print(f"  ({n}) {name}: frames {lo}-{hi}, peak {peak}")
    shutil.rmtree(os.path.join(out_dir, "_frames"), ignore_errors=True)
    json.dump(index, open(os.path.join(out_dir, "review.json"), "w"), indent=2)
    lst = os.path.join(out_dir, "_c.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.join(out_dir, c)}'\n" for c in clips)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                    os.path.join(out_dir, "all_review.mp4")], check=True)
    os.remove(lst)


if __name__ == "__main__":
    main()
