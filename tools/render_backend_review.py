"""render_backend_review.py — skeleton + graphs review video for one pose backend.

For eyeballing a candidate backend the way the app would show it. Each clip
plays at normal speed (30 fps, the frames the model saw), with:
  - the selected person's skeleton (torso selection, as the app does), joints
    under --min-conf hidden, and the bar point with a short trail
  - a side panel: bar height and velocity over the whole clip, the backend's
    detected reps shaded, the YOLO reference's bar height faint for
    comparison, and a cursor at the current frame
Bar point and reps use the app's rule (torso selection, offset at 0.5).

    python tools/render_backend_review.py --run fps30_movenet_thunder_640 \\
        --clip "26-7-6 bench press 1" --clip deadlift4
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
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "yolo_pose")]
from compare_wrist_rules import build, select_by_torso, torso_point, wrist_states  # noqa: E402
from sweep_configs import analyse  # noqa: E402
from utils import extract_frames  # noqa: E402

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
VIDEO_H = 960
PANEL_W = 600
REP_FILL = (70, 110, 40)


def selected_people(kp, src, run_scale):
    sx, sy = run_scale
    out, last = [], None
    for fr in kp["frames"]:
        p = select_by_torso(fr["people"], last, sx, sy)
        if p is not None:
            last = torso_point(p, sx, sy)
        out.append(p)
    return out


def graph(panel, y0, h, series, ref, reps, cur, n, title, unit):
    """One time-series plot in panel[y0:y0+h], whole clip on the x axis."""
    w = panel.shape[1]
    pad_l, pad_r, pad_t, pad_b = 46, 12, 28, 22
    ok = [v for v in series + (ref or []) if v is not None and np.isfinite(v)]
    if not ok:
        return
    lo, hi = min(ok), max(ok)
    rng = (hi - lo) or 1.0

    def px(i):
        return int(pad_l + i / max(1, n - 1) * (w - pad_l - pad_r))

    def py(v):
        return int(y0 + pad_t + (1 - (v - lo) / rng) * (h - pad_t - pad_b))

    for r in reps:
        cv2.rectangle(panel, (px(r["start_frame"]), y0 + pad_t), (px(r["end_frame"]), y0 + h - pad_b),
                      REP_FILL, -1)
    cv2.putText(panel, title, (8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, unit, (8, y0 + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    if lo < 0 < hi:
        cv2.line(panel, (pad_l, py(0)), (w - pad_r, py(0)), (90, 90, 90), 1)
    for s, col, th in ((ref, (120, 120, 120), 1), (series, (255, 255, 255), 2)):
        if not s:
            continue
        pts = [(px(i), py(v)) for i, v in enumerate(s) if v is not None and np.isfinite(v)]
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] <= 6:
                cv2.line(panel, a, b, col, th, cv2.LINE_AA)
    cv2.line(panel, (px(cur), y0 + pad_t), (px(cur), y0 + h - pad_b), (0, 200, 255), 2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--run", required=True)
    p.add_argument("--ref-run", default="fps30_yolo26s_640")
    p.add_argument("--clip", action="append", required=True)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--label", default=None, help="name shown on the video (default: the run)")
    p.add_argument("--out", default=None, help="output dir (default sweep/_review_<run>)")
    a = p.parse_args()
    label = a.label or a.run
    out_dir = a.out or os.path.join(a.sweep, f"_review_{a.run}")
    os.makedirs(out_dir, exist_ok=True)
    made = []

    for clip in a.clip:
        meta = json.load(open(os.path.join(a.sweep, clip, "fps30", "meta.json")))
        src = meta["frame_size"]
        kp = json.load(open(os.path.join(a.sweep, clip, a.run, "keypoints.json")))
        sx, sy = src[0] / kp["frame_size"][0], src[1] / kp["frame_size"][1]
        track = build(wrist_states(kp, src, a.min_conf, select="torso"), "offset")
        reps, sig = analyse(track, 30.0, meta["source_fps"], src[1], scale_constants=True)
        people = selected_people(kp, src, (sx, sy))
        ref_reps = []
        ref_pos = None
        ref_kp_path = os.path.join(a.sweep, clip, a.ref_run, "keypoints.json")
        if os.path.exists(ref_kp_path):
            rkp = json.load(open(ref_kp_path))
            rtrack = build(wrist_states(rkp, src, 0.0, select="torso"), "ignore")
            ref_reps, rsig = analyse(rtrack, 30.0, meta["source_fps"], src[1], scale_constants=True)
            ref_pos = [v * src[1] for v in rsig["y_h"]]
        pos = [v * src[1] for v in sig["y_h"]]
        vel = [v * src[1] for v in sig["v_h_s"]]

        frames_dir = os.path.join(out_dir, "_frames")
        shutil.rmtree(frames_dir, ignore_errors=True)
        extract_frames(os.path.join(ROOT, "videos", f"{clip}.mov"), frames_dir, fps=30)
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        n = min(len(files), len(track))
        scale = VIDEO_H / src[1]
        vw = int(src[0] * scale) // 2 * 2
        size = (vw + PANEL_W, VIDEO_H)
        raw = os.path.join(out_dir, "_raw.mp4")
        wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
        trail = []
        for i in range(n):
            img = cv2.resize(cv2.imread(os.path.join(frames_dir, files[i])), (vw, VIDEO_H))
            person = people[i]
            if person is not None:
                k = [(q[0] * sx * scale, q[1] * sy * scale, q[2]) for q in person["kp"]]
                for e0, e1 in EDGES:
                    if k[e0][2] >= a.min_conf and k[e1][2] >= a.min_conf:
                        cv2.line(img, (int(k[e0][0]), int(k[e0][1])), (int(k[e1][0]), int(k[e1][1])),
                                 (80, 255, 80), 3, cv2.LINE_AA)
                for j, q in enumerate(k):
                    if q[2] >= a.min_conf:
                        cv2.circle(img, (int(q[0]), int(q[1])), 7 if j in (9, 10) else 4,
                                   (0, 0, 255) if j in (9, 10) else (255, 255, 0), -1)
            c = track[i]["center"]
            trail = (trail + [None if c is None else (int(c[0] * scale), int(c[1] * scale))])[-10:]
            pts = [t for t in trail if t is not None]
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(img, p0, p1, (0, 200, 255), 3, cv2.LINE_AA)
            if c is not None:
                cv2.drawMarker(img, (int(c[0] * scale), int(c[1] * scale)), (0, 200, 255),
                               cv2.MARKER_CROSS, 30, 3)
            band = img[0:84].copy()
            img[0:84] = cv2.addWeighted(band, 0.3, np.zeros_like(band), 0.7, 0)
            in_rep = next((r for r in reps if r["start_frame"] <= i <= r["end_frame"]), None)
            cv2.putText(img, f"{clip}   {label}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"t={i / 30:5.2f}s  " + (f"REP {reps.index(in_rep) + 1}" if in_rep else "")
                        + ("" if c is not None else "   no bar point"),
                        (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (80, 255, 80) if in_rep else ((80, 80, 255) if c is None else (200, 200, 200)),
                        2, cv2.LINE_AA)
            panel = np.full((VIDEO_H, PANEL_W, 3), 22, dtype=np.uint8)
            cv2.putText(panel, f"{len(reps)} reps detected (shaded); grey = YOLO reference "
                        f"({len(ref_reps)} reps)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            graph(panel, 34, 450, pos, ref_pos, reps, i, n, "bar height (up = higher)", "px")
            graph(panel, 500, 440, vel, None, reps, i, n, "bar velocity", "px/s")
            wr.write(np.hstack([img, panel]))
        wr.release()
        name = clip.replace(" ", "_") + ".mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "25", "-movflags", "+faststart", os.path.join(out_dir, name)], check=True)
        os.remove(raw)
        shutil.rmtree(frames_dir, ignore_errors=True)
        made.append(name)
        print(f"  {name}: {len(reps)} reps " + " ".join(f"{r['start_s']:.1f}-{r['end_s']:.1f}" for r in reps))

    lst = os.path.join(out_dir, "_c.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.join(out_dir, m)}'\n" for m in made)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart",
                    os.path.join(out_dir, "all_clips.mp4")], check=True)
    os.remove(lst)
    print(f"-> {out_dir}/")


if __name__ == "__main__":
    main()
