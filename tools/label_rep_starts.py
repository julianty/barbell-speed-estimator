"""label_rep_starts.py — slow-motion clips for hand-labelling disputed rep starts.

Part of the pose-backend comparison (tools/compare_pose_backends.py). Rep
lockouts agree across YOLO, Vision and MoveNet to within a few frames, but rep
STARTS disagree by hundreds of ms, and YOLO's starts are known to be weak (app
LEDGER §6 items 7-8), so no model can referee. This cuts one clip per
disputed rep so a human can mark where the concentric phase really starts.

A rep is disputed when the candidate runs' starts for it spread by more than
--spread-ms. Reps that already have a truth_start_s in
sweep/_ground_truth_lifts/disputed_reps.json are skipped.

The candidates are BLINDED: each clip shows letters A, B, C... at the candidate
start times, sorted by time, so the labeller is never told which model is
which. Candidates within one 30 fps frame of each other share a letter. The
letter-to-run mapping is in labels.json, alongside empty truth_start_s fields.

Frames are the source's native rate (60 fps), so a label is finer than any
30 fps candidate. Every clip is written at a fixed 15 fps, one source frame per
output frame: 4x slow from 60 fps, 3.996x from 59.94. A rate derived from the
source (59.94 / 4 = 14.985) made the concatenated file play some clips fast
and then freeze, and it dropped the last clips.

Re-running keeps any truth_start_s already filled in labels.json, matched on
(video, rep). `--only video:rep` renders just those reps, for example to
re-send the ones a labeller missed. Clips are H.264 so they play
on a phone; `all_reps.mp4` is every clip back to back.

    python tools/label_rep_starts.py
    python tools/label_rep_starts.py --spread-ms 100 --runs fps30 fps30_vision
"""

import argparse
import json
import os
import shutil
import string
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from extract_rep_clips import smoothed  # noqa: E402

DEFAULT_RUNS = ["fps30", "fps30_vision", "fps30_movenet_thunder",
                "fps30_yolo26s_640", "fps30_vision_640", "fps30_movenet_thunder_640"]
DEFAULT_VIDEOS = [f"deadlift{i}" for i in range(1, 7)] + ["squat2", "squat3"]
VIDEO_H = 900
PANEL_W = 560
C_MARK = (60, 170, 255)
C_TXT = (255, 255, 255)


def best_overlap(rep, cands):
    best, ov_best = None, 0.0
    for c in cands:
        ov = min(rep["end_s"], c["end_s"]) - max(rep["start_s"], c["start_s"])
        if ov > ov_best:
            best, ov_best = c, ov
    return best


def letters_for(starts):
    """starts: {run: t}. Group runs within 1/30 s, letter the groups by time."""
    groups = []
    for run, t in sorted(starts.items(), key=lambda kv: kv[1]):
        if groups and t - groups[-1]["t"][-1] <= 1 / 30 + 1e-6:
            groups[-1]["runs"].append(run)
            groups[-1]["t"].append(t)
        else:
            groups.append({"runs": [run], "t": [t]})
    return [{"letter": string.ascii_uppercase[i], "t_s": float(np.mean(g["t"])), "runs": g["runs"]}
            for i, g in enumerate(groups)]


def render(frames_dir, pos, sf, lo, hi, marks, title, out_path, out_fps):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    first = cv2.imread(os.path.join(frames_dir, files[lo]))
    fh, fw = first.shape[:2]
    scale = VIDEO_H / fh
    vw, vh = int(round(fw * scale)) // 2 * 2, VIDEO_H
    panel_w = PANEL_W if fh > fw else 0
    size = (vw + panel_w, vh)
    raw_path = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, size)

    seg = pos[lo:hi]
    y0, y1 = float(np.nanmin(seg)), float(np.nanmax(seg))
    rng = (y1 - y0) or 1.0

    def px(i):
        return int((i - lo) / max(1, hi - lo - 1) * (panel_w - 1))

    def py(v):
        return int(vh - 40 - (v - y0) / rng * (vh - 120))

    mark_frames = [(m["letter"], int(round(m["t_s"] * sf))) for m in marks]
    for i in range(lo, hi):
        frame = cv2.resize(cv2.imread(os.path.join(frames_dir, files[i])), (vw, vh))
        band = frame[0:120, :].copy()
        frame[0:120, :] = cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.75, 0)
        cv2.putText(frame, title, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_TXT, 2, cv2.LINE_AA)
        cv2.putText(frame, f"t = {i / sf:.2f} s", (14, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 230, 255), 3, cv2.LINE_AA)
        # Flash the letter for a few frames as playback crosses a candidate.
        for letter, mf in mark_frames:
            if 0 <= i - mf < int(sf * 0.15):
                cv2.putText(frame, letter, (vw - 90, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                            C_MARK, 6, cv2.LINE_AA)
        if panel_w:
            panel = np.full((vh, panel_w, 3), 24, dtype=np.uint8)
            for row, (letter, mf) in enumerate(mark_frames):
                x = px(mf)
                for y in range(60, vh, 12):
                    cv2.line(panel, (x, y), (x, min(vh, y + 6)), C_MARK, 2)
                cv2.putText(panel, letter, (min(x + 5, panel_w - 30), 50 + (row % 2) * 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_MARK, 2, cv2.LINE_AA)
            pts = np.array([[px(lo + k), py(v)] for k, v in enumerate(seg)], dtype=np.int32)
            cv2.polylines(panel, [pts], False, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.line(panel, (px(i), 60), (px(i), vh), C_TXT, 1)
            cv2.circle(panel, (px(i), py(pos[i])), 7, C_TXT, -1)
            cv2.putText(panel, "wrist height, YOLO (up = higher)", (10, vh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
            frame = np.hstack([frame, panel])
        writer.write(frame)
    writer.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw_path, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", out_path],
                   check=True)
    os.remove(raw_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--out", default=None, help="default: <sweep>/_ground_truth_starts")
    p.add_argument("--video", action="append")
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="run dirs whose starts are candidates; the first is the rep list")
    p.add_argument("--arm", default="normalized")
    p.add_argument("--spread-ms", type=float, default=150.0)
    p.add_argument("--before-s", type=float, default=1.0)
    p.add_argument("--after-s", type=float, default=1.2)
    p.add_argument("--out-fps", type=float, default=15.0,
                   help="fixed playback rate for every clip (15 = 4x slow from 60 fps)")
    p.add_argument("--only", nargs="+", default=None, metavar="VIDEO:REP",
                   help="render only these reps, e.g. squat3:3 squat3:5")
    args = p.parse_args()
    out_dir = args.out or os.path.join(args.sweep, "_ground_truth_starts")
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, "_frames")

    gt = json.load(open(os.path.join(args.sweep, "_ground_truth_lifts", "disputed_reps.json")))
    known = {(g["video"], g["rep"]) for g in gt if g["truth_start_s"] is not None}
    labels_path = os.path.join(out_dir, "labels.json")
    previous = {}
    if os.path.exists(labels_path):
        previous = {(x["video"], x["rep"]): x for x in json.load(open(labels_path))}
    only = None if args.only is None else {(o.split(":")[0], int(o.split(":")[1])) for o in args.only}

    index, clips = [], []
    for v in args.video or DEFAULT_VIDEOS:
        runs = [r for r in args.runs if os.path.exists(os.path.join(args.sweep, v, r, args.arm, "reps.json"))]
        reps = {r: json.load(open(os.path.join(args.sweep, v, r, args.arm, "reps.json"))) for r in runs}
        ref = reps[runs[0]]
        todo = []
        for rep in ref:
            if (v, rep["rep"]) in known:
                continue
            starts = {}
            for r in runs:
                c = rep if r == runs[0] else best_overlap(rep, reps[r])
                if c is not None:
                    starts[r] = c["start_s"]
            spread = 1000 * (max(starts.values()) - min(starts.values()))
            if spread > args.spread_ms and (only is None or (v, rep["rep"]) in only):
                todo.append((rep, starts, spread))
        if not todo:
            continue

        meta = json.load(open(os.path.join(args.sweep, v, "native", "meta.json")))
        sf = meta["source_fps"]
        track = json.load(open(os.path.join(args.sweep, v, "native", "barbell_track.json")))
        pos = smoothed(track, sf, sf)
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(ROOT, meta["video"]),
                        "-qscale:v", "3", os.path.join(scratch, "frame_%06d.jpg")], check=True)
        n_frames = len(os.listdir(scratch))

        for rep, starts, spread in todo:
            marks = letters_for(starts)
            lo = max(0, int((min(starts.values()) - args.before_s) * sf))
            hi = min(n_frames, len(pos), int((max(starts.values()) + args.after_s) * sf))
            name = f"{v}_rep{rep['rep']}.mp4"
            title = f"{v}  rep {rep['rep']}   ({len(clips) + 1})"
            render(scratch, pos, sf, lo, hi, marks, title, os.path.join(out_dir, name), args.out_fps)
            clips.append(name)
            index.append({"video": v, "rep": rep["rep"], "clip": name, "source_fps": sf,
                          "window_s": [lo / sf, hi / sf], "spread_ms": round(spread),
                          "lockout_s": rep["end_s"], "candidates": marks,
                          "truth_start_s": None})
            old = previous.get((v, rep["rep"]))
            if old is not None:
                index[-1]["truth_start_s"] = old.get("truth_start_s")
                if "answer" in old:
                    index[-1]["answer"] = old["answer"]
            print(f"  {name}: spread {spread:.0f} ms, " +
                  ", ".join(f"{m['letter']}={m['t_s']:.2f}s" for m in marks))
    shutil.rmtree(scratch, ignore_errors=True)

    # A partial (--only) run must not drop the other reps' entries.
    merged = {(x["video"], x["rep"]): x for x in previous.values()}
    merged.update({(x["video"], x["rep"]): x for x in index})
    with open(labels_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2)
    lst = os.path.join(out_dir, "_concat.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.join(out_dir, c)}'\n" for c in clips)
    combined = "all_reps.mp4" if only is None else "selected_reps.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    "-movflags", "+faststart", os.path.join(out_dir, combined)], check=True)
    os.remove(lst)
    print(f"\n{len(clips)} clip(s) -> {out_dir}/ ({combined} is all of them)")


if __name__ == "__main__":
    main()
