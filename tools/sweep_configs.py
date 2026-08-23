"""Stage-1 FPS sweep for the analysis-speed investigation.

Runs the YOLO-pose pipeline over an fps ladder and, at each fps, analyses the
resulting track *twice* — once with the analysis constants rescaled by
inference_fps/source_fps ("normalized", what `analysis.py` does today) and once
with them left as raw frame counts ("fixed", what the Flutter app's
`AnalysisConfig` does today). The delta between those two arms is blocking
finding A in `barbell-velocity-app/LEDGER.md`, and it is why
every fps is analysed twice instead of once.

Resolution is deliberately NOT swept: Stage 0 showed model time is flat against
input resolution because the graph's input socket is a fixed 640x640 (decision
D1 in the ledger). FPS is the only lever left that scales every term at once.

Inference is run once per (video, fps) and shared by both analysis arms — the
arms differ only in post-processing, so re-inferring would just burn time.

Writes, per run:
  sweep/<video>/fps<N>/barbell_track.json   the raw track
  sweep/<video>/fps<N>/meta.json            fps, source_fps, frame size
  sweep/<video>/fps<N>/timing.json          Tier-3 cost numbers
  sweep/<video>/fps<N>/<arm>/reps.json      per-rep metrics, px and normalized
  sweep/<video>/fps<N>/<arm>/signals.json   per-frame position/velocity signals

`tools/compare_configs.py` consumes those; this script computes no comparisons.

Usage:
  python tools/sweep_configs.py                        # every video in videos/
  python tools/sweep_configs.py --video "videos/26-6-30 225lb bench press 1.mov"
  python tools/sweep_configs.py --fps-ladder 30 15     # native is always added
"""

import argparse
import glob
import json
import os
import resource
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "yolo_pose"))

from utils import extract_frames, get_video_fps, remove_outliers, smooth_positions, compute_velocity  # noqa: E402
from build_barbell_track import compute_barbell_center, MAX_JUMP_FRACTION  # noqa: E402
from detect_concentric import detect_concentric  # noqa: E402
import analysis as ana  # noqa: E402  (for the tuned constants + their scaling helpers)

# The ladder the ledger's Stage 1 calls for. Native fps is always run as the
# baseline every other config is scored against, so it is not listed here.
DEFAULT_FPS_LADDER = [30, 20, 15, 10]

# "normalized" = analysis.py today (constants rescaled by inference/source fps).
# "fixed"      = AnalysisConfig today (raw frame counts, whatever the fps).
ARMS = ("normalized", "fixed")

DEFAULT_MODEL = "yolo_pose/models/yolo26s-pose.pt"


def peak_rss_mb():
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def run_inference(model, frames_dir):
    """Returns (track, frame_size, timings). Mirrors run_pose_inference +
    build_barbell_track_from_keypoints, but in memory and without writing the
    per-frame JSONs or annotated images the sweep has no use for."""
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    track = []
    last_center = None
    rejected_jumps = 0
    frame_size = None
    read_s = 0.0
    model_s = 0.0

    for frame_file in frame_files:
        t = time.perf_counter()
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        read_s += time.perf_counter() - t

        t = time.perf_counter()
        results = model.predict(img, verbose=False)
        model_s += time.perf_counter() - t

        r = results[0]
        kp = r.keypoints.xy.tolist() if r.keypoints is not None else []
        boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
        h, w = r.orig_shape
        frame_size = (w, h)

        max_jump_dist = MAX_JUMP_FRACTION * ((w ** 2 + h ** 2) ** 0.5)
        center, jump_rejected = compute_barbell_center(
            kp, boxes=boxes, last_center=last_center, max_jump_dist=max_jump_dist
        )
        if jump_rejected:
            rejected_jumps += 1
        if center is not None:
            last_center = center

        track.append({"frame": frame_file, "center": center})

    timings = {
        "frames": len(frame_files),
        "read_s": round(read_s, 3),
        "model_s": round(model_s, 3),
        "rejected_jumps": rejected_jumps,
    }
    return track, frame_size, timings


def analyse(track, fps, source_fps, frame_h, scale_constants):
    """One analysis arm. `scale_constants=False` reproduces the app's current
    behaviour (raw frame counts at any fps); True reproduces analysis.py's."""
    fps_scale = (fps / source_fps) if scale_constants else 1.0

    raw_positions = [None if e["center"] is None else -e["center"][1] for e in track]
    detected = sum(p is not None for p in raw_positions)

    outlier_window = ana._frames(ana.OUTLIER_WINDOW_REF, fps_scale)
    clean = remove_outliers(raw_positions, window=outlier_window, threshold=50)

    poly = 3
    window = ana._odd_savgol_window(ana.SMOOTH_WINDOW_REF, fps_scale, len(clean))
    if window <= poly:
        poly = max(0, window - 1)
    smooth = smooth_positions(clean, window=window, poly=poly)

    smooth_vel = compute_velocity(smooth, fps)

    phases, segments = detect_concentric(
        smooth, method="ruptures", pen=20,
        min_frames=ana._frames(ana.MIN_FRAMES_REF, fps_scale),
        smooth_velocity=smooth_vel,
        merge_min_vel=ana.MERGE_MIN_VEL,
        extension_lookahead=ana._frames(ana.EXTENSION_LOOKAHEAD_REF, fps_scale),
        min_rep_ratio=ana.MIN_REP_RATIO,
        return_segments=True,
    )

    v = np.asarray(smooth_vel, dtype=float)
    reps = []
    for i, (start, end) in enumerate(phases):
        seg = v[start:end]
        if seg.size == 0:
            continue
        reps.append({
            "rep": i + 1,
            "start_frame": int(start),
            "end_frame": int(end),
            # Seconds, not frames: the only unit that compares across the ladder.
            "start_s": start / fps,
            "end_s": end / fps,
            "duration_s": (end - start) / fps,
            "avg_vel_px_s": float(seg.mean()),
            "peak_vel_px_s": float(seg.max()),
            "displacement_px": float(smooth[end] - smooth[start]),
            # Frame-height units, so a config at a different resolution or a
            # different clip framing is still comparable.
            "avg_vel_h_s": float(seg.mean()) / frame_h,
            "peak_vel_h_s": float(seg.max()) / frame_h,
            "displacement_h": float(smooth[end] - smooth[start]) / frame_h,
        })

    signals = {
        "fps": fps,
        "t_s": [i / fps for i in range(len(smooth))],
        "y_h": [p / frame_h for p in smooth],
        "v_h_s": [x / frame_h for x in smooth_vel],
        "detected_frames": detected,
        "total_frames": len(raw_positions),
        "detection_rate": detected / len(raw_positions) if raw_positions else 0.0,
        "sg_window": window,
        "sg_poly": poly,
        "outlier_window": outlier_window,
        "n_segments": len(segments),
    }
    return reps, signals


def run_is_complete(run_dir):
    """A run counts as done only with its timings AND both arms' reps on disk.
    Timings are checked because a run interrupted mid-inference can leave a
    track behind, and a resumed sweep must not inherit a half-written one."""
    needed = [os.path.join(run_dir, "timing.json"),
              os.path.join(run_dir, "barbell_track.json")]
    needed += [os.path.join(run_dir, arm, "reps.json") for arm in ARMS]
    return all(os.path.exists(p) for p in needed)


def sweep_video(video, model, fps_ladder, out_root, frames_root, force=False):
    name = os.path.splitext(os.path.basename(video))[0]
    source_fps = get_video_fps(video)
    print(f"\n=== {name}  (source {source_fps:.3f} fps) ===")

    # None = native: ffmpeg gets no -r, so it emits every frame. This is the
    # baseline run, and at native fps both arms are identical by construction
    # (fps_scale == 1.0) — which is itself the check that the arms agree where
    # they should.
    for fps in [None] + list(fps_ladder):
        label = "native" if fps is None else f"fps{fps:g}"
        effective_fps = source_fps if fps is None else float(fps)
        run_dir = os.path.join(out_root, name, label)
        if not force and run_is_complete(run_dir):
            print(f"  {label:>7}  already complete — skipped (--force to redo)")
            continue
        os.makedirs(run_dir, exist_ok=True)

        frames_dir = os.path.join(frames_root, f"{name}_{label}")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

        t0 = time.perf_counter()
        extract_frames(video, frames_dir, fps=fps)
        extract_s = time.perf_counter() - t0

        track, frame_size, timings = run_inference(model, frames_dir)
        shutil.rmtree(frames_dir)

        frame_w, frame_h = frame_size
        n = timings["frames"]

        with open(os.path.join(run_dir, "barbell_track.json"), "w") as f:
            json.dump(track, f)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({"fps": effective_fps, "source_fps": source_fps,
                       "video": video, "frame_size": [frame_w, frame_h],
                       "baseline": fps is None}, f, indent=2)

        timing = {
            "config": label,
            "fps": effective_fps,
            "extract_s": round(extract_s, 3),
            **timings,
            "total_s": round(extract_s + timings["read_s"] + timings["model_s"], 3),
            "model_ms_per_frame": round(1000 * timings["model_s"] / n, 2) if n else None,
            "peak_rss_mb": peak_rss_mb(),
        }
        with open(os.path.join(run_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

        rep_counts = {}
        for arm in ARMS:
            reps, signals = analyse(track, effective_fps, source_fps, frame_h,
                                    scale_constants=(arm == "normalized"))
            arm_dir = os.path.join(run_dir, arm)
            os.makedirs(arm_dir, exist_ok=True)
            with open(os.path.join(arm_dir, "reps.json"), "w") as f:
                json.dump(reps, f, indent=2)
            with open(os.path.join(arm_dir, "signals.json"), "w") as f:
                json.dump(signals, f)
            rep_counts[arm] = len(reps)

        det = signals["detection_rate"]
        print(f"  {label:>7}  {n:>5} frames  extract {extract_s:>6.2f}s  "
              f"infer {timings['model_s']:>7.2f}s  ({timing['model_ms_per_frame']:>6.2f} ms/f)  "
              f"det {det:>5.1%}  reps norm={rep_counts['normalized']} fixed={rep_counts['fixed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", action="append", default=None,
                        help="video to sweep; repeatable (default: every file in videos/)")
    parser.add_argument("--fps-ladder", type=float, nargs="+", default=DEFAULT_FPS_LADDER,
                        help=f"downsampled fps values (default: {DEFAULT_FPS_LADDER}); "
                             "the native-fps baseline is always run as well")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="weights; default matches the iOS .mlpackage's architecture "
                             "AND scale (yolo26s), not the pipeline's usual large")
    parser.add_argument("--out", default="sweep", help="output root (default: sweep/)")
    parser.add_argument("--frames-dir", default=None,
                        help="scratch dir for extracted frames (default: <out>/_frames)")
    parser.add_argument("--force", action="store_true",
                        help="re-run configs that already have complete output "
                             "(default: skip them, so an interrupted sweep resumes)")
    args = parser.parse_args()

    videos = args.video or sorted(glob.glob(os.path.join(ROOT, "videos", "*.mov")))
    if not videos:
        parser.error("no videos found")

    out_root = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    frames_root = args.frames_dir or os.path.join(out_root, "_frames")
    os.makedirs(frames_root, exist_ok=True)

    from ultralytics import YOLO
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    model = YOLO(model_path)

    print(f"model  : {args.model}")
    print(f"ladder : native + {args.fps_ladder}")
    print(f"videos : {len(videos)}")
    print(f"arms   : {', '.join(ARMS)}")

    t0 = time.perf_counter()
    for video in videos:
        sweep_video(video, model, args.fps_ladder, out_root, frames_root, force=args.force)
    shutil.rmtree(frames_root, ignore_errors=True)

    print(f"\nswept {len(videos)} video(s) in {time.perf_counter() - t0:.1f}s -> {out_root}/")
    print("next: python tools/compare_configs.py")


if __name__ == "__main__":
    main()
