# rep_speed_history.py — aggregate rep times across pose_results runs and
# plot progress over time (per week), one scatter plot per exercise.
#
# Video/folder naming convention this relies on (see videos/, pose_results/):
#   "<YY-M-D> <weight>lb <exercise words> <order>"
#   e.g. "26-6-24 225lb bench press 1" — trailing number is the order the
#   exercise was performed within that day's workout, not a set count.
#
# Plots sec/rep rather than px/s velocity: pixel distance isn't calibrated to
# a consistent real-world scale across recording sessions, but rep duration
# is calibration-free (it only depends on fps, which meta.json records exactly).

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import json
import argparse
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    load_track, extract_positions, _frames, _odd_savgol_window,
    OUTLIER_WINDOW_REF, SMOOTH_WINDOW_REF, MIN_FRAMES_REF, EXTENSION_LOOKAHEAD_REF,
    MERGE_MIN_VEL, MIN_REP_RATIO,
)
from detect_concentric import detect_concentric
from utils import compute_velocity, remove_outliers, smooth_positions

NAME_RE = re.compile(
    r"^(?P<date>\d{1,2}-\d{1,2}-\d{1,2})\s+(?P<weight>\d+)lb\s+(?P<exercise>.+?)\s+(?P<set>\d+)$"
)


def parse_video_folder(name):
    """Parse "26-6-24 225lb bench press 1" -> (date, weight_lb, exercise, order)."""
    m = NAME_RE.match(name)
    if not m:
        return None
    yy, mm, dd = (int(p) for p in m.group("date").split("-"))
    date = dt.date(2000 + yy, mm, dd)
    return {
        "date": date,
        "weight_lb": int(m.group("weight")),
        "exercise": m.group("exercise").strip().lower(),
        "order": int(m.group("set")),
    }


def pick_canonical_run(video_dir):
    """Prefer the native-FPS run dir (no _fps suffix); otherwise the highest --fps run."""
    run_dirs = [d for d in glob.glob(os.path.join(video_dir, "*")) if os.path.isdir(d)]
    native = [d for d in run_dirs if not re.search(r"_fps\d+$", os.path.basename(d))]
    if native:
        return max(native, key=os.path.getmtime)
    fps_runs = []
    for d in run_dirs:
        m = re.search(r"_fps(\d+)$", os.path.basename(d))
        if m:
            fps_runs.append((int(m.group(1)), d))
    if not fps_runs:
        return None
    return max(fps_runs)[1]


def compute_rep_speeds(track_dir):
    """Re-run the clean/smooth/PELT pipeline (mirrors analysis.py's CLI logic)
    for a single pose_results/<video>/<model> directory and return per-rep speeds."""
    track_path = os.path.join(track_dir, "barbell_track.json")
    meta_path = os.path.join(track_dir, "meta.json")
    if not (os.path.exists(track_path) and os.path.exists(meta_path)):
        return []

    with open(meta_path) as f:
        meta = json.load(f)
    fps = meta["fps"]
    source_fps = meta.get("source_fps", fps)
    fps_scale = fps / source_fps

    track = load_track(track_path)
    raw_positions = extract_positions(track)

    outlier_window = _frames(OUTLIER_WINDOW_REF, fps_scale)
    clean_positions = remove_outliers(raw_positions, window=outlier_window, threshold=50)

    smooth_poly = 3
    smooth_window = _odd_savgol_window(SMOOTH_WINDOW_REF, fps_scale, len(clean_positions))
    if smooth_window <= smooth_poly:
        smooth_poly = max(0, smooth_window - 1)
    smooth_pos = smooth_positions(clean_positions, window=smooth_window, poly=smooth_poly)

    smooth_velocity = compute_velocity(smooth_pos, fps)

    pen = 20  # matches analysis.py's PELT penalty; not fps-scaled
    min_frames = _frames(MIN_FRAMES_REF, fps_scale)
    extension_lookahead = _frames(EXTENSION_LOOKAHEAD_REF, fps_scale)
    ruptures_phases, _ = detect_concentric(
        smooth_pos, method="ruptures", pen=pen,
        min_frames=min_frames,
        smooth_velocity=smooth_velocity,
        merge_min_vel=MERGE_MIN_VEL,
        extension_lookahead=extension_lookahead,
        min_rep_ratio=MIN_REP_RATIO,
        return_segments=True,
    )

    reps = []
    for i, (start, end) in enumerate(ruptures_phases):
        reps.append({
            "rep": i + 1,
            "avg_vel_px_s": float(np.mean(smooth_velocity[start:end])),
            "duration_s": (end - start) / fps,
        })
    return reps


def collect_rep_rows(pose_results_dir):
    rows = []
    for video_dir in sorted(glob.glob(os.path.join(pose_results_dir, "*"))):
        if not os.path.isdir(video_dir) or os.path.basename(video_dir).startswith("_"):
            continue
        parsed = parse_video_folder(os.path.basename(video_dir))
        if parsed is None:
            print(f"[WARN] Skipping unrecognized folder name: {os.path.basename(video_dir)}")
            continue
        run_dir = pick_canonical_run(video_dir)
        if run_dir is None:
            print(f"[WARN] No runs found under: {video_dir}")
            continue
        reps = compute_rep_speeds(run_dir)
        if not reps:
            print(f"[WARN] No reps detected for: {run_dir}")
            continue
        for r in reps:
            rows.append({**parsed, **r, "video": os.path.basename(video_dir)})
    return rows


def _week_key(date):
    iso_year, iso_week, _ = date.isocalendar()
    return (iso_year, iso_week)


def plot_exercise_history(exercise, rows, out_path):
    rows = sorted(rows, key=lambda r: (r["date"], r["order"], r["rep"]))
    durations = [r["duration_s"] for r in rows]

    unique_weeks = sorted({_week_key(r["date"]) for r in rows})
    week_index = {wk: i for i, wk in enumerate(unique_weeks)}
    week_labels = [dt.date.fromisocalendar(y, w, 1).strftime("%b %d") for y, w in unique_weeks]

    # Deterministic horizontal jitter so same-week reps don't stack on top of each other.
    by_week = {}
    for r in rows:
        by_week.setdefault(_week_key(r["date"]), []).append(r)

    xs = np.empty(len(rows))
    pos = 0
    row_to_x = {}
    for wk, wk_rows in by_week.items():
        n = len(wk_rows)
        jitter = np.linspace(-0.25, 0.25, n) if n > 1 else np.array([0.0])
        for r, j in zip(wk_rows, jitter):
            row_to_x[id(r)] = week_index[wk] + j
    xs = np.array([row_to_x[id(r)] for r in rows])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, durations, color="orange", alpha=0.8, edgecolor="white", linewidth=0.5)

    if len(unique_weeks) >= 2:
        week_x = np.array([week_index[_week_key(r["date"])] for r in rows])
        slope, intercept = np.polyfit(week_x, durations, 1)
        x_fit = np.array([0, len(unique_weeks) - 1])
        ax.plot(x_fit, slope * x_fit + intercept,
                color="limegreen", linestyle="--", label=f"trend: {slope:+.3f} s/rep per week")
        ax.legend()

    ax.set_title(f"{exercise.title()} — Concentric Rep Time Over Time ({len(rows)} reps)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Rep Time (s)")
    ax.set_xticks(range(len(unique_weeks)))
    ax.set_xticklabels(week_labels, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(unique_weeks) - 0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {out_path} ({len(rows)} reps, {len(unique_weeks)} weeks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate rep speeds across pose_results and plot progress over time")
    parser.add_argument("--pose-results", default=os.path.join(os.path.dirname(__file__), "pose_results"),
                         help="Path to pose_results/ directory (default: yolo_pose/pose_results)")
    parser.add_argument("--out-dir", default=None,
                         help="Directory for output plots (default: <pose_results>/_history)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.pose_results, "_history")

    rows = collect_rep_rows(args.pose_results)
    if not rows:
        sys.exit("[ERROR] No rep data collected — check that pose_results/ contains barbell_track.json runs.")

    by_exercise = {}
    for r in rows:
        by_exercise.setdefault(r["exercise"], []).append(r)

    for exercise, ex_rows in by_exercise.items():
        safe_name = re.sub(r"\s+", "_", exercise)
        out_path = os.path.join(out_dir, f"{safe_name}_speed_history.png")
        plot_exercise_history(exercise, ex_rows, out_path)
