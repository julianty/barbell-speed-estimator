"""overlay.py — Overlay centroid and velocity from velocities.csv onto video frames."""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import os
import sys
import tempfile
import shutil

import cv2
import numpy as np

from utils import extract_frames, frames_to_video


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def draw_frame(img: np.ndarray, row: dict) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    cx = float(row["cx_px"]) if row["cx_px"] else None
    cy = float(row["cy_px"]) if row["cy_px"] else None
    vel = float(row["smoothed_velocity_ms"]) if row["smoothed_velocity_ms"] else None
    ts = float(row["timestamp_s"])
    frame_idx = int(row["frame"])

    # Centroid dot + crosshair
    if cx is not None and cy is not None:
        cx_i, cy_i = int(cx), int(cy)
        cv2.circle(out, (cx_i, cy_i), 12, (0, 255, 0), 2)
        cv2.line(out, (cx_i - 20, cy_i), (cx_i + 20, cy_i), (0, 255, 0), 1)
        cv2.line(out, (cx_i, cy_i - 20), (cx_i, cy_i + 20), (0, 255, 0), 1)

        if vel is not None:
            color = (0, 255, 0) if vel > 0.05 else (0, 200, 255)
            label = f"{vel:+.3f} m/s"
            cv2.putText(out, label, (cx_i + 15, cy_i - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, (cx_i + 15, cy_i - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    else:
        # No centroid — show a "lost" indicator
        cv2.putText(out, "tracking lost", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, "tracking lost", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2, cv2.LINE_AA)

    # Timestamp + frame counter (top-left)
    info = f"frame {frame_idx}  t={ts:.2f}s"
    cv2.putText(out, info, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, info, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay centroid/velocity on video frames")
    parser.add_argument("--video", default=None, help="Original video file (opens file picker if omitted)")
    parser.add_argument("--csv", default="velocities.csv", help="Velocities CSV from barbell_velocity.py")
    parser.add_argument("--output", default="overlay.mp4", help="Output annotated video path")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS fallback")
    parser.add_argument("--frame-step", type=int, default=1, dest="frame_step",
                        help="Must match the --frame-step used during inference")
    args = parser.parse_args()

    if args.video is None:
        from utils import pick_video_file
        args.video = pick_video_file("Select a video for overlay")
    print(f"[INFO] Video: {args.video}")

    if not os.path.exists(args.video):
        sys.exit(f"[ERROR] Video not found: {args.video}")
    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] CSV not found: {args.csv}")

    rows = load_csv(args.csv)
    if not rows:
        sys.exit("[ERROR] CSV is empty.")

    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = video_fps if video_fps > 0 else args.fps
    effective_fps = fps / args.frame_step

    frames_dir = tempfile.mkdtemp(prefix="overlay_frames_")
    annotated_dir = tempfile.mkdtemp(prefix="overlay_annotated_")
    try:
        print("[INFO] Extracting frames...")
        extract_frames(args.video, frames_dir, fps=effective_fps if args.frame_step > 1 else None)

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        print(f"[INFO] {len(frame_files)} frames extracted, {len(rows)} CSV rows.")

        n = min(len(frame_files), len(rows))
        for i in range(n):
            src = os.path.join(frames_dir, frame_files[i])
            img = cv2.imread(src)
            if img is None:
                continue
            annotated = draw_frame(img, rows[i])
            dst = os.path.join(annotated_dir, frame_files[i])
            cv2.imwrite(dst, annotated)

        print(f"[INFO] Annotated {n} frames. Building video...")
        frames_to_video(annotated_dir, args.output, fps=effective_fps)
        print(f"[INFO] Saved to {args.output}")

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(annotated_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
