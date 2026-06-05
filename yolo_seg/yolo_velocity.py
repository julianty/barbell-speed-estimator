"""
yolo_velocity.py — YOLOv8-seg barbell plate velocity estimator

Parallel to barbell_velocity.py (SAM2 pipeline). Replaces SAM2 propagation
with YOLOv8 instance segmentation + BotSORT tracking. Everything downstream
(outlier removal, smoothing, velocity, CSV format) is identical; plot_velocity.py
and overlay.py work unchanged.

Usage:
    python yolo_seg/yolo_velocity.py --video videos/deadlift.mov
    python yolo_seg/yolo_velocity.py --video videos/deadlift.mov --model small --no-preview
"""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import os
import shutil
import sys
import tempfile
from typing import Optional

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from utils import compute_velocity, remove_outliers, smooth_positions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CODES = {"nano": "n", "small": "s", "medium": "m"}

MASK_COLORS = [
    (0, 120, 255),
    (0, 200, 80),
    (255, 80, 0),
    (220, 0, 220),
    (0, 200, 200),
    (255, 200, 0),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _extract_frames(
    video_path: str,
    frames_dir: str,
    frame_step: int = 1,
    max_seconds: float | None = None,
) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, "frame_%06d.jpg").replace("\\", "/")
    input_kwargs: dict = {}
    if max_seconds:
        input_kwargs["t"] = max_seconds
    stream = ffmpeg.input(video_path, **input_kwargs)
    if frame_step > 1:
        stream = stream.filter("select", f"not(mod(n,{frame_step}))")
        stream = stream.output(output_pattern, qscale=2, vsync="0")
    else:
        stream = stream.output(output_pattern, qscale=2)
    stream.run(quiet=True)
    print(f"[INFO] Frames extracted to: {frames_dir}")


def _get_masks_and_ids(
    result, frame_hw: tuple[int, int]
) -> list[tuple[int, np.ndarray]]:
    """Return [(track_id, binary_mask_HW), ...] from one tracking result."""
    if result.masks is None:
        return []
    h, w = frame_hw
    raw = result.masks.data.cpu().numpy()  # (N, H_mask, W_mask)
    ids = (
        result.boxes.id.int().tolist()
        if result.boxes.id is not None and len(result.boxes.id) > 0
        else list(range(1, len(raw) + 1))
    )
    out = []
    for tid, m in zip(ids, raw):
        resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((int(tid), resized > 0.5))
    return out


def _render_selection_frame(
    frame_bgr: np.ndarray,
    mask_list: list[tuple[int, np.ndarray]],
    selected_id: Optional[int],
    frame_idx: int = 0,
    n_frames: int = 1,
) -> np.ndarray:
    vis = frame_bgr.copy()
    for i, (tid, binary) in enumerate(mask_list):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        alpha = 0.75 if selected_id == tid else 0.45
        overlay = vis.copy()
        overlay[binary] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        ys, xs = np.where(binary)
        if len(ys):
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            label = f"ID:{tid}" + (" <<" if selected_id == tid else "")
            cv2.putText(
                vis, label, (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )

    if not mask_list:
        cv2.putText(
            vis, "No objects detected on this frame",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA,
        )
        hints = "N = next frame    ESC / Q = quit"
    else:
        hints = "Click plate    ENTER = confirm    N = next frame    ESC / Q = quit"

    cv2.putText(
        vis, f"Frame {frame_idx}/{n_frames - 1}  |  {hints}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return vis


def _select_target_interactive(
    frame_bgr: np.ndarray,
    mask_list: list[tuple[int, np.ndarray]],
    frame_idx: int = 0,
    n_frames: int = 1,
) -> "int | str | None":
    """Show detected masks; user clicks plate.

    Returns:
        int   — chosen track ID (confirmed)
        'next' — user pressed N to advance to the next frame
        None  — user quit (Esc or Q)
    """
    state: dict = {"selected_id": mask_list[0][0] if mask_list else None}

    def on_click(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for tid, binary in mask_list:
            if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1] and binary[y, x]:
                state["selected_id"] = tid
                vis = _render_selection_frame(frame_bgr, mask_list, state["selected_id"], frame_idx, n_frames)
                cv2.imshow(win_name, vis)
                break

    win_name = "Select barbell plate"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_click)

    while True:
        vis = _render_selection_frame(frame_bgr, mask_list, state["selected_id"], frame_idx, n_frames)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32) and mask_list:  # Enter or Space → confirm (only if something detected)
            cv2.destroyWindow(win_name)
            return state["selected_id"]
        if key in (ord("n"), ord("N")):     # N → try next frame
            cv2.destroyWindow(win_name)
            return "next"
        if key in (27, ord("q"), ord("Q")):  # Esc or Q → quit
            cv2.destroyWindow(win_name)
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-seg barbell plate velocity estimator")
    parser.add_argument("--video", default=None, help="Path to input video file (opens file picker if omitted)")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: yolo_results/<video_stem>/velocities.csv)",
    )
    parser.add_argument("--fps", type=float, default=30.0,
                        help="FPS fallback if not readable from file")
    parser.add_argument("--plate-mm", type=float, default=450.0, dest="plate_mm",
                        help="Known plate diameter in mm (default: 450)")
    parser.add_argument("--model", choices=list(MODEL_CODES), default="nano",
                        help="YOLOv8-seg variant: nano/small/medium (default: nano)")
    parser.add_argument("--frame-step", type=int, default=1, dest="frame_step",
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                        help="Only process the first N seconds of the video")
    parser.add_argument("--no-preview", action="store_true", dest="no_preview",
                        help="Skip interactive plate selection; auto-pick largest mask on frame 0")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    args = parser.parse_args()

    if args.video is None:
        from utils import pick_video_file
        args.video = pick_video_file("Select a video for YOLO-seg analysis")
    print(f"[INFO] Video: {args.video}")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join("yolo_results", stem, "velocities.csv")

    if not os.path.exists(args.video):
        sys.exit(f"[ERROR] Video file not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = video_fps if video_fps > 0 else args.fps
    effective_fps = fps / args.frame_step
    print(f"[INFO] Video FPS: {fps:.2f}" + (
        f" -> effective {effective_fps:.2f} (frame-step {args.frame_step})"
        if args.frame_step > 1 else ""
    ))

    from ultralytics import YOLO

    model_file = os.path.join("models", f"yolov8{MODEL_CODES[args.model]}-seg.pt")
    print(f"[INFO] Loading {model_file}...")
    model = YOLO(model_file)

    frames_dir = tempfile.mkdtemp(prefix="yolo_frames_")
    try:
        # 1. Frame extraction
        print("[INFO] Extracting frames...")
        _extract_frames(
            args.video, frames_dir,
            frame_step=args.frame_step,
            max_seconds=args.max_seconds,
        )

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frame_files:
            sys.exit("[ERROR] No frames extracted from video.")
        n_frames = len(frame_files)
        print(f"[INFO] {n_frames} frames extracted.")

        frame0_bgr = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if frame0_bgr is None:
            sys.exit("[ERROR] Could not read frame 0 — JPEG may be corrupted.")
        h, w = frame0_bgr.shape[:2]

        tracker_gen = model.track(
            source=frames_dir,
            task="segment",
            conf=args.conf,
            tracker="botsort.yaml",
            stream=True,
            verbose=False,
            persist=True,
        )

        # 2. Plate selection — advance through frames until user confirms or quits
        if args.no_preview:
            result0 = next(tracker_gen)
            mask_list0 = _get_masks_and_ids(result0, (h, w))
            if not mask_list0:
                sys.exit(
                    "[ERROR] No objects detected on frame 0. "
                    "Try lowering --conf or check that the video has a visible barbell plate."
                )
            target_id = max(mask_list0, key=lambda x: int(x[1].sum()))[0]
            sel_frame_idx = 0
            sel_mask_list = mask_list0
            print(f"[INFO] Auto-selected largest mask: track ID {target_id}")
        else:
            sel_frame_idx = 0
            target_id = None
            sel_mask_list = []

            for sel_frame_idx, tracking_result in enumerate(tracker_gen):
                frame_bgr_sel = cv2.imread(os.path.join(frames_dir, frame_files[sel_frame_idx]))
                if frame_bgr_sel is None:
                    frame_bgr_sel = frame0_bgr
                mask_list_sel = _get_masks_and_ids(tracking_result, (h, w))
                n_det = len(mask_list_sel)
                print(f"[INFO] Frame {sel_frame_idx}: {n_det} object(s) detected.")

                action = _select_target_interactive(
                    frame_bgr_sel, mask_list_sel, sel_frame_idx, n_frames
                )

                if action is None:
                    sys.exit("[INFO] Selection cancelled. Exiting.")
                if action == "next":
                    if sel_frame_idx >= n_frames - 1:
                        sys.exit("[ERROR] Reached last frame with no selection made.")
                    continue
                # confirmed
                target_id = action
                sel_mask_list = mask_list_sel
                break

        # Scale calibration from the confirmed frame's mask
        target_mask0 = next((m for tid, m in sel_mask_list if tid == target_id), None)
        if target_mask0 is None:
            sys.exit("[ERROR] Selected mask not found.")
        ys0, xs0 = np.where(target_mask0)
        if ys0.size == 0:
            sys.exit(
                "[ERROR] Selected mask has no active pixels after thresholding — "
                "try a different frame (N) or lower --conf."
            )
        plate_h_px = float(ys0.max() - ys0.min())
        plate_w_px = float(xs0.max() - xs0.min())
        pixel_diameter = (plate_h_px + plate_w_px) / 2
        px_per_mm = pixel_diameter / args.plate_mm
        print(
            f"[INFO] Selected track ID {target_id} on frame {sel_frame_idx}. "
            f"Scale: {pixel_diameter:.1f} px diameter -> {px_per_mm:.4f} px/mm"
        )

        # 3. Full-video tracking — frames before sel_frame_idx are gaps; continue from sel_frame_idx+1
        masks: dict[int, np.ndarray] = {sel_frame_idx: target_mask0}

        gap_start: int | None = None
        gap_runs: list[tuple[int, int]] = []
        total_lost = 0

        remaining = n_frames - 1 - sel_frame_idx
        with tqdm(total=remaining, desc="Tracking", unit="frame") as pbar:
            for frame_idx, result in enumerate(tracker_gen, start=sel_frame_idx + 1):
                mask_list = _get_masks_and_ids(result, (h, w))
                found = next((m for tid, m in mask_list if tid == target_id), None)

                if found is not None:
                    masks[frame_idx] = found
                    if gap_start is not None:
                        gap_runs.append((gap_start, frame_idx - 1))
                        gap_start = None
                else:
                    total_lost += 1
                    if gap_start is None:
                        gap_start = frame_idx

                pbar.update(1)
                pbar.set_postfix(lost=total_lost, refresh=False)

        if gap_start is not None:
            gap_runs.append((gap_start, n_frames - 1))

        if gap_runs:
            for a, b in gap_runs:
                print(f"[WARN] Track lost frames {a}–{b} ({b - a + 1} frame(s))")

        # 4. Centroid extraction
        centroids: dict[int, tuple[float, float]] = {}
        for idx in range(n_frames):
            mask = masks.get(idx)
            c = extract_centroid(mask) if mask is not None else None
            if c is not None:
                centroids[idx] = c

        print(f"[INFO] Centroids found in {len(centroids)}/{n_frames} frames.")

        # 5. Build position arrays (Y-flipped: upward = positive)
        positions: list[Optional[float]] = []
        cx_list: list[Optional[float]] = []
        cy_list: list[Optional[float]] = []
        for idx in range(n_frames):
            if idx in centroids:
                cx, cy = centroids[idx]
                positions.append(-cy)
                cx_list.append(cx)
                cy_list.append(cy)
            else:
                positions.append(None)
                cx_list.append(None)
                cy_list.append(None)

        # 6. Outlier removal, smoothing, velocity
        outlier_threshold_px = 50.0 * px_per_mm
        clean_pos = remove_outliers(positions, threshold=outlier_threshold_px)
        clean_pos = [None if (v != v) else v for v in clean_pos]  # nan → None

        smooth_pos = smooth_positions(clean_pos)
        raw_vel_pxs = compute_velocity(clean_pos, effective_fps)
        smooth_vel_pxs = compute_velocity(smooth_pos, effective_fps)

        scale_pxm = px_per_mm * 1000
        velocity_ms = [v / scale_pxm for v in raw_vel_pxs]
        smoothed_velocity_ms = [v / scale_pxm for v in smooth_vel_pxs]

        # 7. CSV output
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp_s", "cx_px", "cy_px",
                             "velocity_ms", "smoothed_velocity_ms"])
            for idx in range(n_frames):
                writer.writerow([
                    idx,
                    f"{idx / effective_fps:.4f}",
                    f"{cx_list[idx]:.2f}" if cx_list[idx] is not None else "",
                    f"{cy_list[idx]:.2f}" if cy_list[idx] is not None else "",
                    f"{velocity_ms[idx]:.4f}",
                    f"{smoothed_velocity_ms[idx]:.4f}",
                ])

        print(f"[INFO] Done. Results written to: {args.output}")

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
