#!/usr/bin/env python3
"""barbell_velocity.py — SAM2-based barbell plate tracking and velocity estimation."""

from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import os
import shutil
import sys
import tempfile
from typing import Optional

import cv2
import numpy as np
import torch

from utils import compute_velocity, remove_outliers, smooth_positions

MODEL_IDS: dict[str, str] = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "large": "facebook/sam2-hiera-large",
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[mask] = (
        out[mask].astype(np.float32) * 0.45
        + np.array([0, 200, 0], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    return out


def get_box_from_user(frame_path: str, max_display: int = 900) -> list[float]:
    img = cv2.imread(frame_path)
    if img is None:
        sys.exit(f"[ERROR] Could not read frame: {frame_path}")

    orig_h, orig_w = img.shape[:2]
    scale = min(max_display / orig_w, max_display / orig_h, 1.0)
    display = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    roi = cv2.selectROI(
        "Draw box around plate — ENTER/SPACE to confirm", display, fromCenter=False
    )
    cv2.destroyAllWindows()
    x, y, w, h = (v / scale for v in roi)
    return [x, y, x + w, y + h]


def get_point_from_user(frame_path: str, max_display: int = 900) -> list[float]:
    img = cv2.imread(frame_path)
    if img is None:
        sys.exit(f"[ERROR] Could not read frame: {frame_path}")

    orig_h, orig_w = img.shape[:2]
    scale = min(max_display / orig_w, max_display / orig_h, 1.0)
    display = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    clicked: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))
            preview = display.copy()
            cv2.circle(preview, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(preview, (x, y), 8, (0, 0, 0), 2)
            cv2.imshow(win, preview)

    win = "Click plate center — ENTER to confirm"
    cv2.imshow(win, display)
    cv2.setMouseCallback(win, on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not clicked:
        sys.exit("[ERROR] No point selected.")
    px, py = clicked[0]
    return [px / scale, py / scale]


def _clip_mask_to_box(mask: np.ndarray, box: list[float]) -> np.ndarray:
    """Zero out mask pixels outside [x0,y0,x1,y1] box (original image coords)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    clipped = np.zeros_like(mask)
    clipped[max(y0, 0):y1, max(x0, 0):x1] = mask[max(y0, 0):y1, max(x0, 0):x1]
    return clipped


def refine_prompt_with_preview(
    frame_path: str,
    predictor,
    state,
    initial_prompt_kwargs: dict,
    frame_idx: int = 0,
    max_display: int = 900,
) -> None:
    """Apply initial_prompt_kwargs on frame_idx, show mask overlay, allow refinement.

    L-click=foreground point, R-click=background point, B=draw bounding box,
    Z=undo last point, C=toggle box clipping, R=reset, Enter/Space/ESC=accept.
    Mutates state in-place via add_new_points_or_box.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return

    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display / orig_w, max_display / orig_h, 1.0)
    win = f"Frame {frame_idx} — L/R-click=fg/bg  B=box  Z=undo  C=clip  R=reset  ENTER=accept"

    points_acc: list[list[float]] = []
    labels_acc: list[int] = []
    current_mask: list[np.ndarray] = [np.zeros((orig_h, orig_w), dtype=bool)]

    # Box-draw mode state (all mutable via single-element list for closure access)
    box_mode: list[bool] = [False]
    box_start: list[tuple[int, int] | None] = [None]

    # Clip box: when set, mask is trimmed to this region after every SAM2 call.
    # Initialised from the initial prompt if it has a box.
    _init_box = initial_prompt_kwargs.get("box")
    clip_box: list[list[float] | None] = [
        list(map(float, _init_box)) if _init_box is not None else None
    ]
    clip_enabled: list[bool] = [clip_box[0] is not None]

    def _run_prompt(kwargs: dict) -> np.ndarray:
        """Single SAM2 call; applies clip if active."""
        _, _, logits = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=frame_idx, obj_id=1, **kwargs
        )
        mask = (logits[0, 0] > 0).cpu().numpy()
        if clip_enabled[0] and clip_box[0] is not None:
            mask = _clip_mask_to_box(mask, clip_box[0])
        return mask

    def reapply_all() -> np.ndarray:
        """Rebuild the full prompt in one SAM2 call: initial prompt + all accumulated points."""
        kwargs: dict = dict(initial_prompt_kwargs)  # shallow copy — never mutate the original
        if points_acc:
            # Merge any initial points with accumulated clicks
            init_pts = kwargs.pop("points", None)
            init_lbls = kwargs.pop("labels", None)
            all_pts: list = (init_pts.tolist() if init_pts is not None else []) + points_acc
            all_lbls: list = (init_lbls.tolist() if init_lbls is not None else []) + labels_acc
            kwargs["points"] = np.array(all_pts, dtype=np.float32)
            kwargs["labels"] = np.array(all_lbls, dtype=np.int32)
        # clear_old_points defaults to True — full reset on every call
        return _run_prompt(kwargs)

    current_mask[0] = reapply_all()
    n_px = int(current_mask[0].sum())
    print(f"[INFO] Frame {frame_idx} initial mask: {n_px} pixels")
    if n_px == 0:
        print("[WARN] Mask is empty — try a different prompt or check the SAM2 _C extension.")

    def redraw(live_box: tuple[int, int, int, int] | None = None) -> None:
        small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
        small_mask = cv2.resize(
            current_mask[0].astype(np.uint8),
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        out = _render_mask_overlay(small, small_mask)
        if small_mask.sum() == 0:
            cv2.putText(out, "No mask — try a different prompt",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (px, py), lbl in zip(points_acc, labels_acc):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(out, (int(px * scale), int(py * scale)), 6, color, -1)
            cv2.circle(out, (int(px * scale), int(py * scale)), 6, (0, 0, 0), 1)
        # Draw the active clip box
        if clip_box[0] is not None:
            bx0, by0, bx1, by1 = (int(round(v * scale)) for v in clip_box[0])
            color = (0, 200, 255) if clip_enabled[0] else (100, 100, 100)
            cv2.rectangle(out, (bx0, by0), (bx1, by1), color, 1)
            label_txt = "clip ON" if clip_enabled[0] else "clip OFF"
            cv2.putText(out, label_txt, (bx0 + 4, by0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if box_mode[0]:
            cv2.putText(out, "BOX — click and drag, release to apply",
                        (10, small.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if live_box is not None:
            x0, y0, x1, y1 = live_box
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.imshow(win, out)

    def on_click(event, x, y, flags, param):
        if box_mode[0]:
            if event == cv2.EVENT_LBUTTONDOWN:
                box_start[0] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and box_start[0] is not None:
                x0 = min(box_start[0][0], x)
                y0 = min(box_start[0][1], y)
                x1 = max(box_start[0][0], x)
                y1 = max(box_start[0][1], y)
                redraw((x0, y0, x1, y1))
            elif event == cv2.EVENT_LBUTTONUP and box_start[0] is not None:
                x0 = min(box_start[0][0], x)
                y0 = min(box_start[0][1], y)
                x1 = max(box_start[0][0], x)
                y1 = max(box_start[0][1], y)
                box_start[0] = None
                box_mode[0] = False
                if x1 - x0 > 5 and y1 - y0 > 5:
                    orig_box = [x0 / scale, y0 / scale, x1 / scale, y1 / scale]
                    new_kwargs: dict = {
                        "box": np.array(orig_box, dtype=np.float32),
                        "normalize_coords": False,
                    }
                    points_acc.clear()
                    labels_acc.clear()
                    clip_box[0] = orig_box
                    clip_enabled[0] = True
                    # Replace initial prompt so R resets to this box
                    initial_prompt_kwargs.clear()
                    initial_prompt_kwargs.update(new_kwargs)
                    current_mask[0] = reapply_all()
                redraw()
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                label = 1
            elif event == cv2.EVENT_RBUTTONDOWN:
                label = 0
            else:
                return
            orig_x, orig_y = x / scale, y / scale
            points_acc.append([orig_x, orig_y])
            labels_acc.append(label)
            current_mask[0] = reapply_all()
            redraw()

    redraw()
    cv2.setMouseCallback(win, on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32, 27):  # Enter, Space, ESC
            break
        if key in (ord('r'), ord('R')):
            points_acc.clear()
            labels_acc.clear()
            box_mode[0] = False
            box_start[0] = None
            current_mask[0] = reapply_all()
            redraw()
        if key in (ord('b'), ord('B')):
            box_mode[0] = not box_mode[0]
            box_start[0] = None
            redraw()
        if key in (ord('z'), ord('Z')):
            if points_acc:
                points_acc.pop()
                labels_acc.pop()
                print(f"[INFO] Undo — {len(points_acc)} point(s) remaining.")
                current_mask[0] = reapply_all()
                redraw()
            else:
                print("[INFO] Undo — nothing to undo.")
        if key in (ord('c'), ord('C')):
            if clip_box[0] is not None:
                clip_enabled[0] = not clip_enabled[0]
                current_mask[0] = reapply_all()
                redraw()

    cv2.destroyWindow(win)


def interactive_recovery_prompt(
    frame_path: str,
    predictor,
    state,
    frame_idx: int,
    max_display: int = 900,
) -> bool:
    """Show the lost-tracking frame; let user choose a new prompt. Returns True if re-prompted."""
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[WARN] Could not load frame {frame_idx} for recovery UI.")
        return False

    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display / orig_w, max_display / orig_h, 1.0)
    small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    dimmed = (small.astype(np.float32) * 0.5).astype(np.uint8)
    cv2.putText(dimmed, f"Tracking lost at frame {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(dimmed, "B=draw box  P=click point  S=skip",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    win = "Tracking lost — re-prompt?"
    cv2.imshow(win, dimmed)

    mode = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('s'), ord('S'), 27):
            cv2.destroyWindow(win)
            return False
        if key in (ord('b'), ord('B')):
            mode = 'b'
            break
        if key in (ord('p'), ord('P')):
            mode = 'p'
            break

    cv2.destroyWindow(win)

    if mode == 'b':
        box = get_box_from_user(frame_path, max_display)
        prompt_kwargs: dict = {"box": np.array(box, dtype=np.float32), "normalize_coords": False}
    else:
        pt = get_point_from_user(frame_path, max_display)
        prompt_kwargs = {
            "points": np.array([pt], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
            "normalize_coords": False,
        }

    print(f"[INFO] Recovery prompt at frame {frame_idx}: mode={mode}")
    refine_prompt_with_preview(
        frame_path, predictor, state, prompt_kwargs,
        frame_idx=frame_idx, max_display=max_display,
    )
    return True


# ---------------------------------------------------------------------------
# Core tracking
# ---------------------------------------------------------------------------

def run_sam2_tracking(
    frames_dir: str,
    n_frames: int,
    predictor,
    state,
    recovery_callback=None,
    lost_threshold: int = 5,
) -> dict[int, np.ndarray]:
    """Propagate SAM2 through all frames, with optional interactive recovery on loss.

    Assumes caller holds torch.inference_mode() and autocast context.
    The initial prompt must already be applied to state before calling.
    """
    masks: dict[int, np.ndarray] = {}
    consecutive_lost = 0
    first_lost_frame: int | None = None
    start_frame = 0
    recovery_attempts: dict[int, int] = {}
    total_recoveries = 0
    MAX_RECOVERIES = 10

    while True:
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=start_frame
        ):
            binary = (mask_logits[0, 0] > 0).cpu().numpy()
            masks[int(frame_idx)] = binary

            if binary.sum() == 0:
                if first_lost_frame is None:
                    first_lost_frame = int(frame_idx)
                consecutive_lost += 1
                if (
                    recovery_callback is not None
                    and consecutive_lost >= lost_threshold
                    and total_recoveries < MAX_RECOVERIES
                    and recovery_attempts.get(first_lost_frame, 0) < 2
                ):
                    break  # trigger recovery
            else:
                consecutive_lost = 0
                first_lost_frame = None
        else:
            break  # propagation completed without triggering recovery

        # Recovery path: re-prompt at the first lost frame of this run
        recover_at = first_lost_frame if first_lost_frame is not None else int(frame_idx)
        recovery_attempts[recover_at] = recovery_attempts.get(recover_at, 0) + 1

        # ffmpeg names frames 000001.jpg … so frame_idx N → file N+1
        frame_path = os.path.join(frames_dir, f"{recover_at + 1:06d}.jpg")
        print(f"[INFO] Recovery triggered at frame {recover_at} ({consecutive_lost} lost frames).")
        recovered = recovery_callback(frame_path, predictor, state, recover_at)

        if not recovered:
            print("[INFO] Recovery skipped — continuing without re-prompt.")
            break

        consecutive_lost = 0
        first_lost_frame = None
        start_frame = recover_at
        total_recoveries += 1

    return masks


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def extract_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _extract_frames_for_sam2(
    video_path: str,
    frames_dir: str,
    max_seconds: float | None = None,
    frame_step: int = 1,
) -> None:
    """Extract frames named as bare integers (000001.jpg) as required by SAM2's loader."""
    import ffmpeg
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, "%06d.jpg").replace("\\", "/")

    input_kwargs = {"t": max_seconds} if max_seconds else {}
    stream = ffmpeg.input(video_path, **input_kwargs)

    if frame_step > 1:
        stream = stream.filter("select", f"not(mod(n,{frame_step}))")
        stream = stream.output(output_pattern, qscale=2, vsync="0")
    else:
        stream = stream.output(output_pattern, qscale=2)

    stream.run(quiet=True)
    print(f"[INFO] Frames extracted to: {frames_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 barbell plate velocity estimator")
    parser.add_argument("--video", default=None, help="Path to input video file (opens file picker if omitted)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: sam2_results/<video_stem>/velocities.csv)")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS fallback if not readable from file")
    parser.add_argument("--plate-mm", type=float, default=450.0, dest="plate_mm",
                        help="Known plate diameter in mm (default: 450)")
    parser.add_argument("--model", choices=["tiny", "small", "large"], default="small",
                        help="SAM2 model variant (default: small)")
    parser.add_argument("--box", default=None,
                        help='Bounding box prompt as "x_min,y_min,x_max,y_max" (non-interactive)')
    parser.add_argument("--point", default=None,
                        help='Single foreground point as "x,y" (non-interactive)')
    parser.add_argument("--debug-box", default=None, dest="debug_box",
                        help="Save frame 0 with prompt drawn to this path, then exit")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                        help="Only process the first N seconds of the video")
    parser.add_argument("--frame-step", type=int, default=1, dest="frame_step",
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--no-preview", action="store_true", dest="no_preview",
                        help="Skip the interactive mask preview after the initial prompt")
    parser.add_argument("--no-recovery", action="store_true", dest="no_recovery",
                        help="Disable interactive recovery when tracking is lost")
    parser.add_argument("--lost-threshold", type=int, default=5, dest="lost_threshold",
                        help="Consecutive empty frames before triggering recovery (default: 5)")
    parser.add_argument("--offload-to-cpu", action="store_true", dest="offload_to_cpu",
                        help="Keep frame features and inference state in CPU RAM instead of VRAM")
    parser.add_argument("--num-maskmem", type=int, default=7, dest="num_maskmem",
                        help="SAM2 memory bank size — past frames held in VRAM during propagation "
                             "(default: 7, reduce to 3-4 to cut VRAM use)")
    args = parser.parse_args()

    if args.video is None:
        from utils import pick_video_file
        args.video = pick_video_file("Select a video for SAM2 analysis")
    print(f"[INFO] Video: {args.video}")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join("sam2_results", stem, "velocities.csv")

    if not os.path.exists(args.video):
        sys.exit(f"[ERROR] Video file not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = video_fps if video_fps > 0 else args.fps
    effective_fps = fps / args.frame_step
    print(f"[INFO] Using FPS: {fps:.2f}" + (
        f" -> effective {effective_fps:.2f} (frame-step {args.frame_step})"
        if args.frame_step > 1 else ""
    ))

    frames_dir = tempfile.mkdtemp(prefix="barbell_frames_")
    try:
        # 1. Frame extraction
        print("[INFO] Extracting frames...")
        _extract_frames_for_sam2(
            args.video, frames_dir,
            max_seconds=args.max_seconds,
            frame_step=args.frame_step,
        )

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frame_files:
            sys.exit("[ERROR] No frames extracted from video.")
        n_frames = len(frame_files)
        print(f"[INFO] {n_frames} frames extracted.")

        frame0_path = os.path.join(frames_dir, frame_files[0])

        # 2. Collect initial prompt
        box: list[float] | None = None
        point: list[float] | None = None

        if args.box is not None:
            box = [float(v) for v in args.box.split(",")]
            print(f"[INFO] Using box: {box}")
        elif args.point is not None:
            point = [float(v) for v in args.point.split(",")]
            print(f"[INFO] Using point: {point}")
        else:
            print("[INFO] No --box or --point given — opening interactive selector.")
            print("[INFO]   b = draw bounding box   p = click a point")
            mode = input("Mode [b/p, default=p]: ").strip().lower() or "p"
            if mode == "b":
                box = get_box_from_user(frame0_path)
                print(f"[INFO] Box: {box}")
            else:
                point = get_point_from_user(frame0_path)
                print(f"[INFO] Point: {point}")

        if args.debug_box:
            frame0 = cv2.imread(frame0_path)
            if box is not None:
                x0, y0, x1, y1 = (int(v) for v in box)
                cv2.rectangle(frame0, (x0, y0), (x1, y1), (0, 255, 0), 3)
            if point is not None:
                cv2.circle(frame0, (int(point[0]), int(point[1])), 12, (0, 255, 0), -1)
            cv2.imwrite(args.debug_box, frame0)
            print(f"[INFO] Debug frame saved to {args.debug_box} — verify prompt, then re-run without --debug-box")
            return

        prompt_kwargs: dict = {"normalize_coords": False}
        if box is not None:
            prompt_kwargs["box"] = np.array(box, dtype=np.float32)
        if point is not None:
            prompt_kwargs["points"] = np.array([point], dtype=np.float32)
            prompt_kwargs["labels"] = np.array([1], dtype=np.int32)

        # 3. Load SAM2 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = MODEL_IDS[args.model]
        print(f"[INFO] Loading SAM2 ({args.model}) on {device}...")
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sys.exit("[ERROR] CUDA out of memory loading model. Try --model tiny.")
            raise

        if args.num_maskmem != 7:
            predictor.num_maskmem = args.num_maskmem
            print(f"[INFO] Memory bank size set to {args.num_maskmem} frames.")

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if device == "cuda"
            else contextlib.nullcontext()
        )

        use_recovery = not args.no_recovery

        # 4. Tracking (init_state, optional preview, propagation)
        try:
            with torch.inference_mode(), autocast_ctx:
                state = predictor.init_state(
                    video_path=frames_dir,
                    offload_video_to_cpu=args.offload_to_cpu,
                    offload_state_to_cpu=args.offload_to_cpu,
                    async_loading_frames=args.offload_to_cpu,
                )

                if not args.no_preview:
                    # Apply initial prompt and show live preview for refinement
                    print("[INFO] Showing mask preview — refine if needed, then press ENTER.")
                    refine_prompt_with_preview(frame0_path, predictor, state, prompt_kwargs)
                else:
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=1, **prompt_kwargs
                    )

                print("[INFO] Running SAM2 propagation...")
                recovery_cb = None
                if use_recovery:
                    def recovery_cb(fp, p, s, fi):
                        return interactive_recovery_prompt(fp, p, s, fi)

                masks = run_sam2_tracking(
                    frames_dir, n_frames, predictor, state,
                    recovery_callback=recovery_cb,
                    lost_threshold=args.lost_threshold,
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sys.exit("[ERROR] CUDA out of memory. Try --model tiny, --offload-to-cpu, "
                         "--num-maskmem 3, or --frame-step 2.")
            raise
        finally:
            # Release SAM2 state (holds frame features + memory bank in VRAM/RAM)
            try:
                predictor.reset_state(state)
            except Exception:
                pass
            if device == "cuda":
                torch.cuda.empty_cache()

        # 5. Centroid extraction
        centroids: dict[int, tuple[float, float]] = {}
        lost_runs: list[tuple[int, int]] = []
        _run_start: int | None = None

        for idx in range(n_frames):
            mask = masks.get(idx)
            c = extract_centroid(mask) if mask is not None else None
            if c is None:
                if _run_start is None:
                    _run_start = idx
            else:
                if _run_start is not None:
                    lost_runs.append((_run_start, idx - 1))
                    _run_start = None
                centroids[idx] = c
        if _run_start is not None:
            lost_runs.append((_run_start, n_frames - 1))

        frames_lost = sum(e - s + 1 for s, e in lost_runs)
        if lost_runs:
            parts = [f"{s}-{e}" if s != e else str(s) for s, e in lost_runs]
            print(f"[WARN] Lost tracking for {frames_lost} frame(s) in {len(lost_runs)} run(s): {', '.join(parts)}")

        # 6. Scale calibration — use the first tracked frame in the first 30 frames
        calib_frame = next(
            (i for i in range(min(30, n_frames))
             if i in masks and extract_centroid(masks[i]) is not None),
            None,
        )
        if calib_frame is None:
            sys.exit("[ERROR] No valid mask in first 30 frames — cannot calibrate scale. "
                     "Adjust prompt or check the SAM2 _C extension.")
        if calib_frame > 0:
            print(f"[WARN] Frame 0 mask empty; using frame {calib_frame} for scale calibration.")
        ys_0, xs_0 = np.where(masks[calib_frame])
        h = float(ys_0.max() - ys_0.min())
        w = float(xs_0.max() - xs_0.min())
        pixel_diameter = (h + w) / 2
        px_per_mm = pixel_diameter / args.plate_mm
        print(f"[INFO] Scale: {pixel_diameter:.1f} px diameter (h={h:.0f} w={w:.0f}) -> {px_per_mm:.4f} px/mm")

        # 7. Build position arrays (Y-flipped: upward = positive)
        cx_list: list[Optional[float]] = []
        cy_list: list[Optional[float]] = []
        positions: list[Optional[float]] = []

        for idx in range(n_frames):
            if idx in centroids:
                cx, cy = centroids[idx]
                cx_list.append(cx)
                cy_list.append(cy)
                positions.append(-cy)
            else:
                cx_list.append(None)
                cy_list.append(None)
                positions.append(None)

        # 8. Smooth positions and compute velocity
        outlier_threshold_px = 50.0 * px_per_mm
        clean_pos = [
            None if (v != v) else v
            for v in remove_outliers(positions, threshold=outlier_threshold_px)
        ]
        smooth_pos = smooth_positions(clean_pos)
        raw_vel_pxs = compute_velocity(clean_pos, effective_fps)
        smooth_vel_pxs = compute_velocity(smooth_pos, effective_fps)

        scale_pxm = px_per_mm * 1000
        velocity_ms = [v / scale_pxm for v in raw_vel_pxs]
        smoothed_velocity_ms = [v / scale_pxm for v in smooth_vel_pxs]

        # 9. CSV output
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp_s", "cx_px", "cy_px",
                             "velocity_ms", "smoothed_velocity_ms"])
            for idx in range(n_frames):
                cx = cx_list[idx]
                cy = cy_list[idx]
                writer.writerow([
                    idx,
                    f"{idx / effective_fps:.4f}",
                    f"{cx:.2f}" if cx is not None else "",
                    f"{cy:.2f}" if cy is not None else "",
                    f"{velocity_ms[idx]:.4f}",
                    f"{smoothed_velocity_ms[idx]:.4f}",
                ])

        # 10. Summary
        peak_vel = max((abs(v) for v in smoothed_velocity_ms), default=0.0)
        concentric = [v for v in smoothed_velocity_ms if v > 0.1]
        mean_concentric = float(np.mean(concentric)) if concentric else 0.0

        print(f"\n[SUMMARY]")
        print(f"  Frames tracked:           {n_frames - frames_lost} / {n_frames}")
        print(f"  Frames lost:              {frames_lost}")
        print(f"  Peak velocity:            {peak_vel:.3f} m/s")
        print(f"  Mean concentric velocity: {mean_concentric:.3f} m/s")
        print(f"  Output:                   {args.output}")

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print("[INFO] Cleaned up temp frames.")


if __name__ == "__main__":
    main()
