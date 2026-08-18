# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Estimates barbell velocity from workout videos using computer vision. The investigation evaluated several approaches; the primary working pipeline uses YOLO Pose wrist keypoints as a barbell proxy with Savitzky-Golay smoothing and PELT-based concentric phase detection. Two segmentation-based pipelines (SAM2 and YOLO-seg) were also built; both track the barbell plate directly and output calibrated m/s velocity.

Pipeline-specific commands, CLI arguments, and tuning details live in each pipeline's own `CLAUDE.md`: `yolo_pose/CLAUDE.md`, `sam2/CLAUDE.md`, `yolo_seg/CLAUDE.md`, `tools/CLAUDE.md`.

## Shared utilities

**`utils.py`** — Shared across all pipelines:
- `extract_frames(video_path, frames_dir, fps=None)` — extracts JPEG frames via ffmpeg
- `frames_to_video(frames_dir, output_video_path, fps=30)` — reconstructs video from frames
- `remove_outliers(positions, window=5, threshold=50)` — median-based outlier removal
- `smooth_positions(positions, window=9, poly=3)` — Savitzky-Golay smoothing
- `compute_velocity(positions, fps)` — finite-difference velocity
- `pick_video_file(title)` — opens a file picker dialog (used when `--video` is omitted)
- `get_video_fps(video_path)` — reads FPS via OpenCV

## Key data formats

- **Keypoint JSON** (`pose_results/<video>/<model>/keypoints/frame_XXXXXX.json`): `{ "frame": "...", "keypoints": [...], "boxes": [...], "frame_size": [w, h] }` — COCO 17-keypoint format.
- **Barbell track JSON** (`pose_results/<video>/<model>/barbell_track.json`): `[{ "frame": "...", "center": [x, y] }, ...]` — `center` is `null` when no person was detected.
- **Meta JSON** (`pose_results/<video>/<model>/meta.json`): `{ "fps": ..., "source_fps": ..., "video": "..." }` — written by `pose_velocity.py`; read by `analysis.py` to avoid hardcoding FPS. `fps` is the effective inference FPS (matches `--fps` if passed, else `source_fps`); `source_fps` is always the original video's native FPS.
- **Velocity CSV** (`velocities.csv`): `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms` — used by SAM2 and YOLO-seg pipelines. YOLO-pose outputs px/s via plots rather than CSV.

## Dependencies

**SAM2 pipeline:** `torch`, `torchvision`, `transformers` (HuggingFace), `sam2`, `opencv-python`, `numpy`, `scipy`, `ffmpeg-python`

**YOLO pipelines:** `ultralytics`, `opencv-python`, `ffmpeg-python`, `numpy`, `scipy`, `matplotlib`, `ruptures` (for concentric detection)

## Notable design decisions

- Y-axis is flipped (`-cy`) throughout all pipelines so that upward movement is positive velocity.
- Velocity math (`remove_outliers`, `smooth_positions`, `compute_velocity`) lives in `utils.py` and is shared by all three pipelines.
- YOLO Pose pipeline outputs px/s. The segmentation pipelines (SAM2, YOLO-seg) derive `px/mm` from the plate mask on the calibration frame and output m/s directly.
- PELT changepoint detection runs on the **velocity signal** (first difference of smoothed position), not on raw position. Applying PELT to position over-segments rising trends; differentiating first collapses a clean concentric phase into a near-zero-mean segment that PELT finds as one coherent piece.
- Velocity-ratio filtering in PELT uses the **median** of merged-candidate velocities as the reference (not the max), so a single fast noise burst cannot inflate the rejection threshold.
- `pose_velocity.py --fps` (must be positive) downsamples inference frame rate (via `ffmpeg -r`) to test how low FPS can go before tracking degrades. When set, the output directory gets an `_fps<value>` suffix (e.g. `yolo26_large_fps15`) so sweeps don't overwrite each other. `analysis.py`'s frame-count constants (smoothing windows, PELT `min_frames`/`extension_lookahead`) are tuned as frame counts against each video's own native FPS and rescaled by `inference_fps / source_fps`, so a native-FPS run reproduces the old behavior exactly and only a downsampled `--fps` run rescales them; the PELT `pen` penalty is not auto-scaled and may need manual retuning at very different FPS. Low-FPS runs should also be visually spot-checked in the annotated video, since `build_barbell_track.py`'s nearest-neighbor wrist-proximity lock has no distance guard and could swap targets if the wrist jump between frames grows large.
- Each pipeline script inserts the project root into `sys.path` at startup so `utils` is always importable regardless of CWD.
- `pose_velocity.py` defaults to keeping intermediate frames and keypoint JSONs; pass `--clean` to delete them after video reconstruction.
