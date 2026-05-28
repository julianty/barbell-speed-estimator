# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Estimates barbell velocity from workout videos. It runs YOLOv8n-pose inference on each frame, extracts wrist keypoints as a proxy for barbell position, smooths the resulting signal, and plots position and velocity over time.

## Pipeline (in order)

1. **`video_pose_infer.py`** — Main entry point. Extracts frames via `ffmpeg`, runs YOLOv8n-pose on each, builds `pose_results/barbell_track.json`, and optionally reconstructs an annotated video.
2. **`build_barbell_track.py`** — Called by `video_pose_infer.py`. Aggregates per-frame keypoint JSONs into a single track by averaging left/right wrist positions (COCO keypoints 9 and 10).
3. **`analysis.py`** — Standalone script. Loads `barbell_track.json`, removes outliers (median-based), smooths with Savitzky–Golay, computes velocity, and saves a plot to `pose_results/analysis/bar_speed.png`.
4. **`detect_concentric.py`** — Called by `analysis.py`. Detects concentric phase by comparing position to a baseline (first N frames) and a near-global maximum threshold.
5. **`pose_infer.py`** — Single-image inference helper for quick testing.

## Running the pipeline

```bash
# Full pipeline (frames and keypoints deleted after run)
python video_pose_infer.py

# Keep intermediate frames, keypoints, and annotated images
python video_pose_infer.py --keep

# Analysis and plotting (run after video_pose_infer.py)
python analysis.py
```

The video name and path are hardcoded at the bottom of `video_pose_infer.py` — change `video_name` and `video_extension` there before running.

## Key data formats

- **Keypoint JSON** (`pose_results/keypoints/frame_XXXXXX.json`): `{ "frame": "...", "keypoints": [[[x, y], ...17 points...]] }` — COCO 17-keypoint format, one person assumed.
- **Barbell track JSON** (`pose_results/barbell_track.json`): `[{ "frame": "...", "center": [x, y] }, ...]` — `center` is `null` when no person was detected.

## Dependencies

`ultralytics`, `opencv-python`, `ffmpeg-python`, `numpy`, `scipy`, `matplotlib`. The model weight file `yolov8n-pose.pt` is in the repo root and loaded at runtime.

## Notable design decisions

- Barbell position is estimated from wrist midpoints — there is no separate object detector for the bar itself.
- Y-axis is flipped (`-center[1]`) in `analysis.py` so that upward movement is positive.
- Concentric detection is toggled by `ENABLE_CONCENTRIC = False` in `analysis.py`.
- `--keep` in `video_pose_infer.py` controls whether intermediate frames and per-frame keypoint JSONs are preserved; by default they are deleted to save disk space.
