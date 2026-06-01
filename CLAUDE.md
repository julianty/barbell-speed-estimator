# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Estimates barbell velocity from workout videos. Three pipelines are available, organized into subdirectories. All scripts are run from the project root.

## Directory structure

```
sam2/           SAM2 segmentation pipeline (primary)
yolo_seg/       YOLOv8-seg pipeline (segmentation-based, no manual prompt needed)
yolo_pose/      Legacy YOLOv8-pose pipeline (wrist keypoint proxy)
tools/          Post-processing tools (overlay, plotting)
utils.py        Shared utilities: frame extraction, velocity math
models/         YOLOv8 .pt weight files
videos/         Input videos (Bench/, Deadlift/, Squat/, Other/)
sam2_results/   SAM2 pipeline outputs
yolo_results/   YOLO-seg pipeline outputs
pose_results/   Legacy YOLO-pose pipeline outputs
```

## Primary pipeline — SAM2 plate segmentation

**`sam2/barbell_velocity.py`** — Accepts a video, prompts the user to draw a bounding box around a plate on frame 0 (or accepts `--box` directly), runs SAM2 video propagation, extracts plate centroids, calibrates scale from the known plate diameter, and writes a CSV of per-frame velocity in m/s.

```bash
# Interactive box selection (opens frame 0)
python sam2/barbell_velocity.py --video videos/deadlift.mov

# Non-interactive with a known box
python sam2/barbell_velocity.py --video videos/deadlift.mov --box "100,200,400,500"

# Full options
python sam2/barbell_velocity.py \
  --video videos/deadlift.mov \
  --output results/velocities.csv \
  --fps 60 \
  --plate-mm 450 \
  --model small
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `sam2_results/<stem>/velocities.csv` | Output CSV path |
| `--fps` | `30` | FPS fallback if not readable from file |
| `--plate-mm` | `450` | Known plate diameter in mm |
| `--model` | `small` | SAM2 variant: `tiny` \| `small` \| `large` |
| `--box` | (interactive) | `"x_min,y_min,x_max,y_max"` for frame 0 |
| `--point` | (interactive) | `"x,y"` single foreground point for frame 0 |
| `--no-recovery` | off | Disable interactive recovery on tracking loss |
| `--lost-threshold` | `5` | Consecutive empty frames before prompting recovery |
| `--max-seconds` | (none) | Only process first N seconds |
| `--frame-step` | `1` | Process every Nth frame |
| `--offload-to-cpu` | off | Keep frame features and inference state in CPU RAM instead of VRAM |
| `--num-maskmem` | `7` | SAM2 memory bank size (past frames held during propagation); reduce to 3-4 to cut VRAM use |

**Interactive mode behaviour (stdin is a TTY):**
- After the initial box/point, a preview window shows the SAM2 mask on frame 0. Left-click adds foreground points, right-click adds background points, B enters box-draw mode (drag to draw a bounding box that replaces the current prompt), Z undoes the last added point, R resets to the original/last-committed prompt, Enter/Space accepts.
- During propagation, if tracking is lost for `--lost-threshold` consecutive frames, a recovery window opens on the first lost frame. Press B to draw a box, P to click a point, S to skip.

**Output CSV columns:** `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms`

## YOLO-seg pipeline

**`yolo_seg/yolo_velocity.py`** — Uses YOLOv8 instance segmentation + BotSORT tracking. Interactive plate selection on frame 0; everything downstream (outlier removal, smoothing, velocity, CSV format) is identical to the SAM2 pipeline.

```bash
python yolo_seg/yolo_velocity.py --video videos/deadlift.mov
python yolo_seg/yolo_velocity.py --video videos/deadlift.mov --model small --no-preview
```

Output goes to `yolo_results/<video_stem>/velocities.csv`.

## Legacy pipeline — YOLO wrist proxy

Uses YOLOv8n-pose wrist keypoints as a barbell proxy. Less precise than segmentation pipelines but faster and requires no manual annotation.

1. **`yolo_pose/pose_velocity.py`** — Main entry point. Extracts frames via `ffmpeg`, runs YOLOv8n-pose on each, builds `pose_results/barbell_track.json`, and optionally reconstructs an annotated video.
2. **`yolo_pose/build_barbell_track.py`** — Called by `pose_velocity.py`. Aggregates per-frame keypoint JSONs into a single track by averaging left/right wrist positions (COCO keypoints 9 and 10).
3. **`yolo_pose/analysis.py`** — Standalone script. Loads `barbell_track.json`, removes outliers, smooths with Savitzky–Golay, computes velocity, and saves a plot to `pose_results/analysis/bar_speed.png`.
4. **`yolo_pose/detect_concentric.py`** — Called by `analysis.py`. Detects concentric phase by comparing position to a baseline (first N frames) and a near-global maximum threshold.

```bash
# Full YOLO-pose pipeline
python yolo_pose/pose_velocity.py videos/deadlift.mov

# Keep intermediate frames, keypoints, and annotated images
python yolo_pose/pose_velocity.py videos/deadlift.mov --keep

# Analysis and plotting (run after pose_velocity.py)
python yolo_pose/analysis.py
```

## Post-processing tools

**`tools/overlay.py`** — Draws centroid crosshair and velocity label on each frame and outputs an annotated video. Works with CSV from any pipeline.

```bash
python tools/overlay.py --video videos/deadlift.mov --csv sam2_results/deadlift/velocities.csv
```

**`tools/plot_velocity.py`** — Plots position and velocity from a CSV. Auto-discovers the newest CSV under `sam2_results/` or `yolo_results/` if no `--input` given.

```bash
python tools/plot_velocity.py
python tools/plot_velocity.py --input sam2_results/deadlift/velocities.csv --show
```

## Shared utilities

**`utils.py`** — Shared across all pipelines:
- `extract_frames(video_path, frames_dir, fps=None)` — extracts JPEG frames via ffmpeg
- `frames_to_video(frames_dir, output_video_path, fps=30)` — reconstructs video from frames
- `remove_outliers(positions, window=5, threshold=50)` — median-based outlier removal
- `smooth_positions(positions, window=9, poly=3)` — Savitzky–Golay smoothing
- `compute_velocity(positions, fps)` — finite-difference velocity

## Key data formats

- **Keypoint JSON** (`pose_results/keypoints/frame_XXXXXX.json`): `{ "frame": "...", "keypoints": [[[x, y], ...17 points...]] }` — COCO 17-keypoint format, one person assumed.
- **Barbell track JSON** (`pose_results/barbell_track.json`): `[{ "frame": "...", "center": [x, y] }, ...]` — `center` is `null` when no person was detected.
- **Velocity CSV** (`velocities.csv`): `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms`

## Dependencies

**SAM2 pipeline:** `torch`, `torchvision`, `transformers` (HuggingFace), `opencv-python`, `numpy`, `scipy`, `ffmpeg-python`

**YOLO pipelines:** `ultralytics`, `opencv-python`, `ffmpeg-python`, `numpy`, `scipy`, `matplotlib`

## Notable design decisions

- SAM2 model is loaded from HuggingFace (`facebook/sam2-hiera-{tiny,small,large}`); uses float16 + CUDA autocast when available.
- Scale calibration derives `px/mm` from the vertical span of the plate mask bounding box on frame 0.
- Y-axis is flipped (`-cy`) throughout so that upward movement is positive velocity.
- Velocity math (`remove_outliers`, `smooth_positions`, `compute_velocity`) lives in `utils.py` and is shared by all three pipelines.
- Each pipeline script inserts the project root into `sys.path` at startup so `utils` is always importable regardless of CWD.
- `--clean` in `pose_velocity.py` controls whether intermediate frames and per-frame keypoint JSONs are preserved; by default they are kept.
- Concentric detection is toggled by `ENABLE_CONCENTRIC = False` in `yolo_pose/analysis.py`.
