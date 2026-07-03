# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Estimates barbell velocity from workout videos using computer vision. The investigation evaluated several approaches; the primary working pipeline uses YOLO Pose wrist keypoints as a barbell proxy with Savitzky-Golay smoothing and PELT-based concentric phase detection. Two segmentation-based pipelines (SAM2 and YOLO-seg) were also built; both track the barbell plate directly and output calibrated m/s velocity.

## Directory structure

```
yolo_pose/      Primary pipeline: YOLOv8-pose wrist-proxy tracking + concentric detection
sam2/           SAM2 segmentation pipeline (plate tracking, found unreliable in real gym footage)
yolo_seg/       YOLOv8-seg pipeline (plate tracking, no manual video prompt needed)
tools/          Post-processing tools (overlay, plotting)
utils.py        Shared utilities: frame extraction, velocity math
models/         YOLOv8 .pt weight files
videos/         Input videos (Bench/, Deadlift/, Squat/, Other/)
sam2_results/   SAM2 pipeline outputs
yolo_results/   YOLO-seg pipeline outputs
pose_results/   YOLO-pose pipeline outputs
```

## Primary pipeline — YOLO Pose wrist proxy

Uses YOLOv8-pose keypoints to track wrist position as a proxy for the barbell. Outputs velocity in px/s. Absolute m/s conversion requires a calibration step (not yet implemented in this pipeline).

1. **`yolo_pose/pose_velocity.py`** — Main entry point. Extracts frames via `ffmpeg`, runs YOLOv8-pose on each, builds `pose_results/<video>/<model>/barbell_track.json`, and reconstructs an annotated video.
2. **`yolo_pose/build_barbell_track.py`** — Called by `pose_velocity.py`. Aggregates per-frame keypoint JSONs into a single track by averaging left/right wrist positions (COCO keypoints 9 and 10). Uses largest bounding box on frame 0 for person selection; subsequent frames lock via wrist proximity.
3. **`yolo_pose/analysis.py`** — Standalone script. Loads `barbell_track.json`, removes outliers, smooths with Savitzky-Golay, computes velocity (px/s), and optionally detects concentric phases.
4. **`yolo_pose/detect_concentric.py`** — Called by `analysis.py`. Two algorithms: `ruptures` PELT on the velocity signal (recommended), and `scipy` peak/trough detection. PELT output goes through segment merging, boundary extension, and velocity-ratio filtering.

```bash
# Full YOLO-pose pipeline (default model: small)
python yolo_pose/pose_velocity.py --video videos/deadlift.mov

# Specify model size
python yolo_pose/pose_velocity.py small --video videos/deadlift.mov
python yolo_pose/pose_velocity.py large --video videos/deadlift.mov

# Keep intermediate frames, keypoints, and annotated images
python yolo_pose/pose_velocity.py --video videos/deadlift.mov --keep

# Downsample inference to a lower FPS (e.g. to find the minimum FPS for reliable tracking)
python yolo_pose/pose_velocity.py --video videos/deadlift.mov --fps 15

# Analysis and plotting (run after pose_velocity.py)
python yolo_pose/analysis.py
python yolo_pose/analysis.py --track pose_results/deadlift/small/barbell_track.json
```

**Concentric detection settings** (top of `analysis.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCENTRIC_METHOD` | `"both"` | `None` \| `"ruptures"` \| `"peaks"` \| `"both"` |
| `MIN_FRAMES_REF` | `5` | Minimum segment length (frames, at the video's own native FPS) to be a candidate |
| `MERGE_MIN_VEL` | `50.0` | px/s; segments below this avg velocity excluded from merge runs |
| `EXTENSION_LOOKAHEAD_REF` | `30` | Max boundary extension per side (frames, at the video's own native FPS) after candidate selection |
| `MIN_REP_RATIO` | `0.60` | Fraction of estimated full-rep displacement a candidate must cover |

`MIN_FRAMES_REF`, `EXTENSION_LOOKAHEAD_REF`, and the outlier-removal/Savitzky-Golay smoothing windows are frame counts tuned against each video's own native FPS (from `meta.json`'s `source_fps`), then rescaled by `inference_fps / source_fps`. A run at native FPS (no `--fps`) reproduces these counts exactly, whatever that video's native FPS is; only a downsampled `--fps` run rescales them, so results stay consistent across `--fps` values without changing behavior for existing native-FPS runs. The PELT `pen` value is not auto-scaled — it may need manual retuning at very different FPS.

**Output:** `pose_results/<video>/<model>/barbell_track.json` + annotated video + `bar_speed.png` plot

## SAM2 pipeline (experimental)

Tracks the barbell plate directly using SAM2 video segmentation. Calibrates scale from the known plate diameter and outputs velocity in m/s. Found unreliable on real gym footage — tracking can drift, lose the object, or latch onto the wrong region mid-lift.

**`sam2/barbell_velocity.py`** — User draws a bounding box or clicks a point on the plate in frame 0; SAM2 propagates the mask through the video. Interactive recovery prompt if tracking is lost.

```bash
# Interactive prompt (opens frame 0)
python sam2/barbell_velocity.py --video videos/deadlift.mov

# Non-interactive with a known box
python sam2/barbell_velocity.py --video videos/deadlift.mov --box "100,200,400,500"
```

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `sam2_results/<stem>/velocities.csv` | Output CSV path |
| `--plate-mm` | `450` | Known plate diameter in mm (used for scale calibration) |
| `--model` | `small` | SAM2 variant: `tiny` \| `small` \| `large` |
| `--box` | (interactive) | `"x_min,y_min,x_max,y_max"` for frame 0 |
| `--point` | (interactive) | `"x,y"` single foreground point for frame 0 |
| `--no-recovery` | off | Disable interactive recovery on tracking loss |
| `--lost-threshold` | `5` | Consecutive empty frames before prompting recovery |
| `--max-seconds` | (none) | Only process first N seconds |
| `--frame-step` | `1` | Process every Nth frame |
| `--offload-to-cpu` | off | Keep frame features and inference state in CPU RAM instead of VRAM |
| `--num-maskmem` | `7` | SAM2 memory bank size; reduce to 3-4 to cut VRAM use |

**Output CSV columns:** `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms`

SAM2 model is loaded from HuggingFace (`facebook/sam2-hiera-{tiny,small,large}`); uses float16 + CUDA autocast when available.

## YOLO-seg pipeline

Tracks the barbell plate using YOLOv8 instance segmentation + BotSORT tracking. No manual video prompt: user selects the plate from detected objects on frame 0 (or the first frame with a detection). Same downstream processing and CSV format as the SAM2 pipeline; outputs velocity in m/s.

**`yolo_seg/yolo_velocity.py`**

```bash
python yolo_seg/yolo_velocity.py --video videos/deadlift.mov
python yolo_seg/yolo_velocity.py --video videos/deadlift.mov --model small --no-preview
```

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--output` | `yolo_results/<stem>/velocities.csv` | Output CSV path |
| `--plate-mm` | `450` | Known plate diameter in mm |
| `--model` | `nano` | YOLOv8-seg variant: `nano` \| `small` \| `medium` |
| `--frame-step` | `1` | Process every Nth frame |
| `--max-seconds` | (none) | Only process first N seconds |
| `--no-preview` | off | Auto-pick largest mask on frame 0 (skip interactive selection) |
| `--conf` | `0.25` | Detection confidence threshold |

**Output CSV columns:** `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms`

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
