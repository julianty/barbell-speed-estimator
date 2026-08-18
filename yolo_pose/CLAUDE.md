# yolo_pose/ — Primary pipeline: YOLO Pose wrist proxy

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
