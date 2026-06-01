# Barbell Speed Estimator

Estimates barbell velocity from workout videos. The primary pipeline uses SAM2 to segment and track a barbell plate, computing velocity in m/s with real scale calibration. A legacy pipeline using YOLOv8 wrist keypoints is also available.

## Requirements

```
pip install -r requirements.txt
```

ffmpeg must also be installed on your system and available on PATH.

## Primary pipeline — SAM2 plate tracking

Tracks a plate mask through the video using SAM2, calibrates scale from the known plate diameter, and outputs per-frame velocity in m/s.

```bash
# Interactive: opens frame 0 so you can draw a box around the plate
python barbell_velocity.py --video videos/deadlift.mov

# Non-interactive: provide the box directly
python barbell_velocity.py --video videos/deadlift.mov --box "100,200,400,500"

# Full options
python barbell_velocity.py \
  --video videos/deadlift.mov \
  --output results/velocities.csv \
  --fps 60 \
  --plate-mm 450 \
  --model small
```

Output is a CSV with columns: `frame, timestamp_s, cx_px, cy_px, velocity_ms, smoothed_velocity_ms`

## Legacy pipeline — YOLO wrist proxy

Uses YOLOv8n-pose wrist keypoints as a barbell proxy. Faster, but less precise.

```bash
# Run pipeline (reconstructs annotated video, keeps all output)
python video_pose_infer.py videos/deadlift.mov

# Delete rendered images after reconstruction to save disk space
python video_pose_infer.py videos/deadlift.mov --clean

# Generate position/velocity plots from a saved track
python analysis.py
```

Output is saved to `pose_results/`:
- `barbell_track.json` — bar position per frame
- `analysis/bar_speed.png` — position and velocity plots

## Example output

![Example analysis](results_stephen_1.jpg)
