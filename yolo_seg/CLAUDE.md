# yolo_seg/ — YOLO-seg pipeline

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
