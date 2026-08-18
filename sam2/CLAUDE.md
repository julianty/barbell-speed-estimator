# sam2/ — SAM2 pipeline (experimental)

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
