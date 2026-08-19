# tools/ — Post-processing tools

**`tools/overlay.py`** — Draws centroid crosshair and velocity label on each frame and outputs an annotated video. Works with CSV from any pipeline.

```bash
python tools/overlay.py --video videos/deadlift.mov --csv sam2_results/deadlift/velocities.csv
```

**`tools/profile_stages.py`** — Per-frame cost of each pipeline stage (decode, resize, PNG encode, letterbox, model) as a function of source resolution. Written for the `barbell-velocity-app` analysis-speed investigation; that app's `RESOLUTION_DOWNSCALING.md` is the ledger and holds the results. Perf testing for that thread is done on the laptop, and its scripts live here rather than in the app repo.

```bash
python tools/profile_stages.py --video "videos/26-6-30 225lb bench press 1.mov" --frames 25
python tools/profile_stages.py --skip-model --json out.json   # CPU stages only, no weights
```

**`tools/plot_velocity.py`** — Plots position and velocity from a CSV. Auto-discovers the newest CSV under `sam2_results/` or `yolo_results/` if no `--input` given.

```bash
python tools/plot_velocity.py
python tools/plot_velocity.py --input sam2_results/deadlift/velocities.csv --show
```
