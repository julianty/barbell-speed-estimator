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

**`tools/sweep_configs.py`** — Stage 1 of the same investigation: runs the YOLO-pose pipeline over an FPS ladder (native baseline + 30/20/15/10 by default) and analyses each track twice — `normalized` constants (rescaled by `inference_fps/source_fps`, what `analysis.py` does) and `fixed` constants (raw frame counts, what the app's `AnalysisConfig` does). Inference runs once per (video, fps) and is shared by both arms. Resolution is not swept (settled by decision D1 in the ledger). Writes `sweep/<video>/<config>/{barbell_track.json, meta.json, timing.json, <arm>/{reps.json, signals.json}}`.

```bash
python tools/sweep_configs.py                                  # all videos, default ladder
python tools/sweep_configs.py --video "videos/26-6-30 225lb bench press 1.mov" --fps-ladder 15
```

**`tools/compare_configs.py`** — Scores that sweep against each clip's native-fps baseline using the pre-registered three-tier metric (rep-level agreement as the gate; NRMSE / Pearson r / peak-velocity bias as diagnostics; wall clock as cost). Emits `sweep/_compare/{metrics.csv, REPORT.md, <video>_velocity.png}`. The `native/fixed` row is a harness self-check and must read exactly zero.

```bash
python tools/compare_configs.py
python tools/compare_configs.py --sweep sweep --no-plots
```

**`tools/tune_extension.py`** — Sweeps the concentric boundary-extension rule (stage 7 of `detect_concentric.py`) for FPS robustness, scored on **rep duration** agreement with each clip's native-fps run. Re-analyses the saved `sweep/` tracks, so no inference runs and it is cheap to repeat; it does not modify `detect_concentric.py`. Result was negative — see the ledger's Stage 1b section before re-running: threshold, lookahead cap, and walk symmetry are all inert.

```bash
python tools/tune_extension.py --symmetric
python tools/tune_extension.py --configs fps30 --k 0 0.05 0.1 --lookahead-s 0.2 0.5
```

**`tools/extract_rep_clips.py`** — Cuts the reps where two sweep configs disagree on concentric duration into slow-motion annotated clips (both runs' start/lockout marks drawn over the bar-height trace), so a human can adjudicate which boundary is right. Writes `sweep/_ground_truth/<clip>_repN.mp4` plus `disputed_reps.json` with `truth_start_s`/`truth_end_s` fields to fill in by hand. Re-extracts full clips via ffmpeg rather than seeking, so the overlay stays frame-aligned with the track.

```bash
python tools/extract_rep_clips.py                          # reps disagreeing >10%
python tools/extract_rep_clips.py --config fps15 --threshold 0.2 --out-fps 10
```

**`tools/plot_velocity.py`** — Plots position and velocity from a CSV. Auto-discovers the newest CSV under `sam2_results/` or `yolo_results/` if no `--input` given.

```bash
python tools/plot_velocity.py
python tools/plot_velocity.py --input sam2_results/deadlift/velocities.csv --show
```
