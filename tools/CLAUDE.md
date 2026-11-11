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

**`tools/tune_lockout.py`** — Follow-on to `tune_extension.py`, scoring the same stage-7 rule on **lockout (rep end) error** rather than duration, and on the **forward walk only** so the separate rep-start bug cannot contaminate it. Scores against both the 5 human-adjudicated `truth_end_s` values and each clip's native-fps lockout. Re-analyses the saved `sweep/` tracks; no inference. Result was negative — see the ledger's Stage 1c before re-running: the extension walk contributes only the tail, the stop threshold does not pay for itself, and a wider smoothing window makes it worse.

```bash
python tools/tune_lockout.py
python tools/tune_lockout.py --configs native fps15 --k 0 0.1 0.3 --norm peak mean
```

**`tools/check_other_lifts.py`** — Checks the 30 fps decision on squat and deadlift, which the bench-only fps sweep never covered. The source videos are not in this repo, so it decimates the native-fps tracks checked into the app's golden fixtures (`test/fixtures/*.golden.json`, `input_positions`) and re-analyses them with normalized constants. Tests the **analysis stage only** — it cannot tell you whether YOLO pose still tracks the wrist at 30 fps. Runs every phase offset, so the spread between offsets is the probe's own noise floor. Result: 30 fps transfers (ledger, Stage 1d).

```bash
python tools/check_other_lifts.py
python tools/check_other_lifts.py --strides 2 --min-duration 0.5   # drop the natively-spurious reps
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

**Pose-backend comparison** (the app's LEDGER.md, launch checklist item 1: replacing AGPL yolo26s-pose). Results so far are recorded there.

- **`tools/compare_pose_backends.py`** — Runs Apple Vision, MoveNet Thunder/Lightning, or yolo26s on the regression clips at 30 fps, builds the wrist-midpoint track, analyses it with `sweep_configs.analyse`, and scores reps against the full-res YOLO run in `sweep/<clip>/fps30`. `--max-edge 640` matches the app's decode size and scales keypoints back to source pixels, as the app does. Writes `sweep/<clip>/fps30_<backend>[_640]/` and `sweep/_backends[_640]/REPORT.md`.
- **`tools/vision_pose.swift`** — Runs `VNDetectHumanBodyPoseRequest` (macOS) over a frames dir and writes the joints in COCO-17 order. Called by the script above.
- **`tools/movenet_pose.py`** — MoveNet TFLite with the tutorial's crop-region tracking. Needs `ai-edge-litert`, which has no wheel for this venv's Python 3.13, so it runs from a separate Python 3.12 env passed as `--movenet-python`. Models come from TF Hub: `lite-model/movenet/singlepose/{thunder,lightning}/tflite/float16/4`.
- **`tools/label_rep_starts.py`** — Blinded slow-motion clips of reps whose candidate starts disagree, for hand labelling. Writes `sweep/_ground_truth_starts/{*.mp4, all_reps.mp4, labels.json}`. Keeps labels already given; `--only video:rep` re-renders a subset.
- **`tools/score_rep_starts.py`** — Scores every run's rep starts against the hand labels.

```bash
python tools/compare_pose_backends.py --movenet-python <py3.12 env>/bin/python --movenet-dir <dir>
python tools/compare_pose_backends.py --max-edge 640 --backend vision --backend yolo26s
python tools/label_rep_starts.py && python tools/score_rep_starts.py
```
