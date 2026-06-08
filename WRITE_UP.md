# Write-Up Instructions: Barbell Speed Tracking Investigation

## Purpose

This document contains instructions for writing a technical blog post about an investigation into barbell speed tracking using computer vision. The post will be published on a personal portfolio website and is intended to be read by recruiters and engineers.

- **Target reading time:** ~10 minutes
- **Summary/abstract reading time:** ~1 minute
- **Tone:** Technical but accessible (engineering blog post style)
- **Output format:** `.tsx` file for a portfolio website
- **Figures:** Referenced throughout — placeholders should be kept in the draft

---

## Project Summary (LLM Context)

The author is building a free, open barbell velocity tracking tool for the three powerlifting movements: squat, bench press, and deadlift. The motivation is personal — commercial velocity-based training (VBT) apps exist but are paywalled or limited. The goal is a fast, free alternative built with computer vision.

The investigation proceeded roughly as follows:

- **Model selection:** YOLO models were evaluated first but were noisy out of the box. SAM2 (Meta) was investigated as an alternative but proved unreliable on real gym footage. YOLO Pose was ultimately selected for its reliable person detection and stable wrist tracking, especially at larger model sizes (nano → large).
- **Simplification strategy:** The deadlift was tackled first because the athlete is fully visible, making it the cleanest starting point.
- **Signal processing:** Raw tracking data is noisy. A Savitzky-Golay filter (fits polynomial functions to tunable time windows) was applied to smooth position data.
- **Phase detection:** Two algorithms were evaluated for isolating the concentric (upward) movement phase — the `ruptures` package (PELT: Pruned Exact Linear Time, with an L2 cost model) and `scipy` peak-trough detection. Crucially, PELT is run on the **velocity signal** (first-difference of position) rather than the raw position: running changepoint detection on position over-segments rising trends because a sustained upward movement looks like many small constant-mean segments; differencing first collapses a clean concentric phase into a single near-zero-mean velocity segment, so PELT finds it as one piece. The `ruptures` approach was tunable and more reliable, though with the penalty parameter set low it occasionally captures short lock-out or set-up movements as concentric segments. A `min_vel_ratio` filter was added to reject slow candidates, but does not eliminate all false positives (deemed acceptable).
- **Velocity metric:** px/sec velocity is extractable and useful for self-comparison. Converting to m/s requires knowing the lifter's range of motion in-frame, which is a hard problem without a calibration reference (e.g. a checkerboard pattern). Rep time in seconds is a sufficient relative metric for self-comparison, with the caveat that lift variations with different ranges of motion (e.g. sumo vs. conventional deadlift) are not directly comparable.
- **Generalizing to squat and bench:** Squat works with the constraint that the lifter's wrists remain in frame. Bench press exposed a limitation: the YOLO Pose model struggles when the lifter's head is not visible (lying down), producing unreliable detections. A high-angle camera is the proposed next step to keep both the head and wrists visible.

**Current milestone:** YOLO Pose is usable for deadlift velocity estimation. Squat is functional with constraints. Bench is unresolved pending a camera angle experiment.

---

## Outline

1. **Introduction / Motivation** (~1 min)
   - What velocity-based training is and why it matters
   - Existing apps are paywalled or limited
   - Goal: fast, free barbell speed tracking

2. **Simplifying the Problem** (~1 min)
   - Why deadlift first — full athlete visibility, cleanest signal
   - Scoping down as a deliberate engineering choice

3. **Finding the Right Detector** (~2 min)
   - YOLO out of the box: too noisy
   - SAM2 investigation: promising idea, unreliable on real gym footage
   - _[Figure: SAM2 failure examples]_
   - Landing on YOLO Pose: reliable person detection, stable wrist tracking across model sizes (nano → large)

4. **Taming the Signal** (~2 min)
   - Raw tracking data is noisy — why that matters for velocity
   - Savitzky-Golay filter: what it does, why it worked here specifically — SG preserves peak shape (unlike a moving average), which matters because velocity is computed as a finite difference of position; a peak-flattening smoother would underestimate peak bar speed
   - Isolating the concentric phase: `ruptures` (PELT/L2 on the velocity signal) vs. `scipy` peak-trough detection, why `ruptures` won with tuning
   - _[Figure: smoothed position signal with detected concentric phases marked]_

5. **Extracting Velocity** (~1.5 min)
   - px/sec as a useful relative metric
   - The m/s problem: range of motion, camera distortion, checkerboard calibration
   - Why rep time in seconds is sufficient for self-comparison (with the caveat about lift variation)
   - _[Figure: sample velocity output]_

6. **Generalizing to Squat & Bench** (~2 min)
   - Squat: works with the constraint of keeping wrists in frame
   - Bench: model breaks down — YOLO Pose is a top-down detector that first needs a confident person bounding box; a lying-down athlete seen from the side is out-of-distribution for the person detector itself, so the failure happens before keypoint extraction even runs, not just in keypoint accuracy
   - _[Figure: bench detection failure example]_
   - Current hypothesis: high-angle camera may solve both problems

7. **Summary / Abstract** (~1 min)
   - What was built, what works, what the limits are, what's next

---

## Notes

- The `ruptures` algorithm is **PELT (Pruned Exact Linear Time)** with an L2 cost model, applied to the first-difference (velocity) of the smoothed position signal. The website will include a collapsible section explaining how it works.

  PELT parameter:

  | Parameter | Value used | What it does |
  |---|---|---|
  | `pen` (penalty) | 20 | Controls how many changepoints are found. Lower → more, finer segments; higher → fewer, coarser. The only PELT tuning dial. |

- Raw PELT output required additional post-processing to handle real-world signal quirks. Three steps were added on top:
  - **Segment merging** — adjacent segments that both exceed `merge_min_vel` (50 px/s) are merged into a single candidate, handling cases where a single rep gets split across multiple changepoints.
  - **Boundary extension** — selected phase boundaries are extended forward/backward by up to `extension_lookahead` (30) frames as long as the velocity direction supports it, to better capture the full rep.
  - **Velocity-ratio rejection** — candidates whose average velocity falls below `min_vel_ratio` (0.50) × the median average velocity of merged candidates are discarded, filtering out slow incidental segments. Median is used instead of max so a single fast outlier can't inflate the threshold.
- Commercial app names are intentionally omitted.
- A separate draft file contains the in-progress write-up text.
