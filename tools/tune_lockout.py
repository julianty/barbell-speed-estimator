"""Tune the concentric boundary-extension rule scored on LOCKOUT error.

Follow-on to `tools/tune_extension.py`, which swept the same stage-7 rule but
scored it on rep *duration* at 30 fps and found nothing (Stage 1b, negative).
That negative result does not cover the lockout: at 30 fps the plateau walk is
not the dominant error term, so duration averaged it away. The ledger
(`barbell-velocity-app/RESOLUTION_DOWNSCALING.md`, "15 fps revisited") isolates
the failure at 15 fps instead — at lockout the bar plateaus while the lifter
holds it before racking, and the forward walk crosses the whole hold, ending the
rep where the *descent* begins (worst case +614 ms).

So this script scores the FORWARD boundary only, and does it two ways:

  A. Against the human ground truth in `sweep/_ground_truth/disputed_reps.json`
     (`truth_end_s`). Only 5 reps, but it is the project's only real truth.
  B. Against each clip's native-fps lockout across all reps. Human review scored
     60 fps correct 5/5 at lockout, so native is a defensible lockout reference
     even though it is NOT a valid reference for rep starts (it is 2/5 there).
     This is the wide-sample view; A is the authoritative one.

The backward (start) walk is left exactly as shipped — rep starts are a separate
bug with a separate mechanism (PELT splitting the sticking-point shoulder), and
mixing the two is what made the duration metric blind.

No inference: re-analyses the saved `sweep/<video>/<config>/barbell_track.json`
tracks. Does not modify `detect_concentric.py` — the extension is re-implemented
here over unextended phases (`extension_lookahead=0`).

Usage:
  python tools/tune_lockout.py
  python tools/tune_lockout.py --configs fps15 --k 0 0.05 0.1 --norm peak mean
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "yolo_pose"))

from utils import remove_outliers, smooth_positions, compute_velocity  # noqa: E402
from detect_concentric import detect_concentric  # noqa: E402
import analysis as ana  # noqa: E402

DEFAULT_K = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_LOOKAHEAD_S = 0.5


def extend_forward(start, end, pos, vel, lookahead, k, norm):
    """Shipped forward walk with the `> 0` stop replaced by `> k * reference`.

    `norm` picks the per-rep velocity the threshold is a fraction of: "peak"
    (max over the unextended phase) or "mean". k=0 reproduces today's rule for
    either norm.
    """
    if end >= start:
        window = vel[start:end + 1]
        ref = float(np.max(window)) if norm == "peak" else float(np.mean(window))
    else:
        ref = 0.0
    thresh = k * max(ref, 0.0)

    count = 0
    while end + 1 < len(pos) and count < lookahead:
        if vel[end] > thresh and pos[end + 1] > pos[end]:
            end += 1
            count += 1
        else:
            break
    return min(len(pos) - 1, end)


def extend_backward(start, pos, vel, lookahead):
    """Unchanged from `detect_concentric.py` stage 7."""
    count = 0
    while start - 1 >= 0 and count < lookahead:
        if vel[start - 1] > 0:
            start -= 1
            count += 1
        else:
            break
    return max(0, start)


def analyse(track, fps, source_fps, k, norm, lookahead_s):
    """Returns [(start_s, end_s)] for one saved run under one stop rule."""
    s = fps / source_fps
    raw = [None if e["center"] is None else -e["center"][1] for e in track]
    clean = remove_outliers(raw, window=ana._frames(ana.OUTLIER_WINDOW_REF, s), threshold=50)
    poly = 3
    w = ana._odd_savgol_window(ana.SMOOTH_WINDOW_REF, s, len(clean))
    if w <= poly:
        poly = max(0, w - 1)
    sm = smooth_positions(clean, window=w, poly=poly)
    vel = compute_velocity(sm, fps)

    phases = detect_concentric(
        sm, method="ruptures", pen=20,
        min_frames=ana._frames(ana.MIN_FRAMES_REF, s),
        smooth_velocity=vel, merge_min_vel=ana.MERGE_MIN_VEL,
        extension_lookahead=0, min_rep_ratio=ana.MIN_REP_RATIO)

    pos = np.asarray(sm, dtype=float)
    v = np.asarray(vel, dtype=float)
    lookahead = max(1, round(lookahead_s * fps))
    out = []
    for a, b in phases:
        b2 = extend_forward(a, b, pos, v, lookahead, k, norm)
        a2 = extend_backward(a, pos, v, lookahead)
        out.append((a2 / fps, b2 / fps))
    return out


def match_by_overlap(reps, start_s, end_s):
    """Rep whose window overlaps [start_s, end_s] most; None if nothing overlaps."""
    best, best_ov = None, 0.0
    for a, b in reps:
        ov = min(b, end_s) - max(a, start_s)
        if ov > best_ov:
            best, best_ov = (a, b), ov
    return best


def summarize(errs, signed):
    if not errs:
        return "        no reps"
    return (f"{np.median(errs) * 1000:>7.0f} {np.percentile(errs, 90) * 1000:>7.0f} "
            f"{max(errs) * 1000:>7.0f} {np.mean(signed) * 1000:>+8.0f} "
            f"{sum(1 for e in errs if e > 0.200):>4}/{len(errs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="sweep")
    parser.add_argument("--configs", nargs="+", default=["fps30", "fps15"])
    parser.add_argument("--k", type=float, nargs="+", default=DEFAULT_K)
    parser.add_argument("--norm", nargs="+", default=["peak"], choices=["peak", "mean"])
    parser.add_argument("--lookahead-s", type=float, nargs="+", default=[DEFAULT_LOOKAHEAD_S])
    args = parser.parse_args()

    sweep = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    videos = sorted(d for d in os.listdir(sweep)
                    if not d.startswith("_") and os.path.isdir(os.path.join(sweep, d)))

    truth_path = os.path.join(sweep, "_ground_truth", "disputed_reps.json")
    truth = json.load(open(truth_path)) if os.path.exists(truth_path) else []

    def load(v, c):
        return (json.load(open(f"{sweep}/{v}/{c}/barbell_track.json")),
                json.load(open(f"{sweep}/{v}/{c}/meta.json")))

    # Cache tracks once; the sweep re-analyses them many times.
    tracks = {(v, c): load(v, c)
              for v in videos for c in set(args.configs) | {"native"}}

    print("Lockout (rep END) error, milliseconds. Forward walk only; the backward")
    print("walk is left as shipped. k = stop threshold as a fraction of the rep's")
    print("own velocity (norm); k=0.0 is the shipped `velocity > 0` rule.\n")

    for cfg in args.configs:
        for look in args.lookahead_s:
            for norm in args.norm:
                print(f"=== {cfg}  lookahead={look}s  norm={norm} ===")
                print(f"{'k':>5} | {'--- vs human truth (n=5) ---':^38} | "
                      f"{'--- vs native lockout (all reps) ---':^38}")
                print(f"{'':>5} | {'med':>7} {'p90':>7} {'max':>7} {'bias':>8} {'>200ms':>7} | "
                      f"{'med':>7} {'p90':>7} {'max':>7} {'bias':>8} {'>200ms':>7}")
                print("-" * 92)
                # Native reference uses the shipped rule (k=0): it is the run
                # human review scored 5/5 at lockout, so it is the thing being
                # matched, not another candidate.
                base_reps = {}
                for v in videos:
                    bt, bm = tracks[(v, "native")]
                    base_reps[v] = analyse(bt, bm["fps"], bm["source_fps"], 0.0, norm, look)

                for k in args.k:
                    cfg_reps = {}
                    for v in videos:
                        ct, cm = tracks[(v, cfg)]
                        cfg_reps[v] = analyse(ct, cm["fps"], cm["source_fps"], k, norm, look)
                    # A. human truth
                    h_err, h_signed = [], []
                    for rec in truth:
                        v = rec["video"]
                        if v not in cfg_reps or rec.get("truth_end_s") is None:
                            continue
                        m = match_by_overlap(cfg_reps[v], rec["truth_start_s"], rec["truth_end_s"])
                        if m is None:
                            continue
                        d = m[1] - rec["truth_end_s"]
                        h_err.append(abs(d))
                        h_signed.append(d)
                    # B. vs native lockout, all reps
                    n_err, n_signed = [], []
                    for v in videos:
                        for a, b in base_reps[v]:
                            m = match_by_overlap(cfg_reps[v], a, b)
                            if m is None:
                                continue
                            d = m[1] - b
                            n_err.append(abs(d))
                            n_signed.append(d)
                    print(f"{k:>5.2f} | {summarize(h_err, h_signed)} | "
                          f"{summarize(n_err, n_signed)}")
                print()


if __name__ == "__main__":
    main()
