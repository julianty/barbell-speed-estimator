"""Tune the concentric boundary-extension rule for fps robustness.

Stage 1 of the analysis-speed investigation (see
`barbell-velocity-app/RESOLUTION_DOWNSCALING.md`) found that rep *detection*
survives down to 10 fps but rep *boundaries* drift, and that about half the
velocity error is that drift rather than signal loss. The suspect is stage 7 of
`detect_concentric.py`, which walks each boundary outward while `velocity > 0`.

A zero threshold is the fragile part: velocity approaches zero asymptotically at
a rep edge, so the walk terminates wherever noise happens to cross zero, in the
flattest part of the curve. Lower fps samples that flat region more coarsely, so
the stop point moves — and a boundary that moves changes rep duration, which is
the app's primary measure (it is the most camera-angle-robust quantity we have).

This script replaces that rule with a threshold set as a fraction `k` of the
rep's own peak velocity, which is scale-free and lands the stop on a steep part
of the curve, and sweeps `k` against the saved sweep tracks.

Scored on REP DURATION agreement between a downsampled config and the same
clip's native-fps run, because duration is what the app reports. Note this
measures *consistency across fps*, not correctness — there is no ground-truth
rep duration in this dataset, so a rule could be self-consistently wrong. The
native-fps result is the reference by assumption.

No inference is run: it re-analyses `sweep/<video>/<config>/barbell_track.json`,
so it is cheap to re-run. It does not modify `detect_concentric.py` — the
extension is re-implemented here, over unextended phases obtained by calling
`detect_concentric` with `extension_lookahead=0`.

Usage:
  python tools/tune_extension.py
  python tools/tune_extension.py --configs fps30 fps15 --k 0 0.05 0.1
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

# Fractions of the rep's own peak velocity to stop the boundary walk at.
# 0.0 reproduces today's shipped `velocity > 0` rule exactly.
DEFAULT_K = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

# Max extension per side, in SECONDS (the shipped constant is 30 frames at
# 60 fps = 0.5 s). Expressed in time so it means the same thing at every fps.
DEFAULT_LOOKAHEAD_S = 0.5


def extend(start, end, pos, vel, lookahead, k, symmetric):
    """Walk both boundaries outward while velocity stays above k * peak.

    `symmetric=False` keeps the shipped asymmetry (the forward walk also
    requires the position to be strictly rising; the backward walk does not).
    """
    peak = float(np.max(vel[start:end + 1])) if end >= start else 0.0
    thresh = k * peak

    count = 0
    while end + 1 < len(pos) and count < lookahead:
        rising = pos[end + 1] > pos[end]
        if vel[end] > thresh and (rising or symmetric):
            end += 1
            count += 1
        else:
            break

    count = 0
    while start - 1 >= 0 and count < lookahead:
        falling_back = pos[start] > pos[start - 1]
        if vel[start - 1] > thresh and (falling_back or symmetric):
            start -= 1
            count += 1
        else:
            break

    return max(0, start), min(len(pos) - 1, end)


def analyse(track, fps, source_fps, k, symmetric, lookahead_s):
    """Returns [(start_s, end_s, duration_s)] for one run under one rule."""
    s = fps / source_fps
    raw = [None if e["center"] is None else -e["center"][1] for e in track]
    clean = remove_outliers(raw, window=ana._frames(ana.OUTLIER_WINDOW_REF, s), threshold=50)
    poly = 3
    w = ana._odd_savgol_window(ana.SMOOTH_WINDOW_REF, s, len(clean))
    if w <= poly:
        poly = max(0, w - 1)
    sm = smooth_positions(clean, window=w, poly=poly)
    vel = compute_velocity(sm, fps)

    # extension_lookahead=0 returns the raw selected segments, unextended, so
    # stages 1-6 stay exactly as shipped and only stage 7 is under test.
    phases = detect_concentric(
        sm, method="ruptures", pen=20,
        min_frames=ana._frames(ana.MIN_FRAMES_REF, s),
        smooth_velocity=vel, merge_min_vel=ana.MERGE_MIN_VEL,
        extension_lookahead=0, min_rep_ratio=ana.MIN_REP_RATIO)

    pos = np.asarray(sm)
    v = np.asarray(vel)
    lookahead = max(1, round(lookahead_s * fps))
    out = []
    for a, b in phases:
        a2, b2 = extend(a, b, pos, v, lookahead, k, symmetric)
        out.append((a2 / fps, b2 / fps, (b2 - a2) / fps))
    return out


def score(base, cfg):
    """Duration error per matched rep. Reps are matched positionally, which is
    safe only when the counts agree — mismatches are counted, not matched."""
    if len(base) != len(cfg):
        return None
    return [(abs(c[2] - b[2]) / b[2], abs(c[2] - b[2])) for b, c in zip(base, cfg) if b[2] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="sweep")
    parser.add_argument("--configs", nargs="+", default=["fps30", "fps15"])
    parser.add_argument("--k", type=float, nargs="+", default=DEFAULT_K)
    parser.add_argument("--lookahead-s", type=float, nargs="+", default=[DEFAULT_LOOKAHEAD_S])
    parser.add_argument("--symmetric", action="store_true",
                        help="also try the symmetric variant of the walk")
    args = parser.parse_args()

    sweep = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    videos = sorted(d for d in os.listdir(sweep)
                    if not d.startswith("_") and os.path.isdir(os.path.join(sweep, d)))

    def load(v, c):
        return (json.load(open(f"{sweep}/{v}/{c}/barbell_track.json")),
                json.load(open(f"{sweep}/{v}/{c}/meta.json")))

    variants = [False, True] if args.symmetric else [False]

    print("Rep-duration agreement with the same clip's native-fps run.")
    print("k = stop threshold as a fraction of that rep's peak velocity;")
    print("k=0.0 is the shipped `velocity > 0` rule.\n")

    for cfg in args.configs:
        print(f"=== {cfg} ===")
        print(f"{'k':>6} {'look':>6} {'sym':>5} {'med':>7} {'p90':>7} {'max':>7} "
              f"{'med_s':>8} {'max_s':>8} {'cnt-mismatch':>13}")
        print("-" * 72)
        for look in args.lookahead_s:
            for sym in variants:
                for k in args.k:
                    rel, absd, mismatch = [], [], 0
                    for v in videos:
                        bt, bm = load(v, "native")
                        ct, cm = load(v, cfg)
                        b = analyse(bt, bm["fps"], bm["source_fps"], k, sym, look)
                        c = analyse(ct, cm["fps"], cm["source_fps"], k, sym, look)
                        pairs = score(b, c)
                        if pairs is None:
                            mismatch += 1
                            continue
                        rel += [p[0] for p in pairs]
                        absd += [p[1] for p in pairs]
                    if not rel:
                        print(f"{k:>6.2f} {look:>6.2f} {str(sym):>5}  all clips mismatch")
                        continue
                    print(f"{k:>6.2f} {look:>6.2f} {str(sym):>5} {np.median(rel):>6.1%} "
                          f"{np.percentile(rel, 90):>6.1%} {max(rel):>6.1%} "
                          f"{np.median(absd):>7.3f}s {max(absd):>7.3f}s {mismatch:>10}/{len(videos)}")
        print()


if __name__ == "__main__":
    main()
