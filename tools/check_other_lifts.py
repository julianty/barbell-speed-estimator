"""Check the 30 fps decision on squat and deadlift, not just bench press.

The whole fps investigation (see `barbell-velocity-app/RESOLUTION_DOWNSCALING.md`)
ran on 8 bench-press clips. Bench is the *easiest* case for a wrist-proxy
tracker: short travel, small camera-relative motion, a fixed torso. Squat and
deadlift have longer bar paths, larger and faster whole-body motion, and (for
deadlift) a lockout that is a genuine standing hold rather than an arm extension.
Nothing so far says the 30 fps result transfers to them.

The source squat/deadlift videos are not in this repo, so inference cannot be
re-run. What *is* checked in is the native-fps position track for each, inside
the app's golden fixtures (`barbell-velocity-app/test/fixtures/*.golden.json`,
field `input_positions`). That is enough to test the ANALYSIS half of the
question: given a good track, does the detector still find the same reps with
the same boundaries when it only sees every Nth sample?

**What this does NOT test.** Only the analysis stage. It cannot see whether YOLO
pose still tracks the wrist at 30 fps on a squat, which is the other half and
needs the videos. It also decimates a native track rather than re-extracting,
which the ledger records as *pessimistic*: decimation overstates jitter relative
to the real pipeline (Stage 1 "Real-pipeline probe"). So a PASS here is a real
lower bound on analysis quality; a FAIL would need re-checking against a real
downsampled run before being believed.

Constants are rescaled by `inference_fps / source_fps` exactly as `analysis.py`
does (the "normalized" arm), so this matches the shipped Python reference.

Usage:
  python tools/check_other_lifts.py
  python tools/check_other_lifts.py --strides 2 4 --fixtures <dir>
"""

import argparse
import glob
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

DEFAULT_FIXTURES = os.path.join(ROOT, "barbell-velocity-app", "test", "fixtures")


def analyse(positions, fps, source_fps):
    """The shipped normalized-constants analysis, over a bare position list."""
    s = fps / source_fps
    clean = remove_outliers(positions, window=ana._frames(ana.OUTLIER_WINDOW_REF, s),
                            threshold=50)
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
        extension_lookahead=ana._frames(ana.EXTENSION_LOOKAHEAD_REF, s),
        min_rep_ratio=ana.MIN_REP_RATIO)

    v = np.asarray(vel, dtype=float)
    out = []
    for a, b in phases:
        seg = v[a:b + 1]
        if len(seg) == 0:
            continue
        out.append(dict(start_s=a / fps, end_s=b / fps, duration_s=(b - a) / fps,
                        avg_vel=float(np.mean(seg)), peak_vel=float(np.max(seg))))
    return out


def match(base, cfg, min_iou=0.5):
    """Greedy temporal-IoU matching, the same rule the Stage 1 gate used."""
    pairs, used = [], set()
    for i, b in enumerate(base):
        best, best_iou = None, min_iou
        for j, c in enumerate(cfg):
            if j in used:
                continue
            inter = min(b["end_s"], c["end_s"]) - max(b["start_s"], c["start_s"])
            if inter <= 0:
                continue
            union = max(b["end_s"], c["end_s"]) - min(b["start_s"], c["start_s"])
            iou = inter / union
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs, [i for i in range(len(base)) if i not in {p[0] for p in pairs}], \
        [j for j in range(len(cfg)) if j not in used]


def rel(a, b):
    return abs(a - b) / abs(b) if b else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4],
                        help="decimation factors; 2 = 30 fps from a 60 fps track")
    parser.add_argument("--min-duration", type=float, default=0.0,
                        help="drop reps shorter than this (seconds) before scoring. "
                             "Two of the five fixtures carry reps that are spurious at "
                             "NATIVE fps — sub-0.2 s at 1900-2850 px/s — and scoring "
                             "against those measures detector noise, not the fps change. "
                             "0.5 isolates the physically real reps.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.fixtures, "*.golden.json")))
    if not files:
        sys.exit(f"no golden fixtures under {args.fixtures}")

    for path in files:
        d = json.load(open(path))
        src_fps = d["source"]["fps"]
        pos = d["input_positions"]
        name = os.path.basename(path).replace(".golden.json", "")

        def keep(reps):
            return [r for r in reps if r["duration_s"] >= args.min_duration]

        raw_base = analyse(pos, src_fps, src_fps)
        base = keep(raw_base)
        print(f"\n=== {name}")
        print(f"    {len(pos)} frames @ {src_fps:.2f} fps, "
              f"{len(raw_base)} reps at native (fixture recorded {len(d['reps'])})"
              + (f", {len(base)} kept" if len(base) != len(raw_base) else ""))
        if not base:
            print("    no reps survive the duration filter; skipping")
            continue

        for stride in args.strides:
            fps = src_fps / stride
            # Offsets give two independent samplings of the same motion; their
            # spread is this probe's own noise floor and must be read alongside
            # the error against native.
            for offset in range(stride):
                cfg = keep(analyse(pos[offset::stride], fps, src_fps))
                pairs, missed, spurious = match(base, cfg)
                if not pairs:
                    print(f"    stride {stride} (~{fps:.0f} fps) off{offset}: "
                          f"{len(cfg)} reps, NO MATCHES")
                    continue
                ds = [cfg[j]["start_s"] + offset / src_fps - base[i]["start_s"]
                      for i, j in pairs]
                de = [cfg[j]["end_s"] + offset / src_fps - base[i]["end_s"]
                      for i, j in pairs]
                dur = [rel(cfg[j]["duration_s"], base[i]["duration_s"]) for i, j in pairs]
                av = [rel(cfg[j]["avg_vel"], base[i]["avg_vel"]) for i, j in pairs]
                pv = [rel(cfg[j]["peak_vel"], base[i]["peak_vel"]) for i, j in pairs]
                print(f"    stride {stride} (~{fps:.0f} fps) off{offset}: "
                      f"{len(cfg)} reps, {len(pairs)} matched, "
                      f"{len(missed)} missed, {len(spurious)} spurious")
                print(f"        Δstart  max {max(map(abs, ds)) * 1000:5.0f} ms "
                      f"(bias {np.mean(ds) * 1000:+5.0f})   "
                      f"Δlockout max {max(map(abs, de)) * 1000:5.0f} ms "
                      f"(bias {np.mean(de) * 1000:+5.0f})")
                print(f"        duration med {np.median(dur):5.1%} max {max(dur):5.1%}   "
                      f"avg-vel max {max(av):5.1%}   peak-vel max {max(pv):5.1%}")


if __name__ == "__main__":
    main()
