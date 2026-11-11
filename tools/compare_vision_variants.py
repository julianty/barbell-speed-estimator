"""compare_vision_variants.py — bar point from Vision's body-2D, body-3D and hand requests.

Reads sweep/<clip>/fps30_vision_variants_640/variants.json from
tools/vision_variants.swift and builds one bar-proxy track per source:

  body2d   the app's rule: torso selection, offset at wrist confidence 0.5
  body3d   3D body pose's wrists projected into the image, torso selection.
           It has no per-joint confidence, so both wrists are always used.
  hand     hand pose's wrist joints, confidence >= --hand-conf. Keeps the
           two hands nearest the last bar point (seeded from body2d's first
           point), splits them by x into the two sides, and fills a missing
           side with the offset rule.
  fill     body2d, with a lost wrist replaced by the hand wrist nearest it,
           using the last position of that side

Each track goes through the same analysis. It reports reps against the
full-res YOLO reference, the share of frames with a usable point, and jitter.
With --save, tracks go to <run>/variants/<source>.json for review clips.

    python tools/compare_vision_variants.py --video IMG_4223 --video IMG_4224
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "yolo_pose")]
from compare_wrist_rules import build, overlap, wrist_states  # noqa: E402
from sweep_configs import analyse  # noqa: E402

SOURCES = ["body2d", "body3d", "hand", "fill"]


def scale_of(v, frame_size, sweep):
    src = json.load(open(os.path.join(sweep, v, "fps30", "meta.json")))["frame_size"]
    return src, src[0] / frame_size[0], src[1] / frame_size[1]


def nearest(pts, p):
    return min(pts, key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def body3d_track(var, sx, sy):
    pts, last = [], None
    for fr in var["frames"]:
        ps = [p for p in fr["body3d"] if "lw" in p and "rw" in p]
        if not ps:
            pts.append(None)
            continue
        if last is None:
            p = max(ps, key=lambda p: p.get("height_m", 0))
        else:
            p = min(ps, key=lambda p: (p.get("torso", [0, 0])[0] * sx - last[0]) ** 2
                    + (p.get("torso", [0, 0])[1] * sy - last[1]) ** 2)
        if "torso" in p:
            last = (p["torso"][0] * sx, p["torso"][1] * sy)
        pts.append(((p["lw"][0] + p["rw"][0]) / 2 * sx, (p["lw"][1] + p["rw"][1]) / 2 * sy))
    return pts


def hand_states(var, sx, sy, seed, conf):
    """(A, B) per frame: the two sides of the grip, by x, from hand wrists."""
    states, last_mid, last_side = [], seed, [None, None]
    for fr in var["frames"]:
        hs = [(h["wrist"][0] * sx, h["wrist"][1] * sy) for h in fr["hand"] if h["wrist"][2] >= conf]
        if last_mid is not None and len(hs) > 2:
            hs = sorted(hs, key=lambda q: (q[0] - last_mid[0]) ** 2 + (q[1] - last_mid[1]) ** 2)[:2]
        if len(hs) == 2:
            a, b = sorted(hs)
            states.append((a, b))
            last_side = [a, b]
            last_mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        elif len(hs) == 1:
            h = hs[0]
            # Assign to the side whose last position is nearer.
            if last_side[0] is not None and last_side[1] is not None:
                da = np.hypot(h[0] - last_side[0][0], h[1] - last_side[0][1])
                db = np.hypot(h[0] - last_side[1][0], h[1] - last_side[1][1])
                side = 0 if da <= db else 1
            else:
                side = 0 if last_mid is None or h[0] < last_mid[0] else 1
            states.append((h, None) if side == 0 else (None, h))
            last_side[side] = h
        else:
            states.append((None, None))
    return [(a, b, None) for a, b in states]


def fill_states(body_states, var, sx, sy, conf):
    out, last = [], [None, None]
    for (L, R, raw), fr in zip(body_states, var["frames"]):
        hs = [(h["wrist"][0] * sx, h["wrist"][1] * sy) for h in fr["hand"] if h["wrist"][2] >= conf]
        side = [L, R]
        for k in (0, 1):
            if side[k] is None and hs and last[k] is not None:
                c = nearest(hs, last[k])
                other = side[1 - k]
                if np.hypot(c[0] - last[k][0], c[1] - last[k][1]) < 80 and (
                        other is None or np.hypot(c[0] - other[0], c[1] - other[1]) > 60):
                    side[k] = c
            if side[k] is not None:
                last[k] = side[k]
        out.append((side[0], side[1], raw))
    return out


def jitter(pts, h):
    y = np.array([np.nan if p is None else p[1] for p in pts])
    ok = ~np.isnan(y)
    if ok.sum() < 15:
        return float("nan")
    yy = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    return float(np.std(yy - savgol_filter(yy, 9, 3)) / h * 1000)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--video", action="append", required=True)
    p.add_argument("--hand-conf", type=float, default=0.3)
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    for v in a.video:
        run = os.path.join(a.sweep, v, "fps30_vision_variants_640")
        var = json.load(open(os.path.join(run, "variants.json")))
        src, sx, sy = scale_of(v, var["frame_size"], a.sweep)
        meta = json.load(open(os.path.join(a.sweep, v, "fps30", "meta.json")))
        ref = json.load(open(os.path.join(a.sweep, v, "fps30", "normalized", "reps.json")))
        kp = {"frame_size": var["frame_size"],
              "frames": [{"frame": f["frame"], "people": f["body2d"]} for f in var["frames"]]}
        body = wrist_states(kp, src, 0.5, select="torso")
        tracks = {"body2d": build(body, "offset")}
        tracks["body3d"] = [{"frame": i, "center": None if q is None else [q[0], q[1]]}
                            for i, q in enumerate(body3d_track(var, sx, sy))]
        seed = next((t["center"] for t in tracks["body2d"] if t["center"] is not None), None)
        tracks["hand"] = build(hand_states(var, sx, sy, seed, a.hand_conf), "offset")
        tracks["fill"] = build(fill_states(body, var, sx, sy, a.hand_conf), "offset")

        n = len(var["frames"])
        ms = {k: 1000 * s / n for k, s in var["seconds"].items()}
        print(f"\n{v}  (Mac ms/frame: body2d {ms['body2d']:.1f}, body3d {ms['body3d']:.1f}, "
              f"hand {ms['hand']:.1f})")
        print("  YOLO full-res: " + " ".join(f"{r['start_s']:.1f}-{r['end_s']:.1f}" for r in ref))
        for s in SOURCES:
            tr = tracks[s]
            reps, _ = analyse(tr, 30.0, meta["source_fps"], src[1], scale_constants=True)
            cov = np.mean([t["center"] is not None for t in tr])
            matched = sum(any(overlap(r, c) > 0 for c in reps) for r in ref)
            extra = sum(not any(overlap(r, c) > 0 for r in ref) for c in reps)
            print(f"  {s:7} {matched}/{len(ref)} +{extra} fake  coverage {cov:4.0%}  "
                  f"jitter {jitter([t['center'] for t in tr], src[1]):5.2f}  | "
                  + " ".join(f"{r['start_s']:.1f}-{r['end_s']:.1f}" for r in reps))
            if a.save:
                os.makedirs(os.path.join(run, "variants"), exist_ok=True)
                json.dump(tr, open(os.path.join(run, "variants", f"{s}.json"), "w"))


if __name__ == "__main__":
    main()
