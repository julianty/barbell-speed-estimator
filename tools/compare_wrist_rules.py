"""compare_wrist_rules.py — how to build the bar proxy when a wrist is lost.

Vision drops joints it cannot see instead of guessing. On squats it loses one
wrist in more than half the frames, for gaps of up to 4.4 s. This compares
rules for turning per-wrist keypoints into the bar-proxy point, on the saved
keypoints (no inference), scored against the hand-labelled rep starts.

Rules (a wrist is "lost" below --min-conf):
  ignore    average both wrists whatever their confidence. The app's current
            rule (reference behaviour); with Vision a lost wrist sits at (0, 0).
  fallback  one wrist lost: use the other wrist as is. Steps by half the
            wrist-to-wrist offset each time a wrist drops out or returns.
  drop      one wrist lost: no point; the analysis interpolates across the gap.
  offset    one wrist lost: visible wrist + that wrist's offset to the midpoint
            from the nearest frame where both were seen. No step, since both
            hands are on one rigid bar.
  hybridK   drop for one-wrist gaps of at most K frames, offset for longer ones.
Both wrists lost is no point under every rule except ignore.

    python tools/compare_wrist_rules.py
    python tools/compare_wrist_rules.py --runs fps30_vision_640 --rules offset hybrid3
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "yolo_pose"))

from build_barbell_track import MAX_JUMP_FRACTION  # noqa: E402
from sweep_configs import analyse  # noqa: E402

LW, RW = 9, 10
FPS = 30.0
VIDEOS = [f"deadlift{i}" for i in range(1, 7)] + ["squat2", "squat3"]
DEFAULT_RULES = ["ignore", "fallback", "drop", "offset", "hybrid3", "hybrid6"]


def select_person(people, last, sx, sy, min_conf):
    """build_barbell_track's selection: one person as is; else nearest wrist
    point to the last center; else the largest box. Uses visible wrists only."""
    cands = []
    for p in people:
        l, r = p["kp"][LW], p["kp"][RW]
        vis = [w for w in (l, r) if w[2] >= min_conf]
        if not vis:
            continue
        m = (np.mean([w[0] for w in vis]) * sx, np.mean([w[1] for w in vis]) * sy)
        b = p["box"]
        cands.append((p, m, (b[2] - b[0]) * (b[3] - b[1])))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][0]
    if last is not None:
        return min(cands, key=lambda c: (c[1][0] - last[0]) ** 2 + (c[1][1] - last[1]) ** 2)[0]
    return max(cands, key=lambda c: c[2])[0]


def torso_point(p, sx, sy, min_conf=0.1):
    """Mean of the visible shoulders and hips, else the box centre: where the
    person IS, independent of whether their wrists are visible."""
    pts = [p["kp"][j] for j in (5, 6, 11, 12) if p["kp"][j][2] >= min_conf]
    if pts:
        return (np.mean([q[0] for q in pts]) * sx, np.mean([q[1] for q in pts]) * sy)
    b = p["box"]
    return ((b[0] + b[2]) / 2 * sx, (b[1] + b[3]) / 2 * sy)


def select_by_torso(people, last_torso, sx, sy):
    """Follow a body, not a pair of wrists: the person whose torso is nearest
    the last selected torso; on the first frame, the largest box. A lifter
    whose wrists are hidden stays selected, instead of handing the track to
    whoever's wrists are visible."""
    if not people:
        return None
    if last_torso is None:
        return max(people, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
    return min(people, key=lambda p: (torso_point(p, sx, sy)[0] - last_torso[0]) ** 2
               + (torso_point(p, sx, sy)[1] - last_torso[1]) ** 2)


def wrist_states(kp_json, src_size, min_conf, select="wrist"):
    """Per frame: (L, R) in source px, each None if lost, for the selected
    person. Selection follows the visible-wrist point, so it is shared by every
    rule except ignore."""
    w, h = src_size
    sx, sy = w / kp_json["frame_size"][0], h / kp_json["frame_size"][1]
    max_jump = MAX_JUMP_FRACTION * (w ** 2 + h ** 2) ** 0.5
    out, last, last_torso = [], None, None
    for fr in kp_json["frames"]:
        if select == "torso":
            p = select_by_torso(fr["people"], last_torso, sx, sy)
            if p is None:
                out.append((None, None, None))
                continue
            last_torso = torso_point(p, sx, sy)
            l, r = p["kp"][LW], p["kp"][RW]
            L = (l[0] * sx, l[1] * sy) if l[2] >= min_conf else None
            R = (r[0] * sx, r[1] * sy) if r[2] >= min_conf else None
            out.append((L, R, ((l[0] + r[0]) / 2 * sx, (l[1] + r[1]) / 2 * sy)))
            continue
        p = select_person(fr["people"], last, sx, sy, min_conf)
        if p is None:
            out.append((None, None, None))
            continue
        l, r = p["kp"][LW], p["kp"][RW]
        L = (l[0] * sx, l[1] * sy) if l[2] >= min_conf else None
        R = (r[0] * sx, r[1] * sy) if r[2] >= min_conf else None
        raw = ((l[0] + r[0]) / 2 * sx, (l[1] + r[1]) / 2 * sy)  # for "ignore"
        vis = [q for q in (L, R) if q is not None]
        m = (np.mean([q[0] for q in vis]), np.mean([q[1] for q in vis]))
        if last is not None and np.hypot(m[0] - last[0], m[1] - last[1]) > max_jump:
            out.append((None, None, None))
            continue
        last = m
        out.append((L, R, raw))
    return out


def nearest_offsets(states):
    """For each frame, the (mid - L) and (mid - R) offsets from the nearest
    frame in time where both wrists were seen."""
    both = [i for i, (L, R, _) in enumerate(states) if L is not None and R is not None]
    offs = [None] * len(states)
    if not both:
        return offs
    both_arr = np.array(both)
    for i in range(len(states)):
        j = both[int(np.argmin(np.abs(both_arr - i)))]
        L, R, _ = states[j]
        mid = ((L[0] + R[0]) / 2, (L[1] + R[1]) / 2)
        offs[i] = ((mid[0] - L[0], mid[1] - L[1]), (mid[0] - R[0], mid[1] - R[1]))
    return offs


def build(states, rule):
    offs = nearest_offsets(states)
    pts = []
    for i, (L, R, raw) in enumerate(states):
        if rule == "ignore":
            pts.append(raw)
        elif L is not None and R is not None:
            pts.append(((L[0] + R[0]) / 2, (L[1] + R[1]) / 2))
        elif L is None and R is None:
            pts.append(None)
        elif rule == "fallback":
            pts.append(L or R)
        elif rule == "drop":
            pts.append(None)
        else:  # offset / hybridK
            if offs[i] is None:
                pts.append(L or R)
            elif L is not None:
                pts.append((L[0] + offs[i][0][0], L[1] + offs[i][0][1]))
            else:
                pts.append((R[0] + offs[i][1][0], R[1] + offs[i][1][1]))
    if rule.startswith("hybrid"):
        k = int(rule[len("hybrid"):])
        one = [(L is None) != (R is None) for L, R, _ in states]
        i = 0
        while i < len(one):
            if one[i]:
                j = i
                while j < len(one) and one[j]:
                    j += 1
                if j - i <= k:
                    for q in range(i, j):
                        pts[q] = None
                i = j
            else:
                i += 1
    return [{"frame": i, "center": None if p is None else [float(p[0]), float(p[1])]}
            for i, p in enumerate(pts)]


def overlap(a, b):
    return min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--runs", nargs="+", default=["fps30_vision_640", "fps30_yolo26s_640"])
    p.add_argument("--rules", nargs="+", default=DEFAULT_RULES)
    p.add_argument("--min-conf", type=float, default=0.1)
    p.add_argument("--save-tracks", action="store_true",
                   help="write <run>/rules/<rule>.json tracks, for tools/render_wrist_review.py")
    a = p.parse_args()

    labels = json.load(open(os.path.join(a.sweep, "_ground_truth_starts", "labels.json")))
    old = json.load(open(os.path.join(a.sweep, "_ground_truth_lifts", "disputed_reps.json")))
    t_start, t_end = {}, {}
    for x in old + labels:
        if x["video"] not in VIDEOS:
            continue
        if x.get("truth_start_s") is not None:
            t_start[(x["video"], x["rep"])] = x["truth_start_s"]
        if x.get("truth_end_s") is not None:
            t_end[(x["video"], x["rep"])] = x["truth_end_s"]

    print(f"labels: {len(t_start)} starts, {len(t_end)} lockouts; "
          f"rep counts are against full-res YOLO's 27 reps\n")
    print("| run | rule | reps (missed / extra) | start: median abs / p90 / max (ms) | mean | "
          "squat starts median abs (n) | lockout abs median / max (n) |")
    print("|---|---|---|---|---|---|---|")
    for run in a.runs:
        for rule in a.rules:
            se, se_sq, ee = [], [], []
            missed = extra = 0
            for v in VIDEOS:
                kp = json.load(open(os.path.join(a.sweep, v, run, "keypoints.json")))
                meta = json.load(open(os.path.join(a.sweep, v, "fps30", "meta.json")))
                src = meta["frame_size"]
                states = wrist_states(kp, src, a.min_conf)
                track = build(states, rule)
                if a.save_tracks:
                    d = os.path.join(a.sweep, v, run, "rules")
                    os.makedirs(d, exist_ok=True)
                    # Default threshold keeps the plain name; others are suffixed.
                    tag = rule if a.min_conf == 0.1 else f"{rule}_c{a.min_conf:g}"
                    json.dump(track, open(os.path.join(d, f"{tag}.json"), "w"))
                reps, _ = analyse(track, FPS, meta["source_fps"], src[1], scale_constants=True)
                ref = json.load(open(os.path.join(a.sweep, v, "fps30", "normalized", "reps.json")))
                used = set()
                for r in ref:
                    hits = [i for i, c in enumerate(reps) if overlap(r, c) > 0 and i not in used]
                    if not hits:
                        missed += 1
                        continue
                    i = max(hits, key=lambda i: overlap(r, reps[i]))
                    used.add(i)
                    c = reps[i]
                    if (v, r["rep"]) in t_start:
                        e = 1000 * (c["start_s"] - t_start[(v, r["rep"])])
                        se.append(e)
                        if v.startswith("squat"):
                            se_sq.append(e)
                    if (v, r["rep"]) in t_end:
                        ee.append(1000 * (c["end_s"] - t_end[(v, r["rep"])]))
                extra += len(reps) - len(used)
            ab = np.abs(se)
            print(f"| {run} | {rule} | {27 - missed}/27 ({missed} / {extra}) "
                  f"| {np.median(ab):.0f} / {np.percentile(ab, 90):.0f} / {ab.max():.0f} "
                  f"| {np.mean(se):+.0f} | {np.median(np.abs(se_sq)):.0f} ({len(se_sq)}) "
                  f"| {np.median(np.abs(ee)):.0f} / {np.max(np.abs(ee)):.0f} ({len(ee)}) |")


if __name__ == "__main__":
    main()
