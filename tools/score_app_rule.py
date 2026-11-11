"""score_app_rule.py — score Vision runs under the app's exact bar-point rule.

The app (D25) selects the lifter by torso and fills a lost wrist with the
offset rule at confidence 0.5. compare_pose_backends.py's built-in scoring
uses the older harness rule, so this re-scores saved keypoints the app's way:

  - reps against the full-res YOLO reference (sweep/<clip>/fps30): matched,
    fake (no overlapping YOLO rep)
  - coverage: share of frames with a bar point
  - rep-start error against the hand labels, where a clip has them

    python tools/score_app_rule.py --runs fps30_vision_640 fps30_vision_pad25_640
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "yolo_pose")]
from compare_wrist_rules import build, overlap, wrist_states  # noqa: E402
from sweep_configs import analyse  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--runs", nargs="+", default=["fps30_vision_640", "fps30_vision_pad25_640"])
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--per-clip", action="store_true")
    a = p.parse_args()

    clips = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "videos", "*.mov")))
    clips = [c for c in clips if all(os.path.exists(os.path.join(a.sweep, c, r, "keypoints.json"))
                                     for r in a.runs)]
    labels = []
    for f in ("_ground_truth_starts/labels.json", "_ground_truth_lifts/disputed_reps.json"):
        path = os.path.join(a.sweep, f)
        if os.path.exists(path):
            labels += json.load(open(path))
    truth = {(x["video"], x["rep"]): x["truth_start_s"] for x in labels if x.get("truth_start_s") is not None}

    print(f"{len(clips)} clips; rule: torso selection, offset at {a.min_conf}\n")
    print("| run | group | reps matched | fake | coverage | labelled starts: median / p90 / max ms (n) |")
    print("|---|---|---|---|---|---|")
    per = []
    for run in a.runs:
        groups = {}
        for c in clips:
            g = "bench" if "bench" in c else "deadlift/squat"
            meta = json.load(open(os.path.join(a.sweep, c, "fps30", "meta.json")))
            ref = json.load(open(os.path.join(a.sweep, c, "fps30", "normalized", "reps.json")))
            kp = json.load(open(os.path.join(a.sweep, c, run, "keypoints.json")))
            st = wrist_states(kp, meta["frame_size"], a.min_conf, select="torso")
            tr = build(st, "offset")
            reps, _ = analyse(tr, 30.0, meta["source_fps"], meta["frame_size"][1], scale_constants=True)
            used, m = set(), 0
            for r in ref:
                hits = [i for i, x in enumerate(reps) if overlap(r, x) > 0 and i not in used]
                if not hits:
                    continue
                i = max(hits, key=lambda i: overlap(r, reps[i]))
                used.add(i)
                m += 1
                if (c, r["rep"]) in truth:
                    groups.setdefault(g, {}).setdefault("se", []).append(
                        1000 * (reps[i]["start_s"] - truth[(c, r["rep"])]))
            cov = float(np.mean([t["center"] is not None for t in tr]))
            G = groups.setdefault(g, {})
            G["n"] = G.get("n", 0) + len(ref)
            G["m"] = G.get("m", 0) + m
            G["fake"] = G.get("fake", 0) + len(reps) - len(used)
            G.setdefault("cov", []).append(cov)
            per.append((run, c, len(ref), m, len(reps) - len(used), cov))
        for g in ("bench", "deadlift/squat"):
            if g not in groups:
                continue
            G = groups[g]
            se = np.abs(G.get("se", []))
            st = (f"{np.median(se):.0f} / {np.percentile(se, 90):.0f} / {se.max():.0f} ({len(se)})"
                  if len(se) else "–")
            print(f"| {run} | {g} | {G['m']}/{G['n']} | {G['fake']} | {np.mean(G['cov']):.0%} | {st} |")
    if a.per_clip:
        print()
        for run, c, n, m, fake, cov in per:
            print(f"  {run:28} {c:30} {m}/{n} +{fake}  coverage {cov:.0%}")


if __name__ == "__main__":
    main()
