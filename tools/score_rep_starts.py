"""score_rep_starts.py — score each run's rep starts against hand labels.

Reads truth_start_s from sweep/_ground_truth_starts/labels.json (written by
tools/label_rep_starts.py and filled in by hand) and from the older
sweep/_ground_truth_lifts/disputed_reps.json. Each labelled rep is located in a
run as the rep that overlaps the full-res YOLO rep (sweep/<clip>/fps30) the
label was cut from. Errors are run start minus truth, in ms; negative = early.

    python tools/score_rep_starts.py
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS = ["fps30", "fps30_yolo26s_640", "fps30_vision", "fps30_vision_640",
                "fps30_movenet_thunder", "fps30_movenet_thunder_640"]


def overlap(a, b):
    return min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=os.path.join(ROOT, "sweep"))
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--arm", default="normalized")
    a = p.parse_args()

    new = json.load(open(os.path.join(a.sweep, "_ground_truth_starts", "labels.json")))
    old = json.load(open(os.path.join(a.sweep, "_ground_truth_lifts", "disputed_reps.json")))
    truth = {}
    for x in old + new:  # new labels win on a clash
        if x.get("truth_start_s") is not None and os.path.isdir(os.path.join(a.sweep, x["video"], a.runs[0])):
            truth[(x["video"], x["rep"])] = x["truth_start_s"]

    errs = {r: [] for r in a.runs}
    print("clip rep truth_s " + " ".join(a.runs))
    for (v, rep), t in sorted(truth.items()):
        reps0 = json.load(open(os.path.join(a.sweep, v, a.runs[0], a.arm, "reps.json")))
        ref = next(x for x in reps0 if x["rep"] == rep)
        row = []
        for r in a.runs:
            reps = json.load(open(os.path.join(a.sweep, v, r, a.arm, "reps.json")))
            c = max(reps, key=lambda c: overlap(ref, c))
            e = 1000 * (c["start_s"] - t)
            errs[r].append(e)
            row.append(f"{e:+.0f}")
        print(f"{v} {rep} {t:.2f} " + " ".join(row))

    print(f"\n| run | n | median abs err | p90 | max | mean (signed) | within 100 ms |")
    print("|---|---|---|---|---|---|---|")
    for r in a.runs:
        e = np.array(errs[r])
        ab = np.abs(e)
        print(f"| {r} | {len(e)} | {np.median(ab):.0f} | {np.percentile(ab, 90):.0f} | {ab.max():.0f} "
              f"| {e.mean():+.0f} | {(ab <= 100).sum()}/{len(e)} |")


if __name__ == "__main__":
    main()
