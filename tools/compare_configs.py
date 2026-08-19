"""Stage-1 scorer: compares every config `tools/sweep_configs.py` produced
against the native-fps baseline.

The metric is the one pre-registered in
`barbell-velocity-app/RESOLUTION_DOWNSCALING.md` (written before any results
were seen, deliberately, so the gate can't be tuned to flatter a config):

  Tier 1  rep-level agreement — THE GATE. Config reps are matched to baseline
          reps greedily by temporal IoU over [start_s, end_s] at IoU >= 0.5.
          A spurious rep and a missed rep are counted separately: they are
          different failures to a lifter.
  Tier 2  diagnostics, never the gate — detection rate, NRMSE, Pearson r, and
          peak-velocity *bias* reported apart from spread (a systematic bias is
          correctable by retuning; scatter is not, and one RMS number hides
          which you have).
  Tier 3  cost — wall clock, ms/frame, frames, peak RSS, speedup.

Everything is compared in resolution-independent units (frame heights, frame
heights/second) on the baseline's *time* grid, so differing fps cannot misalign
indices.

Baseline: native fps, "normalized" arm. The native/fixed row is emitted too and
must come out exactly zero — at native fps the two arms are the same code path,
so a nonzero row there means the harness is broken, not the config.

Usage:
  python tools/compare_configs.py
  python tools/compare_configs.py --sweep sweep --out sweep/_compare
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- The PASS gate. Set before results were seen; do not retune downward. ---
GATE_MAX_BOUNDARY_S = 0.05   # ~3 frames at 60 fps
GATE_MAX_AVG_VEL_ERR = 0.05  # VBT acts on 10-20% velocity-loss thresholds
GATE_MAX_PEAK_VEL_ERR = 0.08  # looser: peak is one sample, first thing fps loss damages
IOU_MATCH = 0.5

ARMS = ("normalized", "fixed")
BASELINE_CONFIG = "native"
BASELINE_ARM = "normalized"


def load_run(run_dir):
    def js(*parts):
        path = os.path.join(run_dir, *parts)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    meta, timing = js("meta.json"), js("timing.json")
    if meta is None or timing is None:
        return None
    arms = {}
    for arm in ARMS:
        reps, signals = js(arm, "reps.json"), js(arm, "signals.json")
        if reps is not None and signals is not None:
            arms[arm] = {"reps": reps, "signals": signals}
    return {"meta": meta, "timing": timing, "arms": arms}


def temporal_iou(a, b):
    inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = (max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"]))
    return inter / union if union > 0 else 0.0


def match_reps(base_reps, cfg_reps):
    """Greedy highest-IoU-first matching. Greedy rather than optimal because
    reps are disjoint intervals in time — the assignment is unambiguous unless
    the config is already failing badly, in which case it fails either way."""
    pairs = sorted(
        ((temporal_iou(b, c), i, j) for i, b in enumerate(base_reps) for j, c in enumerate(cfg_reps)),
        key=lambda t: -t[0],
    )
    used_b, used_c, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < IOU_MATCH or i in used_b or j in used_c:
            continue
        used_b.add(i)
        used_c.add(j)
        matches.append((base_reps[i], cfg_reps[j], iou))
    missed = len(base_reps) - len(used_b)      # baseline rep with no config rep
    spurious = len(cfg_reps) - len(used_c)     # config rep with no baseline rep
    return matches, missed, spurious


def rel_err(cfg, base):
    return (cfg - base) / base if base else float("nan")


def _agg(values):
    if not values:
        return {"median": float("nan"), "max": float("nan")}
    a = np.abs(np.asarray(values, dtype=float))
    return {"median": float(np.median(a)), "max": float(a.max())}


def tier1(base_reps, cfg_reps):
    matches, missed, spurious = match_reps(base_reps, cfg_reps)
    d_start, d_end, e_avg, e_peak, e_disp, signed_peak = [], [], [], [], [], []
    for b, c, _ in matches:
        d_start.append(c["start_s"] - b["start_s"])
        d_end.append(c["end_s"] - b["end_s"])
        e_avg.append(rel_err(c["avg_vel_h_s"], b["avg_vel_h_s"]))
        e_peak.append(rel_err(c["peak_vel_h_s"], b["peak_vel_h_s"]))
        e_disp.append(rel_err(c["displacement_h"], b["displacement_h"]))
        signed_peak.append(rel_err(c["peak_vel_h_s"], b["peak_vel_h_s"]))

    boundary = _agg(d_start + d_end)
    out = {
        "base_reps": len(base_reps),
        "cfg_reps": len(cfg_reps),
        "rep_delta": len(cfg_reps) - len(base_reps),
        "matched": len(matches),
        "missed": missed,
        "spurious": spurious,
        "boundary_med_s": boundary["median"],
        "boundary_max_s": boundary["max"],
        "avg_vel_med": _agg(e_avg)["median"],
        "avg_vel_max": _agg(e_avg)["max"],
        "peak_vel_med": _agg(e_peak)["median"],
        "peak_vel_max": _agg(e_peak)["max"],
        "disp_med": _agg(e_disp)["median"],
        "disp_max": _agg(e_disp)["max"],
        "peak_vel_bias": float(np.mean(signed_peak)) if signed_peak else float("nan"),
    }

    reasons = []
    if out["rep_delta"] != 0:
        reasons.append(f"rep count {out['cfg_reps']} vs {out['base_reps']}")
    if missed:
        reasons.append(f"{missed} missed")
    if spurious:
        reasons.append(f"{spurious} spurious")
    if not (out["boundary_max_s"] <= GATE_MAX_BOUNDARY_S):
        reasons.append(f"boundary {out['boundary_max_s']:.3f}s")
    if not (out["avg_vel_max"] <= GATE_MAX_AVG_VEL_ERR):
        reasons.append(f"avg vel {out['avg_vel_max']:.1%}")
    if not (out["peak_vel_max"] <= GATE_MAX_PEAK_VEL_ERR):
        reasons.append(f"peak vel {out['peak_vel_max']:.1%}")
    out["pass"] = not reasons
    out["fail_reasons"] = "; ".join(reasons)
    return out


def tier2(base_sig, cfg_sig):
    """Diagnostics on the baseline's time grid. NRMSE is normalized by the
    baseline's p5-p95 range rather than its std: the signal is dominated by a
    few reps, so std understates the scale that matters."""
    bt = np.asarray(base_sig["t_s"], dtype=float)
    ct = np.asarray(cfg_sig["t_s"], dtype=float)
    # Only compare where the config actually has samples; extrapolating past
    # its last frame would invent agreement (or disagreement) that isn't there.
    lo, hi = ct[0], ct[-1]
    mask = (bt >= lo) & (bt <= hi)
    grid = bt[mask]

    out = {"detection_rate": cfg_sig["detection_rate"], "sg_window": cfg_sig["sg_window"]}
    for key, name in (("y_h", "pos"), ("v_h_s", "vel")):
        b = np.asarray(base_sig[key], dtype=float)[mask]
        c = np.interp(grid, ct, np.asarray(cfg_sig[key], dtype=float))
        rng = np.percentile(b, 95) - np.percentile(b, 5)
        rmse = float(np.sqrt(np.mean((c - b) ** 2)))
        out[f"{name}_nrmse"] = rmse / rng if rng else float("nan")
        if name == "vel":
            out["vel_pearson_r"] = float(np.corrcoef(b, c)[0, 1]) if b.size > 1 else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="sweep", help="sweep output root (default: sweep/)")
    parser.add_argument("--out", default=None, help="report dir (default: <sweep>/_compare)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    sweep_root = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    out_dir = args.out or os.path.join(sweep_root, "_compare")
    os.makedirs(out_dir, exist_ok=True)

    videos = sorted(d for d in os.listdir(sweep_root)
                    if os.path.isdir(os.path.join(sweep_root, d)) and not d.startswith("_"))
    if not videos:
        sys.exit(f"[ERROR] no sweep results under {sweep_root}/")

    rows = []
    for video in videos:
        vdir = os.path.join(sweep_root, video)
        runs = {c: load_run(os.path.join(vdir, c)) for c in sorted(os.listdir(vdir))
                if os.path.isdir(os.path.join(vdir, c))}
        runs = {k: v for k, v in runs.items() if v}
        base = runs.get(BASELINE_CONFIG)
        if not base or BASELINE_ARM not in base["arms"]:
            print(f"[WARN] {video}: no {BASELINE_CONFIG}/{BASELINE_ARM} baseline — skipped")
            continue
        base_reps = base["arms"][BASELINE_ARM]["reps"]
        base_sig = base["arms"][BASELINE_ARM]["signals"]
        base_cost = base["timing"]["total_s"]

        for config, run in runs.items():
            for arm, data in run["arms"].items():
                if config == BASELINE_CONFIG and arm == BASELINE_ARM:
                    continue
                row = {"video": video, "config": config, "arm": arm,
                       "fps": run["meta"]["fps"]}
                row.update(tier1(base_reps, data["reps"]))
                row.update(tier2(base_sig, data["signals"]))
                t = run["timing"]
                row.update({
                    "frames": t["frames"],
                    "extract_s": t["extract_s"],
                    "model_s": t["model_s"],
                    "total_s": t["total_s"],
                    "model_ms_per_frame": t["model_ms_per_frame"],
                    "peak_rss_mb": t["peak_rss_mb"],
                    "speedup": base_cost / t["total_s"] if t["total_s"] else float("nan"),
                })
                rows.append(row)

    if not rows:
        sys.exit("[ERROR] nothing to compare")

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- Roll up per (config, arm) across videos: worst clip decides. ---
    combos = sorted({(r["config"], r["arm"], r["fps"]) for r in rows}, key=lambda c: -c[2])
    summary = []
    for config, arm, fps in combos:
        rs = [r for r in rows if r["config"] == config and r["arm"] == arm]
        failed = [r for r in rs if not r["pass"]]
        summary.append({
            "config": config, "arm": arm, "fps": fps,
            "clips": len(rs),
            "clips_passed": len(rs) - len(failed),
            "pass": not failed,
            "boundary_max_s": max(r["boundary_max_s"] for r in rs),
            "avg_vel_max": max(r["avg_vel_max"] for r in rs),
            "peak_vel_max": max(r["peak_vel_max"] for r in rs),
            "peak_vel_bias": float(np.mean([r["peak_vel_bias"] for r in rs])),
            "vel_nrmse_max": max(r["vel_nrmse"] for r in rs),
            "vel_r_min": min(r["vel_pearson_r"] for r in rs),
            "det_min": min(r["detection_rate"] for r in rs),
            "speedup": float(np.mean([r["speedup"] for r in rs])),
            "fail_clips": ", ".join(sorted({r["video"] for r in failed})),
        })

    lines = ["# Stage 1 — FPS sweep results", "",
             f"Baseline: `{BASELINE_CONFIG}` / `{BASELINE_ARM}` per clip. "
             f"{len(videos)} clip(s). Gate (pre-registered): boundary <= {GATE_MAX_BOUNDARY_S}s, "
             f"avg vel <= {GATE_MAX_AVG_VEL_ERR:.0%}, peak vel <= {GATE_MAX_PEAK_VEL_ERR:.0%}, "
             "rep count equal with no unmatched reps either direction.", "",
             "## Pareto — speedup vs accuracy (worst clip decides)", "",
             "| config | arm | fps | speedup | clips passed | max Δboundary | max avg-vel err | "
             "max peak-vel err | peak-vel bias | vel r (min) | PASS |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda s: (-s["speedup"],)):
        lines.append(
            f"| {s['config']} | {s['arm']} | {s['fps']:.3g} | {s['speedup']:.2f}× | "
            f"{s['clips_passed']}/{s['clips']} | {s['boundary_max_s']:.3f}s | "
            f"{s['avg_vel_max']:.1%} | {s['peak_vel_max']:.1%} | {s['peak_vel_bias']:+.1%} | "
            f"{s['vel_r_min']:.3f} | {'**PASS**' if s['pass'] else 'fail'} |")

    passing = [s for s in summary if s["pass"]]
    lines += ["", "## Selection", ""]
    if passing:
        best = max(passing, key=lambda s: s["speedup"])
        lines.append(f"Cheapest config passing on **every** clip: **{best['config']} / "
                     f"{best['arm']}** at {best['fps']:.3g} fps ({best['speedup']:.2f}× faster).")
    else:
        lines.append("**No config passes the gate on every clip.** Nearest misses, "
                     "by clips passed then speedup:")
        for s in sorted(summary, key=lambda s: (-s["clips_passed"], -s["speedup"]))[:5]:
            lines.append(f"- `{s['config']}/{s['arm']}` — {s['clips_passed']}/{s['clips']} clips, "
                         f"{s['speedup']:.2f}×, failing on: {s['fail_clips']}")

    lines += ["", "## Per-clip failures", "",
              "| video | config | arm | why |", "|---|---|---|---|"]
    for r in rows:
        if not r["pass"]:
            lines.append(f"| {r['video']} | {r['config']} | {r['arm']} | {r['fail_reasons']} |")

    md_path = os.path.join(out_dir, "REPORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        for video in videos:
            vdir = os.path.join(sweep_root, video)
            runs = {c: load_run(os.path.join(vdir, c)) for c in sorted(os.listdir(vdir))
                    if os.path.isdir(os.path.join(vdir, c))}
            runs = {k: v for k, v in runs.items() if v}
            if BASELINE_CONFIG not in runs:
                continue
            fig, axes = plt.subplots(len(ARMS), 1, figsize=(14, 4 * len(ARMS)), sharex=True)
            for ax, arm in zip(np.atleast_1d(axes), ARMS):
                b = runs[BASELINE_CONFIG]["arms"][BASELINE_ARM]["signals"]
                ax.plot(b["t_s"], b["v_h_s"], color="white", linewidth=2, label="baseline (native)")
                for config, run in sorted(runs.items(), key=lambda kv: -kv[1]["meta"]["fps"]):
                    if config == BASELINE_CONFIG or arm not in run["arms"]:
                        continue
                    s = run["arms"][arm]["signals"]
                    ax.plot(s["t_s"], s["v_h_s"], linewidth=1, alpha=0.85,
                            label=f"{config} ({run['meta']['fps']:.3g} fps)")
                ax.set_title(f"{video} — velocity, {arm} constants")
                ax.set_ylabel("frame heights / s")
                ax.legend(fontsize=8)
            plt.xlabel("time (s)")
            plt.tight_layout()
            png = os.path.join(out_dir, f"{video}_velocity.png")
            plt.savefig(png, dpi=110)
            plt.close()

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print()
    print("\n".join(lines[6:6 + len(summary) + 3]))


if __name__ == "__main__":
    main()
