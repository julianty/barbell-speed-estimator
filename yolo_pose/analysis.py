# analysis.py — legacy YOLO-pose pipeline analysis

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

from detect_concentric import detect_concentric
from utils import compute_velocity, remove_outliers, smooth_positions

ENABLE_CONCENTRIC = True


def load_track(track_path):
    with open(track_path, "r") as f:
        return json.load(f)


def extract_positions(track):
    positions = []
    for entry in track:
        center = entry["center"]
        if center is None:
            positions.append(None)
        else:
            positions.append(-center[1])
    return positions


# ---------------------------------------------------------
# Plot raw + smoothed + optional concentric shading
# ---------------------------------------------------------
def plot_analysis(raw_pos, smooth_pos, raw_vel, smooth_vel, concentric, out_path, segments=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if ENABLE_CONCENTRIC:
        start, end = concentric
    else:
        start, end = None, None

    plt.figure(figsize=(14, 8))

    # --- Position ---
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(raw_pos, label="Raw Position", alpha=0.4)
    ax1.plot(smooth_pos, label="Smoothed Position (S-G)", linewidth=2)

    # Draw all ruptures segments for debugging
    if segments:
        y_lo, y_hi = ax1.get_ylim()
        y_range = y_hi - y_lo
        for seg in segments:
            color = "green" if seg["selected"] else ("orange" if seg["skipped_short"] else "steelblue")
            alpha = 0.25 if seg["selected"] else 0.10
            ax1.axvspan(seg["start"], seg["end"], color=color, alpha=alpha)
            ax1.axvline(seg["start"], color=color, linestyle=":", linewidth=0.8)
            label = f'{seg["displacement"]:+.0f}px\n({seg["frames"]}f)'
            mid = (seg["start"] + seg["end"]) / 2
            ax1.text(mid, y_lo + y_range * 0.05, label,
                     fontsize=6, ha="center", va="bottom", color=color,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7))
        # Final boundary line
        ax1.axvline(segments[-1]["end"], color="steelblue", linestyle=":", linewidth=0.8)

    if start is not None:
        ax1.axvspan(start, end, color="green", alpha=0.15, label="Concentric Phase")
        ax1.axvline(start, color="green", linestyle="--", linewidth=1.4)
        ax1.axvline(end, color="darkgreen", linestyle="--", linewidth=1.4)
        y_top = ax1.get_ylim()[1]
        ax1.annotate("Start", xy=(start, y_top), xytext=(start + 1, y_top),
                     fontsize=8, color="green", va="top")
        ax1.annotate("Lockout", xy=(end, y_top), xytext=(end + 1, y_top),
                     fontsize=8, color="darkgreen", va="top")

    ax1.set_title("Barbell Vertical Position (px, upward = positive)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Position")
    ax1.legend()

    # --- Velocity ---
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(raw_vel, label="Raw Velocity", alpha=0.4)
    ax2.plot(smooth_vel, label="Smoothed Velocity (S-G)", linewidth=2)

    if start is not None:
        ax2.axvspan(start, end, color="green", alpha=0.15, label="Concentric Phase")
        ax2.axvline(start, color="green", linestyle="--", linewidth=1.2)
        ax2.axvline(end, color="darkgreen", linestyle="--", linewidth=1.2)
        avg_vel = float(np.mean(smooth_vel[start:end + 1]))
        ax2.text(0.02, 0.95, f"Avg concentric velocity: {avg_vel:.1f} px/s",
                 transform=ax2.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.85))

    ax2.set_title("Barbell Velocity (px/s)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Velocity")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved analysis plot to {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="YOLO-pose analysis and plotting")
    parser.add_argument("--track", default=None,
                        help="Path to barbell_track.json (default: newest under pose_results/)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second (default: 30)")
    args = parser.parse_args()

    if args.track is None:
        candidates = glob.glob("pose_results/*/*/barbell_track.json")
        if not candidates:
            sys.exit("[ERROR] No barbell_track.json found under pose_results/. "
                     "Run pose_velocity.py first or pass --track.")
        args.track = max(candidates, key=os.path.getmtime)
        print(f"[INFO] Auto-selected: {args.track}")

    TRACK_PATH = args.track
    OUTPUT_PATH = os.path.join(os.path.dirname(TRACK_PATH), "bar_speed.png")
    FPS = args.fps

    track = load_track(TRACK_PATH)

    # 1. Raw positions
    raw_positions = extract_positions(track)

    # 2. Remove outliers BEFORE smoothing
    # window: half-width of median comparison; threshold: max deviation (px) before a point is replaced
    clean_positions = remove_outliers(raw_positions, window=3, threshold=50)

    # 3. Smooth the cleaned signal
    # window: S-G filter length in frames (must be odd, larger = smoother); poly: polynomial order
    smooth_positions_arr = smooth_positions(clean_positions, window=9, poly=3)

    # 4. Compute velocities
    raw_velocity = compute_velocity(clean_positions, FPS)
    smooth_velocity = compute_velocity(smooth_positions_arr, FPS)

    # 5. Optional concentric detection
    if ENABLE_CONCENTRIC:
        # pen: ruptures PELT penalty — higher = fewer, larger segments; lower = more splits
        pen = 20
        start, end, segments = detect_concentric(smooth_positions_arr, pen=pen, return_segments=True)
        concentric_phase = (start, end)
        print(f"[DEBUG] {len(segments)} segments detected (pen={pen}):")
        for i, s in enumerate(segments):
            if s["selected"]:
                tag = " <-- SELECTED"
            elif s["skipped_short"]:
                tag = " (short)"
            elif s.get("no_eccentric_following"):
                tag = " (no eccentric follows)"
            else:
                tag = ""
            print(f"  [{i}] frames {s['start']:>4}-{s['end']:>4}  "
                  f"len={s['frames']:>4}  disp={s['displacement']:>+8.1f}px{tag}")
        if start is not None:
            avg_vel = float(np.mean(smooth_velocity[start:end + 1]))
            print(f"[INFO] Avg concentric velocity: {avg_vel:.1f} px/s (frames {start}-{end})")
    else:
        concentric_phase = (None, None)
        segments = None

    # 6. Plot
    plot_analysis(clean_positions, smooth_positions_arr,
                  raw_velocity, smooth_velocity,
                  concentric_phase, OUTPUT_PATH, segments=segments)
