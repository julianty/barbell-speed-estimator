# analysis.py — legacy YOLO-pose pipeline analysis

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

from detect_concentric import detect_concentric
from utils import compute_velocity, remove_outliers, smooth_positions

# None | "ruptures" | "peaks" | "both"
CONCENTRIC_METHOD = "both"


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
# Panel helpers
# ---------------------------------------------------------
def _draw_segment_overlay(ax, segments):
    """Draw ruptures segment debug overlay on an axes."""
    if not segments:
        return
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    for seg in segments:
        is_merged = seg.get("merged", False)
        if seg["selected"]:
            color, alpha = "green", 0.25
        elif is_merged:
            color, alpha = "mediumpurple", 0.12
        elif seg["skipped_short"]:
            color, alpha = "orange", 0.10
        else:
            color, alpha = "steelblue", 0.10
        ax.axvspan(seg["start"], seg["end"], color=color, alpha=alpha,
                   linestyle="--" if is_merged else "-",
                   edgecolor=color if is_merged else None, linewidth=1.2 if is_merged else 0)
        ax.axvline(seg["start"], color=color, linestyle=":", linewidth=0.8)
        prefix = "[M] " if is_merged else ""
        label = f'{prefix}{seg["displacement"]:+.0f}px\n({seg["frames"]}f)'
        mid = (seg["start"] + seg["end"]) / 2
        ax.text(mid, y_lo + y_range * 0.05, label,
                fontsize=6, ha="center", va="bottom", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7))
    ax.axvline(segments[-1]["end"], color="steelblue", linestyle=":", linewidth=0.8)


def _draw_ruptures_panel(ax, raw_pos, smooth_pos, smooth_vel, ruptures_phases, segments):
    ax.plot(raw_pos, label="Raw Position", alpha=0.4)
    ax.plot(smooth_pos, label="Smoothed Position (S-G)", linewidth=2)
    _draw_segment_overlay(ax, segments)

    multi = len(ruptures_phases) > 1
    for i, (start, end) in enumerate(ruptures_phases):
        ax.axvspan(start, end, color="green", alpha=0.15,
                   label="Concentric Phase" if i == 0 else "_nolegend_")
        ax.axvline(start, color="green", linestyle="--", linewidth=1.4)
        ax.axvline(end, color="darkgreen", linestyle="--", linewidth=1.4)
        avg_vel = float(np.mean(smooth_vel[start:end]))
        mid = (start + end) / 2
        y_top = ax.get_ylim()[1]
        if multi:
            ax.text(mid, y_top, f"Rep {i + 1}\n{avg_vel:.1f} px/s",
                    fontsize=8, ha="center", color="darkgreen", va="top")
        else:
            ax.text(start + 1, y_top, "Start", fontsize=8, color="green", va="top")
            ax.text(end + 1, y_top, "Lockout", fontsize=8, color="darkgreen", va="top")
            ax.text(mid, y_top, f"{avg_vel:.1f} px/s",
                    fontsize=8, ha="center", color="darkgreen", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.8))

    n = len(ruptures_phases)
    ax.set_title(f"Ruptures — Detected Concentric Phase{'s' if n != 1 else ''} ({n} rep{'s' if n != 1 else ''})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position")
    ax.legend()


def _draw_peaks_panel(ax, raw_pos, smooth_pos, smooth_vel, peaks_phases):
    ax.plot(raw_pos, label="Raw Position", alpha=0.4)
    ax.plot(smooth_pos, label="Smoothed Position (S-G)", linewidth=2)

    for i, (trough, peak) in enumerate(peaks_phases):
        ax.axvspan(trough, peak, color="green", alpha=0.15)
        mid = (trough + peak) / 2
        avg_vel = float(np.mean(smooth_vel[trough:peak]))
        y_top = ax.get_ylim()[1]
        ax.text(mid, y_top, f"Rep {i + 1}\n{avg_vel:.1f} px/s", fontsize=8, ha="center",
                color="darkgreen", va="top")

    n = len(peaks_phases)
    ax.set_title(f"Peaks — Detected Concentric Phases ({n} rep{'s' if n != 1 else ''})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position")
    ax.legend()


# ---------------------------------------------------------
# Plot raw + smoothed + optional concentric shading
# ---------------------------------------------------------
def plot_analysis(raw_pos, smooth_pos, raw_vel, smooth_vel, out_path,
                  concentric_method=None,
                  ruptures_phases=None, ruptures_segments=None,
                  peaks_phases=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ruptures_phases = ruptures_phases or []
    peaks_phases = peaks_phases or []

    show_ruptures_panel = concentric_method in ("ruptures", "both")
    show_peaks_panel = concentric_method in ("peaks", "both")
    n_panels = 2 + int(show_ruptures_panel) + int(show_peaks_panel)

    figsize_h = {2: 8, 3: 12, 4: 16}[n_panels]
    plt.figure(figsize=(14, figsize_h))

    panel = 1

    # --- Optional ruptures detection panel ---
    if show_ruptures_panel:
        ax = plt.subplot(n_panels, 1, panel)
        _draw_ruptures_panel(ax, raw_pos, smooth_pos, smooth_vel, ruptures_phases, ruptures_segments)
        panel += 1

    # --- Optional peaks detection panel ---
    if show_peaks_panel:
        ax = plt.subplot(n_panels, 1, panel)
        _draw_peaks_panel(ax, raw_pos, smooth_pos, smooth_vel, peaks_phases)
        panel += 1

    # --- Position panel ---
    ax1 = plt.subplot(n_panels, 1, panel)
    ax1.plot(raw_pos, label="Raw Position", alpha=0.4)
    ax1.plot(smooth_pos, label="Smoothed Position (S-G)", linewidth=2)

    if concentric_method == "ruptures" and ruptures_segments:
        _draw_segment_overlay(ax1, ruptures_segments)

    if concentric_method == "ruptures" and ruptures_phases:
        multi = len(ruptures_phases) > 1
        for i, (start, end) in enumerate(ruptures_phases):
            ax1.axvspan(start, end, color="green", alpha=0.15,
                        label="Concentric Phase" if i == 0 else "_nolegend_")
            ax1.axvline(start, color="green", linestyle="--", linewidth=1.4)
            ax1.axvline(end, color="darkgreen", linestyle="--", linewidth=1.4)
            y_top = ax1.get_ylim()[1]
            if multi:
                ax1.text((start + end) / 2, y_top, f"Rep {i + 1}",
                         fontsize=8, ha="center", color="darkgreen", va="top")
            else:
                ax1.annotate("Start", xy=(start, y_top), xytext=(start + 1, y_top),
                             fontsize=8, color="green", va="top")
                ax1.annotate("Lockout", xy=(end, y_top), xytext=(end + 1, y_top),
                             fontsize=8, color="darkgreen", va="top")

    if concentric_method == "peaks":
        for i, (trough, peak) in enumerate(peaks_phases):
            ax1.axvspan(trough, peak, color="green", alpha=0.15,
                        label="Concentric Phase" if i == 0 else "_nolegend_")

    ax1.set_title("Barbell Vertical Position (px, upward = positive)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Position")
    ax1.legend()
    panel += 1

    # --- Velocity panel ---
    ax2 = plt.subplot(n_panels, 1, panel)
    ax2.plot(raw_vel, label="Raw Velocity", alpha=0.4)
    ax2.plot(smooth_vel, label="Smoothed Velocity (S-G)", linewidth=2)

    if concentric_method == "ruptures" and ruptures_phases:
        multi = len(ruptures_phases) > 1
        for i, (start, end) in enumerate(ruptures_phases):
            ax2.axvspan(start, end, color="green", alpha=0.15,
                        label="Concentric Phase" if i == 0 else "_nolegend_")
            ax2.axvline(start, color="green", linestyle="--", linewidth=1.2)
            ax2.axvline(end, color="darkgreen", linestyle="--", linewidth=1.2)
            avg_vel = float(np.mean(smooth_vel[start:end]))
            label = f"R{i + 1}: {avg_vel:.1f} px/s" if multi else f"Avg concentric velocity: {avg_vel:.1f} px/s"
            ax2.text(start, ax2.get_ylim()[1], label,
                     fontsize=7 if multi else 9, color="darkgreen", va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.85))

    if concentric_method == "peaks":
        for i, (trough, peak) in enumerate(peaks_phases):
            ax2.axvspan(trough, peak, color="green", alpha=0.15,
                        label="Concentric Phase" if i == 0 else "_nolegend_")
            avg_vel = float(np.mean(smooth_vel[trough:peak]))
            ax2.text(trough, ax2.get_ylim()[1],
                     f"R{i + 1}: {avg_vel:.1f}",
                     fontsize=7, color="darkgreen", va="top", ha="left")

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

    meta_path = os.path.join(os.path.dirname(TRACK_PATH), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        FPS = meta["fps"]
        print(f"[INFO] Using FPS from meta.json: {FPS:.3f}")
    else:
        FPS = args.fps
        print(f"[INFO] meta.json not found — using --fps default: {FPS}")

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
    ruptures_phases, ruptures_segments = [], None
    peaks_phases = []

    if CONCENTRIC_METHOD in ("ruptures", "both"):
        pen = 20
        ruptures_phases, ruptures_segments = detect_concentric(
            smooth_positions_arr, method="ruptures", pen=pen, return_segments=True
        )
        print(f"[DEBUG] {len(ruptures_segments)} segments detected (pen={pen}):")
        for i, s in enumerate(ruptures_segments):
            if s["selected"]:
                tag = " <-- SELECTED"
            elif s["skipped_short"]:
                tag = " (short)"
            elif s.get("no_eccentric_following"):
                tag = " (no eccentric follows)"
            elif s.get("merged"):
                tag = " (merged)"
            else:
                tag = ""
            print(f"  [{i}] frames {s['start']:>4}-{s['end']:>4}  "
                  f"len={s['frames']:>4}  disp={s['displacement']:>+8.1f}px{tag}")
        for i, (start, end) in enumerate(ruptures_phases):
            avg_vel = float(np.mean(smooth_velocity[start:end]))
            rep_label = f"Rep {i + 1} " if len(ruptures_phases) > 1 else ""
            print(f"[INFO] Ruptures {rep_label}avg concentric velocity: {avg_vel:.1f} px/s (frames {start}-{end})")

    if CONCENTRIC_METHOD in ("peaks", "both"):
        print("[INFO] Peaks detection:")
        peaks_phases = detect_concentric(smooth_positions_arr, method="peaks", fps=FPS)

    # 6. Plot
    plot_analysis(clean_positions, smooth_positions_arr,
                  raw_velocity, smooth_velocity,
                  OUTPUT_PATH,
                  concentric_method=CONCENTRIC_METHOD,
                  ruptures_phases=ruptures_phases,
                  ruptures_segments=ruptures_segments,
                  peaks_phases=peaks_phases)
