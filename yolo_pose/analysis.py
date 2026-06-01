# analysis.py — legacy YOLO-pose pipeline analysis

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt

from detect_concentric import detect_concentric
from utils import compute_velocity, remove_outliers, smooth_positions

ENABLE_CONCENTRIC = False


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
def plot_analysis(raw_pos, smooth_pos, raw_vel, smooth_vel, concentric, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if ENABLE_CONCENTRIC:
        start, end = concentric
    else:
        start, end = None, None

    plt.figure(figsize=(14, 8))

    # --- Position ---
    plt.subplot(2, 1, 1)
    plt.plot(raw_pos, label="Raw Position", alpha=0.4)
    plt.plot(smooth_pos, label="Smoothed Position (S-G)", linewidth=2)

    if start is not None:
        plt.axvspan(start, end, color="green", alpha=0.2, label="Concentric Phase")

    plt.title("Barbell Vertical Position (px, upward = positive)")
    plt.xlabel("Frame")
    plt.ylabel("Position")
    plt.legend()

    # --- Velocity ---
    plt.subplot(2, 1, 2)
    plt.plot(raw_vel, label="Raw Velocity", alpha=0.4)
    plt.plot(smooth_vel, label="Smoothed Velocity (S-G)", linewidth=2)

    if start is not None:
        plt.axvspan(start, end, color="green", alpha=0.2, label="Concentric Phase")

    plt.title("Barbell Velocity (px/s)")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved analysis plot to {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    TRACK_PATH = "pose_results/barbell_track.json"
    OUTPUT_PATH = "pose_results/analysis/bar_speed.png"
    FPS = 30

    track = load_track(TRACK_PATH)

    # 1. Raw positions
    raw_positions = extract_positions(track)

    # 2. Remove outliers BEFORE smoothing
    clean_positions = remove_outliers(raw_positions, window=3, threshold=50)

    # 3. Smooth the cleaned signal
    smooth_positions_arr = smooth_positions(clean_positions, window=9, poly=3)

    # 4. Compute velocities
    raw_velocity = compute_velocity(clean_positions, FPS)
    smooth_velocity = compute_velocity(smooth_positions_arr, FPS)

    # 5. Optional concentric detection
    if ENABLE_CONCENTRIC:
        concentric_phase = detect_concentric(smooth_positions_arr)
    else:
        concentric_phase = (None, None)

    # 6. Plot
    plot_analysis(clean_positions, smooth_positions_arr,
                  raw_velocity, smooth_velocity,
                  concentric_phase, OUTPUT_PATH)
