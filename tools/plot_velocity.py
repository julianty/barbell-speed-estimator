"""plot_velocity.py — Visualize output from barbell_velocity.py"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str):
    frames, timestamps, cy, velocity, smoothed = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            timestamps.append(float(row["timestamp_s"]))
            cy.append(float(row["cy_px"]) if row["cy_px"] else None)
            velocity.append(float(row["velocity_ms"]) if row["velocity_ms"] else None)
            smoothed.append(float(row["smoothed_velocity_ms"]) if row["smoothed_velocity_ms"] else None)
    return frames, timestamps, cy, velocity, smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot barbell velocity CSV")
    parser.add_argument("--input", default=None, help="Path to velocities CSV (default: newest sam2_results/*/velocities.csv)")
    parser.add_argument("--output", default=None, help="Save plot to this path (default: velocity_plot.png alongside the CSV)")
    parser.add_argument("--show", action="store_true", help="Display plot interactively after saving")
    args = parser.parse_args()

    if args.input is None:
        candidates = glob.glob(os.path.join("sam2_results", "*", "velocities.csv"))
        candidates += glob.glob(os.path.join("yolo_results", "*", "velocities.csv"))
        if not candidates:
            raise SystemExit(
                "[ERROR] No --input given and no velocities.csv found under "
                "sam2_results/ or yolo_results/."
            )
        args.input = max(candidates, key=os.path.getmtime)
        print(f"[INFO] Using {args.input}")

    if not os.path.exists(args.input):
        raise SystemExit(f"[ERROR] File not found: {args.input}")

    if args.output is None:
        csv_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(csv_dir, "velocity_plot.png")

    frames, timestamps, cy, velocity, smoothed = load_csv(args.input)
    t = np.array(timestamps)

    # Replace None with nan for plotting
    cy_arr = np.array([v if v is not None else np.nan for v in cy])
    vel_arr = np.array([v if v is not None else np.nan for v in velocity])
    smo_arr = np.array([v if v is not None else np.nan for v in smoothed])

    # Flip cy so upward = positive (image y is inverted)
    cy_arr = -cy_arr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(t, cy_arr, linewidth=1.5, color="steelblue")
    ax1.set_ylabel("Vertical position (px, up = +)")
    ax1.set_title("Barbell Position")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, vel_arr, alpha=0.4, linewidth=1, color="gray", label="Raw")
    ax2.plot(t, smo_arr, linewidth=2, color="tomato", label="Smoothed")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Barbell Velocity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(args.output, dpi=150)
    print(f"[INFO] Saved to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
