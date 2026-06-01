# utils.py — shared utilities

import os
import ffmpeg
import numpy as np
from scipy.signal import savgol_filter


def extract_frames(video_path: str, frames_dir: str, fps: float | None = None) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, "frame_%06d.jpg").replace("\\", "/")

    stream = ffmpeg.input(video_path)
    if fps:
        stream = stream.output(output_pattern, qscale=2, r=fps)
    else:
        stream = stream.output(output_pattern, qscale=2)

    stream.run(quiet=True)

    print(f"[INFO] Frames extracted to: {frames_dir}")


def frames_to_video(frames_dir: str, output_video_path: str, fps: float = 30) -> None:
    input_pattern = os.path.join(frames_dir, "frame_%06d.jpg").replace("\\", "/")

    (
        ffmpeg
        .input(input_pattern, framerate=fps)
        .output(output_video_path, vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"[INFO] Reconstructed video saved to: {output_video_path}")


def remove_outliers(positions, window=5, threshold=50):
    arr = np.array(positions, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window)
        end = min(len(arr), i + window + 1)
        local = arr[start:end]
        local = local[~np.isnan(local)]
        if len(local) < 3:
            continue
        median = np.median(local)
        if abs(arr[i] - median) > threshold:
            arr[i] = median
    return arr.tolist()


def smooth_positions(positions, window=9, poly=3):
    arr = np.array([p if p is not None else np.nan for p in positions])
    mask = np.isnan(arr)
    if np.any(mask):
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return savgol_filter(arr, window_length=window, polyorder=poly).tolist()


def compute_velocity(positions, fps):
    dt = 1.0 / fps
    velocities = [0]
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            velocities.append(0)
        else:
            velocities.append((positions[i] - positions[i - 1]) / dt)
    return velocities
