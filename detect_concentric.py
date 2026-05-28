# detect_concentric.py

import numpy as np

def detect_concentric(smoothed_positions, baseline_frames=100, top_threshold=0.10):
    """
    Detect concentric phase using:
    - Bottom: average of first N frames (bar at rest)
    - Top: first frame within 10% of global maximum
    """

    arr = np.array(smoothed_positions)

    # --- 1. Estimate bottom from baseline frames ---
    N = min(baseline_frames, len(arr))
    bottom_value = np.mean(arr[:N])

    # Find index closest to this baseline value
    bottom_idx = int(np.argmin(np.abs(arr - bottom_value)))

    # --- 2. Estimate top from near-global maximum ---
    global_max = np.max(arr)

    # Threshold: within 10% of global max
    threshold = global_max - (global_max * top_threshold)

    # Find first index where arr >= threshold
    candidates = np.where(arr >= threshold)[0]

    if len(candidates) == 0:
        return None, None

    top_idx = int(candidates[0])

    # --- 3. Sanity checks ---
    if top_idx <= bottom_idx:
        return None, None

    return bottom_idx, top_idx
