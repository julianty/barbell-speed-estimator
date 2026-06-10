import numpy as np
import ruptures as rpt
from scipy.signal import find_peaks


def _interpolate(smoothed_positions):
    arr = np.array([x if x is not None else np.nan for x in smoothed_positions], dtype=float)
    if np.isnan(arr).all():
        return arr
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    return arr


def detect_concentric_ruptures(smoothed_positions, pen: float = 3.0, min_frames: int = 20,
                                smooth_velocity: np.ndarray = None,
                                merge_min_vel: float = 50.0,
                                extension_lookahead: int = 30,
                                min_rep_ratio: float = 0.60,
                                return_segments: bool = False):
    """
    Detect concentric phases using ruptures PELT changepoint detection on the velocity signal.

    Pipeline:
      1. PELT on velocity signal → segment boundaries
      2. Per-segment stats (displacement, length, short flag)
      3. Merge adjacent qualifying segments into synthetic candidates
      4. Collect candidate pool (positive displacement, not too short)
      5. Estimate expected rep displacement from movement range (peak/trough analysis)
      6. Select candidates covering >= min_rep_ratio of expected displacement;
         deduplicate overlapping candidates (prefer merged, then higher displacement)
      7. Extend each selected phase boundary outward while velocity conditions hold

    min_rep_ratio: fraction of the estimated full-rep displacement a candidate must cover
        to be selected. Set to 0.0 to disable (select all positive-displacement candidates).

    Returns:
        list[tuple[int,int]]: [(start_frame, end_frame)] or [] on failure.
        If return_segments=True, returns (phases, segments) where segments is a list of
        dicts: {start, end, displacement, frames, selected, skipped_short}.
    """
    arr = _interpolate(smoothed_positions)

    if np.isnan(arr).all():
        return ([], []) if return_segments else []

    # Stage 1: run PELT on the velocity signal (first diff of position)
    vel = np.diff(arr)
    algo = rpt.Pelt(model="l2").fit(vel)
    vel_breakpoints = algo.predict(pen=pen)
    boundaries = [0] + vel_breakpoints

    # Stage 2: build per-segment stats (displacement, length, short flag)
    segments = []
    for i in range(len(boundaries) - 1):
        v_start = boundaries[i]
        v_end = boundaries[i + 1]
        seg_start = v_start
        seg_end = v_end
        n_frames = seg_end - seg_start + 1
        displacement = float(arr[seg_end] - arr[seg_start])
        too_short = n_frames < min_frames

        seg = dict(start=seg_start, end=seg_end, frames=n_frames,
                   displacement=displacement, skipped_short=too_short, selected=False)
        segments.append(seg)

    # Stage 3: merge adjacent qualifying segments into additional candidates.
    # A run of 2+ consecutive non-short segments may together form a valid concentric
    # phase even if each individual segment was below min_frames on its own.
    def _qualifies(seg):
        if seg["skipped_short"]:
            return False
        s, e = seg["start"], seg["end"]
        if smooth_velocity is not None:
            vel_slice = smooth_velocity[s + 1 : e + 1]
            avg_vel = float(np.mean(vel_slice)) if vel_slice else 0.0
        else:
            avg_vel = seg["displacement"] / seg["frames"] if seg["frames"] > 0 else 0.0
        return seg["displacement"] > 0 and avg_vel >= merge_min_vel

    n_original = len(segments)
    i = 0
    while i < n_original:
        if _qualifies(segments[i]):
            run = [segments[i]]
            j = i + 1
            while j < n_original and _qualifies(segments[j]):
                run.append(segments[j])
                j += 1
            if len(run) > 1:
                m_start = run[0]["start"]
                m_end = run[-1]["end"]
                m_frames = m_end - m_start + 1
                m_disp = float(arr[m_end] - arr[m_start])
                segments.append(dict(
                    start=m_start, end=m_end, frames=m_frames,
                    displacement=m_disp,
                    skipped_short=(m_frames < min_frames),
                    selected=False,
                    merged=True,
                ))
            i = j
        else:
            i += 1

    # Stage 4: collect all positive-displacement non-short segments as candidates
    candidates = [s for s in segments if not s["skipped_short"] and s["displacement"] > 0]

    # Stage 5: estimate expected rep displacement from movement range (peak/trough analysis)
    def _estimate_rep_displacement(a):
        distance = max(1, len(a) // 20)
        prominence = (np.nanmax(a) - np.nanmin(a)) * 0.15
        peaks, _ = find_peaks(a, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(-a, distance=distance, prominence=prominence)
        displacements = []
        peak_cursor = 0
        for trough in troughs:
            while peak_cursor < len(peaks) and peaks[peak_cursor] <= trough:
                peak_cursor += 1
            if peak_cursor < len(peaks):
                displacements.append(float(a[peaks[peak_cursor]] - a[trough]))
        return float(np.median(displacements)) if displacements else float(np.nanmax(a) - np.nanmin(a))

    expected_disp = _estimate_rep_displacement(arr)
    qualified = [c for c in candidates if c["displacement"] >= min_rep_ratio * expected_disp]

    # Stage 6: deduplicate — prefer merged candidates, then higher displacement; greedy non-overlapping
    qualified_sorted = sorted(qualified,
                              key=lambda s: (s.get("merged", False), s["displacement"]),
                              reverse=True)
    all_selected = []
    for cand in qualified_sorted:
        if not any(cand["start"] < sel["end"] and sel["start"] < cand["end"]
                   for sel in all_selected):
            all_selected.append(cand)
    all_selected.sort(key=lambda s: s["start"])

    if not all_selected and candidates:
        all_selected = [max(candidates, key=lambda s: s["displacement"])]

    for seg in segments:
        seg["selected"] = seg in all_selected
    phases = [(s["start"], s["end"]) for s in all_selected]

    # Stage 7: extend each selected phase boundary outward while velocity conditions hold
    def _vel_at(frame):
        if smooth_velocity is not None:
            return float(smooth_velocity[frame])
        return float(arr[frame] - arr[frame - 1]) if frame > 0 else 0.0

    def _extend(start, end):
        count = 0
        while end + 1 < len(arr) and count < extension_lookahead:
            if _vel_at(end) > 0 and arr[end + 1] > arr[end]:
                end += 1
                count += 1
            else:
                break
        count = 0
        while start - 1 >= 0 and count < extension_lookahead:
            if _vel_at(start - 1) > 0:
                start -= 1
                count += 1
            else:
                break
        return max(0, start), min(len(arr) - 1, end)

    phases = [_extend(s, e) for s, e in phases]

    return (phases, segments) if return_segments else phases


def detect_concentric_peaks(smoothed_positions, min_frames: int = 20, fps: float = None):
    """
    Detect concentric phases (one per rep) using scipy peak detection.

    Pairs each trough with the first peak that follows it (merge pairing), so the
    result is correct regardless of where in the rep cycle the signal starts.
    Prints per-rep velocity stats to stdout in px/s when fps is provided, px/frame
    otherwise.

    Returns:
        list[tuple[int,int]]: [(trough_frame, peak_frame), ...] one per rep.
    """
    arr = _interpolate(smoothed_positions)

    if np.isnan(arr).all():
        return []

    distance = max(1, len(arr) // 20)
    prominence = (np.nanmax(arr) - np.nanmin(arr)) * 0.3

    peaks_idx, _ = find_peaks(arr, distance=distance, prominence=prominence)
    troughs_idx, _ = find_peaks(-arr, distance=distance, prominence=prominence)

    vel = np.diff(arr)
    vel_scale = fps if fps is not None else 1.0
    vel_unit = "px/s" if fps is not None else "px/frame"

    phases = []
    peak_cursor = 0
    for trough in troughs_idx:
        while peak_cursor < len(peaks_idx) and peaks_idx[peak_cursor] <= trough:
            peak_cursor += 1
        if peak_cursor >= len(peaks_idx):
            break
        peak = peaks_idx[peak_cursor]
        peak_cursor += 1
        if peak - trough < min_frames:
            continue
        sl = vel[trough:peak]
        avg_vel = float(np.mean(sl)) * vel_scale
        max_vel = float(np.max(sl)) * vel_scale
        print(f"Rep {len(phases) + 1}: avg={avg_vel:.2f} {vel_unit}, max={max_vel:.2f} {vel_unit}")
        phases.append((trough, peak))

    return phases


def detect_concentric(smooth_positions, method: str = "peaks", **kwargs):
    """Dispatcher — method is 'ruptures' or 'peaks'."""
    if method == "ruptures":
        return detect_concentric_ruptures(smooth_positions, **kwargs)
    elif method == "peaks":
        return detect_concentric_peaks(smooth_positions, **kwargs)
    else:
        raise ValueError(f"Unknown concentric detection method: {method!r}")
