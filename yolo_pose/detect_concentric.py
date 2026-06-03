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
                                require_following_eccentric: bool = True,
                                min_eccentric_displacement: float = 20.0,
                                return_segments: bool = False):
    """
    Detect the concentric phase (greatest net upward displacement) using
    ruptures penalty-based changepoint detection.

    If require_following_eccentric is True (default), candidates must be followed by at least
    one segment with downward displacement >= min_eccentric_displacement px. This prevents
    post-lift stand-up movements from being selected. Falls back to the unfiltered best if
    no candidate passes (e.g. video ends at lockout).

    Returns:
        list[tuple[int,int]]: [(start_frame, end_frame)] or [] on failure.
        If return_segments=True, returns (phases, segments) where segments is a list of
        dicts: {start, end, displacement, frames, selected, skipped_short, no_eccentric_following}.
    """
    arr = _interpolate(smoothed_positions)

    if np.isnan(arr).all():
        return ([], []) if return_segments else []

    vel = np.diff(arr)
    algo = rpt.Pelt(model="l2").fit(vel)
    vel_breakpoints = algo.predict(pen=pen)
    boundaries = [0] + vel_breakpoints

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
                   displacement=displacement, skipped_short=too_short,
                   no_eccentric_following=False, selected=False)
        segments.append(seg)

    # Merge adjacent positive-displacement segments into additional candidates.
    # A run of 2+ consecutive positive segments may together form a valid concentric
    # phase even if each individual segment was too short to qualify on its own.
    i = 0
    while i < len(segments):
        if segments[i]["displacement"] > 0:
            run = [segments[i]]
            j = i + 1
            while j < len(segments) and segments[j]["displacement"] > 0:
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
                    no_eccentric_following=False,
                    selected=False,
                    merged=True,
                ))
            i = j
        else:
            i += 1

    def _has_following_eccentric(candidate_seg):
        return any(
            s["start"] > candidate_seg["end"] and s["displacement"] < -min_eccentric_displacement
            for s in segments
        )

    candidates = [s for s in segments if not s["skipped_short"] and s["displacement"] > 0]

    if require_following_eccentric:
        filtered = [s for s in candidates if _has_following_eccentric(s)]
        if not filtered:
            for s in candidates:
                s["no_eccentric_following"] = True
        else:
            candidates = filtered
            for s in segments:
                if not s["skipped_short"] and s["displacement"] > 0 and not _has_following_eccentric(s):
                    s["no_eccentric_following"] = True

    # Prefer merged candidates (multi-rep): return all of them sorted by start frame.
    # Fall back to single best individual segment if no merged candidates survived filtering.
    merged_candidates = [s for s in candidates if s.get("merged")]

    phases = []
    if merged_candidates:
        for seg in segments:
            seg["selected"] = seg.get("merged", False) and seg in merged_candidates
        phases = [(s["start"], s["end"]) for s in sorted(merged_candidates, key=lambda s: s["start"])]
    elif candidates:
        best = max(candidates, key=lambda s: s["displacement"])
        for seg in segments:
            seg["selected"] = (seg["start"] == best["start"] and seg["end"] == best["end"])
        phases = [(best["start"], best["end"])]

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
