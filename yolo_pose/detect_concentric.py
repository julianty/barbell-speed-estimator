import numpy as np
import ruptures as rpt


def detect_concentric(smoothed_positions: list, pen: float = 3.0, min_frames: int = 20,
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
        (start_frame, end_frame) inclusive, or (None, None) on failure.
        If return_segments=True, returns (start, end, segments) where segments is a list of
        dicts: {start, end, displacement, frames, selected, skipped_short, no_eccentric_following}.
    """
    arr = np.array([x if x is not None else np.nan for x in smoothed_positions], dtype=float)

    if np.isnan(arr).all():
        return (None, None, []) if return_segments else (None, None)

    # Linear interpolation over any None / NaN gaps
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])

    # Differentiate first: the concentric phase is a sustained positive-mean region in velocity.
    # L2 detects mean shifts, so running it on the velocity signal finds slope changes in position.
    vel = np.diff(arr)
    algo = rpt.Pelt(model="l2").fit(vel)
    # Breakpoints are end-exclusive indices into vel (length n-1); sentinel is len(vel).
    # Convert to position-frame boundaries: vel[i] = arr[i+1]-arr[i], so a vel segment
    # [v_start, v_end) covers position frames [v_start, v_end] inclusive.
    vel_breakpoints = algo.predict(pen=pen)
    boundaries = [0] + vel_breakpoints  # vel segment i: [boundaries[i], boundaries[i+1])

    best_start, best_end = None, None
    segments = []

    for i in range(len(boundaries) - 1):
        v_start = boundaries[i]
        v_end = boundaries[i + 1]       # exclusive in vel-space
        # Position frames: the segment starts at frame v_start, ends at frame v_end (inclusive)
        seg_start = v_start
        seg_end = v_end                  # inclusive position frame
        n_frames = seg_end - seg_start + 1
        displacement = float(arr[seg_end] - arr[seg_start])
        too_short = n_frames < min_frames

        seg = dict(start=seg_start, end=seg_end, frames=n_frames,
                   displacement=displacement, skipped_short=too_short,
                   no_eccentric_following=False, selected=False)
        segments.append(seg)

    def _has_following_eccentric(candidate_seg):
        return any(
            s["start"] > candidate_seg["end"] and s["displacement"] < -min_eccentric_displacement
            for s in segments
        )

    # Build candidate list: valid (not too short) upward segments
    candidates = [s for s in segments if not s["skipped_short"] and s["displacement"] > 0]

    if require_following_eccentric:
        filtered = [s for s in candidates if _has_following_eccentric(s)]
        if not filtered:
            # Graceful fallback: no eccentric follows any candidate (e.g. video ends at lockout)
            for s in candidates:
                s["no_eccentric_following"] = True
        else:
            candidates = filtered
            for s in segments:
                if not s["skipped_short"] and s["displacement"] > 0 and not _has_following_eccentric(s):
                    s["no_eccentric_following"] = True

    if candidates:
        best = max(candidates, key=lambda s: s["displacement"])
        best_start, best_end = best["start"], best["end"]
        for seg in segments:
            seg["selected"] = (seg["start"] == best_start and seg["end"] == best_end)
    else:
        best_start, best_end = None, None

    result = (best_start, best_end) if best_start is not None else (None, None)
    return (*result, segments) if return_segments else result
