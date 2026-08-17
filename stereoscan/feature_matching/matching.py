"""Circular feature matching (Geiger et al. 2011, Sec. III-A).

Two constraint types, one per leg direction:
  - temporal legs (current-left <-> previous-left, previous-right <->
    current-right): the camera moves only a little between consecutive
    frames, so a genuine match stays within an MxM pixel window.
  - stereo legs (previous-left <-> previous-right, current-right <->
    current-left): rectified stereo images share epipolar geometry, so a
    genuine match lies on (almost) the same image row.

A circle match is accepted only if walking all four legs lands back on the
exact feature that started the circle.
"""

import numpy as np


def _best_match(query_desc, candidate_desc, candidate_indices):
    """SAD-argmin of `query_desc` against `candidate_desc[candidate_indices]`."""
    subset = candidate_desc[candidate_indices].astype(np.int16)
    sad = np.abs(subset - query_desc.astype(np.int16)).sum(axis=1)
    best = np.argmin(sad)
    return candidate_indices[best], int(sad[best])


def match_within_window(query_points, query_desc, candidate_points, candidate_desc, window_radius):
    """Best-SAD match for each query among candidates within an MxM box.

    A candidate qualifies if |dx| <= window_radius and |dy| <= window_radius
    relative to the query point. Returns (match_idx, match_sad), both length
    len(query_points); unmatched queries get -1 in both arrays.
    """
    n = len(query_points)
    match_idx = np.full(n, -1, dtype=np.int64)
    match_sad = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        qx, qy = query_points[i]
        in_window = (np.abs(candidate_points[:, 0] - qx) <= window_radius) & (
            np.abs(candidate_points[:, 1] - qy) <= window_radius
        )
        candidate_indices = np.nonzero(in_window)[0]
        if len(candidate_indices) == 0:
            continue
        match_idx[i], match_sad[i] = _best_match(query_desc[i], candidate_desc, candidate_indices)
    return match_idx, match_sad


def match_within_bounds(query_points, query_desc, candidate_points, candidate_desc, x_min, x_max, y_min, y_max):
    """Best-SAD match for each query within a per-query axis-aligned box.

    Like match_within_window, but the box need not be square or centered on
    the query: x_min/x_max/y_min/y_max are absolute candidate-coordinate
    bounds, each either a scalar (same box for every query) or an array of
    length len(query_points) (a different box per query - e.g. a bucketed
    search window narrowed by observed per-bin displacement).
    """
    n = len(query_points)
    x_min = np.broadcast_to(x_min, (n,))
    x_max = np.broadcast_to(x_max, (n,))
    y_min = np.broadcast_to(y_min, (n,))
    y_max = np.broadcast_to(y_max, (n,))

    match_idx = np.full(n, -1, dtype=np.int64)
    match_sad = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        in_box = (
            (candidate_points[:, 0] >= x_min[i])
            & (candidate_points[:, 0] <= x_max[i])
            & (candidate_points[:, 1] >= y_min[i])
            & (candidate_points[:, 1] <= y_max[i])
        )
        candidate_indices = np.nonzero(in_box)[0]
        if len(candidate_indices) == 0:
            continue
        match_idx[i], match_sad[i] = _best_match(query_desc[i], candidate_desc, candidate_indices)
    return match_idx, match_sad


def match_along_epipolar(query_points, query_desc, candidate_points, candidate_desc, y_tolerance=1):
    """Best-SAD match for each query among candidates on the same row (+/- y_tolerance).

    x position is unconstrained here (disparity is not bounded); returns
    (match_idx, match_sad) in the same format as match_within_window.
    """
    n = len(query_points)
    match_idx = np.full(n, -1, dtype=np.int64)
    match_sad = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        qy = query_points[i, 1]
        on_epipolar_line = np.abs(candidate_points[:, 1] - qy) <= y_tolerance
        candidate_indices = np.nonzero(on_epipolar_line)[0]
        if len(candidate_indices) == 0:
            continue
        match_idx[i], match_sad[i] = _best_match(query_desc[i], candidate_desc, candidate_indices)
    return match_idx, match_sad


def circular_match(
    curr_left_points,
    curr_left_desc,
    prev_left_points,
    prev_left_desc,
    prev_right_points,
    prev_right_desc,
    curr_right_points,
    curr_right_desc,
    window_radius,
    epipolar_tolerance=1,
):
    """Circle-match current-left features through previous-left, previous-right,
    current-right and back to current-left.

    Returns an (K, 4) int64 array of accepted matches, one row per accepted
    circle: [curr_left_idx, prev_left_idx, prev_right_idx, curr_right_idx].
    """
    n = len(curr_left_points)

    to_prev_left, _ = match_within_window(
        curr_left_points, curr_left_desc, prev_left_points, prev_left_desc, window_radius
    )

    to_prev_right = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_prev_left >= 0)[0]
    if len(step):
        m, _ = match_along_epipolar(
            prev_left_points[to_prev_left[step]],
            prev_left_desc[to_prev_left[step]],
            prev_right_points,
            prev_right_desc,
            epipolar_tolerance,
        )
        to_prev_right[step] = m

    to_curr_right = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_prev_right >= 0)[0]
    if len(step):
        m, _ = match_within_window(
            prev_right_points[to_prev_right[step]],
            prev_right_desc[to_prev_right[step]],
            curr_right_points,
            curr_right_desc,
            window_radius,
        )
        to_curr_right[step] = m

    back_to_curr_left = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_curr_right >= 0)[0]
    if len(step):
        m, _ = match_along_epipolar(
            curr_right_points[to_curr_right[step]],
            curr_right_desc[to_curr_right[step]],
            curr_left_points,
            curr_left_desc,
            epipolar_tolerance,
        )
        back_to_curr_left[step] = m

    accepted = back_to_curr_left == np.arange(n)
    idx = np.nonzero(accepted)[0]
    return np.column_stack([idx, to_prev_left[idx], to_prev_right[idx], to_curr_right[idx]])
