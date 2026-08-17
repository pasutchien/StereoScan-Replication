"""Search-window narrowing via coarse-pass bucketing (Sec. III-A, final paragraph).

  1. A coarse first pass matches a much sparser candidate set (found with a
     larger NMS neighborhood, e.g. 3x) using the full MxM window - fast,
     since there are far fewer candidates to search.
  2. Those coarse matches' current-left points are binned into a 50x50
     pixel grid; each occupied bin records the observed [min, max] flow
     (dx, dy) among the coarse matches that landed in it.
  3. The full/fine candidate set is then matched with match_within_bounds
     instead of match_within_window: each fine feature looks up its bin's
     observed flow range and searches only that (usually much smaller,
     non-square) box instead of the generic MxM window.

Bins with fewer than `min_samples` coarse matches (including none at all)
fall back to a symmetric window of `fallback_radius` (normally the same
window_radius used for the coarse pass): a min/max range built from only
one or two observations isn't a reliable estimate of the bin's true
displacement range, and using it verbatim (e.g. a single sample gives
min == max, a zero-width window) starves the fine pass of candidates.
"""

import numpy as np

from stereoscan.feature_matching.matching import (
    circular_match,
    match_along_epipolar,
    match_within_bounds,
    match_within_window,
)


def bin_index(points: np.ndarray, bin_size: int) -> np.ndarray:
    """(N, 2) integer (bin_x, bin_y) grid coordinates for each point."""
    return np.floor_divide(points, bin_size).astype(np.int64)


def compute_bin_flow_bounds(curr_left_points: np.ndarray, flow: np.ndarray, bin_size: int = 50, min_samples: int = 1) -> dict:
    """dict[(bin_x, bin_y)] -> (min_dx, max_dx, min_dy, max_dy) observed flow in that bin.

    Bins with fewer than `min_samples` coarse matches are omitted entirely,
    so lookup_search_bounds falls back to the full window for them instead
    of trusting an under-sampled (possibly zero-width) range.
    """
    bins = bin_index(curr_left_points, bin_size)
    stats = {}
    for bx, by in {tuple(b) for b in bins}:
        mask = (bins[:, 0] == bx) & (bins[:, 1] == by)
        if mask.sum() < min_samples:
            continue
        dx, dy = flow[mask, 0], flow[mask, 1]
        stats[(bx, by)] = (dx.min(), dx.max(), dy.min(), dy.max())
    return stats


def lookup_search_bounds(points: np.ndarray, bin_stats: dict, bin_size: int, fallback_radius: float):
    """Per-point absolute (x_min, x_max, y_min, y_max) search box.

    Points whose bin has coarse-pass statistics get that bin's observed
    flow range added to their own position; points in an empty bin fall
    back to a symmetric fallback_radius box.
    """
    n = len(points)
    bins = bin_index(points, bin_size)
    x_min = np.empty(n)
    x_max = np.empty(n)
    y_min = np.empty(n)
    y_max = np.empty(n)
    for i in range(n):
        x, y = points[i]
        stat = bin_stats.get((bins[i, 0], bins[i, 1]))
        if stat is None:
            x_min[i], x_max[i] = x - fallback_radius, x + fallback_radius
            y_min[i], y_max[i] = y - fallback_radius, y + fallback_radius
        else:
            min_dx, max_dx, min_dy, max_dy = stat
            x_min[i], x_max[i] = x + min_dx, x + max_dx
            y_min[i], y_max[i] = y + min_dy, y + max_dy
    return x_min, x_max, y_min, y_max


def circular_match_bucketed(
    coarse_curr_left_points, coarse_curr_left_desc,
    coarse_prev_left_points, coarse_prev_left_desc,
    coarse_prev_right_points, coarse_prev_right_desc,
    coarse_curr_right_points, coarse_curr_right_desc,
    fine_curr_left_points, fine_curr_left_desc,
    fine_prev_left_points, fine_prev_left_desc,
    fine_prev_right_points, fine_prev_right_desc,
    fine_curr_right_points, fine_curr_right_desc,
    window_radius,
    epipolar_tolerance=1,
    bin_size=50,
    min_bin_samples=3,
):
    """Two-pass circular matching: a coarse pass builds per-bin displacement
    bounds, which narrow the search window for the curr-left -> prev-left
    leg of the fine pass. Other legs are unchanged from circular_match.

    Bins with fewer than `min_bin_samples` coarse observations fall back to
    the full window_radius rather than trusting a narrow/degenerate range
    (see compute_bin_flow_bounds).

    Returns (matches, bin_stats): matches has the same (K, 4) format as
    circular_match; bin_stats is the dict from compute_bin_flow_bounds, for
    inspection/visualization.
    """
    coarse_matches = circular_match(
        coarse_curr_left_points, coarse_curr_left_desc,
        coarse_prev_left_points, coarse_prev_left_desc,
        coarse_prev_right_points, coarse_prev_right_desc,
        coarse_curr_right_points, coarse_curr_right_desc,
        window_radius, epipolar_tolerance,
    )

    coarse_cl = coarse_curr_left_points[coarse_matches[:, 0]]
    coarse_pl = coarse_prev_left_points[coarse_matches[:, 1]]
    flow = coarse_cl - coarse_pl
    bin_stats = compute_bin_flow_bounds(coarse_cl, flow, bin_size, min_bin_samples)

    n = len(fine_curr_left_points)
    x_min, x_max, y_min, y_max = lookup_search_bounds(fine_curr_left_points, bin_stats, bin_size, window_radius)

    to_prev_left, _ = match_within_bounds(
        fine_curr_left_points, fine_curr_left_desc, fine_prev_left_points, fine_prev_left_desc,
        x_min, x_max, y_min, y_max,
    )

    to_prev_right = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_prev_left >= 0)[0]
    if len(step):
        m, _ = match_along_epipolar(
            fine_prev_left_points[to_prev_left[step]], fine_prev_left_desc[to_prev_left[step]],
            fine_prev_right_points, fine_prev_right_desc, epipolar_tolerance,
        )
        to_prev_right[step] = m

    to_curr_right = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_prev_right >= 0)[0]
    if len(step):
        m, _ = match_within_window(
            fine_prev_right_points[to_prev_right[step]], fine_prev_right_desc[to_prev_right[step]],
            fine_curr_right_points, fine_curr_right_desc, window_radius,
        )
        to_curr_right[step] = m

    back_to_curr_left = np.full(n, -1, dtype=np.int64)
    step = np.nonzero(to_curr_right >= 0)[0]
    if len(step):
        m, _ = match_along_epipolar(
            fine_curr_right_points[to_curr_right[step]], fine_curr_right_desc[to_curr_right[step]],
            fine_curr_left_points, fine_curr_left_desc, epipolar_tolerance,
        )
        back_to_curr_left[step] = m

    accepted = back_to_curr_left == np.arange(n)
    idx = np.nonzero(accepted)[0]
    matches = np.column_stack([idx, to_prev_left[idx], to_prev_right[idx], to_curr_right[idx]])
    return matches, bin_stats
