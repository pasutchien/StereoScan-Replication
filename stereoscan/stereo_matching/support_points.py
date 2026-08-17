"""Sparse support points, used to set an adaptive global disparity range
(Sec. III-C, ELAS-inspired - see stereoscan/stereo_matching/__init__.py and
the project's design notes for which parts of real ELAS this simplifies).

Evaluated on a sparse regular grid with a full-range search per candidate
(affordable, since it's only a sparse grid - a Python loop over candidates,
same style as feature_matching/matching.py's per-query loop), with a
left-right consistency check to keep only reliable points.
"""

import numpy as np

from stereoscan.feature_matching.filters import sobel_x, sobel_y


def _extract_window(response: np.ndarray, u: int, v: int, radius: int) -> np.ndarray:
    return response[v - radius : v + radius + 1, u - radius : u + radius + 1]


def _window_sad(w1: np.ndarray, w2: np.ndarray) -> float:
    return float(np.abs(w1.astype(np.int32) - w2.astype(np.int32)).sum())


def _best_disparity(query_gx, query_gy, candidate_gx, candidate_gy, qu: int, v: int, radius: int, max_d: int, shift_sign: int):
    """argmin SAD over d in [0, max_d] where the candidate window is centered
    at qu + shift_sign*d. shift_sign=-1: left-query/right-candidate (KITTI
    convention, right-image match is to the left). shift_sign=+1: the
    reverse, right-query/left-candidate, for the consistency check.
    """
    if max_d < 0:
        return None, None
    query_wx = _extract_window(query_gx, qu, v, radius)
    query_wy = _extract_window(query_gy, qu, v, radius)

    best_d, best_cost = None, None
    for d in range(max_d + 1):
        cu = qu + shift_sign * d
        cost = _window_sad(query_wx, _extract_window(candidate_gx, cu, v, radius)) + _window_sad(
            query_wy, _extract_window(candidate_gy, cu, v, radius)
        )
        if best_cost is None or cost < best_cost:
            best_d, best_cost = d, cost
    return best_d, best_cost


def detect_support_points(
    left_image: np.ndarray,
    right_image: np.ndarray,
    grid_step: int = 5,
    max_disparity: int = 128,
    window_radius: int = 2,
    consistency_tolerance: int = 1,
):
    """(points, disparities): support points on a sparse grid that survive a
    left-right consistency check. `points` is (N, 2) int32 (u, v); `disparities`
    is (N,) float64.
    """
    left_gx, left_gy = sobel_x(left_image), sobel_y(left_image)
    right_gx, right_gy = sobel_x(right_image), sobel_y(right_image)

    h, w = left_image.shape
    r = window_radius

    points = []
    disparities = []

    for v in range(r, h - r, grid_step):
        for u in range(r, w - r, grid_step):
            max_d = min(max_disparity, u - r)
            d_left, _ = _best_disparity(left_gx, left_gy, right_gx, right_gy, u, v, r, max_d, shift_sign=-1)
            if d_left is None:
                continue

            ru = u - d_left
            max_d2 = min(max_disparity, (w - r - 1) - ru)
            d_right, _ = _best_disparity(right_gx, right_gy, left_gx, left_gy, ru, v, r, max_d2, shift_sign=+1)
            if d_right is None or abs(d_right - d_left) > consistency_tolerance:
                continue

            points.append((u, v))
            disparities.append(d_left)

    return np.array(points, dtype=np.int32).reshape(-1, 2), np.array(disparities, dtype=np.float64)
