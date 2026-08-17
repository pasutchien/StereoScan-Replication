"""Fully vectorized dense disparity search over a given (adaptive, not
per-pixel) disparity range (Sec. III-C, ELAS-inspired - see
stereoscan/stereo_matching/__init__.py and the project's design notes for
which parts of real ELAS this simplifies).
"""

import numpy as np
from scipy.ndimage import uniform_filter

from stereoscan.feature_matching.filters import sobel_x, sobel_y


def _cost_volume(query_gx, query_gy, candidate_gx, candidate_gy, disparities: np.ndarray, window_radius: int, direction: int) -> np.ndarray:
    """(D, H, W) SAD cost volume. direction=+1: candidate_u = query_u - d
    (left-query/right-candidate); direction=-1: candidate_u = query_u + d
    (right-query/left-candidate, for the consistency check).
    """
    h, w = query_gx.shape
    query_gx = query_gx.astype(np.float32)
    query_gy = query_gy.astype(np.float32)
    volume = np.full((len(disparities), h, w), np.inf, dtype=np.float32)
    size = 2 * window_radius + 1

    for i, d in enumerate(disparities):
        d = int(d)
        shifted_gx = np.roll(candidate_gx, direction * d, axis=1).astype(np.float32)
        shifted_gy = np.roll(candidate_gy, direction * d, axis=1).astype(np.float32)
        diff = np.abs(query_gx - shifted_gx) + np.abs(query_gy - shifted_gy)
        cost = uniform_filter(diff, size=size, mode="nearest")
        if d > 0:
            if direction > 0:
                cost[:, :d] = np.inf  # candidate wrapped around from the right edge
            else:
                cost[:, -d:] = np.inf  # candidate wrapped around from the left edge
        volume[i] = cost

    return volume


def _best_from_volume(volume: np.ndarray, disparities: np.ndarray):
    idx = np.argmin(volume, axis=0)  # (H, W)
    best_cost = np.take_along_axis(volume, idx[None, :, :], axis=0)[0]
    disparity = disparities[idx].astype(np.float64)
    return disparity, idx, best_cost


def _subpixel_offset(volume: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Parabolic-fit sub-pixel offset around each pixel's best integer disparity
    ("sub-pixel refinement via parabolic fitting", as the paper suggests for
    the sparse case too)."""
    D = volume.shape[0]
    idx_lo = np.clip(idx - 1, 0, D - 1)
    idx_hi = np.clip(idx + 1, 0, D - 1)
    c0 = np.take_along_axis(volume, idx_lo[None], axis=0)[0]
    c1 = np.take_along_axis(volume, idx[None], axis=0)[0]
    c2 = np.take_along_axis(volume, idx_hi[None], axis=0)[0]

    denom = c0 - 2 * c1 + c2
    offset = np.zeros_like(c1)
    valid = (denom != 0) & (idx > 0) & (idx < D - 1) & np.isfinite(c0) & np.isfinite(c2)
    offset[valid] = 0.5 * (c0[valid] - c2[valid]) / denom[valid]
    return np.clip(offset, -1.0, 1.0)


def _left_right_consistency(left_disp: np.ndarray, right_disp: np.ndarray, tolerance: float) -> np.ndarray:
    h, w = left_disp.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h)[:, None], (1, w))
    ru = np.clip(np.round(u - left_disp).astype(np.int32), 0, w - 1)

    matched_right_disp = right_disp[v, ru]
    consistent = np.abs(left_disp - matched_right_disp) <= tolerance

    result = left_disp.copy()
    result[~consistent] = np.nan
    return result


def compute_disparity_map(
    left_image: np.ndarray,
    right_image: np.ndarray,
    min_disparity: int,
    max_disparity: int,
    window_radius: int = 2,
    consistency_tolerance: float = 1.0,
) -> np.ndarray:
    """Dense (H, W) disparity map (float64, NaN where left-right inconsistent),
    searching only `[min_disparity, max_disparity]` everywhere (not narrowed
    per-pixel - see module docstring)."""
    left_gx, left_gy = sobel_x(left_image), sobel_y(left_image)
    right_gx, right_gy = sobel_x(right_image), sobel_y(right_image)

    disparities = np.arange(min_disparity, max_disparity + 1)

    left_volume = _cost_volume(left_gx, left_gy, right_gx, right_gy, disparities, window_radius, direction=+1)
    left_disp, left_idx, _ = _best_from_volume(left_volume, disparities)
    left_disp = left_disp + _subpixel_offset(left_volume, left_idx)

    right_volume = _cost_volume(right_gx, right_gy, left_gx, left_gy, disparities, window_radius, direction=-1)
    right_disp, right_idx, _ = _best_from_volume(right_volume, disparities)
    right_disp = right_disp + _subpixel_offset(right_volume, right_idx)

    return _left_right_consistency(left_disp, right_disp, consistency_tolerance)
