"""Dense stereo matching (Sec. III-C), ELAS-inspired.

A sparse set of robustly-matched support points (support_points.py) sets an
adaptive global disparity search range - the "automatically determines the
required disparity search range" idea from the paper - and a fully
vectorized block-matching search (dense_matching.py) then fills in a dense
disparity map over that range.

This deliberately does NOT build a per-pixel disparity prior from the
support-point triangulation the way real ELAS does (that would narrow the
search window individually per pixel, not just set one global range), and
does not include ELAS's Bayesian MAP blending or gap-filling post-process.
See the project's design notes for the full list of simplifications.
"""

from stereoscan.stereo_matching.dense_matching import compute_disparity_map
from stereoscan.stereo_matching.support_points import detect_support_points


def compute_dense_disparity(
    left_image,
    right_image,
    grid_step: int = 5,
    max_disparity: int = 128,
    range_padding: int = 5,
    window_radius: int = 2,
    consistency_tolerance: float = 1.0,
):
    """Support points -> adaptive [min, max] disparity range -> dense map.

    Returns (disparity_map, support_points, support_disparities).
    """
    points, disparities = detect_support_points(
        left_image,
        right_image,
        grid_step=grid_step,
        max_disparity=max_disparity,
        window_radius=window_radius,
        consistency_tolerance=consistency_tolerance,
    )
    if len(disparities) == 0:
        raise RuntimeError("no support points found - cannot determine an adaptive disparity range")

    min_disp = max(0, int(disparities.min() - range_padding))
    max_disp = min(max_disparity, int(disparities.max() + range_padding))

    disparity_map = compute_disparity_map(
        left_image,
        right_image,
        min_disp,
        max_disp,
        window_radius=window_radius,
        consistency_tolerance=consistency_tolerance,
    )
    return disparity_map, points, disparities
