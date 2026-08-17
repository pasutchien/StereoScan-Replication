"""Sporadic outlier rejection via Delaunay neighbor support (Sec. III-A).

Sporadic outliers are removed by establishing neighborhood relations as
edges of a 2d Delaunay triangulation on the feature locations in the
current left image. A match is retained only if it is supported by at
least `min_support` neighboring matches, where a neighbor supports a match
if their disparity and flow differences fall within tau_disp / tau_flow
(both default to 5px, per the paper's Sec. IV parameterization).
"""

from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay


def reject_sporadic_outliers(
    curr_left_points: np.ndarray,
    prev_left_points: np.ndarray,
    curr_right_points: np.ndarray,
    tau_disp: float = 5.0,
    tau_flow: float = 5.0,
    min_support: int = 2,
) -> np.ndarray:
    """Return a boolean inlier mask over row-aligned matches.

    `curr_left_points`, `prev_left_points`, `curr_right_points` are (K, 2)
    (x, y) arrays for the same K accepted circular matches, e.g.
    curr_left_points = cl_pts[matches[:, 0]], etc.
    """
    n = len(curr_left_points)
    if n < 3:
        # Not enough points to triangulate, so nothing can be supported.
        return np.zeros(n, dtype=bool)

    disparity = curr_left_points[:, 0].astype(np.float64) - curr_right_points[:, 0].astype(np.float64)
    flow = (curr_left_points - prev_left_points).astype(np.float64)

    triangulation = Delaunay(curr_left_points.astype(np.float64))
    edges = {
        (min(a, b), max(a, b))
        for simplex in triangulation.simplices
        for a, b in combinations(simplex, 2)
    }

    support_count = np.zeros(n, dtype=np.int32)
    for i, j in edges:
        disp_ok = abs(disparity[i] - disparity[j]) <= tau_disp
        flow_ok = np.linalg.norm(flow[i] - flow[j]) <= tau_flow
        if disp_ok and flow_ok:
            support_count[i] += 1
            support_count[j] += 1

    return support_count >= min_support
