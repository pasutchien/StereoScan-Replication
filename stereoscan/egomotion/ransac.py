"""RANSAC-robust egomotion estimation (Sec. III-B).

"To be robust against outliers, we wrap our estimation approach into a
RANSAC scheme, by first estimating (r, t) for 50 times independently using
3 randomly drawn correspondences. All inliers of the winning iteration are
then used for refining the parameters, yielding the final transformation
(r, t)."

Plain Gauss-Newton over all correspondences (gauss_newton.py) isn't robust:
a handful of leftover bad matches can dominate a least-squares fit even
after Delaunay-based outlier rejection, as our own sanity check showed (RMS
dropped 15.9px -> 8.3px, but max residual stayed at 121px). RANSAC fixes
this by fitting on tiny, cheap-to-verify samples and keeping only the trial
whose model most of the data agrees with.
"""

import numpy as np

from stereoscan.egomotion.gauss_newton import estimate_egomotion
from stereoscan.egomotion.reprojection import project


def reprojection_errors(points_3d: np.ndarray, obs_left: np.ndarray, obs_right: np.ndarray, r: np.ndarray, t: np.ndarray, calib) -> np.ndarray:
    """Per-point pixel error: mean of the left/right Euclidean reprojection distances."""
    pred_left = project(points_3d, r, t, calib, shift=0.0)
    pred_right = project(points_3d, r, t, calib, shift=calib.baseline)
    err_left = np.linalg.norm(obs_left - pred_left, axis=1)
    err_right = np.linalg.norm(obs_right - pred_right, axis=1)
    return 0.5 * (err_left + err_right)


def estimate_egomotion_ransac(
    points_3d: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    calib,
    n_ransac_iterations: int = 50,
    sample_size: int = 3,
    inlier_threshold: float = 1.5,
    gn_iterations: int = 8,
    seed: int = 0,
):
    """RANSAC-robust (r, t): `n_ransac_iterations` trials of a `sample_size`-point
    Gauss-Newton fit, keep the trial with the most inliers (reprojection
    error < inlier_threshold px), then refine Gauss-Newton on that full
    inlier set for the final answer.

    Returns (r, t, inlier_mask): inlier_mask is over ALL input points,
    evaluated at the final refined (r, t) (not the winning trial's own fit).
    """
    n = len(points_3d)
    if n < sample_size:
        raise ValueError(f"need at least {sample_size} correspondences, got {n}")

    rng = np.random.default_rng(seed)

    best_inliers = None
    best_count = -1
    for _ in range(n_ransac_iterations):
        sample = rng.choice(n, size=sample_size, replace=False)
        r, t = estimate_egomotion(
            points_3d[sample], obs_left[sample], obs_right[sample], calib, n_iterations=gn_iterations
        )
        errors = reprojection_errors(points_3d, obs_left, obs_right, r, t, calib)
        inliers = errors < inlier_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    r, t = estimate_egomotion(
        points_3d[best_inliers], obs_left[best_inliers], obs_right[best_inliers], calib, n_iterations=gn_iterations
    )
    inlier_mask = reprojection_errors(points_3d, obs_left, obs_right, r, t, calib) < inlier_threshold

    return r, t, inlier_mask
