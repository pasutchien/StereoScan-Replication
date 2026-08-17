"""Gauss-Newton reprojection-error minimization (Sec. III-B, Eq. 2).

Jacobians are estimated numerically (central differences) rather than
derived analytically from Eq. 1 - simpler to get right without an algebra
mistake, at some speed cost we aren't chasing yet in this recreation.
"""

import numpy as np

from stereoscan.egomotion.reprojection import project


def _residuals(params: np.ndarray, points_3d: np.ndarray, obs_left: np.ndarray, obs_right: np.ndarray, calib) -> np.ndarray:
    r, t = params[:3], params[3:]
    pred_left = project(points_3d, r, t, calib, shift=0.0)
    pred_right = project(points_3d, r, t, calib, shift=calib.baseline)
    return np.concatenate([(obs_left - pred_left).ravel(), (obs_right - pred_right).ravel()])


def estimate_egomotion(
    points_3d: np.ndarray,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    calib,
    n_iterations: int = 8,
    eps: float = 1e-6,
):
    """Gauss-Newton estimate of (r, t) minimizing Eq. 2's reprojection error.

    points_3d: (N, 3) previous-frame points, triangulated via `triangulate`.
    obs_left, obs_right: (N, 2) observed CURRENT-frame (x, y) feature
    locations that points_3d should reproject onto once transformed by
    (r, t) - i.e. curr_left / curr_right from the matched correspondences.

    Initializes at (r, t) = 0 and runs a fixed `n_iterations` (the paper
    notes 4-8 is typically enough for convergence). Returns (r, t), each a
    length-3 np.ndarray.
    """
    params = np.zeros(6)
    for _ in range(n_iterations):
        r0 = _residuals(params, points_3d, obs_left, obs_right, calib)

        J = np.empty((len(r0), 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = eps
            r_plus = _residuals(params + step, points_3d, obs_left, obs_right, calib)
            r_minus = _residuals(params - step, points_3d, obs_left, obs_right, calib)
            J[:, k] = (r_plus - r_minus) / (2 * eps)

        delta, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        params = params + delta

    return params[:3], params[3:]
