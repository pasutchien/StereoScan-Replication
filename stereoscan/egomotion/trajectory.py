"""Chain per-frame relative (r, t) estimates into a global camera trajectory.

Each frame pair's (R, t) maps points from the previous camera frame's
coordinates into the current camera frame's coordinates:
X_i = R_i @ X_{i-1} + t_i (Eq. 1's convention). Composing these gives the
world(=frame 0)-to-camera-i transform (R_w, t_w); the camera center in
world coordinates is then the standard -R_w.T @ t_w.
"""

import numpy as np

from stereoscan.egomotion.reprojection import rotation_matrix


def accumulate_poses(relative_r: list, relative_t: list) -> list:
    """World(=frame 0)-to-camera-i transforms (R_i, t_i), one per frame,
    including the identity pose at frame 0 (index 0).

    relative_r, relative_t: length-N sequences of per-frame-pair (r, t)
    estimates (small-angle rotation vector + translation), one per
    consecutive frame pair, in order.

    Used wherever a WORLD point needs to be reprojected into a given
    frame's image plane (Sec. III-D), not just the camera-center
    trajectory accumulate_trajectory derives from these same poses.
    """
    R_world = np.eye(3)
    t_world = np.zeros(3)
    poses = [(R_world.copy(), t_world.copy())]

    for r, t in zip(relative_r, relative_t):
        R_rel = rotation_matrix(r)
        R_world = R_rel @ R_world
        t_world = R_rel @ t_world + t
        poses.append((R_world.copy(), t_world.copy()))

    return poses


def accumulate_trajectory(relative_r: list, relative_t: list) -> np.ndarray:
    """Camera-center trajectory in the frame-0 world frame.

    Returns an (N+1, 3) array; row 0 is the origin (frame 0's own camera
    center, by definition), row i is frame i's camera center. Derived from
    accumulate_poses via the standard -R.T @ t.
    """
    return np.array([-R.T @ t for R, t in accumulate_poses(relative_r, relative_t)])
