"""3D triangulation and stereo reprojection (Sec. III-B, Eq. 1)."""

import numpy as np


def triangulate(left_points: np.ndarray, right_points: np.ndarray, calib) -> np.ndarray:
    """3D points, in the left camera's frame, for rectified stereo correspondences.

    left_points, right_points: (N, 2) (x, y) pixel coordinates from the same
    stereo pair (same frame, left vs right camera). Points with non-positive
    disparity (invalid / at-infinity) produce non-finite z and should be
    filtered by the caller.
    """
    disparity = left_points[:, 0] - right_points[:, 0]
    z = calib.focal_length * calib.baseline / disparity
    x = (left_points[:, 0] - calib.cu) * z / calib.focal_length
    y = (left_points[:, 1] - calib.cv) * z / calib.focal_length
    return np.column_stack([x, y, z])


def rotation_matrix(r: np.ndarray) -> np.ndarray:
    """R(r) = Rx(rx) @ Ry(ry) @ Rz(rz), per Eq. 1."""
    rx, ry, rz = r
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def project(points_3d: np.ndarray, r: np.ndarray, t: np.ndarray, calib, shift: float) -> np.ndarray:
    """pi(X; r, t) for a batch of points at once (Eq. 1).

    `shift` is Eq. 1's s: 0 for the left image, calib.baseline for the right.
    """
    R = rotation_matrix(r)
    cam = points_3d @ R.T + t
    cam = cam - np.array([shift, 0.0, 0.0])
    u = calib.focal_length * cam[:, 0] / cam[:, 2] + calib.cu
    v = calib.focal_length * cam[:, 1] / cam[:, 2] + calib.cv
    return np.column_stack([u, v])
