"""Render a PointCloudModel from an arbitrary virtual viewpoint (Fig. 8 style).

Depth-buffering is done with a single vectorized fancy-index assignment:
points are sorted far-to-near before assignment, so NumPy's documented
"last write wins" behavior for duplicate indices naturally keeps the
nearest point at each pixel, with no explicit z-buffer array or Python loop.
"""

import cv2
import numpy as np


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    calib,
    image_size,
    point_radius: int = 1,
    background=(255, 255, 255),
) -> np.ndarray:
    """(H, W, 3) BGR uint8 render of `points` (world-frame, (N,3)) colored by
    `colors` (grayscale, (N,)) from a virtual camera at world-to-camera
    pose (R, t). `image_size` is (w, h).
    """
    w, h = image_size
    cam = points @ R.T + t
    in_front = cam[:, 2] > 0
    safe_z = np.where(in_front, cam[:, 2], 1.0)

    u = calib.focal_length * cam[:, 0] / safe_z + calib.cu
    v = calib.focal_length * cam[:, 1] / safe_z + calib.cv
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    visible = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, depth, color = ui[visible], vi[visible], cam[visible, 2], colors[visible]

    order = np.argsort(-depth)  # far to near, so nearer points are written last
    ui, vi, color = ui[order], vi[order], color[order]

    # Work on a black canvas + separate coverage mask so dilation (a
    # max-filter) can only ever grow colored splats into empty space, never
    # the other way around - doing this directly against a light/white
    # background would let the background "win" the max and erase points.
    image = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.uint8)
    image[vi, ui] = color[:, None]  # broadcasts grayscale to all 3 channels
    coverage[vi, ui] = 255

    if point_radius > 1:
        kernel = np.ones((point_radius, point_radius), np.uint8)
        image = cv2.dilate(image, kernel)
        coverage = cv2.dilate(coverage, kernel)

    result = np.full((h, w, 3), background, dtype=np.uint8)
    mask = coverage > 0
    result[mask] = image[mask]
    return result


def project_trajectory(trajectory: np.ndarray, R: np.ndarray, t: np.ndarray, calib, image_size) -> np.ndarray:
    """(M, 2) float pixel coords of a world-frame trajectory in the same
    virtual view as render_point_cloud. Points behind the camera become NaN,
    breaking the polyline there rather than drawing a bogus wraparound segment.
    """
    cam = trajectory @ R.T + t
    in_front = cam[:, 2] > 0
    safe_z = np.where(in_front, cam[:, 2], 1.0)

    u = calib.focal_length * cam[:, 0] / safe_z + calib.cu
    v = calib.focal_length * cam[:, 1] / safe_z + calib.cv
    pixels = np.column_stack([u, v])
    pixels[~in_front] = np.nan
    return pixels


def draw_dashed_trajectory(
    image: np.ndarray,
    pixel_trajectory: np.ndarray,
    color=(0, 0, 255),
    thickness: int = 2,
    dash_len: float = 8.0,
    gap_len: float = 6.0,
) -> None:
    """Draw a dashed polyline in place (mirrors Fig. 8's dashed red trajectory)."""
    for p1, p2 in zip(pixel_trajectory[:-1], pixel_trajectory[1:]):
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            continue
        p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1e-6:
            continue
        direction = (p2 - p1) / seg_len

        dist = 0.0
        while dist < seg_len:
            dash_end = min(dist + dash_len, seg_len)
            start = p1 + direction * dist
            end = p1 + direction * dash_end
            cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)
            dist += dash_len + gap_len
