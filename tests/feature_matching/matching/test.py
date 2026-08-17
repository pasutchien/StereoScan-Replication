"""Visualize circular matching results on frame 1 of the KITTI sequence.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate the images in this folder:

    python tests/feature_matching/matching/test.py

Frame 1 is used as "current" (frame 0 as "previous") since frame 0 has no
predecessor to circle-match against.
"""

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors  # noqa: E402
from stereoscan.feature_matching.features import detect_features  # noqa: E402
from stereoscan.feature_matching.filters import sobel_x, sobel_y  # noqa: E402
from stereoscan.feature_matching.matching import circular_match  # noqa: E402
from stereoscan.feature_matching.outliers import reject_sporadic_outliers  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"

CURR_FRAME = 1
PREV_FRAME = 0
FEATURE_CLASS = "blob_max"
WINDOW_RADIUS = 50
EPIPOLAR_TOLERANCE = 1
STEREO_SAMPLE_FRACTION = 0.3  # dense correspondences make stereo_correspondences.png unreadable otherwise
TAU_DISP = 5.0
TAU_FLOW = 5.0
MIN_SUPPORT = 2

POINT_RADIUS = 3
LINE_THICKNESS = 1
SUPPORT_EDGE_COLOR = (0, 255, 255)  # yellow: edge satisfies the disp/flow thresholds
NON_SUPPORT_EDGE_COLOR = (70, 70, 70)  # dim gray: mesh edge, but doesn't count as support
INLIER_COLOR = (0, 200, 0)
OUTLIER_COLOR = (0, 0, 255)


def load_image(frame_idx: int, camera: str) -> np.ndarray:
    path = DATA_ROOT / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def prepare(image: np.ndarray):
    """Detect + describe one feature class, dropping candidates too close to the border."""
    gx, gy = sobel_x(image), sobel_y(image)
    candidates = getattr(detect_features(image), FEATURE_CLASS)

    h, w = image.shape
    m = DESCRIPTOR_MARGIN
    keep = (candidates[:, 0] >= m) & (candidates[:, 0] < w - m) & (candidates[:, 1] >= m) & (candidates[:, 1] < h - m)
    points = candidates[keep]

    descriptors = compute_descriptors(gx, gy, points)
    return points, descriptors


def colors_from_flow(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Hue = motion direction, matching Fig. 4(b)'s 'colors encode track orientation'."""
    angle = np.arctan2(dy, dx)  # [-pi, pi]
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=1).reshape(-1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)


def colors_from_disparity(disparity: np.ndarray) -> np.ndarray:
    """Colormap disparity, matching Fig. 4(a)'s 'colors encode disparities'."""
    lo, hi = np.percentile(disparity, [1, 99])
    hi = max(hi, lo + 1e-6)
    normalized = np.clip((disparity - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)


def draw_temporal_flow(curr_image: np.ndarray, curr_pts: np.ndarray, prev_pts: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    flow = curr_pts - prev_pts
    colors = colors_from_flow(flow[:, 0], flow[:, 1])
    for (cx, cy), (px, py), color in zip(curr_pts, prev_pts, colors):
        color = tuple(int(c) for c in color)
        cv2.line(vis, (px, py), (cx, cy), color, LINE_THICKNESS, cv2.LINE_AA)
        cv2.circle(vis, (px, py), POINT_RADIUS, color, 1)  # hollow: previous position
        cv2.circle(vis, (cx, cy), POINT_RADIUS, color, -1)  # filled: current position
    return vis


def sample_rows(n: int, fraction: float, seed: int = 0) -> np.ndarray:
    """Indices of a random `fraction` of `n` rows, for a less cluttered plot."""
    if n == 0:
        return np.empty(0, dtype=np.int64)
    count = max(1, round(n * fraction))
    return np.random.default_rng(seed).choice(n, size=count, replace=False)


def draw_stereo_correspondences(left_image: np.ndarray, right_image: np.ndarray, left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
    canvas = np.hstack([cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)])
    width = left_image.shape[1]
    disparity = left_pts[:, 0] - right_pts[:, 0]
    colors = colors_from_disparity(disparity)
    for (lx, ly), (rx, ry), color in zip(left_pts, right_pts, colors):
        color = tuple(int(c) for c in color)
        p1, p2 = (lx, ly), (rx + width, ry)
        cv2.line(canvas, p1, p2, color, LINE_THICKNESS, cv2.LINE_AA)
        cv2.circle(canvas, p1, POINT_RADIUS, color, -1)
        cv2.circle(canvas, p2, POINT_RADIUS, color, -1)
    return canvas


def draw_delaunay_mesh(
    image: np.ndarray,
    curr_left_pts: np.ndarray,
    prev_left_pts: np.ndarray,
    curr_right_pts: np.ndarray,
    inlier_mask: np.ndarray,
    tau_disp: float,
    tau_flow: float,
) -> np.ndarray:
    """The Delaunay triangulation reject_sporadic_outliers builds on curr_left_pts.

    Yellow edges satisfy the disp/flow support thresholds, gray edges don't.
    Points are green if kept as an inlier, red if rejected.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    disparity = curr_left_pts[:, 0].astype(np.float64) - curr_right_pts[:, 0].astype(np.float64)
    flow = (curr_left_pts - prev_left_pts).astype(np.float64)

    triangulation = Delaunay(curr_left_pts.astype(np.float64))
    edges = {
        (min(a, b), max(a, b))
        for simplex in triangulation.simplices
        for a, b in combinations(simplex, 2)
    }

    for i, j in edges:
        disp_ok = abs(disparity[i] - disparity[j]) <= tau_disp
        flow_ok = np.linalg.norm(flow[i] - flow[j]) <= tau_flow
        color = SUPPORT_EDGE_COLOR if (disp_ok and flow_ok) else NON_SUPPORT_EDGE_COLOR
        p1 = (int(curr_left_pts[i, 0]), int(curr_left_pts[i, 1]))
        p2 = (int(curr_left_pts[j, 0]), int(curr_left_pts[j, 1]))
        cv2.line(vis, p1, p2, color, LINE_THICKNESS, cv2.LINE_AA)

    for (x, y), kept in zip(curr_left_pts, inlier_mask):
        color = INLIER_COLOR if kept else OUTLIER_COLOR
        cv2.circle(vis, (int(x), int(y)), POINT_RADIUS, color, -1)

    return vis


def main():
    curr_left_img = load_image(CURR_FRAME, "00")
    prev_left_img = load_image(PREV_FRAME, "00")
    prev_right_img = load_image(PREV_FRAME, "01")
    curr_right_img = load_image(CURR_FRAME, "01")

    cl_pts, cl_desc = prepare(curr_left_img)
    pl_pts, pl_desc = prepare(prev_left_img)
    pr_pts, pr_desc = prepare(prev_right_img)
    cr_pts, cr_desc = prepare(curr_right_img)

    matches = circular_match(
        cl_pts, cl_desc, pl_pts, pl_desc, pr_pts, pr_desc, cr_pts, cr_desc,
        window_radius=WINDOW_RADIUS, epipolar_tolerance=EPIPOLAR_TOLERANCE,
    )
    print(f"{FEATURE_CLASS}: {len(cl_pts)} curr_left candidates -> {len(matches)} accepted circular matches")

    curr_left = cl_pts[matches[:, 0]]
    prev_left = pl_pts[matches[:, 1]]
    curr_right = cr_pts[matches[:, 3]]

    inlier_mask = reject_sporadic_outliers(
        curr_left, prev_left, curr_right, tau_disp=TAU_DISP, tau_flow=TAU_FLOW, min_support=MIN_SUPPORT
    )
    print(f"outlier rejection: {inlier_mask.sum()} kept, {(~inlier_mask).sum()} rejected")

    cv2.imwrite(str(OUTPUT_DIR / "current_left.png"), curr_left_img)
    cv2.imwrite(str(OUTPUT_DIR / "prev_left.png"), prev_left_img)
    cv2.imwrite(str(OUTPUT_DIR / "current_right.png"), curr_right_img)

    temporal_vis = draw_temporal_flow(curr_left_img, curr_left, prev_left)
    cv2.imwrite(str(OUTPUT_DIR / "temporal_flow.png"), temporal_vis)

    sample = sample_rows(len(matches), STEREO_SAMPLE_FRACTION)
    stereo_vis = draw_stereo_correspondences(curr_left_img, curr_right_img, curr_left[sample], curr_right[sample])
    cv2.imwrite(str(OUTPUT_DIR / "stereo_correspondences.png"), stereo_vis)

    curr_left_in = curr_left[inlier_mask]
    prev_left_in = prev_left[inlier_mask]
    curr_right_in = curr_right[inlier_mask]

    temporal_vis_in = draw_temporal_flow(curr_left_img, curr_left_in, prev_left_in)
    cv2.imwrite(str(OUTPUT_DIR / "temporal_flow_inliers.png"), temporal_vis_in)

    sample_in = sample_rows(len(curr_left_in), STEREO_SAMPLE_FRACTION)
    stereo_vis_in = draw_stereo_correspondences(
        curr_left_img, curr_right_img, curr_left_in[sample_in], curr_right_in[sample_in]
    )
    cv2.imwrite(str(OUTPUT_DIR / "stereo_correspondences_inliers.png"), stereo_vis_in)

    mesh_vis = draw_delaunay_mesh(curr_left_img, curr_left, prev_left, curr_right, inlier_mask, TAU_DISP, TAU_FLOW)
    cv2.imwrite(str(OUTPUT_DIR / "delaunay_mesh.png"), mesh_vis)

    print(f"wrote images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
