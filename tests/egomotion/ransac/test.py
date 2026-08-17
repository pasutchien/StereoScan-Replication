"""Compare plain (non-robust) Gauss-Newton against RANSAC-wrapped Gauss-Newton
for egomotion estimation, and write the results as a markdown table.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate results.md in this folder:

    python tests/egomotion/ransac/test.py

Frame 1 is used as "current" (frame 0 as "previous"), matching
tests/feature_matching/matching/test.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.egomotion.bucketing import bucket_matches  # noqa: E402
from stereoscan.egomotion.calibration import load_stereo_calibration  # noqa: E402
from stereoscan.egomotion.gauss_newton import estimate_egomotion  # noqa: E402
from stereoscan.egomotion.ransac import estimate_egomotion_ransac, reprojection_errors  # noqa: E402
from stereoscan.egomotion.reprojection import triangulate  # noqa: E402
from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors  # noqa: E402
from stereoscan.feature_matching.features import detect_features  # noqa: E402
from stereoscan.feature_matching.filters import sobel_x, sobel_y  # noqa: E402
from stereoscan.feature_matching.matching import circular_match  # noqa: E402
from stereoscan.feature_matching.outliers import reject_sporadic_outliers  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"
CALIB_PATH = PROJECT_ROOT / "config" / "calib_cam_to_cam.txt"

CURR_FRAME = 1
PREV_FRAME = 0
FEATURE_CLASS = "blob_max"
WINDOW_RADIUS = 50
EPIPOLAR_TOLERANCE = 1
INLIER_THRESHOLD_PX = 1.5


def load_image(frame_idx, camera):
    path = DATA_ROOT / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def prepare(image):
    gx, gy = sobel_x(image), sobel_y(image)
    candidates = getattr(detect_features(image), FEATURE_CLASS)
    h, w = image.shape
    m = DESCRIPTOR_MARGIN
    keep = (candidates[:, 0] >= m) & (candidates[:, 0] < w - m) & (candidates[:, 1] >= m) & (candidates[:, 1] < h - m)
    points = candidates[keep]
    return points, compute_descriptors(gx, gy, points)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def main():
    calib = load_stereo_calibration(CALIB_PATH)

    cl_pts, cl_desc = prepare(load_image(CURR_FRAME, "00"))
    pl_pts, pl_desc = prepare(load_image(PREV_FRAME, "00"))
    pr_pts, pr_desc = prepare(load_image(PREV_FRAME, "01"))
    cr_pts, cr_desc = prepare(load_image(CURR_FRAME, "01"))

    matches = circular_match(
        cl_pts, cl_desc, pl_pts, pl_desc, pr_pts, pr_desc, cr_pts, cr_desc,
        window_radius=WINDOW_RADIUS, epipolar_tolerance=EPIPOLAR_TOLERANCE,
    )
    curr_left = cl_pts[matches[:, 0]]
    prev_left = pl_pts[matches[:, 1]]
    prev_right = pr_pts[matches[:, 2]]
    curr_right = cr_pts[matches[:, 3]]

    inlier_mask = reject_sporadic_outliers(curr_left, prev_left, curr_right)
    curr_left, prev_left, prev_right, curr_right = (
        curr_left[inlier_mask], prev_left[inlier_mask], prev_right[inlier_mask], curr_right[inlier_mask],
    )

    bucket_mask = bucket_matches(curr_left)
    curr_left, prev_left, prev_right, curr_right = (
        curr_left[bucket_mask], prev_left[bucket_mask], prev_right[bucket_mask], curr_right[bucket_mask],
    )

    disparity = (prev_left[:, 0] - prev_right[:, 0]).astype(np.float64)
    valid = disparity > 0
    points_3d = triangulate(prev_left[valid].astype(np.float64), prev_right[valid].astype(np.float64), calib)
    obs_left = curr_left[valid].astype(np.float64)
    obs_right = curr_right[valid].astype(np.float64)
    n = len(points_3d)

    r_gn, t_gn = estimate_egomotion(points_3d, obs_left, obs_right, calib)
    err_gn = reprojection_errors(points_3d, obs_left, obs_right, r_gn, t_gn, calib)
    gn_inliers = err_gn < INLIER_THRESHOLD_PX

    r_rs, t_rs, rs_inliers = estimate_egomotion_ransac(
        points_3d, obs_left, obs_right, calib, inlier_threshold=INLIER_THRESHOLD_PX
    )
    err_rs = reprojection_errors(points_3d, obs_left, obs_right, r_rs, t_rs, calib)

    rows = [
        ("plain GN", gn_inliers.sum(), rms(err_gn[gn_inliers]), rms(err_gn), r_gn, t_gn),
        ("RANSAC + refined GN", rs_inliers.sum(), rms(err_rs[rs_inliers]), rms(err_rs), r_rs, t_rs),
    ]

    lines = []
    lines.append("# Plain Gauss-Newton vs. RANSAC-wrapped Gauss-Newton\n")
    lines.append(
        f"Frame {PREV_FRAME} -> {CURR_FRAME}, `{FEATURE_CLASS}` class, "
        f"{n} correspondences after outlier rejection + bucketing, "
        f"inlier threshold = {INLIER_THRESHOLD_PX}px.\n"
    )
    lines.append("| approach | inliers @ 1.5px | RMS on those inliers | RMS on full set |")
    lines.append("|---|---|---|---|")
    for name, count, rms_inliers, rms_full, _, _ in rows:
        lines.append(f"| {name} | {count} / {n} ({count / n:.1%}) | {rms_inliers:.2f}px | {rms_full:.2f}px |")

    lines.append("\n## Estimated motion\n")
    for name, _, _, _, r, t in rows:
        deg = np.degrees(r)
        lines.append(
            f"- **{name}**: rotation (deg) = ({deg[0]:.3f}, {deg[1]:.3f}, {deg[2]:.3f}), "
            f"translation (m) = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"
        )

    lines.append(
        "\nPlain GN's least-squares fit is dragged toward a compromise by leftover bad matches, so it "
        "barely satisfies the threshold for anyone. RANSAC finds a 3-point sample most of the data agrees "
        "with, then refines on that consensus set - producing a sub-pixel fit on a much larger inlier set, "
        "at the cost of a higher full-set RMS/max since it correctly stops averaging in the true outliers."
    )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote results to {OUTPUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
