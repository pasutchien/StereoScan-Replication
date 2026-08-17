"""Run the Kalman filter across several consecutive frame pairs and compare
raw RANSAC+GN (r, t) against the filtered estimate, writing a markdown table.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate results.md in this folder:

    python tests/egomotion/kalman/test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.egomotion.bucketing import bucket_matches  # noqa: E402
from stereoscan.egomotion.calibration import load_stereo_calibration  # noqa: E402
from stereoscan.egomotion.kalman import EgomotionKalmanFilter  # noqa: E402
from stereoscan.egomotion.ransac import estimate_egomotion_ransac  # noqa: E402
from stereoscan.egomotion.reprojection import triangulate  # noqa: E402
from stereoscan.egomotion.timestamps import frame_dt, load_timestamps  # noqa: E402
from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors  # noqa: E402
from stereoscan.feature_matching.features import detect_features  # noqa: E402
from stereoscan.feature_matching.filters import sobel_x, sobel_y  # noqa: E402
from stereoscan.feature_matching.matching import circular_match  # noqa: E402
from stereoscan.feature_matching.outliers import reject_sporadic_outliers  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"
CALIB_PATH = PROJECT_ROOT / "config" / "calib_cam_to_cam.txt"
TIMESTAMPS_PATH = DATA_ROOT / "image_00" / "timestamps.txt"

FRAME_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
FEATURE_CLASS = "blob_max"
WINDOW_RADIUS = 50
EPIPOLAR_TOLERANCE = 1


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


def estimate_pair_motion(calib, prev_frame, curr_frame):
    cl_pts, cl_desc = prepare(load_image(curr_frame, "00"))
    pl_pts, pl_desc = prepare(load_image(prev_frame, "00"))
    pr_pts, pr_desc = prepare(load_image(prev_frame, "01"))
    cr_pts, cr_desc = prepare(load_image(curr_frame, "01"))

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

    r, t, _ = estimate_egomotion_ransac(points_3d, obs_left, obs_right, calib)
    return r, t


def main():
    calib = load_stereo_calibration(CALIB_PATH)
    timestamps = load_timestamps(TIMESTAMPS_PATH)
    kf = EgomotionKalmanFilter()

    rows = []
    for prev_frame, curr_frame in FRAME_PAIRS:
        dt = frame_dt(timestamps, curr_frame)
        r_raw, t_raw = estimate_pair_motion(calib, prev_frame, curr_frame)
        r_filt, t_filt = kf.step(r_raw, t_raw, dt)
        rows.append((prev_frame, curr_frame, dt, r_raw, t_raw, r_filt, t_filt))
        print(f"({prev_frame},{curr_frame}) dt={dt:.4f}s raw_t_z={t_raw[2]:.3f} filtered_t_z={t_filt[2]:.3f}")

    raw_tz = np.array([row[4][2] for row in rows])
    filt_tz = np.array([row[6][2] for row in rows])

    lines = []
    lines.append("# Raw RANSAC+GN vs. Kalman-filtered egomotion\n")
    lines.append(
        f"`{FEATURE_CLASS}` class, window_radius={WINDOW_RADIUS}, "
        f"frame pairs: {FRAME_PAIRS}, real per-frame dt from timestamps.txt.\n"
    )
    lines.append("| pair | dt (s) | raw t_z (m) | filtered t_z (m) | raw rotation (deg) | filtered rotation (deg) |")
    lines.append("|---|---|---|---|---|---|")
    for prev_frame, curr_frame, dt, r_raw, t_raw, r_filt, t_filt in rows:
        raw_deg = np.degrees(r_raw)
        filt_deg = np.degrees(r_filt)
        lines.append(
            f"| ({prev_frame},{curr_frame}) | {dt:.4f} | {t_raw[2]:.3f} | {t_filt[2]:.3f} | "
            f"({raw_deg[0]:.3f}, {raw_deg[1]:.3f}, {raw_deg[2]:.3f}) | "
            f"({filt_deg[0]:.3f}, {filt_deg[1]:.3f}, {filt_deg[2]:.3f}) |"
        )

    lines.append("\n## Smoothing effect (forward translation t_z)\n")
    lines.append(f"- raw t_z std dev: {raw_tz.std():.4f} m")
    lines.append(f"- filtered t_z std dev: {filt_tz.std():.4f} m")

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote results to {OUTPUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
