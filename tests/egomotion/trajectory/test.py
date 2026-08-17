"""Run the full egomotion pipeline across the whole KITTI sequence and plot
the accumulated camera trajectory against GPS/IMU ground truth, matching the
paper's Fig. 7 (x vs z, top-down).

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate trajectory.png + results.md:

    python tests/egomotion/trajectory/test.py
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.egomotion.bucketing import bucket_matches  # noqa: E402
from stereoscan.egomotion.calibration import load_stereo_calibration  # noqa: E402
from stereoscan.egomotion.kalman import EgomotionKalmanFilter  # noqa: E402
from stereoscan.egomotion.oxts import load_oxts, oxts_to_local_trajectory  # noqa: E402
from stereoscan.egomotion.ransac import estimate_egomotion_ransac  # noqa: E402
from stereoscan.egomotion.reprojection import triangulate  # noqa: E402
from stereoscan.egomotion.timestamps import frame_dt, load_timestamps  # noqa: E402
from stereoscan.egomotion.trajectory import accumulate_trajectory  # noqa: E402
from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors  # noqa: E402
from stereoscan.feature_matching.features import detect_features  # noqa: E402
from stereoscan.feature_matching.filters import sobel_x, sobel_y  # noqa: E402
from stereoscan.feature_matching.matching import circular_match  # noqa: E402
from stereoscan.feature_matching.outliers import reject_sporadic_outliers  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"
CALIB_PATH = PROJECT_ROOT / "config" / "calib_cam_to_cam.txt"
TIMESTAMPS_PATH = DATA_ROOT / "image_00" / "timestamps.txt"
OXTS_DATA_DIR = DATA_ROOT / "oxts" / "data"

FEATURE_CLASS = "blob_max"
WINDOW_RADIUS = 50
EPIPOLAR_TOLERANCE = 1
N_FRAMES = len(list((DATA_ROOT / "image_00" / "data").glob("*.png")))


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


def estimate_pair_motion(calib, prev_cache, curr_frame):
    """prev_cache holds the previous frame's (points, desc) for left/right so
    each frame's detection work is only ever done once, not twice."""
    cl_pts, cl_desc = prepare(load_image(curr_frame, "00"))
    cr_pts, cr_desc = prepare(load_image(curr_frame, "01"))
    pl_pts, pl_desc, pr_pts, pr_desc = prev_cache

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
    return r, t, (cl_pts, cl_desc, cr_pts, cr_desc)


def main():
    calib = load_stereo_calibration(CALIB_PATH)
    timestamps = load_timestamps(TIMESTAMPS_PATH)
    kf = EgomotionKalmanFilter()

    print(f"running egomotion across {N_FRAMES} frames...")
    t_start = time.time()

    prev_cache = (
        *prepare(load_image(0, "00")),
        *prepare(load_image(0, "01")),
    )
    raw_r, raw_t, filtered_r, filtered_t = [], [], [], []
    for curr_frame in range(1, N_FRAMES):
        dt = frame_dt(timestamps, curr_frame)
        r_raw, t_raw, prev_cache = estimate_pair_motion(calib, prev_cache, curr_frame)
        r_filt, t_filt = kf.step(r_raw, t_raw, dt)
        raw_r.append(r_raw)
        raw_t.append(t_raw)
        filtered_r.append(r_filt)
        filtered_t.append(t_filt)
        if curr_frame % 20 == 0 or curr_frame == N_FRAMES - 1:
            elapsed = time.time() - t_start
            print(f"  frame {curr_frame}/{N_FRAMES - 1}  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t_start
    print(f"done in {total_time:.1f}s")

    raw_traj = accumulate_trajectory(raw_r, raw_t)
    filtered_traj = accumulate_trajectory(filtered_r, filtered_t)

    oxts_records = load_oxts(OXTS_DATA_DIR)[:N_FRAMES]
    gt_traj = oxts_to_local_trajectory(oxts_records)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_traj[:, 0], gt_traj[:, 2], color="red", label="GPS/IMU")
    ax.plot(raw_traj[:, 0], raw_traj[:, 2], color="green", linestyle="--", label="our method (raw)")
    ax.plot(filtered_traj[:, 0], filtered_traj[:, 2], color="blue", label="our method (Kalman-filtered)")
    ax.set_xlabel("x [meters]")
    ax.set_ylabel("z [meters]")
    ax.set_title("2011_09_26_drive_0001_sync: visual odometry vs. GPS/IMU")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory.png", dpi=150)

    def path_length(traj):
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    def final_error(a, b):
        return float(np.linalg.norm(a[-1] - b[-1]))

    report = "\n".join([
        "# Visual odometry trajectory vs. GPS/IMU\n",
        f"{N_FRAMES} frames, `{FEATURE_CLASS}` class, total pipeline time {total_time:.1f}s.\n",
        "| trajectory | path length (m) | final-position error vs. GPS/IMU (m) |",
        "|---|---|---|",
        f"| GPS/IMU | {path_length(gt_traj):.2f} | - |",
        f"| our method (raw) | {path_length(raw_traj):.2f} | {final_error(raw_traj, gt_traj):.2f} |",
        f"| our method (Kalman-filtered) | {path_length(filtered_traj):.2f} | {final_error(filtered_traj, gt_traj):.2f} |",
        "\nSee trajectory.png for the plotted paths.\n",
    ])
    (OUTPUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote trajectory.png and results.md to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
