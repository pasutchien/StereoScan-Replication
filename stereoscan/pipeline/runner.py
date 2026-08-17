"""Run the full feature-matching + egomotion algorithm across an arbitrary
KITTI-raw-layout dataset directory, caching per-frame results.

This only orchestrates stereoscan.feature_matching / stereoscan.egomotion -
no algorithm changes. It exists to decouple *computing* the trajectory
(slow, ~1.5s/frame, variable) from *replaying* it (fast, fixed cadence):
run once, cache to disk, then any playback tool just reads the cache.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stereoscan.egomotion.bucketing import bucket_matches
from stereoscan.egomotion.calibration import load_stereo_calibration
from stereoscan.egomotion.kalman import EgomotionKalmanFilter
from stereoscan.egomotion.ransac import estimate_egomotion_ransac
from stereoscan.egomotion.reprojection import triangulate
from stereoscan.egomotion.timestamps import frame_dt, load_timestamps
from stereoscan.egomotion.trajectory import accumulate_trajectory
from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors
from stereoscan.feature_matching.features import detect_features
from stereoscan.feature_matching.filters import sobel_x, sobel_y
from stereoscan.feature_matching.matching import circular_match
from stereoscan.feature_matching.outliers import reject_sporadic_outliers

_CALIB_CANDIDATES = ("calib_cam_to_cam.txt",)


@dataclass
class SequenceResult:
    start_frame: int
    end_frame: int
    frame_indices: np.ndarray  # (N,) curr_frame index for each step
    dt: np.ndarray  # (N,)
    raw_r: np.ndarray  # (N,3)
    raw_t: np.ndarray  # (N,3)
    filtered_r: np.ndarray  # (N,3)
    filtered_t: np.ndarray  # (N,3)
    raw_trajectory: np.ndarray  # (N+1,3)
    filtered_trajectory: np.ndarray  # (N+1,3)


def resolve_calib_path(dataset_dir: Path, calib_path) -> Path:
    if calib_path is not None:
        return Path(calib_path)
    for candidate in (dataset_dir / "calib_cam_to_cam.txt", dataset_dir.parent / "config" / "calib_cam_to_cam.txt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find calib_cam_to_cam.txt near {dataset_dir}; pass calib_path explicitly"
    )


def _load_image(dataset_dir: Path, frame_idx: int, camera: str) -> np.ndarray:
    path = dataset_dir / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def _prepare(image: np.ndarray, feature_class: str):
    gx, gy = sobel_x(image), sobel_y(image)
    candidates = getattr(detect_features(image), feature_class)
    h, w = image.shape
    m = DESCRIPTOR_MARGIN
    keep = (candidates[:, 0] >= m) & (candidates[:, 0] < w - m) & (candidates[:, 1] >= m) & (candidates[:, 1] < h - m)
    points = candidates[keep]
    return points, compute_descriptors(gx, gy, points)


def run_sequence(
    dataset_dir,
    calib_path=None,
    start_frame: int = 0,
    end_frame: int = None,
    feature_class: str = "blob_max",
    window_radius: int = 50,
    epipolar_tolerance: int = 1,
    progress: bool = True,
) -> SequenceResult:
    """Run detect -> match -> outlier-reject -> bucket -> triangulate ->
    RANSAC+GN -> Kalman across [start_frame, end_frame] of a KITTI-raw-layout
    dataset directory (must contain image_00/, image_01/, image_00/timestamps.txt).
    """
    dataset_dir = Path(dataset_dir)
    calib = load_stereo_calibration(resolve_calib_path(dataset_dir, calib_path))
    timestamps = load_timestamps(dataset_dir / "image_00" / "timestamps.txt")

    if end_frame is None:
        end_frame = len(sorted((dataset_dir / "image_00" / "data").glob("*.png"))) - 1

    kf = EgomotionKalmanFilter()
    prev_left, prev_left_desc = _prepare(_load_image(dataset_dir, start_frame, "00"), feature_class)
    prev_right, prev_right_desc = _prepare(_load_image(dataset_dir, start_frame, "01"), feature_class)

    frame_indices, dts, raw_r, raw_t, filtered_r, filtered_t = [], [], [], [], [], []

    for curr_frame in range(start_frame + 1, end_frame + 1):
        curr_left, curr_left_desc = _prepare(_load_image(dataset_dir, curr_frame, "00"), feature_class)
        curr_right, curr_right_desc = _prepare(_load_image(dataset_dir, curr_frame, "01"), feature_class)

        matches = circular_match(
            curr_left, curr_left_desc, prev_left, prev_left_desc, prev_right, prev_right_desc,
            curr_right, curr_right_desc, window_radius=window_radius, epipolar_tolerance=epipolar_tolerance,
        )
        cl = curr_left[matches[:, 0]]
        pl = prev_left[matches[:, 1]]
        pr = prev_right[matches[:, 2]]
        cr = curr_right[matches[:, 3]]

        inlier_mask = reject_sporadic_outliers(cl, pl, cr)
        cl, pl, pr, cr = cl[inlier_mask], pl[inlier_mask], pr[inlier_mask], cr[inlier_mask]

        bucket_mask = bucket_matches(cl)
        cl, pl, pr, cr = cl[bucket_mask], pl[bucket_mask], pr[bucket_mask], cr[bucket_mask]

        disparity = (pl[:, 0] - pr[:, 0]).astype(np.float64)
        valid = disparity > 0
        points_3d = triangulate(pl[valid].astype(np.float64), pr[valid].astype(np.float64), calib)
        obs_left = cl[valid].astype(np.float64)
        obs_right = cr[valid].astype(np.float64)

        r_raw, t_raw, _ = estimate_egomotion_ransac(points_3d, obs_left, obs_right, calib)
        dt = frame_dt(timestamps, curr_frame)
        r_filt, t_filt = kf.step(r_raw, t_raw, dt)

        frame_indices.append(curr_frame)
        dts.append(dt)
        raw_r.append(r_raw)
        raw_t.append(t_raw)
        filtered_r.append(r_filt)
        filtered_t.append(t_filt)

        if progress and (curr_frame % 20 == 0 or curr_frame == end_frame):
            print(f"  frame {curr_frame}/{end_frame}")

        prev_left, prev_left_desc = curr_left, curr_left_desc
        prev_right, prev_right_desc = curr_right, curr_right_desc

    raw_trajectory = accumulate_trajectory(raw_r, raw_t)
    filtered_trajectory = accumulate_trajectory(filtered_r, filtered_t)

    return SequenceResult(
        start_frame=start_frame,
        end_frame=end_frame,
        frame_indices=np.array(frame_indices, dtype=np.int64),
        dt=np.array(dts, dtype=np.float64),
        raw_r=np.array(raw_r), raw_t=np.array(raw_t),
        filtered_r=np.array(filtered_r), filtered_t=np.array(filtered_t),
        raw_trajectory=raw_trajectory, filtered_trajectory=filtered_trajectory,
    )


def save_cache(result: SequenceResult, path) -> None:
    np.savez(
        path,
        start_frame=result.start_frame, end_frame=result.end_frame,
        frame_indices=result.frame_indices, dt=result.dt,
        raw_r=result.raw_r, raw_t=result.raw_t,
        filtered_r=result.filtered_r, filtered_t=result.filtered_t,
        raw_trajectory=result.raw_trajectory, filtered_trajectory=result.filtered_trajectory,
    )


def load_cache(path) -> SequenceResult:
    with np.load(path) as data:
        return SequenceResult(
            start_frame=int(data["start_frame"]), end_frame=int(data["end_frame"]),
            frame_indices=data["frame_indices"], dt=data["dt"],
            raw_r=data["raw_r"], raw_t=data["raw_t"],
            filtered_r=data["filtered_r"], filtered_t=data["filtered_t"],
            raw_trajectory=data["raw_trajectory"], filtered_trajectory=data["filtered_trajectory"],
        )
