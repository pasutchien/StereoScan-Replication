"""Run egomotion + dense stereo + greedy fusion across a dataset directory,
producing a persistent colored PointCloudModel (Sec. III-C+D).

Generalizes the loop prototyped in tests/reconstruction/render/test.py into
a reusable, dataset-path-parameterized function, the same way
pipeline.runner.run_sequence generalized the egomotion-only trajectory
script. Builds directly on run_sequence for poses - no duplicated egomotion
logic.
"""

from pathlib import Path

import cv2
import numpy as np

from stereoscan.egomotion.calibration import load_stereo_calibration
from stereoscan.egomotion.trajectory import accumulate_poses
from stereoscan.pipeline.runner import resolve_calib_path, run_sequence
from stereoscan.reconstruction.model import PointCloudModel
from stereoscan.stereo_matching import compute_dense_disparity


def _load_image(dataset_dir: Path, frame_idx: int, camera: str) -> np.ndarray:
    path = dataset_dir / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def run_reconstruction(
    dataset_dir,
    calib_path=None,
    start_frame: int = 0,
    end_frame: int = None,
    grid_step: int = 10,
    progress: bool = True,
    **egomotion_kwargs,
):
    """Dense colored point-cloud reconstruction over [start_frame, end_frame].

    `grid_step` is the stereo-matching support-point grid spacing (see
    stereoscan.stereo_matching.compute_dense_disparity) - the module default
    (5) is quite slow per frame (~40s); a coarser grid (this function's
    default, 10) trades some adaptive-range precision for speed (~13s/frame),
    matching what the render demo used. `**egomotion_kwargs` pass through to
    run_sequence (e.g. feature_class, window_radius).

    Returns (model, result): the PointCloudModel and the SequenceResult from
    run_sequence (trajectory, poses' raw material, etc.).
    """
    dataset_dir = Path(dataset_dir)
    calib = load_stereo_calibration(resolve_calib_path(dataset_dir, calib_path))

    result = run_sequence(
        dataset_dir, calib_path=calib_path, start_frame=start_frame, end_frame=end_frame,
        progress=progress, **egomotion_kwargs,
    )
    poses = accumulate_poses(list(result.filtered_r), list(result.filtered_t))
    frame_indices = [start_frame] + list(result.frame_indices)

    model = PointCloudModel()
    for i, frame_idx in enumerate(frame_indices):
        left = _load_image(dataset_dir, frame_idx, "00")
        right = _load_image(dataset_dir, frame_idx, "01")
        disp, _, _ = compute_dense_disparity(left, right, grid_step=grid_step)
        R, t = poses[i]
        model.integrate_frame(disp, left, R, t, calib)
        if progress:
            print(f"  reconstruction frame {frame_idx}: model now {len(model.points)} points")

    return model, result
