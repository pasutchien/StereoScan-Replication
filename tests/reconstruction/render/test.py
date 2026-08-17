"""Build a small dense 3D reconstruction and render it (Fig. 8 style): one
view from an actual historical camera pose (comparable to that frame's real
photo) and one from a synthetic "pulled back and up" viewpoint the sequence
never actually visited, to demonstrate a genuinely novel render rather than
just replaying a stored photo.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate the images in this folder:

    python tests/reconstruction/render/test.py

Uses a coarser support-point grid (grid_step=10, vs. the module default 5)
purely to keep this demo's runtime down - dense stereo matching's current
bottleneck is support-point detection at ~40s/frame with the default grid.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.egomotion.calibration import load_stereo_calibration  # noqa: E402
from stereoscan.egomotion.trajectory import accumulate_poses  # noqa: E402
from stereoscan.pipeline.runner import run_sequence  # noqa: E402
from stereoscan.reconstruction.model import PointCloudModel  # noqa: E402
from stereoscan.stereo_matching import compute_dense_disparity  # noqa: E402
from stereoscan.visualization.point_cloud_render import (  # noqa: E402
    draw_dashed_trajectory,
    project_trajectory,
    render_point_cloud,
)

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"
CALIB_PATH = PROJECT_ROOT / "config" / "calib_cam_to_cam.txt"

N_FRAMES = 8  # frames 0..N_FRAMES-1
GRID_STEP = 10  # coarser than the module default (5), for demo speed


def load_image(frame_idx: int, camera: str) -> np.ndarray:
    path = DATA_ROOT / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def main():
    calib = load_stereo_calibration(CALIB_PATH)

    print(f"running egomotion across {N_FRAMES} frames...")
    result = run_sequence(DATA_ROOT, calib_path=CALIB_PATH, start_frame=0, end_frame=N_FRAMES - 1)
    poses = accumulate_poses(list(result.filtered_r), list(result.filtered_t))

    model = PointCloudModel()
    print("building dense reconstruction...")
    left = None
    for i in range(N_FRAMES):
        t0 = time.time()
        left = load_image(i, "00")
        right = load_image(i, "01")
        disp, _, _ = compute_dense_disparity(left, right, grid_step=GRID_STEP)
        R, t = poses[i]
        model.integrate_frame(disp, left, R, t, calib)
        print(f"  frame {i}: {time.time() - t0:.1f}s, model now {len(model.points)} points")

    print(f"final model: {len(model.points)} points")
    image_size = (left.shape[1], left.shape[0])  # (w, h)
    trajectory = result.filtered_trajectory

    # View 1: frame 0's own historical pose. Deliberately NOT the last
    # frame's pose: a forward-facing camera only ever sees the trajectory
    # that lies ahead of it, and frame 0 is the one pose the whole
    # subsequently-driven path is guaranteed to be in front of.
    R_hist, t_hist = poses[0]
    view1 = render_point_cloud(model.points, model.colors, R_hist, t_hist, calib, image_size, point_radius=2)
    draw_dashed_trajectory(view1, project_trajectory(trajectory, R_hist, t_hist, calib, image_size))
    cv2.imwrite(str(OUTPUT_DIR / "view_historical_pose.png"), view1)
    cv2.imwrite(str(OUTPUT_DIR / "view_historical_pose_source_photo.png"), load_image(0, "00"))

    # View 2: pulled back 8m and up 3m in frame 0's OWN local frame (y is
    # down, z is forward, so "up" = -y, "back" = -z), same orientation.
    # Camera center: C = -R.T@t: solving for the new t at a shifted center
    # C_novel = C_hist + R_hist.T @ local_offset with R_novel = R_hist
    # collapses to t_novel = t_hist - local_offset.
    local_offset = np.array([0.0, -3.0, -8.0])  # (x, y=-up, z=-back)
    R_novel = R_hist
    t_novel = t_hist - local_offset
    view2 = render_point_cloud(model.points, model.colors, R_novel, t_novel, calib, image_size, point_radius=2)
    draw_dashed_trajectory(view2, project_trajectory(trajectory, R_novel, t_novel, calib, image_size))
    cv2.imwrite(str(OUTPUT_DIR / "view_novel_pulled_back.png"), view2)

    print(f"wrote renders to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
