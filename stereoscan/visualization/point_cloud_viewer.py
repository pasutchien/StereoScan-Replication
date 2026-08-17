"""Free-navigation 3D viewer for a PointCloudModel + trajectory, via Open3D.

Mirrors live_player.py's precompute-once-then-view split: run_reconstruction
(stereoscan.pipeline.reconstruction) is the slow part (~13-40s/frame), so
this module only ever loads an already-computed/cached model and opens an
interactive window - viewing never has to re-pay the reconstruction cost.

Like live_player's cv2.imshow loop, the interactive Open3D window is a live
GUI I can't drive or observe from my own tool calls - render_offscreen()
below is the part I actually verify myself (a static PNG snapshot of the
same geometry), the same role compose_frame() played for live_player.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from stereoscan.egomotion.calibration import load_stereo_calibration
from stereoscan.pipeline.reconstruction import run_reconstruction
from stereoscan.pipeline.runner import resolve_calib_path
from stereoscan.reconstruction.model import PointCloudModel


def build_point_cloud_geometry(model: PointCloudModel) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(model.points)
    gray = model.colors.astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(np.tile(gray[:, None], (1, 3)))
    return pcd


def build_trajectory_geometry(trajectory: np.ndarray, color=(1.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def _geometries(model: PointCloudModel, trajectory: np.ndarray):
    geometries = [build_point_cloud_geometry(model)]
    if trajectory is not None and len(trajectory) > 1:
        geometries.append(build_trajectory_geometry(trajectory))
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    return geometries


def _default_initial_view(calib, image_size):
    """Pinhole camera params for frame 0's pose, pulled back 8m and up 3m in
    its own local frame - the same viewpoint validated in the cv2 renderer
    (tests/reconstruction/render/test.py's "novel pulled back" view).

    Frame 0's pose is always (R=I, t=0) by accumulate_poses' construction,
    so this needs no actual pose data, just the calibration.
    """
    local_offset = np.array([0.0, -3.0, -8.0])  # (x, y=-up, z=-back)
    t = -local_offset

    w, h = image_size
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, calib.focal_length, calib.focal_length, calib.cu, calib.cv)
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = t
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extrinsic
    return params


def view(
    model: PointCloudModel,
    trajectory: np.ndarray = None,
    calib=None,
    image_size=(1242, 375),
    window_name: str = "StereoScan: 3D reconstruction",
) -> None:
    """Open an interactive free-navigation window (drag to orbit, scroll to
    zoom, ctrl/shift-drag to pan - standard Open3D mouse controls).

    Opens from a pulled-back, street-level-ish vantage by default (see
    _default_initial_view) rather than Open3D's auto-fit framing, which
    tends to end up looking nearly edge-on down a long thin scan like ours -
    geometrically fine, but unrecognizable. Pass calib=None to fall back to
    the auto-fit default instead (e.g. if you don't have calibration handy).
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for geom in _geometries(model, trajectory):
        vis.add_geometry(geom)

    if calib is not None:
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            _default_initial_view(calib, image_size), allow_arbitrary=True
        )

    vis.run()
    vis.destroy_window()


def render_offscreen(
    model: PointCloudModel, trajectory: np.ndarray, output_path, calib=None, image_size=(1242, 375), width: int = 1280, height: int = 720
) -> None:
    """Headless snapshot of the same geometry view() would show, from the
    same default viewpoint - used to verify the data without a live GUI."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for geom in _geometries(model, trajectory):
        vis.add_geometry(geom)
    if calib is not None:
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            _default_initial_view(calib, image_size), allow_arbitrary=True
        )
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path), do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Free-navigation 3D viewer for a dense reconstruction.")
    parser.add_argument("dataset_dir", help="KITTI-raw-layout dataset directory (contains image_00/, image_01/, ...)")
    parser.add_argument("--calib", default=None, help="path to calib_cam_to_cam.txt (auto-detected if omitted)")
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--end", type=int, default=None, help="last frame index (default: end of sequence)")
    parser.add_argument("--grid-step", type=int, default=10, help="support-point grid spacing (smaller = slower, more precise)")
    args = parser.parse_args()

    # {dataset folder name}_reconstruction_cache.npz - same auto-naming idea
    # as live_player, but a different suffix (this is a much larger/slower
    # cache than the egomotion-only one, and the two shouldn't collide).
    cache_path = Path(f"{Path(args.dataset_dir).name}_reconstruction_cache.npz")
    trajectory_path = cache_path.with_suffix(".trajectory.npy")

    if cache_path.exists() and trajectory_path.exists():
        print(f"loading cached reconstruction from {cache_path}")
        model = PointCloudModel.load(cache_path)
        trajectory = np.load(trajectory_path)
    else:
        print("running reconstruction (this can take a while: ~13-40s/frame)...")
        model, result = run_reconstruction(
            args.dataset_dir, calib_path=args.calib, start_frame=args.start, end_frame=args.end, grid_step=args.grid_step
        )
        trajectory = result.filtered_trajectory
        model.save(cache_path)
        np.save(trajectory_path, trajectory)
        print(f"cached reconstruction to {cache_path}")

    calib = load_stereo_calibration(resolve_calib_path(Path(args.dataset_dir), args.calib))

    print(f"viewing {len(model.points)} points, {len(trajectory)}-point trajectory")
    view(model, trajectory, calib=calib)


if __name__ == "__main__":
    main()
