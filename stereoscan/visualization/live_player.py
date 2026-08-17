"""Play a dataset's grayscale video and its precomputed trajectory side by
side, in sync, paced by the real inter-frame capture time.

Deliberately decoupled from computation: `run_sequence` (stereoscan.pipeline)
does the ~1.5s/frame algorithm work once, ahead of time; this module only
ever replays already-computed results, so playback pacing is independent of
how long the original computation took.

The trajectory panel is drawn with direct cv2 primitives rather than
matplotlib - per-frame Agg-to-array rendering has enough overhead to make a
live ~10fps loop stutter, while cv2.line/circle is effectively free at this
scale.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from stereoscan.egomotion.oxts import load_oxts, oxts_to_local_trajectory
from stereoscan.pipeline.runner import SequenceResult, load_cache, run_sequence, save_cache


def compute_trajectory_bounds(*trajectories: np.ndarray, padding: float = 5.0):
    """(x_min, x_max, z_min, z_max) covering all given trajectories, with padding.

    Computed once up front from the FULL trajectory/trajectories (not
    per-frame) so the plot's scale stays fixed for the whole playback
    instead of rescaling/jumping around as new points arrive. Pass both the
    estimated and ground-truth trajectories so neither gets clipped.
    """
    all_x = np.concatenate([t[:, 0] for t in trajectories])
    all_z = np.concatenate([t[:, 2] for t in trajectories])
    return all_x.min() - padding, all_x.max() + padding, all_z.min() - padding, all_z.max() + padding


def _draw_legend(canvas: np.ndarray, entries, origin=(14, 14)) -> None:
    """entries: list of (label, BGR color) tuples, drawn as a small
    boxed legend with a color-swatch line next to each label."""
    if not entries:
        return
    x0, y0 = origin
    line_h = 18
    swatch_w = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    label_w = max(cv2.getTextSize(label, font, font_scale, thickness)[0][0] for label, _ in entries)
    box_w = swatch_w + 10 + label_w + 12
    box_h = line_h * len(entries) + 10

    cv2.rectangle(canvas, (x0 - 6, y0 - 6), (x0 - 6 + box_w, y0 - 6 + box_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0 - 6, y0 - 6), (x0 - 6 + box_w, y0 - 6 + box_h), (0, 0, 0), 1)

    for i, (label, color) in enumerate(entries):
        y = y0 + i * line_h + 8
        cv2.line(canvas, (x0, y), (x0 + swatch_w, y), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + swatch_w + 8, y + 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _world_to_canvas(x: np.ndarray, z: np.ndarray, bounds, panel_size, margin: int = 20):
    x_min, x_max, z_min, z_max = bounds
    w, h = panel_size
    usable_w, usable_h = w - 2 * margin, h - 2 * margin
    x_range = max(x_max - x_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)
    scale = min(usable_w / x_range, usable_h / z_range)

    x_offset = margin + (usable_w - x_range * scale) / 2
    y_offset = margin + (usable_h - z_range * scale) / 2

    col = x_offset + (x - x_min) * scale
    row = h - y_offset - (z - z_min) * scale  # +z is "up" the panel, like Fig. 7
    return col, row


def compose_frame(
    video_frame: np.ndarray,
    trajectory_so_far: np.ndarray,
    bounds,
    ground_truth: np.ndarray = None,
    panel_size=None,
) -> np.ndarray:
    """Grayscale video frame + a top-down trajectory-so-far plot, side by side.

    `ground_truth`, if given, is drawn as a full static reference line (we
    already know the whole GPS/IMU path in advance, unlike the estimate,
    which only grows as playback proceeds).
    """
    video_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
    h, w = video_frame.shape
    if panel_size is None:
        panel_size = (h, h)  # square panel, matched to video height
    panel_w, panel_h = panel_size

    canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    legend = []

    if ground_truth is not None:
        gt_cols, gt_rows = _world_to_canvas(ground_truth[:, 0], ground_truth[:, 2], bounds, panel_size)
        gt_points = np.stack([gt_cols, gt_rows], axis=1).astype(np.int32)
        for p1, p2 in zip(gt_points[:-1], gt_points[1:]):
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 0, 255), 2, cv2.LINE_AA)  # red
        legend.append(("GPS/IMU", (0, 0, 255)))

    cols, rows = _world_to_canvas(trajectory_so_far[:, 0], trajectory_so_far[:, 2], bounds, panel_size)
    points = np.stack([cols, rows], axis=1).astype(np.int32)
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(canvas, tuple(p1), tuple(p2), (200, 0, 0), 2, cv2.LINE_AA)  # blue
    if len(points):
        cv2.circle(canvas, tuple(points[0]), 4, (0, 128, 0), -1)  # start
        cv2.circle(canvas, tuple(points[-1]), 6, (255, 0, 255), -1)  # current position (magenta, distinct from red GT)
    legend.append(("our method", (200, 0, 0)))

    _draw_legend(canvas, legend)

    if panel_h != h:
        canvas = cv2.resize(canvas, (panel_w, h))

    return np.hstack([video_bgr, canvas])


def play(
    dataset_dir,
    result: SequenceResult,
    ground_truth: np.ndarray = None,
    fps_override: float = None,
    panel_size=None,
) -> None:
    """Open a window and step through the cached result frame by frame.

    Paced by each frame's real capture dt (result.dt) unless fps_override
    is given. 'q' quits, spacebar pauses/resumes. `ground_truth`, if given,
    must be aligned to result.start_frame/end_frame (e.g. via
    oxts_to_local_trajectory on the matching oxts record slice).
    """
    dataset_dir = Path(dataset_dir)
    bounds = (
        compute_trajectory_bounds(result.filtered_trajectory, ground_truth)
        if ground_truth is not None
        else compute_trajectory_bounds(result.filtered_trajectory)
    )
    window_name = "StereoScan: video | trajectory"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    n = len(result.frame_indices)
    i = 0
    paused = False
    try:
        while i < n:
            frame_idx = int(result.frame_indices[i])
            video_path = dataset_dir / "image_00" / "data" / f"{frame_idx:010d}.png"
            video_frame = cv2.imread(str(video_path), cv2.IMREAD_GRAYSCALE)
            if video_frame is None:
                raise FileNotFoundError(f"could not read {video_path}")

            trajectory_so_far = result.filtered_trajectory[: i + 2]
            composed = compose_frame(video_frame, trajectory_so_far, bounds, ground_truth=ground_truth, panel_size=panel_size)
            cv2.imshow(window_name, composed)

            if paused:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = False
                continue

            delay_ms = int(1000 / fps_override) if fps_override else max(1, int(result.dt[i] * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = True
                continue
            i += 1
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play grayscale video + live trajectory side by side.")
    parser.add_argument("dataset_dir", help="KITTI-raw-layout dataset directory (contains image_00/, image_01/, ...)")
    parser.add_argument("--calib", default=None, help="path to calib_cam_to_cam.txt (auto-detected if omitted)")
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--end", type=int, default=None, help="last frame index (default: end of sequence)")
    parser.add_argument("--fps", type=float, default=None, help="override playback fps (default: real capture cadence)")
    args = parser.parse_args()

    # {dataset folder name}_cache.npz, next to wherever this is run from.
    # Note: the filename doesn't encode --start/--end, so switching frame
    # ranges on a dataset that already has a cache will silently reload the
    # old range - delete the cache file to force a recompute.
    cache_path = Path(f"{Path(args.dataset_dir).name}_cache.npz")

    if cache_path.exists():
        print(f"loading cached results from {cache_path}")
        result = load_cache(cache_path)
    else:
        print("running pipeline (this can take a while: ~1.5s/frame)...")
        result = run_sequence(args.dataset_dir, calib_path=args.calib, start_frame=args.start, end_frame=args.end)
        save_cache(result, cache_path)
        print(f"cached results to {cache_path}")

    oxts_dir = Path(args.dataset_dir) / "oxts" / "data"
    ground_truth = None
    if oxts_dir.is_dir():
        # Slice to [start_frame, end_frame] BEFORE converting, so the GT
        # trajectory is re-based to the same origin/heading as `result`
        # (oxts_to_local_trajectory always treats its first row as the
        # origin) - otherwise the two paths wouldn't share a reference frame.
        records = load_oxts(oxts_dir)[result.start_frame : result.end_frame + 1]
        ground_truth = oxts_to_local_trajectory(records)
    else:
        print(f"no oxts/ ground truth found at {oxts_dir}, playing without it")

    play(args.dataset_dir, result, ground_truth=ground_truth, fps_override=args.fps)


if __name__ == "__main__":
    main()
