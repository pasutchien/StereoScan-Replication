"""Greedy multi-view point fusion (Sec. III-D, Fig. 5).

"[We] propose a greedy approach which solves the association problem by
reprojecting reconstructed 3d points of the previous frame into the image
plane of the current frame. In case a point falls onto a valid disparity,
we fuse both 3d points by computing their 3d mean."

The paper doesn't state an explicit consistency threshold beyond "falls
onto a valid disparity" - this implementation additionally requires the
measured disparity there to be within `disparity_tolerance` of what the
existing point's own depth predicts, so a background point doesn't
silently get fused with an unrelated closer object that now happens to
reproject onto the same pixel.
"""

import numpy as np

from stereoscan.reconstruction.points import disparity_map_to_points


class PointCloudModel:
    """Persistent world-frame point cloud, built up incrementally frame by frame."""

    def __init__(self):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0,), dtype=np.uint8)

    def save(self, path) -> None:
        np.savez(path, points=self.points, colors=self.colors)

    @classmethod
    def load(cls, path) -> "PointCloudModel":
        model = cls()
        with np.load(path) as data:
            model.points = data["points"]
            model.colors = data["colors"]
        return model

    def integrate_frame(
        self,
        disparity_map: np.ndarray,
        left_image: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        calib,
        disparity_tolerance: float = 2.0,
        min_disparity: float = 2.0,
    ) -> None:
        """Fuse one frame's dense disparity map into the model.

        left_image: this frame's left grayscale image, for sampling each
        new point's color. R, t: this frame's world-to-camera pose (e.g. a
        row from stereoscan.egomotion.trajectory.accumulate_poses).
        `min_disparity` excludes near-zero-disparity pixels whose implied
        depth is too noise-sensitive to trust (see disparity_map_to_points).
        """
        pixel_uv, points_local, colors_new = disparity_map_to_points(disparity_map, calib, left_image, min_disparity=min_disparity)
        if len(points_local) == 0:
            return

        # R.T @ (X_cam - t), batched: world = local camera-frame points,
        # rotated/translated out of this frame's pose into the world frame.
        points_world_new = (points_local - t) @ R

        h, w = disparity_map.shape
        claimed = np.zeros(len(points_local), dtype=bool)

        if len(self.points):
            cam = self.points @ R.T + t  # existing world points -> this frame's camera coords
            in_front = cam[:, 2] > 0
            safe_z = np.where(in_front, cam[:, 2], 1.0)

            u = calib.focal_length * cam[:, 0] / safe_z + calib.cu
            v = calib.focal_length * cam[:, 1] / safe_z + calib.cv
            predicted_disp = calib.focal_length * calib.baseline / safe_z

            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            in_bounds = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

            measured_disp = np.full(len(self.points), np.nan)
            measured_disp[in_bounds] = disparity_map[vi[in_bounds], ui[in_bounds]]

            match = in_bounds & np.isfinite(measured_disp) & (np.abs(measured_disp - predicted_disp) <= disparity_tolerance)

            if np.any(match):
                pixel_index = np.full((h, w), -1, dtype=np.int64)
                pixel_index[pixel_uv[:, 1], pixel_uv[:, 0]] = np.arange(len(pixel_uv))

                matched_existing_idx = np.nonzero(match)[0]
                new_idx = pixel_index[vi[matched_existing_idx], ui[matched_existing_idx]]
                found = new_idx >= 0
                matched_existing_idx = matched_existing_idx[found]
                new_idx = new_idx[found]

                self.points[matched_existing_idx] = 0.5 * (self.points[matched_existing_idx] + points_world_new[new_idx])
                fused_colors = 0.5 * (self.colors[matched_existing_idx].astype(np.float64) + colors_new[new_idx].astype(np.float64))
                self.colors[matched_existing_idx] = np.round(fused_colors).astype(np.uint8)
                claimed[new_idx] = True

        self.points = np.vstack([self.points, points_world_new[~claimed]])
        self.colors = np.concatenate([self.colors, colors_new[~claimed]])
