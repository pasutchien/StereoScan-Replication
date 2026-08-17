"""Convert a dense disparity map into 3D points (Sec. III-D)."""

import numpy as np


def disparity_map_to_points(disparity_map: np.ndarray, calib, left_image: np.ndarray, min_disparity: float = 2.0):
    """(pixel_uv, points_3d, colors) for every pixel with disparity >= min_disparity.

    points_3d are in the LEFT camera's own frame for that map (same formula
    as egomotion.reprojection.triangulate, applied densely across the map
    rather than at a sparse set of matched points). `colors` is each point's
    grayscale intensity, sampled from `left_image` at its originating pixel
    (Fig. 8's renders are shaded by real image intensity, not a colormap).

    A plain `disparity > 0` filter isn't enough on its own: z = f*baseline/d
    is extremely sensitive to small errors in d as d -> 0 (sub-pixel noise
    near zero can imply a point hundreds of meters away), so a small
    positive floor is needed to keep near-zero-disparity noise out of the
    model rather than just excluding exactly-zero/negative disparity.
    """
    v, u = np.nonzero(np.isfinite(disparity_map) & (disparity_map >= min_disparity))
    d = disparity_map[v, u]
    z = calib.focal_length * calib.baseline / d
    x = (u - calib.cu) * z / calib.focal_length
    y = (v - calib.cv) * z / calib.focal_length

    pixel_uv = np.column_stack([u, v]).astype(np.int64)
    points_3d = np.column_stack([x, y, z])
    colors = left_image[v, u].astype(np.uint8)
    return pixel_uv, points_3d, colors
