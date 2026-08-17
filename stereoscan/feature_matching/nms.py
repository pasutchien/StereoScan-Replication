"""Non-maximum / non-minimum suppression (Neubeck & Van Gool, 2006)."""

import numpy as np
from scipy import ndimage


def non_maximum_suppression(response: np.ndarray, radius: int = 3, threshold: float = 50.0) -> np.ndarray:
    """Locate strict local maxima of `response` exceeding `threshold`.

    Returns an (N, 2) int32 array of (x, y) pixel coordinates. Pixels
    within `radius` of the image border are excluded since their
    (2*radius+1) square window would be incomplete.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")

    size = 2 * radius + 1
    local_max = ndimage.maximum_filter(response, size=size, mode="nearest")
    mask = (response == local_max) & (response > threshold)
    mask[:radius, :] = False
    mask[-radius:, :] = False
    mask[:, :radius] = False
    mask[:, -radius:] = False

    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(np.int32)


def non_minimum_suppression(response: np.ndarray, radius: int = 3, threshold: float = 50.0) -> np.ndarray:
    """Locate strict local minima of `response` below `-threshold`.

    Equivalent to non_maximum_suppression(-response, ...); see that
    function for the return format and border handling.
    """
    return non_maximum_suppression(-response, radius=radius, threshold=threshold)
