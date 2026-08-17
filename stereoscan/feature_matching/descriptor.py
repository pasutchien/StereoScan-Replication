"""16-location Sobel descriptor and SAD matching (Geiger et al. 2011, Fig. 3c)."""

import numpy as np

# (dx, dy) sample offsets relative to the feature center, given as absolute
# (row, col) positions in an 11x11 patch (0..10, center at 5) and recentered
# here by subtracting 5. Point-symmetric about the center, per Fig. 3c.
OFFSETS = np.array(
    [
        (-1, -5), (1, -5),
        (-5, -3), (5, -3),
        (-3, -1), (-1, -1), (1, -1), (3, -1),
        (-3, 1), (-1, 1), (1, 1), (3, 1),
        (-5, 3), (5, 3),
        (-1, 5), (1, 5),
    ],
    dtype=np.int32,
)

# Minimum distance a feature must keep from the image border for every
# offset to land inside the image.
DESCRIPTOR_MARGIN = int(np.abs(OFFSETS).max())


def quantize_to_uint8(response: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Clamp response*scale to a signed 8-bit range and shift it to unsigned.

    Matches the paper's "quantize the Sobel responses to 8 bits": values are
    clipped to [-128, 127] then shifted by +128 into [0, 255].
    """
    clipped = np.clip(response * scale, -128, 127)
    return (clipped + 128).astype(np.uint8)


def compute_descriptors(
    sobel_x_response: np.ndarray,
    sobel_y_response: np.ndarray,
    points: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Build 32-byte descriptors (16 quantized Sobel-x + 16 quantized Sobel-y samples).

    `points` is an (N, 2) array of (x, y) feature locations. Every point
    must be at least DESCRIPTOR_MARGIN pixels from the image border, since
    the sample offsets reach that far from the center; points that violate
    this raise ValueError rather than silently wrapping around the array.
    """
    points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    height, width = sobel_x_response.shape

    xs = points[:, 0:1] + OFFSETS[:, 0]  # (N, 16)
    ys = points[:, 1:2] + OFFSETS[:, 1]  # (N, 16)

    if len(points) and (xs.min() < 0 or xs.max() >= width or ys.min() < 0 or ys.max() >= height):
        raise ValueError(
            f"compute_descriptors: points must be >= {DESCRIPTOR_MARGIN}px from the image border"
        )

    gx = quantize_to_uint8(sobel_x_response[ys, xs], scale)
    gy = quantize_to_uint8(sobel_y_response[ys, xs], scale)
    return np.concatenate([gx, gy], axis=1)


def sad_matrix(descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> np.ndarray:
    """Pairwise sum-of-absolute-differences between two sets of descriptors.

    Returns an (Na, Nb) int32 cost matrix, computed as one vectorized batch
    (Na * Nb * 32 element ops) rather than the paper's per-pair SSE SAD
    instruction, since NumPy has no direct SIMD intrinsic access. This scales
    fine for small/pre-filtered candidate sets, but matching thousands of
    features against thousands of others should first narrow `descriptors_b`
    to a local search window (per the paper's circular matching scheme)
    rather than comparing every feature against every other feature.
    """
    a = descriptors_a.astype(np.int16)
    b = descriptors_b.astype(np.int16)
    return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2).astype(np.int32)
