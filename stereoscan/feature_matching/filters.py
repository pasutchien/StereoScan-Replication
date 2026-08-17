"""5x5 blob/corner masks and 3x3 Sobel masks (Geiger et al. 2011, Fig. 3a-c)."""

import cv2
import numpy as np

SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1],
    ],
    dtype=np.float32,
)

SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [+1, +2, +1],
    ],
    dtype=np.float32,
)

BLOB_KERNEL = np.array(
    [
        [-1, -1, -1, -1, -1],
        [-1, +1, +1, +1, -1],
        [-1, +1, +8, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, -1, -1, -1],
    ],
    dtype=np.float32,
)

CORNER_KERNEL = np.array(
    [
        [-1, -1, 0, +1, +1],
        [-1, -1, 0, +1, +1],
        [0, 0, 0, 0, 0],
        [+1, +1, 0, -1, -1],
        [+1, +1, 0, -1, -1],
    ],
    dtype=np.float32,
)


def filter_blob(image: np.ndarray, border_type: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Apply the 5x5 blob mask, returning a float32 response map same size as image."""
    return cv2.filter2D(image.astype(np.float32), cv2.CV_32F, BLOB_KERNEL, borderType=border_type)


def filter_corner(image: np.ndarray, border_type: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Apply the 5x5 corner mask, returning a float32 response map same size as image."""
    return cv2.filter2D(image.astype(np.float32), cv2.CV_32F, CORNER_KERNEL, borderType=border_type)


def sobel_x(image: np.ndarray, border_type: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Horizontal-gradient Sobel response (positive where intensity rises left-to-right)."""
    return cv2.filter2D(image.astype(np.float32), cv2.CV_32F, SOBEL_X_KERNEL, borderType=border_type)


def sobel_y(image: np.ndarray, border_type: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Vertical-gradient Sobel response (positive where intensity rises top-to-bottom)."""
    return cv2.filter2D(image.astype(np.float32), cv2.CV_32F, SOBEL_Y_KERNEL, borderType=border_type)
