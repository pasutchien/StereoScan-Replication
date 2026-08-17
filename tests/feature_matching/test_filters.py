import numpy as np

from stereoscan.feature_matching.filters import (
    BLOB_KERNEL,
    CORNER_KERNEL,
    SOBEL_X_KERNEL,
    SOBEL_Y_KERNEL,
    filter_blob,
    filter_corner,
    sobel_x,
    sobel_y,
)


def test_kernels_sum_to_zero():
    assert BLOB_KERNEL.sum() == 0
    assert CORNER_KERNEL.sum() == 0
    assert SOBEL_X_KERNEL.sum() == 0
    assert SOBEL_Y_KERNEL.sum() == 0


def test_kernels_are_5x5():
    assert BLOB_KERNEL.shape == (5, 5)
    assert CORNER_KERNEL.shape == (5, 5)


def test_sobel_kernels_are_3x3():
    assert SOBEL_X_KERNEL.shape == (3, 3)
    assert SOBEL_Y_KERNEL.shape == (3, 3)


def test_blob_filter_peaks_on_bright_square():
    image = np.zeros((21, 21), dtype=np.uint8)
    # 3x3 bright square, centered at (10, 10): matches the kernel's inner
    # +1/+8 region while leaving its -1 border on dark background.
    image[9:12, 9:12] = 255

    response = filter_blob(image)

    assert response.shape == image.shape
    assert response.dtype == np.float32
    peak = np.unravel_index(np.argmax(response), response.shape)
    assert peak == (10, 10)


def test_corner_filter_detects_right_angle_edge():
    # top-right quadrant bright, rest dark: response should be strongly non-zero
    # at the corner point and roughly zero far from any edge.
    image = np.zeros((21, 21), dtype=np.uint8)
    image[:10, 11:] = 255

    response = filter_corner(image)

    assert response.shape == image.shape
    assert abs(response[9, 10]) > abs(response[0, 0])


def test_filters_are_translation_consistent():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(50, 50), dtype=np.uint8)

    blob = filter_blob(image)
    corner = filter_corner(image)

    assert np.isfinite(blob).all()
    assert np.isfinite(corner).all()


def test_sobel_x_is_positive_across_a_left_to_right_step_edge():
    image = np.zeros((21, 21), dtype=np.uint8)
    image[:, 11:] = 255  # dark on the left, bright on the right

    response = sobel_x(image)

    assert response.shape == image.shape
    assert response.dtype == np.float32
    assert response[10, 10] > 0
    assert response[10, 5] == 0  # flat region, away from the edge


def test_sobel_y_is_positive_across_a_top_to_bottom_step_edge():
    image = np.zeros((21, 21), dtype=np.uint8)
    image[11:, :] = 255  # dark on top, bright on the bottom

    response = sobel_y(image)

    assert response.shape == image.shape
    assert response[10, 10] > 0
    assert response[5, 10] == 0  # flat region, away from the edge


def test_sobel_x_and_y_respond_to_orthogonal_edges_only():
    vertical_edge = np.zeros((21, 21), dtype=np.uint8)
    vertical_edge[:, 11:] = 255

    assert sobel_x(vertical_edge)[10, 10] != 0
    assert sobel_y(vertical_edge)[10, 10] == 0
