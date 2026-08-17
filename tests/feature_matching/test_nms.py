import numpy as np

from stereoscan.feature_matching.nms import non_maximum_suppression, non_minimum_suppression


def test_finds_single_isolated_peak():
    response = np.zeros((21, 21), dtype=np.float32)
    response[10, 15] = 100.0  # row=y=10, col=x=15

    peaks = non_maximum_suppression(response, radius=3, threshold=10.0)

    assert peaks.shape == (1, 2)
    assert tuple(peaks[0]) == (15, 10)  # (x, y)


def test_suppresses_weaker_nearby_peak():
    response = np.zeros((21, 21), dtype=np.float32)
    response[10, 10] = 100.0
    response[10, 12] = 80.0  # within radius=3 of the stronger peak

    peaks = non_maximum_suppression(response, radius=3, threshold=10.0)

    assert peaks.shape == (1, 2)
    assert tuple(peaks[0]) == (10, 10)


def test_keeps_two_peaks_far_apart():
    response = np.zeros((21, 21), dtype=np.float32)
    response[5, 5] = 100.0
    response[15, 15] = 90.0

    peaks = non_maximum_suppression(response, radius=3, threshold=10.0)

    coords = {tuple(p) for p in peaks}
    assert coords == {(5, 5), (15, 15)}


def test_threshold_rejects_weak_peak():
    response = np.zeros((21, 21), dtype=np.float32)
    response[10, 10] = 5.0

    peaks = non_maximum_suppression(response, radius=3, threshold=10.0)

    assert peaks.shape == (0, 2)


def test_border_peak_excluded():
    response = np.zeros((21, 21), dtype=np.float32)
    response[0, 0] = 100.0  # window near (0, 0) is never fully inside the image

    peaks = non_maximum_suppression(response, radius=3, threshold=10.0)

    assert peaks.shape == (0, 2)


def test_non_minimum_suppression_finds_dip():
    response = np.zeros((21, 21), dtype=np.float32)
    response[8, 12] = -100.0

    dips = non_minimum_suppression(response, radius=3, threshold=10.0)

    assert dips.shape == (1, 2)
    assert tuple(dips[0]) == (12, 8)
