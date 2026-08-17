import numpy as np

from stereoscan.feature_matching.features import detect_features


def test_detect_features_returns_four_classes_with_valid_xy_coords():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)

    candidates = detect_features(image, nms_radius=3, nms_threshold=20.0)

    for coords in (
        candidates.blob_max,
        candidates.blob_min,
        candidates.corner_max,
        candidates.corner_min,
    ):
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        if len(coords):
            assert (coords[:, 0] >= 0).all() and (coords[:, 0] < image.shape[1]).all()
            assert (coords[:, 1] >= 0).all() and (coords[:, 1] < image.shape[0]).all()


def test_lower_threshold_finds_more_features():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)

    loose = detect_features(image, nms_radius=3, nms_threshold=5.0)
    strict = detect_features(image, nms_radius=3, nms_threshold=100.0)

    assert len(loose.blob_max) >= len(strict.blob_max)
    assert len(loose.corner_max) >= len(strict.corner_max)
