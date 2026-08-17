"""Feature candidate detection: blob/corner filtering + 4-class NMS (Sec. III-A)."""

from dataclasses import dataclass

import numpy as np

from stereoscan.feature_matching.filters import filter_blob, filter_corner
from stereoscan.feature_matching.nms import non_maximum_suppression, non_minimum_suppression


@dataclass
class FeatureCandidates:
    """(x, y) pixel coordinates for each of the paper's four feature classes."""

    blob_max: np.ndarray
    blob_min: np.ndarray
    corner_max: np.ndarray
    corner_min: np.ndarray


def detect_features(
    image: np.ndarray,
    nms_radius: int = 3,
    nms_threshold: float = 50.0,
) -> FeatureCandidates:
    """Detect blob/corner max/min feature candidates in a grayscale image.

    Candidates are only ever matched within their own class (Sec. III-A),
    so the four classes are kept separate rather than merged.
    """
    blob = filter_blob(image)
    corner = filter_corner(image)

    return FeatureCandidates(
        blob_max=non_maximum_suppression(blob, nms_radius, nms_threshold),
        blob_min=non_minimum_suppression(blob, nms_radius, nms_threshold),
        corner_max=non_maximum_suppression(corner, nms_radius, nms_threshold),
        corner_min=non_minimum_suppression(corner, nms_radius, nms_threshold),
    )
