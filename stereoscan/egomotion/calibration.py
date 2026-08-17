"""Stereo calibration parameters for Eq. 1, parsed from calib_cam_to_cam.txt."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StereoCalibration:
    focal_length: float
    cu: float
    cv: float
    baseline: float


def load_stereo_calibration(calib_path, left_cam: str = "00", right_cam: str = "01") -> StereoCalibration:
    """Parse P_rect_<left_cam>/P_rect_<right_cam> from a KITTI calib_cam_to_cam.txt.

    Eq. 1 shares one (f, cu, cv) between the left and right projections and
    instead shifts the 3D point by `s` (0 for left, baseline for right).
    P_rect_<right_cam>[0, 3] encodes that shift as -f * baseline, so
    baseline = -P_rect_<right_cam>[0, 3] / f.
    """
    matrices = {}
    for line in Path(calib_path).read_text().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key in (f"P_rect_{left_cam}", f"P_rect_{right_cam}"):
            matrices[key] = np.array([float(v) for v in rest.split()], dtype=np.float64).reshape(3, 4)

    p_left = matrices[f"P_rect_{left_cam}"]
    p_right = matrices[f"P_rect_{right_cam}"]

    focal_length = p_left[0, 0]
    cu = p_left[0, 2]
    cv = p_left[1, 2]
    baseline = -p_right[0, 3] / focal_length

    return StereoCalibration(focal_length=focal_length, cu=cu, cv=cv, baseline=baseline)
