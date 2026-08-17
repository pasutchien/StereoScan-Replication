"""KITTI OXTS (GPS/IMU) ground truth, converted to a local top-down trajectory
comparable to the camera-frame (x, z) trajectory from accumulate_trajectory.

The paper treats this as "weak" ground truth (Sec. IV-B: "localization errors
of up to two meters may occur"), and this conversion is itself an
approximation - it aligns the GPS track's initial heading with the camera's
initial forward (+z) axis, but doesn't chain the full IMU->Velodyne->camera
extrinsic calibration, so treat it as visually comparable, not millimeter-exact.
"""

from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6378137.0  # WGS84


def load_oxts(data_dir) -> np.ndarray:
    """(N, 6) array of (lat, lon, alt, roll, pitch, yaw) for each frame's
    oxts/data/NNNNNNNNNN.txt file, in frame order."""
    paths = sorted(Path(data_dir).glob("*.txt"))
    records = [np.loadtxt(p)[:6] for p in paths]
    return np.array(records)


def oxts_to_local_trajectory(records: np.ndarray) -> np.ndarray:
    """(N, 3) local (x, y, z) ground-truth positions in a frame-0-relative,
    camera-like convention: z = distance traveled along the frame-0 heading,
    x = lateral offset to the right of it, y = altitude change.

    Uses a Mercator projection (scaled by the first frame's latitude, same
    convention as the KITTI devkit) to turn lat/lon into local east/north
    meters, then rotates by the initial yaw so "forward" lines up with +z.
    """
    lat, lon, alt, _, _, yaw = records.T

    scale = np.cos(np.radians(lat[0]))
    east = scale * EARTH_RADIUS_M * np.radians(lon)
    north = scale * EARTH_RADIUS_M * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

    east -= east[0]
    north -= north[0]

    yaw0 = yaw[0]
    z = east * np.cos(yaw0) + north * np.sin(yaw0)
    x = east * np.sin(yaw0) - north * np.cos(yaw0)
    y = alt - alt[0]

    return np.column_stack([x, y, z])
