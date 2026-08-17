"""KITTI per-frame timestamps, used for the Kalman filter's dt (Sec. III-B)."""

from pathlib import Path

import numpy as np


def load_timestamps(path) -> np.ndarray:
    """Seconds-of-day for each line of a KITTI timestamps.txt.

    Lines look like "2011-09-26 13:02:25.967790592" - 9-digit (nanosecond)
    fractional seconds, which exceeds what datetime.fromisoformat/strptime
    support (6-digit microsecond max), so this parses the time-of-day part
    manually instead of going through datetime.
    """
    seconds = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, time_part = line.split(" ", 1)
        hh, mm, ss = time_part.split(":")
        seconds.append(int(hh) * 3600 + int(mm) * 60 + float(ss))
    return np.array(seconds, dtype=np.float64)


def frame_dt(timestamps: np.ndarray, index: int) -> float:
    """Elapsed time between frame `index - 1` and frame `index`, in seconds."""
    return float(timestamps[index] - timestamps[index - 1])
