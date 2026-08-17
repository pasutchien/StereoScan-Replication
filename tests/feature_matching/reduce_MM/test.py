"""Compare naive full-MxM-window matching against the bucketed (coarse-pass
narrowed-window) matcher, sweeping min_bin_samples, and write the results
as a markdown table + conclusion.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate results.md in this folder:

    python tests/feature_matching/reduce_MM/test.py

Frame 1 is used as "current" (frame 0 as "previous") since frame 0 has no
predecessor to circle-match against, matching tests/feature_matching/matching/test.py.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.feature_matching.bucketing import circular_match_bucketed  # noqa: E402
from stereoscan.feature_matching.descriptor import DESCRIPTOR_MARGIN, compute_descriptors  # noqa: E402
from stereoscan.feature_matching.features import detect_features  # noqa: E402
from stereoscan.feature_matching.filters import sobel_x, sobel_y  # noqa: E402
from stereoscan.feature_matching.matching import circular_match  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "2011_09_26_drive_0001_sync"

CURR_FRAME = 1
PREV_FRAME = 0
FEATURE_CLASS = "blob_max"
FINE_RADIUS = 3
COARSE_RADIUS = FINE_RADIUS * 3
WINDOW_RADIUS = 50
EPIPOLAR_TOLERANCE = 1
BIN_SIZE = 50
MIN_BIN_SAMPLES_SWEEP = [1, 2, 3, 5, 8]


def load_image(frame_idx, camera):
    path = DATA_ROOT / f"image_{camera}" / "data" / f"{frame_idx:010d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {path}")
    return image


def prepare(image, nms_radius):
    """Detect + describe one feature class at a given NMS radius, dropping border candidates."""
    gx, gy = sobel_x(image), sobel_y(image)
    candidates = getattr(detect_features(image, nms_radius=nms_radius), FEATURE_CLASS)

    h, w = image.shape
    m = DESCRIPTOR_MARGIN
    keep = (candidates[:, 0] >= m) & (candidates[:, 0] < w - m) & (candidates[:, 1] >= m) & (candidates[:, 1] < h - m)
    points = candidates[keep]

    descriptors = compute_descriptors(gx, gy, points)
    return points, descriptors


def main():
    images = {
        "cl": load_image(CURR_FRAME, "00"),
        "pl": load_image(PREV_FRAME, "00"),
        "pr": load_image(PREV_FRAME, "01"),
        "cr": load_image(CURR_FRAME, "01"),
    }

    fine = {name: prepare(img, FINE_RADIUS) for name, img in images.items()}
    coarse = {name: prepare(img, COARSE_RADIUS) for name, img in images.items()}

    t0 = time.time()
    naive_matches = circular_match(
        *fine["cl"], *fine["pl"], *fine["pr"], *fine["cr"],
        window_radius=WINDOW_RADIUS, epipolar_tolerance=EPIPOLAR_TOLERANCE,
    )
    naive_time = time.time() - t0
    naive_set = set(naive_matches[:, 0].tolist())

    rows = [("naive (full MxM window)", len(naive_matches), naive_time, "-", "-")]

    for min_samples in MIN_BIN_SAMPLES_SWEEP:
        t0 = time.time()
        bucketed_matches, bin_stats = circular_match_bucketed(
            *coarse["cl"], *coarse["pl"], *coarse["pr"], *coarse["cr"],
            *fine["cl"], *fine["pl"], *fine["pr"], *fine["cr"],
            window_radius=WINDOW_RADIUS, epipolar_tolerance=EPIPOLAR_TOLERANCE,
            bin_size=BIN_SIZE, min_bin_samples=min_samples,
        )
        elapsed = time.time() - t0
        bucketed_set = set(bucketed_matches[:, 0].tolist())
        overlap = len(naive_set & bucketed_set)
        rows.append((f"bucketed, min_bin_samples={min_samples}", len(bucketed_matches), elapsed, len(bin_stats), f"{overlap}/{len(bucketed_matches)}"))

    lines = []
    lines.append("# MxM window reduction: naive vs. bucketed matching\n")
    lines.append(
        f"Frame {PREV_FRAME} -> {CURR_FRAME}, `{FEATURE_CLASS}` class, "
        f"window_radius={WINDOW_RADIUS}, bin_size={BIN_SIZE}, "
        f"fine NMS radius={FINE_RADIUS}, coarse NMS radius={COARSE_RADIUS}.\n"
    )
    lines.append("| approach | matches | time (s) | occupied bins | overlap w/ naive |")
    lines.append("|---|---|---|---|---|")
    for name, count, elapsed, bins, overlap in rows:
        lines.append(f"| {name} | {count} | {elapsed:.3f} | {bins} | {overlap} |")

    lines.append("\n## Conclusion\n")
    lines.append(
        "Even though the bucketed approach can speed up matching, the number of matches "
        "drops a lot without finding many new matches in return. We're sticking with the "
        "full MxM search window rather than the coarse-pass bucketed narrowing."
    )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote results to {OUTPUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
