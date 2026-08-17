"""Visualize blob/corner max/min feature candidates on a sample KITTI frame.

Not a pytest test (filename doesn't match pytest's test_*.py collection
pattern) - run it directly to (re)generate the images in this folder:

    python tests/feature_matching/feature_detection/test.py
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from stereoscan.feature_matching.features import detect_features  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "2011_09_26_drive_0001_sync" / "image_00" / "data" / "0000000000.png"

POINT_COLOR_BGR = (0, 0, 255)
POINT_RADIUS = 2


def draw_points(image, points):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), POINT_RADIUS, POINT_COLOR_BGR, -1)
    return vis


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read {IMAGE_PATH}")

    candidates = detect_features(image)

    cv2.imwrite(str(OUTPUT_DIR / "original.png"), image)
    print(f"original -> {OUTPUT_DIR / 'original.png'}")

    for name in ("blob_max", "blob_min", "corner_max", "corner_min"):
        points = getattr(candidates, name)
        vis = draw_points(image, points)
        out_path = OUTPUT_DIR / f"{name}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"{name}: {len(points)} points -> {out_path}")


if __name__ == "__main__":
    main()
