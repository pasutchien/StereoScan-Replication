"""Bucketing for egomotion estimation (Geiger et al. 2011, Sec. III-B).

"First, bucketing is used to reduce the number of features (in practice we
retain between 200 and 500 features) and spread them uniformly over the
image domain."

This is a different operation from stereoscan.feature_matching.bucketing:
that one narrows a *search window* during matching using per-bin min/max
displacement statistics from a coarse pass. This one runs after matching
(and outlier rejection) are done, and subsamples the already-accepted
matches down to a smaller, spatially-uniform set for the reprojection-error
minimization (Eq. 2) that follows. They share the word "bucketing" and the
grid-binning mechanic, not the purpose.
"""

import numpy as np

from stereoscan.feature_matching.bucketing import bin_index


def bucket_matches(curr_left_points: np.ndarray, bin_size: int = 50, max_per_bucket: int = 3, seed: int = 0) -> np.ndarray:
    """Boolean mask keeping at most `max_per_bucket` matches per grid cell.

    Caps dense clusters (e.g. foliage) so the retained set spreads evenly
    over the image domain instead of being dominated by one region - which
    would otherwise leave the egomotion optimization poorly conditioned for
    some of the 6 motion parameters. Cells with <= max_per_bucket matches
    keep all of them; over-full cells are subsampled uniformly at random
    (seeded for reproducibility - the paper doesn't specify a quality-based
    selection criterion, so this is the simplest defensible default).
    """
    n = len(curr_left_points)
    bins = bin_index(curr_left_points, bin_size)
    rng = np.random.default_rng(seed)

    keep = np.zeros(n, dtype=bool)
    for bx, by in {tuple(b) for b in bins}:
        bucket_idx = np.nonzero((bins[:, 0] == bx) & (bins[:, 1] == by))[0]
        if len(bucket_idx) > max_per_bucket:
            bucket_idx = rng.choice(bucket_idx, size=max_per_bucket, replace=False)
        keep[bucket_idx] = True
    return keep
