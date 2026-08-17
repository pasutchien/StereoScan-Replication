# Plain Gauss-Newton vs. RANSAC-wrapped Gauss-Newton

Frame 0 -> 1, `blob_max` class, 341 correspondences after outlier rejection + bucketing, inlier threshold = 1.5px.

| approach | inliers @ 1.5px | RMS on those inliers | RMS on full set |
|---|---|---|---|
| plain GN | 22 / 341 (6.5%) | 1.12px | 11.74px |
| RANSAC + refined GN | 258 / 341 (75.7%) | 0.84px | 15.21px |

## Estimated motion

- **plain GN**: rotation (deg) = (-0.069, 0.323, -0.077), translation (m) = (-0.199, -0.035, -1.047)
- **RANSAC + refined GN**: rotation (deg) = (-0.042, -0.097, -0.039), translation (m) = (-0.005, -0.004, -1.384)

Plain GN's least-squares fit is dragged toward a compromise by leftover bad matches, so it barely satisfies the threshold for anyone. RANSAC finds a 3-point sample most of the data agrees with, then refines on that consensus set - producing a sub-pixel fit on a much larger inlier set, at the cost of a higher full-set RMS/max since it correctly stops averaging in the true outliers.
