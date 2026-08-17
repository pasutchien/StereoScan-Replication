# Raw RANSAC+GN vs. Kalman-filtered egomotion

`blob_max` class, window_radius=50, frame pairs: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], real per-frame dt from timestamps.txt.

| pair | dt (s) | raw t_z (m) | filtered t_z (m) | raw rotation (deg) | filtered rotation (deg) |
|---|---|---|---|---|---|
| (0,1) | 0.1031 | -1.384 | -1.371 | (-0.042, -0.097, -0.039) | (-0.042, -0.096, -0.039) |
| (1,2) | 0.1034 | -1.368 | -1.373 | (-0.108, -0.130, 0.023) | (-0.093, -0.122, 0.008) |
| (2,3) | 0.1027 | -1.351 | -1.355 | (-0.070, -0.149, 0.023) | (-0.081, -0.147, 0.026) |
| (3,4) | 0.1031 | -1.355 | -1.355 | (0.055, -0.127, 0.031) | (0.025, -0.136, 0.035) |
| (4,5) | 0.1030 | -1.366 | -1.362 | (0.119, -0.161, 0.007) | (0.113, -0.156, 0.016) |

## Smoothing effect (forward translation t_z)

- raw t_z std dev: 0.0115 m
- filtered t_z std dev: 0.0077 m
