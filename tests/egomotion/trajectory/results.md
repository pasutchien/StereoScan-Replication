# Visual odometry trajectory vs. GPS/IMU

108 frames, `blob_max` class, total pipeline time 163.6s.

| trajectory | path length (m) | final-position error vs. GPS/IMU (m) |
|---|---|---|
| GPS/IMU | 106.97 | - |
| our method (raw) | 108.61 | 2.07 |
| our method (Kalman-filtered) | 108.61 | 2.07 |

See trajectory.png for the plotted paths.
