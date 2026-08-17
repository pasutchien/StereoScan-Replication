# Sec. III-B's Kalman filter (Eq. 3/4)

## Context

The egomotion pipeline (`calibration.py`, `reprojection.py`, `gauss_newton.py`,
`ransac.py`) already produces one `(r, t)` motion estimate per consecutive
frame pair via circular matching → outlier rejection → bucketing →
triangulation → RANSAC-wrapped Gauss-Newton. That estimate is per-frame-pair
only - it has no memory of recent motion, so it can be noisy/jittery from
frame to frame (each RANSAC run is independently subject to which 3-point
sample it drew, which correspondences survived bucketing, etc.).

Sec. III-B's Kalman filter sits on top of that as a second, temporal stage:
a constant-acceleration filter that treats each frame's raw `(r, t)` as a
noisy velocity measurement, blends it with a kinematic prediction built
from recent frames, and outputs a smoothed, more physically-plausible
motion estimate. `kalman.py` implements exactly that (Eq. 3 state
transition, Eq. 4 measurement model, Sec. IV's empirical noise constants) -
it does not change anything upstream.

## Design

**1. `timestamps.py`**

KITTI's `timestamps.txt` lines look like `2011-09-26 13:02:25.967790592` -
9-digit (nanosecond) fractional seconds, which exceeds what
`datetime.fromisoformat`/`strptime` support (6-digit microsecond max), so
parse manually rather than via `datetime`:
- `load_timestamps(path) -> np.ndarray[float]`: for each line, split off the
  time-of-day part, split on `:`, compute `hh*3600 + mm*60 + float(ss_frac)`
  as seconds-of-day. Confirmed against the actual file: consecutive frames
  are ~0.103s apart (not a flat 0.1s), so using real timestamps instead of
  an assumed fixed rate is a meaningful accuracy win here.
- `frame_dt(timestamps, index) -> float`: `timestamps[index] - timestamps[index-1]`.

**2. `kalman.py`**

Unlike the rest of `egomotion/`, which is all stateless pure functions, a
Kalman filter is inherently sequential/stateful, so this is a small class,
`EgomotionKalmanFilter`:

- state = `(v; a)` in R^12 (`v`: 3 rot-rate + 3 trans-rate, `a = d(v)/dt`)
- `Q = block_diag(process_noise_v * I6, process_noise_a * I6)` - the
  paper's flat `epsilon_1..6 ~ N(0, 1e-8 I)`, `epsilon_7..12 ~ N(0, I)`;
  `Q` is NOT dt-scaled, matching the paper's literal formulation
- `R = measurement_noise * I6`
- `predict(dt)`: `F = [[I6, dt*I6], [0, I6]]` (Eq. 3);
  `state = F @ state`, `covariance = F @ covariance @ F.T + Q`
- `update(r, t, dt)`: `z = concat(r, t) / dt` (Eq. 4: "we directly observe
  v"), `H = [I6, 0]` (6x12), standard KF innovation/gain/update
- `step(r, t, dt)`: `predict(dt)` then `update(r, t, dt)`; filtered
  `(r, t)` for this frame = `state[:6] * dt`; returns `(filtered_r, filtered_t)`

Defaults are exactly Sec. IV's empirical constants
(`nu ~ N(0, 10^-2 I)`, `epsilon_1..6 ~ N(0, 10^-8 I)`, `epsilon_7..12 ~ N(0, I)`),
so calling `step()` with no extra tuning reproduces the paper's
parameterization. Initial state = zeros (no prior motion assumed at the
very first frame); initial covariance = `1.0 * I12` (moderate starting
uncertainty) - the paper doesn't specify these, but they wash out quickly
given how confident the velocity process noise is.

No changes to `calibration.py`, `reprojection.py`, `gauss_newton.py`,
`ransac.py`, or `bucketing.py` - the filter only ever consumes their
`(r, t)` output as an external caller would.

## Verification

`tests/egomotion/kalman/test.py` (same non-pytest-collected, self-contained
script convention as `tests/egomotion/ransac/test.py`):

- Run the full pipeline across several consecutive real frame pairs from
  `2011_09_26_drive_0001_sync` (e.g. (0,1), (1,2), (2,3), (3,4), (4,5)),
  loading real per-frame timestamps for each pair's `dt`.
- Feed each pair's raw `(r, t)` through one `EgomotionKalmanFilter`
  instance sequentially, carrying state across the whole sequence.
- Write a `results.md` table: raw RANSAC+GN `(r, t)` vs. filtered `(r, t)`
  per frame pair, plus frame-to-frame variance of the dominant
  forward-translation component (z) for raw vs. filtered.
