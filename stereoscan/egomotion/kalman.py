"""Constant-acceleration Kalman filter over the RANSAC+GN motion estimate
(Geiger et al. 2011, Sec. III-B, Eq. 3/4).

Unlike the rest of egomotion/, which is stateless pure functions operating
on one frame pair at a time, a Kalman filter is inherently sequential: it
carries state across the whole frame sequence, smoothing each frame's raw
(r, t) estimate against a kinematic prediction built from recent motion.
"""

import numpy as np


class EgomotionKalmanFilter:
    """State = (v; a) in R^12: v is rotation-rate + translation-rate
    ((r, t) per second), a = d(v)/dt. Only v is ever directly measured
    (Eq. 4's "we directly observe v") - a is inferred purely from how v
    changes over time.
    """

    def __init__(
        self,
        process_noise_v: float = 1e-8,
        process_noise_a: float = 1.0,
        measurement_noise: float = 1e-2,
        initial_covariance: float = 1.0,
    ):
        self.state = np.zeros(12)
        self.covariance = initial_covariance * np.eye(12)
        # Sec. IV: epsilon_1..6 ~ N(0, 1e-8 I), epsilon_7..12 ~ N(0, I).
        # Flat, not dt-scaled - matches the paper's literal formulation.
        self.process_noise = np.diag(
            np.concatenate([np.full(6, process_noise_v), np.full(6, process_noise_a)])
        )
        self.measurement_noise = measurement_noise * np.eye(6)
        self._H = np.hstack([np.eye(6), np.zeros((6, 6))])

    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        """F = [[I6, dt*I6], [0, I6]] (Eq. 3)."""
        F = np.eye(12)
        F[:6, 6:] = dt * np.eye(6)
        return F

    def predict(self, dt: float) -> None:
        F = self._transition_matrix(dt)
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, r: np.ndarray, t: np.ndarray, dt: float) -> None:
        z = np.concatenate([r, t]) / dt  # Eq. 4: directly-observed velocity
        innovation = z - self._H @ self.state
        S = self._H @ self.covariance @ self._H.T + self.measurement_noise
        gain = self.covariance @ self._H.T @ np.linalg.inv(S)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(12) - gain @ self._H) @ self.covariance

    def step(self, r: np.ndarray, t: np.ndarray, dt: float):
        """Predict, then update with this frame's raw (r, t); returns the
        filtered (r, t) for this frame (= filtered velocity * dt)."""
        self.predict(dt)
        self.update(r, t, dt)
        velocity = self.state[:6]
        filtered = velocity * dt
        return filtered[:3], filtered[3:]
