"""Meta-labeling: does acting on this forecast pay after cost?

An online logistic model over six numbers the forecast itself gives (its score, its conviction,
the size of its median, its width, the agreement of its two direction heads and its tails next to
the median), fitted on matured forecasts' outcomes: 1 when the move in the called direction beat
a round trip of `cost_bps`. It is a second opinion on the net's own forecast, so the net is
untouched and the features are per forecast, not per name: the model survives a universe change
as it is.

Walk-forward over the 2026-10-11 relative run's stored forecasts (refit every five days on the
days before) it was calibrated to a few points, a bucket's claim matching its realised rate from
0.33 to 0.68, and in its top decile a forecast netted +5.6 bps after cost against -3.1 for the
rest. As a gate at even odds over the sweep's proposed gates it lifted the confirming half from
+62 to +84 bps and Sharpe 4.8 to 6.2 on a third fewer trades. The band-implied P(move clears
the cost) was tried first and is not a probability: at a claimed 0.7 the move cleared 49% of the
time, because the band calibration widens the quantiles around a median it leaves alone.

The same class runs live (`Engine._suggest` asks it, `Engine._score` teaches it), in the backtest
(`backtest.decide` asks it, the resolve one horizon later teaches it, and each forecast's answer
is stored with the forecasts) and in the replay (`sweep.replay_policy` reads the stored answers,
so `meta_min` can be swept like any gate). Pure numpy.
"""
from __future__ import annotations

import math

import numpy as np

K = 6                                   # features per forecast
NAMES = ("|score|", "conviction", "log|q50|", "log IQR", "heads agree", "tails/|q50|")


def features(q, p_up, p_down) -> np.ndarray:
    """(..., 6) from the raw quantiles (..., 5) and the raw P(up) and P(down): |q50/IQR|,
    |2 P(up) - 1|, log1p |q50|, log1p IQR, P(up) - P(down) in the median's direction, and the
    10-90 width over max(|q50|, 1)."""
    q = np.asarray(q, dtype=np.float64)
    pu = np.asarray(p_up, dtype=np.float64)
    pd_ = np.asarray(p_down, dtype=np.float64)
    q50 = q[..., 2]
    iqr = np.maximum(q[..., 3] - q[..., 1], 1e-3)
    sgn = np.sign(q50)
    return np.stack([np.abs(q50 / iqr), np.abs(2.0 * pu - 1.0), np.log1p(np.abs(q50)), np.log1p(iqr),
                     (pu - pd_) * sgn, (q[..., -1] - q[..., 0]) / np.maximum(np.abs(q50), 1.0)], axis=-1)


def label(y, q50, cost_bps: float) -> np.ndarray:
    """1 where the realised move in the median's direction beat the round trip."""
    return (np.asarray(y, dtype=np.float64) * np.sign(np.asarray(q50, dtype=np.float64)) > float(cost_bps)).astype(np.float32)


class MetaLabeler:
    """A ridge logistic regression on standardized features, refit every `refit_every` new rows
    over a ring of the last `cap` matured forecasts; silent (`predict` is None) until `min_rows`
    rows have matured, so an unfitted model gates nothing."""

    def __init__(self, cap: int = 400_000, min_rows: int = 2000, refit_every: int = 5000, ridge: float = 1e-2):
        self.cap, self.min_rows, self.refit_every, self.ridge = int(cap), int(min_rows), int(refit_every), float(ridge)
        self.x = np.zeros((0, K), dtype=np.float32)
        self.t = np.zeros(0, dtype=np.float32)
        self.w: np.ndarray | None = None                  # (K + 1,): bias first, on standardized features
        self.mu = np.zeros(K)
        self.sd = np.ones(K)
        self.n = 0                                        # rows ever seen
        self.since = 0                                    # rows since the last fit
        self.fits = 0

    @property
    def fitted(self) -> bool:
        return self.w is not None

    def update(self, x, t) -> None:
        """Matured forecasts: `x` (m, 6), `t` (m,) in {0, 1}. Refits when due."""
        x = np.asarray(x, dtype=np.float32).reshape(-1, K)
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        keep = np.isfinite(x).all(axis=1) & np.isfinite(t)
        if not keep.all():
            x, t = x[keep], t[keep]
        if not len(t):
            return
        self.x = np.concatenate([self.x, x])[-self.cap:]
        self.t = np.concatenate([self.t, t])[-self.cap:]
        self.n += len(t)
        self.since += len(t)
        if len(self.t) >= self.min_rows and (self.w is None or self.since >= self.refit_every):
            self.refit()

    def refit(self) -> None:
        x = self.x.astype(np.float64)
        t = self.t.astype(np.float64)
        if len(t) < max(K + 2, 10):
            return
        self.mu = x.mean(axis=0)
        self.sd = np.maximum(x.std(axis=0), 1e-6)
        xs = np.concatenate([np.ones((len(x), 1)), (x - self.mu) / self.sd], axis=1)
        w = np.zeros(K + 1)
        rate = float(np.clip(t.mean(), 1e-3, 1 - 1e-3))
        w[0] = math.log(rate / (1 - rate))
        eye = np.eye(K + 1) * self.ridge
        eye[0, 0] = 0.0                                                   # the bias is free
        for _ in range(10):
            p = 1.0 / (1.0 + np.exp(-(xs @ w)))
            g = xs.T @ (t - p) - eye @ w
            h = xs.T @ (xs * (p * (1 - p))[:, None]) + eye + 1e-9 * np.eye(K + 1)
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            w += step
            if float(np.abs(step).max()) < 1e-6:
                break
        if np.isfinite(w).all():
            self.w = w
            self.fits += 1
        self.since = 0

    def predict(self, x) -> np.ndarray | None:
        """P(the move pays) per row of `x` (m, 6), or None until fitted."""
        if self.w is None:
            return None
        x = np.asarray(x, dtype=np.float64).reshape(-1, K)
        z = self.w[0] + ((x - self.mu) / self.sd) @ self.w[1:]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def state(self) -> dict:
        return {"v": 1, "w": None if self.w is None else self.w.astype(np.float64).tolist(),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "n": int(self.n), "since": int(self.since),
                "fits": int(self.fits), "x": self.x.copy(), "t": self.t.copy()}

    def load(self, st) -> bool:
        """Restore; False (and untouched) when the state is not this model's shape."""
        try:
            st = dict(st)
            x = np.asarray(st.get("x", np.zeros((0, K))), dtype=np.float32).reshape(-1, K)
            t = np.asarray(st.get("t", np.zeros(0)), dtype=np.float32).reshape(-1)
            if len(x) != len(t):
                return False
            w = st.get("w")
            if w is not None:
                w = np.asarray(w, dtype=np.float64).reshape(-1)
                if w.shape != (K + 1,) or not np.isfinite(w).all():
                    return False
            mu = np.asarray(st.get("mu", np.zeros(K)), dtype=np.float64).reshape(-1)
            sd = np.asarray(st.get("sd", np.ones(K)), dtype=np.float64).reshape(-1)
            if mu.shape != (K,) or sd.shape != (K,):
                return False
        except Exception:  # noqa: BLE001
            return False
        self.x, self.t, self.w, self.mu, self.sd = x[-self.cap:], t[-self.cap:], w, mu, np.maximum(sd, 1e-6)
        self.n, self.since, self.fits = int(st.get("n", len(t))), int(st.get("since", 0)), int(st.get("fits", 0))
        return True
