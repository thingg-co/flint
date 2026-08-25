"""Turn aligned bar rows into standardized model inputs."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .bars import Bar

# Base market microstructure features, then exogenous signals appended (news, retail, derivatives, gurus).
# The exogenous slots are already bounded to [-1, 1] / [0, 1] by their providers, so they pass through
# standardization untouched. There is deliberately no clock feature.
EXOGENOUS = ["news_sent", "news_attn", "wsb_sent", "wsb_attn", "fng", "funding", "oi_chg", "longshort", "guru_net", "ethos_bias", "f_value", "f_quality", "breadth", "vix"]
FEATURES = ["ret", "range", "logvol", "imbalance", "spread", "logtrades", "closepos", "mom", "rvol"] + EXOGENOUS
NORMALIZE = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1] + [0] * len(EXOGENOUS), dtype=bool)
N_FEATURES = len(FEATURES)


class RunningNorm:
    """Per-symbol, per-feature running mean and variance.

    Cumulative for the first samples, then exponentially weighted so the
    standardization tracks slow drifts in volatility and volume.
    """

    def __init__(self, n_assets: int, n_features: int, alpha: float = 0.003):
        self.alpha = alpha
        self.mean = np.zeros((n_assets, n_features), dtype=np.float64)
        self.var = np.ones((n_assets, n_features), dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        if self.count == 0:
            self.mean[:] = x
            self.count = 1
            return
        a = max(self.alpha, 1.0 / (self.count + 1))
        delta = x - self.mean
        self.mean += a * delta
        self.var = (1 - a) * (self.var + a * delta * delta)
        self.count += 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        z = np.where(NORMALIZE[None, :], z, x)
        return np.clip(z, -5.0, 5.0).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load(self, st: dict) -> None:
        self.mean[:] = st["mean"]
        self.var[:] = st["var"]
        self.count = int(st["count"])


class FeatureBuilder:
    def __init__(self, symbols: list[str], window: int, mom_bars: int = 12):
        self.symbols = symbols
        self.window_len = window
        self.norm = RunningNorm(len(symbols), N_FEATURES)
        self.prev_close: dict[str, float] = {}
        self.rets: dict[str, deque[float]] = {s: deque(maxlen=mom_bars) for s in symbols}
        self.window: deque[np.ndarray] = deque(maxlen=window)
        self.news: dict[str, tuple[float, float]] = {s: (0.0, 0.0) for s in symbols}
        # exogenous signal slots (everything after news), held constant between refreshes
        self.exo_keys = EXOGENOUS[2:]
        self.exo: dict[str, dict[str, float]] = {s: {k: 0.0 for k in self.exo_keys} for s in symbols}

    def set_news(self, symbol: str, sentiment: float, attention: float) -> None:
        """Exogenous news state, held constant until the next skim."""
        self.news[symbol] = (float(np.clip(sentiment, -1, 1)), float(np.clip(attention, 0, 1)))

    def set_exo(self, symbol: str, values: dict) -> None:
        """Exogenous signal vector (WSB, Fear&Greed, derivatives, guru 13Fs), held between refreshes."""
        cur = self.exo[symbol]
        for k in self.exo_keys:
            if k in values and values[k] is not None:
                cur[k] = float(np.clip(values[k], -1, 1))

    def push(self, row: dict[str, Bar]) -> np.ndarray | None:
        """Add one bar row. Returns the (assets, window, features) input once the window is full."""
        raw = np.zeros((len(self.symbols), N_FEATURES), dtype=np.float64)
        for i, s in enumerate(self.symbols):
            b = row[s]
            prev = self.prev_close.get(s, b.open)
            ret = math.log(b.close / prev) * 1e4 if prev > 0 else 0.0
            rng = math.log(b.high / b.low) * 1e4 if b.low > 0 else 0.0
            tot = b.buy_volume + b.sell_volume
            imb = (b.buy_volume - b.sell_volume) / tot if tot > 0 else 0.0
            spread = b.spread_bps if math.isfinite(b.spread_bps) else self.norm.mean[i, 4]
            span = b.high - b.low
            closepos = (b.close - b.low) / span - 0.5 if span > 0 else 0.0
            r = self.rets[s]
            r.append(ret)
            mom = float(sum(r))
            rvol = float(np.std(r)) if len(r) >= 2 else 0.0
            sent, attn = self.news[s]
            ex = self.exo[s]
            raw[i] = [ret, rng, math.log1p(b.volume), imb, spread, math.log1p(b.trades), closepos, mom, rvol,
                      sent, attn] + [ex[k] for k in self.exo_keys]
            self.prev_close[s] = b.close
        self.norm.update(raw)
        self.window.append(self.norm.apply(raw))
        if len(self.window) < self.window_len:
            return None
        return np.stack(self.window, axis=1)

    def peek_window(self) -> np.ndarray | None:
        """A predict-only window from whatever history exists, left-edge padded to
        window_len. Seeds forecasts before a full real window has accumulated; never
        used as a training sample, so training data stays real."""
        if not self.window:
            return None
        frames = list(self.window)
        if len(frames) < self.window_len:
            frames = [frames[0]] * (self.window_len - len(frames)) + frames
        return np.stack(frames, axis=1)
