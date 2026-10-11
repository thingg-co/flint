"""Turn aligned bar rows into standardized model inputs."""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from .bars import Bar

log = logging.getLogger(__name__)

# Base market microstructure features, then exogenous signals appended (news, retail, derivatives, gurus).
# The exogenous slots are already bounded to [-1, 1] / [0, 1] by their providers, so they pass through
# standardization untouched. There is deliberately no clock feature.
EXOGENOUS = ["news_sent", "news_attn", "wsb_sent", "wsb_attn", "fng", "funding", "oi_chg", "longshort", "guru_net", "ethos_bias", "f_value", "f_quality", "breadth", "vix", "opt_iv", "opt_skew"]
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
        self.count = np.zeros(n_assets, dtype=np.int64)   # per name: one added by a universe change starts fresh while the rest keep their history

    def update(self, x: np.ndarray) -> None:
        fresh = self.count == 0                                           # a row's first sample is its mean
        a = np.maximum(self.alpha, 1.0 / (self.count + 1.0))[:, None]
        delta = x - self.mean
        self.mean = np.where(fresh[:, None], x, self.mean + a * delta)
        self.var = np.where(fresh[:, None], 1.0, (1 - a) * (self.var + a * delta * delta))
        self.count += 1

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        z = np.where(NORMALIZE[None, :], z, x)
        return np.clip(z, -5.0, 5.0).astype(np.float32)

    def state(self, symbols: list[str] | None = None) -> dict:
        st = {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count.copy()}
        if symbols is not None:
            st["symbols"] = list(symbols)                                   # names the rows, so a later universe can map them
        return st

    def load(self, st: dict, symbols: list[str] | None = None) -> bool:
        """Restore a saved state. Rows travel by name when both sides carry a symbol list, so a
        name the universe loop promoted or demoted keeps its own statistics or starts fresh; a
        state without names loads by position when the shape matches. Never raises: the start
        must not die over normalization statistics (found in Steel 2026-10-11, where the
        positional copy of an (old N, features) array into a (new N, features) normalizer died
        on a broadcast error after the first promotion), so an unmatchable state is logged and
        the normalizer stays fresh. Returns whether anything was restored."""
        n, f = self.mean.shape
        mean = np.asarray(st.get("mean"), dtype=np.float64)
        var = np.asarray(st.get("var"), dtype=np.float64)
        if mean.ndim != 2 or mean.shape != var.shape or mean.shape[1] != f:
            log.warning("normalization state of shape %s does not fit %d features; starting fresh", mean.shape, f)
            return False
        count = np.asarray(st.get("count", 0))
        if count.ndim == 0 or count.shape != (mean.shape[0],):             # pre-0.5.15: one count for every name
            count = np.full(mean.shape[0], int(count.flat[0]) if count.size else 0, dtype=np.int64)
        count = count.astype(np.int64)
        saved = [str(x) for x in (st.get("symbols") or [])]
        cur = [str(x) for x in symbols] if symbols is not None and len(symbols) == n else []
        if saved and cur and len(saved) == mean.shape[0]:
            pos = {s: i for i, s in enumerate(saved)}
            self.mean[:] = 0.0
            self.var[:] = 1.0
            self.count[:] = 0
            kept = 0
            for i, s in enumerate(cur):
                j = pos.get(s)
                if j is not None:
                    self.mean[i], self.var[i], self.count[i] = mean[j], var[j], count[j]
                    kept += 1
            if cur != saved:
                log.info("normalization state mapped onto the new universe by name: %d names kept, %d start fresh", kept, n - kept)
            return True
        if mean.shape[0] == n:
            self.mean[:] = mean
            self.var[:] = var
            self.count[:] = count
            return True
        log.warning("normalization state for %d names cannot be mapped onto %d without a symbol list; starting fresh", mean.shape[0], n)
        return False


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
