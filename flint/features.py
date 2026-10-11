"""Turn aligned bar rows into standardized model inputs.

Everything the net sees is computable from the bar stream itself plus dated news, so paged
history and the live feed produce the same numbers (2a, 2026-10-12): before, the net trained on
sixteen exogenous slots that paged history could not carry and met nonzero values only live,
and three bar features (bid/ask imbalance, the quoted spread, the trade count) that history
candles do not have. Those live-only signals now stay in the policy and the dashboard.

Per name, from its own bars: the bar's return, range, log volume and close position; the hour's
momentum and realised vol; Garman-Klass (this bar) and Yang-Zhang (the hour) volatility; the
day's and the week's realised vol (HAR terms) and the day's return; the same-slot return one day
back and its five-day mean; Amihud illiquidity and the Roll spread over the day; the overnight
and the intraday legs; the time of day. Across names: breadth this bar and over the hour, the
market's return, momentum and vol (the first index name when modeled, else the equal-weight
universe), each name's rolling beta to it, and the market's first half hour. Then the two news
slots, from the live skim or, for history, a dated store (`NewsHistory`). There is no day-of-week
or calendar-date feature on purpose.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .bars import Bar

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
BARS_H, BARS_D, BARS_W = 12, 78, 390                 # an hour, a session, a week of 5-minute bars
RING = BARS_W + 1
OPEN_BARS = 6                                        # the first half hour
BETA_SHRINK = 20.0                                   # pseudo-bars behind a beta of 1

OWN = ["ret", "range", "logvol", "closepos", "mom", "rvol", "gk", "yz", "rv_d", "rv_w", "mom_d", "lag_d", "slot5",
       "amihud", "roll", "overnight", "intraday", "tod_sin", "tod_cos"]
MARKET = ["breadth", "breadth_h", "mkt_ret", "mkt_mom", "mkt_rvol", "beta", "open30"]
NEWS = ["news_sent", "news_attn"]
FEATURES = OWN + MARKET + NEWS
EXOGENOUS = NEWS                                     # the only exogenous slots left in the net
_BOUNDED = {"closepos", "tod_sin", "tod_cos", "breadth", "breadth_h", "news_sent", "news_attn"}   # already on a fixed scale
NORMALIZE = np.array([f not in _BOUNDED for f in FEATURES], dtype=bool)
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


def _safe_log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ln(a / b) where both are positive, else 0."""
    ok = (a > 0) & (b > 0)
    out = np.zeros(len(a))
    out[ok] = np.log(a[ok] / b[ok])
    return out


def session_minutes(ts: float) -> tuple[int, float]:
    """(the ET date's ordinal, minutes since 09:30 ET) for a bar's close time."""
    d = datetime.fromtimestamp(ts, NY)
    return d.date().toordinal(), (d.hour * 60 + d.minute - 570) + d.second / 60.0


class FeatureBuilder:
    """One bar row in, one standardized (assets, features) row out, and the (assets, window,
    features) input once `window` rows have accumulated. Holds a week of returns per name in a
    ring so the longer features are exact once a week of bars has passed; before that they are
    computed on what there is (an offline chunk warms up on a prefix it does not emit)."""

    def __init__(self, symbols: list[str], window: int, mom_bars: int = BARS_H, index: str | None = None):
        self.symbols = list(symbols)
        n = len(self.symbols)
        self.window_len = window
        self.mom_bars = int(mom_bars)
        self.norm = RunningNorm(n, N_FEATURES)
        self.index = self.symbols.index(index) if index in self.symbols else None
        self.prev_close = np.full(n, np.nan)
        self.R = np.zeros((n, RING))                 # returns, bps
        self.A = np.zeros((n, RING))                 # |return| per $M traded (Amihud terms)
        self.M = np.zeros(RING)                      # the market's return, bps
        self.O = np.zeros((n, BARS_H)); self.H = np.zeros((n, BARS_H)); self.L = np.zeros((n, BARS_H)); self.C = np.zeros((n, BARS_H))
        self.PC = np.zeros((n, BARS_H))              # the previous close behind each of the hour's bars (Yang-Zhang's overnight leg)
        self.p = 0                                   # next ring slot
        self.n_bars = 0
        self.day: int | None = None
        self.day_open = np.full(n, np.nan)
        self.prev_day_close = np.full(n, np.nan)
        self.bars_today = 0
        self.open30 = 0.0
        self.mkt_day_open = math.nan
        self.window: deque[np.ndarray] = deque(maxlen=window)
        self.news_sent = np.zeros(n)
        self.news_attn = np.zeros(n)
        self.news_history = None                     # a `NewsHistory`: when set, the news slots come from it at the bar's time
        self.exo: dict[str, dict[str, float]] = {s: {} for s in self.symbols}   # live-only signals, kept for the policy and the dashboard, not the net

    # -- exogenous inputs ----------------------------------------------------------------
    def set_news(self, symbol: str, sentiment: float, attention: float) -> None:
        """News state from the live skim, held constant until the next one."""
        if symbol in self.symbols:
            i = self.symbols.index(symbol)
            self.news_sent[i] = float(np.clip(sentiment, -1, 1))
            self.news_attn[i] = float(np.clip(attention, 0, 1))

    @property
    def news(self) -> dict[str, tuple[float, float]]:
        return {s: (float(self.news_sent[i]), float(self.news_attn[i])) for i, s in enumerate(self.symbols)}

    def set_exo(self, symbol: str, values: dict) -> None:
        """Live-only signals (WSB, Fear & Greed, derivatives, guru 13Fs, options): kept for the
        policy and the dashboard; since 2a they are not model inputs."""
        if symbol in self.exo:
            for k, v in values.items():
                if v is not None:
                    self.exo[symbol][k] = float(np.clip(v, -1, 1))

    # -- the ring --------------------------------------------------------------------------
    def _last(self, ring: np.ndarray, m: int) -> np.ndarray:
        """The newest min(m, n_bars) entries of a (..., RING) ring, oldest first."""
        k = min(m, self.n_bars)
        idx = (self.p - k + np.arange(k)) % RING
        return ring[..., idx]

    def _lag(self, ring: np.ndarray, lag: int) -> np.ndarray:
        """The entry `lag` bars before the newest (lag 1 = the bar before this one), or zeros."""
        if self.n_bars <= lag:
            return np.zeros(ring.shape[:-1])
        return ring[..., (self.p - 1 - lag) % RING]

    def push(self, row: dict[str, Bar]) -> np.ndarray | None:
        """Add one bar row. Returns the (assets, window, features) input once the window is full."""
        n = len(self.symbols)
        o = np.array([row[s].open for s in self.symbols], dtype=np.float64)
        h = np.array([row[s].high for s in self.symbols], dtype=np.float64)
        lo = np.array([row[s].low for s in self.symbols], dtype=np.float64)
        c = np.array([row[s].close for s in self.symbols], dtype=np.float64)
        v = np.array([row[s].volume for s in self.symbols], dtype=np.float64)
        ts = float(row[self.symbols[0]].ts)
        prev = np.where(np.isnan(self.prev_close), o, self.prev_close)
        ret = _safe_log_ratio(c, prev) * 1e4
        rng = np.where((h > 0) & (lo > 0) & (h >= lo), np.log(np.maximum(h, 1e-12) / np.maximum(lo, 1e-12)), 0.0) * 1e4
        span = h - lo
        closepos = np.where(span > 0, (c - lo) / np.where(span > 0, span, 1.0) - 0.5, 0.0)
        active = v > 0
        # the session clock and the day's legs
        day, minutes = session_minutes(ts)
        if day != self.day:
            self.prev_day_close = self.prev_close.copy()
            self.day_open = np.where(o > 0, o, np.nan)
            self.day = day
            self.bars_today = 0
            self.open30 = 0.0
            self.mkt_day_open = math.nan
        self.bars_today += 1
        pdc = np.where(np.isnan(self.prev_day_close), 0.0, self.prev_day_close)
        dop = np.where(np.isnan(self.day_open), 0.0, self.day_open)
        overnight = _safe_log_ratio(dop, pdc) * 1e4
        intraday = _safe_log_ratio(c, dop) * 1e4
        frac = min(max(minutes, -30.0), 420.0) / 390.0
        tod_sin, tod_cos = math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)
        # the market: the index name when modeled, else the equal-weight universe of real bars
        if self.index is not None:
            mkt_ret = float(ret[self.index])
        else:
            mkt_ret = float(ret[active].mean()) if active.any() else 0.0
        # rings
        p = self.p
        self.R[:, p] = ret
        dv = c * v
        self.A[:, p] = np.where(dv > 0, np.abs(ret) / np.maximum(dv * 1e-6, 1e-9), 0.0)
        self.M[p] = mkt_ret
        q = self.n_bars % BARS_H
        self.O[:, q], self.H[:, q], self.L[:, q], self.C[:, q], self.PC[:, q] = o, h, lo, c, prev
        self.n_bars += 1
        self.p = (p + 1) % RING
        # own-bar rolling features
        rh = self._last(self.R, self.mom_bars)
        mom = rh.sum(axis=1)
        rvol = rh.std(axis=1) if rh.shape[1] >= 2 else np.zeros(n)
        rd = self._last(self.R, BARS_D)
        rw = self._last(self.R, BARS_W)
        rv_d = rd.std(axis=1) if rd.shape[1] >= 2 else np.zeros(n)
        rv_w = rw.std(axis=1) if rw.shape[1] >= 2 else np.zeros(n)
        mom_d = rd.sum(axis=1)
        lag_d = self._lag(self.R, BARS_D)
        lags = [self._lag(self.R, k * BARS_D) for k in range(1, 6) if self.n_bars > k * BARS_D]
        slot5 = np.mean(lags, axis=0) if lags else np.zeros(n)
        amihud = np.log1p(self._last(self.A, BARS_D).mean(axis=1))
        if rd.shape[1] >= 3:
            x1, x0 = rd[:, 1:], rd[:, :-1]
            cov = (x1 * x0).mean(axis=1) - x1.mean(axis=1) * x0.mean(axis=1)
            roll = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
        else:
            roll = np.zeros(n)
        gk = np.sqrt(np.maximum(0.5 * _safe_log_ratio(h, lo) ** 2 - (2 * math.log(2) - 1) * _safe_log_ratio(c, o) ** 2, 0.0)) * 1e4
        yz = self._yang_zhang()
        # across names
        breadth = (float(np.mean(ret[active] > 0)) - 0.5) * 2 if active.any() else 0.0
        breadth_h = (float(np.mean(mom[active] > 0)) - 0.5) * 2 if active.any() else 0.0
        mh = self._last(self.M, self.mom_bars)
        mkt_mom = float(mh.sum())
        mkt_rvol = float(rvol[active].mean()) if active.any() else 0.0
        md = self._last(self.M, BARS_D)
        k = md.shape[0]
        if k >= 3:
            mx = md - md.mean()
            vx = float((mx * mx).mean())
            cv = ((rd - rd.mean(axis=1, keepdims=True)) * mx[None, :]).mean(axis=1)
            braw = cv / vx if vx > 0 else np.ones(n)
            beta = (k * braw + BETA_SHRINK) / (k + BETA_SHRINK)
        else:
            beta = np.ones(n)
        if self.bars_today == 1:
            self.mkt_day_open = float(o[self.index]) if self.index is not None else (float(o[active].mean()) if active.any() else math.nan)
        if self.bars_today == OPEN_BARS:
            mkt_c = float(c[self.index]) if self.index is not None else (float(c[active].mean()) if active.any() else math.nan)
            self.open30 = math.log(mkt_c / self.mkt_day_open) * 1e4 if mkt_c > 0 and self.mkt_day_open > 0 else 0.0
        # news: the dated store for history, the live skim otherwise
        if self.news_history is not None:
            sent, attn = self.news_history.at(ts, self.symbols)
        else:
            sent, attn = self.news_sent, self.news_attn
        raw = np.column_stack([ret, rng, np.log1p(v), closepos, mom, rvol, gk, yz, rv_d, rv_w, mom_d, lag_d, slot5,
                               amihud, roll, overnight, intraday, np.full(n, tod_sin), np.full(n, tod_cos),
                               np.full(n, breadth), np.full(n, breadth_h), np.full(n, mkt_ret), np.full(n, mkt_mom),
                               np.full(n, mkt_rvol), beta, np.full(n, self.open30), sent, attn])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_close = np.where(c > 0, c, self.prev_close)
        self.norm.update(raw)
        self.window.append(self.norm.apply(raw))
        if len(self.window) < self.window_len:
            return None
        return np.stack(self.window, axis=1)

    def _yang_zhang(self) -> np.ndarray:
        """Yang-Zhang volatility over the hour's bars, in bps: the overnight (close-to-open)
        variance, the open-to-close variance and the Rogers-Satchell term, weighted so the
        estimate is unbiased whatever the drift."""
        m = min(self.n_bars, BARS_H)
        n = self.O.shape[0]
        if m < 2:
            return np.zeros(n)
        cols = (self.n_bars - m + np.arange(m)) % BARS_H
        o, h, lo, c, pc = self.O[:, cols], self.H[:, cols], self.L[:, cols], self.C[:, cols], self.PC[:, cols]
        ok = (o > 0) & (h > 0) & (lo > 0) & (c > 0) & (pc > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(ok, np.log(o / pc), 0.0)
            r = np.where(ok, np.log(c / o), 0.0)
            rs = np.where(ok, np.log(h / o) * np.log(h / c) + np.log(lo / o) * np.log(lo / c), 0.0)
        kf = 0.34 / (1.34 + (m + 1) / (m - 1))
        var = u.var(axis=1) + kf * r.var(axis=1) + (1 - kf) * rs.mean(axis=1)
        return np.sqrt(np.maximum(np.nan_to_num(var), 0.0)) * 1e4

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
