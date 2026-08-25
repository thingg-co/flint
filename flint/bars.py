"""Aggregate ticks into fixed-interval bars, aligned on one clock across all symbols."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .feed import Tick


@dataclass(slots=True)
class Bar:
    ts: float          # bar close time, unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trades: int = 0
    spread_bps: float = math.nan

    def to_json(self) -> dict:
        return {"t": self.ts, "o": self.open, "h": self.high, "l": self.low, "c": self.close, "v": self.volume}


class BarBuilder:
    def __init__(self, symbols: list[str], bar_seconds: float):
        self.symbols = symbols
        self.dt = bar_seconds
        self.last_close: dict[str, float] = {}
        self.open_bars: dict[int, dict[str, Bar]] = {}
        self.next_index: int | None = None

    def add(self, tick: Tick) -> None:
        idx = int(min(tick.ts, time.time() + 2.0) // self.dt)
        if self.next_index is not None and idx < self.next_index:
            idx = self.next_index  # late tick: fold into the oldest bar still open
        bars = self.open_bars.setdefault(idx, {})
        b = bars.get(tick.symbol)
        p = tick.price
        if tick.o is not None:                       # pre-aggregated OHLC bar from a history feed
            if b is None:
                b = Bar(ts=(idx + 1) * self.dt, open=tick.o, high=tick.h, low=tick.l, close=p)
                bars[tick.symbol] = b
            else:
                b.high = max(b.high, tick.h)
                b.low = min(b.low, tick.l)
                b.close = p
        elif b is None:
            b = Bar(ts=(idx + 1) * self.dt, open=p, high=p, low=p, close=p)
            bars[tick.symbol] = b
        else:
            b.high = max(b.high, p)
            b.low = min(b.low, p)
            b.close = p
        if tick.size > 0:
            b.volume += tick.size
            b.trades += 1
            if tick.taker_buy is True:
                b.buy_volume += tick.size
            elif tick.taker_buy is False:
                b.sell_volume += tick.size
        if tick.bid and tick.ask and tick.ask > tick.bid:
            b.spread_bps = (tick.ask - tick.bid) / (0.5 * (tick.ask + tick.bid)) * 1e4

    def roll(self, now: float, grace: float = 0.5) -> list[dict[str, Bar]]:
        """Close every bar whose interval ended before now - grace. Rows come back oldest first.

        A row is emitted only for intervals with real activity (at least one symbol
        traded), so closed-market gaps are skipped and the chart shows trading time;
        within an active interval, symbols that did not trade get a flat bar.
        """
        end_idx = int((now - grace) // self.dt)
        if self.next_index is None:
            if not self.open_bars:
                return []
            self.next_index = min(self.open_bars)
        rows: list[dict[str, Bar]] = []
        while self.next_index < end_idx:
            idx = self.next_index
            self.next_index += 1
            partial = self.open_bars.pop(idx, {})
            for s, b in partial.items():
                self.last_close[s] = b.close
            if not partial:
                continue          # no trades this interval (market closed / overnight) — skip the flat gap
            if any(s not in self.last_close for s in self.symbols):
                continue
            row: dict[str, Bar] = {}
            for s in self.symbols:
                b = partial.get(s)
                if b is None:
                    lp = self.last_close[s]
                    b = Bar(ts=(idx + 1) * self.dt, open=lp, high=lp, low=lp, close=lp)
                row[s] = b
            rows.append(row)
        return rows
