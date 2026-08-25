"""Core tick type and the offline simulator.

Live and REST sources live in sources.py; this module holds the shared Tick
dataclass and the synthetic SimFeed used as an offline fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import httpx
import websockets

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Tick:
    symbol: str
    ts: float                  # unix seconds (exchange time when available)
    price: float
    size: float                # base units traded in this event; 0 for quote-only updates
    taker_buy: bool | None     # aggressor side; None when unknown
    bid: float | None = None
    ask: float | None = None
    o: float | None = None   # set when this tick is a pre-aggregated OHLC bar (history feeds)
    h: float | None = None
    l: float | None = None


def _parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class SimFeed:
    """Synthetic market used when there is no network.

    A hidden regime chain drives the lead asset (trend up, trend down, mean
    reverting, choppy), the other assets follow it with a lag, and taker order
    flow leaks a little of the next move. That gives the model real structure to
    discover offline without pretending to be a market.
    """

    name = "sim"
    BASE = {"BTC-USD": 60000.0, "ETH-USD": 3000.0, "SOL-USD": 150.0, "XRP-USD": 0.6}

    def __init__(self, symbols: list[str], seed: int | None = None, step: float = 0.25):
        self.symbols = symbols
        self.rng = random.Random(seed)
        self.step = step
        self.prices = {s: self.BASE.get(s, 10.0) for s in symbols}
        self.slow = dict(self.prices)
        self.vol = {s: 0.5 * (1.0 if i == 0 else 1.3) for i, s in enumerate(symbols)}  # bps per 0.25s step
        self.regime = 0
        self.t = time.time()
        self.lead_hist: deque[float] = deque([0.0] * 4, maxlen=4)
        self.next_r = {s: 0.0 for s in symbols}

    def _draw(self) -> dict[str, float]:
        lead = self.symbols[0]
        drift = {0: 0.035, 1: -0.035, 2: 0.0, 3: 0.0}[self.regime]
        out: dict[str, float] = {}
        for i, s in enumerate(self.symbols):
            v = self.vol[s]
            if i == 0:
                r = drift + v * self.rng.gauss(0, 1)
                if self.regime == 2:
                    r -= 0.01 * math.log(self.prices[s] / self.slow[s]) * 1e4
            else:
                lagged = self.lead_hist[-2] if len(self.lead_hist) >= 2 else 0.0
                r = 0.5 * lagged + 0.5 * drift + 0.85 * v * self.rng.gauss(0, 1)
            out[s] = r
        return out

    def _advance(self) -> list[Tick]:
        if self.rng.random() < self.step / 240.0:
            self.regime = self.rng.randrange(4)
        cur = self.next_r
        self.next_r = self._draw()
        self.lead_hist.append(cur[self.symbols[0]])
        ticks: list[Tick] = []
        for s in self.symbols:
            r = cur[s]
            p0 = self.prices[s]
            p1 = p0 * math.exp(r / 1e4)
            self.prices[s] = p1
            self.slow[s] += 0.002 * (p1 - self.slow[s])
            # Order flow: the taker side is tilted toward the *next* step's move.
            tilt = 1.0 / (1.0 + math.exp(-0.6 * self.next_r[s] / self.vol[s]))
            n = 1 + (self.rng.random() < 0.6) + (self.rng.random() < 0.3)
            half = 1.2e-4 * (0.7 + 0.6 * self.rng.random())
            for k in range(n):
                f = (k + 1) / n
                price = p0 + (p1 - p0) * f
                size = self.rng.lognormvariate(-3.0, 1.0) * (60000.0 / self.BASE.get(s, 10.0)) ** 0.7
                ticks.append(Tick(s, self.t + self.step * f, price, size, self.rng.random() < tilt,
                                  price * (1 - half), price * (1 + half)))
        self.t += self.step
        return ticks

    async def backfill(self, seconds: float) -> list[Tick]:
        self.t = time.time() - seconds
        ticks: list[Tick] = []
        while self.t < time.time():
            ticks.extend(self._advance())
        return ticks

    async def stream(self):
        while True:
            for tick in self._advance():
                yield tick
            delay = self.t - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
