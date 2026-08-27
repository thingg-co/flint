"""Paper-trading simulator: follows the model's live suggestions with a $100k book that
runs continuously and compounds. Long/short, no leverage (gross exposure capped at the
book's equity), per-name weight capped, marked to market on live prices. Its state is
persisted in the checkpoint, so positions, cash, and P&L survive restarts.
"""
from __future__ import annotations

from collections import deque


class PaperBook:
    def __init__(self, start: float = 100_000.0, cost_bps: float = 1.0,
                 max_weight: float = 0.15, min_trade_frac: float = 0.004):
        self.start = float(start)
        self.cash = float(start)
        self.cost_bps = float(cost_bps)
        self.max_weight = float(max_weight)      # cap on a single name's gross weight
        self.min_trade_frac = float(min_trade_frac)  # skip rebalances smaller than this share of equity
        self.pos: dict[str, float] = {}          # signed shares
        self.avg: dict[str, float] = {}          # average entry price
        self.realized = 0.0
        self.fees = 0.0
        self.n_trades = 0
        self.spread_cost = 0.0        # half-spread paid crossing bid/ask, cumulative (reporting only)
        self.trades: deque = deque(maxlen=250)
        self.curve: deque = deque(maxlen=3000)   # {t, eq}
        self.last: dict[str, float] = {}         # last seen price per symbol
        self.started = None

    def _px(self, s, prices):
        return prices.get(s) or self.last.get(s)

    def equity(self, prices) -> float:
        v = self.cash
        for s, sh in self.pos.items():
            p = self._px(s, prices)
            if p:
                v += sh * p
        return v

    def rebalance(self, targets: dict, prices: dict, ts: float, quotes: dict | None = None) -> None:
        """targets: symbol -> desired signed weight (side * size, in [-1, 1]).
        quotes: optional {symbol: {"bid": float, "ask": float}} -- when present, buys fill at the
        ask and sells at the bid so the book pays the real spread; equity still marks to mid."""
        quotes = quotes or {}
        for s, p in prices.items():
            if p:
                self.last[s] = p
        if self.started is None:
            self.started = ts
        eq = self.equity(prices)
        tw = {s: max(-self.max_weight, min(self.max_weight, float(w))) for s, w in targets.items()}
        gross = sum(abs(w) for w in tw.values())
        scale = (1.0 / gross) if gross > 1 else 1.0        # no leverage
        min_notional = max(200.0, eq * self.min_trade_frac)
        for s, w in tw.items():
            ref = self._px(s, prices)
            if not ref or ref <= 0:
                continue
            desired = eq * w * scale / ref
            cur = self.pos.get(s, 0.0)
            delta = desired - cur
            if abs(delta * ref) < min_notional:
                continue
            self._fill(s, cur, delta, self._exec_price(s, delta, ref, quotes), ts)
        self.curve.append({"t": ts, "eq": round(self.equity(prices), 2)})

    def _exec_price(self, s, delta, ref, quotes) -> float:
        """Fill a buy at the ask and a sell at the bid; fall back to the reference (mid/last)
        price when no quote is available. Accrues the half-spread paid, for reporting."""
        q = quotes.get(s) or {}
        bid, ask = q.get("bid"), q.get("ask")
        if bid and ask and ask > bid > 0:
            self.spread_cost += abs(delta) * (ask - bid) / 2.0
            return ask if delta > 0 else bid
        return ref

    def _fill(self, s, cur, delta, p, ts) -> None:
        notional = delta * p
        fee = abs(notional) * self.cost_bps / 1e4
        self.cash -= notional + fee
        self.fees += fee
        new = cur + delta
        if cur == 0 or (cur > 0) == (delta > 0):           # opening or adding to a side
            tot = abs(cur) + abs(delta)
            self.avg[s] = (self.avg.get(s, p) * abs(cur) + p * abs(delta)) / tot if tot else p
        else:                                              # reducing, closing, or flipping
            closed = min(abs(delta), abs(cur))
            self.realized += (p - self.avg.get(s, p)) * (closed if cur > 0 else -closed)
            if (new > 0) != (cur > 0) and abs(new) > 1e-9:  # flipped through zero
                self.avg[s] = p
        if abs(new) < 1e-9:
            self.pos.pop(s, None)
            self.avg.pop(s, None)
        else:
            self.pos[s] = new
        self.n_trades += 1
        self.trades.appendleft({"t": ts, "sym": s, "side": "buy" if delta > 0 else "sell",
                                "shares": round(abs(delta), 2), "price": round(p, 4),
                                "notional": round(abs(notional), 2)})

    def to_state(self) -> dict:
        return {"cash": self.cash, "pos": dict(self.pos), "avg": dict(self.avg),
                "realized": self.realized, "fees": self.fees, "n_trades": self.n_trades,
                "start": self.start, "started": self.started, "last": dict(self.last),
                "spread_cost": self.spread_cost,
                "trades": list(self.trades), "curve": list(self.curve)}

    def load_state(self, d: dict, symbols=None) -> None:
        self.cash = float(d.get("cash", self.cash))
        self.pos = {k: float(v) for k, v in (d.get("pos") or {}).items() if symbols is None or k in symbols}
        self.avg = {k: float(v) for k, v in (d.get("avg") or {}).items() if k in self.pos}
        self.realized = float(d.get("realized", 0.0))
        self.fees = float(d.get("fees", 0.0))
        self.n_trades = int(d.get("n_trades", 0))
        self.spread_cost = float(d.get("spread_cost", 0.0))
        self.start = float(d.get("start", self.start))
        self.started = d.get("started")
        self.last = dict(d.get("last") or {})
        self.trades = deque(d.get("trades") or [], maxlen=self.trades.maxlen)
        self.curve = deque(d.get("curve") or [], maxlen=self.curve.maxlen)

    def snapshot(self, prices: dict) -> dict:
        eq = self.equity(prices)
        positions, upnl = [], 0.0
        for s, sh in self.pos.items():
            p = self._px(s, prices) or 0.0
            a = self.avg.get(s, p)
            val = sh * p
            u = (p - a) * sh
            upnl += u
            positions.append({"sym": s, "shares": round(sh, 2), "avg": round(a, 4), "price": round(p, 4),
                              "value": round(val, 2), "upnl": round(u, 2),
                              "weight": round(val / eq, 4) if eq else 0.0})
        positions.sort(key=lambda x: -abs(x["value"]))
        gross = sum(abs(x["value"]) for x in positions)
        net = sum(x["value"] for x in positions)
        return {"start": self.start, "equity": round(eq, 2), "cash": round(self.cash, 2),
                "gross": round(gross, 2), "net_exposure": round(net, 2),
                "realized": round(self.realized, 2), "unrealized": round(upnl, 2), "fees": round(self.fees, 2),
                "spread_cost": round(self.spread_cost, 2),
                "pnl": round(eq - self.start, 2), "return_pct": round((eq / self.start - 1) * 100, 3),
                "n_trades": self.n_trades, "started": self.started,
                "positions": positions, "trades": list(self.trades)[:50], "curve": list(self.curve)}
