"""Paper-trading simulator: follows the model's live suggestions with a $100k book that
runs continuously and compounds. Longs are stock; shorts are expressed as long puts so the
loss is capped at the premium (never unlimited). No leverage (gross exposure capped at the
book's equity), per-name weight capped, marked to market. State is persisted in the
checkpoint, so positions, cash, and P&L survive restarts.
"""
from __future__ import annotations

import datetime
from collections import deque

from .options import bs_put, years_to


class PaperBook:
    def __init__(self, start: float = 100_000.0, cost_bps: float = 1.0, max_weight: float = 0.15,
                 min_trade_frac: float = 0.004, option_commission: float = 0.65, put_min_hold_s: float = 0.0):
        self.start = float(start)
        self.cash = float(start)
        self.cost_bps = float(cost_bps)
        self.max_weight = float(max_weight)          # cap on a single name's gross weight
        self.min_trade_frac = float(min_trade_frac)  # skip rebalances smaller than this share of equity
        self.option_commission = float(option_commission)
        self.put_min_hold_s = float(put_min_hold_s)   # hold a put at least this long before closing (anti-churn)
        self.pos: dict[str, float] = {}              # signed shares (longs only, in practice)
        self.avg: dict[str, float] = {}              # average entry price
        self.puts: dict[str, dict] = {}              # sym -> {contracts, strike, expiry_ts, entry_prem, iv}
        self.realized = 0.0
        self.fees = 0.0
        self.option_fees = 0.0
        self.n_trades = 0
        self.spread_cost = 0.0
        self.trades: deque = deque(maxlen=250)
        self.curve: deque = deque(maxlen=3000)
        self.last: dict[str, float] = {}
        self.started = None
        self._now = 0.0

    def _px(self, s, prices):
        return prices.get(s) or self.last.get(s)

    def _put_value(self, s, prices, now) -> float:
        pt = self.puts.get(s)
        spot = self._px(s, prices) if pt else None
        if not pt or not spot:
            return 0.0
        return bs_put(spot, pt["strike"], years_to(pt["expiry_ts"], now), pt["iv"]) * pt["contracts"] * 100.0

    def equity(self, prices, now=None) -> float:
        now = self._now if now is None else now
        v = self.cash
        for s, sh in self.pos.items():
            p = self._px(s, prices)
            if p:
                v += sh * p
        for s in self.puts:
            v += self._put_value(s, prices, now)
        return v

    def rebalance(self, targets: dict, prices: dict, ts: float, quotes: dict | None = None,
                  put_quotes: dict | None = None) -> None:
        """targets: symbol -> desired signed weight (side * size). A negative weight is a short,
        opened as a delta-equivalent long put from put_quotes[sym] (fetched from a live chain)."""
        quotes = quotes or {}
        put_quotes = put_quotes or {}
        self._now = ts
        for s, p in prices.items():
            if p:
                self.last[s] = p
        if self.started is None:
            self.started = ts
        self._settle_expired(prices, ts)
        eq = self.equity(prices, ts)
        tw = {s: max(-self.max_weight, min(self.max_weight, float(w))) for s, w in targets.items()}
        gross = sum(abs(w) for w in tw.values())
        scale = (1.0 / gross) if gross > 1 else 1.0        # no leverage
        min_notional = max(200.0, eq * self.min_trade_frac)
        for s, w in tw.items():
            ref = self._px(s, prices)
            if not ref or ref <= 0:
                continue
            if w >= 0:                                      # long or flat: close any put (once past min hold), manage stock
                if s in self.puts and ts - self.puts[s].get("opened", 0.0) >= self.put_min_hold_s:
                    self._close_put(s, ref, ts)
                desired = eq * w * scale / ref
                cur = self.pos.get(s, 0.0)
                delta = desired - cur
                if abs(delta * ref) >= min_notional:
                    self._fill(s, cur, delta, self._exec_price(s, delta, ref, quotes), ts)
            else:                                           # short: express as a long put, no naked share short
                cur = self.pos.get(s, 0.0)
                if cur > 0:                                 # close any long stock first
                    self._fill(s, cur, -cur, self._exec_price(s, -cur, ref, quotes), ts)
                if s not in self.puts and s in put_quotes:
                    self._open_put(s, eq * abs(w) * scale, ref, put_quotes[s], ts)
                # already holding a put -> hold it (marked); wanted-short-but-no-chain -> stay flat
        self.curve.append({"t": ts, "eq": round(self.equity(prices, ts), 2)})

    def _exec_price(self, s, delta, ref, quotes) -> float:
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
        if cur == 0 or (cur > 0) == (delta > 0):
            tot = abs(cur) + abs(delta)
            self.avg[s] = (self.avg.get(s, p) * abs(cur) + p * abs(delta)) / tot if tot else p
        else:
            closed = min(abs(delta), abs(cur))
            self.realized += (p - self.avg.get(s, p)) * (closed if cur > 0 else -closed)
            if (new > 0) != (cur > 0) and abs(new) > 1e-9:
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

    def _open_put(self, s, notional, spot, pq, ts) -> None:
        prem = pq.get("premium")
        if not prem or prem <= 0 or not spot:
            return
        dl = max(0.1, abs(pq.get("delta") or 0.5))          # size to delta-equivalent short exposure
        contracts = notional / (100.0 * spot * dl)
        if contracts < 0.01:
            return
        cost = contracts * prem * 100.0
        fee = contracts * self.option_commission
        self.cash -= cost + fee
        self.option_fees += fee
        self.n_trades += 1
        self.puts[s] = {"contracts": contracts, "strike": float(pq["strike"]), "expiry_ts": float(pq["expiry_ts"]),
                        "entry_prem": float(prem), "iv": float(pq["iv"]), "opened": ts}
        self.trades.appendleft({"t": ts, "sym": s, "side": "buy put", "shares": round(contracts, 2),
                                "price": round(prem, 2), "notional": round(cost, 2),
                                "note": f"P{pq['strike']:.0f} {self._exp(pq['expiry_ts'])}"})

    def _close_put(self, s, spot, ts) -> None:
        pt = self.puts.pop(s, None)
        if not pt:
            return
        val = bs_put(spot, pt["strike"], years_to(pt["expiry_ts"], ts), pt["iv"]) * pt["contracts"] * 100.0
        fee = pt["contracts"] * self.option_commission
        self.cash += val - fee
        self.option_fees += fee
        self.realized += val - pt["entry_prem"] * pt["contracts"] * 100.0
        self.n_trades += 1
        px = val / (pt["contracts"] * 100.0) if pt["contracts"] else 0.0
        self.trades.appendleft({"t": ts, "sym": s, "side": "sell put", "shares": round(pt["contracts"], 2),
                                "price": round(px, 2), "notional": round(val, 2)})

    def _settle_expired(self, prices, ts) -> None:
        for s in [s for s, pt in self.puts.items() if years_to(pt["expiry_ts"], ts) <= 0]:
            self._close_put(s, self._px(s, prices) or self.puts[s]["strike"], ts)

    @staticmethod
    def _exp(ts) -> str:
        try:
            return datetime.datetime.fromtimestamp(ts).strftime("%b%d")
        except (ValueError, OSError, OverflowError):
            return ""

    def to_state(self) -> dict:
        return {"cash": self.cash, "pos": dict(self.pos), "avg": dict(self.avg), "puts": dict(self.puts),
                "realized": self.realized, "fees": self.fees, "option_fees": self.option_fees,
                "n_trades": self.n_trades, "start": self.start, "started": self.started, "last": dict(self.last),
                "spread_cost": self.spread_cost, "trades": list(self.trades), "curve": list(self.curve)}

    def load_state(self, d: dict, symbols=None) -> None:
        self.cash = float(d.get("cash", self.cash))
        self.pos = {k: float(v) for k, v in (d.get("pos") or {}).items() if symbols is None or k in symbols}
        self.avg = {k: float(v) for k, v in (d.get("avg") or {}).items() if k in self.pos}
        self.puts = {k: v for k, v in (d.get("puts") or {}).items() if symbols is None or k in symbols}
        self.realized = float(d.get("realized", 0.0))
        self.fees = float(d.get("fees", 0.0))
        self.option_fees = float(d.get("option_fees", 0.0))
        self.n_trades = int(d.get("n_trades", 0))
        self.spread_cost = float(d.get("spread_cost", 0.0))
        self.start = float(d.get("start", self.start))
        self.started = d.get("started")
        self.last = dict(d.get("last") or {})
        self.trades = deque(d.get("trades") or [], maxlen=self.trades.maxlen)
        self.curve = deque(d.get("curve") or [], maxlen=self.curve.maxlen)

    def snapshot(self, prices: dict) -> dict:
        now = self._now
        eq = self.equity(prices, now)
        positions, upnl = [], 0.0
        for s, sh in self.pos.items():
            p = self._px(s, prices) or 0.0
            a = self.avg.get(s, p)
            val = sh * p
            u = (p - a) * sh
            upnl += u
            positions.append({"sym": s, "kind": "stock", "shares": round(sh, 2), "avg": round(a, 4),
                              "price": round(p, 4), "value": round(val, 2), "upnl": round(u, 2),
                              "weight": round(val / eq, 4) if eq else 0.0})
        for s, pt in self.puts.items():
            val = self._put_value(s, prices, now)
            paid = pt["entry_prem"] * pt["contracts"] * 100.0
            u = val - paid
            upnl += u
            positions.append({"sym": s, "kind": "put", "shares": round(pt["contracts"], 2),
                              "strike": pt["strike"], "expiry": self._exp(pt["expiry_ts"]),
                              "avg": round(pt["entry_prem"], 2), "price": round(val / (pt["contracts"] * 100.0), 2) if pt["contracts"] else 0.0,
                              "value": round(val, 2), "upnl": round(u, 2), "risk": round(paid, 2),
                              "weight": round(val / eq, 4) if eq else 0.0})
        positions.sort(key=lambda x: -abs(x["value"]))
        gross = sum(abs(x["value"]) for x in positions)
        net = sum(x["value"] for x in positions if x["kind"] == "stock") - sum(x["value"] for x in positions if x["kind"] == "put")
        return {"start": self.start, "equity": round(eq, 2), "cash": round(self.cash, 2),
                "gross": round(gross, 2), "net_exposure": round(net, 2),
                "realized": round(self.realized, 2), "unrealized": round(upnl, 2),
                "fees": round(self.fees, 2), "option_fees": round(self.option_fees, 2),
                "spread_cost": round(self.spread_cost, 2),
                "pnl": round(eq - self.start, 2), "return_pct": round((eq / self.start - 1) * 100, 3),
                "n_trades": self.n_trades, "started": self.started, "n_puts": len(self.puts),
                "positions": positions, "trades": list(self.trades)[:50], "curve": list(self.curve)}
