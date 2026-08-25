"""Whole-market scanner: breadth, sector rotation, volatility regime, index levels,
crypto dominance, and market-wide movers.

The DNN forecasts a focused tradeable universe, but the market it trades in is much
bigger. This module surveys that broader market on a slow cadence and produces (a) a
human-readable overview and (b) two market-wide features — breadth and a volatility/risk
signal — that are shared across every tracked asset to condition the regime gate. No key
is required beyond the Finnhub key already in use; Yahoo/CoinGecko cover the rest.
"""
from __future__ import annotations

import asyncio
import math
import time

import httpx

UA = "Mozilla/5.0 (flint-market)"

INDEX_ETFS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000"}
SECTOR_ETFS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
               "XLY": "Cons. Discretionary", "XLI": "Industrials", "XLC": "Communication",
               "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"}


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class MarketScanner:
    def __init__(self, finnhub_key: str = "", radar_top: int = 250):
        self.finnhub_key = finnhub_key
        self.radar_top = radar_top
        self.state: dict = {}
        self.status = ""

    async def _finnhub_quotes(self, c, syms):
        out = {}
        for s in syms:
            try:
                r = await c.get("https://finnhub.io/api/v1/quote", params={"symbol": s, "token": self.finnhub_key})
                if r.status_code == 200:
                    d = r.json()
                    if d.get("pc"):
                        out[s] = {"price": d.get("c"), "chg": (d["c"] - d["pc"]) / d["pc"] * 100.0}
                await asyncio.sleep(0.12)
            except Exception:  # noqa: BLE001
                pass
        return out

    async def _vix(self, c):
        try:
            r = await c.get("https://query1.finance.yahoo.com/v8/finance/spark",
                            params={"symbols": "^VIX", "range": "1d", "interval": "1d"}, headers={"User-Agent": UA})
            if r.status_code == 200:
                res = (r.json().get("spark", {}).get("result") or [{}])[0]
                resp = (res.get("response") or [{}])[0]
                return resp.get("meta", {}).get("regularMarketPrice")
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _crypto_global(self, c):
        try:
            r = await c.get("https://api.coingecko.com/api/v3/global", headers={"User-Agent": UA})
            if r.status_code == 200:
                g = r.json()["data"]
                return {"total_mcap": g["total_market_cap"]["usd"], "btc_dom": g["market_cap_percentage"]["btc"],
                        "eth_dom": g["market_cap_percentage"].get("eth"), "chg24": g["market_cap_change_percentage_24h_usd"]}
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _movers(self, c, scr):
        try:
            r = await c.get("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                            params={"scrIds": scr, "count": 8}, headers={"User-Agent": UA})
            if r.status_code == 200:
                q = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
                return [{"symbol": x.get("symbol"), "chg": round(x.get("regularMarketChangePercent") or 0, 2),
                         "price": x.get("regularMarketPrice")} for x in q[:8]]
        except Exception:  # noqa: BLE001
            pass
        return []

    async def _radar(self, c):
        """Merge the three big predefined screeners into one ranked movers watchlist."""
        seen = {}
        for scr, cat in (("most_actives", "active"), ("day_gainers", "gainer"), ("day_losers", "loser"),
                         ("small_cap_gainers", "smallcap"), ("aggressive_small_caps", "smallcap")):  # include penny/small-cap movers
            try:
                r = await c.get("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                                params={"scrIds": scr, "count": 100}, headers={"User-Agent": UA})
                if r.status_code != 200:
                    continue
                for x in r.json().get("finance", {}).get("result", [{}])[0].get("quotes", []):
                    sym = x.get("symbol")
                    if not sym or sym in seen:
                        continue
                    seen[sym] = {"symbol": sym, "name": (x.get("shortName") or "")[:24],
                                 "chg": round(x.get("regularMarketChangePercent") or 0, 2),
                                 "price": x.get("regularMarketPrice"),
                                 "vol": x.get("regularMarketVolume"), "cat": cat}
            except Exception:  # noqa: BLE001
                pass
        rows = sorted(seen.values(), key=lambda r: -abs(r["chg"]))
        return rows[:self.radar_top]

    async def scan(self, say=None) -> dict:
        say = say or (lambda *a, **k: None)
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            idx = await self._finnhub_quotes(c, list(INDEX_ETFS)) if self.finnhub_key else {}
            sect = await self._finnhub_quotes(c, list(SECTOR_ETFS)) if self.finnhub_key else {}
            vix = await self._vix(c)
            crypto = None   # crypto removed from flint; equities only
            actives = await self._movers(c, "most_actives")
            gainers = await self._movers(c, "day_gainers")
            losers = await self._movers(c, "day_losers")
            radar = await self._radar(c)

        allq = {**idx, **sect}
        up = sum(1 for v in allq.values() if v["chg"] > 0)
        n = len(allq)
        breadth = up / n if n else 0.5
        sectors = sorted(({"etf": k, "name": SECTOR_ETFS[k], **v} for k, v in sect.items()), key=lambda x: -x["chg"])
        indices = [{"etf": k, "name": INDEX_ETFS[k], **v} for k, v in idx.items()]

        # market-wide model features
        breadth_norm = round(clip(2 * breadth - 1), 3)
        vix_sig = round(clip((vix - 20) / 20.0), 3) if vix else 0.0     # + = elevated volatility / risk-off
        # risk-on composite: breadth up, VIX low, crypto up
        risk = 0.6 * breadth_norm - 0.7 * vix_sig + (0.3 * clip(crypto["chg24"] / 5.0) if crypto else 0.0)
        regime = "risk-on" if risk > 0.2 else "risk-off" if risk < -0.2 else "mixed"

        self.state = {
            "t": time.time(), "breadth": round(breadth, 3), "breadth_up": up, "breadth_n": n,
            "vix": vix, "vix_sig": vix_sig, "risk": round(clip(risk), 3), "regime": regime,
            "indices": indices, "sectors": sectors, "crypto": crypto,
            "movers": {"actives": actives, "gainers": gainers, "losers": losers},
            "radar": radar,
            "feat": {"breadth": breadth_norm, "vix": vix_sig},
        }
        self.status = f"{regime}, breadth {breadth*100:.0f}%, {len(radar)} movers" + (f", VIX {vix:.1f}" if vix else "")
        say(f"market: {regime} — breadth {up}/{n} up" + (f", VIX {vix:.1f}" if vix else "") +
            (f", BTC dominance {crypto['btc_dom']:.1f}%" if crypto else ""))
        if sectors:
            say(f"sector leaders: {sectors[0]['name']} {sectors[0]['chg']:+.1f}%, "
                f"{sectors[1]['name']} {sectors[1]['chg']:+.1f}%; laggard {sectors[-1]['name']} {sectors[-1]['chg']:+.1f}%")
        return self.state
