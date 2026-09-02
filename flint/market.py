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
import json
import os
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


# Finnhub industries folded into GICS-style sectors for the treemap; unmapped industries keep their own name
SECTOR_OF = {
    "Semiconductors": "Technology", "Technology": "Technology", "Software": "Technology", "Electronic Equipment": "Technology",
    "Communications": "Technology", "Computers": "Technology", "Electrical Equipment": "Industrials",
    "Banking": "Financials", "Financial Services": "Financials", "Insurance": "Financials", "Diversified Financial": "Financials",
    "Real Estate": "Real Estate", "Pharmaceuticals": "Health Care", "Biotechnology": "Health Care", "Health Care": "Health Care",
    "Medical Devices": "Health Care", "Life Sciences Tools & Services": "Health Care",
    "Energy": "Energy", "Oil & Gas": "Energy", "Utilities": "Utilities", "Chemicals": "Materials", "Metals & Mining": "Materials",
    "Packaging": "Materials", "Building": "Industrials", "Machinery": "Industrials", "Aerospace & Defense": "Industrials",
    "Industrial Conglomerates": "Industrials", "Transportation": "Industrials", "Airlines": "Industrials", "Logistics & Transportation": "Industrials",
    "Auto Components": "Consumer Discretionary", "Automobiles": "Consumer Discretionary", "Retail": "Consumer Discretionary",
    "Hotels, Restaurants & Leisure": "Consumer Discretionary", "Textiles, Apparel & Luxury Goods": "Consumer Discretionary",
    "Leisure Products": "Consumer Discretionary", "Distributors": "Consumer Discretionary", "Consumer products": "Consumer Staples",
    "Beverages": "Consumer Staples", "Food Products": "Consumer Staples", "Tobacco": "Consumer Staples", "Food & Staples Retailing": "Consumer Staples",
    "Media": "Communication Services", "Telecommunication": "Communication Services", "Entertainment": "Communication Services",
    "Interactive Media & Services": "Communication Services",
}


class MarketScanner:
    def __init__(self, finnhub_key: str = "", radar_top: int = 250, radar_count: int = 100, state_dir: str = ""):
        self.finnhub_key = finnhub_key
        self.radar_top = radar_top
        self.radar_count = radar_count
        self.state: dict = {}
        self.status = ""
        self.sector_file = os.path.join(state_dir, "sectors.json") if state_dir else ""
        self.sectors: dict = {}                       # symbol -> sector (persisted; industries do not change)
        try:
            self.sectors = json.loads(open(self.sector_file).read()) if self.sector_file else {}
        except (OSError, ValueError):
            self.sectors = {}

    async def _fill_sectors(self, c, syms, budget: int = 40) -> None:
        """Look up the industry for radar names we have not seen, a few per scan (Finnhub allows
        60 calls/min), and persist. Names Finnhub does not know get "Other" so they are not retried."""
        if not self.finnhub_key:
            return
        todo = [s for s in syms if s not in self.sectors][:budget]
        for s in todo:
            try:
                r = await c.get("https://finnhub.io/api/v1/stock/profile2", params={"symbol": s, "token": self.finnhub_key})
                if r.status_code == 429:
                    break
                ind = (r.json().get("finnhubIndustry") or "") if r.status_code == 200 else ""
                self.sectors[s] = SECTOR_OF.get(ind, ind or "Other")
            except Exception:  # noqa: BLE001
                self.sectors[s] = "Other"
            await asyncio.sleep(0.15)
        if todo and self.sector_file:
            try:
                tmp = self.sector_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self.sectors, f)
                os.replace(tmp, self.sector_file)
            except OSError:
                pass

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
                            params={"scrIds": scr, "count": 12}, headers={"User-Agent": UA})
            if r.status_code == 200:
                q = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
                return [{"symbol": x.get("symbol"), "chg": round(x.get("regularMarketChangePercent") or 0, 2),
                         "price": x.get("regularMarketPrice")} for x in q[:12]]
        except Exception:  # noqa: BLE001
            pass
        return []

    async def _radar(self, c):
        """Merge the three big predefined screeners into one ranked movers watchlist."""
        seen = {}
        for scr, cat in (("most_actives", "active"), ("day_gainers", "gainer"), ("day_losers", "loser"),
                         ("small_cap_gainers", "smallcap"), ("aggressive_small_caps", "smallcap"),
                         ("undervalued_large_caps", "value"), ("growth_technology_stocks", "growth"),
                         ("undervalued_growth_stocks", "growth"), ("most_shorted_stocks", "shorted")):  # broad supply for a large radar
            try:
                r = await c.get("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                                params={"scrIds": scr, "count": self.radar_count}, headers={"User-Agent": UA})
                if r.status_code != 200:
                    continue
                for x in r.json().get("finance", {}).get("result", [{}])[0].get("quotes", []):
                    sym = x.get("symbol")
                    if not sym or sym in seen:
                        continue
                    seen[sym] = {"symbol": sym, "name": (x.get("shortName") or "")[:24],
                                 "chg": round(x.get("regularMarketChangePercent") or 0, 2),
                                 "price": x.get("regularMarketPrice"),
                                 "vol": x.get("regularMarketVolume"), "cat": cat,
                                 "mcap": x.get("marketCap")}
            except Exception:  # noqa: BLE001
                pass
        rows = sorted(seen.values(), key=lambda r: -abs(r["chg"]))[:self.radar_top]
        await self._fill_sectors(c, [r["symbol"] for r in rows])
        for r in rows:
            r["sector"] = self.sectors.get(r["symbol"])
        return rows

    async def scan(self, say=None, light=False) -> dict:
        say = say or (lambda *a, **k: None)
        prev = self.state or {}
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            # index/sector quotes are Finnhub-heavy and slow-moving -- only fetch them on a full scan
            idx = (await self._finnhub_quotes(c, list(INDEX_ETFS)) if self.finnhub_key else {}) if not light else {}
            sect = (await self._finnhub_quotes(c, list(SECTOR_ETFS)) if self.finnhub_key else {}) if not light else {}
            vix = await self._vix(c)
            crypto = None   # crypto removed from flint; equities only
            actives = await self._movers(c, "most_actives")
            gainers = await self._movers(c, "day_gainers")
            losers = await self._movers(c, "day_losers")
            radar = prev.get("radar", []) if light else await self._radar(c)   # light refresh keeps the last radar

        # preserve slow-moving sectors/indices/breadth when this scan skipped them or a fetch returned empty,
        # so the whole-market panel never flickers to empty between full scans
        allq = {**idx, **sect}
        if allq:
            up = sum(1 for v in allq.values() if v["chg"] > 0)
            n = len(allq)
            breadth = up / n
        else:
            breadth, up, n = prev.get("breadth", 0.5), prev.get("breadth_up", 0), prev.get("breadth_n", 0)
        sectors = sorted(({"etf": k, "name": SECTOR_ETFS[k], **v} for k, v in sect.items()), key=lambda x: -x["chg"]) if sect else prev.get("sectors", [])
        indices = [{"etf": k, "name": INDEX_ETFS[k], **v} for k, v in idx.items()] if idx else prev.get("indices", [])
        vix = vix if vix else prev.get("vix")

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
