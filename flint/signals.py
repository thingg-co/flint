"""Exogenous signal providers: retail attention, sentiment, derivatives positioning,
and Michael Burry's disclosed 13F holdings.

These do not produce prices. Each provider returns slow-moving, bounded signals that
are fed to the model as extra per-asset features and combined into a "crowding /
euphoria" index that drives the Burry contrarian overlay in the policy. Everything is
toggleable and degrades to neutral (zero) when a provider is off or failing.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time

import httpx

from .market import MarketScanner
from .sources import asset_class

log = logging.getLogger(__name__)

UA = "flint/0.1 (research; contact flint@localhost)"


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def base_of(sym: str) -> str:
    return sym.split("-")[0].upper() if "-" in sym else sym.upper()


# Common company-name -> ticker aliases so 13F issuer names and WSB names line up
# with configured equity symbols.
NAME_TO_TICKER = {
    "nvidia": "NVDA", "apple": "AAPL", "tesla": "TSLA", "microstrategy": "MSTR",
    "strategy": "MSTR", "micron": "MU", "amazon": "AMZN", "alphabet": "GOOGL",
    "meta platforms": "META", "microsoft": "MSFT", "advanced micro": "AMD",
    "sandisk": "SNDK", "broadcom": "AVGO", "palantir": "PLTR", "coinbase": "COIN",
    "bank of america": "BAC", "american express": "AXP", "coca-cola": "KO", "coca cola": "KO",
    "chevron": "CVX", "occidental": "OXY", "kraft heinz": "KHC", "moody": "MCO", "chubb": "CB",
    "davita": "DVA", "kroger": "KR", "visa": "V", "mastercard": "MA", "citigroup": "C",
    "ally financial": "ALLY", "sirius": "SIRI", "domino": "DPZ", "constellation": "STZ", "netflix": "NFLX",
    "uber": "UBER", "brookfield": "BN", "chipotle": "CMG", "hilton": "HLT", "howard hughes": "HHH",
    "restaurant brands": "QSR", "nike": "NKE", "canadian pacific": "CP", "seaport": "SEG", "amazon.com": "AMZN",
}
# Crypto proxies traded on WSB whose sentiment we can lend to a crypto asset.
CRYPTO_PROXY = {"BTC": ["MSTR", "IBIT", "BITO"], "ETH": ["ETHA"]}

# Investor council. Each entry: SEC 13F CIK plus an "ethos" profile encoding that
# investor's documented, public investment philosophy (not fabricated quotes) across a
# few dimensions in [-1, 1], and a one-line basis. Holdings come from the 13F; the ethos
# shapes HOW the overlay trades. Bill Gross is intentionally absent — he files no 13F
# (bonds, retired). Adding an investor is one line here.
ETHOS_DIMS = ["contrarian", "value", "momentum", "quality", "macro_bear", "activist", "safety"]
GURUS = [
    {"id": "scion", "name": "Burry / Scion", "cik": "0001649339",
     "ethos": {"contrarian": 0.95, "value": 0.9, "momentum": -0.4, "quality": 0.3, "macro_bear": 0.7, "activist": 0.1, "safety": 0.9},
     "basis": "Deep-value contrarian, margin of safety; shorts bubbles (subprime, later big index/AI put bets)."},
    {"id": "berkshire", "name": "Buffett / Berkshire", "cik": "0001067983",
     "ethos": {"contrarian": 0.5, "value": 0.9, "momentum": -0.2, "quality": 0.95, "macro_bear": 0.0, "activist": 0.1, "safety": 0.85},
     "basis": "Wonderful businesses at fair prices, long horizon; greedy when others are fearful."},
    {"id": "ackman", "name": "Ackman / Pershing Square", "cik": "0001336528",
     "ethos": {"contrarian": 0.5, "value": 0.7, "momentum": 0.0, "quality": 0.9, "macro_bear": 0.2, "activist": 0.9, "safety": 0.6},
     "basis": "Concentrated quality with activist catalysts; occasional cheap macro tail hedges."},
    {"id": "point72", "name": "Cohen / Point72", "cik": "0001603466",
     "ethos": {"contrarian": 0.1, "value": 0.3, "momentum": 0.85, "quality": 0.5, "macro_bear": 0.0, "activist": 0.2, "safety": 0.3},
     "basis": "Fast information-edge trading; momentum and catalysts; high turnover."},
    {"id": "icahn", "name": "Icahn / Icahn Capital", "cik": "0000921669",
     "ethos": {"contrarian": 0.8, "value": 0.8, "momentum": -0.2, "quality": 0.5, "macro_bear": 0.4, "activist": 0.95, "safety": 0.6},
     "basis": "Activist deep value; unlock value via breakups and pressure; has run large market hedges."},
    {"id": "europac", "name": "Schiff / Euro Pacific", "cik": "0001796651",
     "ethos": {"contrarian": 0.7, "value": 0.6, "momentum": -0.3, "quality": 0.4, "macro_bear": 0.95, "activist": 0.1, "safety": 0.7},
     "basis": "Perma-bearish on US bubbles and the dollar; favors gold, commodities, foreign value."},
    {"id": "dailyjournal", "name": "Munger / Daily Journal", "cik": "0000783412",
     "ethos": {"contrarian": 0.6, "value": 0.95, "momentum": -0.5, "quality": 0.95, "macro_bear": 0.2, "activist": 0.1, "safety": 0.9},
     "basis": "Great businesses at fair prices, extreme patience and concentration, avoid stupidity (Daily Journal; Munger d. 2023)."},
]



class WSBProvider:
    """r/wallstreetbets attention and sentiment via free aggregators.

    Reddit's own API is OAuth-walled, so we use ApeWisdom (mention counts and
    momentum) and Tradestie (bullish/bearish scores) and merge them per ticker.
    """

    id = "wsb"
    name = "WallStreetBets"

    def __init__(self):
        self.last: dict[str, dict] = {}
        self.status = ""

    async def gather(self, say=None) -> dict:
        say = say or (lambda *a, **k: None)
        tickers: dict[str, dict] = {}
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}, follow_redirects=True) as c:
            try:
                r = await c.get("https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/1")
                r.raise_for_status()
                for row in r.json().get("results", []):
                    m = int(row.get("mentions") or 0)
                    m0 = int(row.get("mentions_24h_ago") or 0) or m
                    tickers[row["ticker"].upper()] = {
                        "mentions": m, "mention_chg": (m - m0) / max(m0, 1),
                        "upvotes": int(row.get("upvotes") or 0), "rank": row.get("rank"),
                        "name": row.get("name", ""), "sent": None,
                    }
                say(f"ApeWisdom: {len(tickers)} WSB tickers")
            except Exception as e:  # noqa: BLE001
                say(f"ApeWisdom failed: {type(e).__name__}", "warn")
            try:
                r = await c.get("https://tradestie.com/api/v1/apps/reddit")
                r.raise_for_status()
                for row in r.json():
                    t = row["ticker"].upper()
                    d = tickers.setdefault(t, {"mentions": row.get("no_of_comments", 0), "mention_chg": 0.0,
                                               "upvotes": 0, "rank": None, "name": "", "sent": None})
                    d["sent"] = float(row.get("sentiment_score") or 0.0)
                    d["comments"] = int(row.get("no_of_comments") or 0)
                say(f"Tradestie: sentiment for {len(r.json())} WSB tickers")
            except Exception as e:  # noqa: BLE001
                say(f"Tradestie failed: {type(e).__name__}", "warn")
        maxm = max((d["mentions"] for d in tickers.values()), default=1) or 1
        for d in tickers.values():
            d["attention"] = min(1.0, math.log1p(d["mentions"]) / math.log1p(maxm))
        self.last = tickers
        self.status = f"{len(tickers)} tickers"
        return tickers

    def for_symbol(self, sym: str) -> dict:
        base = base_of(sym)
        names = [base] + CRYPTO_PROXY.get(base, [])
        rows = [self.last[n] for n in names if n in self.last]
        if not rows:
            return {"wsb_sent": 0.0, "wsb_attn": 0.0, "mentions": 0, "top": None}
        attn = max(r["attention"] for r in rows)
        sents = [r["sent"] for r in rows if r.get("sent") is not None]
        chg = max((r.get("mention_chg", 0.0) for r in rows), default=0.0)
        sent = sum(sents) / len(sents) if sents else clip(chg)  # fall back to mention momentum
        top = max(rows, key=lambda r: r["mentions"])
        return {"wsb_sent": clip(sent), "wsb_attn": attn, "mentions": sum(r["mentions"] for r in rows),
                "mention_chg": chg, "top": top.get("name") or top.get("ticker")}


class FearGreedProvider:
    """Crypto Fear & Greed index (market-wide) from alternative.me."""

    id = "feargreed"
    name = "Fear & Greed"

    def __init__(self):
        self.value = 50
        self.classification = ""
        self.status = ""

    async def gather(self, say=None) -> dict:
        say = say or (lambda *a, **k: None)
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as c:
            r = await c.get("https://api.alternative.me/fng/", params={"limit": 1})
            r.raise_for_status()
            d = r.json()["data"][0]
        self.value = int(d["value"])
        self.classification = d["value_classification"]
        self.status = f"{self.value} ({self.classification})"
        say(f"Fear & Greed: {self.value} ({self.classification})")
        return {"value": self.value, "class": self.classification, "norm": clip((self.value - 50) / 50.0)}


class DerivativesProvider:
    """Binance USDT-perp positioning per crypto asset: funding, open interest change,
    and the retail long/short account ratio. No key."""

    id = "derivatives"
    name = "Derivatives (Binance)"
    FAPI = "https://fapi.binance.com"

    def __init__(self):
        self.prev_oi: dict[str, float] = {}
        self.last: dict[str, dict] = {}
        self.status = ""
        self.note = ""

    def pair(self, sym: str) -> str:
        return base_of(sym) + "USDT"

    async def gather(self, symbols, say=None) -> dict:
        say = say or (lambda *a, **k: None)
        out: dict[str, dict] = {}
        cryptos = [s for s in symbols if "-" in s]
        pairs = {self.pair(s): s for s in cryptos}
        if not cryptos:
            self.last = {}
            self.status = "no crypto symbols"
            return {}
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as c:
            for sym in cryptos:
                pair = self.pair(sym)
                funding = 0.0
                oiv = None
                lsr = 1.0
                try:
                    fr = await c.get(f"{self.FAPI}/fapi/v1/fundingRate", params={"symbol": pair, "limit": 1})
                    if fr.status_code == 200 and fr.json():
                        funding = float(fr.json()[0].get("fundingRate") or 0.0)
                    else:
                        self.note = f"fundingRate {fr.status_code}"
                    await asyncio.sleep(0.25)
                    oi = await c.get(f"{self.FAPI}/fapi/v1/openInterest", params={"symbol": pair})
                    if oi.status_code == 200:
                        oiv = float(oi.json().get("openInterest") or 0.0)
                    await asyncio.sleep(0.25)
                    ls = await c.get(f"{self.FAPI}/futures/data/globalLongShortAccountRatio",
                                     params={"symbol": pair, "period": "5m", "limit": 1})
                    if ls.status_code == 200 and ls.json():
                        lsr = float(ls.json()[0]["longShortRatio"])
                    await asyncio.sleep(0.25)
                except Exception as e:  # noqa: BLE001
                    say(f"{pair}: {type(e).__name__}", "warn")
                if oiv is None:
                    continue
                oiv = oiv if oiv is not None else self.prev_oi.get(sym, 0.0)
                prev = self.prev_oi.get(sym, oiv)
                oi_chg = (oiv - prev) / prev if prev else 0.0
                self.prev_oi[sym] = oiv
                out[sym] = {
                    "funding": funding, "funding_sig": clip(math.tanh(funding * 2000)),
                    "oi": oiv, "oi_chg": clip(oi_chg, -1, 1), "oi_chg_sig": clip(math.tanh(oi_chg * 5)),
                    "longshort": lsr, "longshort_sig": clip(math.tanh(math.log(max(lsr, 1e-3)))),
                }
        self.last = out
        self.status = f"{len(out)} perps" if out else "no data"
        if out:
            say("Binance perps: " + ", ".join(
                f"{base_of(s)} fund {d['funding']*100:+.3f}% OI {d['oi_chg']*100:+.1f}% L/S {d['longshort']:.2f}"
                for s, d in out.items()))
        return out


class Guru13F:
    """A fund's disclosed positions from its latest 13F-HR (SEC EDGAR). Puts are read as
    bearish (negative weight), shares/calls as bullish. Used for Burry (Scion) and, as a
    long-value counterpoint, Buffett (Berkshire Hathaway)."""

    def __init__(self, pid: str, name: str, cik: str):
        self.id = pid
        self.name = name
        self.CIK = cik.zfill(10)
        self.holdings: list[dict] = []
        self.asof: str | None = None
        self.status = ""
        self._by_ticker: dict[str, float] = {}
        self._fetched_at = 0.0

    async def gather(self, say=None, max_age: float = 21600.0) -> dict:
        say = say or (lambda *a, **k: None)
        if self.holdings and time.time() - self._fetched_at < max_age:
            return {"holdings": self.holdings, "asof": self.asof}
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}, follow_redirects=True) as c:
            sub = (await c.get(f"https://data.sec.gov/submissions/CIK{self.CIK}.json")).json()
            rec = sub["filings"]["recent"]
            acc = period = None
            for f, a, rd in zip(rec["form"], rec["accessionNumber"], rec.get("reportDate", rec["filingDate"])):
                if f in ("13F-HR", "13F-HR/A"):   # holdings report (skip 13F-NT notices with no table)
                    acc, period = a, rd
                    break
            if not acc:
                self.status = "no 13F found"
                return {"holdings": [], "asof": None}
            accn = acc.replace("-", "")
            idx = (await c.get(f"https://www.sec.gov/Archives/edgar/data/{int(self.CIK)}/{accn}/index.json")).json()
            info_name = None
            for item in idx.get("directory", {}).get("item", []):
                n = item["name"].lower()
                if n.endswith(".xml") and ("infotable" in n or "form13f" in n) and "primary_doc" not in n:
                    info_name = item["name"]
            if not info_name:  # fall back to any xml that is not the cover page
                for item in idx.get("directory", {}).get("item", []):
                    if item["name"].lower().endswith(".xml") and "primary_doc" not in item["name"].lower():
                        info_name = item["name"]
            xml = (await c.get(f"https://www.sec.gov/Archives/edgar/data/{int(self.CIK)}/{accn}/{info_name}")).text
        self.holdings = self._parse(xml)
        self.asof = period
        total = sum(abs(h["value"]) for h in self.holdings) or 1
        for h in self.holdings:
            h["weight"] = round(h["value"] / total, 4)
        self.holdings.sort(key=lambda h: -abs(h["value"]))
        self._by_ticker = {}
        for h in self.holdings:
            if h["ticker"]:
                self._by_ticker[h["ticker"]] = self._by_ticker.get(h["ticker"], 0.0) + h["weight"]
        self._fetched_at = time.time()
        self.status = f"{len(self.holdings)} holdings as of {self.asof}"
        say(f"{self.name} ({self.asof}): " + ", ".join(
            f"{h['ticker'] or h['issuer'][:10]} {h['weight']*100:.0f}%{'(PUT)' if h['put'] else ''}"
            for h in self.holdings[:6]))
        return {"holdings": self.holdings, "asof": self.asof}

    def _parse(self, xml: str) -> list[dict]:
        xml = re.sub(r"<(\w+):", "<", xml).replace("</ns1:", "</").replace("</n1:", "</")
        out = []
        for block in re.findall(r"<infoTable>(.*?)</infoTable>", xml, re.S | re.I):
            def grab(tag):
                m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S | re.I)
                return m.group(1).strip() if m else ""
            issuer = grab("nameOfIssuer")
            value = float(re.sub(r"[^\d.]", "", grab("value")) or 0)   # in USD thousands
            put_call = grab("putCall").upper()
            put = put_call == "PUT"
            ticker = NAME_TO_TICKER.get(issuer.lower().replace(",", "").strip())
            if not ticker:
                for name, tk in NAME_TO_TICKER.items():
                    if name in issuer.lower():
                        ticker = tk
                        break
            out.append({"issuer": issuer, "ticker": ticker, "value": -value if put else value,
                        "put": put, "class": grab("titleOfClass")})
        # merge duplicate issuer/putCall rows
        merged: dict[tuple, dict] = {}
        for h in out:
            k = (h["issuer"], h["put"])
            if k in merged:
                merged[k]["value"] += h["value"]
            else:
                merged[k] = h
        return list(merged.values())

    def signal(self, sym: str) -> float:
        """Signed Burry exposure for a symbol: + long, - puts, 0 if not held."""
        return clip(self._by_ticker.get(base_of(sym), 0.0) * 3.0)


class FundamentalsProvider:
    """Per-equity fundamentals from Finnhub (P/E, EPS, growth, margins, ROE, 52w range).

    Fundamentals move slowly, so the whole set is cached for hours. They are turned into
    two bounded features — value (cheapness, from the earnings multiple) and quality (from
    margins, ROE and growth) — and the raw numbers are kept for display. The key is never
    logged.
    """

    id = "fundamentals"
    name = "Fundamentals (Finnhub)"
    REST = "https://finnhub.io/api/v1"

    def __init__(self, key: str):
        self.key = key
        self.data: dict = {}
        self._fetched = 0.0
        self.status = ""

    async def gather(self, symbols, say=None, max_age: float = 21600.0) -> dict:
        say = say or (lambda *a, **k: None)
        if not self.key:
            self.status = "no Finnhub key"
            return self.data
        equities = [s for s in symbols if asset_class(s) == "equity"]
        if not equities:
            self.status = "no equities"
            return {}
        if self.data and time.time() - self._fetched < max_age:
            return self.data
        out = {}
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in equities:
                try:
                    r = await c.get(f"{self.REST}/stock/metric", params={"symbol": sym, "metric": "all", "token": self.key})
                    if r.status_code != 200:
                        continue
                    m = r.json().get("metric", {})
                    pe = m.get("peTTM"); eps = m.get("epsTTM"); margin = m.get("netProfitMarginTTM")
                    roe = m.get("roeTTM"); rev_g = m.get("revenueGrowthTTMYoy"); eps_g = m.get("epsGrowthTTMYoy")
                    value = clip(math.tanh((25 - pe) / 25)) if pe and pe > 0 else 0.0
                    quality = clip(0.4 * math.tanh((margin or 0) / 25) + 0.3 * math.tanh((roe or 0) / 25) +
                                   0.3 * math.tanh((rev_g or 0) / 20))
                    out[sym] = {"pe": pe, "eps": eps, "ps": m.get("psTTM"), "pb": m.get("pbAnnual"),
                                "roe": roe, "margin": margin, "rev_growth": rev_g, "eps_growth": eps_g,
                                "div_yield": m.get("dividendYieldIndicatedAnnual"), "beta": m.get("beta"),
                                "w52_high": m.get("52WeekHigh"), "w52_low": m.get("52WeekLow"),
                                "value": round(value, 3), "quality": round(quality, 3)}
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    say(f"{sym} fundamentals: {type(e).__name__}", "warn")
        self.data = out
        self._fetched = time.time()
        self.status = f"{len(out)} equities"
        say(f"fundamentals: {len(out)} equities (e.g. " +
            ", ".join(f"{s} P/E {d['pe']:.0f}" for s, d in list(out.items())[:3] if d.get("pe")) + ")")
        return out

    def feat(self, sym: str) -> dict:
        d = self.data.get(sym)
        return {"f_value": d["value"], "f_quality": d["quality"]} if d else {"f_value": 0.0, "f_quality": 0.0}


class SignalHub:
    """Runs the toggleable signal providers and the investor council, assembling per-asset
    features (including a guru consensus and an ethos bias) plus a crowding index."""

    def __init__(self, cfg, symbols):
        self.cfg = cfg
        self.symbols = symbols
        self.providers = {p.id: p for p in (WSBProvider(), FundamentalsProvider(cfg.finnhub_key))}
        self.gurus = {g["id"]: Guru13F(g["id"], g["name"], g["cik"]) for g in GURUS}
        self.guru_meta = {g["id"]: g for g in GURUS}
        off = {x.strip() for x in cfg.signals_off.split(",") if x.strip()}
        self.enabled = {pid: pid not in off for pid in self.providers}
        for gid in self.gurus:
            self.enabled[gid] = gid not in off
        self.market = MarketScanner(cfg.finnhub_key, getattr(cfg, "radar_top", 250))
        self.enabled["market"] = "market" not in off
        self.state = {"per_asset": {}, "market": {}, "gurus": [], "council": {}, "crowding": {},
                      "guru_tilt": {}, "ethos_bias": {}, "providers": self.status(), "t": None}

    def toggle(self, pid: str, on: bool) -> dict:
        if pid not in self.enabled:
            return {"error": "unknown signal provider"}
        self.enabled[pid] = on
        return {"ok": True}

    def status(self) -> list[dict]:
        out = [{"id": p.id, "name": p.name, "enabled": self.enabled.get(p.id, False),
                "status": getattr(p, "status", "")} for p in self.providers.values()]
        for gid, g in self.gurus.items():
            out.append({"id": gid, "name": g.name, "enabled": self.enabled.get(gid, False), "status": g.status})
        out.append({"id": "market", "name": "Whole-market scan", "enabled": self.enabled.get("market", False),
                    "status": self.market.status})
        return out

    def _council(self) -> dict:
        active = [self.guru_meta[gid]["ethos"] for gid in self.gurus if self.enabled.get(gid)]
        if not active:
            return {d: 0.0 for d in ETHOS_DIMS}
        return {d: round(sum(e[d] for e in active) / len(active), 3) for d in ETHOS_DIMS}

    async def gather(self, say=None) -> dict:
        say = say or (lambda *a, **k: None)
        wsb = self.providers["wsb"]
        market = {"fng": 0.0, "fng_value": None, "fng_class": None}   # fng retired with crypto; stays neutral
        if self.enabled["wsb"]:
            try:
                await wsb.gather(say)
            except Exception as e:  # noqa: BLE001
                say(f"WSB failed: {type(e).__name__}", "warn")
        for gid, guru in self.gurus.items():
            if self.enabled.get(gid):
                try:
                    await guru.gather(say)
                except Exception as e:  # noqa: BLE001
                    say(f"{guru.name} 13F failed: {type(e).__name__}", "warn")

        fundamentals = self.providers["fundamentals"]
        if self.enabled.get("fundamentals"):
            try:
                await fundamentals.gather(self.symbols, say)
            except Exception as e:  # noqa: BLE001
                say(f"fundamentals failed: {type(e).__name__}", "warn")

        market_feat = {"breadth": 0.0, "vix": 0.0}
        if self.enabled.get("market"):
            try:
                market_feat = (await self.market.scan(say)).get("feat", market_feat)
            except Exception as e:  # noqa: BLE001
                say(f"market scan failed: {type(e).__name__}", "warn")

        # enrich the movers radar with our own signals
        radar = (self.market.state or {}).get("radar", [])
        wsb_last = self.providers["wsb"].last if self.enabled.get("wsb") else {}
        for row in radar:
            sym = row["symbol"]
            row["wsb"] = wsb_last.get(sym, {}).get("mentions", 0) if wsb_last else 0
            held = [self.guru_meta[gid]["name"].split(" / ")[0] for gid, g in self.gurus.items()
                    if self.enabled.get(gid) and abs(g.signal(sym)) > 0.02]
            row["gurus"] = held

        council = self._council()
        per_asset, crowding, guru_tilt, ethos_bias = {}, {}, {}, {}
        for sym in self.symbols:
            w = wsb.for_symbol(sym) if self.enabled["wsb"] else {"wsb_sent": 0.0, "wsb_attn": 0.0, "mentions": 0, "top": None}
            d = {}   # derivatives (crypto perps) retired; funding/oi/longshort stay neutral
            funding_sig, oi_sig, ls_sig = d.get("funding_sig", 0.0), d.get("oi_chg_sig", 0.0), d.get("longshort_sig", 0.0)
            crowd = (0.9 * market["fng"] + 1.1 * w["wsb_attn"] * (0.3 + 0.7 * clip(w["wsb_sent"] + 0.15)) +
                     1.0 * funding_sig + 0.7 * ls_sig + 0.4 * oi_sig)
            crowding[sym] = round(clip(crowd / 3.2), 3)

            # Per-guru stance on this asset from their 13F (long +, puts -).
            gsig = {gid: g.signal(sym) for gid, g in self.gurus.items() if self.enabled.get(gid)}
            guru_net = clip(sum(gsig.values()))
            # Ethos bias blends holdings with philosophy: quality investors' longs weigh more,
            # bearish investors' shorts weigh more, the council fades crowded names, and a
            # macro-bearish council (Schiff, Burry) drags everything down a little.
            long_w = sum(max(0.0, v) * (0.5 + 0.5 * self.guru_meta[gid]["ethos"]["quality"]) for gid, v in gsig.items())
            short_w = sum(max(0.0, -v) * (0.5 + 0.5 * self.guru_meta[gid]["ethos"]["macro_bear"]) for gid, v in gsig.items())
            eb = 0.7 * (long_w - short_w) - 0.4 * council["macro_bear"] - 0.5 * council["contrarian"] * crowding[sym]
            ethos_bias[sym] = round(clip(eb), 3)
            guru_tilt[sym] = round(guru_net, 3)

            feats = {"wsb_sent": w["wsb_sent"], "wsb_attn": w["wsb_attn"], "fng": market["fng"],
                     "funding": funding_sig, "oi_chg": oi_sig, "longshort": ls_sig,
                     "guru_net": round(guru_net, 3), "ethos_bias": ethos_bias[sym],
                     **(fundamentals.feat(sym) if self.enabled.get("fundamentals") else {"f_value": 0.0, "f_quality": 0.0}),
                     "breadth": market_feat["breadth"], "vix": market_feat["vix"]}
            per_asset[sym] = {**feats, "mentions": w["mentions"], "wsb_top": w.get("top"),
                              "funding": d.get("funding"), "oi_chg": d.get("oi_chg"), "longshort": d.get("longshort"),
                              "gurus": {gid: round(v, 3) for gid, v in gsig.items() if abs(v) > 1e-6},
                              "fundamentals": fundamentals.data.get(sym), "feat": feats}

        gurus_view = [{"id": gid, "name": g.name, "asof": g.asof, "enabled": self.enabled.get(gid),
                       "ethos": self.guru_meta[gid]["ethos"], "basis": self.guru_meta[gid]["basis"],
                       "n_holdings": len(g.holdings), "holdings": g.holdings[:20]} for gid, g in self.gurus.items()]
        self.state = {"per_asset": per_asset, "market": self.market.state, "gurus": gurus_view, "council": council,
                      "fear_greed": {"value": market.get("fng_value"), "class": market.get("fng_class"), "norm": market["fng"]},
                      "crowding": crowding, "guru_tilt": guru_tilt, "ethos_bias": ethos_bias,
                      "providers": self.status(), "t": time.time()}
        return self.state
