"""Pluggable market-data sources and the manager that routes them.

Each canonical symbol (e.g. BTC-USD, AAPL) is served by the highest-priority
source that is enabled and currently producing data. Sources stream ticks for
every symbol they support; the manager's router forwards only the active
provider's ticks for each symbol and drops the rest, so enabling a second venue
gives instant failover without double-counting volume. Sources can be toggled at
runtime from the control panel.

Sources fall into two shapes:
  * push  - a websocket the source keeps open (Coinbase, Binance)
  * poll  - a REST endpoint sampled on an interval (Yahoo, Kraken, CoinGecko,
            Alpha Vantage). Polled sources emit one tick per fresh observation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field

import httpx
import websockets

from .feed import SimFeed, Tick
from .schwab import SchwabAuth

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "BNB", "MATIC", "BCH", "ATOM"}
COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin",
                 "ADA": "cardano", "AVAX": "avalanche-2", "LINK": "chainlink", "DOT": "polkadot", "LTC": "litecoin",
                 "BNB": "binancecoin", "MATIC": "matic-network", "BCH": "bitcoin-cash", "ATOM": "cosmos"}
KRAKEN_BASE = {"BTC": "XBT"}
COMMODITY_METALS = {"XAU": "Gold", "XAG": "Silver", "XPT": "Platinum", "XPD": "Palladium"}


def asset_class(sym: str) -> str:
    """Classify a canonical symbol: crypto (BASE-USD), commodity (metal or =F future),
    fx (BASE=X), or equity."""
    s = sym.upper()
    if s.endswith("=F"):
        return "commodity"
    if s.endswith("=X"):
        return "fx"
    if "-" in s and s.split("-", 1)[1] in ("USD", "USDT", "USDC"):
        return "commodity" if s.split("-")[0] in COMMODITY_METALS else "crypto"
    return "equity"


def crypto_base(sym: str) -> str:
    return sym.split("-")[0].upper()


# Shared Alpha Vantage client ------------------------------------------------------

class AlphaVantage:
    """One client per process, pacing every call to at most one request per second.

    The lock is held across the pacing sleep so concurrent callers (the quote
    source and the news source) genuinely serialize instead of bursting.
    """

    def __init__(self, key: str, min_interval: float = 1.0):
        self.key = key
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next = 0.0
        self.calls = 0
        self.last_note: str | None = None
        self.premium_blocked: set[str] = set()
        self.exhausted = False          # set when the daily free-tier cap is reported

    @property
    def available(self) -> bool:
        return bool(self.key) and not self.exhausted

    @property
    def masked(self) -> str:
        return (self.key[:2] + "***" + self.key[-2:]) if len(self.key) >= 6 else "***"

    def scrub(self, text: str) -> str:
        return (text or "").replace(self.key, self.masked) if self.key else (text or "")

    async def get(self, function: str, **params) -> dict:
        if not self.key:
            raise RuntimeError("no Alpha Vantage key configured")
        if self.exhausted:
            raise AlphaVantageLimited("Alpha Vantage daily free limit reached; re-enable later or upgrade")
        async with self._lock:
            loop = asyncio.get_running_loop()
            wait = self._next - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}) as client:
                    r = await client.get("https://www.alphavantage.co/query",
                                         params={"function": function, **params, "apikey": self.key})
                    r.raise_for_status()
                    d = r.json()
            finally:
                self._next = loop.time() + self.min_interval
                self.calls += 1
        for k in ("Note", "Information", "Error Message"):
            if k in d:
                msg = self.scrub(d[k])
                self.last_note = msg
                low = d[k].lower()
                if "premium" in low:
                    self.premium_blocked.add(function)
                if "per day" in low or "25 requests" in low or "rate limit is" in low:
                    self.exhausted = True
                raise AlphaVantageLimited(msg)
        return d


class AlphaVantageLimited(Exception):
    pass


# Base source ----------------------------------------------------------------------

@dataclass
class SourceStatus:
    id: str
    name: str
    kind: str            # crypto | equity | fx | multi
    mechanism: str       # websocket | poll | simulated
    enabled: bool
    running: bool
    priority: int
    detail: str
    supported: list[str]
    owned: list[str]
    ticks: int
    last_tick: float | None
    note: str | None


class Source:
    id = "base"
    name = "Base"
    kind = "multi"
    mechanism = "poll"
    priority = 100
    classes: tuple[str, ...] = ()          # asset classes this source can serve
    poll_interval = 15.0
    fresh_after = 40.0                      # a symbol is "live" if seen within this many seconds

    def __init__(self, cfg, symbols: list[str]):
        self.cfg = cfg
        self.all_symbols = symbols
        self.symbols = [s for s in symbols if self.supports(s)]
        self.ticks = 0
        self.last_tick: float | None = None
        self.note: str | None = None

    def supports(self, sym: str) -> bool:
        return asset_class(sym) in self.classes

    async def backfill(self, symbols: list[str], cutoff: float) -> list[Tick]:
        return []

    async def run(self, emit) -> None:
        """Stream ticks for self.symbols via emit(tick). Override per source."""
        raise NotImplementedError

    def _emit(self, emit, tick: Tick) -> None:
        self.ticks += 1
        self.last_tick = time.time()
        emit(tick)

    async def _poll_loop(self, emit) -> None:
        while True:
            try:
                await self.poll_once(emit)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.note = f"{type(e).__name__}: {str(e)[:80]}"
            await asyncio.sleep(self.poll_interval)

    async def poll_once(self, emit) -> None:
        raise NotImplementedError


class YahooSource(Source):
    id = "yahoo"
    name = "Yahoo Finance"
    kind = "multi"
    mechanism = "poll"
    priority = 50
    classes = ("equity", "fx", "crypto", "commodity")
    poll_interval = 45.0
    fresh_after = 150.0

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self._seen: dict[str, float] = {}
        self.market_state: dict[str, str] = {}
        self._cooldown_until = 0.0     # set when Yahoo returns 429

    async def _chart(self, client, sym, rng, interval, retries=1):
        for attempt in range(retries + 1):
            r = await client.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                                 params={"interval": interval, "range": rng}, headers={"User-Agent": UA})
            if r.status_code == 429:
                self._cooldown_until = time.time() + 120
                if attempt < retries:
                    await asyncio.sleep(2.0)
                    continue
                raise httpx.HTTPStatusError("429 rate limited", request=r.request, response=r)
            r.raise_for_status()
            res = r.json().get("chart", {}).get("result")
            return res[0] if res else None
        return None

    async def _spark(self, client, syms):
        """One request for every symbol's recent 1-minute sparkline."""
        r = await client.get("https://query1.finance.yahoo.com/v8/finance/spark",
                             params={"symbols": ",".join(syms), "range": "1d", "interval": "1m"},
                             headers={"User-Agent": UA})
        if r.status_code == 429:
            self._cooldown_until = time.time() + 120
            raise httpx.HTTPStatusError("429 rate limited", request=r.request, response=r)
        r.raise_for_status()
        out = {}
        for row in r.json().get("spark", {}).get("result", []):
            resp = (row.get("response") or [{}])[0]
            out[row.get("symbol")] = resp
        return out

    def _sane(self, c, ref):
        return c is not None and c > 0 and (ref is None or 0.2 * ref <= c <= 5 * ref)

    def _emit_from(self, emit, sym, chart, cutoff):
        meta = chart.get("meta", {})
        self.market_state[sym] = meta.get("marketState", "")
        ref = meta.get("regularMarketPrice") or meta.get("previousClose")
        ts = chart.get("timestamp") or []
        quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
        closes, vols = quote.get("close") or [], quote.get("volume") or []
        opens, highs, lows = quote.get("open") or [], quote.get("high") or [], quote.get("low") or []
        n = 0
        for i, t in enumerate(ts):
            if t <= self._seen.get(sym, cutoff):
                continue
            c = closes[i] if i < len(closes) else None
            if not self._sane(c, ref):
                continue
            v = float(vols[i]) if i < len(vols) and vols[i] else 0.0
            c = float(c)
            o = float(opens[i]) if i < len(opens) and opens[i] is not None else c
            hi = float(highs[i]) if i < len(highs) and highs[i] is not None else c
            lo = float(lows[i]) if i < len(lows) and lows[i] is not None else c
            self._emit(emit, Tick(sym, float(t), c, v, None, o=o, h=hi, l=lo))
            self._seen[sym] = t
            n += 1
        price = meta.get("regularMarketPrice")
        if price and n == 0:
            self._emit(emit, Tick(sym, time.time(), float(price), 0.0, None, quote=True))
        return n

    def _bars_from(self, sym, resp, lo, cap=170):
        meta = resp.get("meta", {})
        self.market_state[sym] = meta.get("marketState", "")
        ref = meta.get("regularMarketPrice") or meta.get("previousClose")
        ts = resp.get("timestamp") or []
        quote = (resp.get("indicators", {}).get("quote") or [{}])[0]
        closes, vols = quote.get("close") or [], quote.get("volume") or []
        opens, highs, lows = quote.get("open") or [], quote.get("high") or [], quote.get("low") or []
        rows = []
        for i, t in enumerate(ts):
            if t < lo or i >= len(closes) or not self._sane(closes[i], ref):
                continue
            v = float(vols[i]) if i < len(vols) and vols[i] else 0.0
            c0 = float(closes[i])
            o = float(opens[i]) if i < len(opens) and opens[i] is not None else c0
            hi = float(highs[i]) if i < len(highs) and highs[i] is not None else c0
            lw = float(lows[i]) if i < len(lows) and lows[i] is not None else c0
            rows.append(Tick(sym, float(t), c0, v, None, o=o, h=hi, l=lw))
        rows = rows[-cap:]
        if rows:
            self._seen[sym] = rows[-1].ts
        return rows

    async def backfill(self, symbols: list[str], cutoff: float) -> list[Tick]:
        syms = [s for s in symbols if self.supports(s)]
        if not syms:
            return []
        sec = self.cfg.bar_seconds
        iv, rng = ("1m", "1d") if sec <= 60 else ("5m", "5d") if sec <= 900 else ("1h", "1mo")
        ticks: list[Tick] = []
        async with httpx.AsyncClient(timeout=25) as client:
            for sym in syms:
                lo = cutoff if asset_class(sym) == "crypto" else 0.0   # equities: keep the last sessions
                try:
                    chart = await self._chart(client, sym, rng, iv)
                    if chart:
                        ticks.extend(self._bars_from(sym, chart, lo))
                    await asyncio.sleep(0.35)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        if time.time() < self._cooldown_until:
            self.note = f"rate-limited by Yahoo; backing off {self._cooldown_until - time.time():.0f}s"
            return
        got = 0
        async with httpx.AsyncClient(timeout=25) as client:
            try:
                sparks = await self._spark(client, self.symbols)
                for sym, chart in sparks.items():
                    if chart:
                        got += self._emit_from(emit, sym, chart, time.time() - 3600)
            except Exception as e:  # noqa: BLE001
                self.note = f"spark: {type(e).__name__}"
                return
        states = {v for v in self.market_state.values() if v}
        self.note = ("markets " + ",".join(sorted(states)) if states else "polled") + f"; {got} new bar(s)"


class AlphaVantageQuoteSource(Source):
    id = "av_quote"
    name = "Alpha Vantage (quote)"
    kind = "equity"
    mechanism = "poll"
    priority = 70
    classes = ("equity",)
    fresh_after = 1e9  # delayed end-of-day data never goes "stale" for failover purposes

    def __init__(self, cfg, symbols, av: AlphaVantage):
        super().__init__(cfg, symbols)
        self.av = av
        self.poll_interval = max(60.0, cfg.av_quote_seconds)

    async def run(self, emit):
        if not self.av.available:
            self.note = "no API key"
            return
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        if not self.av.available:
            self.note = "daily free limit reached (25/day)" if self.av.exhausted else "no API key"
            return
        for sym in self.symbols:
            try:
                d = await self.av.get("GLOBAL_QUOTE", symbol=sym)
                q = d.get("Global Quote", {})
                price = q.get("05. price")
                if price:
                    self._emit(emit, Tick(sym, time.time(), float(price), float(q.get("06. volume") or 0.0), None, quote=True))
                    self.note = f"{sym} {q.get('07. latest trading day', '')} (delayed)"
            except AlphaVantageLimited as e:
                self.note = str(e)[:90]
                return
            except Exception as e:  # noqa: BLE001
                self.note = f"{sym}: {type(e).__name__}"


class FMPSource(Source):
    """Financial Modeling Prep: reliable 5-minute intraday history (for backfill) plus
    real-time quotes. Paid (Starter), so no free-tier throttling — this is the primary
    equity feed. The API key is never placed in notes or logs."""

    id = "fmp"
    name = "Financial Modeling Prep"
    kind = "equity"
    mechanism = "poll"
    priority = 33               # above Finnhub/Yahoo: paid, reliable history + quotes
    classes = ("equity",)
    fresh_after = 120.0
    REST = "https://financialmodelingprep.com/stable"

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self.key = cfg.fmp_key
        self.poll_interval = max(10.0, cfg.fmp_seconds)
        if not self.key:
            self.note = "no API key (set fmp.json or FLINT_FMP_KEY)"

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms or not self.key:
            return []
        from datetime import datetime, date, timedelta
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except Exception:  # noqa: BLE001
            et = None
        frm = (date.fromtimestamp(cutoff) - timedelta(days=1)).isoformat()
        to = date.fromtimestamp(time.time() + 86400).isoformat()
        ticks = []
        async with httpx.AsyncClient(timeout=25) as c:
            for sym in syms:
                try:
                    r = await c.get(f"{self.REST}/historical-chart/5min",
                                    params={"symbol": sym, "from": frm, "to": to, "apikey": self.key})
                    if r.status_code != 200:
                        self.note = f"history {r.status_code}"
                        continue
                    for row in r.json():
                        try:
                            dt = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                            ts = (dt.replace(tzinfo=et).timestamp() if et else dt.timestamp()) + self.cfg.bar_seconds
                        except (ValueError, KeyError):
                            continue
                        if row.get("close"):
                            c0 = float(row["close"])
                            ticks.append(Tick(sym, ts, c0, float(row.get("volume") or 0.0), None,
                                              o=float(row.get("open") or c0), h=float(row.get("high") or c0), l=float(row.get("low") or c0)))
                    await asyncio.sleep(0.1)
                except Exception as e:  # noqa: BLE001
                    self.note = f"backfill: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        if not self.key or not self.symbols:
            return
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols or not self.key:
            return
        n = 0
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in self.symbols:
                try:
                    r = await c.get(f"{self.REST}/quote", params={"symbol": sym, "apikey": self.key})
                    if r.status_code == 200 and r.json():
                        d = r.json()[0]
                        if d.get("price"):
                            self._emit(emit, Tick(sym, time.time(), float(d["price"]), float(d.get("volume") or 0.0), None, quote=True))
                            n += 1
                    elif r.status_code in (401, 402, 403):
                        self.note = f"quote {r.status_code} (plan?)"
                        return
                    await asyncio.sleep(0.05)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        self.note = f"{n} quotes"


class SchwabSource(Source):
    """Charles Schwab market data: real-time equity quotes and 1-minute price history.

    Needs a Schwab developer app and a one-time OAuth login (run `flint schwab-auth`).
    Market data only — no trading endpoints are used. When not authenticated the source
    is present but idle, with a status explaining what to do.
    """

    id = "schwab"
    name = "Charles Schwab"
    kind = "equity"
    mechanism = "poll"
    priority = 30              # above Yahoo (50): real-time beats delayed polling
    classes = ("equity",)
    fresh_after = 150.0

    BASE = "https://api.schwabapi.com/marketdata/v1"

    def __init__(self, cfg, symbols, auth: SchwabAuth):
        super().__init__(cfg, symbols)
        self.auth = auth
        self.poll_interval = max(4.0, cfg.schwab_seconds)
        if not auth.has_creds:
            self.note = "no app key; set schwab.json or FLINT_SCHWAB_APP_KEY/SECRET"
        elif not auth.authenticated:
            self.note = "app key set; run `flint schwab-auth` to log in"

    async def _headers(self) -> dict:
        return {"Authorization": f"Bearer {await self.auth.token()}", "Accept": "application/json"}

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms or not self.auth.authenticated:
            return []
        headers = await self._headers()
        sem = asyncio.Semaphore(10)          # paid tier, no rate-limit worry: fetch the whole universe in parallel

        async def _fetch(c, sym):
            async with sem:
                try:
                    r = await c.get(f"{self.BASE}/pricehistory", headers=headers,
                                    params={"symbol": sym, "periodType": "day", "period": "10",
                                            "frequencyType": "minute", "frequency": "5", "needExtendedHoursData": "false"})
                    if r.status_code != 200:
                        self.note = f"pricehistory {r.status_code}"
                        return []
                    out = []
                    for k in r.json().get("candles", []):
                        c0 = k.get("close")
                        kt = k.get("datetime", 0) / 1000.0
                        if c0 and kt >= cutoff:                   # respect backfill_seconds; Schwab over-fetches to cover weekends
                            c0 = float(c0)
                            out.append(Tick(sym, kt, c0, float(k.get("volume") or 0.0), None,
                                            o=float(k.get("open") or c0), h=float(k.get("high") or c0), l=float(k.get("low") or c0)))
                    return out
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
                    return []

        async with httpx.AsyncClient(timeout=25) as c:
            groups = await asyncio.gather(*[_fetch(c, sym) for sym in syms])
        return [t for g in groups for t in g]

    async def run(self, emit):
        if not self.symbols:
            return
        if not self.auth.authenticated:
            self.note = ("no app key; add schwab.json" if not self.auth.has_creds
                         else "not logged in; run `flint schwab-auth`")
            return
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols or not self.auth.authenticated:
            return
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{self.BASE}/quotes", headers=await self._headers(),
                            params={"symbols": ",".join(self.symbols), "fields": "quote"})
            if r.status_code == 401:
                self.note = "unauthorized; refresh may have expired — re-run `flint schwab-auth`"
                return
            if r.status_code != 200:
                self.note = f"quotes {r.status_code}"
                return
            data = r.json()
        n = 0
        for sym in self.symbols:
            q = (data.get(sym) or {}).get("quote") or {}
            price = q.get("lastPrice") or q.get("mark") or q.get("closePrice")
            if price:
                ts = float(q.get("quoteTime", 0) or 0) / 1000.0 or time.time()
                bid = q.get("bidPrice") or None
                ask = q.get("askPrice") or None
                self._emit(emit, Tick(sym, ts, float(price), float(q.get("totalVolume") or 0.0), None,
                                      float(bid) if bid else None, float(ask) if ask else None, quote=True))
                n += 1
        self.note = f"real-time quotes for {n} symbols"


class FinnhubSource(Source):
    """Finnhub real-time US equity data: a websocket trade stream (market hours) plus a
    quote heartbeat that keeps the last price flowing off-hours. Free tier has no
    intraday history (candles are premium), so history is left to other sources.
    The API key is never placed in notes or logs."""

    id = "finnhub"
    name = "Finnhub"
    kind = "equity"
    mechanism = "websocket"
    priority = 28              # above FMP (33): real-time trade ws beats 20s polling for live prices
    classes = ("equity",)
    fresh_after = 150.0
    WS = "wss://ws.finnhub.io"
    REST = "https://finnhub.io/api/v1"

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self.key = cfg.finnhub_key
        self.poll_interval = max(5.0, cfg.finnhub_seconds)
        if not self.key:
            self.note = "no API key (set finnhub.json or FLINT_FINNHUB_KEY)"

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms or not self.key:
            return []
        ticks = []           # one current-price seed per symbol; Yahoo provides real history when it can
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in syms:
                try:
                    r = await c.get(f"{self.REST}/quote", params={"symbol": sym, "token": self.key})
                    if r.status_code == 200 and r.json().get("c"):
                        ticks.append(Tick(sym, time.time(), float(r.json()["c"]), 0.0, None))
                    await asyncio.sleep(0.1)
                except Exception as e:  # noqa: BLE001
                    self.note = f"backfill: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        if not self.key or not self.symbols:
            return
        await asyncio.gather(self._ws(emit), self._poll(emit))

    async def _poll(self, emit):
        last = {}
        while True:
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    for sym in self.symbols:
                        r = await c.get(f"{self.REST}/quote", params={"symbol": sym, "token": self.key})
                        if r.status_code == 200:
                            d = r.json()
                            c0 = d.get("c")
                            # Only emit when the quote actually moves. Finnhub's quote "c" is frozen at the
                            # regular-session close outside market hours; re-emitting it would pin every bar's
                            # close to a stale price and corrupt the candles. Real trades still arrive on the ws.
                            if c0 and c0 != last.get(sym):
                                last[sym] = c0
                                self._emit(emit, Tick(sym, float(d.get("t") or time.time()), float(c0), 0.0, None, quote=True))
                        elif r.status_code in (401, 403):
                            self.note = f"quote unauthorized ({r.status_code}); check API key"
                        await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.note = f"poll: {type(e).__name__}"
            await asyncio.sleep(self.poll_interval)

    async def _ws(self, emit):
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(f"{self.WS}?token={self.key}", ping_interval=20, max_queue=4096) as ws:
                    for sym in self.symbols:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                    backoff = 1.0
                    async for raw in ws:
                        m = json.loads(raw)
                        if m.get("type") == "trade":
                            for t in m.get("data", []):
                                sym = t.get("s")
                                if sym in self.symbols and t.get("p"):
                                    self._emit(emit, Tick(sym, t["t"] / 1000.0, float(t["p"]), float(t.get("v") or 0.0), None))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.note = f"ws {type(e).__name__}; reconnecting"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


class EODHDSource(Source):
    """EODHD delayed real-time US equity quotes (polled). Intraday history is premium on
    the free tier, so this contributes live-ish prices as a backup to Finnhub. The API
    token is never placed in notes or logs."""

    id = "eodhd"
    name = "EODHD"
    kind = "equity"
    mechanism = "poll"
    priority = 45
    classes = ("equity",)
    fresh_after = 180.0
    REST = "https://eodhd.com/api"

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self.key = cfg.eodhd_key
        self.poll_interval = max(10.0, cfg.eodhd_seconds)
        if not self.key:
            self.note = "no API token (set eodhd.json or FLINT_EODHD_KEY)"

    def _tk(self, sym: str) -> str:
        return sym if "." in sym else f"{sym}.US"

    async def _quote(self, c, syms):
        first = self._tk(syms[0])
        params = {"api_token": self.key, "fmt": "json"}
        if len(syms) > 1:
            params["s"] = ",".join(self._tk(x) for x in syms[1:])
        r = await c.get(f"{self.REST}/real-time/{first}", params=params)
        if r.status_code != 200:
            self.note = f"quote {r.status_code}"
            return []
        d = r.json()
        return d if isinstance(d, list) else [d]

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms or not self.key:
            return []
        ticks = []
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                for row in await self._quote(c, syms):
                    sym = str(row.get("code", "")).split(".")[0]
                    if sym in syms and row.get("close") not in (None, "NA"):
                        ticks.append(Tick(sym, time.time(), float(row["close"]), 0.0, None))  # seed at now (no intraday history)
        except Exception as e:  # noqa: BLE001
            self.note = f"backfill: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        if not self.key or not self.symbols:
            return
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols or not self.key:
            return
        async with httpx.AsyncClient(timeout=20) as c:
            rows = await self._quote(c, self.symbols)
        n = 0
        for row in rows:
            sym = str(row.get("code", "")).split(".")[0]
            if sym in self.symbols and row.get("close") not in (None, "NA"):
                self._emit(emit, Tick(sym, float(row.get("timestamp") or time.time()), float(row["close"]), 0.0, None, quote=True))
                n += 1
        self.note = f"delayed quotes for {n} symbols"


class GoldApiSource(Source):
    """Spot precious-metals prices (gold, silver, platinum, palladium) from gold-api.com.
    No key. Canonical symbols XAU-USD, XAG-USD, XPT-USD, XPD-USD."""

    id = "goldapi"
    name = "Metals (gold-api)"
    kind = "commodity"
    mechanism = "poll"
    priority = 40
    classes = ("commodity",)
    poll_interval = 15.0
    fresh_after = 60.0

    def supports(self, sym):
        return crypto_base(sym) in COMMODITY_METALS

    async def _price(self, c, sym):
        r = await c.get(f"https://api.gold-api.com/price/{crypto_base(sym)}", headers={"User-Agent": UA})
        if r.status_code == 200:
            return r.json().get("price")
        return None

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        ticks = []
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in syms:
                try:
                    px = await self._price(c, sym)
                    if px:
                        ticks.append(Tick(sym, time.time(), float(px), 0.0, None))
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in self.symbols:
                try:
                    px = await self._price(c, sym)
                    if px:
                        self._emit(emit, Tick(sym, time.time(), float(px), 0.0, None, quote=True))
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        self.note = f"spot metals for {len(self.symbols)} symbols"


class SimSource(Source):
    id = "sim"
    name = "Simulator"
    kind = "crypto"
    mechanism = "simulated"
    priority = 200            # lowest: only owns a symbol when every live source is down
    classes = ("crypto", "equity", "fx", "commodity")
    fresh_after = 10.0

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self.feed = SimFeed(self.symbols) if self.symbols else None

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms:
            return []
        feed = SimFeed(syms)
        return await feed.backfill(time.time() - cutoff if cutoff < 1e6 else self.cfg.backfill_seconds)

    async def run(self, emit):
        if not self.feed:
            return
        async for tick in self.feed.stream():
            self._emit(emit, tick)


# Manager --------------------------------------------------------------------------

class IBKRSource(Source):
    """Interactive Brokers market data via a local IB Gateway / TWS (opt-in).

    Uses the ib_async library to talk to a running IB Gateway or TWS on ibkr_port,
    streaming real-time (or delayed, if the account lacks a subscription) L1 quotes and
    backfilling true intraday OHLC bars. Off by default and fails gracefully when the
    Gateway is not running, so it never blocks the other feeds. Market data only -- it
    places no orders."""
    id = "ibkr"
    name = "Interactive Brokers"
    kind = "equity"
    mechanism = "websocket"
    priority = 22                 # above FMP (33) and Yahoo (50): the user's own real-time feed
    classes = ("equity",)
    fresh_after = 30.0
    _SIZES = {30: "30 secs", 60: "1 min", 120: "2 mins", 180: "3 mins", 300: "5 mins",
              600: "10 mins", 900: "15 mins", 1800: "30 mins", 3600: "1 hour"}

    @staticmethod
    def _imp():
        try:
            from ib_async import IB, Stock
            return IB, Stock
        except ImportError:
            try:
                from ib_insync import IB, Stock
                return IB, Stock
            except ImportError:
                return None, None

    @staticmethod
    def _num(x):
        try:
            x = float(x)
            return x if x == x else None      # NaN -> None
        except (TypeError, ValueError):
            return None

    def _barsize(self):
        return self._SIZES.get(int(self.cfg.bar_seconds), "5 mins")

    def _duration(self):
        days = max(1, (int(self.cfg.backfill_seconds) + 86399) // 86400)
        return f"{days} D"

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        if not syms:
            return []
        IB, Stock = self._imp()
        if IB is None:
            self.note = "pip/uv add ib_async to use IBKR"
            return []
        ib = IB()
        try:
            await ib.connectAsync(self.cfg.ibkr_host, self.cfg.ibkr_port,
                                  clientId=self.cfg.ibkr_client_id + 1, timeout=8)
        except Exception as e:  # noqa: BLE001
            self.note = f"backfill: no Gateway at {self.cfg.ibkr_host}:{self.cfg.ibkr_port} ({type(e).__name__})"
            return []
        ticks = []
        try:
            try:
                ib.reqMarketDataType(self.cfg.ibkr_market_data_type)
            except Exception:  # noqa: BLE001
                pass
            dur, size = self._duration(), self._barsize()
            contracts = [Stock(s, "SMART", "USD") for s in syms]
            try:
                await ib.qualifyContractsAsync(*contracts)
            except Exception:  # noqa: BLE001
                pass
            for c in contracts:
                try:
                    bars = await ib.reqHistoricalDataAsync(
                        c, endDateTime="", durationStr=dur, barSizeSetting=size,
                        whatToShow="TRADES", useRTH=False, formatDate=2)
                    for b in bars:
                        ts = b.date.timestamp() if hasattr(b.date, "timestamp") else float(b.date)
                        ticks.append(Tick(c.symbol, ts + self.cfg.bar_seconds, float(b.close),
                                          float(b.volume or 0.0), None,
                                          o=float(b.open), h=float(b.high), l=float(b.low)))
                except Exception as e:  # noqa: BLE001
                    self.note = f"hist {c.symbol}: {type(e).__name__}"
        finally:
            try:
                ib.disconnect()
            except Exception:  # noqa: BLE001
                pass
        return ticks

    async def run(self, emit):
        IB, Stock = self._imp()
        if IB is None:
            self.note = "pip/uv add ib_async to use IBKR"
            return
        ib = IB()
        try:
            await ib.connectAsync(self.cfg.ibkr_host, self.cfg.ibkr_port,
                                  clientId=self.cfg.ibkr_client_id, timeout=8)
        except Exception as e:  # noqa: BLE001
            self.note = f"no IB Gateway at {self.cfg.ibkr_host}:{self.cfg.ibkr_port} -- start it and enable the API ({type(e).__name__})"
            return
        self.note = f"connected {self.cfg.ibkr_host}:{self.cfg.ibkr_port}"

        def on_err(reqId, code, msg, contract=None, *a):
            if code in (354, 10089, 10090, 10167, 10168, 10197):   # market data not subscribed / not available
                self.note = "no real-time subscription; using delayed data"
                try:
                    ib.reqMarketDataType(3)
                except Exception:  # noqa: BLE001
                    pass
        ib.errorEvent += on_err
        try:
            ib.reqMarketDataType(self.cfg.ibkr_market_data_type)
        except Exception:  # noqa: BLE001
            pass

        want = set(self.symbols)
        contracts = [Stock(s, "SMART", "USD") for s in self.symbols]
        try:
            await ib.qualifyContractsAsync(*contracts)
        except Exception as e:  # noqa: BLE001
            self.note = f"qualify failed: {type(e).__name__}"
        for c in contracts:
            try:
                ib.reqMktData(c, "", False, False)
            except Exception:  # noqa: BLE001
                pass

        def on_ticks(tickers):
            now = time.time()
            for t in tickers:
                sym = getattr(t.contract, "symbol", None)
                if sym not in want:
                    continue
                px = self._num(t.last)
                if px is None:
                    px = self._num(t.close)
                bid, ask = self._num(t.bid), self._num(t.ask)
                if px is None and bid and ask:
                    px = (bid + ask) / 2
                if px is None or px <= 0:
                    continue
                self._emit(emit, Tick(sym, now, float(px), float(self._num(t.lastSize) or 0.0),
                                      None, bid=bid, ask=ask))
        ib.pendingTickersEvent += on_ticks
        try:
            while ib.isConnected():
                await asyncio.sleep(5)
            self.note = "disconnected from IB Gateway"
        finally:
            try:
                ib.pendingTickersEvent -= on_ticks
                ib.disconnect()
            except Exception:  # noqa: BLE001
                pass


REGISTRY = [IBKRSource, SchwabSource, FMPSource, FinnhubSource, EODHDSource, GoldApiSource, YahooSource, AlphaVantageQuoteSource]


class SourceManager:
    """Owns every source, routes ticks to the active provider per symbol, and
    lets sources be toggled at runtime."""

    def __init__(self, cfg, symbols: list[str], av: AlphaVantage, on_tick, on_provider_change=None, trace=None, schwab: SchwabAuth = None, on_quote=None):
        self.cfg = cfg
        self.symbols = symbols
        self.av = av
        self.schwab = schwab
        self.on_tick = on_tick
        self.on_quote = on_quote
        self.on_provider_change = on_provider_change
        self.trace = trace
        default_off = {s.strip() for s in cfg.sources_off.split(",") if s.strip()}
        default_on = {s.strip() for s in cfg.sources_on.split(",") if s.strip()}
        self.sources: dict[str, Source] = {}
        self.enabled: dict[str, bool] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        for cls in REGISTRY:
            if cls is AlphaVantageQuoteSource:
                src = cls(cfg, symbols, av)
            elif cls is SchwabSource:
                src = cls(cfg, symbols, schwab)
            else:
                src = cls(cfg, symbols)
            self.sources[src.id] = src
            on = src.id not in default_off
            # Yahoo is the primary equity/fx feed; as a pure crypto backup it just adds load,
            # so default it on only when there are non-crypto symbols to serve.
            if src.id == "yahoo":
                on = any(asset_class(x) != "crypto" for x in symbols)
            if default_on:
                on = src.id in default_on or src.id == "sim"
            if src.id == "av_quote" and not av.key:
                on = False
            if src.id == "schwab":
                on = bool(schwab and schwab.authenticated)   # present but idle until you log in
            if src.id == "finnhub":
                on = bool(cfg.finnhub_key)
            if src.id == "eodhd":
                on = bool(cfg.eodhd_key)
            if src.id == "fmp":
                on = bool(cfg.fmp_key)
            if src.id == "ibkr":
                on = bool(cfg.ibkr_enabled)   # opt-in: needs IB Gateway running
            self.enabled[src.id] = on and bool(src.symbols)
        self.seen: dict[str, dict[str, float]] = {s: {} for s in symbols}   # sym -> source_id -> last ts
        self.active: dict[str, str | None] = {s: None for s in symbols}

    def ordered(self) -> list[Source]:
        return sorted(self.sources.values(), key=lambda s: s.priority)

    def provider_for(self, sym: str) -> str | None:
        now = time.time()
        for src in self.ordered():
            if not self.enabled.get(src.id) or not src.supports(sym):
                continue
            seen = self.seen[sym].get(src.id)
            if seen is not None and now - seen <= src.fresh_after:
                return src.id
        return None

    def _route(self, src_id: str, tick: Tick) -> None:
        sym = tick.symbol
        if sym not in self.seen:
            return
        self.seen[sym][src_id] = time.time()
        active = self.provider_for(sym)
        if active != self.active.get(sym):
            prev = self.active.get(sym)
            self.active[sym] = active
            if self.on_provider_change:
                self.on_provider_change(sym, prev, active)
        if src_id == active:
            self.on_tick(tick)
        elif self.on_quote and tick.bid and tick.ask:
            self.on_quote(tick.symbol, tick.bid, tick.ask)   # keep the spread fresh even when a trade feed owns the price

    def _emitter(self, src_id: str):
        return lambda tick: self._route(src_id, tick)

    async def backfill(self, cutoff: float) -> tuple[list[Tick], dict[str, str]]:
        """Backfill each symbol from the highest-priority enabled source that returns data."""
        active = [src for src in self.ordered()
                  if self.enabled.get(src.id) and any(src.supports(s) for s in self.symbols)]

        async def _one(src):
            want = [s for s in self.symbols if src.supports(s)]
            try:
                return src.id, await asyncio.wait_for(src.backfill(want, cutoff), timeout=90)
            except Exception as e:  # noqa: BLE001
                if self.trace:
                    self.trace.emit("feed", f"{src.name} backfill failed: {type(e).__name__}: {str(e)[:80]}", "warn")
                return src.id, []

        # query every source concurrently, then merge in priority order so startup stays fast as the universe grows
        results = dict(await asyncio.gather(*[_one(src) for src in active]))
        chosen: dict[str, str] = {}
        collected: dict[str, list[Tick]] = {s: [] for s in self.symbols}
        for src in self.ordered():
            ticks = results.get(src.id)
            if not ticks:
                continue
            by_sym: dict[str, list[Tick]] = {}
            for t in ticks:
                by_sym.setdefault(t.symbol, []).append(t)
            # keep the source with the RICHEST history per symbol (real bars beat a single seed)
            for s in self.symbols:
                if len(by_sym.get(s, [])) > len(collected[s]):
                    collected[s] = by_sym[s]
                    chosen[s] = src.id
                    self.seen[s][src.id] = time.time()
            if self.trace:
                self.trace.emit("feed", f"{src.name}: {len(ticks)} ticks for {len(by_sym)} symbols")
        merged: list[Tick] = []
        for s in self.symbols:
            merged.extend(collected[s])
            self.active[s] = chosen.get(s)
        merged.sort(key=lambda t: t.ts)
        return merged, chosen

    def start(self) -> None:
        for src in self.sources.values():
            if self.enabled.get(src.id) and src.id not in self.tasks:
                self.tasks[src.id] = asyncio.create_task(self._supervise(src))

    async def _supervise(self, src: Source) -> None:
        try:
            await src.run(self._emitter(src.id))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            src.note = f"stopped: {type(e).__name__}"
            if self.trace:
                self.trace.emit("feed", f"{src.name} stopped: {type(e).__name__}: {str(e)[:80]}", "error")

    def toggle(self, src_id: str, on: bool) -> dict:
        src = self.sources.get(src_id)
        if not src:
            return {"error": "unknown source"}
        if on and not src.symbols:
            return {"error": f"{src.name} supports none of the active symbols"}
        if on and src_id == "av_quote" and not self.av.key:
            return {"error": "no Alpha Vantage key configured"}
        if on and src_id == "schwab" and not (self.schwab and self.schwab.authenticated):
            return {"error": "Schwab not authenticated — run: flint schwab-auth"}
        if on and src_id == "finnhub" and not self.cfg.finnhub_key:
            return {"error": "no Finnhub API key configured"}
        if on and src_id == "eodhd" and not self.cfg.eodhd_key:
            return {"error": "no EODHD API token configured"}
        if on and src_id == "fmp" and not self.cfg.fmp_key:
            return {"error": "no FMP API key configured"}
        self.enabled[src_id] = on
        if on:
            if src_id not in self.tasks:
                self.tasks[src_id] = asyncio.create_task(self._supervise(src))
        else:
            task = self.tasks.pop(src_id, None)
            if task:
                task.cancel()
            for s in self.symbols:                     # drop its freshness so failover is immediate
                self.seen[s].pop(src_id, None)
        for s in self.symbols:
            self.active[s] = self.provider_for(s)
        if self.trace:
            self.trace.emit("feed", f"{src.name} turned {'on' if on else 'off'}", "act")
        return {"ok": True}

    async def stop(self) -> None:
        for task in self.tasks.values():
            task.cancel()

    def status(self) -> list[dict]:
        now = time.time()
        owned: dict[str, list[str]] = {sid: [] for sid in self.sources}
        for s in self.symbols:
            a = self.active.get(s)
            if a:
                owned[a].append(s)
        out = []
        for src in self.ordered():
            st = SourceStatus(
                id=src.id, name=src.name, kind=src.kind, mechanism=src.mechanism,
                enabled=self.enabled.get(src.id, False), running=src.id in self.tasks,
                priority=src.priority, detail=src.note or "", supported=list(src.symbols),
                owned=owned.get(src.id, []), ticks=src.ticks,
                last_tick=src.last_tick, note=src.note)
            out.append(st.__dict__)
        return out

    def provider_map(self) -> dict[str, str | None]:
        return {s: self.active.get(s) for s in self.symbols}
