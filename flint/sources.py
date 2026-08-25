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


# Crypto websocket sources ---------------------------------------------------------

class CoinbaseSource(Source):
    id = "coinbase"
    name = "Coinbase"
    kind = "crypto"
    mechanism = "websocket"
    priority = 10
    classes = ("crypto",)
    fresh_after = 15.0
    WS = "wss://ws-feed.exchange.coinbase.com"
    REST = "https://api.exchange.coinbase.com"

    def _parse(self, s: str) -> float:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    async def backfill(self, symbols: list[str], cutoff: float) -> list[Tick]:
        syms = [s for s in symbols if self.supports(s)]
        ticks: list[Tick] = []
        async with httpx.AsyncClient(base_url=self.REST, timeout=20, headers={"User-Agent": "flint/0.1"}) as client:
            for sym in syms:
                after = None
                for _ in range(self.cfg.backfill_pages):
                    params = {"limit": 1000}
                    if after:
                        params["after"] = after
                    r = await client.get(f"/products/{sym}/trades", params=params)
                    r.raise_for_status()
                    rows = r.json()
                    if not rows:
                        break
                    oldest = math.inf
                    for row in rows:
                        ts = self._parse(row["time"])
                        oldest = min(oldest, ts)
                        if ts >= cutoff:
                            ticks.append(Tick(sym, ts, float(row["price"]), float(row["size"]), row.get("side") == "sell"))
                    if oldest < cutoff:
                        break
                    after = r.headers.get("CB-AFTER") or str(min(int(x["trade_id"]) for x in rows))
                    await asyncio.sleep(0.12)
        return ticks

    async def run(self, emit) -> None:
        if not self.symbols:
            return
        backoff = 1.0
        sub = json.dumps({"type": "subscribe", "product_ids": self.symbols, "channels": ["ticker", "heartbeat"]})
        while True:
            try:
                async with websockets.connect(self.WS, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                    await ws.send(sub)
                    backoff = 1.0
                    self.note = "connected"
                    async for raw in ws:
                        m = json.loads(raw)
                        if m.get("type") != "ticker" or not m.get("price"):
                            continue
                        ts = self._parse(m["time"]) if m.get("time") else time.time()
                        side = m.get("side")
                        bid = float(m["best_bid"]) if m.get("best_bid") else None
                        ask = float(m["best_ask"]) if m.get("best_ask") else None
                        self._emit(emit, Tick(m["product_id"], ts, float(m["price"]), float(m.get("last_size") or 0.0),
                                              (side == "buy") if side else None, bid, ask))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.note = f"reconnecting: {type(e).__name__}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


class BinanceSource(Source):
    id = "binance"
    name = "Binance"
    kind = "crypto"
    mechanism = "websocket"
    priority = 20
    classes = ("crypto",)
    fresh_after = 15.0
    HOSTS = ("api.binance.com", "data-api.binance.vision")
    WS = "wss://stream.binance.com:9443/stream"

    def pair(self, sym: str) -> str:
        return crypto_base(sym) + "USDT"

    async def backfill(self, symbols: list[str], cutoff: float) -> list[Tick]:
        syms = [s for s in symbols if self.supports(s)]
        ticks: list[Tick] = []
        start = int(cutoff * 1000)
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as client:
            for sym in syms:
                host_ok = None
                cur = start
                for _ in range(self.cfg.backfill_pages):
                    data = None
                    for host in ([host_ok] if host_ok else self.HOSTS):
                        try:
                            r = await client.get(f"https://{host}/api/v3/klines",
                                                 params={"symbol": self.pair(sym), "interval": "1s", "startTime": cur, "limit": 1000})
                            if r.status_code == 200:
                                data = r.json()
                                host_ok = host
                                break
                        except Exception:  # noqa: BLE001
                            continue
                    if not data:
                        break
                    for k in data:
                        ts = k[0] / 1000.0
                        # One synthetic tick per 1s kline close; taker buy volume is field 9.
                        vol = float(k[5]); buyv = float(k[9])
                        ticks.append(Tick(sym, ts + 1.0, float(k[4]), vol, (buyv > vol / 2) if vol else None))
                    cur = data[-1][0] + 1000
                    if cur >= time.time() * 1000 or len(data) < 1000:
                        break
                    await asyncio.sleep(0.1)
        return ticks

    async def run(self, emit) -> None:
        if not self.symbols:
            return
        streams = "/".join(f"{self.pair(s).lower()}@trade" for s in self.symbols)
        url = f"{self.WS}?streams={streams}"
        rev = {self.pair(s): s for s in self.symbols}
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                    backoff = 1.0
                    self.note = "connected"
                    async for raw in ws:
                        m = json.loads(raw).get("data", {})
                        if m.get("e") != "trade":
                            continue
                        sym = rev.get(m["s"])
                        if not sym:
                            continue
                        # m["m"] is true when the buyer is the maker, i.e. the taker sold.
                        self._emit(emit, Tick(sym, m["T"] / 1000.0, float(m["p"]), float(m["q"]), not m["m"]))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.note = f"reconnecting: {type(e).__name__}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


# Poll sources ---------------------------------------------------------------------

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
        n = 0
        for i, t in enumerate(ts):
            if t <= self._seen.get(sym, cutoff):
                continue
            c = closes[i] if i < len(closes) else None
            if not self._sane(c, ref):
                continue
            v = float(vols[i]) if i < len(vols) and vols[i] else 0.0
            self._emit(emit, Tick(sym, float(t), float(c), v, None))
            self._seen[sym] = t
            n += 1
        price = meta.get("regularMarketPrice")
        if price and n == 0:
            self._emit(emit, Tick(sym, time.time(), float(price), 0.0, None))
        return n

    async def backfill(self, symbols: list[str], cutoff: float) -> list[Tick]:
        syms = [s for s in symbols if self.supports(s)]
        ticks: list[Tick] = []
        cap = 240  # most recent bars to keep per symbol
        async with httpx.AsyncClient(timeout=25) as client:
            for sym in syms:
                # Crypto trades 24/7 so the tight window applies; equities/fx may be closed,
                # so take the last available session instead of an empty recent window.
                rng = "1d"
                lo = cutoff if asset_class(sym) == "crypto" else 0.0
                try:
                    chart = await self._chart(client, sym, rng, "1m")
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
                    continue
                if not chart:
                    continue
                meta = chart.get("meta", {})
                self.market_state[sym] = meta.get("marketState", "")
                ref = meta.get("regularMarketPrice") or meta.get("previousClose")
                ts = chart.get("timestamp") or []
                quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
                closes, vols = quote.get("close") or [], quote.get("volume") or []
                rows = []
                for i, t in enumerate(ts):
                    if t < lo or i >= len(closes) or not self._sane(closes[i], ref):
                        continue
                    v = float(vols[i]) if i < len(vols) and vols[i] else 0.0
                    rows.append(Tick(sym, float(t), float(closes[i]), v, None))
                rows = rows[-cap:]
                if rows:
                    self._seen[sym] = rows[-1].ts
                ticks.extend(rows)
                await asyncio.sleep(0.2)
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


class KrakenSource(Source):
    id = "kraken"
    name = "Kraken"
    kind = "crypto"
    mechanism = "poll"
    priority = 40
    classes = ("crypto",)
    poll_interval = 12.0
    fresh_after = 40.0

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self._seen: dict[str, float] = {}

    def pair(self, sym: str) -> str:
        b = crypto_base(sym)
        return KRAKEN_BASE.get(b, b) + "USD"

    async def _ohlc(self, client, sym, since=None):
        params = {"pair": self.pair(sym), "interval": 1}
        if since:
            params["since"] = int(since)
        r = await client.get("https://api.kraken.com/0/public/OHLC", params=params, headers={"User-Agent": UA})
        r.raise_for_status()
        res = r.json().get("result", {})
        rows = next((v for k, v in res.items() if k != "last"), [])
        return rows

    async def backfill(self, symbols, cutoff):
        syms = [s for s in symbols if self.supports(s)]
        ticks = []
        async with httpx.AsyncClient(timeout=20) as client:
            for sym in syms:
                try:
                    for row in await self._ohlc(client, sym, since=cutoff):
                        t = float(row[0])
                        if t >= cutoff:
                            ticks.append(Tick(sym, t + 60.0, float(row[4]), float(row[6]), None))
                            self._seen[sym] = t
                    await asyncio.sleep(0.3)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        async with httpx.AsyncClient(timeout=20) as client:
            for sym in self.symbols:
                try:
                    rows = await self._ohlc(client, sym, since=self._seen.get(sym, time.time() - 300))
                    for row in rows:
                        t = float(row[0])
                        if t > self._seen.get(sym, 0):
                            self._emit(emit, Tick(sym, t + 60.0, float(row[4]), float(row[6]), None))
                            self._seen[sym] = t
                    await asyncio.sleep(0.3)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"


class CoinGeckoSource(Source):
    id = "coingecko"
    name = "CoinGecko"
    kind = "crypto"
    mechanism = "poll"
    priority = 60
    classes = ("crypto",)
    poll_interval = 20.0
    fresh_after = 60.0

    def supports(self, sym):
        return asset_class(sym) == "crypto" and crypto_base(sym) in COINGECKO_IDS

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        ids = ",".join(COINGECKO_IDS[crypto_base(s)] for s in self.symbols)
        rev = {COINGECKO_IDS[crypto_base(s)]: s for s in self.symbols}
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as client:
            r = await client.get("https://api.coingecko.com/api/v3/simple/price",
                                 params={"ids": ids, "vs_currencies": "usd", "include_last_updated_at": "true"})
            r.raise_for_status()
            d = r.json()
        for cid, row in d.items():
            sym = rev.get(cid)
            if sym and row.get("usd"):
                self._emit(emit, Tick(sym, float(row.get("last_updated_at", time.time())), float(row["usd"]), 0.0, None))
        self.note = f"polled {len(d)} prices"


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
                    self._emit(emit, Tick(sym, time.time(), float(price), float(q.get("06. volume") or 0.0), None))
                    self.note = f"{sym} {q.get('07. latest trading day', '')} (delayed)"
            except AlphaVantageLimited as e:
                self.note = str(e)[:90]
                return
            except Exception as e:  # noqa: BLE001
                self.note = f"{sym}: {type(e).__name__}"


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
        ticks = []
        async with httpx.AsyncClient(timeout=25) as c:
            for sym in syms:
                try:
                    r = await c.get(f"{self.BASE}/pricehistory", headers=await self._headers(),
                                    params={"symbol": sym, "periodType": "day", "period": "1",
                                            "frequencyType": "minute", "frequency": "1", "needExtendedHoursData": "false"})
                    if r.status_code != 200:
                        self.note = f"pricehistory {r.status_code}"
                        continue
                    for k in r.json().get("candles", []):
                        c0 = k.get("close")
                        if c0:
                            ticks.append(Tick(sym, k["datetime"] / 1000.0, float(c0), float(k.get("volume") or 0.0), None))
                    await asyncio.sleep(0.2)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        return ticks

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
                                      float(bid) if bid else None, float(ask) if ask else None))
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
    priority = 35
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
        ticks = []
        async with httpx.AsyncClient(timeout=15) as c:
            for sym in syms:
                try:
                    r = await c.get(f"{self.REST}/quote", params={"symbol": sym, "token": self.key})
                    if r.status_code == 200 and r.json().get("c"):
                        d = r.json()
                        ticks.append(Tick(sym, float(d.get("t") or time.time()), float(d["c"]), 0.0, None))
                    await asyncio.sleep(0.2)
                except Exception as e:  # noqa: BLE001  (never interpolate the URL/key)
                    self.note = f"backfill: {type(e).__name__}"
        return ticks

    async def run(self, emit):
        if not self.key or not self.symbols:
            return
        await asyncio.gather(self._ws(emit), self._poll(emit))

    async def _poll(self, emit):
        while True:
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    for sym in self.symbols:
                        r = await c.get(f"{self.REST}/quote", params={"symbol": sym, "token": self.key})
                        if r.status_code == 200:
                            d = r.json()
                            if d.get("c"):
                                self._emit(emit, Tick(sym, float(d.get("t") or time.time()), float(d["c"]), 0.0, None))
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
                        ticks.append(Tick(sym, float(row.get("timestamp") or time.time()), float(row["close"]), 0.0, None))
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
                self._emit(emit, Tick(sym, float(row.get("timestamp") or time.time()), float(row["close"]), 0.0, None))
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
                        self._emit(emit, Tick(sym, time.time(), float(px), 0.0, None))
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{sym}: {type(e).__name__}"
        self.note = f"spot metals for {len(self.symbols)} symbols"


class BitfinexSource(Source):
    """Bitfinex spot ticker (poll). Crypto, no key."""

    id = "bitfinex"
    name = "Bitfinex"
    kind = "crypto"
    mechanism = "poll"
    priority = 24
    classes = ("crypto",)
    poll_interval = 8.0
    fresh_after = 30.0

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as c:
            for sym in self.symbols:
                try:
                    r = await c.get(f"https://api-pub.bitfinex.com/v2/ticker/t{crypto_base(sym)}USD")
                    if r.status_code == 200:
                        d = r.json()
                        if isinstance(d, list) and len(d) > 7:
                            self._emit(emit, Tick(sym, time.time(), float(d[6]), float(d[7]), None))
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{crypto_base(sym)}: {type(e).__name__}"


class GeminiSource(Source):
    """Gemini spot ticker (poll). Crypto, no key."""

    id = "gemini"
    name = "Gemini"
    kind = "crypto"
    mechanism = "poll"
    priority = 26
    classes = ("crypto",)
    poll_interval = 8.0
    fresh_after = 30.0

    async def run(self, emit):
        await self._poll_loop(emit)

    async def poll_once(self, emit):
        if not self.symbols:
            return
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as c:
            for sym in self.symbols:
                try:
                    r = await c.get(f"https://api.gemini.com/v1/pubticker/{crypto_base(sym).lower()}usd")
                    if r.status_code == 200 and r.json().get("last"):
                        self._emit(emit, Tick(sym, time.time(), float(r.json()["last"]), 0.0, None))
                    await asyncio.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    self.note = f"{crypto_base(sym)}: {type(e).__name__}"


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

REGISTRY = [CoinbaseSource, BinanceSource, BitfinexSource, GeminiSource, KrakenSource, SchwabSource, FinnhubSource,
            EODHDSource, GoldApiSource, YahooSource, CoinGeckoSource, AlphaVantageQuoteSource, SimSource]


class SourceManager:
    """Owns every source, routes ticks to the active provider per symbol, and
    lets sources be toggled at runtime."""

    def __init__(self, cfg, symbols: list[str], av: AlphaVantage, on_tick, on_provider_change=None, trace=None, schwab: SchwabAuth = None):
        self.cfg = cfg
        self.symbols = symbols
        self.av = av
        self.schwab = schwab
        self.on_tick = on_tick
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
            if src.id == "sim":
                on = True  # always available as the fallback provider
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

    def _emitter(self, src_id: str):
        return lambda tick: self._route(src_id, tick)

    async def backfill(self, cutoff: float) -> tuple[list[Tick], dict[str, str]]:
        """Backfill each symbol from the highest-priority enabled source that returns data."""
        chosen: dict[str, str] = {}
        collected: dict[str, list[Tick]] = {s: [] for s in self.symbols}
        remaining = set(self.symbols)
        for src in self.ordered():
            if not remaining or not self.enabled.get(src.id):
                continue
            want = [s for s in remaining if src.supports(s)]
            if not want:
                continue
            try:
                ticks = await asyncio.wait_for(src.backfill(want, cutoff), timeout=90)
            except Exception as e:  # noqa: BLE001
                if self.trace:
                    self.trace.emit("feed", f"{src.name} backfill failed: {type(e).__name__}: {str(e)[:80]}", "warn")
                continue
            by_sym: dict[str, list[Tick]] = {}
            for t in ticks:
                by_sym.setdefault(t.symbol, []).append(t)
            for s in want:
                if by_sym.get(s):
                    collected[s] = by_sym[s]
                    chosen[s] = src.id
                    remaining.discard(s)
                    self.seen[s][src.id] = time.time()
            if ticks and self.trace:
                covered = ", ".join(sorted({t.symbol for t in ticks}))
                self.trace.emit("feed", f"{src.name}: backfilled {len(ticks)} ticks for {covered}")
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
        if src_id == "sim" and not on:
            return {"error": "the simulator is the fallback provider and cannot be turned off"}
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
