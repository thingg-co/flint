"""Skim news pages with a headless browser and turn the headlines into per-asset signals.

The skimmer renders each source in headless Chromium (Playwright), pulls
headline-shaped link text out of the DOM, tags each headline with the assets it
mentions and a lexicon sentiment score, and aggregates that into a per-asset
tone and attention level. If the browser is unavailable it falls back to plain
HTTP and regex extraction.
"""
from __future__ import annotations

import asyncio
import html
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

DEFAULT_SOURCES = [
    ("CoinDesk", "https://www.coindesk.com/"),
    ("Cointelegraph", "https://cointelegraph.com/"),
    ("Google News", "https://news.google.com/search?q=bitcoin%20OR%20ethereum%20OR%20solana%20OR%20xrp%20OR%20crypto&hl=en-US&gl=US&ceid=US%3Aen"),
]

ASSET_TERMS = {
    "BTC": [r"bitcoin", r"\bbtc\b"],
    "ETH": [r"ethereum", r"\bether\b", r"\beth\b"],
    "SOL": [r"solana", r"\bsol\b"],
    "XRP": [r"\bxrp\b", r"\bripple\b"],
    "DOGE": [r"dogecoin", r"\bdoge\b"],
    "ADA": [r"cardano", r"\bada\b"],
    "AVAX": [r"avalanche", r"\bavax\b"],
    "LINK": [r"chainlink"],
    "DOT": [r"polkadot"],
    "LTC": [r"litecoin", r"\bltc\b"],
    "BNB": [r"\bbnb\b", r"binance coin"],
    "MATIC": [r"\bpolygon\b", r"\bmatic\b"],
}
GENERIC_TERMS = [r"crypto", r"cryptocurrenc", r"digital asset", r"stablecoin", r"altcoin", r"\betf\b", r"\bsec\b",
                 r"blockchain", r"\bdefi\b", r"\btoken"]

POSITIVE = ["surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps", "gain", "gains", "record", "high",
            "highs", "approve", "approves", "approval", "adoption", "inflow", "inflows", "upgrade", "breakout", "bull",
            "bullish", "rebound", "rebounds", "partnership", "launch", "launches", "buy", "buys", "buying", "accumulate",
            "milestone", "boost", "boosts", "climb", "climbs", "recover", "recovers", "recovery", "optimism", "outperform",
            "rise", "rises", "rising", "up", "growth", "wins", "win"]
NEGATIVE = ["plunge", "plunges", "crash", "crashes", "drop", "drops", "fall", "falls", "slump", "slumps", "slide", "slides",
            "hack", "hacked", "exploit", "lawsuit", "sue", "sues", "sued", "ban", "bans", "outflow", "outflows", "sell-off",
            "selloff", "liquidation", "liquidations", "bear", "bearish", "fraud", "halt", "halts", "fine", "fined", "probe",
            "delay", "delays", "dump", "dumps", "low", "lows", "decline", "declines", "loss", "losses", "warning", "warns",
            "risk", "fear", "tumble", "tumbles", "sink", "sinks", "collapse", "scam", "charges", "charged", "bankrupt",
            "bankruptcy", "down", "falling", "dip", "dips", "crackdown"]

_POS = re.compile(r"\b(" + "|".join(map(re.escape, POSITIVE)) + r")\b", re.I)
_NEG = re.compile(r"\b(" + "|".join(map(re.escape, NEGATIVE)) + r")\b", re.I)
_ASSET = {k: re.compile("|".join(v), re.I) for k, v in ASSET_TERMS.items()}
_GENERIC = re.compile("|".join(GENERIC_TERMS), re.I)


@dataclass
class Headline:
    title: str
    source: str
    url: str
    assets: list[str]
    generic: bool
    sentiment: float
    new: bool
    ts: float

    def to_json(self) -> dict:
        return dict(self.__dict__)


def score_headline(text: str) -> tuple[list[str], bool, float]:
    assets = [a for a, rx in _ASSET.items() if rx.search(text)]
    generic = bool(_GENERIC.search(text))
    pos = len(_POS.findall(text))
    neg = len(_NEG.findall(text))
    sent = (pos - neg) / (pos + neg) if pos + neg else 0.0
    return assets, generic, sent


JS_EXTRACT = r"""() => {
  const sel = 'h1 a, h2 a, h3 a, h4 a, a h1, a h2, a h3, a h4, article a, article h2, article h3, article h4, [class*="headline"] a, [class*="title"] a';
  const seen = new Set(); const out = [];
  document.querySelectorAll(sel).forEach(el => {
    const a = el.closest('a') || el.querySelector('a');
    const text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
    if (text.length < 25 || text.length > 220 || !/[a-zA-Z]{3}/.test(text)) return;
    if (seen.has(text)) return; seen.add(text);
    out.push([text, a && a.href ? a.href : '']);
  });
  return out.slice(0, 400);
}"""

_TAG = re.compile(r"<[^>]+>")
_ANCHOR = re.compile(r"<a\b[^>]*?href=[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", re.I | re.S)
_HEAD = re.compile(r"<h[1-4]\b[^>]*>(.*?)</h[1-4]>", re.I | re.S)


def _clean_text(fragment: str) -> str:
    t = html.unescape(_TAG.sub(" ", fragment))
    return re.sub(r"\s+", " ", t).strip()


def _http_candidates(page: str, base: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _ANCHOR.finditer(page):
        t = _clean_text(m.group(2))
        if 25 <= len(t) <= 220 and t not in seen and re.search(r"[a-zA-Z]{3}", t):
            seen.add(t)
            href = m.group(1)
            try:
                url = str(httpx.URL(base).join(href)) if href else ""
            except Exception:  # noqa: BLE001
                url = ""
            out.append((t, url))
    for m in _HEAD.finditer(page):
        t = _clean_text(m.group(1))
        if 25 <= len(t) <= 220 and t not in seen and re.search(r"[a-zA-Z]{3}", t):
            seen.add(t)
            out.append((t, ""))
    return out[:400]


def parse_sources(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, url = item.partition("|")
        if url:
            out.append((name.strip(), url.strip()))
    return out


class NewsSkimmer:
    id = "browser"
    name = "Headless browser"

    def __init__(self, symbols: list[str], sources_spec: str = "", use_browser: bool = True):
        self.symbols = symbols
        self.bases = {s: s.split("-")[0].upper() for s in symbols}
        self.sources = parse_sources(sources_spec) or DEFAULT_SOURCES
        self.use_browser = use_browser
        self.browser_ok: bool | None = None
        self.seen: dict[str, float] = {}
        self.status: dict[str, str] = {}
        self.method = "none"

    async def skim(self, say=None) -> dict:
        say = say or (lambda text, level="info": None)
        now = time.time()
        self.seen = {t: ts for t, ts in self.seen.items() if now - ts < 86400}
        raw: dict[str, list[tuple[str, str]]] = {}
        if self.use_browser and self.browser_ok is not False:
            try:
                raw = await self._with_browser(say)
                self.browser_ok = True
                self.method = "headless chromium"
            except Exception as e:  # noqa: BLE001
                self.browser_ok = False
                self.method = "http"
                say(f"headless browser unavailable ({type(e).__name__}: {str(e)[:80]}); using plain HTTP", "warn")
        else:
            self.method = "http"
        for name, url in self.sources:
            if raw.get(name):
                continue
            try:
                raw[name] = await self._with_http(url)
                self.status[name] = f"http: {len(raw[name])} candidates"
                say(f"{name}: fetched over HTTP, {len(raw[name])} headline candidates")
            except Exception as e:  # noqa: BLE001
                self.status[name] = f"http failed: {type(e).__name__}"
                raw[name] = []
                say(f"{name}: fetch failed ({type(e).__name__})", "warn")

        headlines: list[Headline] = []
        titles: set[str] = set()
        for name, _ in self.sources:
            for title, url in raw.get(name, []):
                key = title.lower()
                if key in titles:
                    continue
                titles.add(key)
                assets, generic, sent = score_headline(title)
                if not assets and not generic:
                    continue
                is_new = key not in self.seen
                self.seen.setdefault(key, now)
                headlines.append(Headline(title, name, url, assets, generic, sent, is_new, now))

        per_asset: dict[str, dict] = {}
        for sym, base in self.bases.items():
            direct = [h for h in headlines if base in h.assets]
            generic = [h for h in headlines if h.generic and not h.assets]
            weight = len(direct) + 0.5 * len(generic)
            sent = (sum(h.sentiment for h in direct) + 0.5 * sum(h.sentiment for h in generic)) / weight if weight else 0.0
            attention = min(1.0, math.log1p(weight) / math.log1p(120.0))
            per_asset[sym] = {
                "mentions": len(direct),
                "generic": len(generic),
                "sentiment": round(sent, 3),
                "attention": round(attention, 3),
                "top": [h.title for h in sorted(direct, key=lambda h: -abs(h.sentiment))[:3]],
            }
        return {
            "t": now,
            "method": self.method,
            "status": dict(self.status),
            "headlines": [h.to_json() for h in headlines],
            "per_asset": per_asset,
            "ideas": self._ideas(per_asset, headlines),
            "new": sum(1 for h in headlines if h.new),
        }

    def _ideas(self, per_asset: dict, headlines: list[Headline]) -> list[str]:
        ideas = []
        for sym, pa in per_asset.items():
            base = self.bases[sym]
            if pa["mentions"] == 0:
                ideas.append(f"{base}: no direct coverage this pass; general crypto tone {pa['sentiment']:+.2f}")
                continue
            s = pa["sentiment"]
            tilt = "bullish" if s > 0.15 else "bearish" if s < -0.15 else "mixed"
            lead = pa["top"][0] if pa["top"] else ""
            line = f"{base}: {pa['mentions']} headline{'s' if pa['mentions'] != 1 else ''}, tone {s:+.2f} ({tilt}), attention {pa['attention']:.2f}"
            if lead:
                line += f' | "{lead[:100]}"'
            ideas.append(line)
        gen = [h for h in headlines if h.generic and not h.assets]
        if gen:
            mean = sum(h.sentiment for h in gen) / len(gen)
            ideas.append(f"Market: {len(gen)} general crypto headlines, tone {mean:+.2f}")
        return ideas

    async def _with_browser(self, say) -> dict[str, list[tuple[str, str]]]:
        from playwright.async_api import async_playwright

        out: dict[str, list[tuple[str, str]]] = {}
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 900}, locale="en-US")
                for name, url in self.sources:
                    page = await ctx.new_page()
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        await page.wait_for_timeout(2000)
                        cands = await page.evaluate(JS_EXTRACT)
                        out[name] = [(c[0], c[1]) for c in cands]
                        self.status[name] = f"browser: {len(cands)} candidates"
                        say(f"{name}: rendered in headless chromium, {len(cands)} headline candidates")
                    except Exception as e:  # noqa: BLE001
                        self.status[name] = f"browser failed: {type(e).__name__}"
                        say(f"{name}: page failed in browser ({type(e).__name__}), will retry over HTTP", "warn")
                    finally:
                        await page.close()
            finally:
                await browser.close()
        return out

    async def _with_http(self, url: str) -> list[tuple[str, str]]:
        headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"}
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
            r = await client.get(url)
            r.raise_for_status()
            return _http_candidates(r.text, str(r.url))


class AlphaVantageNews:
    """News + sentiment from Alpha Vantage's NEWS_SENTIMENT endpoint.

    Each article carries per-ticker relevance and sentiment scores. Canonical
    symbols map to Alpha Vantage tickers (CRYPTO:BTC for crypto, the bare ticker
    for equities). Crypto and equity tickers cannot be queried together (the API
    returns nothing for a mixed list) and multiple topics are rejected, so we
    issue one unfiltered call per asset class and merge. This is a scored feed,
    generally better than the browser lexicon.
    """

    id = "alphavantage"
    name = "Alpha Vantage sentiment"

    def __init__(self, symbols: list[str], av, min_interval: float = 1800.0):
        self.symbols = symbols
        self.av = av
        self.min_interval = min_interval
        self._last_at = 0.0
        self._cache: dict | None = None
        self.tickers = {s: (f"CRYPTO:{s.split('-')[0].upper()}" if "-" in s else s.upper()) for s in symbols}
        groups: dict[str, list[str]] = {}
        for sym, tk in self.tickers.items():
            groups.setdefault("crypto" if tk.startswith("CRYPTO:") else "equity", []).append(tk)
        self.groups = groups

    async def skim(self, say=None) -> dict:
        say = say or (lambda text, level="info": None)
        now = time.time()
        if not self.av.available:
            say("Alpha Vantage news skipped: daily free limit reached (25/day)" if self.av.exhausted
                else "Alpha Vantage news skipped: no API key", "warn")
            return self._cache or {"headlines": [], "per_asset": {}, "method": "alpha vantage (unavailable)"}
        if self._cache is not None and now - self._last_at < self.min_interval:
            say(f"Alpha Vantage news cached ({(now - self._last_at) / 60:.0f} min old; refreshes every "
                f"{self.min_interval / 60:.0f} min to stay under the daily cap)")
            return self._cache
        want = {tk: sym for sym, tk in self.tickers.items()}
        articles: list[dict] = []
        seen_titles: set[str] = set()
        for cls, tks in self.groups.items():
            try:
                d = await self.av.get("NEWS_SENTIMENT", tickers=",".join(tks), sort="LATEST", limit="50")
            except Exception as e:  # noqa: BLE001
                say(f"Alpha Vantage {cls} news: {type(e).__name__}: {str(e)[:80]}", "warn")
                continue
            for art in d.get("feed", []) or []:
                key = art.get("title", "").lower()[:120]
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    articles.append(art)
        headlines: list[Headline] = []
        agg: dict[str, list[tuple[float, float]]] = {s: [] for s in self.symbols}
        for art in articles:
            tsent = {t["ticker"]: t for t in art.get("ticker_sentiment", [])}
            assets = []
            for tk, sym in want.items():
                if tk in tsent:
                    rel = float(tsent[tk].get("relevance_score", 0) or 0)
                    sc = float(tsent[tk].get("ticker_sentiment_score", 0) or 0)
                    agg[sym].append((rel, sc))
                    if rel >= 0.05:
                        assets.append(sym.split("-")[0].upper())
            try:
                ts = datetime.strptime(art.get("time_published", ""), "%Y%m%dT%H%M%S").timestamp()
            except ValueError:
                ts = now
            overall = float(art.get("overall_sentiment_score", 0) or 0)
            headlines.append(Headline(art.get("title", "")[:220], art.get("source", "AV"), art.get("url", ""),
                                      assets, not assets, overall, True, ts))
        per_asset = {}
        for sym in self.symbols:
            rows = agg[sym]
            wsum = sum(r for r, _ in rows)
            sent = sum(r * sc for r, sc in rows) / wsum if wsum else 0.0
            mentions = sum(1 for r, _ in rows if r >= 0.1)
            per_asset[sym] = {"mentions": mentions, "generic": len(rows) - mentions, "sentiment": round(sent, 3),
                              "attention": round(min(1.0, wsum / 6.0), 3),
                              "top": [h.title for h in sorted((h for h in headlines if sym.split("-")[0].upper() in h.assets),
                                                              key=lambda h: -abs(h.sentiment))[:3]]}
        say(f"Alpha Vantage: {len(articles)} scored articles across {', '.join(self.groups)} "
            f"({sum(len(v) for v in self.groups.values())} tickers)")
        result = {"headlines": [h.to_json() for h in headlines], "per_asset": per_asset, "method": "alpha vantage sentiment"}
        if articles:
            self._cache = result
            self._last_at = now
        return result


class NewsHub:
    """Holds the toggleable news sources and merges their output per asset."""

    def __init__(self, cfg, symbols: list[str], av):
        self.cfg = cfg
        self.symbols = symbols
        self.sources: dict[str, object] = {}
        self.enabled: dict[str, bool] = {}
        off = {x.strip() for x in cfg.news_off.split(",") if x.strip()}
        browser = NewsSkimmer(symbols, cfg.news_sources, cfg.news_browser)
        self.sources[browser.id] = browser
        self.enabled[browser.id] = "browser" not in off
        if av.key:
            avn = AlphaVantageNews(symbols, av, cfg.av_news_minutes * 60)
            self.sources[avn.id] = avn
            self.enabled[avn.id] = "alphavantage" not in off

    def toggle(self, src_id: str, on: bool) -> dict:
        if src_id not in self.sources:
            return {"error": "unknown news source"}
        self.enabled[src_id] = on
        return {"ok": True}

    def status(self) -> list[dict]:
        return [{"id": sid, "name": src.name, "enabled": self.enabled.get(sid, False)}
                for sid, src in self.sources.items()]

    async def skim(self, say=None) -> dict:
        say = say or (lambda text, level="info": None)
        results = []
        for sid, src in self.sources.items():
            if not self.enabled.get(sid):
                continue
            try:
                results.append((sid, await asyncio.wait_for(src.skim(say), timeout=240)))
            except Exception as e:  # noqa: BLE001
                say(f"{src.name} skim failed: {type(e).__name__}: {str(e)[:100]}", "error")
        headlines: list[dict] = []
        seen: set[str] = set()
        for _, res in results:
            for h in res.get("headlines", []):
                key = h["title"].lower()[:120]
                if key and key not in seen:
                    seen.add(key)
                    headlines.append(h)
        headlines.sort(key=lambda h: -h.get("ts", 0))
        per_asset = {}
        for sym in self.symbols:
            weighted, wsum, mentions, tops = 0.0, 0.0, 0, []
            attn = 0.0
            for _, res in results:
                pa = res.get("per_asset", {}).get(sym)
                if not pa:
                    continue
                w = 1.0 + pa["mentions"]
                weighted += pa["sentiment"] * w
                wsum += w
                mentions += pa["mentions"]
                attn = max(attn, pa["attention"])
                tops += pa.get("top", [])
            per_asset[sym] = {"mentions": mentions, "generic": 0, "sentiment": round(weighted / wsum, 3) if wsum else 0.0,
                              "attention": round(attn, 3), "top": tops[:3]}
        methods = ", ".join(r.get("method", sid) for sid, r in results) or "no sources enabled"
        ideas = self._ideas(per_asset)
        return {"t": time.time(), "method": methods, "sources": self.status(), "headlines": headlines,
                "per_asset": per_asset, "ideas": ideas, "new": sum(1 for h in headlines if h.get("new"))}

    def _ideas(self, per_asset: dict) -> list[str]:
        ideas = []
        for sym, pa in per_asset.items():
            base = sym.split("-")[0].upper()
            s = pa["sentiment"]
            tilt = "bullish" if s > 0.15 else "bearish" if s < -0.15 else "mixed"
            lead = pa["top"][0] if pa["top"] else ""
            line = f"{base}: {pa['mentions']} scored mention(s), tone {s:+.2f} ({tilt}), attention {pa['attention']:.2f}"
            if lead:
                line += f' | "{lead[:100]}"'
            ideas.append(line)
        return ideas
