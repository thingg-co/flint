"""Dated company news, so paged history carries the news features the live skim gives.

`NewsHistory` keeps, per name, the timestamps and tones of its headlines over the backfill
range (Finnhub /company-news by date, scored like the live skim: FinBERT when available, the
word lists otherwise), persisted in `state/news_history.json`. `at(ts)` answers what the live
skim would have said at that bar: a name's tone is its own headlines of the last three days
(the eight newest, as the skim keeps) with the market tone, here the mean tone of every name's
headlines in the window, as one headline of prior; attention is the square root of its share
of the window's most-covered name. The fetch keeps the six newest headlines per name per day,
so a heavily covered name stays within the skim's scale and the store stays small.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np

from .news import score_headline

log = logging.getLogger(__name__)

KEEP_S = 3 * 86400                      # the skim's horizon
PER_PASS = 8                            # headlines per name the skim keeps per pass
PER_DAY = 6                             # headlines per name per day the store keeps
SLICE_DAYS = 10                         # the span one pass fetches for every name: the deep loop's chunk, so paging can follow the news
CAP_HINT = 200                          # Finnhub answers with about 244 of the newest items at most: an answer this long was probably cut


class NewsHistory:
    def __init__(self, path: str | None = None):
        self.path = path
        self.ts: dict[str, np.ndarray] = {}
        self.tone: dict[str, np.ndarray] = {}
        self.covered: dict[str, list[list[float]]] = {}
        self.status = "empty"
        self.fetched_at: float | None = None
        self._busy = False

    # -- persistence -----------------------------------------------------------------------
    def load(self) -> bool:
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                d = json.load(f)
            for sym, v in (d.get("symbols") or {}).items():
                self.ts[sym] = np.asarray(v.get("ts") or [], dtype=np.float64)
                self.tone[sym] = np.asarray(v.get("tone") or [], dtype=np.float64)
                self.covered[sym] = [[float(a), float(b)] for a, b in (v.get("covered") or [])]
            self.fetched_at = d.get("fetched_at")
            self.status = f"{sum(len(a) for a in self.ts.values())} headlines for {len(self.ts)} names"
            return True
        except (OSError, ValueError, TypeError) as e:
            log.warning("news history unreadable (%s); starting empty", e)
            return False

    def save(self) -> None:
        if not self.path:
            return
        d = {"v": 1, "fetched_at": self.fetched_at,
             "symbols": {s: {"ts": [round(float(t), 0) for t in self.ts[s]], "tone": [round(float(x), 3) for x in self.tone[s]],
                             "covered": self.covered.get(s, [])} for s in self.ts}}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)

    # -- coverage ----------------------------------------------------------------------------
    def _merge(self, sym: str, lo: float, hi: float) -> None:
        spans = sorted(self.covered.get(sym, []) + [[float(lo), float(hi)]])
        out: list[list[float]] = []
        for a, b in spans:
            if out and a <= out[-1][1] + 60:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        self.covered[sym] = out

    def covers(self, symbols, lo: float, hi: float) -> bool:
        """Whether every name's fetched spans contain [lo, hi]."""
        for s in symbols:
            if not any(a <= lo and b >= hi for a, b in self.covered.get(s, [])):
                return False
        return True

    def missing(self, symbols, lo: float, hi: float) -> list[str]:
        return [s for s in symbols if not self.covers([s], lo, hi)]

    def add(self, sym: str, rows: list[tuple[float, float]], lo: float, hi: float) -> int:
        """Merge (ts, tone) rows for a name, keeping the PER_DAY newest per day, and record the span."""
        old = list(zip(self.ts.get(sym, np.zeros(0)).tolist(), self.tone.get(sym, np.zeros(0)).tolist()))
        keep = [(t, x) for t, x in old if not (lo <= t <= hi)] + [(float(t), float(x)) for t, x in rows if lo <= t <= hi]
        by_day: dict[int, list[tuple[float, float]]] = {}
        for t, x in keep:
            by_day.setdefault(int(t // 86400), []).append((t, x))
        rows_out = []
        for d in sorted(by_day):
            rows_out.extend(sorted(by_day[d])[-PER_DAY:])
        rows_out.sort()
        self.ts[sym] = np.asarray([t for t, _ in rows_out], dtype=np.float64)
        self.tone[sym] = np.asarray([x for _, x in rows_out], dtype=np.float64)
        self._merge(sym, lo, hi)
        return len(rows_out)

    # -- the features at a time ------------------------------------------------------------
    def at(self, ts: float, symbols) -> tuple[np.ndarray, np.ndarray]:
        """(sentiment, attention) per name as the live skim would have reported at `ts`."""
        n = len(symbols)
        counts = np.zeros(n, dtype=np.int64)
        sums = np.zeros(n)
        for i, s in enumerate(symbols):
            a = self.ts.get(s)
            if a is None or not len(a):
                continue
            i0 = int(np.searchsorted(a, ts - KEEP_S, side="right"))
            i1 = int(np.searchsorted(a, ts, side="right"))
            if i1 <= i0:
                continue
            direct = self.tone[s][max(i0, i1 - PER_PASS):i1]
            counts[i] = len(direct)
            sums[i] = float(direct.sum())
        total = int(counts.sum())
        market = float(sums.sum() / total) if total else 0.0
        sent = np.where(counts > 0, (sums + market) / (counts + 1.0), market)
        top = int(counts.max()) if n else 0
        attn = np.sqrt(counts / top) if top else np.zeros(n)
        return np.clip(sent, -1, 1), np.clip(attn, 0, 1)

    # -- the fetch ---------------------------------------------------------------------------
    async def fetch(self, symbols, lo: float, hi: float, key: str, scorer=None, say=None, rate_s: float = 1.25,
                    client=None, deadline_s: float | None = None) -> dict:
        """Fill the store for [lo, hi], newest slice first and every name per slice, so the deep
        loop can page a chunk as soon as its slice is in. One request per name per slice, paced
        under Finnhub's limit next to the live skim; a capped answer is paged backwards by its
        oldest item; a 429 waits a minute. Titles are scored off the loop. Returns a summary;
        progress goes to `say`."""
        import httpx  # noqa: PLC0415

        if self._busy:
            return {"error": "already fetching"}
        self._busy = True
        t0 = time.time()
        names = list(symbols)
        slices: list[tuple[float, float]] = []
        end = hi
        while end > lo:
            start = max(lo, end - SLICE_DAYS * 86400)
            slices.append((start, end))
            end = start
        asked = done = fetched = requests = 0
        failed: set[str] = set()
        try:
            own = client is None
            c = client or httpx.AsyncClient(timeout=20)
            try:
                for si, (start, end) in enumerate(slices):
                    if deadline_s is not None and time.time() - t0 > deadline_s:
                        break
                    todo = self.missing(names, start, end)
                    asked += len(todo)
                    for sym in todo:
                        rows = await self._slice(c, sym, start, end, key, rate_s)
                        requests += 1
                        if rows is None:
                            failed.add(sym)
                            continue
                        titles = [t for _, t in rows]
                        tones = await asyncio.to_thread(self._score, scorer, titles) if titles else []
                        self.add(sym, [(ts, x) for (ts, _), x in zip(rows, tones)], start, end)
                        fetched += len(rows)
                        done += 1
                    if todo:
                        self.fetched_at = time.time()
                        self.save()
                        if say:
                            say(f"news history: {datetime.fromtimestamp(start):%b %d} – {datetime.fromtimestamp(end):%b %d} in "
                                f"({si + 1}/{len(slices)} slices, {fetched} headlines, {len(failed)} names failing)")
            finally:
                if own:
                    await c.aclose()
        finally:
            self._busy = False
            self.fetched_at = time.time()
            try:
                self.save()
            except OSError as e:
                log.warning("could not save the news history: %s", e)
        self.status = (f"{sum(len(a) for a in self.ts.values())} headlines for {len(self.ts)} names"
                       + (f"; {len(failed)} names failed" if failed else ""))
        return {"names": asked, "done": done, "headlines": fetched, "requests": requests, "failed": sorted(failed),
                "elapsed_s": time.time() - t0}

    async def _slice(self, c, sym: str, start: float, end: float, key: str, rate_s: float) -> list[tuple[float, str]] | None:
        """The (ts, title) pairs Finnhub files under `sym` in [start, end], or None on failure.
        Finnhub answers with its newest items only, about 244 of them, so an answer that long
        which does not reach `start` is followed by another for the days before its oldest item."""
        out: list[tuple[float, str]] = []
        hi = end
        for _page in range(60):
            items = await self._request(c, sym, start, hi, key, rate_s)
            if items is None:
                return None
            rows = []
            for it in items:
                title = html.unescape(str(it.get("headline") or "")).strip()
                ts = float(it.get("datetime") or 0)
                if len(title) >= 12 and start <= ts <= hi:
                    rows.append((ts, title))
            out.extend(rows)
            oldest = min((ts for ts, _ in rows), default=start)
            if len(items) < CAP_HINT or oldest <= start + 60 or not rows:
                return out
            hi = oldest - 1                                                   # the cut answer covered [oldest, hi]: ask for what is older
        return out

    async def _request(self, c, sym: str, start: float, end: float, key: str, rate_s: float) -> list | None:
        frm = datetime.fromtimestamp(start, timezone.utc).date().isoformat()
        to = datetime.fromtimestamp(end, timezone.utc).date().isoformat()
        for _attempt in range(3):
            try:
                r = await c.get("https://finnhub.io/api/v1/company-news", params={"symbol": sym, "from": frm, "to": to, "token": key})
            except Exception as e:  # noqa: BLE001
                log.debug("news history %s %s-%s: %s", sym, frm, to, e)
                await asyncio.sleep(rate_s * 4)
                continue
            await asyncio.sleep(rate_s)
            if r.status_code == 429:
                await asyncio.sleep(61.0)
                continue
            if r.status_code != 200:
                return None
            try:
                items = r.json()
            except ValueError:
                return None
            return items if isinstance(items, list) else None
        return None

    @staticmethod
    def _score(scorer, titles: list[str]) -> list[float]:
        if scorer is not None:
            try:
                return [float(x) for x in scorer.score(titles)]
            except Exception as e:  # noqa: BLE001
                log.warning("news history: scorer failed (%s); word lists instead", e)
        return [score_headline(t)[2] for t in titles]


def backfill_span(now: float, days: float) -> tuple[float, float]:
    """The [lo, hi] the deep backfill walks, with the skim's horizon in front so the first bars
    see their trailing three days of news."""
    return now - days * 86400.0 - KEEP_S, now

