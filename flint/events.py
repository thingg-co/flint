"""A dated event calendar, so paged history and the live feed carry the same event features.

`EarningsCalendar` keeps, per session date, the names that report that day and the dividends
going ex that day (rate and announcement date), from Nasdaq's public calendars (two requests per
day, no key, historical as far back as asked, where Finnhub's free earnings calendar reaches only
a few weeks back and its dividend endpoints are premium), persisted in `state/events.json`.
`at(ts)` answers per name as of a bar's session date: proximity to the next report,
exp(-days / DECAY_DAYS), 1 on the day itself and 0 beyond NEAR_DAYS; the same decay since the
last one; and the dividend about to come off the price, its rate weighted by exp(-(sessions
ahead - 1)), so the session before an ex-date reads the full rate and the ex-date itself 0 (the
drop is in the overnight leg by then), only once the dividend was announced. The report's hour
is not used: Nasdaq leaves it unsupplied for most names, so the day itself reads 1 on both
earnings flags and the session clock tells the net the rest. A report date counts as known once
it is on the calendar, which for most names is weeks ahead, so a bar a month before sees it coming.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .features import NY

log = logging.getLogger(__name__)

NEAR_DAYS = 30                          # a report further off than this reads 0
DECAY_DAYS = 3.0                        # e-folding of both flags, in calendar days
AHEAD_S = 45 * 86400.0                  # the forward window the live fetch keeps current
URL = "https://api.nasdaq.com/api/calendar/earnings"
DIV_URL = "https://api.nasdaq.com/api/calendar/dividends"
DIV_SESSIONS = 5                        # an ex-date further off than this reads 0
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
           "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/market-activity/earnings"}


def _day_of(ts: float) -> date:
    return datetime.fromtimestamp(ts, timezone.utc).date()


class EarningsCalendar:
    def __init__(self, path: str | None = None):
        self.path = path
        self.days: dict[str, list[str]] = {}            # ISO date -> names reporting that day
        self.divs: dict[str, list[list]] = {}            # ISO ex-date -> [[name, rate, ISO announcement date or ""], ...]
        self.covered: set[str] = set()                   # ISO dates fetched, both calendars (an empty day counts)
        self.dates: dict[str, np.ndarray] = {}           # name -> sorted ordinals of its report dates
        self.ex: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # name -> (ex ordinals, rates, announced ordinals), sorted by ex
        self.fetched_at: float | None = None
        self.status = "empty"
        self._busy = False
        self._cache: tuple[int, int, np.ndarray, np.ndarray] | None = None

    # -- persistence -----------------------------------------------------------------------
    def load(self) -> bool:
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.days = {str(k): [str(s) for s in v] for k, v in (d.get("days") or {}).items()}
            self.divs = {str(k): [[str(a), float(b), str(c)] for a, b, c in v] for k, v in (d.get("divs") or {}).items()}
            self.covered = {str(k) for k in (d.get("covered") or [])} | set(self.days) | set(self.divs)
            if int(d.get("v") or 1) < 2:                                     # before the dividends: fetch those days again
                self.covered = set()
            self.fetched_at = d.get("fetched_at")
            self._reindex()
            return True
        except (OSError, ValueError, TypeError) as e:
            log.warning("earnings calendar unreadable (%s); starting empty", e)
            return False

    def save(self) -> None:
        if not self.path:
            return
        d = {"v": 2, "fetched_at": self.fetched_at, "days": {k: self.days[k] for k in sorted(self.days)},
             "divs": {k: self.divs[k] for k in sorted(self.divs)}, "covered": sorted(self.covered)}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)

    def _reindex(self) -> None:
        by: dict[str, list[int]] = {}
        for k, names in self.days.items():
            o = date.fromisoformat(k).toordinal()
            for s in names:
                by.setdefault(s, []).append(o)
        self.dates = {s: np.asarray(sorted(set(v)), dtype=np.int64) for s, v in by.items()}
        dv: dict[str, list[tuple[int, float, int]]] = {}
        for k, rows in self.divs.items():
            o = date.fromisoformat(k).toordinal()
            for s, rate, ann in rows:
                a = date.fromisoformat(ann).toordinal() if ann else o - 7      # unannounced: known a week ahead, Nasdaq's usual lead at the least
                dv.setdefault(s, []).append((o, float(rate), a))
        self.ex = {}
        for s, v in dv.items():
            v.sort()
            self.ex[s] = (np.asarray([x[0] for x in v], dtype=np.int64), np.asarray([x[1] for x in v]), np.asarray([x[2] for x in v], dtype=np.int64))
        self._cache = None
        self.status = (f"{sum(len(v) for v in self.days.values())} reports and {sum(len(v) for v in self.divs.values())} dividends "
                       f"on {len(self.covered)} days")

    # -- coverage ----------------------------------------------------------------------------
    @staticmethod
    def weekdays(lo: float, hi: float) -> list[date]:
        """The weekdays (UTC dates) in [lo, hi]."""
        d, end = _day_of(lo), _day_of(hi)
        out = []
        while d <= end:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out

    def missing(self, lo: float, hi: float) -> list[date]:
        return [d for d in self.weekdays(lo, hi) if d.isoformat() not in self.covered]

    def covers(self, lo: float, hi: float) -> bool:
        return not self.missing(lo, hi)

    def add(self, day: date, symbols, dividends=None) -> None:
        """Record the names reporting on `day` and the dividends going ex that day, as
        (name, rate, announcement date or None) rows; empty lists still mark the day fetched."""
        k = day.isoformat()
        names = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        if names:
            self.days[k] = names
        else:
            self.days.pop(k, None)
        rows = []
        for s, rate, ann in (dividends or []):
            s = str(s).strip().upper()
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                continue
            if s and rate > 0:
                rows.append([s, rate, ann.isoformat() if isinstance(ann, date) else (str(ann) if ann else "")])
        if rows:
            self.divs[k] = sorted(rows)
        else:
            self.divs.pop(k, None)
        self.covered.add(k)
        self._reindex()

    def forget(self, since: date) -> None:
        """Drop the days from `since` on, so a refresh fetches them again (a date can move)."""
        k0 = since.isoformat()
        self.covered = {k for k in self.covered if k < k0}
        self.days = {k: v for k, v in self.days.items() if k < k0}
        self.divs = {k: v for k, v in self.divs.items() if k < k0}
        self._reindex()

    # -- the flags at a time -----------------------------------------------------------------
    @staticmethod
    def sessions_between(today: int, later: int) -> int:
        """Weekdays after `today` up to and including `later` (ordinals): 1 for the next session."""
        n = 0
        for o in range(today + 1, later + 1):
            if date.fromordinal(o).weekday() < 5:
                n += 1
        return n

    def at(self, ts: float, symbols) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(near, post, dividend) per name as of the bar's ET session date: the earnings flags and
        the announced dividend rate about to come off the price, weighted by exp(-(sessions ahead - 1))."""
        today = datetime.fromtimestamp(ts, NY).date().toordinal()
        key = hash(tuple(symbols))
        if self._cache is not None and self._cache[0] == today and self._cache[1] == key:
            return self._cache[2]
        n = len(symbols)
        near = np.zeros(n)
        post = np.zeros(n)
        div = np.zeros(n)
        for i, s in enumerate(symbols):
            a = self.dates.get(s)
            if a is not None and len(a):
                j = int(np.searchsorted(a, today, side="left"))
                if j < len(a):
                    ahead = int(a[j]) - today
                    if ahead <= NEAR_DAYS:
                        near[i] = math.exp(-ahead / DECAY_DAYS)
                k = int(np.searchsorted(a, today, side="right")) - 1
                if k >= 0:
                    since = today - int(a[k])
                    if since <= NEAR_DAYS:
                        post[i] = math.exp(-since / DECAY_DAYS)
            e = self.ex.get(s)
            if e is not None and len(e[0]):
                xs, rates, anns = e
                j = int(np.searchsorted(xs, today, side="right"))               # the first ex-date after today
                while j < len(xs) and anns[j] > today:                         # not announced yet as of this bar: not knowable
                    j += 1
                if j < len(xs):
                    k = self.sessions_between(today, int(xs[j]))
                    if 1 <= k <= DIV_SESSIONS:
                        div[i] = float(rates[j]) * math.exp(-(k - 1))
        out = (near, post, div)
        self._cache = (today, key, out)
        return out

    # -- the fetch ---------------------------------------------------------------------------
    async def fetch(self, lo: float, hi: float, say=None, rate_s: float = 0.75, client=None,
                    deadline_s: float | None = None, refresh_from: date | None = None) -> dict:
        """Fill the calendar for the weekdays of [lo, hi] not yet fetched, newest first (the deep
        loop pages the newest chunk first), one request per day, paced. `refresh_from` forgets
        the days from that date on first, so a live refresh picks up moved and newly confirmed
        dates. Returns a summary; progress goes to `say`."""
        import httpx  # noqa: PLC0415

        if self._busy:
            return {"error": "already fetching"}
        self._busy = True
        t0 = time.time()
        if refresh_from is not None:
            self.forget(refresh_from)
        todo = sorted(self.missing(lo, hi), reverse=True)
        done = failed = names = 0
        try:
            own = client is None
            c = client or httpx.AsyncClient(timeout=20, headers=HEADERS)
            try:
                for i, day in enumerate(todo):
                    if deadline_s is not None and time.time() - t0 > deadline_s:
                        break
                    rows = await self._request(c, day, rate_s)
                    divs = await self._request(c, day, rate_s, dividends=True) if rows is not None else None
                    if rows is None or divs is None:
                        failed += 1
                        continue
                    self.add(day, rows, divs)
                    names += len(rows) + len(divs)
                    done += 1
                    if done % 25 == 0:
                        self.fetched_at = time.time()
                        self.save()
                        if say:
                            say(f"earnings calendar: {done}/{len(todo)} days in, back to {day:%b %d}")
            finally:
                if own:
                    await c.aclose()
        finally:
            self._busy = False
            self.fetched_at = time.time()
            try:
                self.save()
            except OSError as e:
                log.warning("could not save the earnings calendar: %s", e)
        return {"days": len(todo), "done": done, "names": names, "failed": failed, "elapsed_s": time.time() - t0}

    async def _request(self, c, day: date, rate_s: float, dividends: bool = False) -> list | None:
        """The names Nasdaq lists as reporting on `day`, or with `dividends` the (name, rate,
        announcement date) rows going ex that day; [] for a day with none, None on failure."""
        for _attempt in range(3):
            try:
                r = await c.get(DIV_URL if dividends else URL, params={"date": day.isoformat()}, headers=HEADERS)
            except Exception as e:  # noqa: BLE001
                log.debug("earnings calendar %s: %s", day, e)
                await asyncio.sleep(rate_s * 4)
                continue
            await asyncio.sleep(rate_s)
            if r.status_code == 429:
                await asyncio.sleep(30.0)
                continue
            if r.status_code != 200:
                return None
            try:
                d = r.json()
            except ValueError:
                return None
            data = d.get("data") if isinstance(d, dict) else None
            if dividends and isinstance(data, dict):
                data = data.get("calendar")
            rows = (data or {}).get("rows") if isinstance(data, dict) else None
            if rows is None:
                return [] if isinstance(d, dict) else None
            if not dividends:
                return [str(x.get("symbol") or "") for x in rows if isinstance(x, dict) and x.get("symbol")]
            out = []
            for x in rows:
                if not isinstance(x, dict) or not x.get("symbol"):
                    continue
                out.append((str(x["symbol"]), x.get("dividend_Rate"), _us_date(x.get("announcement_Date"))))
            return out
        return None


def _us_date(s) -> date | None:
    """Nasdaq's M/D/YYYY, or None."""
    try:
        m, d, y = str(s).strip().split("/")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None
