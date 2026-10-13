"""Dated company facts and filings from SEC EDGAR, so paged history and the live feed carry the
same valuation and filing features (TODO items 2c and the index part of 8).

`EdgarStore` keeps, per name, the XBRL facts the valuation ratios need (`TAGS`: equity, shares,
debt, cash, operating cash flow, capex, operating income, depreciation, net income, dividends per
share) from EDGAR's companyfacts API, each with the day it became public, and the acceptance
times of its 8-K, Form 4 and Schedule 13D/G filings from the submissions API; persisted in
`state/edgar.json`. `at(ts, symbols, closes)` answers point in time: a fact or a filing is seen
only from the session after its filing date, and a flow is the trailing twelve months as the
filings available then would have given it (the latest annual figure plus the year-to-date since
it less the same stretch a year before). The valuation ratios are logs of market-cap multiples
(price to book, free-cash-flow yield, enterprise value to EBITDA, dividend yield) and a quality
read (return on equity), each relative to the median of the name's sector among the names
modeled (the universe median when a sector has fewer than three names with a value), so what the
net sees is "rich or cheap against its peers today", not the level. The filing features are the
decay since the last 8-K, the count of Form 4s in the trailing month (insider activity; the
direction needs the forms' XML, not read here) and the decay since the last 13D/G.

EDGAR needs a contact in the User-Agent (`FLINT_SEC_CONTACT`); without one the store fetches
nothing and every feature reads 0, which is also what an ETF, a foreign filer on IFRS tags or a
name without a CIK reads. Fair use is ten requests a second; the fetch paces well under it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta

import numpy as np

from .features import FILINGS, NY, VALUATION  # noqa: F401  (the feature names live with the sets)

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DAY = 86400.0
KEEP_S = 3.0 * 365.25 * DAY             # facts older than this are dropped (a TTM needs two years)
FILINGS_S = 450 * DAY                   # filings kept, and how far back the submissions pages are read
EIGHTK_DAYS = 3.0                       # e-folding of the 8-K flag, calendar days
EIGHTK_MAX = 30.0                       # beyond this the flag reads 0
STAKE_DAYS = 10.0                       # e-folding of the 13D/G flag
STAKE_MAX = 60.0
INSIDER_WINDOW = 30.0                   # Form 4s counted over this many days
MIN_SECTOR = 3                          # names with a value a sector needs for its own median
REFRESH_S = 6 * 3600.0                  # a name's submissions are read again after this

# the facts kept, by the names they go under: the first tag found wins
TAGS: dict[str, tuple[str, ...]] = {
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "shares": ("EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
               "CommonStockSharesOutstanding"),
    "debt": ("LongTermDebt", "LongTermDebtNoncurrent", "DebtInstrumentCarryingAmount", "LongTermDebtAndCapitalLeaseObligations",
             "DebtLongtermAndShorttermCombinedAmount"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "opinc": ("OperatingIncomeLoss",),
    "da": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"),
    "netinc": ("NetIncomeLoss", "ProfitLoss"),
    "dps": ("CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"),
}
INSTANT = {"equity", "shares", "debt", "cash"}       # balance-sheet facts: the latest reading; the rest are flows (TTM)
# a filer that reports a total in a per-share tag, or a penny stock's book, would dominate a sector's median: raw ratios outside these are unknown
BOUNDS = {"pb": (-4.0, 6.0), "fcfy": (-1.0, 1.0), "evx": (-3.0, 6.0), "dy": (0.0, 0.25), "roe": (-2.0, 2.0)}
FORMS = {"8-K": "8k", "8-K/A": "8k", "4": "ins", "4/A": "ins", "SC 13D": "stake", "SC 13D/A": "stake", "SC 13G": "stake", "SC 13G/A": "stake"}
# EDGAR renamed the schedules in 2025 ("SCHEDULE 13G/A" beside the older "SC 13G/A"); both spellings count


def _form(form: str) -> str:
    return str(form).strip().upper().replace("SCHEDULE ", "SC ")
REPORTS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}   # a new one of these means the facts changed


def _avail(filed: str) -> float:
    """When a filing dated `filed` (ISO) is public to the next bar: the following midnight, New York."""
    d = date.fromisoformat(filed[:10]) + timedelta(days=1)
    return datetime(d.year, d.month, d.day, tzinfo=NY).timestamp()


def _ts(iso: str) -> float:
    d = date.fromisoformat(iso[:10])
    return datetime(d.year, d.month, d.day, tzinfo=NY).timestamp()


def _accept_ts(accept: str, filed: str) -> float:
    """A filing's acceptance time; the filing date's next midnight when the time is missing."""
    try:
        return datetime.fromisoformat(accept.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return _avail(filed)


def ttm(rows: list[list[float]], ts: float) -> float:
    """The trailing twelve months of a flow as the filings public at `ts` give it: the latest
    annual figure, plus the year-to-date reported since it, less the same stretch a year before
    (so a Q3 10-Q gives FY + 9M - prior 9M). Without an annual figure, the last four distinct
    quarters. Rows are [avail, start, end, value]; NaN when nothing is known."""
    seen = [r for r in rows if r[0] <= ts]
    if not seen:
        return math.nan
    ann = [r for r in seen if 350 * DAY <= r[2] - r[1] <= 380 * DAY]
    if not ann:
        q = {}
        for r in seen:
            if 80 * DAY <= r[2] - r[1] <= 100 * DAY:
                q[r[2]] = r
        last4 = sorted(q.values(), key=lambda r: r[2])[-4:]
        return float(sum(r[3] for r in last4)) if len(last4) == 4 else math.nan
    a = max(ann, key=lambda r: (r[2], r[0]))
    ytd = [r for r in seen if r[2] > a[2] and r[2] - r[1] < 350 * DAY]
    if not ytd:
        return float(a[3])
    y = max(ytd, key=lambda r: (r[2], r[0]))
    dur = y[2] - y[1]
    prev = [r for r in seen if abs(r[2] + 365.25 * DAY - y[2]) <= 12 * DAY and abs((r[2] - r[1]) - dur) <= 12 * DAY]
    if not prev:
        return float(a[3])
    return float(a[3] + y[3] - max(prev, key=lambda r: r[0])[3])


def latest(rows: list[list[float]], ts: float) -> float:
    """The latest balance-sheet reading public at `ts` (by period end, then by filing)."""
    seen = [r for r in rows if r[0] <= ts]
    if not seen:
        return math.nan
    return float(max(seen, key=lambda r: (r[2], r[0]))[3])


def _within(vals: np.ndarray, which: str) -> np.ndarray:
    """A raw ratio outside its plausible range (BOUNDS) is unknown, not an outlier to standardize."""
    lo, hi = BOUNDS[which]
    return np.where((vals >= lo) & (vals <= hi), vals, np.nan)


def sector_relative(vals: np.ndarray, sectors: list[str | None]) -> np.ndarray:
    """Each value less the median of its sector's values among the names given (the median of all
    values when the sector has fewer than MIN_SECTOR with one); NaN stays NaN."""
    out = np.full(len(vals), np.nan)
    ok = np.isfinite(vals)
    if not ok.any():
        return out
    med_all = float(np.median(vals[ok]))
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(sectors):
        if ok[i]:
            groups.setdefault(s or "", []).append(i)
    for s, idx in groups.items():
        med = float(np.median(vals[idx])) if s and len(idx) >= MIN_SECTOR else med_all
        out[idx] = vals[idx] - med
    return out


class EdgarStore:
    def __init__(self, path: str | None = None, sectors_path: str | None = None, contact: str = ""):
        self.path = path
        self.sectors_path = sectors_path
        self.contact = (contact or "").strip()
        self.cik: dict[str, int] = {}                    # name -> CIK
        self.nocik: set[str] = set()                     # names EDGAR does not know (ETFs, some ADRs)
        self.facts: dict[str, dict[str, list[list[float]]]] = {}   # name -> fact -> [[avail, start, end, value], ...] by avail
        self.filings: dict[str, dict[str, list[float]]] = {}       # name -> group -> acceptance timestamps, sorted
        self.fetched: dict[str, float] = {}              # name -> when its submissions were last read
        self.reports: dict[str, int] = {}                # name -> count of 10-K/10-Q style filings seen (a change refetches the facts)
        self.sectors: dict[str, str] = {}
        self.status = "empty"
        self._busy = False
        self._cache: tuple[int, int, dict] | None = None
        self._tickers: dict[str, int] | None = None

    # -- persistence -----------------------------------------------------------------------
    def load(self) -> bool:
        if self.sectors_path and os.path.exists(self.sectors_path):
            try:
                with open(self.sectors_path) as f:
                    self.sectors = {str(k): str(v) for k, v in json.load(f).items()}
            except (OSError, ValueError):
                self.sectors = {}
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.cik = {str(k): int(v) for k, v in (d.get("cik") or {}).items()}
            self.nocik = {str(s) for s in (d.get("nocik") or [])}
            self.facts = {str(s): {str(k): [[float(a), float(b), float(c), float(v)] for a, b, c, v in rows]
                                   for k, rows in facts.items()} for s, facts in (d.get("facts") or {}).items()}
            self.filings = {str(s): {str(k): [float(t) for t in v] for k, v in groups.items()}
                            for s, groups in (d.get("filings") or {}).items()}
            self.fetched = {str(k): float(v) for k, v in (d.get("fetched") or {}).items()}
            self.reports = {str(k): int(v) for k, v in (d.get("reports") or {}).items()}
            self.status = f"{len(self.fetched)} names"
            self._cache = None
            return True
        except (OSError, ValueError, TypeError) as e:
            log.warning("edgar store unreadable (%s); starting empty", e)
            return False

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"v": 1, "cik": self.cik, "nocik": sorted(self.nocik), "facts": self.facts, "filings": self.filings,
                       "fetched": self.fetched, "reports": self.reports}, f)
        os.replace(tmp, self.path)

    # -- coverage --------------------------------------------------------------------------
    def ready(self, symbols) -> bool:
        """Whether every name has been looked up at least once (a name EDGAR does not know counts)."""
        return all(s in self.fetched or s in self.nocik for s in symbols)

    def stale(self, symbols, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        return [s for s in symbols if s not in self.nocik and now - self.fetched.get(s, 0.0) >= REFRESH_S]

    # -- the features ----------------------------------------------------------------------
    def at(self, ts: float, symbols: list[str], closes: np.ndarray) -> dict[str, np.ndarray]:
        """The eight features per name as of a bar at `ts` with the names' closes; one answer per
        session date per universe for the ratios' inputs, the closes applied fresh."""
        n = len(symbols)
        day = int(datetime.fromtimestamp(ts, NY).toordinal())
        key = (day, hash(tuple(symbols)))
        if self._cache is None or self._cache[:2] != key:
            self._cache = (*key, self._inputs(ts, symbols))
        inp = self._cache[2]
        c = np.where(np.asarray(closes, dtype=float) > 0, np.asarray(closes, dtype=float), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            mcap = c * inp["shares"]
            ev = mcap + np.nan_to_num(inp["debt"], nan=0.0) - np.nan_to_num(inp["cash"], nan=0.0)
            ebitda = inp["opinc"] + np.nan_to_num(inp["da"], nan=0.0)
            fcf = inp["ocf"] - np.nan_to_num(inp["capex"], nan=0.0)
            pb = _within(np.where(inp["equity"] > 0, np.log(mcap / inp["equity"]), np.nan), "pb")
            fcfy = _within(np.where(mcap > 0, fcf / mcap, np.nan), "fcfy")
            evx = _within(np.where((ebitda > 0) & (ev > 0), np.log(ev / ebitda), np.nan), "evx")
            dy = _within(np.where(c > 0, inp["dps"] / c, np.nan), "dy")
            roe = _within(np.where(inp["equity"] > 0, inp["netinc"] / inp["equity"], np.nan), "roe")
        secs = [self.sectors.get(s) for s in symbols]
        out = {"val_pb": sector_relative(pb, secs), "val_fcf": sector_relative(fcfy, secs),
               "val_ev": sector_relative(evx, secs), "val_dy": sector_relative(dy, secs),
               "qual_roe": sector_relative(roe, secs)}
        for k in out:
            out[k] = np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
        eightk, ins, stake = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, s in enumerate(symbols):
            g = self.filings.get(s)
            if not g:
                continue
            t8 = [t for t in g.get("8k", ()) if t <= ts]
            if t8:
                d = (ts - t8[-1]) / DAY
                eightk[i] = math.exp(-d / EIGHTK_DAYS) if d <= EIGHTK_MAX else 0.0
            ins[i] = math.log1p(sum(1 for t in g.get("ins", ()) if ts - INSIDER_WINDOW * DAY < t <= ts))
            t13 = [t for t in g.get("stake", ()) if t <= ts]
            if t13:
                d = (ts - t13[-1]) / DAY
                stake[i] = math.exp(-d / STAKE_DAYS) if d <= STAKE_MAX else 0.0
        out.update({"filing_8k": eightk, "insider_n": ins, "stake_13d": stake})
        return out

    def _inputs(self, ts: float, symbols: list[str]) -> dict[str, np.ndarray]:
        n = len(symbols)
        inp = {k: np.full(n, np.nan) for k in TAGS}
        for i, s in enumerate(symbols):
            facts = self.facts.get(s)
            if not facts:
                continue
            for k in TAGS:
                rows = facts.get(k)
                if rows:
                    inp[k][i] = latest(rows, ts) if k in INSTANT else ttm(rows, ts)
        return inp

    # -- ingestion -------------------------------------------------------------------------
    def add_facts(self, symbol: str, companyfacts: dict, now: float | None = None) -> int:
        """Keep the facts under TAGS from a companyfacts document; returns the rows kept."""
        now = time.time() if now is None else now
        units = {}
        for tax in ("us-gaap", "dei", "ifrs-full"):
            for tag, body in ((companyfacts.get("facts") or {}).get(tax) or {}).items():
                units.setdefault(tag, body.get("units") or {})
        kept: dict[str, list[list[float]]] = {}
        for k, tags in TAGS.items():
            rows: dict[tuple, list[float]] = {}
            for tag in tags:                                 # every spelling the filer used over the years
                u = units.get(tag)
                if not u:
                    continue
                series = next(iter(u.values()), [])
                for e in series:
                    try:
                        end, filed, val = e["end"], e["filed"], float(e["val"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    end_ts = _ts(end)
                    if end_ts < now - KEEP_S:
                        continue
                    start_ts = _ts(e["start"]) if e.get("start") else end_ts
                    avail = _avail(filed)
                    key = (start_ts, end_ts)
                    if key not in rows or avail < rows[key][0]:      # the first filing of a period is when it became known
                        rows[key] = [avail, start_ts, end_ts, val]
            if rows:
                kept[k] = sorted(rows.values(), key=lambda r: (r[0], r[2]))
        if kept:
            self.facts[symbol] = kept
        self._cache = None
        return sum(len(v) for v in kept.values())

    def add_filings(self, symbol: str, recent: dict, now: float | None = None) -> int:
        """Keep the acceptance times of the 8-K, Form 4 and 13D/G filings in a submissions
        `recent` block (parallel lists); returns the filings kept and counts the reports seen."""
        now = time.time() if now is None else now
        groups = self.filings.setdefault(symbol, {})
        forms = recent.get("form") or []
        filed = recent.get("filingDate") or []
        accepted = recent.get("acceptanceDateTime") or []
        n_rep = 0
        for i, form in enumerate(forms):
            form = _form(form)
            if form in REPORTS:
                n_rep += 1
            g = FORMS.get(form)
            if not g or i >= len(filed):
                continue
            t = _accept_ts(accepted[i] if i < len(accepted) else "", filed[i])
            if t < now - FILINGS_S:
                continue
            groups.setdefault(g, []).append(t)
        for g in groups:
            groups[g] = sorted(set(groups[g]))
        self.reports[symbol] = self.reports.get(symbol, 0) + n_rep
        self._cache = None
        return sum(len(v) for v in groups.values())

    # -- fetching --------------------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        try:
            from importlib.metadata import version
            ver = version("flint")
        except Exception:  # noqa: BLE001
            ver = "dev"
        return {"User-Agent": f"Flint/{ver} ({self.contact})", "Accept-Encoding": "gzip, deflate"}

    async def fetch(self, symbols, say=None, rate_s: float = 0.15, client=None, deadline_s: float | None = None,
                    now: float | None = None) -> dict:
        """Look every name up: its CIK from the ticker file, its recent submissions (and the older
        pages back to FILINGS_S), and its companyfacts when never read or when the submissions
        show a report filed since. Names read within REFRESH_S are skipped. Returns counts."""
        say = say or (lambda t: None)
        res = {"names": 0, "facts": 0, "filings": 0, "failed": 0, "skipped": 0}
        if not self.contact:
            if self.status != "no contact":
                say("EDGAR needs a contact in the User-Agent (FLINT_SEC_CONTACT); the valuation and filing features read 0")
                self.status = "no contact"
            return res
        if self._busy:
            return res
        self._busy = True
        t0 = now if now is not None else time.time()
        start = time.monotonic()
        try:
            import httpx
            own = client is None
            c = client or httpx.AsyncClient(timeout=30, headers=self.headers(), follow_redirects=True)
            try:
                if self._tickers is None:
                    r = await c.get(TICKERS_URL)
                    if r.status_code == 200:
                        self._tickers = {str(v["ticker"]).upper().replace("-", "."): int(v["cik_str"]) for v in r.json().values()}
                    else:
                        say(f"EDGAR ticker file: HTTP {r.status_code}; filings not read")
                        return res
                    await asyncio.sleep(rate_s)
                todo = self.stale(symbols, t0)
                res["skipped"] = len(list(symbols)) - len(todo)
                for s in todo:
                    if deadline_s is not None and time.monotonic() - start > deadline_s:
                        break
                    cik = self.cik.get(s) or self._tickers.get(s.upper()) or self._tickers.get(s.upper().replace(".", "-"))
                    if not cik:
                        self.nocik.add(s)
                        continue
                    self.cik[s] = int(cik)
                    try:
                        r = await c.get(SUBS_URL.format(cik=cik))
                        await asyncio.sleep(rate_s)
                        if r.status_code != 200:
                            res["failed"] += 1
                            continue
                        sub = r.json()
                        before = self.reports.get(s, 0)
                        self.filings[s] = {}
                        self.reports[s] = 0
                        recent = (sub.get("filings") or {}).get("recent") or {}
                        res["filings"] += self.add_filings(s, recent, t0)
                        dates = recent.get("filingDate") or []
                        oldest = _ts(dates[-1]) if dates else 0.0
                        for page in (sub.get("filings") or {}).get("files") or []:
                            if oldest <= t0 - FILINGS_S:
                                break
                            r = await c.get(SUBS_PAGE_URL.format(name=page["name"]))
                            await asyncio.sleep(rate_s)
                            if r.status_code == 200:
                                pr = r.json()
                                res["filings"] += self.add_filings(s, pr, t0)
                                pd = pr.get("filingDate") or []
                                oldest = _ts(pd[-1]) if pd else 0.0
                        if s not in self.facts or self.reports[s] != before:
                            r = await c.get(FACTS_URL.format(cik=cik))
                            await asyncio.sleep(rate_s)
                            if r.status_code == 200:
                                res["facts"] += self.add_facts(s, r.json(), t0)
                            elif r.status_code != 404:
                                res["failed"] += 1
                        self.fetched[s] = t0
                        res["names"] += 1
                    except Exception as e:  # noqa: BLE001
                        res["failed"] += 1
                        log.debug("edgar %s: %s", s, e)
                        await asyncio.sleep(rate_s)
            finally:
                if own:
                    await c.aclose()
            self.status = f"{len(self.fetched)} names, {len(self.nocik)} unknown to EDGAR"
            self.save()
        finally:
            self._busy = False
        return res
