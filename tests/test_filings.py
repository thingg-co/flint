"""The dated EDGAR store (`flint.filings`): point-in-time facts and filings, the trailing twelve
months, the sector-relative ratios, the filing flags, persistence, the paced fetch and its place
in the feature pipeline."""
import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from flint.features import CORE, EXTENDED, FILINGS, NY, VALUATION, FeatureBuilder
from flint.filings import (EIGHTK_DAYS, STAKE_DAYS, EdgarStore, _avail, latest, sector_relative, ttm)


def _d(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=NY).timestamp()


def _iso(y, m, d):
    return f"{y:04d}-{m:02d}-{d:02d}"


def _row(filed, start, end, val):
    return [_avail(filed), datetime(*start, tzinfo=NY).timestamp(), datetime(*end, tzinfo=NY).timestamp(), float(val)]


def test_a_filing_is_public_from_the_next_session_and_the_latest_reading_wins():
    assert _avail("2026-07-31") == datetime(2026, 8, 1, tzinfo=NY).timestamp()
    rows = [_row("2026-05-01", (2026, 3, 28), (2026, 3, 28), 100.0), _row("2026-07-31", (2026, 6, 27), (2026, 6, 27), 110.0)]
    assert latest(rows, _d(2026, 7, 31, 15)) == 100.0            # filed on the 31st: not yet seen that afternoon
    assert latest(rows, _d(2026, 8, 3)) == 110.0
    assert np.isnan(latest(rows, _d(2026, 4, 1)))


def test_ttm_is_the_annual_plus_the_year_to_date_less_the_same_stretch_a_year_before():
    rows = [
        _row("2025-11-01", (2024, 9, 29), (2025, 9, 27), 1000.0),   # FY2025, 10-K
        _row("2025-02-01", (2024, 9, 29), (2024, 12, 28), 300.0),   # Q1 FY2025 (a quarter)
        _row("2025-05-02", (2024, 9, 29), (2025, 3, 29), 560.0),    # 6M FY2025, year to date
        _row("2026-01-31", (2025, 9, 28), (2025, 12, 27), 330.0),   # Q1 FY2026
        _row("2026-05-01", (2025, 9, 28), (2026, 3, 28), 620.0),    # 6M FY2026
    ]
    assert ttm(rows, _d(2025, 12, 1)) == 1000.0                     # only the annual so far
    assert ttm(rows, _d(2026, 2, 15)) == 1000.0 + 330.0 - 300.0     # FY + Q1 - prior Q1
    assert ttm(rows, _d(2026, 6, 1)) == 1000.0 + 620.0 - 560.0      # FY + 6M - prior 6M
    assert np.isnan(ttm(rows, _d(2024, 6, 1)))
    quarters = [_row("2026-02-01", (2025, 10, 1), (2025, 12, 31), 10.0), _row("2026-05-01", (2026, 1, 1), (2026, 3, 31), 20.0),
                _row("2026-08-01", (2026, 4, 1), (2026, 6, 30), 30.0), _row("2026-11-01", (2026, 7, 1), (2026, 9, 30), 40.0),
                _row("2026-11-15", (2026, 7, 1), (2026, 9, 30), 40.0)]   # the same quarter filed twice counts once
    assert ttm(quarters, _d(2026, 12, 1)) == 100.0                  # no annual figure: the last four distinct quarters
    assert np.isnan(ttm(quarters[:3], _d(2026, 12, 1)))


def test_sector_relative_uses_the_sector_median_when_it_has_three_names_else_the_universe():
    vals = np.array([1.0, 2.0, 3.0, 10.0, np.nan, 20.0])
    secs = ["T", "T", "T", "F", "F", None]
    out = sector_relative(vals, secs)
    assert out[:3].tolist() == [-1.0, 0.0, 1.0]                     # tech: its own median 2
    med_all = np.median([1.0, 2.0, 3.0, 10.0, 20.0])
    assert out[3] == 10.0 - med_all and out[5] == 20.0 - med_all and np.isnan(out[4])
    assert np.isnan(sector_relative(np.array([np.nan, np.nan]), ["T", "T"])).all()


def _store():
    st = EdgarStore(contact="dev@example.com")
    st.sectors = {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "JPM": "Financials"}
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
                 {"end": "2026-07-17", "val": 100.0, "filed": "2026-07-31"}]}}},
             "us-gaap": {
                 "StockholdersEquity": {"units": {"USD": [{"end": "2026-06-27", "val": 500.0, "filed": "2026-07-31"},
                                                          {"end": "2026-03-28", "val": 400.0, "filed": "2026-05-01"}]}},
                 "LongTermDebt": {"units": {"USD": [{"end": "2026-06-27", "val": 200.0, "filed": "2026-07-31"}]}},
                 "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"end": "2026-06-27", "val": 100.0, "filed": "2026-07-31"}]}},
                 "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                     {"start": "2024-09-29", "end": "2025-09-27", "val": 300.0, "filed": "2025-11-01"}]}},
                 "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                     {"start": "2024-09-29", "end": "2025-09-27", "val": 50.0, "filed": "2025-11-01"}]}},
                 "OperatingIncomeLoss": {"units": {"USD": [{"start": "2024-09-29", "end": "2025-09-27", "val": 200.0, "filed": "2025-11-01"}]}},
                 "DepreciationDepletionAndAmortization": {"units": {"USD": [{"start": "2024-09-29", "end": "2025-09-27", "val": 50.0, "filed": "2025-11-01"}]}},
                 "NetIncomeLoss": {"units": {"USD": [{"start": "2024-09-29", "end": "2025-09-27", "val": 150.0, "filed": "2025-11-01"}]}},
                 "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": [
                     {"start": "2024-09-29", "end": "2025-09-27", "val": 1.0, "filed": "2025-11-01"}]}},
             }}}
    now = _d(2026, 10, 1)
    st.add_facts("AAPL", facts, now)
    subs = {"form": ["8-K", "4", "4", "SC 13G/A", "10-Q", "4"],
            "filingDate": ["2026-09-28", "2026-09-20", "2026-09-10", "2026-09-01", "2026-07-31", "2026-05-15"],
            "acceptanceDateTime": ["2026-09-28T20:30:00.000Z", "2026-09-20T22:00:00.000Z", "2026-09-10T21:00:00.000Z",
                                   "2026-09-01T16:00:00.000Z", "2026-07-31T00:30:00.000Z", "2026-05-15T21:00:00.000Z"]}
    st.add_filings("AAPL", subs, now)
    return st, now


def test_the_ratios_are_point_in_time_and_relative_to_the_sector_and_the_flags_decay():
    st, now = _store()
    syms = ["AAPL", "MSFT", "NVDA", "JPM"]
    closes = np.array([10.0, 50.0, 60.0, 70.0])
    out = st.at(_d(2026, 9, 29), syms, closes)
    assert set(out) == set(VALUATION) | set(FILINGS) and all(v.shape == (4,) for v in out.values())
    # AAPL alone has facts: its sector has one value, so the universe median is its own and every ratio reads 0 (nothing to compare with)
    assert out["val_pb"].tolist() == [0.0, 0.0, 0.0, 0.0]
    # with a peer the difference shows: give MSFT the same facts at twice the price-to-book
    st.facts["MSFT"] = {k: [list(r) for r in v] for k, v in st.facts["AAPL"].items()}
    st._cache = None
    out = st.at(_d(2026, 9, 29), syms, np.array([10.0, 20.0, 60.0, 70.0]))
    pb_a, pb_m = np.log(10.0 * 100 / 500.0), np.log(20.0 * 100 / 500.0)
    med = (pb_a + pb_m) / 2
    assert abs(out["val_pb"][0] - (pb_a - med)) < 1e-9 and abs(out["val_pb"][1] - (pb_m - med)) < 1e-9
    assert abs(out["val_fcf"][0] - ((300.0 - 50.0) / 1000.0 - ((300.0 - 50.0) / 1000.0 + (300.0 - 50.0) / 2000.0) / 2)) < 1e-9
    ev_a = np.log((1000.0 + 200.0 - 100.0) / 250.0)
    ev_m = np.log((2000.0 + 200.0 - 100.0) / 250.0)
    assert abs(out["val_ev"][0] - (ev_a - (ev_a + ev_m) / 2)) < 1e-9
    assert abs(out["val_dy"][0] - (1.0 / 10.0 - (1.0 / 10.0 + 1.0 / 20.0) / 2)) < 1e-9
    assert out["qual_roe"][0] == 0.0                                   # the same ROE: no difference
    # point in time: before the 10-Q of 2026-07-31 was public the equity read 400 and the shares were unknown
    out_old = st.at(_d(2026, 7, 31, 15), syms, np.array([10.0, 20.0, 60.0, 70.0]))
    assert out_old["val_pb"].tolist() == [0.0, 0.0, 0.0, 0.0]          # no shares yet: no market cap, nothing to compare
    # the filing flags: an 8-K accepted 2026-09-28 16:30 ET reads exp(-d/3) a day later, two Form 4s in the month, a 13G
    t = datetime(2026, 9, 29, 16, 30, tzinfo=NY).timestamp()
    out = st.at(t, syms, closes)
    assert abs(out["filing_8k"][0] - np.exp(-1.0 / EIGHTK_DAYS)) < 1e-9 and out["filing_8k"][1] == 0.0
    assert abs(out["insider_n"][0] - np.log1p(2)) < 1e-9
    t13 = datetime.fromisoformat("2026-09-01T16:00:00+00:00").timestamp()        # accepted at 16:00 UTC, 12:00 ET
    assert abs(out["stake_13d"][0] - np.exp(-(t - t13) / 86400.0 / STAKE_DAYS)) < 1e-9
    before = st.at(datetime(2026, 9, 28, 12, tzinfo=NY).timestamp(), syms, closes)
    assert before["filing_8k"][0] == 0.0                              # accepted after that bar
    assert st.at(_d(2026, 11, 15), syms, closes)["filing_8k"][0] == 0.0   # past EIGHTK_MAX


def test_persistence_round_trips(tmp_path):
    st, now = _store()
    st.path = str(tmp_path / "edgar.json")
    st.cik["AAPL"] = 320193
    st.nocik.add("SPY")
    st.fetched["AAPL"] = now
    st.save()
    st2 = EdgarStore(st.path)
    assert st2.load() and st2.cik == {"AAPL": 320193} and st2.nocik == {"SPY"} and st2.fetched == {"AAPL": now}
    assert st2.facts == st.facts and st2.filings == st.filings and st2.reports == st.reports
    assert st2.ready(["AAPL", "SPY"]) and not st2.ready(["AAPL", "MSFT"])


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class _Client:
    """A fake EDGAR: the ticker file, one submissions page, one companyfacts document."""
    def __init__(self):
        self.urls: list[str] = []
        self.facts_calls = 0

    async def get(self, url, **kw):
        self.urls.append(url)
        if url.endswith("company_tickers.json"):
            return _Resp(200, {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                               "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMorgan"}})
        if "submissions/CIK0000320193.json" in url:
            return _Resp(200, {"filings": {"recent": {"form": ["8-K", "10-Q"], "filingDate": ["2026-09-28", "2026-07-31"],
                                                      "acceptanceDateTime": ["2026-09-28T20:30:00.000Z", "2026-07-31T00:30:00.000Z"]},
                                           "files": []}})
        if "submissions/CIK0000019617.json" in url:
            return _Resp(503)
        if "companyfacts/CIK0000320193.json" in url:
            self.facts_calls += 1
            return _Resp(200, {"facts": {"us-gaap": {"StockholdersEquity": {"units": {"USD": [
                {"end": "2026-06-27", "val": 500.0, "filed": "2026-07-31"}]}}}}})
        return _Resp(404)

    async def aclose(self):
        pass


def test_fetch_maps_tickers_reads_filings_and_facts_and_skips_fresh_names():
    st = EdgarStore(contact="dev@example.com")
    c = _Client()
    now = _d(2026, 10, 1)
    res = asyncio.run(st.fetch(["AAPL", "JPM", "SPY"], client=c, rate_s=0.0, now=now))
    assert res["names"] == 1 and res["failed"] == 1 and res["facts"] == 1 and res["filings"] == 1
    assert st.cik == {"AAPL": 320193, "JPM": 19617} and st.nocik == {"SPY"}
    assert st.ready(["AAPL", "SPY"]) and not st.ready(["JPM"])      # JPM failed: read again next pass
    assert c.facts_calls == 1
    res = asyncio.run(st.fetch(["AAPL", "SPY"], client=c, rate_s=0.0, now=now + 60))
    assert res["names"] == 0 and res["skipped"] == 2 and c.facts_calls == 1   # fresh: nothing read
    res = asyncio.run(st.fetch(["AAPL"], client=c, rate_s=0.0, now=now + 7 * 3600))
    assert res["names"] == 1 and c.facts_calls == 1                 # stale: submissions again, no new report, facts kept
    empty = EdgarStore(contact="")
    said = []
    res = asyncio.run(empty.fetch(["AAPL"], say=said.append, client=c, rate_s=0.0, now=now))
    assert res["names"] == 0 and said and "contact" in said[0]       # no contact: nothing fetched, one warning


def test_the_builder_reads_the_store_and_a_core_builder_ignores_it():
    from flint.bars import Bar
    st, now = _store()
    syms = ["AAPL", "MSFT"]
    st.sectors = {"AAPL": "Technology", "MSFT": "Technology"}
    st.facts["MSFT"] = {k: [list(r) for r in v] for k, v in st.facts["AAPL"].items()}
    fb = FeatureBuilder(syms, 4, features=EXTENDED)
    fb.filings = st
    assert fb.wants_filings
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=NY).timestamp()
    for k in range(6):
        row = {s: Bar(ts=t0 + 300 * k, open=p, high=p * 1.001, low=p * 0.999, close=p, volume=100.0, buy_volume=60.0,
                      sell_volume=40.0, trades=5, spread_bps=5.0) for s, p in (("AAPL", 10.0), ("MSFT", 20.0))}
        x = fb.push(row)
    assert x is not None and x.shape[2] == len(EXTENDED)
    i8 = EXTENDED.index("filing_8k")
    assert x[0, -1, i8] > 0.0 and x[1, -1, i8] == 0.0              # AAPL's 8-K the day before, nothing on MSFT (bounded: unnormalized)
    core = FeatureBuilder(syms, 4, features=CORE)
    assert not core.wants_filings and not hasattr(core, "no_such") and core.filings is None


def test_history_ready_waits_for_the_store():
    from flint.engine import Engine
    st = EdgarStore(contact="dev@example.com")
    news = SimpleNamespace(covers=lambda names, lo, hi: True)
    cal = SimpleNamespace(covers=lambda lo, hi: True)
    fake = SimpleNamespace(news_history=news, events=cal, filings=st, symbols=["AAPL", "SPY"],
                           features=SimpleNamespace(wants_events=True, wants_filings=True), _news_names=lambda: ["AAPL"])
    assert not Engine._history_ready(fake, 0.0, 1.0)
    st.fetched["AAPL"] = 1.0
    st.nocik.add("SPY")
    assert Engine._history_ready(fake, 0.0, 1.0)
    fake.features.wants_filings = False
    st.fetched.clear()
    assert Engine._history_ready(fake, 0.0, 1.0)
