"""The dated earnings calendar (`flint.events`) and its place in the feature pipeline."""
import asyncio
import json
from datetime import date, datetime, timezone, timedelta
from types import SimpleNamespace

import numpy as np

from flint.events import DECAY_DAYS, EarningsCalendar
from flint.features import CORE, EXTENDED, NY, FeatureBuilder

UTC = timezone.utc


def _t(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=NY).timestamp()


def test_add_covers_and_the_flags_decay_from_the_report_date():
    cal = EarningsCalendar()
    cal.add(date(2026, 9, 3), ["AAPL", "msft "])
    cal.add(date(2026, 9, 4), [])                                             # a day fetched with no reports counts as covered
    assert cal.days == {"2026-09-03": ["AAPL", "MSFT"]} and cal.covered == {"2026-09-03", "2026-09-04"}
    lo, hi = datetime(2026, 9, 1, tzinfo=UTC).timestamp(), datetime(2026, 9, 4, tzinfo=UTC).timestamp()
    assert [d.isoformat() for d in cal.missing(lo, hi)] == ["2026-09-01", "2026-09-02"] and not cal.covers(lo, hi)
    assert cal.covers(datetime(2026, 9, 3, tzinfo=UTC).timestamp(), hi)
    syms = ["AAPL", "MSFT", "NVDA"]
    near, post = cal.at(_t(2026, 9, 3), syms)                                 # the day itself: both flags 1, the hour unknown
    assert near.tolist() == [1.0, 1.0, 0.0] and post.tolist() == [1.0, 1.0, 0.0]
    near, post = cal.at(_t(2026, 8, 31), syms)                                # three days ahead
    assert abs(near[0] - np.exp(-3 / DECAY_DAYS)) < 1e-12 and post[0] == 0.0
    near, post = cal.at(_t(2026, 9, 8), syms)                                 # five days after
    assert near[0] == 0.0 and abs(post[0] - np.exp(-5 / DECAY_DAYS)) < 1e-12
    near, post = cal.at(_t(2026, 11, 3), syms)                                # beyond NEAR_DAYS: nothing
    assert post[0] == 0.0
    cal.add(date(2026, 10, 1), ["AAPL"])
    near, _ = cal.at(_t(2026, 9, 8), syms)                                    # the next report counts from the bar's own date
    assert abs(near[0] - np.exp(-23 / DECAY_DAYS)) < 1e-12
    assert cal.at(_t(2026, 9, 8, 10), syms)[0] is cal.at(_t(2026, 9, 8, 15), syms)[0]   # one answer per session date
    cal.forget(date(2026, 9, 4))
    assert cal.covered == {"2026-09-03"} and "2026-10-01" not in cal.days


def test_persists_and_reloads(tmp_path):
    p = str(tmp_path / "events.json")
    cal = EarningsCalendar(p)
    cal.add(date(2026, 9, 3), ["AAPL"])
    cal.add(date(2026, 9, 4), [])
    cal.save()
    d = json.load(open(p))
    assert d["days"] == {"2026-09-03": ["AAPL"]} and d["covered"] == ["2026-09-03", "2026-09-04"]
    again = EarningsCalendar(p)
    assert again.load() and again.days == cal.days and again.covered == cal.covered
    assert again.at(_t(2026, 9, 3), ["AAPL"])[0][0] == 1.0
    assert "1 reports on 2 days" in again.status


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _Client:
    """Answers like Nasdaq's calendar: rows per day, one 429 first, a holiday with no data, one day down."""
    def __init__(self):
        self.calls, self.limited = [], True

    async def get(self, url, params=None, headers=None):
        day = params["date"]
        self.calls.append(day)
        if self.limited:
            self.limited = False
            return _Resp(429, None)
        if day == "2026-09-10":
            return _Resp(500, None)
        if day == "2026-09-09":
            return _Resp(200, {"data": None, "message": "No data"})
        rows = [{"symbol": "AAPL", "time": "time-after-hours"}, {"symbol": "MSFT", "time": "time-not-supplied"}] if day == "2026-09-03" else []
        return _Resp(200, {"data": {"rows": rows}})


def test_fetch_walks_the_missing_weekdays_newest_first_and_persists(tmp_path, monkeypatch):
    import flint.events as ev
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ev.asyncio, "sleep", lambda s: real_sleep(0))
    cal = EarningsCalendar(str(tmp_path / "events.json"))
    lo, hi = datetime(2026, 9, 1, tzinfo=UTC).timestamp(), datetime(2026, 9, 11, tzinfo=UTC).timestamp()
    said = []
    c = _Client()
    res = asyncio.run(cal.fetch(lo, hi, say=said.append, client=c, rate_s=0))
    assert res["days"] == 9 and res["done"] == 8 and res["failed"] == 1 and res["names"] == 2
    assert c.calls[0] == c.calls[1] == "2026-09-11" and c.calls[1:] == sorted(c.calls[1:], reverse=True)   # a 429 retries; newest first
    assert cal.days == {"2026-09-03": ["AAPL", "MSFT"]}
    assert [d.isoformat() for d in cal.missing(lo, hi)] == ["2026-09-10"] and not cal.covers(lo, hi)
    assert cal.covers(lo, datetime(2026, 9, 9, tzinfo=UTC).timestamp())
    assert EarningsCalendar(cal.path).load()
    # a refresh forgets the days from a date on and fetches them again
    c2 = _Client(); c2.limited = False
    res = asyncio.run(cal.fetch(lo, hi, client=c2, rate_s=0, refresh_from=date(2026, 9, 9)))
    assert sorted(c2.calls) == ["2026-09-09", "2026-09-10", "2026-09-11"] and res["failed"] == 1


def test_the_extended_builder_reads_the_calendar_for_live_and_paged_bars_alike():
    from flint.bars import Bar
    cal = EarningsCalendar()
    cal.add(date(2026, 10, 5), ["A"])
    fb = FeatureBuilder(["A", "B"], window=1, features=EXTENDED)
    fb.events = cal
    ts = _t(2026, 10, 5, 10)
    fb.push({s: Bar(ts=ts, open=100, high=101, low=99, close=100, volume=10.0) for s in ("A", "B")})
    x = fb.window[-1]
    F = EXTENDED.index
    assert x[0, F("earn_near")] == 1.0 and x[0, F("earn_post")] == 1.0 and x[1, F("earn_near")] == 0.0
    bare = FeatureBuilder(["A", "B"], window=1, features=EXTENDED)                # no calendar: zeros, never an error
    bare.push({s: Bar(ts=ts, open=100, high=101, low=99, close=100, volume=10.0) for s in ("A", "B")})
    assert bare.window[-1][0, F("earn_near")] == 0.0
    core = FeatureBuilder(["A", "B"], window=1, features=CORE)
    assert not core.wants_events and "earn_near" not in core.features


def test_history_ready_asks_the_calendar_only_when_the_set_reads_it():
    from flint.engine import Engine
    from flint.newshist import NewsHistory
    news = NewsHistory()
    news.add("A", [], 0.0, 10 * 86400.0)
    cal = EarningsCalendar()
    eng = SimpleNamespace(news_history=news, _news_names=lambda: ["A"], events=cal, features=SimpleNamespace(wants_events=False))
    ready = Engine._history_ready
    assert ready(eng, 86400.0, 5 * 86400.0)                                      # the news is in and no calendar is needed
    eng.features.wants_events = True
    assert not ready(eng, 86400.0, 5 * 86400.0)                                  # the calendar is wanted and empty
    for d in EarningsCalendar.weekdays(86400.0, 5 * 86400.0):
        cal.add(d, [])
    assert ready(eng, 86400.0, 5 * 86400.0)
    assert not ready(eng, 20 * 86400.0, 25 * 86400.0)                            # the news does not cover these days
