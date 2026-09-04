"""Alpaca's latest 1-minute bars become pre-aggregated ticks, once each, once complete."""
import asyncio
import types

import pytest

import flint.sources as sources
from flint.sources import AlpacaSource

T0 = "2026-09-04T13:30:00Z"      # 09:30 ET
T1 = "2026-09-04T13:31:00Z"
T0_S = 1788528600.0


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeClient:
    script = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        key = "bars" if "bars/latest" in url else "quotes"
        return self.script[key]


def make(monkeypatch, bars, bars_status=200, now=T0_S + 30.0):
    monkeypatch.setattr(sources.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(sources.time, "time", lambda: now)
    FakeClient.script = {"quotes": FakeResp(200, {"quotes": {"AAPL": {"bp": 199.9, "ap": 200.1}}}),
                         "bars": FakeResp(bars_status, {"bars": bars})}
    cfg = types.SimpleNamespace(alpaca_creds=("k", "s"), alpaca_feed="iex", alpaca_seconds=6.0)
    return AlpacaSource(cfg, ["AAPL"])


def bar(t, c=200.0, v=1000.0):
    return {"t": t, "o": 199.0, "h": 201.0, "l": 198.5, "c": c, "v": v}


def poll(src):
    out = []
    asyncio.run(src.poll_once(out.append))
    return out


def test_first_poll_holds_the_forming_bar_and_emits_the_quote(monkeypatch):
    src = make(monkeypatch, {"AAPL": bar(T0)})
    out = poll(src)
    assert [t.quote for t in out] == [True]
    assert src._pending_bar["AAPL"]["t"] == T0
    assert "1 symbols, 0 bars" in src.note


def test_a_newer_minute_completes_the_pending_bar(monkeypatch):
    src = make(monkeypatch, {"AAPL": bar(T0)})
    poll(src)
    FakeClient.script["bars"] = FakeResp(200, {"bars": {"AAPL": bar(T1, c=201.0)}})
    out = [t for t in poll(src) if not t.quote]
    assert len(out) == 1
    t = out[0]
    assert (t.symbol, t.ts, t.price, t.size, t.o, t.h, t.l) == ("AAPL", T0_S + 59.0, 200.0, 1000.0, 199.0, 201.0, 198.5)
    assert src._pending_bar["AAPL"]["t"] == T1


def test_an_old_pending_bar_is_flushed_once(monkeypatch):
    src = make(monkeypatch, {"AAPL": bar(T0)}, now=T0_S + 30.0)
    poll(src)
    monkeypatch.setattr(sources.time, "time", lambda: T0_S + 100.0)
    assert len([t for t in poll(src) if not t.quote]) == 1
    assert len([t for t in poll(src) if not t.quote]) == 0      # same minute again: nothing new
    assert len([t for t in poll(src) if not t.quote]) == 0


def test_bars_error_keeps_quotes_flowing(monkeypatch):
    src = make(monkeypatch, {}, bars_status=403)
    out = poll(src)
    assert [t.quote for t in out] == [True]
    assert "bars 403" in src.note


def test_zero_close_is_never_emitted(monkeypatch):
    src = make(monkeypatch, {"AAPL": bar(T0, c=0.0)}, now=T0_S + 200.0)
    assert [t for t in poll(src) if not t.quote] == []
    assert src._pending_bar == {}


def test_bar_tick_lands_in_its_own_five_minute_bucket():
    assert int((T0_S + 59.0) // 300) == int(T0_S // 300)
