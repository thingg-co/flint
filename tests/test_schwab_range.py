"""Schwab ranged history: the deep-backfill page [cutoff, until) by date range, with a 429 retried
once. Exists because the deep backfill was FMP-only, so a Schwab-only box never paged history."""
import asyncio
from types import SimpleNamespace

import flint.sources as srcmod
from flint.config import Config
from flint.sources import SchwabSource


def make_source():
    cfg = Config()
    cfg.symbols = ["AAPL", "MSFT"]

    async def token():
        return "tok"

    auth = SimpleNamespace(authenticated=True, has_creds=True, token=token)
    return SchwabSource(cfg, ["AAPL", "MSFT"], auth)


class FakeClient:
    """Serves candles at cutoff-300, cutoff, until-300 and until for every symbol; a scripted
    status sequence per symbol lets a test inject a 429."""
    calls: list = []
    statuses: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        FakeClient.calls.append(dict(params))
        seq = FakeClient.statuses.get(params["symbol"], [200])
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        lo, hi = params["startDate"] / 1000, params["endDate"] / 1000
        candles = [{"datetime": int(t * 1000), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 5}
                   for t in (lo - 300, lo, hi - 300, hi)]
        return SimpleNamespace(status_code=status, json=lambda: {"candles": candles}, text="")


def test_range_page_is_bounded_and_by_date(monkeypatch):
    monkeypatch.setattr(srcmod.httpx, "AsyncClient", FakeClient)
    FakeClient.calls, FakeClient.statuses = [], {}
    src = make_source()
    cutoff, until = 1_750_000_000.0, 1_750_864_000.0
    ticks = asyncio.run(src.backfill_range(["AAPL", "MSFT", "BTC-USD"], cutoff, until))
    assert {c["symbol"] for c in FakeClient.calls} == {"AAPL", "MSFT"}        # equities only
    c = FakeClient.calls[0]
    assert (c["startDate"], c["endDate"]) == (int(cutoff * 1000), int(until * 1000))
    assert (c["frequencyType"], c["frequency"]) == ("minute", "5") and "period" not in c
    assert sorted(t.ts for t in ticks if t.symbol == "AAPL") == [cutoff, until - 300]   # [cutoff, until)
    assert len(ticks) == 4


def test_429_is_retried_once(monkeypatch):
    monkeypatch.setattr(srcmod.httpx, "AsyncClient", FakeClient)
    slept = []

    async def nap(s):
        slept.append(s)

    monkeypatch.setattr(srcmod.asyncio, "sleep", nap)
    FakeClient.calls, FakeClient.statuses = [], {"AAPL": [429, 200], "MSFT": [429, 429]}
    src = make_source()
    ticks = asyncio.run(src.backfill_range(["AAPL", "MSFT"], 1_750_000_000.0, 1_750_864_000.0))
    assert slept and all(s >= 1 for s in slept)
    assert {t.symbol for t in ticks} == {"AAPL"}                              # MSFT gave up after one retry
    assert len([c for c in FakeClient.calls if c["symbol"] == "MSFT"]) == 2
