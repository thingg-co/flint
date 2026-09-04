"""Tests for SourceManager routing and failover."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from flint.feed import Tick
from flint.sources import Source, SourceManager


class Prim(Source):
    """Test source with lower priority."""

    id = "prim"
    name = "Prim"
    kind = "equity"
    mechanism = "poll"
    priority = 10
    classes = ("equity",)
    poll_interval = 15.0
    fresh_after = 40.0

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self._ticks = []

    async def run(self, emit):
        for t in self._ticks:
            emit(t)

    async def backfill(self, symbols, cutoff):
        return self._ticks


class Back(Source):
    """Test source with higher priority."""

    id = "back"
    name = "Back"
    kind = "equity"
    mechanism = "poll"
    priority = 20
    classes = ("equity",)
    poll_interval = 15.0
    fresh_after = 40.0

    def __init__(self, cfg, symbols):
        super().__init__(cfg, symbols)
        self._ticks = []

    async def run(self, emit):
        for t in self._ticks:
            emit(t)

    async def backfill(self, symbols, cutoff):
        return self._ticks


def tick(sym="AAPL", price=100.0, bid=None, ask=None, ts=None):
    """Build a Tick with size 1.0, taker_buy None."""
    return Tick(
        symbol=sym,
        ts=ts if ts is not None else time.time(),
        price=price,
        size=1.0,
        taker_buy=None,
        bid=bid,
        ask=ask,
        quote=False,
    )


@pytest.fixture
def cfg():
    """Minimal config for testing."""
    return SimpleNamespace(
        bar_seconds=300,
        backfill_seconds=86400,
        sources_off="",
        sources_on="",
        av_key="",
        fmp_key="",
        finnhub_key="",
        eodhd_key="",
        schwab_key="",
        schwab_secret="",
        alpaca_creds=("", ""),
        tradier_token="",
        etrade_key="",
        etrade_secret="",
        ibkr_enabled=False,
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_client_id=1,
        ibkr_market_data_type=1,
        alpaca_seconds=5,
        alpaca_feed="iex",
        tradier_seconds=5,
        schwab_seconds=5,
        finnhub_seconds=10,
        eodhd_seconds=10,
        fmp_seconds=10,
        av_quote_seconds=60,
        etrade_seconds=10,
    )


class TestSourceManagerFailover:
    """Tests for SourceManager routing and failover behavior."""

    def test_primary_wins_when_both_fresh(self, cfg):
        """At t=1000 route a tick from 'back' then from 'prim'; provider_for == 'prim'."""
        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        on_tick_calls = []
        on_quote_calls = []
        on_provider_calls = []

        def on_tick(tick):
            on_tick_calls.append(tick)

        def on_quote(sym, bid, ask):
            on_quote_calls.append((sym, bid, ask))

        def on_provider(sym, prev, active):
            on_provider_calls.append((sym, prev, active))

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = on_tick
        mgr.on_quote = on_quote
        mgr.on_provider_change = on_provider
        mgr.trace = None

        # Route back tick first at t=1000
        back_tick = tick("AAPL", 99.0, ts=1000.0)
        mgr._route("back", back_tick)

        # Route prim tick at t=1000
        prim_tick = tick("AAPL", 100.0, ts=1000.0)
        mgr._route("prim", prim_tick)

        # Prim should be active because it has higher priority (lower priority value)
        assert mgr.provider_for("AAPL") == "prim"
        assert mgr.active["AAPL"] == "prim"

    def test_fails_over_when_primary_goes_stale(self, cfg, monkeypatch):
        """Route prim at t=1000, back at t=1000; advance to t=1050; route back; provider_for == 'back'."""
        # Use a mutable list to track time (since we can't patch time.time after fixtures are set up)
        current_time = [1000.0]

        def mock_time():
            return current_time[0]

        monkeypatch.setattr(time, "time", mock_time)

        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        on_provider_calls = []

        def on_provider(sym, prev, active):
            on_provider_calls.append((sym, prev, active))

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = lambda t: None  # no-op callback
        mgr.on_quote = None
        mgr.on_provider_change = on_provider
        mgr.trace = None

        # Route prim tick at t=1000
        prim_tick = tick("AAPL", 100.0, ts=1000.0)
        mgr._route("prim", prim_tick)

        # Route back tick at t=1000
        back_tick = tick("AAPL", 99.0, ts=1000.0)
        mgr._route("back", back_tick)

        # Advance to t=1050 (50s later, past prim's 40s fresh_after)
        current_time[0] = 1050.0

        # Route another back tick
        back_tick2 = tick("AAPL", 98.0, ts=1050.0)
        mgr._route("back", back_tick2)

        # Now back should be active because prim is stale
        assert mgr.provider_for("AAPL") == "back"
        assert mgr.active["AAPL"] == "back"

        # Check provider change was recorded
        assert ("AAPL", "prim", "back") in on_provider_calls

    def test_returns_to_primary_when_fresh_again(self, cfg, monkeypatch):
        """Continue from test 2 state; at t=1060 route prim tick; provider_for == 'prim'."""
        current_time = [1000.0]

        def mock_time():
            return current_time[0]

        monkeypatch.setattr(time, "time", mock_time)

        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        on_provider_calls = []

        def on_provider(sym, prev, active):
            on_provider_calls.append((sym, prev, active))

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = lambda t: None  # no-op callback
        mgr.on_quote = None
        mgr.on_provider_change = on_provider
        mgr.trace = None

        # Route prim tick at t=1000
        prim_tick = tick("AAPL", 100.0, ts=1000.0)
        mgr._route("prim", prim_tick)

        # Route back tick at t=1000
        back_tick = tick("AAPL", 99.0, ts=1000.0)
        mgr._route("back", back_tick)

        # Advance to t=1050
        current_time[0] = 1050.0

        # Route another back tick
        back_tick2 = tick("AAPL", 98.0, ts=1050.0)
        mgr._route("back", back_tick2)

        # Now advance to t=1060 and route a prim tick
        current_time[0] = 1060.0
        prim_tick2 = tick("AAPL", 101.0, ts=1060.0)
        mgr._route("prim", prim_tick2)

        # Prim should be active again because it's fresh
        assert mgr.provider_for("AAPL") == "prim"
        assert mgr.active["AAPL"] == "prim"

        # Check provider change was recorded
        assert ("AAPL", "back", "prim") in on_provider_calls

    def test_only_active_source_ticks_reach_on_tick(self, cfg):
        """Both fresh, prim active; back ticks without bid/ask reach neither on_tick nor on_quote;
        back ticks with bid/ask reach on_quote; prim ticks reach on_tick."""
        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        on_tick_calls = []
        on_quote_calls = []

        def on_tick(tick):
            on_tick_calls.append(tick)

        def on_quote(sym, bid, ask):
            on_quote_calls.append((sym, bid, ask))

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = on_tick
        mgr.on_quote = on_quote
        mgr.on_provider_change = None
        mgr.trace = None

        # Route prim tick first (it has higher priority)
        prim_tick = tick("AAPL", 100.0, ts=1000.0)
        mgr._route("prim", prim_tick)

        # Prim should be active now
        assert mgr.active["AAPL"] == "prim"

        # A back tick without bid/ask reaches neither on_tick nor on_quote
        back_tick_no_quote = tick("AAPL", 99.0, bid=None, ask=None, ts=1000.0)
        mgr._route("back", back_tick_no_quote)

        assert len(on_tick_calls) == 1  # Only prim tick
        assert len(on_quote_calls) == 0

        # A back tick with bid/ask reaches on_quote but not on_tick
        back_tick_with_quote = tick("AAPL", 99.5, bid=99.9, ask=100.1, ts=1000.0)
        mgr._route("back", back_tick_with_quote)

        assert len(on_tick_calls) == 1  # Still only prim tick
        assert len(on_quote_calls) == 1
        assert on_quote_calls[0] == ("AAPL", 99.9, 100.1)

        # A prim tick reaches on_tick
        prim_tick2 = tick("AAPL", 101.0, ts=1000.0)
        mgr._route("prim", prim_tick2)

        assert len(on_tick_calls) == 2  # Now prim has 2 ticks
        assert on_tick_calls[0] is prim_tick
        assert on_tick_calls[1] is prim_tick2

    def test_disabled_source_is_never_chosen(self, cfg):
        """mgr.enabled["prim"] = False; route prim and back ticks; provider_for == 'back'."""
        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": False, "back": True}  # prim is disabled
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = lambda t: None  # no-op callback
        mgr.on_quote = None
        mgr.on_provider_change = None
        mgr.trace = None

        # Route prim tick (should be ignored since disabled)
        prim_tick = tick("AAPL", 100.0, ts=1000.0)
        mgr._route("prim", prim_tick)

        # Route back tick
        back_tick = tick("AAPL", 99.0, ts=1000.0)
        mgr._route("back", back_tick)

        # Only back should be active
        assert mgr.provider_for("AAPL") == "back"
        assert mgr.active["AAPL"] == "back"

    def test_unknown_symbol_is_ignored(self, cfg):
        """Route a tick for 'ZZZ' from prim; mgr.seen still has only 'AAPL', on_tick not called."""
        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        on_tick_calls = []

        def on_tick(tick):
            on_tick_calls.append(tick)

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = on_tick
        mgr.on_quote = None
        mgr.on_provider_change = None
        mgr.trace = None

        # Route a tick for unknown symbol 'ZZZ'
        zzz_tick = tick("ZZZ", 50.0, ts=1000.0)
        mgr._route("prim", zzz_tick)

        # mgr.seen should still have only 'AAPL'
        assert list(mgr.seen.keys()) == ["AAPL"]
        # on_tick should not have been called
        assert len(on_tick_calls) == 0

    def test_backfill_keeps_richest_history(self, cfg):
        """Give Prim.backfill returning 1 tick and Back.backfill returning 3 ticks;
        assert the chosen provider for AAPL is 'back' and merged ticks number 3."""
        prim = Prim(cfg, ["AAPL"])
        back = Back(cfg, ["AAPL"])

        # Set up backfill returns
        prim_ticks = [tick("AAPL", 100.0, ts=900.0)]
        prim._ticks = prim_ticks

        back_ticks = [
            tick("AAPL", 98.0, ts=900.0),
            tick("AAPL", 99.0, ts=910.0),
            tick("AAPL", 99.5, ts=920.0),
        ]
        back._ticks = back_ticks

        mgr = SourceManager.__new__(SourceManager)
        mgr.cfg = cfg
        mgr.symbols = ["AAPL"]
        mgr.sources = {"prim": prim, "back": back}
        mgr.enabled = {"prim": True, "back": True}
        mgr.seen = {"AAPL": {}}
        mgr.active = {"AAPL": None}
        mgr.on_tick = None
        mgr.on_quote = None
        mgr.on_provider_change = None
        mgr.trace = None

        async def run_backfill():
            return await mgr.backfill(cutoff=0.0)

        merged, chosen = asyncio.run(run_backfill())

        # Back should be chosen because it has more ticks (richest history)
        assert chosen["AAPL"] == "back"
        # The merged ticks should be 3 (from back)
        assert len(merged) == 3
        assert all(t.symbol == "AAPL" for t in merged)
