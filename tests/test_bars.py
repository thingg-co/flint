"""Tests for BarBuilder in flint/bars.py"""
import time
import types

import pytest

from flint.bars import BarBuilder
from flint.feed import Tick


def tick(sym, ts, price, size=1.0, taker_buy=None, **kw):
    """Helper to create ticks with a base time offset."""
    return Tick(
        symbol=sym,
        ts=ts,
        price=price,
        size=size,
        taker_buy=taker_buy,
        **kw
    )


class TestBarBuilder:
    """Tests for BarBuilder."""

    def test_trades_build_ohlc_and_volume(self):
        """OHLC, volume, trades, buy_volume, sell_volume accumulate correctly."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        # Use current time, rounded down to nearest bar boundary
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        # A trades at T0+10, T0+20, T0+30, T0+40
        builder.add(tick("A", T0 + 10, 10.0, size=2.0, taker_buy=True))
        builder.add(tick("A", T0 + 20, 12.0, size=1.0, taker_buy=False))
        builder.add(tick("A", T0 + 30, 9.0, size=3.0, taker_buy=None))
        builder.add(tick("A", T0 + 40, 11.0, size=1.0, taker_buy=True))
        # B trades at T0+15
        builder.add(tick("B", T0 + 15, 100.0, size=1.0))

        # Roll at T0+300+1 to close the T0 interval
        rows = builder.roll(now=T0 + 300 + 1)

        assert len(rows) == 1
        row = rows[0]
        bar_a = row["A"]
        bar_b = row["B"]

        assert bar_a.open == 10.0
        assert bar_a.high == 12.0
        assert bar_a.low == 9.0
        assert bar_a.close == 11.0
        assert bar_a.volume == 7.0  # 2 + 1 + 3 + 1
        assert bar_a.trades == 4
        assert bar_a.buy_volume == 3.0  # 2 + 1
        assert bar_a.sell_volume == 1.0  # only the one with taker_buy=False
        assert bar_a.ts == T0 + 300

        # B should be a flat bar since it only had one trade
        assert bar_b.open == 100.0
        assert bar_b.high == 100.0
        assert bar_b.low == 100.0
        assert bar_b.close == 100.0
        assert bar_b.volume == 1.0
        assert bar_b.trades == 1
        assert bar_b.ts == T0 + 300

    def test_roll_waits_for_grace(self):
        """Roll with grace period: inside grace returns [], just past grace returns row."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        builder.add(tick("A", T0 + 10, 10.0, size=1.0))
        builder.add(tick("B", T0 + 15, 100.0, size=1.0))

        # Inside grace (grace is 0.5 by default)
        rows = builder.roll(now=T0 + 300 + 0.2)
        assert rows == []

        # Just past grace
        rows = builder.roll(now=T0 + 301)
        assert len(rows) == 1

    def test_gap_intervals_are_skipped(self):
        """Empty intervals between trades are skipped."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        # T0 interval: both A and B trade
        builder.add(tick("A", T0 + 10, 10.0, size=1.0))
        builder.add(tick("B", T0 + 15, 100.0, size=1.0))

        # T0+300 interval: no trades (market closed)

        # T0+600 interval: both A and B trade again
        builder.add(tick("A", T0 + 610, 11.0, size=1.0))
        builder.add(tick("B", T0 + 615, 101.0, size=1.0))

        # Roll at T0+901 - should return 2 rows, skipping the empty T0+300 interval
        rows = builder.roll(now=T0 + 901)

        assert len(rows) == 2
        assert rows[0]["A"].ts == T0 + 300
        assert rows[1]["A"].ts == T0 + 900

    def test_untraded_symbol_gets_flat_bar(self):
        """Symbol that didn't trade in an interval gets a flat bar from last close."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 900  # Start 3 bars ago for better separation

        # T0 interval: both A and B trade (B at 100)
        builder.add(tick("A", T0 + 10, 10.0, size=1.0))
        builder.add(tick("B", T0 + 15, 100.0, size=1.0))

        # T0+300 interval: only A trades
        builder.add(tick("A", T0 + 310, 11.0, size=1.0))

        # Roll at T0+1201 - should get 2 rows: T0 (both traded) and T0+300 (A traded, B flat)
        # But only T0+300 is requested to be closed, and we need B's last_close to be set
        rows = builder.roll(now=T0 + 1201)

        # We should get 2 rows - T0 interval (both traded) and T0+300 interval (A traded, B flat)
        # Note: T0+600 is also included because we're rolling up to T0+1201
        # Let's be more precise about the assertion
        assert len(rows) == 2
        assert rows[0]["A"].ts == T0 + 300  # First interval (T0)
        assert rows[1]["A"].ts == T0 + 600  # Second interval (T0+300)

        # In the second row (T0+300 interval), B should be flat
        bar_b = rows[1]["B"]
        assert bar_b.open == 100.0  # last_close from T0 interval
        assert bar_b.high == 100.0
        assert bar_b.low == 100.0
        assert bar_b.close == 100.0
        assert bar_b.volume == 0.0
        assert bar_b.trades == 0

    def test_no_row_until_every_symbol_has_a_price(self):
        """A row is only emitted when every symbol has traded at least once."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 900  # Start 3 bars ago

        # T0 interval: only A trades
        builder.add(tick("A", T0 + 10, 10.0, size=1.0))

        # Roll - B has never traded, so no row is emitted
        # The T0 interval is skipped because B has no last_close
        rows = builder.roll(now=T0 + 301)
        assert rows == []

        # T0+300 interval: A and B both trade
        builder.add(tick("A", T0 + 310, 11.0, size=1.0))
        builder.add(tick("B", T0 + 315, 100.0, size=1.0))

        # Now B has traded, so a row is emitted for the T0+300 interval
        # The T0 interval was already skipped (discarded) because B had no price
        rows = builder.roll(now=T0 + 1201)
        # Only 1 row for the T0+300 interval (B's first trade interval)
        assert len(rows) == 1
        # The ts for the T0+300 interval is T0 + 600 (bar close time)
        assert rows[0]["A"].ts == T0 + 600

    def test_late_tick_folds_into_oldest_open_bar(self):
        """Late ticks (stamped inside an already-closed interval) fold into the oldest bar."""
        builder = BarBuilder(symbols=["A", "B"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        # T0 interval: both A and B trade
        builder.add(tick("A", T0 + 10, 10.0, size=1.0))
        builder.add(tick("B", T0 + 15, 100.0, size=1.0))

        # Roll - closes T0 interval
        rows = builder.roll(now=T0 + 301)
        assert len(rows) == 1

        # Late tick for A stamped at T0+100 (inside the T0 interval, now closed)
        # This should fold into the T0 bar (which is already closed but bar builder
        # will still fold it into the oldest open bar, which is T0+300)
        builder.add(tick("A", T0 + 100, 10.5, size=2.0))

        # T0+300 interval: A trades normally
        builder.add(tick("A", T0 + 310, 11.0, size=1.0))
        builder.add(tick("B", T0 + 315, 101.0, size=1.0))

        # Roll at T0+601 - A's bar should include both the late tick and the normal tick
        rows = builder.roll(now=T0 + 601)

        assert len(rows) == 1
        bar_a = rows[0]["A"]
        # The late tick (T0+100) is folded into the T0+300 bar because by then next_index=2
        # (T0+300 interval). Since the bar is new for A in this interval, its open is
        # set to the late tick's price (10.5). The normal tick (T0+310) updates
        # high to max(10.5, 11) = 11, low to min(10.5, 11) = 10.5, close to 11.
        # The actual behavior is that the bar open is the first tick in the bar it's assigned to
        assert bar_a.open == 10.5  # Late tick's price sets the open
        assert bar_a.high == 11.0
        assert bar_a.low == 10.5  # min(10.5, 11)
        assert bar_a.close == 11.0
        assert bar_a.volume == 3.0  # 2 (late) + 1 (normal)
        assert bar_a.trades == 2

    def test_spread_from_bid_ask(self):
        """Spread is computed correctly in bps from bid/ask."""
        builder = BarBuilder(symbols=["A"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        # Tick with bid 99.5, ask 100.5, price 100
        # spread = (100.5 - 99.5) / ((100.5 + 99.5) / 2) * 10000
        #        = 1.0 / 100.0 * 10000 = 100 bps
        builder.add(tick("A", T0 + 10, 100.0, size=1.0, bid=99.5, ask=100.5))

        rows = builder.roll(now=T0 + 301)

        assert len(rows) == 1
        bar = rows[0]["A"]
        assert bar.spread_bps == pytest.approx(100.0)

    def test_preaggregated_bars_merge(self):
        """Pre-aggregated OHLC bars (with o/h/l set) merge correctly."""
        builder = BarBuilder(symbols=["A"], bar_seconds=300)
        T0 = int(time.time() // 300) * 300 - 600  # Start 2 bars ago

        # First pre-aggregated bar
        builder.add(Tick(
            symbol="A",
            ts=T0 + 10,
            price=12.0,
            size=0,
            taker_buy=None,
            o=10.0,  # open
            h=15.0,  # high
            l=9.0,   # low
        ))

        # Second pre-aggregated bar in same interval
        builder.add(Tick(
            symbol="A",
            ts=T0 + 20,
            price=11.0,
            size=0,
            taker_buy=None,
            o=12.0,
            h=13.0,
            l=8.0,
        ))

        rows = builder.roll(now=T0 + 301)

        assert len(rows) == 1
        bar = rows[0]["A"]

        # First bar sets open=10, then second bar updates:
        # high = max(15, 13) = 15
        # low = min(9, 8) = 8
        # close = 11
        assert bar.open == 10.0
        assert bar.high == 15.0
        assert bar.low == 8.0
        assert bar.close == 11.0

    def test_quote_ticks_never_build_bars(self):
        """
        Test that _on_live_tick filters out quote ticks before builder.add.

        The _on_live_tick method in engine.py (line 813) checks:
            if not tick.quote and (tick.size > 0 or tick.o is not None):
                self.builder.add(tick)

        So quote=True with size=0 should NOT call builder.add.
        A trade tick (quote=False, size=1) SHOULD call builder.add.

        We also need to stub _on_tick since the method calls it unconditionally.
        """
        import time

        # Create a mock builder whose add() records ticks
        recorded_ticks = []

        class MockBuilder:
            def add(self, tick):
                recorded_ticks.append(tick)

        # Create a stand-in for Engine with minimal attributes
        mock_engine = types.SimpleNamespace(
            builder=MockBuilder(),
            _on_tick=lambda tick, live: None  # stub to avoid AttributeError
        )

        # Call _on_live_tick unbound on the Engine class
        from flint.engine import Engine

        # T0 is arbitrary but must be defined
        T0 = int(time.time() // 300) * 300

        # Quote tick with size 0 - should NOT be passed to builder
        quote_tick = Tick(
            symbol="A",
            ts=T0 + 10,
            price=100.0,
            size=0,
            taker_buy=None,
            quote=True
        )
        Engine._on_live_tick(mock_engine, quote_tick)
        assert len(recorded_ticks) == 0

        # Reset for next test
        recorded_ticks.clear()

        # Trade tick (quote=False, size=1) - SHOULD be passed to builder
        trade_tick = Tick(
            symbol="A",
            ts=T0 + 20,
            price=100.0,
            size=1,
            taker_buy=True
        )
        Engine._on_live_tick(mock_engine, trade_tick)
        assert len(recorded_ticks) == 1
        assert recorded_ticks[0] is trade_tick
