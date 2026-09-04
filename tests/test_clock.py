"""Tests for flint.clock module - wall-clock session logic."""
from datetime import datetime
from zoneinfo import ZoneInfo

import flint.clock as clock


class TestRegularSession:
    """Tests for clock.regular_session()."""

    def test_regular_session_before_open(self):
        """Thursday 2026-09-03 at 09:29 ET -> False."""
        now = datetime(2026, 9, 3, 9, 29, tzinfo=ZoneInfo("America/New_York"))
        assert clock.regular_session(now) is False

    def test_regular_session_at_open(self):
        """Thursday 2026-09-03 at 09:30 ET -> True."""
        now = datetime(2026, 9, 3, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert clock.regular_session(now) is True

    def test_regular_session_during_session(self):
        """Thursday 2026-09-03 at 15:59 ET -> True."""
        now = datetime(2026, 9, 3, 15, 59, tzinfo=ZoneInfo("America/New_York"))
        assert clock.regular_session(now) is True

    def test_regular_session_at_close(self):
        """Thursday 2026-09-03 at 16:00 ET -> False."""
        now = datetime(2026, 9, 3, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.regular_session(now) is False

    def test_regular_session_weekend(self):
        """Saturday 2026-09-05 at 12:00 ET -> False."""
        now = datetime(2026, 9, 5, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.regular_session(now) is False

    def test_regular_session_with_different_timezone(self):
        """Time in UTC should convert correctly to ET."""
        # 12:00 UTC is 08:00 ET (before market open)
        now = datetime(2026, 9, 3, 12, 0, tzinfo=ZoneInfo("UTC"))
        assert clock.regular_session(now) is False
        # 21:30 UTC is 17:30 ET (after market close)
        now = datetime(2026, 9, 3, 21, 30, tzinfo=ZoneInfo("UTC"))
        assert clock.regular_session(now) is False


class TestExtendedSession:
    """Tests for clock.extended_session()."""

    def test_extended_session_pre_market(self):
        """Thursday 03:59 -> False; 04:00 -> True."""
        now = datetime(2026, 9, 3, 3, 59, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is False

        now = datetime(2026, 9, 3, 4, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is True

    def test_extended_session_transition_to_regular(self):
        """09:29 -> True; 09:30 -> False (regular takes over)."""
        now = datetime(2026, 9, 3, 9, 29, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is True

        now = datetime(2026, 9, 3, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is False

    def test_extended_session_post_market(self):
        """16:00 -> True; 19:59 -> True; 20:00 -> False."""
        now = datetime(2026, 9, 3, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is True

        now = datetime(2026, 9, 3, 19, 59, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is True

        now = datetime(2026, 9, 3, 20, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is False

    def test_extended_session_weekend(self):
        """Sunday 2026-09-06 at 18:00 -> False."""
        now = datetime(2026, 9, 6, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.extended_session(now) is False


class TestStockSession:
    """Tests for clock.stock_session()."""

    def test_stock_session_regular_hours_either_way(self):
        """Thursday 12:00 -> True either way."""
        now = datetime(2026, 9, 3, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.stock_session(False, now) is True
        assert clock.stock_session(True, now) is True

    def test_stock_session_respects_extended_hours_switch(self):
        """Thursday 17:00 ET with extended_hours False -> stock_session False;
        with True -> True."""
        now = datetime(2026, 9, 3, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.stock_session(False, now) is False
        assert clock.stock_session(True, now) is True

    def test_stock_session_after_hours_false_either_way(self):
        """Thursday 21:00 -> False either way."""
        now = datetime(2026, 9, 3, 21, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.stock_session(False, now) is False
        assert clock.stock_session(True, now) is False

    def test_stock_session_pre_market_false_without_extended(self):
        """Thursday 06:00 ET with extended_hours False -> False."""
        now = datetime(2026, 9, 3, 6, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.stock_session(False, now) is False

    def test_stock_session_pre_market_true_with_extended(self):
        """Thursday 06:00 ET with extended_hours True -> True."""
        now = datetime(2026, 9, 3, 6, 0, tzinfo=ZoneInfo("America/New_York"))
        assert clock.stock_session(True, now) is True
