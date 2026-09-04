"""Tests for engine helpers: clean, _temper, _calibrate, and session methods."""
import collections
import pytest
import types
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

import flint.engine as eng
from flint.engine import Engine


class TestClean:
    """Tests for the clean() function."""

    def test_clean_makes_json_safe(self):
        """clean() converts numpy types, NaN/inf to None, tuples/deques to lists."""
        result = eng.clean({
            "a": np.float64(1.5),
            "b": float("nan"),
            "c": np.int64(3),
            "d": np.bool_(True),
            "e": (1, 2),
            "f": collections.deque([np.float32("inf")]),
            "g": np.array([1.0, 2.0]),
            7: "x",
        })
        assert result == {
            "a": 1.5,
            "b": None,
            "c": 3,
            "d": True,
            "e": [1, 2],
            "f": [None],
            "g": [1.0, 2.0],
            "7": "x",
        }

    def test_clean_string_and_none(self):
        """clean() returns strings and None unchanged."""
        assert eng.clean("s") == "s"
        assert eng.clean(None) is None


class TestTemper:
    """Tests for Engine._temper()."""

    def test_temper_identity_with_p_scale_1(self):
        """With p_scale 1.0, _temper is essentially identity."""
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._temper(fake, 0.7) == pytest.approx(0.7, abs=1e-6)

    def test_temper_all_0_5_with_p_scale_0(self):
        """With p_scale 0.0, every input gives 0.5."""
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=0.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._temper(fake, 0.7) == pytest.approx(0.5, abs=1e-6)
        assert Engine._temper(fake, 0.3) == pytest.approx(0.5, abs=1e-6)
        assert Engine._temper(fake, 0.0) == pytest.approx(0.5, abs=1e-6)
        assert Engine._temper(fake, 1.0) == pytest.approx(0.5, abs=1e-6)

    def test_temper_sharpened_with_p_scale_2(self):
        """With p_scale 2.0, inputs are sharpened."""
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=2.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        # p=0.7 should be > 0.7, p=0.3 should be < 0.3
        assert Engine._temper(fake, 0.7) > 0.7
        assert Engine._temper(fake, 0.3) < 0.3

    def test_temper_extremes_do_not_raise(self):
        """_temper(0.0) and _temper(1.0) do not raise and stay within (0, 1)."""
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        result_0 = Engine._temper(fake, 0.0)
        result_1 = Engine._temper(fake, 1.0)
        assert 0 < result_0 < 1
        assert 0 < result_1 < 1


class TestCalibrate:
    """Tests for Engine._calibrate()."""

    def test_calibrate_scales_bands_around_median(self):
        """Calibration scales bands around the median q[2]."""
        # Mock _temper as a standalone lambda since _calibrate calls it
        def temper_mock(p):
            return p

        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=2.0),
            cfg=types.SimpleNamespace(extended_hours=False),
            _temper=temper_mock,
        )
        q = [-20, -10, 0, 10, 20]
        qc, p = Engine._calibrate(fake, q, 0.6)
        # With band_scale=2.0, distances from median 0 are doubled
        # [-20-0, -10-0, 0, 10-0, 20-0] * 2 + 0 = [-40, -20, 0, 20, 40]
        assert qc == pytest.approx([-40, -20, 0, 20, 40], abs=1e-6)
        # p equals _temper(p_raw) since band_scale doesn't affect p
        assert p == pytest.approx(0.6, abs=1e-6)

    def test_calibrate_with_median_pivot(self):
        """With q=[90,95,100,105,110] and band_scale=0.5, median is pivot."""
        def temper_mock(p):
            return p

        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=0.5),
            cfg=types.SimpleNamespace(extended_hours=False),
            _temper=temper_mock,
        )
        q = [90, 95, 100, 105, 110]
        qc, p = Engine._calibrate(fake, q, 0.5)
        # With band_scale=0.5, distances from median 100 are halved
        # [90-100, 95-100, 100, 105-100, 110-100] * 0.5 + 100
        # = [-10*0.5+100, -5*0.5+100, 100, 5*0.5+100, 10*0.5+100]
        # = [95, 97.5, 100, 102.5, 105]
        assert qc == pytest.approx([95, 97.5, 100, 102.5, 105], abs=1e-6)


class _FakeDateTime:
    """Fake datetime class for testing session hours."""

    def __init__(self, year, month, day, hour, minute):
        self._dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))

    @classmethod
    def now(cls, tz=None):
        return cls._dt


class TestRegularSession:
    """Tests for Engine._regular_session()."""

    @pytest.fixture
    def patch_datetime(self, monkeypatch):
        """Patch flint.engine.datetime to return a fixed time."""

        def at(year, month, day, hour, minute):
            fake_dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))

            class FakeDateTime:
                @classmethod
                def now(cls, tz=None):
                    return fake_dt

            monkeypatch.setattr(eng, "datetime", FakeDateTime)

        return at

    def test_regular_session_before_open(self, patch_datetime):
        """Thursday 2026-09-03 at 09:29 ET -> False."""
        patch_datetime(2026, 9, 3, 9, 29)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._regular_session(fake) is False

    def test_regular_session_at_open(self, patch_datetime):
        """Thursday 2026-09-03 at 09:30 ET -> True."""
        patch_datetime(2026, 9, 3, 9, 30)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._regular_session(fake) is True

    def test_regular_session_during_session(self, patch_datetime):
        """Thursday 2026-09-03 at 15:59 ET -> True."""
        patch_datetime(2026, 9, 3, 15, 59)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._regular_session(fake) is True

    def test_regular_session_at_close(self, patch_datetime):
        """Thursday 2026-09-03 at 16:00 ET -> False."""
        patch_datetime(2026, 9, 3, 16, 0)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._regular_session(fake) is False

    def test_regular_session_weekend(self, patch_datetime):
        """Saturday 2026-09-05 at 12:00 ET -> False."""
        patch_datetime(2026, 9, 5, 12, 0)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._regular_session(fake) is False


class TestExtendedSession:
    """Tests for Engine._extended_session()."""

    @pytest.fixture
    def patch_datetime(self, monkeypatch):
        """Patch flint.engine.datetime to return a fixed time."""

        def at(year, month, day, hour, minute):
            fake_dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))

            class FakeDateTime:
                @classmethod
                def now(cls, tz=None):
                    return fake_dt

            monkeypatch.setattr(eng, "datetime", FakeDateTime)

        return at

    def test_extended_session_pre_market(self, patch_datetime):
        """Thursday 03:59 -> False; 04:00 -> True."""
        patch_datetime(2026, 9, 3, 3, 59)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._extended_session(fake) is False

        patch_datetime(2026, 9, 3, 4, 0)
        assert Engine._extended_session(fake) is True

    def test_extended_session_transition_to_regular(self, patch_datetime):
        """09:29 -> True; 09:30 -> False (regular takes over)."""
        patch_datetime(2026, 9, 3, 9, 29)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._extended_session(fake) is True

        patch_datetime(2026, 9, 3, 9, 30)
        assert Engine._extended_session(fake) is False

    def test_extended_session_post_market(self, patch_datetime):
        """16:00 -> True; 19:59 -> True; 20:00 -> False."""
        patch_datetime(2026, 9, 3, 16, 0)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._extended_session(fake) is True

        patch_datetime(2026, 9, 3, 19, 59)
        assert Engine._extended_session(fake) is True

        patch_datetime(2026, 9, 3, 20, 0)
        assert Engine._extended_session(fake) is False

    def test_extended_session_weekend(self, patch_datetime):
        """Sunday 2026-09-06 at 18:00 -> False."""
        patch_datetime(2026, 9, 6, 18, 0)
        fake = types.SimpleNamespace(
            metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
            cfg=types.SimpleNamespace(extended_hours=False),
        )
        assert Engine._extended_session(fake) is False


class TestStockSession:
    """Tests for Engine._stock_session()."""

    @pytest.fixture
    def patch_datetime(self, monkeypatch):
        """Patch flint.engine.datetime to return a fixed time."""

        def at(year, month, day, hour, minute, extended_hours=False):
            fake_dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))

            class FakeDateTime:
                @classmethod
                def now(cls, tz=None):
                    return fake_dt

            monkeypatch.setattr(eng, "datetime", FakeDateTime)

            # Mock session methods to return the actual time-based values for the patched time
            mins = hour * 60 + minute
            is_regular = 570 <= mins < 960
            is_extended = (240 <= mins < 570) or (960 <= mins < 1200)

            fake = types.SimpleNamespace(
                metrics=types.SimpleNamespace(p_scale=1.0, band_scale=1.0),
                cfg=types.SimpleNamespace(extended_hours=extended_hours),
                _regular_session=lambda: is_regular,
                _extended_session=lambda: is_extended,
            )
            return fake

        return at

    def test_stock_session_respects_extended_hours_switch(self, patch_datetime):
        """Thursday 17:00 ET with cfg.extended_hours False -> _stock_session False;
        with True -> True."""
        fake_no_ext = patch_datetime(2026, 9, 3, 17, 0, extended_hours=False)
        assert Engine._stock_session(fake_no_ext) is False

        fake_ext = patch_datetime(2026, 9, 3, 17, 0, extended_hours=True)
        assert Engine._stock_session(fake_ext) is True

    def test_stock_session_regular_hours_either_way(self, patch_datetime):
        """Thursday 12:00 -> True either way."""
        fake_no_ext = patch_datetime(2026, 9, 3, 12, 0, extended_hours=False)
        assert Engine._stock_session(fake_no_ext) is True

        fake_ext = patch_datetime(2026, 9, 3, 12, 0, extended_hours=True)
        assert Engine._stock_session(fake_ext) is True

    def test_stock_session_after_hours_false_either_way(self, patch_datetime):
        """Thursday 21:00 -> False either way."""
        fake_no_ext = patch_datetime(2026, 9, 3, 21, 0, extended_hours=False)
        assert Engine._stock_session(fake_no_ext) is False

        fake_ext = patch_datetime(2026, 9, 3, 21, 0, extended_hours=True)
        assert Engine._stock_session(fake_ext) is False
