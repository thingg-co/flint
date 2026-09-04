"""Tests for the risk gates in Engine._suggest"""
import collections
import pytest
import types
import numpy as np

from flint.bars import Bar
from flint.learner import Prediction
from flint.config import Config


def make_fake(**overrides):
    """Create a fake Engine instance with all attributes _suggest touches."""
    fake = types.SimpleNamespace()
    cfg = Config()
    cfg.cost_bps = 8.0
    cfg.confirm_bars = 2
    cfg.skill_min_n = 8
    cfg.min_hit_rate = 0.5
    cfg.max_spread_bps = 25.0
    cfg.min_price = 5.0
    cfg.size_by_coverage = True
    cfg.extended_hours = False
    cfg.move_floor_bps = 8.0
    cfg.score_threshold = 0.35
    cfg.prob_margin = 0.06
    cfg.kelly_fraction = 0.15
    cfg.max_size = 1.0
    cfg.straddle_enabled = True
    cfg.straddle_band_bps = 120.0
    cfg.straddle_max = 3
    cfg.straddle_min_coverage = 0.55
    cfg.straddle_hold_bars = 12
    cfg.muted_symbols = ""
    cfg.burry_enabled = False

    # Apply overrides to cfg
    if 'cfg' in overrides:
        for k, v in overrides.pop('cfg').items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if k == 'cfg':
            continue
        if k == 'metrics':
            continue
        if k == 'prices':
            continue
        if k == 'bars':
            continue
        if k == 'outcomes':
            continue
        if k == '_streak':
            continue
        if k == 'muted':
            continue
        if k == 'burry_enabled':
            continue
        if k == 'crowding':
            continue
        if k == 'guru_tilt':
            continue
        if k == '_calibrate':
            continue
        if k == '_temper':
            continue
        if k == '_regular_session':
            continue
        if k == '_extended_session':
            continue
        if k == '_stock_session':
            continue
        setattr(fake, k, v)

    fake.cfg = cfg
    fake.metrics = types.SimpleNamespace(
        trusted=True,
        hit_ema=0.6,
        coverage_ema=0.8,
        band_scale=1.0,
        p_scale=1.0
    )
    fake.prices = {"AAPL": {"price": 200.0, "bid": 199.9, "ask": 200.1, "ts": 0.0}}
    bar = Bar(
        ts=1_800_000_000.0,
        open=200.0,
        high=200.5,
        low=199.8,
        close=200.0,
        volume=1000.0,
        buy_volume=600.0,
        sell_volume=400.0,
        trades=15,
        spread_bps=5.0
    )
    fake.bars = {"AAPL": collections.deque([bar])}
    fake.outcomes = {"AAPL": [{"hit": True}] * 7 + [{"hit": False}] * 3}
    fake._streak = {}
    fake.muted = set()
    fake.burry_enabled = False
    fake.crowding = {}
    fake.guru_tilt = {}
    fake._calibrate = lambda q, p_raw: ([float(v) for v in q], p_raw)
    fake._temper = lambda p: p
    fake._regular_session = lambda: True
    fake._extended_session = lambda: True
    fake._stock_session = lambda: True

    # Apply remaining overrides
    for k, v in overrides.items():
        setattr(fake, k, v)

    return fake


def make_pred(q, p_up, p_down):
    """Create a Prediction with the given quantiles and probabilities."""
    return Prediction(
        q=np.array([q]),  # (1, 5)
        p_up=np.array([p_up]),
        p_down=np.array([p_down]),
        gate=np.zeros(1),
        attn=np.zeros((1, 1))
    )


class TestSuggestGates:
    """Tests for the policy risk gates in Engine._suggest."""

    def test_buy_needs_confirmation(self):
        """First call returns HOLD waiting for confirmation; second call BUYs."""
        fake = make_fake()

        # First call - should HOLD because of confirmation
        result1 = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result1["action"] == "HOLD"
        assert "waiting for the signal to persist (1/2 bars)" in result1["why"]

        # Second call - should BUY after confirmation
        result2 = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result2["action"] == "BUY"
        assert result2["size"] > 0

    def test_no_track_record_holds(self):
        """No outcomes -> HOLD with no track record message."""
        fake = make_fake(outcomes={"AAPL": []})

        # First call
        result1 = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result1["action"] == "HOLD"

        # Second call - still HOLD due to no track record
        result2 = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result2["action"] == "HOLD"
        assert "no track record" in result2["why"]

    def test_low_hit_rate_holds(self):
        """Low hit rate (30%) -> HOLD."""
        fake = make_fake(outcomes={"AAPL": [{"hit": True}] * 3 + [{"hit": False}] * 7})

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result["action"] == "HOLD"
        assert "its record on this name is 30%" in result["why"]

    def test_wide_spread_holds(self):
        """Wide spread (500 bps) -> HOLD."""
        fake = make_fake(prices={"AAPL": {"price": 200.0, "bid": 195.0, "ask": 205.0, "ts": 0.0}})

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result["action"] == "HOLD"
        assert "wider than the 25 bps limit" in result["why"]

    def test_penny_price_holds(self):
        """Penny stock (price 3.0) -> HOLD."""
        bar = Bar(ts=1_800_000_000.0, open=3.0, high=3.01, low=2.99, close=3.0, volume=1000.0)
        fake = make_fake(
            prices={"AAPL": {"price": 3.0, "bid": 2.99, "ask": 3.01, "ts": 0.0}},
            bars={"AAPL": collections.deque([bar])}
        )

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result["action"] == "HOLD"
        assert "below the 5 floor" in result["why"]

    def test_flat_bar_holds(self):
        """Flat bar (volume 0) -> HOLD."""
        bar = Bar(ts=1_800_000_000.0, open=200.0, high=200.0, low=200.0, close=200.0, volume=0.0)
        fake = make_fake(bars={"AAPL": collections.deque([bar])})

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result["action"] == "HOLD"
        assert "no live trade this bar" in result["why"]

    def test_short_needs_regular_hours(self):
        """Bearish forecast outside regular hours -> HOLD (shorts are puts)."""
        fake = make_fake(_regular_session=lambda: False, _stock_session=lambda: True)

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-100, -70, -40, -10, 20], 0.3, 0.7))
        assert result["action"] == "HOLD"
        assert "shorts are puts" in result["why"]

    def test_short_in_regular_hours_sells(self):
        """Bearish forecast in regular hours -> SELL after confirmation."""
        fake = make_fake()

        # First call - holds due to confirmation
        Engine._suggest(fake, 0, "AAPL", make_pred([-100, -70, -40, -10, 20], 0.3, 0.7))

        # Second call - SELL after confirmation
        result = Engine._suggest(fake, 0, "AAPL", make_pred([-100, -70, -40, -10, 20], 0.3, 0.7))
        assert result["action"] == "SELL"
        assert result["side"] == -1

    def test_size_scales_with_coverage(self):
        """Size scales with coverage_ema: 0.8 -> 0.4 -> 0.1 (floor)."""
        fake_high = make_fake(metrics=types.SimpleNamespace(
            trusted=True, hit_ema=0.6, coverage_ema=0.8, band_scale=1.0, p_scale=1.0
        ))

        fake_mid = make_fake(metrics=types.SimpleNamespace(
            trusted=True, hit_ema=0.6, coverage_ema=0.4, band_scale=1.0, p_scale=1.0
        ))

        fake_low = make_fake(metrics=types.SimpleNamespace(
            trusted=True, hit_ema=0.6, coverage_ema=0.1, band_scale=1.0, p_scale=1.0
        ))

        # Second calls (after confirmation)
        result_high = Engine._suggest(fake_high, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        result_mid = Engine._suggest(fake_mid, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        result_low = Engine._suggest(fake_low, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))

        size_high = result_high["size"]
        size_mid = result_mid["size"]
        size_low = result_low["size"]

        # 0.4 / 0.8 = 0.5
        assert size_mid == pytest.approx(size_high * 0.5, rel=1e-6)
        # 0.1 / 0.8 = 0.125, but floor is 0.25, so 0.25 / 0.8 = 0.3125
        assert size_low == pytest.approx(size_high * 0.25, rel=1e-6)

    def test_muted_holds(self):
        """Muted symbol -> HOLD with muted flag."""
        fake = make_fake(muted={"AAPL"})

        result = Engine._suggest(fake, 0, "AAPL", make_pred([-20, 10, 40, 70, 100], 0.7, 0.2))
        assert result["action"] == "HOLD"
        assert result["muted"] is True


# Import Engine after defining test classes so it can find them
from flint.engine import Engine
