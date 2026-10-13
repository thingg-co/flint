"""Tests for the risk gates in Engine._suggest"""
import collections
import pytest
import types
import numpy as np

from flint.bars import Bar
from flint.learner import Prediction
from flint.config import Config
from flint.engine import Engine

KNOBS = dict(cost_bps=8.0, confirm_bars=2, skill_min_n=8, min_hit_rate=0.5, max_spread_bps=25.0, min_price=5.0,
             size_by_coverage=True, extended_hours=False, move_floor_bps=8.0, score_threshold=0.35,
             prob_margin=0.06, kelly_fraction=0.15, max_size=1.0)   # pinned so the tests ignore FLINT_* env


def make_fake(**overrides):
    """Create a fake Engine instance with all attributes _suggest touches."""
    cfg = Config()
    for k, v in KNOBS.items():
        setattr(cfg, k, v)
    defaults = {
        "cfg": cfg,
        "metrics": types.SimpleNamespace(
            trusted=True, hit_ema=0.6, coverage_ema=0.8, band_scale=1.0, p_scale=1.0
        ),
        "prices": {"AAPL": {"price": 200.0, "bid": 199.9, "ask": 200.1, "ts": 0.0}},
        "bars": {"AAPL": collections.deque([Bar(
            ts=1_800_000_000.0, open=200.0, high=200.5, low=199.8, close=200.0,
            volume=1000.0, buy_volume=600.0, sell_volume=400.0, trades=15, spread_bps=5.0
        )])},
        "outcomes": {"AAPL": [{"hit": True}] * 7 + [{"hit": False}] * 3},
        "_streak": {},
        "muted": set(),
        "burry_enabled": False,
        "crowding": {},
        "guru_tilt": {},
        "_calibrate": lambda q, p_raw: ([float(v) for v in q], p_raw),
        "_temper": lambda p: p,
        "_regular_session": lambda: True,
        "_extended_session": lambda: True,
        "_stock_session": lambda: True,
    }

    # Apply cfg overrides: merge dict onto Config with setattr
    if "cfg" in overrides:
        cfg_overrides = overrides.pop("cfg")
        for k, v in cfg_overrides.items():
            setattr(defaults["cfg"], k, v)

    # Apply non-cfg overrides
    for k, v in overrides.items():
        defaults[k] = v

    # Return SimpleNamespace with merged defaults
    return types.SimpleNamespace(**defaults)


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



class TestMetaGate:
    """The meta-labeler's odds gate the live call once it has learned; before that it is silent."""

    def test_low_odds_hold_and_an_unfitted_model_gates_nothing(self):
        pred = make_pred([-20, 10, 40, 70, 100], 0.7, 0.2)
        for _ in range(2):
            fake = make_fake(meta=types.SimpleNamespace(fitted=True, predict=lambda x: np.array([0.31])), cfg={"meta_min": 0.5})
            result = Engine._suggest(fake, 0, "AAPL", pred)
        assert result["action"] == "HOLD" and "meta-labeler gives it 31% odds" in result["why"] and result["meta"] == 0.31
        fake = make_fake(meta=types.SimpleNamespace(fitted=True, predict=lambda x: np.array([0.62])), cfg={"meta_min": 0.5})
        assert Engine._suggest(fake, 0, "AAPL", pred)["action"] == "HOLD"        # confirmation first
        assert Engine._suggest(fake, 0, "AAPL", pred)["action"] == "BUY"
        fake = make_fake(meta=types.SimpleNamespace(fitted=False, predict=lambda x: None), cfg={"meta_min": 0.9})
        Engine._suggest(fake, 0, "AAPL", pred)
        r = Engine._suggest(fake, 0, "AAPL", pred)
        assert r["action"] == "BUY" and r["meta"] is None


class TestTicketFloor:
    """A position too small to be worth its ticket is lifted to the floor or not placed; a put's
    commission has to be earned back over the hold."""

    def test_a_tiny_signal_trades_at_the_floor(self):
        pred = make_pred([-20, 10, 40, 70, 100], 0.7, 0.2)
        fake = make_fake(cfg={"kelly_fraction": 0.001, "min_position_dollars": 1000.0})   # Kelly alone would size this at 0.1%
        Engine._suggest(fake, 0, "AAPL", pred)
        r = Engine._suggest(fake, 0, "AAPL", pred)
        assert r["action"] == "BUY" and r["size"] == 0.01 and r["ticket_bps"] == 0.0     # $1,000 of a $100k book
        fake = make_fake(cfg={"kelly_fraction": 0.001, "min_position_dollars": 0.0})
        Engine._suggest(fake, 0, "AAPL", pred)
        assert Engine._suggest(fake, 0, "AAPL", pred)["size"] <= 0.001                   # off: Kelly's size stands

    def test_a_put_ticket_has_to_pay_for_itself(self):
        pred = make_pred([-100, -70, -40, -10, 20], 0.2, 0.7)
        fake = make_fake(cfg={"option_commission": 80.0, "min_position_dollars": 1000.0, "kelly_fraction": 0.001})
        Engine._suggest(fake, 0, "AAPL", pred)
        r = Engine._suggest(fake, 0, "AAPL", pred)
        assert r["action"] == "HOLD" and "contract ticket costs" in r["why"] and r["ticket_bps"] > 100
        fake = make_fake(cfg={"option_commission": 0.65, "min_position_dollars": 1000.0, "kelly_fraction": 0.001})
        Engine._suggest(fake, 0, "AAPL", pred)
        r = Engine._suggest(fake, 0, "AAPL", pred)
        assert r["action"] == "SELL" and 0 < r["ticket_bps"] < 20                        # one contract on $1,000: 13 bps round trip



def test_the_session_gates_need_the_clock_the_calendar_and_a_fresh_status_to_agree(monkeypatch):
    import time as _time
    from types import SimpleNamespace
    import flint.clock as clock
    from flint.engine import Engine
    monkeypatch.setattr(clock, "regular_session", lambda now=None: True)
    monkeypatch.setattr(clock, "extended_session", lambda now=None: False)
    now = _time.time()
    eng = SimpleNamespace(market_status={}, cfg=SimpleNamespace(extended_hours=False))
    assert Engine._regular_session(eng) is True                          # no status at all: the clock and the calendar decide
    eng.market_status = {"isOpen": False, "session": None, "t": now}
    assert Engine._regular_session(eng) is False                         # a fresh status saying closed overrules the clock
    assert Engine._stock_session(eng) is False
    eng.market_status = {"isOpen": True, "session": "regular", "t": now}
    assert Engine._regular_session(eng) is True
    eng.market_status = {"isOpen": False, "session": None, "t": now - 2 * Engine.STATUS_FRESH_S}
    assert Engine._regular_session(eng) is True                          # stale: no longer a voice
    monkeypatch.setattr(clock, "regular_session", lambda now=None: False)
    monkeypatch.setattr(clock, "extended_session", lambda now=None: True)
    eng.cfg.extended_hours = True
    eng.market_status = {"isOpen": False, "session": "post-market", "t": now}
    assert Engine._regular_session(eng) is False and Engine._stock_session(eng) is True
    eng.market_status = {"isOpen": False, "session": None, "t": now}      # the status says the day is over
    assert Engine._extended_session(eng) is False and Engine._stock_session(eng) is False


def test_paper_reset_action_starts_a_fresh_book():
    from types import SimpleNamespace
    from flint.engine import Engine
    from flint.paper import PaperBook
    said = []
    old = PaperBook(100_000.0, 5.0)
    old.cash = 99_800.0
    old.n_trades = 4
    eng = SimpleNamespace(paper=old, cfg=SimpleNamespace(paper_start=100_000.0), _live_prices=lambda: {},
                          _new_paper=lambda: PaperBook(100_000.0, 5.0), _save=lambda: None,
                          trace=SimpleNamespace(emit=lambda ch, text, lvl="info": said.append(text)),
                          learning_enabled=True, metrics=SimpleNamespace(live_labels=0, trusted=False))
    r = Engine.apply_control(eng, {"action": "paper_reset"})
    assert r == {"ok": True, "paper": "reset"} and eng.paper is not old and eng.paper.cash == 100_000.0
    assert said and "99,800.00" in said[0] and "4 trades" in said[0]
