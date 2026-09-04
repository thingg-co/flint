"""Tests for flint.features module."""
import numpy as np
import pytest

from flint.bars import Bar
from flint.features import FEATURES, N_FEATURES, NORMALIZE, RunningNorm, FeatureBuilder


def bar(o, h, l, c, v=100.0, buy=60.0, sell=40.0, trades=10, spread=5.0):
    """Build a Bar with ts 0.0."""
    return Bar(ts=0.0, open=o, high=h, low=l, close=c, volume=v, buy_volume=buy, sell_volume=sell, trades=trades, spread_bps=spread)


def rows(n, price=100.0):
    """Return n identical rows {"A": bar(...), "B": bar(...)} at that price."""
    return [bar(price, price, price, price, v=100.0) for _ in range(n)]


class TestRunningNorm:
    """Tests for RunningNorm class."""

    def test_norm_first_update_sets_mean(self):
        """RunningNorm(2, N_FEATURES); update with x of ones*3; mean equals x everywhere, var is still all ones, count 1."""
        norm = RunningNorm(2, N_FEATURES)
        x = np.ones((2, N_FEATURES), dtype=np.float64)
        norm.update(x)
        # After first update, mean should equal x, var should remain ones, count should be 1
        assert np.array_equal(norm.mean, x)
        assert np.array_equal(norm.var, np.ones((2, N_FEATURES), dtype=np.float64))
        assert norm.count == 1

    def test_norm_apply_clips_and_passes_exogenous_through(self):
        """After one update with zeros, apply to an array whose normalized columns hold 100.0 and exogenous columns hold 0.7."""
        norm = RunningNorm(2, N_FEATURES)
        # First update with zeros to set mean=0, var=1
        zeros = np.zeros((2, N_FEATURES), dtype=np.float64)
        norm.update(zeros)

        # Create test array: normalized columns = 100.0, exogenous columns = 0.7
        x = np.zeros((2, N_FEATURES), dtype=np.float64)
        for i in range(2):
            for j in range(N_FEATURES):
                if NORMALIZE[j]:
                    x[i, j] = 100.0
                else:
                    x[i, j] = 0.7

        result = norm.apply(x)
        # Normalized columns should be clipped to 5.0
        for i in range(2):
            for j in range(N_FEATURES):
                if NORMALIZE[j]:
                    assert result[i, j] == 5.0
                else:
                    assert result[i, j] == 0.7
        assert result.dtype == np.float32

    def test_norm_state_round_trip(self):
        """Update three times with random arrays; st = norm.state(); fresh RunningNorm loaded with st has equal mean, var, count."""
        np.random.seed(42)
        norm = RunningNorm(2, N_FEATURES)

        # Update three times
        for _ in range(3):
            x = np.random.randn(2, N_FEATURES).astype(np.float64)
            norm.update(x)

        # Get state
        st = norm.state()
        assert st["count"] == 3

        # Create fresh norm and load state
        fresh = RunningNorm(2, N_FEATURES)
        fresh.load(st)

        assert np.array_equal(fresh.mean, norm.mean)
        assert np.array_equal(fresh.var, norm.var)
        assert fresh.count == norm.count

        # Mutate fresh norm's mean and verify st is unchanged
        fresh.mean[0, 0] = 999.0
        assert st["mean"][0, 0] != 999.0  # st should be a copy

    def test_push_returns_none_until_window_full(self):
        """FeatureBuilder(["A", "B"], window=4); pushing 3 rows returns None; 4th returns array of shape (2, 4, N_FEATURES)."""
        fb = FeatureBuilder(["A", "B"], window=4)

        # First 3 pushes should return None
        for _ in range(3):
            r = rows(2, price=100.0)
            row = {"A": r[0], "B": r[1]}
            result = fb.push(row)
            assert result is None

        # 4th push should return array
        r = rows(2, price=100.0)
        row = {"A": r[0], "B": r[1]}
        result = fb.push(row)

        assert result is not None
        assert result.shape == (2, 4, N_FEATURES)
        assert result.dtype == np.float32

    def test_flat_seeded_symbol_never_yields_nan(self):
        """FeatureBuilder(["A", "Z"], window=3) where Z's bars are all zero; push 3 rows; no NaN or inf."""
        # Z is flat-seeded: all zeros including spread=nan
        z_bar = Bar(ts=0.0, open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0, buy_volume=0.0, sell_volume=0.0, trades=0, spread_bps=float("nan"))

        fb = FeatureBuilder(["A", "Z"], window=3)

        for _ in range(3):
            r = rows(2, price=100.0)
            row = {"A": r[0], "Z": z_bar}
            result = fb.push(row)

        assert result is not None
        assert np.isfinite(result).all()

    def test_news_and_exo_are_clipped_and_pass_through(self):
        """FeatureBuilder(["A"], window=1); set_news then set_exo; check feature values."""
        fb = FeatureBuilder(["A"], window=1)

        # Set news
        fb.set_news("A", 2.0, -1.0)  # clipped to (1.0, 0.0)

        # Set exogenous
        fb.set_exo("A", {"fng": 3.0, "vix": None, "bogus": 1.0})

        # Push one row
        r = rows(1, price=100.0)
        row = {"A": r[0]}
        result = fb.push(row)

        assert result is not None
        assert result.shape == (1, 1, N_FEATURES)

        # Check feature values
        news_sent_idx = FEATURES.index("news_sent")
        news_attn_idx = FEATURES.index("news_attn")
        fng_idx = FEATURES.index("fng")
        vix_idx = FEATURES.index("vix")

        assert result[0, 0, news_sent_idx] == 1.0  # clipped from 2.0
        assert result[0, 0, news_attn_idx] == 0.0  # clipped from -1.0
        assert result[0, 0, fng_idx] == 1.0  # clipped from 3.0
        assert result[0, 0, vix_idx] == 0.0  # None becomes 0.0

        # "bogus" should not be in FEATURES
        assert "bogus" not in FEATURES

    def test_momentum_window_length(self):
        """FeatureBuilder(["A"], window=2, mom_bars=5); push 8 rows with prices 100..107; len(rets["A"]) == 5 and last ret > 0."""
        fb = FeatureBuilder(["A"], window=2, mom_bars=5)

        for i in range(8):
            price = 100 + i
            r = rows(1, price=price)
            row = {"A": r[0]}
            result = fb.push(row)

        assert result is not None
        assert len(fb.rets["A"]) == 5
        # Last raw return should be positive (price increased from 106 to 107)
        assert fb.rets["A"][-1] > 0

    def test_peek_window_pads_left(self):
        """FeatureBuilder(["A"], window=4); peek_window() is None before any push; after 2 pushes shape is (1, 4, N_FEATURES) with left padding; after 4 pushes equals last push result."""
        fb = FeatureBuilder(["A"], window=4)

        # Before any push
        assert fb.peek_window() is None

        # Push 2 rows
        for _ in range(2):
            r = rows(1, price=100.0)
            row = {"A": r[0]}
            result = fb.push(row)

        peeked = fb.peek_window()
        assert peeked is not None
        assert peeked.shape == (1, 4, N_FEATURES)
        # First two frames should be equal (left padding repeats first frame)
        assert np.array_equal(peeked[0, 0], peeked[0, 1])

        # Push 2 more to fill window
        for _ in range(2):
            r = rows(1, price=100.0)
            row = {"A": r[0]}
            result = fb.push(row)

        peeked_after_full = fb.peek_window()
        assert peeked_after_full is not None
        assert np.array_equal(peeked_after_full, result)
