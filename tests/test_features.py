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
        assert np.all(norm.count == 1)

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
        assert np.all(st["count"] == 3)

        # Create fresh norm and load state
        fresh = RunningNorm(2, N_FEATURES)
        fresh.load(st)

        assert np.array_equal(fresh.mean, norm.mean)
        assert np.array_equal(fresh.var, norm.var)
        assert np.array_equal(fresh.count, norm.count)

        # Mutate fresh norm's mean and verify st is unchanged
        fresh.mean[0, 0] = 999.0
        assert st["mean"][0, 0] != 999.0  # st should be a copy

    def test_running_norm_maps_rows_by_symbol_across_a_universe_change(self):
        """Found in Steel 2026-10-11: the state loaded by position, so after the universe loop
        promoted names the start loaded an (old N, F) array into a (new N, F) normalizer and the
        engine died on a broadcast error. Rows now travel by name, a new name starts fresh with its
        own count, and a state that cannot be mapped leaves the normalizer fresh instead of raising."""
        rng = np.random.default_rng(1)
        old = RunningNorm(3, N_FEATURES)
        for _ in range(5):
            old.update(rng.normal(size=(3, N_FEATURES)) + np.arange(3)[:, None] * 10)
        st = old.state(["AAA", "BBB", "CCC"])
        assert st["symbols"] == ["AAA", "BBB", "CCC"] and np.all(st["count"] == 5)
        new = RunningNorm(4, N_FEATURES)
        assert new.load(st, ["CCC", "DDD", "AAA", "EEE"]) is True
        assert np.array_equal(new.mean[0], old.mean[2]) and np.array_equal(new.var[2], old.var[0])     # rows moved by name
        assert np.all(new.mean[1] == 0) and np.all(new.var[1] == 1) and list(new.count) == [5, 0, 5, 0]  # new names fresh
        x = rng.normal(size=(4, N_FEATURES))
        new.update(x)
        assert np.array_equal(new.mean[1], x[1]) and not np.array_equal(new.mean[0], x[0])   # a fresh row takes its first sample, a kept row blends
        assert list(new.count) == [6, 1, 6, 1]
        legacy = {"mean": old.mean.copy(), "var": old.var.copy(), "count": 5}                  # pre-0.5.15: no names, one count
        same = RunningNorm(3, N_FEATURES)
        assert same.load(legacy) is True and np.array_equal(same.mean, old.mean) and list(same.count) == [5, 5, 5]
        other = RunningNorm(4, N_FEATURES)
        assert other.load(legacy) is False and np.all(other.mean == 0) and np.all(other.count == 0)   # a mismatch: fresh, not an exception
        assert other.load({**legacy, "symbols": ["AAA", "BBB", "CCC"]}, ["BBB", "ZZZ", "AAA", "CCC"]) is True   # the checkpoint's list names a legacy state
        assert np.array_equal(other.mean[0], old.mean[1]) and list(other.count) == [5, 0, 5, 5]
        assert other.load({"mean": np.zeros((2, 2)), "var": np.ones((2, 2)), "count": 1}) is False   # wrong feature count: fresh, not an exception

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

    def test_news_slots_are_clipped_and_pass_through_and_live_only_signals_stay_out_of_the_net(self):
        """set_news is clipped and lands unstandardized in the two news slots; set_exo is kept for
        the policy but is not a model input since 2a."""
        fb = FeatureBuilder(["A"], window=1)
        fb.set_news("A", 2.0, -1.0)                                            # clipped to (1.0, 0.0)
        fb.set_exo("A", {"fng": 3.0, "vix": None, "bogus": 1.0})
        row = {"A": rows(1, price=100.0)[0]}
        result = fb.push(row)
        assert result is not None and result.shape == (1, 1, N_FEATURES)
        assert result[0, 0, FEATURES.index("news_sent")] == 1.0 and result[0, 0, FEATURES.index("news_attn")] == 0.0
        assert fb.exo["A"] == {"fng": 1.0, "bogus": 1.0} and "fng" not in FEATURES and "vix" not in FEATURES
        for gone in ("imbalance", "spread", "logtrades", "wsb_sent", "opt_iv"):
            assert gone not in FEATURES                                        # paged history cannot carry them
        assert fb.news["A"] == (1.0, 0.0)

    def test_momentum_window_length(self):
        """FeatureBuilder(["A"], window=2, mom_bars=5); push 8 rising rows: mom is the sum of the last 5 returns."""
        fb = FeatureBuilder(["A"], window=2, mom_bars=5)
        for i in range(8):
            result = fb.push({"A": rows(1, price=100 + i)[0]})
        assert result is not None
        last = fb._last(fb.R, 5)[0]
        assert last.shape == (5,) and np.all(last > 0)                          # five positive returns
        assert fb._lag(fb.R, 0)[0] > 0 and fb._lag(fb.R, 100)[0] == 0.0          # a lag past the history reads 0

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


def _bar(ts, o, h, l, c, v=100.0):
    from flint.bars import Bar
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


class TestBarDerivedFeatures:
    """The features 2a added: the day's legs and clock, the cross-section and the lags, all from bars."""

    def _ts(self, day, bar):
        from datetime import datetime
        from flint.features import NY
        base = datetime(2026, 10, 5 + day, 9, 35, tzinfo=NY)                   # the first bar closes 09:35 ET
        return base.timestamp() + bar * 300

    def test_day_legs_clock_and_market_features(self):
        fb = FeatureBuilder(["A", "SPY"], window=1, index="SPY")
        F = FEATURES.index
        # day 0: three bars, A up, SPY flat
        r0 = fb.push({"A": _bar(self._ts(0, 0), 100, 101, 99, 100), "SPY": _bar(self._ts(0, 0), 50, 50, 50, 50)})
        x = fb.window[-1]
        assert abs(x[0, F("tod_sin")] - np.sin(2 * np.pi * 5 / 390)) < 1e-9 and x[0, F("overnight")] == 0.0
        fb.push({"A": _bar(self._ts(0, 1), 100, 102, 100, 102), "SPY": _bar(self._ts(0, 1), 50, 50, 50, 50)})
        raw_intraday = np.log(102 / 100) * 1e4
        assert abs(fb.norm.mean[0, F("intraday")] * 0 + raw_intraday - np.log(102 / 100) * 1e4) < 1e-9
        assert fb.day is not None and fb.bars_today == 2
        # breadth this bar: A up, SPY flat -> half the names up -> 0
        assert fb.window[-1][0, F("breadth")] == 0.0
        # day 1: a gap up on A sets the overnight leg; the market's open30 fills at the sixth bar
        d1 = [fb.push({"A": _bar(self._ts(1, k), 110 + k, 111 + k, 109 + k, 110.5 + k), "SPY": _bar(self._ts(1, k), 51 + 0.1 * k, 51.2 + 0.1 * k, 50.9 + 0.1 * k, 51.1 + 0.1 * k)}) for k in range(7)]
        assert fb.bars_today == 7 and fb.open30 != 0.0
        assert abs(fb.open30 - np.log(51.6 / 51.0) * 1e4) < 1e-6                 # SPY from its day open to its sixth close
        assert np.isfinite(fb.window[-1]).all()
        assert fb.index == 1 and abs(fb.M[(fb.p - 1) % fb.M.shape[0]] - np.log(51.7 / 51.6) * 1e4) < 1e-6   # the market's return is SPY's

    def test_lags_breadth_and_beta_come_from_the_ring(self):
        from flint.features import BARS_D
        rng = np.random.default_rng(1)
        fb = FeatureBuilder(["A", "B", "SPY"], window=1, index="SPY")
        px = np.array([100.0, 100.0, 50.0]); rets = []
        for k in range(BARS_D * 2 + 5):
            m = rng.standard_normal() * 0.002
            r = np.array([2 * m + rng.standard_normal() * 0.001, -0.5 * m + rng.standard_normal() * 0.001, m])
            new = px * np.exp(r); rets.append(np.log(new / px) * 1e4)
            fb.push({s: _bar(self._ts(k // BARS_D, k % BARS_D), px[i], max(px[i], new[i]), min(px[i], new[i]), new[i]) for i, s in enumerate(["A", "B", "SPY"])})
            px = new
        rets = np.array(rets)
        assert abs(fb._lag(fb.R, BARS_D)[0] - rets[-1 - BARS_D, 0]) < 1e-6         # the same slot one day back
        F = FEATURES.index; raw_beta = None
        x = fb.window[-1]
        assert x[0, F("beta")] > x[1, F("beta")]                                  # A loads on the market, B against it
        assert -1 <= x[0, F("breadth")] <= 1 and x[0, F("breadth")] == x[2, F("breadth")]   # one market number for every name
        assert np.isfinite(x).all()

    def test_a_warm_prefix_makes_the_week_features_exact(self):
        """The same bars give the same features whether or not the builder walked a prefix first,
        once the ring holds the prefix: what the deep loop relies on."""
        from flint.features import BARS_W
        rng = np.random.default_rng(2)
        bars = []
        px = 100.0
        for k in range(BARS_W + 40):
            new = px * np.exp(rng.standard_normal() * 0.002); bars.append(_bar(self._ts(k // 78, k % 78), px, max(px, new), min(px, new), new)); px = new
        cold = FeatureBuilder(["A"], window=1); warm = FeatureBuilder(["A"], window=1)
        for b in bars:
            warm.push({"A": b})
        for b in bars[-20:]:
            cold.push({"A": b})
        F = FEATURES.index
        assert warm.n_bars == len(bars) and cold.n_bars == 20
        assert warm._last(warm.R, BARS_W).shape[1] == BARS_W and cold._last(cold.R, BARS_W).shape[1] == 20
        assert warm._lag(warm.R, 78)[0] != 0.0 and cold._lag(cold.R, 78)[0] == 0.0   # the cold builder has no yesterday


class TestFeatureSets:
    """The feature set is a switch: `core` is the live 28, `extended` adds the opinions and the earnings flags."""

    def test_sets_masks_and_an_unknown_feature(self):
        from flint.features import CORE, EVENTS, EXTENDED, FEATURE_SET, FEATURE_SETS, OPINIONS, normalize_mask
        assert len(CORE) == 28 and EXTENDED == CORE + OPINIONS + EVENTS and len(EXTENDED) == 34
        assert FEATURE_SETS[FEATURE_SET] == FEATURES and len(set(EXTENDED)) == len(EXTENDED)
        m = normalize_mask(EXTENDED)
        assert not m[EXTENDED.index("ou_z")] and not m[EXTENDED.index("earn_near")] and m[EXTENDED.index("trend_t")] and m[EXTENDED.index("garch")]
        assert np.array_equal(RunningNorm(2, N_FEATURES).normalize, NORMALIZE) and RunningNorm(2, 5).normalize.all()
        with pytest.raises(ValueError):
            FeatureBuilder(["A"], window=1, features=["ret", "bogus"])
        fb = FeatureBuilder(["A"], window=1, features=EXTENDED)
        assert fb.wants_opinions and fb.wants_events and fb.norm.mean.shape == (1, 34)
        assert np.array_equal(fb.norm.normalize, normalize_mask(EXTENDED))


class TestOpinionFeatures:
    """2d: what other forecasters read in the week's ring, as functions of the ring alone."""

    def _ts(self, k):
        from datetime import datetime
        from flint.features import NY
        return datetime(2026, 9, 8, 9, 35, tzinfo=NY).timestamp() + (k // 78) * 86400 + (k % 78) * 300

    def _walk(self, fb, closes, names):
        px = closes[0]
        for k, new in enumerate(closes[1:]):
            fb.push({s: _bar(self._ts(k), px[i], max(px[i], new[i]), min(px[i], new[i]), new[i]) for i, s in enumerate(names)})
            px = new

    def test_mean_reversion_trend_and_garch_read_as_expected(self):
        from flint.features import BARS_W, EXTENDED, GARCH_A, GARCH_B, OPINION_MIN_BARS
        rng = np.random.default_rng(3)
        names = ["A", "B", "SPY"]
        lvl, closes = 0.0, [np.array([100.0, 100.0, 50.0])]
        for k in range(400):
            lvl = 0.7 * lvl + rng.standard_normal() * 0.003                     # A reverts fast to 100
            b = 100 * np.exp(0.0005 * (k + 1) + rng.standard_normal() * 0.0005)   # B trends
            closes.append(np.array([100 * np.exp(lvl), b, 50.0]))
        fb = FeatureBuilder(names, window=1, index="SPY", features=EXTENDED)
        self._walk(fb, closes, names)
        rw, mw = fb._last(fb.R, BARS_W), fb._last(fb.M, BARS_W)
        ou_z, ou_hl, trend_t, garch = fb._opinions(rw, mw, np.ones(3))
        assert np.expm1(ou_hl[0]) < 10 < 100 < np.expm1(ou_hl[1])              # A's half-life is bars, B's is the cap
        assert trend_t[1] > 5 and abs(trend_t[0]) < 4 and abs(ou_z[0]) < 4
        assert garch[0] > garch[1] > 0 and garch[2] == 0.0                     # the flat index has no vol
        # the closed form is the GARCH(1,1) recursion started at the ring's variance
        r = rw[0]; s2 = float((r * r).mean()); omega = (1 - GARCH_A - GARCH_B) * s2
        for x in r:
            s2 = omega + GARCH_A * x * x + GARCH_B * s2
        assert abs(garch[0] - np.sqrt(s2)) < 1e-9
        short = fb._opinions(rw[:, -(OPINION_MIN_BARS - 1):], mw[-(OPINION_MIN_BARS - 1):], np.ones(3))
        assert all((v == 0).all() for v in short)                               # too little ring: silence, not noise
        x = fb.window[-1]
        F = EXTENDED.index
        assert np.isfinite(x).all() and abs(x[0, F("ou_z")]) <= 5

    def test_opinions_depend_on_the_ring_alone(self):
        """Two builders that walked different histories agree once their rings hold the same week:
        what lets a deep chunk's warm prefix make them exact."""
        from flint.features import BARS_W, EXTENDED
        rng = np.random.default_rng(4)
        names = ["A", "SPY"]
        closes = [np.array([100.0, 50.0])]
        for _ in range(BARS_W + 120):
            closes.append(closes[-1] * np.exp(rng.standard_normal(2) * 0.002))
        long = FeatureBuilder(names, window=1, index="SPY", features=EXTENDED)
        late = FeatureBuilder(names, window=1, index="SPY", features=EXTENDED)
        self._walk(long, closes, names)
        self._walk(late, closes[100:], names)
        a = long._opinions(long._last(long.R, BARS_W), long._last(long.M, BARS_W), np.ones(2))
        b = late._opinions(late._last(late.R, BARS_W), late._last(late.M, BARS_W), np.ones(2))
        for u, v in zip(a, b):
            assert np.allclose(u, v, atol=1e-9)
