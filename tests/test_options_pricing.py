"""Tests for Black-Scholes option pricing and years_to utility."""
import math
import time
import pytest

from flint.options import bs_put, bs_call, years_to


class TestPutCallParity:
    """Test put-call parity relationship."""

    def test_put_call_parity(self):
        """Put-call parity: C - P == S - K * exp(-r * t)"""
        spot = 100.0
        strike = 100.0
        t = 0.25
        iv = 0.3
        r = 0.04

        call = bs_call(spot, strike, t, iv, r)
        put = bs_put(spot, strike, t, iv, r)
        lhs = call - put
        rhs = spot - strike * math.exp(-r * t)

        assert lhs == pytest.approx(rhs, rel=1e-9)


class TestKnownValues:
    """Test Black-Scholes against known reference values."""

    def test_known_values(self):
        """With spot=100, strike=100, t=0.25, iv=0.3, r=0.04:
        Expected call ~6.46, put ~5.46 (Black-Scholes formula).
        Note: Values computed from flint.options.bs_call/bs_put."""
        spot = 100.0
        strike = 100.0
        t = 0.25
        iv = 0.3
        r = 0.04

        call = bs_call(spot, strike, t, iv, r)
        put = bs_put(spot, strike, t, iv, r)

        # Values from flint.options: call=6.459483, put=5.464467
        assert call == pytest.approx(6.46, abs=0.05)
        assert put == pytest.approx(5.46, abs=0.05)


class TestIntrinsicAtExpiry:
    """Test intrinsic value at and after expiry."""

    def test_intrinsic_at_and_after_expiry(self):
        """At t=0 (expiry) and t<0 (after expiry), prices equal intrinsic value."""
        # Put: strike - spot = 100 - 80 = 20 when spot < strike
        assert bs_put(80.0, 100.0, 0.0, 0.3) == pytest.approx(20.0)
        assert bs_put(80.0, 100.0, -1.0, 0.3) == pytest.approx(20.0)

        # Call: spot - strike = 80 - 100 = 0 when spot < strike
        assert bs_call(80.0, 100.0, 0.0, 0.3) == pytest.approx(0.0)
        assert bs_call(80.0, 100.0, -1.0, 0.3) == pytest.approx(0.0)

        # Call: spot - strike = 120 - 100 = 20 when spot > strike
        assert bs_call(120.0, 100.0, 0.0, 0.3) == pytest.approx(20.0)


class TestDegenerateInputs:
    """Test handling of degenerate inputs (no volatility, zero prices)."""

    def test_degenerate_inputs_fall_back_to_intrinsic(self):
        """Degenerate inputs return intrinsic value without raising."""
        # iv=0: returns intrinsic
        assert bs_put(90.0, 100.0, 0.5, 0.0) == pytest.approx(10.0)

        # spot=0: put returns strike (intrinsic = strike - 0)
        assert bs_put(0.0, 100.0, 0.5, 0.3) == pytest.approx(100.0)

        # strike=0: call returns spot (intrinsic = spot - 0)
        assert bs_call(100.0, 0.0, 0.5, 0.3) == pytest.approx(100.0)


class TestPricesBounds:
    """Test that prices are non-negative and bounded correctly."""

    def test_prices_never_negative_and_bounded(self):
        """Call <= spot, Put <= strike, both >= 0."""
        spot_values = (1, 50, 100, 150, 1000)
        t_values = (0.01, 0.5, 2.0)

        for spot in spot_values:
            for t in t_values:
                # With strike = spot for ATM options
                put = bs_put(spot, spot, t, 0.3)
                call = bs_call(spot, spot, t, 0.3)

                assert 0 <= put <= spot
                assert 0 <= call <= spot

                # Also test with different strike
                put2 = bs_put(spot, 120.0, t, 0.3)
                call2 = bs_call(spot, 120.0, t, 0.3)

                assert 0 <= put2 <= 120.0
                assert 0 <= call2 <= spot


class TestMonotonicity:
    """Test monotonicity of prices w.r.t. spot and volatility."""

    def test_put_price_monotone_in_spot_and_vol(self):
        """Put falls as spot rises; both put and call rise with iv."""
        t = 0.5
        iv = 0.3
        strike = 100.0

        # Put price falls as spot rises (spot 90 > 100 > 110)
        put_90 = bs_put(90.0, strike, t, iv)
        put_100 = bs_put(100.0, strike, t, iv)
        put_110 = bs_put(110.0, strike, t, iv)

        assert put_90 > put_100 > put_110

        # Put rises with iv (0.2 < 0.4 < 0.8)
        put_iv_low = bs_put(100.0, strike, t, 0.2)
        put_iv_med = bs_put(100.0, strike, t, 0.4)
        put_iv_high = bs_put(100.0, strike, t, 0.8)

        assert put_iv_low < put_iv_med < put_iv_high

        # Call also rises with iv
        call_iv_low = bs_call(100.0, strike, t, 0.2)
        call_iv_med = bs_call(100.0, strike, t, 0.4)
        call_iv_high = bs_call(100.0, strike, t, 0.8)

        assert call_iv_low < call_iv_med < call_iv_high


class TestTimeValue:
    """Test that more time is worth more for calls."""

    def test_more_time_is_worth_more_for_calls(self):
        """At-the-money call value increases with time to expiry."""
        spot = 100.0
        strike = 100.0
        iv = 0.3

        call_01 = bs_call(spot, strike, 0.1, iv)
        call_05 = bs_call(spot, strike, 0.5, iv)
        call_10 = bs_call(spot, strike, 1.0, iv)

        assert call_01 < call_05 < call_10


class TestYearsTo:
    """Test years_to utility function."""

    def test_years_to(self):
        """Test years_to calculations."""
        now = time.time()

        # 1 year in the future
        assert years_to(now + 365 * 86400.0, now) == pytest.approx(1.0)

        # In the past should return 0 (never negative)
        assert years_to(now - 1.0, now) == 0.0

        # Same time returns 0
        assert years_to(now, now) == 0.0

        # Far future
        assert years_to(now + 365 * 2 * 86400.0, now) == pytest.approx(2.0)
