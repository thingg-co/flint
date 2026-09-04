"""Tests for PaperBook options: straddles and expiry logic."""
import pytest

from flint.paper import PaperBook

ts = 1_800_000_000.0
DAY = 86400.0


class TestPaperBookStraddles:
    """Tests for straddle opening, closing, budget capping, and expiry."""

    def test_straddle_budget_capped_by_option_max_frac(self):
        # Book: PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        # prices = {"AAPL": 200.0}
        # Straddle quote SQ = {"put_premium": 4.0, "call_premium": 4.0, "strike": 200.0,
        #                     "expiry_ts": ts + 30 * DAY, "put_iv": 0.4, "call_iv": 0.4}
        # Open a straddle with book.rebalance({}, prices, ts,
        #     straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})

        st = book.straddles["AAPL"]
        # Budget capped by option_max_frac: contracts * 8.0 * 100.0 <= 100_000.0 * 0.02 + 1e-6
        assert st["contracts"] * 8.0 * 100.0 <= 100_000.0 * 0.02 + 1e-6

        # Newest trade has side "buy straddle"
        assert book.trades[0]["side"] == "buy straddle"

    def test_straddle_loss_is_bounded_by_premium(self):
        # Open the straddle; cash_after = book.cash
        # For every spot in (50.0, 100.0, 150.0, 200.0, 300.0, 400.0):
        # book.equity({"AAPL": spot}, ts + 10 * DAY) >= cash_after - 1e-6
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})
        cash_after = book.cash

        for spot in (50.0, 100.0, 150.0, 200.0, 300.0, 400.0):
            # The pair is never worth less than zero, so the worst case is the premium plus fees already paid
            equity = book.equity({"AAPL": spot}, ts + 10 * DAY)
            assert equity >= cash_after - 1e-6

    def test_straddle_respects_min_hold(self):
        # Default straddle_min_hold_s is 3600. After opening, rebalance({}, prices, ts + 600, straddle_targets={})
        # leaves the straddle open; rebalance({}, prices, ts + 3601, straddle_targets={})
        # closes it and the newest trade has side "sell straddle"
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})
        assert "AAPL" in book.straddles

        # Rebalance at ts + 600 (10 minutes, less than 3600s) with empty targets
        book.rebalance({}, prices, ts + 600, straddle_targets={})
        # Straddle should still be open (min hold not reached)
        assert "AAPL" in book.straddles

        # Rebalance at ts + 3601 (1 hour + 1 second, past min hold)
        book.rebalance({}, prices, ts + 3601, straddle_targets={})
        # Straddle should be closed
        assert "AAPL" not in book.straddles

        # Newest trade has side "sell straddle"
        assert book.trades[0]["side"] == "sell straddle"

    def test_straddle_persists_while_signal_stays(self):
        # After opening, rebalance at ts + 4000 with straddle_targets {"AAPL": 0.05}
        # and the same quotes keeps exactly one straddle (same contracts as before)
        # and adds no new "buy straddle" trade
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})
        initial_contracts = book.straddles["AAPL"]["contracts"]
        buy_straddle_count = sum(1 for t in book.trades if t["side"] == "buy straddle")

        # Rebalance at ts + 4000 with same straddle target (min hold already passed)
        book.rebalance({}, prices, ts + 4000, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})

        # Exactly one straddle (same contracts as before)
        assert "AAPL" in book.straddles
        assert book.straddles["AAPL"]["contracts"] == initial_contracts

        # No new "buy straddle" trade
        new_buy_straddle_count = sum(1 for t in book.trades if t["side"] == "buy straddle")
        assert new_buy_straddle_count == buy_straddle_count


class TestPaperBookExpiry:
    """Tests for expired option settlement."""

    def test_expired_put_settles_at_intrinsic(self):
        # Open the put; contracts = book.puts["AAPL"]["contracts"]; cash_before = book.cash
        # rebalance({}, {"AAPL": 150.0}, ts + 31 * DAY):
        #   "AAPL" not in book.puts
        #   book.cash == pytest.approx(cash_before + (195.0 - 150.0) * contracts * 100.0 - contracts * book.option_commission)
        #   newest trade has side "sell put"
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        PQ = {
            "premium": 5.0,
            "delta": -0.45,
            "strike": 195.0,
            "expiry_ts": ts + 30 * DAY,
            "iv": 0.4,
        }

        book.rebalance({"AAPL": -0.1}, prices, ts, put_quotes={"AAPL": PQ})
        contracts = book.puts["AAPL"]["contracts"]
        cash_before = book.cash

        # Advance past expiry
        book.rebalance({}, {"AAPL": 150.0}, ts + 31 * DAY)

        # Put should be removed
        assert "AAPL" not in book.puts

        # Cash should be updated: intrinsic value = strike - spot = 195 - 150 = 45
        # Cash = cash_before + intrinsic * contracts * 100 - fees
        expected_intrinsic = (195.0 - 150.0) * contracts * 100.0
        expected_fee = contracts * book.option_commission
        expected_cash = cash_before + expected_intrinsic - expected_fee
        assert book.cash == pytest.approx(expected_cash)

        # Newest trade has side "sell put"
        assert book.trades[0]["side"] == "sell put"

    def test_expired_straddle_at_the_strike_is_worthless(self):
        # Open the straddle; cash_before = book.cash
        # rebalance({}, {"AAPL": 200.0}, ts + 31 * DAY):
        #   no straddle left
        #   book.cash == pytest.approx(cash_before - book.straddles_fee_estimate)
        #   where fee = contracts * option_commission * 2
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})
        st = book.straddles["AAPL"]
        contracts = st["contracts"]
        cash_before = book.cash

        # At expiry with spot == strike, straddle is worthless (intrinsic = 0)
        # fee = contracts * option_commission * 2
        expected_fee = contracts * book.option_commission * 2.0

        # Advance past expiry
        book.rebalance({}, {"AAPL": 200.0}, ts + 31 * DAY)

        # No straddle left
        assert "AAPL" not in book.straddles

        # Cash = cash_before - fee (value is 0 at-the-money)
        assert book.cash == pytest.approx(cash_before - expected_fee)


class TestPaperBookMinHold:
    """Tests for put and straddle minimum hold periods."""

    def test_put_min_hold(self):
        # PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02, put_min_hold_s=1800.0)
        # open the put; rebalance({"AAPL": 0.0}, prices, ts + 600) keeps the put
        # rebalance({"AAPL": 0.0}, prices, ts + 1801) closes it
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02, put_min_hold_s=1800.0)
        prices = {"AAPL": 200.0}
        PQ = {
            "premium": 5.0,
            "delta": -0.45,
            "strike": 195.0,
            "expiry_ts": ts + 30 * DAY,
            "iv": 0.4,
        }

        book.rebalance({"AAPL": -0.1}, prices, ts, put_quotes={"AAPL": PQ})
        assert "AAPL" in book.puts

        # Rebalance at ts + 600 (10 minutes, less than 1800s)
        book.rebalance({"AAPL": 0.0}, prices, ts + 600)
        # Put should still be held (min hold not reached)
        assert "AAPL" in book.puts

        # Rebalance at ts + 1801 (30 min + 1 second, past min hold)
        book.rebalance({"AAPL": 0.0}, prices, ts + 1801)
        # Put should be closed
        assert "AAPL" not in book.puts


class TestPaperBookEquity:
    """Tests for equity calculation marking options to market."""

    def test_equity_marks_options(self):
        # open the put; book.equity(prices, ts) == pytest.approx(book.cash + book._put_value("AAPL", prices, ts))
        # then also open a straddle at the same ts and equity == pytest.approx(book.cash + book._put_value(...) + book._straddle_value("AAPL", prices, ts))
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}
        PQ = {
            "premium": 5.0,
            "delta": -0.45,
            "strike": 195.0,
            "expiry_ts": ts + 30 * DAY,
            "iv": 0.4,
        }
        SQ = {
            "put_premium": 4.0,
            "call_premium": 4.0,
            "strike": 200.0,
            "expiry_ts": ts + 30 * DAY,
            "put_iv": 0.4,
            "call_iv": 0.4,
        }

        # Open the put
        book.rebalance({"AAPL": -0.1}, prices, ts, put_quotes={"AAPL": PQ})

        # Equity should be cash + put value
        equity = book.equity(prices, ts)
        put_value = book._put_value("AAPL", prices, ts)
        assert equity == pytest.approx(book.cash + put_value)

        # Open a straddle at the same ts
        book.rebalance({}, prices, ts, straddle_targets={"AAPL": 0.05}, straddle_quotes={"AAPL": SQ})

        # Equity should be cash + put value + straddle value
        straddle_value = book._straddle_value("AAPL", prices, ts)
        equity = book.equity(prices, ts)
        assert equity == pytest.approx(book.cash + put_value + straddle_value)
