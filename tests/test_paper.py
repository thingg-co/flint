"""Tests for PaperBook in flint/paper.py"""
import pytest

from flint.paper import PaperBook


ts = 1_800_000_000.0


class TestPaperBook:
    def test_short_target_never_shorts_stock(self):
        # First rebalance with negative target and no put_quotes
        # Should not trade because short requires put_quotes
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        targets = {"AAPL": -0.1}
        prices = {"AAPL": 200.0}
        book.rebalance(targets, prices, ts)

        assert book.pos.get("AAPL", 0.0) == 0.0
        assert "AAPL" not in book.puts
        assert book.cash == 100_000.0

        # Rebalance again with put_quotes for AAPL
        put_quotes = {
            "AAPL": {
                "premium": 5.0,
                "delta": -0.45,
                "strike": 195.0,
                "expiry_ts": ts + 30 * 86400,
                "iv": 0.4
            }
        }
        book.rebalance(targets, prices, ts, put_quotes=put_quotes)

        assert "AAPL" in book.puts
        assert book.puts["AAPL"]["contracts"] > 0
        assert "AAPL" not in book.pos  # No share position

        # Assert every share position is >= 0
        for sym, pos in book.pos.items():
            assert pos >= 0

    def test_put_premium_capped_by_option_max_frac(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        targets = {"AAPL": -0.15}
        prices = {"AAPL": 200.0}
        put_quotes = {
            "AAPL": {
                "premium": 5.0,
                "delta": -0.45,
                "strike": 195.0,
                "expiry_ts": ts + 30 * 86400,
                "iv": 0.4
            }
        }
        book.rebalance(targets, prices, ts, put_quotes=put_quotes)

        # Premium at risk capped as a share of equity
        max_premium_allowed = 100_000.0 * 0.02 + 1e-6
        actual_premium = book.puts["AAPL"]["contracts"] * 5.0 * 100.0
        assert actual_premium <= max_premium_allowed

    def test_short_closes_long_stock_first(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}

        # First open a long position
        targets = {"AAPL": 0.1}
        book.rebalance(targets, prices, ts)

        assert book.pos.get("AAPL", 0.0) > 0

        # Then rebalance to short with put_quotes
        put_quotes = {
            "AAPL": {
                "premium": 5.0,
                "delta": -0.45,
                "strike": 195.0,
                "expiry_ts": ts + 30 * 86400,
                "iv": 0.4
            }
        }
        targets = {"AAPL": -0.1}
        book.rebalance(targets, prices, ts, put_quotes=put_quotes)

        # Long stock should be closed first, then short position opened via put
        assert "AAPL" not in book.pos
        assert "AAPL" in book.puts

    def test_fill_fee_and_spread(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        targets = {"AAPL": 0.1}
        prices = {"AAPL": 200.0}
        quotes = {"AAPL": {"bid": 199.9, "ask": 200.1}}

        book.rebalance(targets, prices, ts, quotes=quotes)

        # The buy must fill at the ask
        assert book.trades[0]["price"] == 200.1
        assert book.trades[0]["side"] == "buy"

        trade_notional = book.trades[0]["notional"]
        expected_fee = trade_notional * 8.0 / 1e4
        assert book.fees == pytest.approx(expected_fee)

        shares = book.trades[0]["shares"]
        expected_spread_cost = shares * 0.1
        assert book.spread_cost == pytest.approx(expected_spread_cost, rel=1e-3)

    def test_weights_clipped_and_no_leverage(self):
        book = PaperBook(start=100_000.0, cost_bps=0.0, max_weight=0.15)

        # Targets of 0.5 on eight symbols (total 4.0, way over 1.0)
        prices = {f"S{i}": 100.0 for i in range(8)}
        targets = {f"S{i}": 0.5 for i in range(8)}

        book.rebalance(targets, prices, ts)

        # Each position notional must be <= 0.15 * 100_000
        max_weight_notional = 0.15 * 100_000 + 1e-6
        for sym in prices.keys():
            pos = book.pos.get(sym, 0.0)
            notional = pos * prices[sym]
            assert notional <= max_weight_notional

        # Sum of all position notionals must be <= 100_000 (gross clipped to 1)
        total_gross = sum(book.pos.get(sym, 0.0) * prices[sym] for sym in prices.keys())
        assert total_gross <= 100_000 + 1e-6

    def test_small_rebalance_skipped(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        targets = {"AAPL": 0.001}  # 100 notional, below the 200 floor
        prices = {"AAPL": 200.0}

        book.rebalance(targets, prices, ts)

        assert "AAPL" not in book.pos
        assert len(book.trades) == 0

    def test_round_trip_costs_only_fees(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}

        # Buy
        targets = {"AAPL": 0.1}
        book.rebalance(targets, prices, ts)

        initial_fees = book.fees
        assert initial_fees > 0

        # Sell (close position)
        targets = {"AAPL": 0.0}
        book.rebalance(targets, prices, ts)

        assert "AAPL" not in book.pos

        # Equity should be start minus fees (no position left)
        final_equity = book.equity(prices, ts)
        expected_equity = 100_000.0 - book.fees
        assert final_equity == pytest.approx(expected_equity, rel=1e-9)

    def test_state_round_trip(self):
        book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)
        prices = {"AAPL": 200.0}

        # Open a long position
        targets = {"AAPL": 0.1}
        book.rebalance(targets, prices, ts)

        # Open a put
        put_quotes = {
            "AAPL": {
                "premium": 5.0,
                "delta": -0.45,
                "strike": 195.0,
                "expiry_ts": ts + 30 * 86400,
                "iv": 0.4
            }
        }
        targets = {"AAPL": -0.1}
        book.rebalance(targets, prices, ts, put_quotes=put_quotes)

        # Serialize
        d = book.to_state()

        # Deserialize into a new book
        b2 = PaperBook()
        b2.load_state(d)

        assert b2.cash == pytest.approx(book.cash)
        assert b2.pos == book.pos
        assert b2.puts["AAPL"]["contracts"] == pytest.approx(book.puts["AAPL"]["contracts"])
