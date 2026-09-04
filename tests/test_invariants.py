"""Tests that pin the project's hard rules from CLAUDE.md"""
import re
from pathlib import Path

import numpy
import pytest

from flint.config import Config
from flint.paper import PaperBook

TESTS_DIR = Path(__file__).resolve().parents[1]
FLINT_DIR = TESTS_DIR / "flint"


def test_flint_never_places_orders():
    """Flint never places, changes, or cancels orders on any provider."""
    order_endpoints = ["/orders", "placeorder", "previeworder", "cancelorder", "replaceorder"]

    for pyfile in FLINT_DIR.glob("*.py"):
        source = pyfile.read_text()
        source_lower = source.lower()

        # Check for order endpoints in all Python files
        for endpoint in order_endpoints:
            if endpoint in source_lower:
                # Find the line number
                for i, line in enumerate(source.split("\n"), 1):
                    if endpoint in line.lower():
                        raise AssertionError(
                            f"{pyfile.name}:{i} contains order endpoint '{endpoint}'"
                        )

    # Additional checks for portfolio.py - only GET, no .post/.put/.delete/.patch
    portfolio_source = (FLINT_DIR / "portfolio.py").read_text()
    for method in [".post(", ".put(", ".delete(", ".patch("]:
        if method in portfolio_source:
            for i, line in enumerate(portfolio_source.split("\n"), 1):
                if method in line:
                    raise AssertionError(
                        f"flint/portfolio.py:{i} contains {method} - should only use GET"
                    )

    # Check schwab.py - all .post() calls must target TOKEN_URL
    schwab_source = (FLINT_DIR / "schwab.py").read_text()
    for i, line in enumerate(schwab_source.split("\n"), 1):
        if ".post(" in line and "TOKEN_URL" not in line:
            # Allow .post() if it's in a comment or string
            if not (line.strip().startswith("#") or '"""' in line or "'''" in line):
                raise AssertionError(
                    f"flint/schwab.py:{i} contains .post() that does not target TOKEN_URL: {line.strip()}"
                )


def test_brief_backend_defaults_are_local():
    """The narrative brief runs on a local LLM backend only."""
    cfg = Config()

    # Check brief_openai_base defaults to localhost
    from urllib.parse import urlparse
    host = urlparse(cfg.brief_openai_base).hostname
    assert host in ("localhost", "127.0.0.1"), f"brief_openai_base hostname is {host}, expected localhost or 127.0.0.1"

    # Check ollama_host defaults to localhost
    ollama_host = urlparse(cfg.ollama_host).hostname
    assert ollama_host in ("localhost", "127.0.0.1"), f"ollama_host hostname is {ollama_host}, expected localhost or 127.0.0.1"

    # Check source text of flint/brief.py does not contain cloud API domains
    brief_source = (FLINT_DIR / "brief.py").read_text()
    cloud_domains = ["api.openai.com", "anthropic.com", "googleapis.com", "openrouter.ai", "api.together", "groq.com"]
    for domain in cloud_domains:
        assert domain not in brief_source, f"flint/brief.py contains {domain} - cloud APIs not allowed"


def test_config_builds_from_an_empty_environment(monkeypatch):
    """Config constructs without environment variables set."""
    # Delete every FLINT_ environment variable
    for key in list(monkeypatch.getenv("FLINT_")) if hasattr(monkeypatch, 'getenv') else []:
        if key.startswith("FLINT_"):
            monkeypatch.delenv(key, raising=False)

    # Alternative approach: set them to empty
    import os
    for key in os.environ:
        if key.startswith("FLINT_"):
            monkeypatch.delenv(key, raising=False)

    # Also clear any that might have been set
    cfg = Config()

    assert isinstance(cfg.port, int)
    assert isinstance(cfg.symbols, list)
    assert len(cfg.symbols) > 0
    assert cfg.extended_hours is False
    assert cfg.brief_enabled is False


def test_paper_book_never_holds_short_stock_or_leverage():
    """PaperBook constraints: no short stock, no leverage, defined option risk."""
    book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)

    # Generate test symbols
    symbols = [f"S{i}" for i in range(3)]
    rng = numpy.random.default_rng(0)
    ts = 1_800_000_000.0

    prices = {s: 100.0 for s in symbols}
    targets = {}

    # Run 300 steps
    for step in range(300):
        # Prices drift multiplicatively
        for s in symbols:
            drift = numpy.exp(rng.normal(0, 0.01))
            prices[s] = prices[s] * drift

        # Targets are rng.uniform(-0.3, 0.3), about 1/3 at 0
        targets = {}
        for s in symbols:
            if rng.random() < 0.33:
                targets[s] = 0.0
            else:
                targets[s] = rng.uniform(-0.3, 0.3)

        # Quotes: bid/ask at price*(1 +/- 0.0005)
        quotes = {
            s: {"bid": prices[s] * (1 - 0.0005), "ask": prices[s] * (1 + 0.0005)}
            for s in symbols
        }

        # Put quotes for every symbol
        put_quotes = {}
        for s in symbols:
            price = prices[s]
            put_quotes[s] = {
                "premium": max(0.5, 0.03 * price),
                "delta": -0.45,
                "strike": round(price * 0.98, 2),
                "expiry_ts": ts + 30 * 86400,
                "iv": 0.4
            }

        # Every 20th step also straddle_targets
        straddle_targets = {}
        straddle_quotes = {}
        if step % 20 == 0:
            for s in symbols:
                straddle_targets[s] = 0.05
                price = prices[s]
                straddle_quotes[s] = {
                    "put_premium": max(0.5, 0.03 * price),
                    "call_premium": max(0.5, 0.03 * price),
                    "strike": round(price, 2),
                    "put_iv": 0.4,
                    "call_iv": 0.4,
                    "expiry_ts": ts + 30 * 86400
                }

        book.rebalance(targets, prices, ts, quotes=quotes, put_quotes=put_quotes,
                      straddle_targets=straddle_targets, straddle_quotes=straddle_quotes)

        # After every rebalance assert:
        # 1. Every value in book.pos is > 0 (no short stock)
        for sym, pos in book.pos.items():
            assert pos > 0, f"Short stock found: {sym} = {pos}"

        # 2. Every put and straddle has contracts > 0
        for sym, pt in book.puts.items():
            assert pt["contracts"] > 0, f"Put contracts <= 0 for {sym}: {pt['contracts']}"

        for sym, st in book.straddles.items():
            assert st["contracts"] > 0, f"Straddle contracts <= 0 for {sym}: {st['contracts']}"

        # 3. book.cash is finite
        assert numpy.isfinite(book.cash), "book.cash is not finite"

        # 4. No leverage: sum(pos * price) <= book.equity * (1 + 1e-6) + 1e-6
        stock_notional = sum(book.pos[s] * prices.get(s, 0) for s in book.pos)
        equity = book.equity(prices, ts)
        assert stock_notional <= equity * (1 + 1e-6) + 1e-6, \
            f"Leverage violation: stock_notional {stock_notional} > equity {equity}"

        # 5. Option premium at risk <= option_max_frac of equity (dynamic bound)
        equity = book.equity(prices, ts)
        for sym, pt in book.puts.items():
            premium_at_risk = pt["contracts"] * pt["entry_prem"] * 100
            max_allowed = 0.02 * equity + 1e-6
            assert premium_at_risk <= max_allowed, \
                f"Option premium too high for {sym}: {premium_at_risk} > {max_allowed}"

        for sym, st in book.straddles.items():
            premium_at_risk = st["contracts"] * st["entry_prem"] * 100
            max_allowed = 0.02 * equity + 1e-6
            assert premium_at_risk <= max_allowed, \
                f"Straddle premium too high for {sym}: {premium_at_risk} > {max_allowed}"

        ts += 300

    # At the end assert book.n_trades > 0
    assert book.n_trades > 0, "The fuzz did not trade"


def test_defined_risk_the_book_cannot_lose_more_than_it_paid_for_options():
    """Options never mark negative; equity >= cash - premium paid."""
    # Start with a new book
    book = PaperBook(start=100_000.0, cost_bps=8.0, option_max_frac=0.02)

    ts = 1_800_000_000.0

    # Open one put and one straddle at price 200
    prices_200 = {"S0": 200.0}

    # Put quote
    put_quote = {
        "premium": 5.0,
        "delta": -0.45,
        "strike": 195.0,
        "expiry_ts": ts + 30 * 86400,
        "iv": 0.4
    }

    # Straddle quote
    straddle_quote = {
        "put_premium": 5.0,
        "call_premium": 5.0,
        "strike": 200.0,
        "put_iv": 0.4,
        "call_iv": 0.4,
        "expiry_ts": ts + 30 * 86400
    }

    # Open put first
    targets = {"S0": -0.1}  # Short via put
    book.rebalance(targets, prices_200, ts, put_quotes={"S0": put_quote})

    # At this point, we have a put but no straddle yet
    # Mark at price 1.0 and 10_000.0 with just the put
    equity_low = book.equity({"S0": 1.0}, ts)
    equity_high = book.equity({"S0": 10000.0}, ts)
    cash = book.cash

    # Both equities must be >= cash - 1e-6 (options never mark negative)
    # Even at price 10000, the put's intrinsic value is negative (max(strike - spot, 0) = 0)
    # So equity should be >= cash (since put is worth 0, and we paid for it)
    assert equity_low >= cash - 1e-6, \
        f"Equity at low price ({equity_low}) < cash - epsilon ({cash - 1e-6})"
    assert equity_high >= cash - 1e-6, \
        f"Equity at high price ({equity_high}) < cash - epsilon ({cash - 1e-6})"

    # Both equities must be finite
    assert numpy.isfinite(equity_low), "Equity at low price is not finite"
    assert numpy.isfinite(equity_high), "Equity at high price is not finite"

    # Now open straddle - it will close the put immediately (since target goes positive)
    targets = {"S0": 0.05}  # Straddle (this will close the put since target is positive)
    book.rebalance(targets, prices_200, ts, straddle_targets={"S0": 0.05},
                  straddle_quotes={"S0": straddle_quote})

    # After straddle is opened (and put is closed), verify equity is still finite
    equity_final = book.equity({"S0": 200.0}, ts)
    assert numpy.isfinite(equity_final), "Final equity is not finite"
    assert equity_final >= cash - 1e-6, "Final equity < cash - epsilon"
