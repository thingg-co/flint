"""Tests for signals.py pure functions: base_of, NAME_TO_TICKER, Guru13F._parse, Guru13F.signal,
DerivativesProvider.pair, and SignalHub.__init__/toggle/status/_council."""

import tempfile
from types import SimpleNamespace

import pytest

from flint.signals import (
    DerivativesProvider,
    Guru13F,
    SignalHub,
    base_of,
    GURUS,
    ETHOS_DIMS,
)


class TestBaseOf:
    """Tests for base_of function."""

    def test_btc_with_dash(self):
        assert base_of("BTC-USD") == "BTC"

    def test_aapl_no_dash(self):
        assert base_of("aapl") == "AAPL"

    def test_eth_with_dash(self):
        assert base_of("ETH-USDT") == "ETH"


class TestDerivativesProviderPair:
    """Tests for DerivativesProvider.pair method."""

    def test_pair_btc_usd(self):
        dp = DerivativesProvider()
        assert dp.pair("BTC-USD") == "BTCUSDT"

    def test_pair_eth_usdt(self):
        dp = DerivativesProvider()
        assert dp.pair("ETH-USDT") == "ETHUSDT"

    def test_pair_aapl(self):
        dp = DerivativesProvider()
        # Note: pair() always appends "USDT" regardless of symbol type
        assert dp.pair("AAPL") == "AAPLUSDT"


class TestGuru13FParse:
    """Tests for Guru13F._parse method."""

    def test_parse_13f_maps_issuers_and_puts(self):
        """Parse a 13F XML with three infoTable blocks."""
        xml = """<?xml version="1.0"?>
<ns1:superRoot xmlns:ns1="http://www.sec.gov">
    <ns1:infoTable>
        <ns1:nameOfIssuer>NVIDIA CORP</ns1:nameOfIssuer>
        <ns1:titleOfClass>COM</ns1:titleOfClass>
        <ns1:value>1,500</ns1:value>
    </ns1:infoTable>
    <ns1:infoTable>
        <ns1:nameOfIssuer>APPLE INC</ns1:nameOfIssuer>
        <ns1:titleOfClass>COM</ns1:titleOfClass>
        <ns1:value>2,000</ns1:value>
        <ns1:putCall>Put</ns1:putCall>
    </ns1:infoTable>
    <ns1:infoTable>
        <ns1:nameOfIssuer>UNKNOWN WIDGETS CO</ns1:nameOfIssuer>
        <ns1:titleOfClass>COM</ns1:titleOfClass>
        <ns1:value>300</ns1:value>
    </ns1:infoTable>
</ns1:superRoot>"""
        g = Guru13F("t", "Test", "123")
        rows = g._parse(xml)

        assert len(rows) == 3

        # NVIDIA row: ticker "NVDA", value 1500.0, put False
        nvidia = next(r for r in rows if r["issuer"] == "NVIDIA CORP")
        assert nvidia["ticker"] == "NVDA"
        assert nvidia["value"] == 1500.0
        assert nvidia["put"] is False

        # APPLE row: ticker "AAPL", value -2000.0, put True
        apple = next(r for r in rows if r["issuer"] == "APPLE INC")
        assert apple["ticker"] == "AAPL"
        assert apple["value"] == -2000.0
        assert apple["put"] is True

        # UNKNOWN row: ticker None, value 300.0
        unknown = next(r for r in rows if r["issuer"] == "UNKNOWN WIDGETS CO")
        assert unknown["ticker"] is None
        assert unknown["value"] == 300.0

    def test_parse_merges_duplicate_issuer_rows(self):
        """Two blocks for same issuer merge, one with putCall stays separate."""
        xml = """<?xml version="1.0"?>
<ns1:superRoot xmlns:ns1="http://www.sec.gov">
    <ns1:infoTable>
        <ns1:nameOfIssuer>NVIDIA CORP</ns1:nameOfIssuer>
        <ns1:titleOfClass>COM</ns1:titleOfClass>
        <ns1:value>100</ns1:value>
    </ns1:infoTable>
    <ns1:infoTable>
        <ns1:nameOfIssuer>NVIDIA CORP</ns1:nameOfIssuer>
        <ns1:titleOfClass>COM</ns1:titleOfClass>
        <ns1:value>250</ns1:value>
    </ns1:infoTable>
    <ns1:infoTable>
        <ns1:nameOfIssuer>NVIDIA CORP</ns1:nameOfIssuer>
        <ns1:titleOfClass>PUT</ns1:titleOfClass>
        <ns1:value>50</ns1:value>
        <ns1:putCall>Put</ns1:putCall>
    </ns1:infoTable>
</ns1:superRoot>"""
        g = Guru13F("t", "Test", "123")
        rows = g._parse(xml)

        # Should have two rows: one long (100+250=350), one put (-50)
        assert len(rows) == 2

        long_row = next(r for r in rows if not r["put"])
        assert long_row["value"] == 350.0

        put_row = next(r for r in rows if r["put"])
        assert put_row["value"] == -50.0

    def test_parse_handles_empty_and_garbage(self):
        """_parse returns empty list for empty string and invalid XML."""
        g = Guru13F("t", "Test", "123")

        assert g._parse("") == []
        assert g._parse("<html>nope</html>") == []

    def test_parse_handles_mixed_namespaces(self):
        """Parse with different namespace prefixes."""
        # Note: The code only strips ns1: and n1: prefixes, not ns2:
        # This test documents actual behavior - ns2: will not be stripped
        xml = """<?xml version="1.0"?>
<ns2:superRoot xmlns:ns2="http://www.sec.gov">
    <ns2:infoTable>
        <ns2:nameOfIssuer>APPLE INC</ns2:nameOfIssuer>
        <ns2:titleOfClass>COM</ns2:titleOfClass>
        <ns2:value>1,000</ns2:value>
    </ns2:infoTable>
</ns2:superRoot>"""
        g = Guru13F("t", "Test", "123")
        rows = g._parse(xml)

        # With ns2: prefix, the regex doesn't strip it, so infoTable won't be found
        # This documents actual behavior - only ns1: and n1: work
        assert rows == []


class TestGuru13FSignal:
    """Tests for Guru13F.signal method."""

    def test_signal_is_clipped_and_signed(self):
        """Signal clips to [-1,1], scales by 3, and handles base_of lookup."""
        g = Guru13F("t", "Test", "123")
        g._by_ticker = {"NVDA": 0.5, "AAPL": -0.1}

        # 0.5 * 3 = 1.5, clipped to 1.0
        assert g.signal("NVDA") == 1.0

        # -0.1 * 3 = -0.3
        assert g.signal("AAPL") == pytest.approx(-0.3)

        # MSFT not in _by_ticker -> 0.0
        assert g.signal("MSFT") == 0.0

        # nvda-usd uses base_of so it equals signal("NVDA")
        assert g.signal("nvda-usd") == g.signal("NVDA")


class TestSignalHubToggleAndStatus:
    """Tests for SignalHub.toggle and status methods."""

    def test_hub_toggle_and_status(self):
        """Test toggle, unknown provider, and status structure."""
        with tempfile.TemporaryDirectory() as tmp_path:
            cfg = SimpleNamespace(
                finnhub_key="",
                signals_off="wsb",
                state_dir=tmp_path,
            )
            hub = SignalHub(cfg, ["AAPL"])

            # wsb is off (in signals_off), market is on
            assert hub.enabled["wsb"] is False
            assert hub.enabled["market"] is True

            # Toggle wsb on
            result = hub.toggle("wsb", True)
            assert result == {"ok": True}
            assert hub.enabled["wsb"] is True

            # Toggle unknown provider
            result = hub.toggle("nope", True)
            assert result == {"error": "unknown signal provider"}

            # Check status() structure
            status = hub.status()
            assert isinstance(status, list)

            # Every entry has required keys
            for entry in status:
                assert "id" in entry
                assert "name" in entry
                assert "enabled" in entry
                assert "status" in entry

            # Contains market entry
            market_entry = next(e for e in status if e["id"] == "market")
            assert market_entry["name"] == "Whole-market scan"

            # Contains guru entries
            guru_ids = {g["id"] for g in GURUS}
            status_ids = {e["id"] for e in status}
            assert guru_ids.issubset(status_ids)


class TestSignalHubCouncil:
    """Tests for SignalHub._council method."""

    def test_council_averages_enabled_gurus(self):
        """Council averages ethos dimensions of enabled gurus."""
        with tempfile.TemporaryDirectory() as tmp_path:
            cfg = SimpleNamespace(
                finnhub_key="",
                signals_off="",
                state_dir=tmp_path,
            )
            hub = SignalHub(cfg, ["AAPL"])

            # All gurus disabled -> all dims 0.0
            for gid in hub.gurus:
                hub.enabled[gid] = False
            council = hub._council()
            assert council == {d: 0.0 for d in ETHOS_DIMS}

            # Enable exactly first guru
            first_guru_id = GURUS[0]["id"]
            for gid in hub.gurus:
                hub.enabled[gid] = gid == first_guru_id

            council = hub._council()
            for d in ETHOS_DIMS:
                assert council[d] == pytest.approx(GURUS[0]["ethos"][d], abs=1e-3)

            # Enable two gurus and check average
            second_guru_id = GURUS[1]["id"]
            for gid in hub.gurus:
                hub.enabled[gid] = gid in (first_guru_id, second_guru_id)

            council = hub._council()
            for d in ETHOS_DIMS:
                expected = round(
                    (GURUS[0]["ethos"][d] + GURUS[1]["ethos"][d]) / 2,
                    3,
                )
                assert council[d] == pytest.approx(expected, abs=1e-3)
