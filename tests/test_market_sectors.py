"""Tests for sector bookkeeping in flint/market.py: SECTOR_OF, clip, and MarketScanner."""
import asyncio
import json
import os

import pytest

from flint.market import MarketScanner, SECTOR_OF, clip


class TestClip:
    """Tests for the clip function."""

    def test_clip_positive(self):
        assert clip(2.0) == 1.0
        assert clip(0.3) == 0.3

    def test_clip_negative(self):
        assert clip(-2.0) == -1.0

    def test_clip_custom_bounds(self):
        assert clip(5, 0, 2) == 2


class TestSectorOf:
    """Tests for SECTOR_OF mapping."""

    def test_folds_industries(self):
        assert SECTOR_OF["Semiconductors"] == "Technology"
        assert SECTOR_OF["Banking"] == "Financials"
        assert SECTOR_OF["Biotechnology"] == "Health Care"

    def test_all_values_are_valid_sectors(self):
        valid_sectors = set(SECTOR_OF.values())
        # Assert there are at most 12 GICS-style sectors
        assert len(valid_sectors) <= 12
        assert "Technology" in valid_sectors


class TestMarketScannerInit:
    """Tests for MarketScanner.__init__."""

    def test_loads_persisted_sectors(self, tmp_path):
        # Write a persisted sector file
        sector_file = tmp_path / "sectors.json"
        sector_file.write_text(json.dumps({"AAPL": "Technology"}))

        scanner = MarketScanner(state_dir=str(tmp_path))
        assert scanner.sectors == {"AAPL": "Technology"}
        assert scanner.sector_file == str(sector_file)

    def test_corrupt_file_gives_empty_dict(self, tmp_path):
        sector_file = tmp_path / "sectors.json"
        sector_file.write_bytes(b"{not json")

        scanner = MarketScanner(state_dir=str(tmp_path))
        assert scanner.sectors == {}

    def test_no_state_dir_gives_empty_sectors(self):
        scanner = MarketScanner(state_dir="")
        assert scanner.sectors == {}
        assert scanner.sector_file == ""


class TestFillSectors:
    """Tests for MarketScanner._fill_sectors."""

    def test_fill_sectors_folds_and_persists(self, tmp_path):
        """Test that _fill_sectors correctly maps industries to sectors and persists."""
        scanner = MarketScanner(finnhub_key="k", state_dir=str(tmp_path))

        # Track what symbols were requested
        requested_symbols = []

        class FakeClient:
            async def get(self, url, params=None):
                symbol = params.get("symbol") if params else None
                requested_symbols.append(symbol)

                # Scripted responses
                responses = {
                    "NVDA": {"status_code": 200, "json_data": {"finnhubIndustry": "Semiconductors"}},
                    "JPM": {"status_code": 200, "json_data": {"finnhubIndustry": "Banking"}},
                    "XYZ": {"status_code": 200, "json_data": {"finnhubIndustry": "Something Odd"}},
                    "ZZZ": {"status_code": 200, "json_data": {}},  # Empty industry
                }
                resp = responses.get(symbol, {"status_code": 404, "json_data": {}})
                return type("SimpleNamespace", (), {
                    "status_code": resp["status_code"],
                    "json": lambda *args: resp["json_data"]
                })()

        # Run the async method
        asyncio.run(scanner._fill_sectors(FakeClient(), ["NVDA", "JPM", "XYZ", "ZZZ"]))

        # Verify sector mapping
        assert scanner.sectors["NVDA"] == "Technology"  # Semiconductors -> Technology
        assert scanner.sectors["JPM"] == "Financials"   # Banking -> Financials
        assert scanner.sectors["XYZ"] == "Something Odd"  # Unknown industry kept as-is
        assert scanner.sectors["ZZZ"] == "Other"  # Empty industry -> Other

        # Verify the sector file was written
        sector_file = tmp_path / "sectors.json"
        assert sector_file.exists()
        assert json.loads(sector_file.read_text()) == scanner.sectors

    def test_fill_sectors_skips_known_and_respects_budget(self, tmp_path):
        """Test that known symbols are skipped and budget is respected."""
        scanner = MarketScanner(finnhub_key="k", state_dir=str(tmp_path))
        # Pre-seed with one symbol
        scanner.sectors = {"AAPL": "Technology"}

        requested_symbols = []

        class FakeClient:
            async def get(self, url, params=None):
                requested_symbols.append(params.get("symbol"))
                return type("SimpleNamespace", (), {
                    "status_code": 200,
                    "json": lambda *args: {"finnhubIndustry": "Banking"}
                })()

        # Request AAPL + 50 new symbols with budget=3
        syms = ["AAPL"] + [f"S{i}" for i in range(50)]
        asyncio.run(scanner._fill_sectors(FakeClient(), syms, budget=3))

        # AAPL was skipped, only first 3 unknown symbols were requested
        assert requested_symbols == ["S0", "S1", "S2"]

    def test_fill_sectors_stops_on_rate_limit(self, tmp_path):
        """Test that a 429 response stops the loop."""
        scanner = MarketScanner(finnhub_key="k", state_dir=str(tmp_path))

        requested_symbols = []

        class FakeClient:
            async def get(self, url, params=None):
                symbol = params.get("symbol")
                requested_symbols.append(symbol)

                # S0 -> 200, S1 -> 429 (rate limit), S2 -> never reached
                if symbol == "S0":
                    return type("SimpleNamespace", (), {
                        "status_code": 200,
                        "json": lambda *args: {"finnhubIndustry": "Banking"}
                    })()
                elif symbol == "S1":
                    return type("SimpleNamespace", (), {
                        "status_code": 429
                    })()
                else:
                    return type("SimpleNamespace", (), {
                        "status_code": 200,
                        "json": lambda *args: {"finnhubIndustry": "Energy"}
                    })()

        asyncio.run(scanner._fill_sectors(FakeClient(), ["S0", "S1", "S2"]))

        # S0 was fetched, S1 triggered 429 and stopped the loop
        assert requested_symbols == ["S0", "S1"]
        assert scanner.sectors.get("S0") == "Financials"  # Banking -> Financials
        assert "S1" not in scanner.sectors
        assert "S2" not in scanner.sectors

    def test_fill_sectors_without_key_is_noop(self, tmp_path):
        """Test that empty finnhub_key makes _fill_sectors a no-op."""
        scanner = MarketScanner(finnhub_key="", state_dir=str(tmp_path))

        class FakeClient:
            called = False

            async def get(self, url, params=None):
                FakeClient.called = True
                return type("SimpleNamespace", (), {
                    "status_code": 200,
                    "json": lambda *args: {"finnhubIndustry": "Banking"}
                })()

        asyncio.run(scanner._fill_sectors(FakeClient(), ["AAPL"]))

        assert not FakeClient.called
        assert scanner.sectors == {}
