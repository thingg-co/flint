"""Tests for fetch_schwab_positions and _mask in flint/portfolio.py"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from flint.portfolio import _mask, fetch_schwab_positions


async def _get_token():
    return "tok"


class TestMask:
    def test_mask_formats_account_number(self):
        assert _mask("12345678") == "****5678"

    def test_mask_short_number(self):
        assert _mask("123") == "****"

    def test_mask_none(self):
        assert _mask(None) == "****"


class TestFetchSchwabPositions:
    """Test fetch_schwab_positions with mocked HTTP responses."""

    def test_accounts_are_masked_named_and_sorted(self):
        # Account A (MARGIN, $50k), Account B (CASH, $120k)
        # userPreference gives "Roth" nickname to account 12345678
        accounts_resp = SimpleNamespace(
            status_code=200,
            json=lambda: [
                {
                    "securitiesAccount": {
                        "accountNumber": "12345678",
                        "type": "MARGIN",
                        "currentBalances": {"liquidationValue": 50000.0},
                        "positions": [
                            {
                                "instrument": {
                                    "symbol": "AAPL",
                                    "assetType": "EQUITY",
                                },
                                "longQuantity": 10,
                                "shortQuantity": 0,
                                "averagePrice": 150.0,
                                "marketValue": 2000.0,
                                "longOpenProfitLoss": 500.0,
                            },
                            {
                                "instrument": {
                                    "symbol": "TSLA",
                                    "assetType": "EQUITY",
                                },
                                "longQuantity": 0,
                                "shortQuantity": 2,
                                "averagePrice": 100.0,
                                "marketValue": -180.0,
                                "longOpenProfitLoss": 0.0,
                                "currentDayProfitLoss": -20.0,
                            },
                            {
                                "instrument": {},
                                "longQuantity": 5,
                            },
                        ],
                    }
                },
                {
                    "securitiesAccount": {
                        "accountNumber": "99990000",
                        "type": "CASH",
                        "currentBalances": {"liquidationValue": 120000.0},
                        "positions": [],
                    }
                },
            ],
            raise_for_status=lambda: None,
        )
        userpref_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "accounts": [
                    {"accountNumber": "12345678", "nickName": "Roth"}
                ]
            },
            raise_for_status=lambda: None,
        )

        # Monkeypatch httpx.AsyncClient
        original_async_client = httpx.AsyncClient

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                self.methods_called = set()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                self.methods_called.add("get")
                if "accounts" in url:
                    return accounts_resp
                else:
                    return userpref_resp

        httpx.AsyncClient = MonkeyClient

        auth = SimpleNamespace(token=_get_token)

        try:
            result = asyncio.run(fetch_schwab_positions(auth))

            # Check structure
            assert "t" in result
            assert "accounts" in result

            # Check accounts are sorted by liquidation descending
            accounts = result["accounts"]
            assert accounts[0]["id"] == "****0000"  # CASH, $120k
            assert accounts[0]["name"] == "CASH"  # No nickname, falls back to type
            assert accounts[1]["id"] == "****5678"  # MARGIN, $50k
            assert accounts[1]["name"] == "Roth"  # From userPreference

            # Verify raw account numbers do not appear in JSON
            json_str = json.dumps(result)
            assert "12345678" not in json_str
            assert "99990000" not in json_str
        finally:
            httpx.AsyncClient = original_async_client

    def test_positions_parsed_and_sorted(self):
        # Account A (MARGIN, $50k) with positions
        accounts_resp = SimpleNamespace(
            status_code=200,
            json=lambda: [
                {
                    "securitiesAccount": {
                        "accountNumber": "12345678",
                        "type": "MARGIN",
                        "currentBalances": {"liquidationValue": 50000.0},
                        "positions": [
                            {
                                "instrument": {
                                    "symbol": "AAPL",
                                    "assetType": "EQUITY",
                                },
                                "longQuantity": 10,
                                "shortQuantity": 0,
                                "averagePrice": 150.0,
                                "marketValue": 2000.0,
                                "longOpenProfitLoss": 500.0,
                            },
                            {
                                "instrument": {
                                    "symbol": "TSLA",
                                    "assetType": "EQUITY",
                                },
                                "longQuantity": 0,
                                "shortQuantity": 2,
                                "averagePrice": 100.0,
                                "marketValue": -180.0,
                                "longOpenProfitLoss": 0.0,
                                "currentDayProfitLoss": -20.0,
                            },
                            {
                                "instrument": {},
                                "longQuantity": 5,
                            },
                        ],
                    }
                },
            ],
            raise_for_status=lambda: None,
        )
        userpref_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"accounts": []},
            raise_for_status=lambda: None,
        )

        original_async_client = httpx.AsyncClient

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                if "accounts" in url:
                    return accounts_resp
                else:
                    return userpref_resp

        httpx.AsyncClient = MonkeyClient

        auth = SimpleNamespace(token=_get_token)

        try:
            result = asyncio.run(fetch_schwab_positions(auth))
            accounts = result["accounts"]

            # Account ****5678 has exactly two positions (the one with no symbol is skipped)
            assert accounts[0]["id"] == "****5678"
            positions = accounts[0]["positions"]
            assert len(positions) == 2

            # AAPL first (higher absolute value: 2000 vs 180)
            assert positions[0]["symbol"] == "AAPL"
            assert positions[0]["qty"] == 10.0
            assert positions[0]["avg"] == 150.0
            assert positions[0]["value"] == 2000.0
            assert positions[0]["pnl"] == 500.0
            # pnl_pct = 500 / (10 * 150) * 100 = 33.33...
            assert positions[0]["pnl_pct"] == pytest.approx(500 / 1500 * 100, abs=0.01)
            assert positions[0]["asset_type"] == "EQUITY"

            # TSLA second
            assert positions[1]["symbol"] == "TSLA"
            assert positions[1]["qty"] == -2.0  # shortQuantity - longQuantity
            assert positions[1]["avg"] == 100.0
            assert positions[1]["value"] == -180.0
            assert positions[1]["pnl"] == -20.0  # currentDayProfitLoss (fallback when longOpenProfitLoss is 0)
            assert positions[1]["pnl_pct"] == pytest.approx(-20 / 200 * 100, abs=0.01)
            assert positions[1]["asset_type"] == "EQUITY"
        finally:
            httpx.AsyncClient = original_async_client

    def test_zero_cost_gives_zero_pct(self):
        # Position with averagePrice 0 should give pnl_pct 0.0 (no division by zero)
        accounts_resp = SimpleNamespace(
            status_code=200,
            json=lambda: [
                {
                    "securitiesAccount": {
                        "accountNumber": "11111111",
                        "type": "MARGIN",
                        "currentBalances": {"liquidationValue": 1000.0},
                        "positions": [
                            {
                                "instrument": {
                                    "symbol": "ZRO",
                                    "assetType": "EQUITY",
                                },
                                "longQuantity": 1,
                                "shortQuantity": 0,
                                "averagePrice": 0,
                                "marketValue": 10.0,
                                "longOpenProfitLoss": 10.0,
                            },
                        ],
                    }
                },
            ],
            raise_for_status=lambda: None,
        )
        userpref_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"accounts": []},
            raise_for_status=lambda: None,
        )

        original_async_client = httpx.AsyncClient

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                if "accounts" in url:
                    return accounts_resp
                else:
                    return userpref_resp

        httpx.AsyncClient = MonkeyClient

        auth = SimpleNamespace(token=_get_token)

        try:
            result = asyncio.run(fetch_schwab_positions(auth))
            positions = result["accounts"][0]["positions"]
            assert positions[0]["symbol"] == "ZRO"
            assert positions[0]["qty"] == 1.0
            assert positions[0]["avg"] == 0.0
            assert positions[0]["pnl"] == 10.0
            assert positions[0]["pnl_pct"] == 0.0  # No division by zero
        finally:
            httpx.AsyncClient = original_async_client

    def test_nickname_failure_is_tolerated(self):
        # userPreference get raises an exception; accounts still return with type fallback
        accounts_resp = SimpleNamespace(
            status_code=200,
            json=lambda: [
                {
                    "securitiesAccount": {
                        "accountNumber": "12345678",
                        "type": "MARGIN",
                        "currentBalances": {"liquidationValue": 50000.0},
                        "positions": [],
                    }
                },
            ],
            raise_for_status=lambda: None,
        )

        original_async_client = httpx.AsyncClient

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                if "accounts" in url:
                    return accounts_resp
                else:
                    # Simulate userPreference failing
                    raise RuntimeError("Network error")

        httpx.AsyncClient = MonkeyClient

        auth = SimpleNamespace(token=_get_token)

        try:
            result = asyncio.run(fetch_schwab_positions(auth))
            accounts = result["accounts"]

            # Both accounts returned with type fallback for name
            assert len(accounts) == 1
            assert accounts[0]["id"] == "****5678"
            assert accounts[0]["name"] == "MARGIN"  # Fallback to type when userPreference fails
        finally:
            httpx.AsyncClient = original_async_client

    def test_non_2xx_accounts_raises(self):
        # accounts response status 401 should raise httpx.HTTPStatusError
        auth = SimpleNamespace(token=_get_token)

        original_async_client = httpx.AsyncClient

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                # Simulate 401 error
                request = httpx.Request("GET", url)
                response = httpx.Response(401, request=request)
                return SimpleNamespace(
                    status_code=401,
                    raise_for_status=lambda: (
                        raise_(httpx.HTTPStatusError("401 error", request=request, response=response))
                    ),
                )

        def raise_(exc):
            raise exc

        httpx.AsyncClient = MonkeyClient

        try:
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(fetch_schwab_positions(auth))
        finally:
            httpx.AsyncClient = original_async_client

    def test_only_get_is_used(self):
        # Verify only "get" method is called on the client (read-only account access)
        accounts_resp = SimpleNamespace(
            status_code=200,
            json=lambda: [
                {
                    "securitiesAccount": {
                        "accountNumber": "12345678",
                        "type": "MARGIN",
                        "currentBalances": {"liquidationValue": 50000.0},
                        "positions": [],
                    }
                },
            ],
            raise_for_status=lambda: None,
        )
        userpref_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"accounts": []},
            raise_for_status=lambda: None,
        )

        original_async_client = httpx.AsyncClient

        methods_called = set()

        class MonkeyClient:
            def __init__(self, *args, **kwargs):
                self.methods_called = methods_called

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def get(self, url, headers=None, params=None):
                self.methods_called.add("get")
                if "accounts" in url:
                    return accounts_resp
                else:
                    return userpref_resp

        httpx.AsyncClient = MonkeyClient

        auth = SimpleNamespace(token=_get_token)

        try:
            result = asyncio.run(fetch_schwab_positions(auth))

            # Only "get" method was called
            assert methods_called == {"get"}
        finally:
            httpx.AsyncClient = original_async_client
