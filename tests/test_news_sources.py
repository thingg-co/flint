"""News for the universe actually modeled: feeds by asset mix, ticker matching that does not
fire on English words, Finnhub company news keyed by ticker, and ideas only for covered names."""
import asyncio
import re
import time
from types import SimpleNamespace

import flint.news as news
from flint.news import FinnhubNews, NewsHub, NewsSkimmer, asset_patterns, default_sources, score_headline


def test_feeds_follow_the_asset_mix():
    eq = [n for n, _ in default_sources(["AAPL", "MSFT"])]
    assert "CNBC Markets" in eq and "CoinDesk" not in eq
    both = [n for n, _ in default_sources(["AAPL", "BTC-USD"])]
    assert "CNBC Markets" in both and "CoinDesk" in both
    assert "CoinDesk" in [n for n, _ in default_sources(["BTC-USD"])]


def test_ticker_matching_is_conservative():
    rx = asset_patterns({"NVDA": "NVDA", "NOW": "NOW", "F": "F", "BTC-USD": "BTC", "MSFT": "MSFT"})
    assert "NOW" not in rx and "F" not in rx                                   # a word and a letter
    assert score_headline("NVDA shares jump after earnings beat", rx)[0] == ["NVDA"]
    assert score_headline("Nvidia shares jump right now", rx)[0] == []           # lower-case name is not the ticker
    assert "BTC" in score_headline("Bitcoin falls 5%", rx)[0]
    assert score_headline("Stocks slide as the Fed warns on inflation", rx)[1] is True   # market tone counts as generic


def test_skimmer_ideas_list_only_covered_names():
    sk = NewsSkimmer(["AAPL", "MSFT", "NVDA"], "", use_browser=False)
    per = {"AAPL": {"mentions": 2, "sentiment": 0.4, "attention": 0.3, "top": ["Apple beats"]},
           "MSFT": {"mentions": 0, "sentiment": 0.1, "attention": 0.1, "top": []},
           "NVDA": {"mentions": 0, "sentiment": 0.1, "attention": 0.1, "top": []}}
    ideas = sk._ideas(per, [])
    assert len(ideas) == 2 and ideas[0].startswith("AAPL: 2 headlines") and "2 names without direct coverage" in ideas[1]


class FakeClient:
    calls: list = []
    status = 200

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        FakeClient.calls.append((url.rsplit("/", 1)[-1], dict(params)))
        if url.endswith("/company-news"):
            s = params["symbol"]
            body = [{"headline": f"{s} surges on record demand", "source": "Wire", "url": "https://x/1", "datetime": time.time() - 300},
                    {"headline": f"{s} faces lawsuit", "source": "Wire", "url": "https://x/2", "datetime": time.time() - 200}]
        else:
            body = [{"headline": "Stocks rally as inflation cools", "source": "Wire", "url": "https://x/3", "datetime": time.time() - 100}]
        return SimpleNamespace(status_code=FakeClient.status, json=lambda: body)


def test_finnhub_company_news_is_keyed_by_ticker(monkeypatch):
    monkeypatch.setattr(news.httpx, "AsyncClient", FakeClient)
    FakeClient.calls = []
    fn = FinnhubNews(["AAPL", "MSFT", "BTC-USD"], lambda: "k", per_skim=1)
    r1 = asyncio.run(fn.skim())
    assert [c[1]["symbol"] for c in FakeClient.calls if c[0] == "company-news"] == ["AAPL"]
    assert r1["per_asset"]["AAPL"]["mentions"] == 2 and r1["per_asset"]["MSFT"]["mentions"] == 0
    assert "BTC-USD" not in r1["per_asset"]
    assert all(h["assets"] == ["AAPL"] for h in r1["headlines"] if "AAPL" in h["title"])
    assert any(h["generic"] and not h["assets"] for h in r1["headlines"])           # general market news carries the tone
    r2 = asyncio.run(fn.skim())                                                      # rotation: the next name
    assert [c[1]["symbol"] for c in FakeClient.calls if c[0] == "company-news"] == ["AAPL", "MSFT"]
    assert r2["per_asset"]["AAPL"]["mentions"] == 2 and r2["per_asset"]["MSFT"]["mentions"] == 2   # kept across passes
    assert not asyncio.run(FinnhubNews(["AAPL"], lambda: "").skim())["headlines"]      # no key: quiet


def test_hub_adds_finnhub_for_equities_and_reads_the_key_live():
    cfg = SimpleNamespace(news_off="", news_sources="", news_browser=False, av_news_minutes=30.0, finnhub_key="")
    hub = NewsHub(cfg, ["AAPL", "MSFT"], SimpleNamespace(key=""))
    assert "finnhub" in hub.sources and hub.enabled["finnhub"]
    cfg.finnhub_key = "late"
    assert hub.sources["finnhub"].key_getter() == "late"
    crypto = NewsHub(cfg, ["BTC-USD"], SimpleNamespace(key=""))
    assert "finnhub" not in crypto.sources
