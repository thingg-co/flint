"""Keys saved from the Control Panel: written to the file the config reads on the next start,
and applied live to every consumer. Exists because set_key raised NameError on every service
(2026-10-10), so no key could be entered without a restart, and a Finnhub key entered at
runtime never reached the market-status poll, the radar or the fundamentals signal."""
import asyncio
import json
import stat
import types

import pytest

import flint.engine as eng
from flint.config import Config
from flint.engine import KEY_SERVICES, Engine
from flint.signals import FundamentalsProvider


def fake_engine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLINT_FINNHUB_KEY", raising=False)
    ns = types.SimpleNamespace
    log = {"emitted": [], "published": [], "toggled": []}
    fake = ns(
        cfg=Config(),
        sources=ns(sources={"finnhub": ns(key="", note="no API key"), "alpaca": ns(kid="", sec=""),
                            "tradier": ns(token="")},
                   toggle=lambda sid, on: log["toggled"].append((sid, on))),
        signals=ns(market=ns(finnhub_key=""), providers={"fundamentals": FundamentalsProvider("")}),
        av=ns(key="", exhausted=True),
        schwab_auth=ns(app_key="", app_secret="", callback="https://127.0.0.1"),
        etrade_auth=ns(ck="", cs=""),
        trace=ns(emit=lambda *a, **k: log["emitted"].append(a)),
        _publish=lambda m: log["published"].append(m),
        market_status={"isOpen": None, "session": None, "holiday": None, "t": None},
        _status_http=None,
        log=log,
    )
    fake.status_dict = lambda: {"market_status": fake.market_status}
    for name in ("set_key", "_apply_key", "keys_status", "_key_values", "_poll_market_status"):
        setattr(fake, name, types.MethodType(getattr(Engine, name), fake))
    return fake


def test_finnhub_key_is_saved_and_applied_everywhere(monkeypatch, tmp_path):
    e = fake_engine(monkeypatch, tmp_path)
    assert e.cfg.finnhub_key == ""
    out = e.set_key("finnhub", {"key": "  abc123xyz  "})
    assert out["ok"] is True
    f = tmp_path / "finnhub.json"
    assert json.loads(f.read_text()) == {"key": "abc123xyz"}
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    # live, without a restart: cfg (the market-status poll reads it), the source, the radar, fundamentals
    assert e.cfg.finnhub_key == "abc123xyz"
    assert e.sources.sources["finnhub"].key == "abc123xyz"
    assert ("finnhub", True) in e.log["toggled"]
    assert e.signals.market.finnhub_key == "abc123xyz"
    assert e.signals.providers["fundamentals"].key == "abc123xyz"
    # the panel reports it present, masked, never in clear
    fin = next(k for k in out["keys"] if k["id"] == "finnhub")
    assert fin["present"] and "abc123xyz" not in json.dumps(out)
    # the next start reads the same file
    assert Config().finnhub_key == "abc123xyz"


# how each service's saved values come back out of a fresh Config(), i.e. after a restart
CONFIG_READ = {
    "alphavantage": lambda c: {"key": c.av_key},
    "finnhub": lambda c: {"key": c.finnhub_key},
    "eodhd": lambda c: {"key": c.eodhd_key},
    "fmp": lambda c: {"key": c.fmp_key},
    "schwab": lambda c: {"app_key": c.schwab_creds[0], "app_secret": c.schwab_creds[1]},
    "alpaca": lambda c: {"key_id": c.alpaca_creds[0], "secret_key": c.alpaca_creds[1]},
    "tradier": lambda c: {"token": c.tradier_token},
    "etrade": lambda c: {"consumer_key": c.etrade_creds[0], "consumer_secret": c.etrade_creds[1]},
}
ENV_OF = {"alphavantage": ["FLINT_AV_KEY", "FLINT_ALPHAVANTAGE_KEY"], "finnhub": ["FLINT_FINNHUB_KEY"], "eodhd": ["FLINT_EODHD_KEY"],
          "fmp": ["FLINT_FMP_KEY"], "schwab": ["FLINT_SCHWAB_APP_KEY", "FLINT_SCHWAB_APP_SECRET"],
          "alpaca": ["FLINT_ALPACA_KEY_ID", "FLINT_ALPACA_SECRET_KEY"], "tradier": ["FLINT_TRADIER_TOKEN"],
          "etrade": ["FLINT_ETRADE_CONSUMER_KEY", "FLINT_ETRADE_CONSUMER_SECRET"]}


@pytest.mark.parametrize("svc", KEY_SERVICES, ids=[s["id"] for s in KEY_SERVICES])
def test_every_service_saves_shows_present_and_survives_a_restart(monkeypatch, tmp_path, svc):
    """For each service in KEY_SERVICES: the panel save succeeds, the panel then reports the
    service present (it did not for Alpaca, Tradier and E*TRADE: the status map lacked them),
    and a fresh Config() in that directory, which is what a restart does, reads the same values."""
    for v in ENV_OF[svc["id"]]:
        monkeypatch.delenv(v, raising=False)
    e = fake_engine(monkeypatch, tmp_path)
    vals = {f["id"]: f"v-{f['id']}" for f in svc["fields"]}
    out = e.set_key(svc["id"], vals)
    assert out.get("ok") is True, out
    assert (tmp_path / svc["file"]).exists()
    row = next(k for k in out["keys"] if k["id"] == svc["id"])
    assert row["present"], f"{svc['id']} saved but the panel says not set"
    assert all(f["present"] and f["masked"] for f in row["fields"])
    assert CONFIG_READ[svc["id"]](Config()) == vals, f"{svc['id']} does not survive a restart"


def test_apply_key_reaches_the_real_sources(monkeypatch, tmp_path):
    """_apply_key against a real SourceManager (every REGISTRY source, built offline), not a
    stand-in: the attribute names it sets must be the ones the sources read, the source must
    end up enabled with its note cleared, and the toggle must not refuse it."""
    from types import SimpleNamespace as NS
    from flint.sources import SourceManager

    monkeypatch.chdir(tmp_path)
    for vs in ENV_OF.values():
        for v in vs:
            monkeypatch.delenv(v, raising=False)
    cfg = Config()
    cfg.symbols = ["AAPL", "MSFT"]

    async def tok():
        return "t"

    schwab = NS(authenticated=False, has_creds=False, app_key="", app_secret="", callback="https://127.0.0.1", token=tok)
    etrade = NS(authenticated=False, has_creds=False, ck="", cs="", token=tok)
    e = fake_engine(monkeypatch, tmp_path)
    e.cfg = cfg
    e.schwab_auth, e.etrade_auth = schwab, etrade
    e.sources = SourceManager(cfg, cfg.symbols, NS(key="", exhausted=False), lambda t: None, schwab=schwab, etrade=etrade)
    e.av = e.sources.av

    async def run():
        # every source that reads a key at construction starts "keyless"; the live apply must flip it
        for sid in ("alpaca", "tradier", "finnhub", "eodhd", "fmp"):
            e.sources.enabled[sid] = False
        assert e.set_key("alpaca", {"key_id": "AKID", "secret_key": "ASEC"})["ok"]
        assert e.set_key("tradier", {"token": "TTOK"})["ok"]
        assert e.set_key("finnhub", {"key": "FKEY"})["ok"]
        assert e.set_key("eodhd", {"key": "EKEY"})["ok"]
        assert e.set_key("fmp", {"key": "MKEY"})["ok"]
        assert e.set_key("schwab", {"app_key": "SK", "app_secret": "SS"})["ok"]
        assert e.set_key("etrade", {"consumer_key": "CK", "consumer_secret": "CS"})["ok"]
        assert e.set_key("alphavantage", {"key": "AVKEY"})["ok"]
        src = e.sources.sources
        assert (src["alpaca"].kid, src["alpaca"].sec) == ("AKID", "ASEC") and "no API keys" not in src["alpaca"].note
        assert src["tradier"].token == "TTOK" and "no token" not in src["tradier"].note
        assert src["finnhub"].key == "FKEY" and src["eodhd"].key == "EKEY" and src["fmp"].key == "MKEY"
        assert (schwab.app_key, schwab.app_secret) == ("SK", "SS") and (etrade.ck, etrade.cs) == ("CK", "CS")
        assert e.sources.av.key == "AVKEY"
        for sid in ("alpaca", "tradier", "finnhub", "eodhd", "fmp"):
            assert e.sources.enabled[sid], f"{sid} not enabled after its key was saved"
            assert sid in e.sources.tasks, f"{sid} not started after its key was saved"
        for t in e.sources.tasks.values():
            t.cancel()

    asyncio.run(run())


def _client(monkeypatch, status, body, calls):
    class Resp:
        status_code = status
        def json(self):
            return body
    class Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            calls.append(params); return Resp()
    monkeypatch.setattr(eng.httpx, "AsyncClient", Client)


def test_market_status_poll_starts_when_a_key_arrives(monkeypatch, tmp_path):
    e = fake_engine(monkeypatch, tmp_path)
    calls = []
    _client(monkeypatch, 200, {"isOpen": False, "session": None, "holiday": None, "t": 1}, calls)
    assert asyncio.run(e._poll_market_status()) is False and calls == []     # no key: no request, keep waiting
    e.set_key("finnhub", {"key": "abc123xyz"})
    assert asyncio.run(e._poll_market_status()) is True
    assert calls[0]["token"] == "abc123xyz"
    assert e.market_status["isOpen"] is False                                # what backfill and idle training wait on
    assert any(m.get("type") == "status" for m in e.log["published"])


def test_rejected_key_is_reported_once(monkeypatch, tmp_path):
    e = fake_engine(monkeypatch, tmp_path)
    e.set_key("finnhub", {"key": "wrong"})
    calls = []
    _client(monkeypatch, 401, {}, calls)
    asyncio.run(e._poll_market_status()); asyncio.run(e._poll_market_status())
    warns = [a for a in e.log["emitted"] if "HTTP 401" in a[1]]
    assert len(warns) == 1 and "check the key" in warns[0][1]
    assert e.market_status["isOpen"] is None
