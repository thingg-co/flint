"""Keys saved from the Control panel: written to the file the config reads on the next start,
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


@pytest.mark.parametrize("svc", KEY_SERVICES, ids=[s["id"] for s in KEY_SERVICES])
def test_every_service_saves_its_file(monkeypatch, tmp_path, svc):
    e = fake_engine(monkeypatch, tmp_path)
    out = e.set_key(svc["id"], {f["id"]: f"v-{f['id']}" for f in svc["fields"]})
    assert out.get("ok") is True, out
    assert (tmp_path / svc["file"]).exists()


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
