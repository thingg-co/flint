"""The dated news store behind the history features, and the replay's rewrite of a window."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np

from flint.config import Config
from flint.learner import OnlineLearner
from flint.newshist import KEEP_S, PER_DAY, PER_PASS, NewsHistory, backfill_span

DAY = 86400.0


def test_add_keeps_the_newest_per_day_and_records_coverage(tmp_path):
    h = NewsHistory(str(tmp_path / "nh.json"))
    t0 = 19676 * DAY                                                          # a UTC day's start
    rows = [(t0 + k * 600, 0.1 * (k % 5)) for k in range(20)]                 # twenty headlines in one day
    assert h.add("AAPL", rows, t0 - DAY, t0 + DAY) == PER_DAY
    assert np.allclose(h.ts["AAPL"], [t for t, _ in rows[-PER_DAY:]])
    assert h.covers(["AAPL"], t0 - DAY, t0 + DAY) and not h.covers(["AAPL"], t0 - 2 * DAY, t0) and not h.covers(["MSFT"], t0, t0)
    h.add("AAPL", [(t0 + 2 * DAY, 0.5)], t0 + DAY, t0 + 3 * DAY)                  # an adjacent span merges
    assert h.covered["AAPL"] == [[t0 - DAY, t0 + 3 * DAY]] and h.missing(["AAPL", "MSFT"], t0, t0 + DAY) == ["MSFT"]
    h.save()
    again = NewsHistory(str(tmp_path / "nh.json"))
    assert again.load() and np.allclose(again.ts["AAPL"], h.ts["AAPL"]) and again.covers(["AAPL"], t0, t0 + 2 * DAY)


def test_at_reads_like_the_live_skim():
    h = NewsHistory()
    t = 1_700_000_000.0
    h.add("AAPL", [(t - 7 * 3600 * k, 0.8) for k in range(1, 11)], t - 10 * DAY, t)   # ten headlines over the last three days, all positive
    h.add("MSFT", [(t - 3600, -0.4)], t - 10 * DAY, t)                             # one, negative
    h.add("OLD", [(t - KEEP_S - 10, 0.9)], t - 10 * DAY, t)                        # outside the window
    sent, attn = h.at(t, ["AAPL", "MSFT", "OLD", "NONE"])
    market = (PER_PASS * 0.8 - 0.4) / (PER_PASS + 1)                              # the mean over the window's headlines (AAPL capped at the skim's eight)
    assert abs(sent[0] - (PER_PASS * 0.8 + market) / (PER_PASS + 1)) < 1e-9
    assert abs(sent[1] - (-0.4 + market) / 2) < 1e-9
    assert sent[2] == market and sent[3] == market                                # no coverage: the market tone
    assert attn[0] == 1.0 and abs(attn[1] - np.sqrt(1 / PER_PASS)) < 1e-9 and attn[2] == 0.0 and attn[3] == 0.0
    s2, a2 = h.at(t - 20 * DAY, ["AAPL"])                                          # before any headline
    assert s2[0] == 0.0 and a2[0] == 0.0
    lo, hi = backfill_span(t, 10)
    assert hi == t and lo == t - 10 * DAY - KEEP_S


class _Resp:
    def __init__(self, status, body):
        self.status_code = status; self._body = body
    def json(self):
        return self._body


class _Client:
    """Answers company-news with three headlines per day in the asked range, one 429 first."""
    def __init__(self):
        self.calls = []; self.limited = True
    async def get(self, url, params=None):
        self.calls.append(dict(params))
        if self.limited:
            self.limited = False
            return _Resp(429, None)
        from datetime import datetime, timezone
        a = datetime.fromisoformat(params["from"]).replace(tzinfo=timezone.utc).timestamp()
        b = datetime.fromisoformat(params["to"]).replace(tzinfo=timezone.utc).timestamp()
        items = []
        d = a
        while d < b:
            for k in range(3):
                items.append({"headline": f"{params['symbol']} beats estimates again, number {k}", "datetime": d + 3600 * (k + 1)})
            d += DAY
        return _Resp(200, items)


def test_fetch_pages_the_missing_names_scores_titles_and_persists(tmp_path, monkeypatch):
    import flint.newshist as nh
    real_sleep = asyncio.sleep
    monkeypatch.setattr(nh.asyncio, "sleep", lambda *a, **k: real_sleep(0))          # no pacing in the test
    h = NewsHistory(str(tmp_path / "nh.json"))
    t = 1_700_000_000.0
    scorer = SimpleNamespace(score=lambda titles: [0.6] * len(titles), name="stub")
    c = _Client()
    res = asyncio.run(h.fetch(["AAPL", "MSFT"], t - 100 * DAY, t, "key", scorer=scorer, client=c, rate_s=0))
    assert res["done"] == 2 and res["headlines"] > 0 and not res["failed"]
    assert len(c.calls) >= 6 and all(p["token"] == "key" for p in c.calls)         # three 45-day slices per name, plus the 429 retry
    assert h.covers(["AAPL", "MSFT"], t - 100 * DAY, t)
    assert np.all(h.tone["AAPL"] == 0.6) and len(h.ts["AAPL"]) <= PER_DAY * 101
    sent, attn = h.at(t - 50 * DAY, ["AAPL", "MSFT"])
    assert sent[0] > 0.5 and attn[0] == 1.0
    assert json.load(open(tmp_path / "nh.json"))["symbols"]["MSFT"]["covered"]
    again = asyncio.run(h.fetch(["AAPL", "MSFT"], t - 100 * DAY, t, "key", scorer=scorer, client=c))
    assert again["names"] == 0                                                       # nothing missing: no requests


def test_replace_rewrites_a_stored_window_in_place():
    cfg = Config()
    cfg.symbols = ["A", "B", "C"]; cfg.device = "cpu"; cfg.torch_threads = 1; cfg.d_model = 16; cfg.dilations = (1, 2)
    cfg.n_experts = 2; cfg.n_heads = 2; cfg.window = 8; cfg.replay_size = 16; cfg.batch_size = 4; cfg.compile = False
    L = OnlineLearner(cfg, n_features=3)
    x = np.ones((3, 8, 3), np.float32); y = np.array([1.0, 2.0, 3.0], np.float32)
    assert L.add(x, y, None, ts=1000.0) and not L.add(x, y, None, ts=1000.0)
    assert L.replace(1000.0, x * 2, y * 2, np.array([1, 0, 1], np.float32))
    assert np.all(L.rx[0] == 2) and np.all(L.ry[0] == y * 2) and list(L.rmask[0]) == [1, 0, 1] and L.size == 1
    assert not L.replace(999.0, x, y) and not L.replace(None, x, y)
