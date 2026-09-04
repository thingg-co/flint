"""The bar clock must say so when quotes tick but no candle forms during the open session."""
import types

from flint.engine import Engine


def make_fake(now, last_row_at, is_open=True, stalled=False):
    events = []
    fake = types.SimpleNamespace(
        _last_row_at=last_row_at, _stalled=stalled,
        cfg=types.SimpleNamespace(bar_seconds=300.0),
        market_status={"isOpen": is_open},
        sources=types.SimpleNamespace(status=lambda: [{"id": "alpaca", "enabled": True, "ticks": 10, "note": "quotes"},
                                                      {"id": "finnhub", "enabled": True, "ticks": 5, "note": "ws InvalidStatus"},
                                                      {"id": "yahoo", "enabled": True, "ticks": 0}]),
        trace=types.SimpleNamespace(emit=lambda ch, text, level="info", data=None: events.append((ch, level, text))),
        STALL_BARS=Engine.STALL_BARS,
    )
    return fake, events


def test_no_alarm_before_three_bars():
    fake, events = make_fake(now=1000.0, last_row_at=1000.0 - 800.0)
    Engine._check_stall(fake, 1000.0)
    assert events == [] and fake._stalled is False


def test_alarm_once_when_open_and_stale():
    fake, events = make_fake(now=1000.0, last_row_at=1000.0 - 1000.0)
    Engine._check_stall(fake, 1000.0)
    Engine._check_stall(fake, 1300.0)
    assert fake._stalled is True
    assert len(events) == 1
    ch, level, text = events[0]
    assert (ch, level) == ("system", "warn")
    assert "no bar for 17 minutes" in text and "alpaca (quotes)" in text and "finnhub (ws InvalidStatus)" in text
    assert "yahoo" not in text          # sources that never ticked are noise here


def test_silent_when_market_closed():
    fake, events = make_fake(now=1000.0, last_row_at=0.0, is_open=False)
    Engine._check_stall(fake, 1000.0)
    assert events == [] and fake._stalled is False


def test_falls_back_to_the_wall_clock_when_status_unknown(monkeypatch):
    fake, events = make_fake(now=1000.0, last_row_at=0.0, is_open=None)
    monkeypatch.setattr("flint.engine.clock.regular_session", lambda: True)
    Engine._check_stall(fake, 1000.0)
    assert fake._stalled is True and len(events) == 1
