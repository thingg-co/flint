"""Tests for flint.trace.Trace."""
import pytest
from flint.trace import Trace


def test_emit_publishes_and_keeps():
    """emit publishes via the callback and keeps in buffer."""
    recorded = []

    def recorder(msg: dict) -> None:
        recorded.append(msg)

    t = Trace(publish=recorder, keep=3)
    t.emit("feed", "hello", "warn", {"n": 1})

    # Publish received exactly one message with the expected keys
    assert len(recorded) == 1
    msg = recorded[0]
    assert msg["type"] == "trace"
    ev = msg["ev"]
    assert ev["ch"] == "feed"
    assert ev["text"] == "hello"
    assert ev["lvl"] == "warn"
    assert ev["data"] == {"n": 1}
    assert "seq" in ev
    assert "t" in ev  # timestamp

    # recent() returns the event in the feed list
    assert len(t.recent()["feed"]) == 1
    assert t.recent()["feed"][0] is ev


def test_recent_is_bounded_per_channel():
    """recent() returns at most `keep` events per channel, in emission order."""
    recorded = []

    def recorder(msg: dict) -> None:
        pass  # ignore for this test

    t = Trace(publish=recorder, keep=3)

    # Emit 5 events on "feed"
    for i in range(5):
        t.emit("feed", f"msg{i}", "info", None)

    # Emit 2 events on "model"
    t.emit("model", "m0", "info", None)
    t.emit("model", "m1", "info", None)

    recent = t.recent()
    # feed keeps only the last 3 (in emission order)
    assert len(recent["feed"]) == 3
    assert recent["feed"][0]["text"] == "msg2"
    assert recent["feed"][1]["text"] == "msg3"
    assert recent["feed"][2]["text"] == "msg4"
    # model keeps all 2
    assert len(recent["model"]) == 2
    assert recent["model"][0]["text"] == "m0"
    assert recent["model"][1]["text"] == "m1"


def test_default_level_and_data():
    """emit with minimal args uses level='info' and no data field."""
    recorded = []

    def recorder(msg: dict) -> None:
        recorded.append(msg)

    t = Trace(publish=recorder, keep=10)
    # Use a valid channel from CHANNELS
    t.emit("feed", "t")

    ev = recorded[0]["ev"]
    assert ev["lvl"] == "info"
    # data key is absent when None is passed
    assert "data" not in ev


def test_publish_errors_do_propagate():
    """A publishing error propagates; emit does not guard the publisher."""
    call_count = 0

    def failing_recorder(msg: dict) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("publish failed")

    t = Trace(publish=failing_recorder, keep=3)
    # The emit raises because the publisher has no try/except guard
    with pytest.raises(RuntimeError, match="publish failed"):
        t.emit("feed", "hello", "info", None)
    # Publisher was called once
    assert call_count == 1
