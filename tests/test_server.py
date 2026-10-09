"""Tests for flint/server.py routes using FastAPI TestClient."""
import asyncio
from types import SimpleNamespace

import pytest

from flint.engine import clean
from flint.server import create_app


def make_engine():
    """Create a stub engine for testing routes."""
    # Pre-populated queue for first subscribe
    q = asyncio.Queue()
    q.put_nowait('{"type":"update"}')

    # Use lists on the namespace to record calls
    recorded_keys = []
    recorded_control = []
    unsubscribed = []

    async def async_stop():
        pass

    def set_key(service, values):
        recorded_keys.append((service, values))
        return {"ok": True}

    def apply_control(payload):
        recorded_control.append(payload)
        return {"ok": True, "applied": payload}

    def subscribe():
        new_q = asyncio.Queue()
        new_q.put_nowait('{"type":"update"}')
        return new_q

    def unsubscribe(q):
        unsubscribed.append(q)

    # Attach recording lists to the namespace for test access
    stub = SimpleNamespace(
        start=lambda: None,
        stop=async_stop,
        snapshot=lambda: {"status": {"phase": "training"}, "bad": float("nan")},
        news=[{"headline": "x"}],
        sources=SimpleNamespace(
            status=lambda: [{"id": "alpaca"}],
            provider_map=lambda: {"AAPL": "alpaca"}
        ),
        news_hub=SimpleNamespace(status=lambda: []),
        signals_state={"fng": 55},
        signals=SimpleNamespace(status=lambda: []),
        burry_enabled=False,
        keys_status=lambda: {"schwab": {"configured": False}},
        set_key=set_key,
        apply_control=apply_control,
        subscribe=subscribe,
        snapshot_json=lambda: '{"type":"snapshot"}',
        unsubscribe=unsubscribe,
        _recorded_keys=recorded_keys,
        _recorded_control=recorded_control,
        _unsubscribed=unsubscribed,
    )
    return stub


class TestIndex:
    def test_index_serves_dashboard(self):
        """GET "/" is 200, content-type starts with "text/html", body contains "Flint"."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "Flint" in response.text


class TestState:
    def test_state_is_cleaned_json(self):
        """GET "/api/state" is 200; json()["status"]["phase"] == "training";
        the "bad" NaN field comes back as None (clean() turns NaN into None)
        and the raw body text contains no "NaN"."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            response = client.get("/api/state")
            assert response.status_code == 200
            data = response.json()
            assert data["status"]["phase"] == "training"
            # clean() turns NaN/inf into None
            assert data["bad"] is None
            # raw body should not contain "NaN" string
            assert "NaN" not in response.text


class TestNewsSourcesSignalsKeys:
    def test_news_sources_signals_keys(self):
        """Test all GET endpoints for news, sources, signals, and keys."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            # /api/news
            response = client.get("/api/news")
            assert response.status_code == 200
            assert response.json() == [{"headline": "x"}]

            # /api/sources
            response = client.get("/api/sources")
            assert response.status_code == 200
            sources_data = response.json()
            assert sources_data["providers"] == {"AAPL": "alpaca"}
            assert sources_data["sources"] == [{"id": "alpaca"}]

            # /api/signals
            response = client.get("/api/signals")
            assert response.status_code == 200
            signals_data = response.json()
            assert signals_data["signals"] == {"fng": 55}
            assert signals_data["burry"] is False

            # /api/keys
            response = client.get("/api/keys")
            assert response.status_code == 200
            assert response.json() == {"schwab": {"configured": False}}


class TestControl:
    def test_control_posts_payload_through(self):
        """POST /api/control with json {"name": "cost_bps", "value": 8}
        returns {"ok": True, "applied": {...same payload...}}."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            payload = {"name": "cost_bps", "value": 8}
            response = client.post("/api/control", json=payload)
            assert response.status_code == 200
            assert response.json() == {"ok": True, "applied": payload}
            # Check the stub recorded the payload once
            assert engine._recorded_control == [payload]


class TestSetKeys:
    def test_set_keys_passes_service_and_values(self):
        """POST /api/keys with {"service": "finnhub", "values": {"key": "abc"}}
        returns {"ok": True} and the stub recorded ("finnhub", {"key": "abc"})."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            payload = {"service": "finnhub", "values": {"key": "abc"}}
            response = client.post("/api/keys", json=payload)
            assert response.status_code == 200
            assert response.json() == {"ok": True}
            # Check the stub recorded the call
            assert ("finnhub", {"key": "abc"}) in engine._recorded_keys

    def test_set_keys_empty_body(self):
        """POST /api/keys with empty body {} records ("", {})."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            response = client.post("/api/keys", json={})
            assert response.status_code == 200
            assert response.json() == {"ok": True}
            # Check the stub recorded ("", {})
            assert ("", {}) in engine._recorded_keys


class TestWebsocket:
    def test_websocket_sends_snapshot_then_updates(self):
        """WebSocket /ws sends snapshot then update; after close, unsubscribe was called."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            with client.websocket_connect("/ws") as ws:
                # First message should be snapshot
                msg1 = ws.receive_text()
                assert msg1 == '{"type":"snapshot"}'
                # Second message should be update
                msg2 = ws.receive_text()
                assert msg2 == '{"type":"update"}'
            # After the with block, unsubscribe should have been called
            assert len(engine._unsubscribed) == 1

    def test_websocket_handler_returns_on_disconnect_without_being_cancelled(self):
        """uvicorn's graceful shutdown closes every socket and then waits for each handler
        task to *return*; it cancels nothing. A handler that only pushes would sit on its
        queue until the next publish, which with an idle or dead engine is never, and
        SIGTERM would hang until systemd's SIGKILL. Drive the ASGI app directly (TestClient
        cancels the handler on exit, so it cannot see this): connect, let it push, send the
        disconnect, and require the coroutine to finish on its own."""
        engine = make_engine()
        app = create_app(engine)

        async def run():
            sent = []
            pushed = asyncio.Event()
            inbox = asyncio.Queue()
            inbox.put_nowait({"type": "websocket.connect"})

            async def receive():
                return await inbox.get()

            async def send(msg):
                sent.append(msg)
                if sum(m["type"] == "websocket.send" for m in sent) == 2:
                    pushed.set()

            scope = {"type": "websocket", "asgi": {"version": "3.0"}, "path": "/ws", "raw_path": b"/ws",
                     "root_path": "", "scheme": "ws", "query_string": b"", "headers": [],
                     "client": ("testclient", 50000), "server": ("testserver", 80), "subprotocols": []}
            handler = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(pushed.wait(), 2)              # snapshot and the one queued update
            inbox.put_nowait({"type": "websocket.disconnect", "code": 1012})
            await asyncio.wait_for(handler, 2)                    # returns by itself: nothing cancels it
            return sent

        sent = asyncio.run(run())
        assert [m["type"] for m in sent][:3] == ["websocket.accept", "websocket.send", "websocket.send"]
        assert len(engine._unsubscribed) == 1


class TestCORS:
    def test_cors_allows_any_origin(self):
        """GET /api/state with header Origin "http://tauri.localhost"
        returns access-control-allow-origin "*"."""
        from fastapi.testclient import TestClient

        engine = make_engine()
        with TestClient(create_app(engine)) as client:
            response = client.get("/api/state", headers={"Origin": "http://tauri.localhost"})
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "*"
