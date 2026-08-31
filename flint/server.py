"""FastAPI app: serves the dashboard, a JSON snapshot, a control endpoint and the live websocket."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine, clean

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / "web"


def create_app(engine: Engine) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        try:
            yield
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await engine.stop()

    app = FastAPI(title="Flint", lifespan=lifespan)
    from fastapi.middleware.cors import CORSMiddleware
    # The mobile app's webview runs on its own origin (http://tauri.localhost) and probes
    # /api/state before handing the view over to the dashboard; without CORS that probe is
    # blocked. The server is LAN-only and read-only over HTTP, so a wildcard is fine.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/state")
    async def state():
        return JSONResponse(clean(engine.snapshot()))

    @app.get("/api/news")
    async def news():
        return JSONResponse(clean(engine.news))

    @app.get("/api/sources")
    async def sources():
        return JSONResponse(clean({"sources": engine.sources.status(), "news_sources": engine.news_hub.status(),
                                   "providers": engine.sources.provider_map()}))

    @app.get("/api/signals")
    async def signals():
        return JSONResponse(clean({"signals": engine.signals_state, "providers": engine.signals.status(),
                                   "burry": engine.burry_enabled}))

    @app.get("/api/keys")
    async def keys():
        return JSONResponse(clean(engine.keys_status()))

    @app.post("/api/keys")
    async def set_keys(payload: dict):
        return JSONResponse(clean(engine.set_key(payload.get("service", ""), payload.get("values", {}))))

    @app.post("/api/control")
    async def control(payload: dict):
        return JSONResponse(clean(engine.apply_control(payload)))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        q = engine.subscribe()
        try:
            await websocket.send_text(engine.snapshot_json())
            while True:
                await websocket.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            engine.unsubscribe(q)

    return app
