"""FastAPI app: serves the dashboard, a JSON snapshot, a control endpoint and the live websocket."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine, clean

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / "web"


def index_html() -> str:
    """The dashboard page with its script and stylesheet URLs versioned by the files' mtimes, so a
    changed file is a new URL and no cache, in the browser or in between, serves the old one."""
    html = (WEB / "index.html").read_text()
    for name in ("app.js", "style.css"):
        try:
            v = int((WEB / name).stat().st_mtime)
        except OSError:
            continue
        html = html.replace(f"/static/{name}\"", f"/static/{name}?v={v}\"")
    return html


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

    @app.middleware("http")
    async def never_stale(request, call_next):
        # The page and its script and stylesheet carry no Cache-Control, so a browser kept them by
        # the heuristic rule (a tenth of the file's age) and a normal reload re-ran a script fixed
        # hours earlier (2026-10-11: the Control panel's "cannot be parsed" warnings outlived their
        # fix by a day). no-cache makes every load revalidate; the ETag answers 304 when unchanged.
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/")
    async def index():
        return HTMLResponse(index_html())

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

    @app.get("/api/backtest")
    async def backtest():
        # the same shape the websocket pushes (totals and day rows from the live run while one is on,
        # a finished variant as the baseline), plus the raw saved report for scripts
        return JSONResponse(clean({**engine.backtest_summary(), "backtest": engine.backtest}))

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

        async def push() -> None:
            await websocket.send_text(engine.snapshot_json())
            while True:
                await websocket.send_text(await q.get())

        async def listen() -> None:
            # The dashboard never sends anything, but only a read notices the peer going away
            # or uvicorn closing the socket on SIGTERM. A handler that only pushes sits on its
            # queue until the next publish; with the engine idle (or dead) that is forever, and
            # uvicorn's graceful shutdown waits for every handler to return before it stops.
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                while (await websocket.receive())["type"] != "websocket.disconnect":
                    pass

        pusher = asyncio.create_task(push())
        listener = asyncio.create_task(listen())
        try:
            await asyncio.wait({pusher, listener}, return_when=asyncio.FIRST_COMPLETED)
            if pusher.done() and not pusher.cancelled():
                with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                    pusher.result()                 # anything else is a real bug: let uvicorn log it
        finally:
            pusher.cancel()
            listener.cancel()
            await asyncio.gather(pusher, listener, return_exceptions=True)
            engine.unsubscribe(q)

    return app
