"""Charles Schwab API OAuth token manager (market data only).

Schwab uses OAuth 2.0 three-legged auth. You register an app at
developer.schwab.com (App Key + Secret, callback https://127.0.0.1), approve the
Market Data product, then log in once to mint a refresh token (valid 7 days) from
which short-lived access tokens (30 min) are minted automatically.

This module handles the token lifecycle and persists tokens to a JSON file. It
never logs the secret or tokens. Trading endpoints are intentionally not used —
flint consumes market data only.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

AUTH_BASE = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"


REFRESH_TOKEN_LIFE_S = 7 * 86400.0   # Schwab's refresh token lives seven days from the login; only the browser login mints a new one


class SchwabAuth:
    def __init__(self, app_key: str, app_secret: str, callback: str, token_path: str | Path):
        self.app_key = app_key or ""
        self.app_secret = app_secret or ""
        self.callback = callback or "https://127.0.0.1"
        self.token_path = Path(token_path)
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.expires_at: float = 0.0
        self.minted_at: float | None = None   # when the refresh token was minted by a login; the file's birth time for one saved before 0.5.24
        self.status = ""
        self._stamp: int | None = None   # mtime_ns of the token file as last read or written here
        self._load()

    def sync(self) -> None:
        """Re-read the token file when another process rewrote it: `flint schwab-auth` runs beside
        a live engine, which otherwise kept the old login in memory (the notice counted down to the
        old expiry) and wrote it back over the new one at its next refresh."""
        try:
            st = self.token_path.stat().st_mtime_ns
        except OSError:
            return
        if st != self._stamp:
            self._load()

    @property
    def login_expires_at(self) -> float | None:
        """When the refresh token dies and `flint schwab-auth` is needed again; None when unknown."""
        self.sync()
        return (self.minted_at + REFRESH_TOKEN_LIFE_S) if (self.refresh_token and self.minted_at) else None

    @property
    def has_creds(self) -> bool:
        return bool(self.app_key and self.app_secret)

    @property
    def authenticated(self) -> bool:
        return bool(self.has_creds and self.refresh_token)

    def _basic(self) -> str:
        raw = f"{self.app_key}:{self.app_secret}".encode()
        return base64.b64encode(raw).decode()

    def authorize_url(self) -> str:
        from urllib.parse import urlencode
        return AUTH_BASE + "?" + urlencode({"client_id": self.app_key, "redirect_uri": self.callback,
                                            "response_type": "code"})

    def _load(self) -> None:
        try:
            self._stamp = self.token_path.stat().st_mtime_ns
            d = json.loads(self.token_path.read_text())
            self.refresh_token = d.get("refresh_token")
            self.access_token = d.get("access_token")
            self.expires_at = float(d.get("expires_at", 0))
            self.minted_at = float(d["minted_at"]) if d.get("minted_at") else _birth_time(self.token_path)
            self.status = "loaded tokens"
        except (OSError, ValueError):
            self.status = "no tokens on disk"

    def _save(self) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps({
            "refresh_token": self.refresh_token, "access_token": self.access_token,
            "expires_at": self.expires_at, "saved_at": time.time(), "minted_at": self.minted_at,
        }))
        try:
            self.token_path.chmod(0o600)
            self._stamp = self.token_path.stat().st_mtime_ns
        except OSError:
            pass

    async def exchange_code(self, code: str) -> None:
        """Exchange an authorization code (from the redirect URL) for tokens."""
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(TOKEN_URL, headers={"Authorization": f"Basic {self._basic()}",
                                                 "Content-Type": "application/x-www-form-urlencoded"},
                             data={"grant_type": "authorization_code", "code": code, "redirect_uri": self.callback})
            r.raise_for_status()
            self._apply(r.json())
        self.minted_at = time.time()
        self._save()

    async def _refresh(self) -> None:
        self.sync()
        if not self.refresh_token:
            raise RuntimeError("no Schwab refresh token; run: flint schwab-auth")
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(TOKEN_URL, headers={"Authorization": f"Basic {self._basic()}",
                                                 "Content-Type": "application/x-www-form-urlencoded"},
                             data={"grant_type": "refresh_token", "refresh_token": self.refresh_token})
            if r.status_code != 200:
                self.status = f"refresh failed ({r.status_code}); re-run flint schwab-auth"
                raise RuntimeError(self.status)
            self._apply(r.json())
        self._save()

    def _apply(self, d: dict) -> None:
        self.access_token = d.get("access_token")
        if d.get("refresh_token"):
            self.refresh_token = d["refresh_token"]
        self.expires_at = time.time() + float(d.get("expires_in", 1800)) - 60
        self.status = "authenticated"

    async def token(self) -> str:
        """Return a valid access token, refreshing if it has expired."""
        if not self.authenticated:
            raise RuntimeError("Schwab not authenticated")
        self.sync()
        if not self.access_token or time.time() >= self.expires_at:
            await self._refresh()
        return self.access_token


def _birth_time(path: Path) -> float | None:
    """The file's creation time, for a token file saved before the mint time was recorded: the
    file is rewritten in place at every refresh, so its birth is the login. Linux `stat`; None
    where the filesystem does not keep it."""
    try:
        import subprocess
        out = subprocess.run(["stat", "-c", "%W", str(path)], capture_output=True, text=True, timeout=5).stdout.strip()
        t = float(out or 0)
        return t if t > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
