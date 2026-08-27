"""E*TRADE OAuth 1.0a auth (market data only).

E*TRADE uses three-legged OAuth 1.0a with HMAC-SHA1 request signing: mint a request
token, have the user authorize it in a browser to get a 5-character verifier, then
exchange it for an access token. Access tokens expire at midnight US-Eastern and go
inactive after ~2h idle; `renew()` reactivates an idle (not expired) token. Every API
request is signed. This module never logs the secret or tokens. Trading endpoints are
never used -- Flint consumes market data only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qs, quote

import httpx

log = logging.getLogger(__name__)

BASE = "https://api.etrade.com"
REQUEST_TOKEN_URL = f"{BASE}/oauth/request_token"
ACCESS_TOKEN_URL = f"{BASE}/oauth/access_token"
RENEW_URL = f"{BASE}/oauth/renew_access_token"
AUTHORIZE_URL = "https://us.etrade.com/e/t/etws/authorize"


def _pct(s: str) -> str:
    return quote(str(s), safe="~")


class ETradeAuth:
    def __init__(self, consumer_key: str, consumer_secret: str, token_path: str | Path):
        self.ck = consumer_key or ""
        self.cs = consumer_secret or ""
        self.token_path = Path(token_path)
        self.oauth_token: str | None = None
        self.oauth_token_secret: str | None = None
        self.request_secret: str | None = None      # transient, during the auth dance
        self.status = ""
        self._load()

    @property
    def has_creds(self) -> bool:
        return bool(self.ck and self.cs)

    @property
    def authenticated(self) -> bool:
        return bool(self.has_creds and self.oauth_token and self.oauth_token_secret)

    def _load(self) -> None:
        try:
            d = json.loads(self.token_path.read_text())
            self.oauth_token = d.get("oauth_token")
            self.oauth_token_secret = d.get("oauth_token_secret")
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            self.token_path.write_text(json.dumps(
                {"oauth_token": self.oauth_token, "oauth_token_secret": self.oauth_token_secret}))
            self.token_path.chmod(0o600)
        except OSError:
            pass

    def _sign(self, method: str, url: str, params: dict, token_secret: str = "") -> str:
        norm = "&".join(f"{_pct(k)}={_pct(v)}" for k, v in sorted(params.items()))
        base = "&".join([method.upper(), _pct(url), _pct(norm)])
        key = f"{_pct(self.cs)}&{_pct(token_secret)}"
        digest = hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()
        return base64.b64encode(digest).decode()

    def _auth_header(self, method: str, url: str, token: str = "", token_secret: str = "", extra: dict | None = None) -> str:
        p = {"oauth_consumer_key": self.ck, "oauth_nonce": secrets.token_hex(16),
             "oauth_signature_method": "HMAC-SHA1", "oauth_timestamp": str(int(time.time())),
             "oauth_version": "1.0"}
        if token:
            p["oauth_token"] = token
        if extra:
            p.update(extra)
        p["oauth_signature"] = self._sign(method, url, p, token_secret)
        return "OAuth " + ", ".join(f'{_pct(k)}="{_pct(v)}"' for k, v in sorted(p.items()))

    async def request_token(self) -> str:
        async with httpx.AsyncClient(timeout=20) as c:
            hdr = self._auth_header("GET", REQUEST_TOKEN_URL, extra={"oauth_callback": "oob"})
            r = await c.get(REQUEST_TOKEN_URL, headers={"Authorization": hdr})
            r.raise_for_status()
            d = parse_qs(r.text)
            self.oauth_token = d["oauth_token"][0]
            self.request_secret = d["oauth_token_secret"][0]
            return self.oauth_token

    def authorize_url(self) -> str:
        return f"{AUTHORIZE_URL}?key={_pct(self.ck)}&token={_pct(self.oauth_token or '')}"

    async def exchange_code(self, verifier: str) -> None:
        """Exchange the request token + browser verifier for an access token."""
        async with httpx.AsyncClient(timeout=20) as c:
            hdr = self._auth_header("GET", ACCESS_TOKEN_URL, token=self.oauth_token or "",
                                    token_secret=self.request_secret or "", extra={"oauth_verifier": verifier.strip()})
            r = await c.get(ACCESS_TOKEN_URL, headers={"Authorization": hdr})
            r.raise_for_status()
            d = parse_qs(r.text)
            self.oauth_token = d["oauth_token"][0]
            self.oauth_token_secret = d["oauth_token_secret"][0]
        self._save()

    async def renew(self) -> bool:
        """Reactivate an idle (not day-expired) access token. Returns True on success."""
        if not self.authenticated:
            return False
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                hdr = self._auth_header("GET", RENEW_URL, token=self.oauth_token, token_secret=self.oauth_token_secret)
                r = await c.get(RENEW_URL, headers={"Authorization": hdr})
                return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def signed_headers(self, method: str, url: str) -> dict:
        """OAuth header for a signed market-data request (symbols go in the path, no query params)."""
        return {"Authorization": self._auth_header(method, url, token=self.oauth_token or "",
                                                   token_secret=self.oauth_token_secret or "")}
