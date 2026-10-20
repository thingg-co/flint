"""The token file is re-read when another process rewrites it. Exists because `flint schwab-auth`
run beside a live engine left the engine on the old login: the notice kept counting down to the old
expiry, and the next access-token refresh wrote the old refresh token back over the new one."""
import asyncio
import json
import os
import time

from flint.schwab import REFRESH_TOKEN_LIFE_S, SchwabAuth


def write(path, refresh, minted, expires, bump=0):
    path.write_text(json.dumps({"refresh_token": refresh, "access_token": "acc-" + refresh,
                                "expires_at": expires, "saved_at": minted, "minted_at": minted}))
    if bump:   # make the rewrite visible on filesystems with coarse mtimes
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump))


def test_new_login_on_disk_is_picked_up(tmp_path):
    p = tmp_path / "schwab_tokens.json"
    old = time.time() - 6 * 86400
    write(p, "old", old, time.time() + 600)
    auth = SchwabAuth("k", "s", "https://127.0.0.1", p)
    assert auth.login_expires_at == old + REFRESH_TOKEN_LIFE_S

    new = time.time()
    write(p, "new", new, time.time() + 1800, bump=10**9)
    assert auth.login_expires_at == new + REFRESH_TOKEN_LIFE_S
    assert asyncio.run(auth.token()) == "acc-new"


def test_own_save_does_not_reload(tmp_path):
    p = tmp_path / "schwab_tokens.json"
    write(p, "r", time.time(), time.time() + 600)
    auth = SchwabAuth("k", "s", "https://127.0.0.1", p)
    auth._apply({"access_token": "fresh", "expires_in": 1800})
    auth._save()
    auth.sync()
    assert auth.access_token == "fresh" and auth.status == "authenticated"


def test_refresh_uses_the_newer_login(tmp_path, monkeypatch):
    p = tmp_path / "schwab_tokens.json"
    write(p, "old", time.time() - 86400, 0)
    auth = SchwabAuth("k", "s", "https://127.0.0.1", p)
    write(p, "new", time.time(), 0, bump=10**9)
    sent = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"access_token": "minted", "expires_in": 1800}

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, data=None):
            sent.update(data)
            return Resp()

    monkeypatch.setattr("flint.schwab.httpx.AsyncClient", Client)
    assert asyncio.run(auth.token()) == "minted"
    assert sent["refresh_token"] == "new"
    assert json.loads(p.read_text())["refresh_token"] == "new"
