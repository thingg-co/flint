"""Entry point: python -m flint [--feed sim|coinbase|auto] [--symbols BTC-USD,ETH-USD] [--port 8000]"""
from __future__ import annotations

import argparse
import logging
import os


def _schwab_auth() -> None:
    """Interactive one-time Schwab OAuth login to mint a refresh token."""
    import asyncio
    import os
    import webbrowser
    from urllib.parse import urlparse, parse_qs, unquote

    from .config import Config
    from .schwab import SchwabAuth

    cfg = Config()
    key, secret, callback = cfg.schwab_creds
    if not (key and secret):
        print("No Schwab app credentials found.\n"
              "Create schwab.json in this directory:\n"
              '  {"app_key": "YOUR_APP_KEY", "app_secret": "YOUR_APP_SECRET", "callback": "https://127.0.0.1"}\n'
              "or set FLINT_SCHWAB_APP_KEY and FLINT_SCHWAB_APP_SECRET, then run `flint schwab-auth` again.")
        return
    token_file = cfg.schwab_token_file or os.path.join(cfg.state_dir, "schwab_tokens.json")
    auth = SchwabAuth(key, secret, callback, token_file)
    url = auth.authorize_url()
    print("\nSchwab login")
    print("1. Open this URL, log in, and approve the app:\n")
    print("   " + url + "\n")
    print(f"2. Your browser will redirect to {callback}/?code=...  (the page itself may not load — that's fine).")
    print("3. Copy the FULL redirected URL from the address bar and paste it below.\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    pasted = input("Paste the redirect URL (or just the code): ").strip()
    code = pasted
    if pasted.startswith("http"):
        qs = parse_qs(urlparse(pasted).query)
        code = qs.get("code", [""])[0]
    code = unquote(code)
    if not code:
        print("No authorization code found in that input. Try again.")
        return
    try:
        asyncio.run(auth.exchange_code(code))
    except Exception as e:  # noqa: BLE001
        print(f"Token exchange failed: {type(e).__name__}: {e}")
        return
    print(f"\nAuthenticated. Tokens saved to {token_file} (refresh token valid ~7 days).")
    print("The Charles Schwab source is now enabled for equity symbols. Start flint normally: `uv run flint`.")


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "schwab-auth":
        _schwab_auth()
        return
    ap = argparse.ArgumentParser(prog="flint", description="continuously learning market model with a live dashboard")
    ap.add_argument("--feed", choices=["auto", "coinbase", "sim"], help="market data source (default: auto)")
    ap.add_argument("--symbols", help="comma separated products, e.g. BTC-USD,ETH-USD")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-news", action="store_true", help="disable the news skimmer")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    if args.feed:
        os.environ["FLINT_FEED"] = args.feed
    if args.symbols:
        os.environ["FLINT_SYMBOLS"] = args.symbols
    if args.host:
        os.environ["FLINT_HOST"] = args.host
    if args.port:
        os.environ["FLINT_PORT"] = str(args.port)
    if args.no_news:
        os.environ["FLINT_NEWS"] = "0"
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import uvicorn

    from .config import Config
    from .engine import Engine
    from .server import create_app

    cfg = Config()
    app = create_app(Engine(cfg))
    print(f"Flint listening on http://{cfg.host}:{cfg.port}  (feed={cfg.feed}, symbols={','.join(cfg.symbols)})")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
