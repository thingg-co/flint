"""Entry point: python -m flint [--feed sim|coinbase|auto] [--symbols BTC-USD,ETH-USD] [--port 8000]"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _schwab_auth() -> None:
    """Interactive one-time Schwab OAuth login to mint a refresh token."""
    import asyncio
    import webbrowser
    from urllib.parse import parse_qs, urlparse, unquote

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


def _etrade_auth() -> None:
    """Interactive one-time E*TRADE OAuth 1.0a login (out-of-band verifier code)."""
    import asyncio
    import webbrowser

    from .config import Config
    from .etrade import ETradeAuth

    cfg = Config()
    ck, cs = cfg.etrade_creds
    if not (ck and cs):
        print("No E*TRADE app credentials found.\n"
              "Create etrade.json in this directory:\n"
              '  {"consumer_key": "YOUR_CONSUMER_KEY", "consumer_secret": "YOUR_CONSUMER_SECRET"}\n'
              "or set FLINT_ETRADE_CONSUMER_KEY and FLINT_ETRADE_CONSUMER_SECRET, then run `flint etrade-auth` again.")
        return
    token_file = cfg.etrade_token_file or os.path.join(cfg.state_dir, "etrade_tokens.json")
    auth = ETradeAuth(ck, cs, token_file)
    try:
        asyncio.run(auth.request_token())
    except Exception as e:  # noqa: BLE001
        print(f"Could not get a request token: {type(e).__name__}: {e}")
        return
    url = auth.authorize_url()
    print("\nE*TRADE login")
    print("1. Open this URL, log in, and approve the app:\n")
    print("   " + url + "\n")
    print("2. E*TRADE shows a short verification code on the page after you approve.")
    print("3. Paste that code below.\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    code = input("Paste the verification code: ").strip()
    if not code:
        print("No code entered. Try again.")
        return
    try:
        asyncio.run(auth.exchange_code(code))
    except Exception as e:  # noqa: BLE001
        print(f"Token exchange failed: {type(e).__name__}: {e}")
        return
    print(f"\nAuthenticated. Tokens saved to {token_file}.")
    print("Note: E*TRADE access tokens expire at midnight ET -- re-run `flint etrade-auth` when that happens.")
    print("The E*TRADE source is now enabled for equity symbols. Start flint normally: `uv run flint`.")


def _replay() -> None:
    """Print a short report about the saved training state without starting the server."""
    import sys
    import os
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch

    from .config import Config
    from .features import N_FEATURES

    # Read state_dir from environment first, then default to "state"
    state_dir_str = os.environ.get("FLINT_STATE_DIR", "state")
    state_dir = Path(state_dir_str)

    cfg = Config()
    model_path = state_dir / "model.pt"
    replay_path = state_dir / "replay.npz"

    # Check if model.pt exists
    if not model_path.exists():
        print(f"model.pt: {model_path} does not exist")
        print("No checkpoint found")
        sys.exit(1)

    # Try to load the checkpoint
    try:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"model.pt: {model_path} exists but is unreadable ({type(e).__name__}: {e})")
        print("Corrupt checkpoint")
        sys.exit(2)

    # Load replay.npz
    replay_info = None
    if replay_path.exists():
        try:
            z = np.load(replay_path, allow_pickle=False)
            replay_info = {
                "x_shape": z["x"].shape,
                "y_shape": z["y"].shape,
                "mask_shape": z["mask"].shape if "mask" in z else None,
                "ptr": int(z.get("ptr", 0)),
            }
        except Exception as e:
            replay_info = {"error": str(e)}

    # Extract saved info
    saved_shape = tuple(ck.get("shape", ()))
    saved_symbols = ck.get("symbols", [])
    saved_steps = ck.get("steps", 0)
    saved_labels = ck.get("labels", 0)

    # Get the current window: from machine.json cache if it exists, else cfg.window
    machine_path = state_dir / "machine.json"
    if machine_path.exists():
        try:
            machine = json.loads(machine_path.read_text())
            current_window = machine.get("choice", {}).get("window", cfg.window)
        except (OSError, ValueError, KeyError):
            current_window = cfg.window
    else:
        current_window = cfg.window

    # Get the current symbol list: merge universe.json into config symbols like Engine does
    try:
        saved_universe = json.loads((state_dir / "universe.json").read_text()).get("symbols", [])
        extra_syms = [s for s in saved_universe if s not in set(cfg.symbols)]
        room = cfg.max_universe - len(cfg.symbols)
        current_symbols = cfg.symbols + extra_syms[:max(0, room)]
    except (OSError, ValueError):
        current_symbols = cfg.symbols

    current_shape = (len(current_symbols), current_window, N_FEATURES)

    # Check if checkpoint would load or backup
    shape_matches = saved_shape == current_shape
    symbols_match = saved_symbols == current_symbols
    would_load = shape_matches and symbols_match

    print(f"state_dir: {state_dir}")
    print(f"model.pt: {model_path} exists")

    # File age in hours
    try:
        mtime = model_path.stat().st_mtime
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        print(f"checkpoint age: {age_hours:.1f} hours")
    except Exception:
        print("checkpoint age: unknown")

    print(f"saved symbols: {len(saved_symbols)}")
    print(f"saved shape: {saved_shape}")
    print(f"current symbols: {len(current_symbols)}")
    print(f"current shape: {current_shape}")

    if would_load:
        print("config match: yes — would load checkpoint")
    else:
        print("config match: no — would back up to model.pt.bak")

    print(f"steps: {saved_steps}")
    print(f"labels: {saved_labels}")

    if replay_info:
        if "error" in replay_info:
            print(f"replay.npz: unreadable ({replay_info['error']})")
            print("replay windows: 0")
            print("zero labels: N/A")
            print("label mean (bps): N/A")
            print("label std (bps): N/A")
        else:
            x_shape = replay_info["x_shape"]
            # replay_size = x_shape[0], windows stored = size (capped at replay_size)
            replay_size = cfg.replay_size
            windows_stored = min(x_shape[0], replay_size)
            print(f"replay.npz: {replay_path} exists")
            print(f"replay windows: {windows_stored}")

            # Calculate zero label fraction and stats
            try:
                y = z["y"][:x_shape[0]]
                total_labels = y.size
                zero_count = np.sum(y == 0.0)
                zero_frac = zero_count / total_labels if total_labels > 0 else 0

                # Convert to bps for mean/std
                y_bps = y[y != 0.0]  # exclude fake zeros for stats
                if len(y_bps) > 0:
                    y_bps = y_bps * 100  # convert to bps (returns are already in decimal form)
                    label_mean = np.mean(y_bps)
                    label_std = np.std(y_bps)
                else:
                    label_mean = 0.0
                    label_std = 0.0

                print(f"zero labels: {zero_count:,} / {total_labels:,} ({zero_frac * 100:.1f}%)")
                print(f"label mean (bps): {label_mean:.2f}")
                print(f"label std (bps): {label_std:.2f}")
            except Exception as e:
                print(f"label stats: could not compute ({e})")
    else:
        print("replay.npz: none")
        print("replay windows: 0")
        print("zero labels: N/A")
        print("label mean (bps): N/A")
        print("label std (bps): N/A")

    if would_load:
        sys.exit(0)
    else:
        # Even if config mismatch, we could read the checkpoint
        sys.exit(0)


def _check() -> None:
    """Print what a start would do without loading a model or opening the port."""
    import json
    import sys

    from .autotune import (
        NotEnoughMemory,
        _check_start_memory,
        free_memory_gb,
        other_gpu_tenants,
        pick_device,
    )
    from .config import Config
    from .features import N_FEATURES

    cfg = Config()
    device = pick_device(cfg.device)

    # Device
    print(f"device: {device}")

    # Free memory
    free_gb = free_memory_gb(device)
    print(f"free memory: {free_gb:.1f} GB" if free_gb is not None else "free memory: unknown")

    # Other GPU tenants (llama-server units)
    tenants = other_gpu_tenants()
    if tenants:
        print(f"llama-server units: {', '.join(tenants)}")
    else:
        print("llama-server units: none")

    # Check for cached machine.json
    cache_path = Path(cfg.state_dir) / "machine.json"
    if cache_path.exists():
        try:
            c = json.loads(cache_path.read_text())
            choice = c.get("choice", {})
            preset = choice.get("preset", "unknown")
            params = choice.get("params", 0)
            ms_per_step = choice.get("ms_per_step", 0)
            peak_gb = choice.get("peak_gb", 0)

            print(f"cache exists: {cache_path}")
            print(f"  preset: {preset}")
            print(f"  params: {params / 1e6:.1f}M")
            print(f"  ms_per_step: {ms_per_step}")
            print(f"  peak_gb: {peak_gb:.1f}")

            # Check memory and print verdict
            try:
                _check_start_memory(cfg, N_FEATURES, choice, say=print)
                print("verdict: would start")
            except NotEnoughMemory as e:
                print(f"verdict: {e}")
                sys.exit(2)
        except (OSError, ValueError, KeyError):
            print(f"cache exists: {cache_path} (read error)")
            print("next start: will run autotune")
            return
    else:
        print(f"cache: {cache_path} does not exist")
        print("next start: will run autotune ladder")
        print(f"free memory for autotune: {free_gb:.1f} GB" if free_gb is not None else "free memory for autotune: unknown")


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "schwab-auth":
        _schwab_auth()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "etrade-auth":
        _etrade_auth()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        _check()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        _replay()
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

    from .autotune import NotEnoughMemory
    from .config import Config
    from .engine import Engine
    from .server import create_app

    cfg = Config()
    import socket
    _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _probe.settimeout(0.4)
    if _probe.connect_ex(("127.0.0.1" if cfg.host in ("0.0.0.0", "") else cfg.host, cfg.port)) == 0:
        _probe.close()
        print(f"Flint is already running on {cfg.host}:{cfg.port} -- refusing to start a second instance "
              f"(it would thrash the GPU during tuning). Stop it first:  pkill -9 -f 'python -m flint'")
        raise SystemExit(1)
    _probe.close()
    try:
        engine = Engine(cfg)
    except NotEnoughMemory as e:
        print(f"\n{e}")
        raise SystemExit(2)
    app = create_app(engine)
    print(f"Flint listening on http://{cfg.host}:{cfg.port}  (feed={cfg.feed}, symbols={','.join(cfg.symbols)})")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
