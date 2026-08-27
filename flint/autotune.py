"""First-run machine benchmark and model auto-sizing.

Picks the compute device (CUDA > MPS > CPU), benchmarks a ladder of model sizes on
it, and chooses the biggest whose training step is fast enough to keep up with the
bar interval. The result is cached to <state_dir>/machine.json so it only benchmarks
once per machine — this is what lets the app adapt when shared to different hardware.
"""
from __future__ import annotations

import json
import logging
import fcntl
import os
import time
from pathlib import Path

import torch

from .model import FlintNet, flint_loss

log = logging.getLogger(__name__)

# name, d_model, dilations, n_experts, n_heads, window
PRESETS = [
    ("S",    64,  (1, 2, 4, 8, 16),          3, 4, 64),
    ("M",    96,  (1, 2, 4, 8, 16, 32),      5, 6, 96),
    ("L",    128, (1, 2, 4, 8, 16, 32),      6, 8, 96),
    ("XL",   192, (1, 2, 4, 8, 16, 32),      8, 8, 128),
    ("XXL",  256, (1, 2, 4, 8, 16, 32),      8, 8, 128),
    ("XXXL", 384, (1, 2, 4, 8, 16, 32, 64),  8, 8, 128),
]


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _bench(device: str, preset, n_assets: int, n_features: int, batch: int, n_quant: int, iters: int = 6):
    _, d, dil, exp, heads, win = preset
    dev = torch.device(device)
    quant = tuple((i + 1) / (n_quant + 1) for i in range(n_quant))
    m = FlintNet(n_features, n_assets, n_quant, d_model=d, dilations=dil, n_experts=exp, n_heads=heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(batch, n_assets, win, n_features, device=dev)
    y = torch.randn(batch, n_assets, device=dev)

    def step():
        q, l, g, _ = m(x)
        loss, _ = flint_loss(q, l, g, y, quant)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    step(); step()
    _sync(device)
    t = time.time()
    for _ in range(iters):
        step()
    _sync(device)
    ms = (time.time() - t) / iters * 1000
    n = sum(p.numel() for p in m.parameters())
    del m, opt, x, y
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return ms, n


def _best_threads(device: str, n_assets: int, n_features: int, n_quant: int, say) -> int:
    cores = os.cpu_count() or 4
    if device != "cpu":
        return max(1, min(4, cores))          # GPU does the math; a few CPU threads feed it
    best_ms, best_th = 1e9, 2
    for th in sorted({2, cores // 2, max(2, int(cores * 0.6)), cores - 2}):
        if th < 1:
            continue
        torch.set_num_threads(th)
        ms, _ = _bench("cpu", PRESETS[1], n_assets, n_features, 16, n_quant, iters=4)
        say(f"thread sweep: {th} threads -> {ms:.0f} ms/step")
        if ms < best_ms:
            best_ms, best_th = ms, th
    return best_th


def autotune(cfg, n_features: int, say=None) -> dict:
    say = say or (lambda *a, **k: log.info(*a))
    device = pick_device(cfg.device)
    cache = Path(cfg.state_dir) / "machine.json"
    key = (f"{device}|{os.cpu_count()}|{cfg.n_assets}|{n_features}|{cfg.bar_seconds}|{cfg.batch_size}"
           f"|{cfg.autotune_util}|{cfg.max_warmup_seconds}|{cfg.warmup_steps}")
    try:
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            ch = c["choice"]
            say(f"using cached tuning: {ch['preset']} ({ch['params'] / 1e6:.2f}M params) on {ch['device']} "
                f"at {ch['ms_per_step']} ms/step")
            return ch
    except (OSError, ValueError, KeyError):
        pass

    # Serialize benchmarking across processes -- two benchmarks on one GPU thrash memory into swap.
    lock_f = None
    try:
        Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
        lock_f = open(Path(cfg.state_dir) / "autotune.lock", "w")
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say("another benchmark is running; waiting for it to finish (one benchmark at a time)...")
            fcntl.flock(lock_f, fcntl.LOCK_EX)
    except OSError:
        lock_f = None
    try:
        try:                                    # a concurrent run may have just tuned this exact config
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                ch = c["choice"]
                say(f"using tuning from a concurrent run: {ch['preset']} ({ch['params'] / 1e6:.2f}M params)")
                return ch
        except (OSError, ValueError, KeyError):
            pass

        threads = _best_threads(device, cfg.n_assets, n_features, cfg.n_quantiles, say)
        torch.set_num_threads(threads)
        steps_per_bar = max(1, cfg.steps_per_label) + 1
        cadence = cfg.bar_seconds * 1000.0 * cfg.autotune_util / steps_per_bar        # must keep up with online training
        warmup_ceiling = cfg.max_warmup_seconds * 1000.0 / max(1, cfg.warmup_steps)   # keep go-live warmup bearable
        budget = min(cadence, warmup_ceiling)
        say(f"device {device} ({threads} cpu threads); step budget {budget:.0f} ms "
            f"(min of {cadence:.0f} ms training cadence, {warmup_ceiling:.0f} ms warmup ceiling "
            f"= {cfg.max_warmup_seconds:.0f}s over {cfg.warmup_steps} steps)")

        chosen, chosen_ms, chosen_n = PRESETS[0], None, None
        for preset in PRESETS:
            try:
                ms, n = _bench(device, preset, cfg.n_assets, n_features, cfg.batch_size, cfg.n_quantiles)
            except Exception as e:  # noqa: BLE001
                say(f"preset {preset[0]} failed ({type(e).__name__}); stopping ladder")
                break
            say(f"benchmark {preset[0]:4}: {n / 1e6:5.2f}M params  {ms:5.0f} ms/step" +
                ("  <- fits" if ms <= budget else "  (over budget)"))
            if ms <= budget:
                chosen, chosen_ms, chosen_n = preset, ms, n
                if ms > budget * 0.85:        # near the ceiling: the next preset is bigger and will be over --
                    say(f"{preset[0]} is near budget; skipping larger presets")   # skip it (a thrashing over-budget benchmark can take minutes + spike swap)
                    break
            else:
                break
        name, d, dil, exp, heads, win = chosen
        if chosen_ms is None:
            chosen_ms, chosen_n = _bench(device, chosen, cfg.n_assets, n_features, cfg.batch_size, cfg.n_quantiles)
        choice = {"device": device, "threads": threads, "d_model": d, "dilations": list(dil), "n_experts": exp,
                  "n_heads": heads, "window": win, "preset": name, "params": chosen_n,
                  "ms_per_step": round(chosen_ms), "budget_ms": round(budget)}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "choice": choice}))
        except OSError:
            pass
        say(f"selected {name}: {chosen_n / 1e6:.2f}M params on {device}, {round(chosen_ms)} ms/step")
        return choice
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
                lock_f.close()
            except OSError:
                pass
