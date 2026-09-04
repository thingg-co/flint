"""First-run machine benchmark and model auto-sizing.

Picks the compute device (CUDA > MPS > CPU), benchmarks a ladder of model sizes on
it, and chooses the biggest whose training step is fast enough to keep up with the
bar interval. The result is cached to <state_dir>/machine.json so it only benchmarks
once per machine — this is what lets the app adapt when shared to different hardware.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
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
    ("4XL",  512, (1, 2, 4, 8, 16, 32, 64),  8, 8, 128),    # GPU-class presets: a Mac never reaches these
    ("5XL",  768, (1, 2, 4, 8, 16, 32, 64), 12, 12, 128),
]

MEMORY_HEADROOM_GB = 3.0     # left for the OS, the feeds and the dashboard, on top of what the model needs
LADDER_GROWTH = 3.0          # the next rung's peak vs this one's: d_model x1.5 gives ~2.25x params + optimizer state


class NotEnoughMemory(RuntimeError):
    """The box cannot hold the model at its tuned size right now; starting anyway would push a
    unified-memory machine into the OOM killer (the Spark hung this way on 2026-09-04 with an
    84 GB llama-server loaded next to the 5XL benchmark)."""


def free_memory_gb(device: str) -> float | None:
    """Memory the model could still take, in GB, or None when the platform gives no answer.
    Linux reports MemAvailable, which is the truth on unified memory (CUDA's own free figure there
    counts page cache as used: 0.9 GB "free" on the Spark with 34 GB available). A discrete GPU
    is its own pool, so there CUDA's figure is the one that matters."""
    host = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    host = int(line.split()[1]) / 1e6
                    break
    except (OSError, ValueError):
        pass
    if device == "cuda":
        try:
            free, total = torch.cuda.mem_get_info()
            unified = host is not None and abs(total / 1e9 - _host_total_gb()) < 0.15 * total / 1e9
            if not unified:
                return free / 1e9
        except Exception:  # noqa: BLE001
            pass
    if host is None and device == "mps":
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout
            total = int(out.strip()) / 1e9
            host = max(0.0, total - torch.mps.driver_allocated_memory() / 1e9 - 8.0)   # rough: leave 8 GB for everything else
        except Exception:  # noqa: BLE001
            pass
    return host


def _host_total_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except (OSError, ValueError):
        pass
    return 0.0


def other_gpu_tenants() -> list[str]:
    """Names of systemd services that share the accelerator with Flint (llama-server@<model> on the
    Spark), so a memory refusal can say what to stop."""
    try:
        import subprocess
        out = subprocess.run(["systemctl", "list-units", "--type=service", "--state=active", "llama-server@*",
                              "--no-legend", "--plain"], capture_output=True, text=True, timeout=3).stdout
        return [ln.split()[0] for ln in out.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


def _memory_message(need_gb: float, free_gb: float, what: str) -> str:
    msg = (f"not enough memory to {what}: needs about {need_gb:.0f} GB free, {free_gb:.0f} GB available. "
           f"Refusing to start rather than push the machine into the OOM killer.")
    tenants = other_gpu_tenants()
    if tenants:
        msg += f" Running on the same memory: {', '.join(tenants)} -- stop it first (sudo systemctl stop {tenants[0]})."
    return msg


def _peak_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_backend(device: str) -> None:
    """One-time precision/kernel settings. On CUDA: TF32 matmuls and cuDNN autotuning; the
    training step then runs the forward/backward under bf16 autocast (see `autocast`). fp32
    everywhere else. Idempotent."""
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def autocast(device: str):
    """bf16 mixed precision on CUDA; a no-op context elsewhere (MPS bf16 is still patchy)."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _bench(device: str, preset, n_assets: int, n_features: int, batch: int, n_quant: int, iters: int = 6):
    _, d, dil, exp, heads, win = preset
    configure_backend(device)
    dev = torch.device(device)
    quant = tuple((i + 1) / (n_quant + 1) for i in range(n_quant))
    m = FlintNet(n_features, n_assets, n_quant, d_model=d, dilations=dil, n_experts=exp, n_heads=heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(batch, n_assets, win, n_features, device=dev)
    y = torch.randn(batch, n_assets, device=dev)
    _reset_peak(device)

    def step():
        with autocast(device):
            q, up, down, g, _ = m(x)
        loss, _ = flint_loss(q.float(), up.float(), down.float(), g.float(), y, quant)
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
    peak = _peak_gb(device)
    del m, opt, x, y
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return ms, n, peak


def _best_threads(device: str, n_assets: int, n_features: int, n_quant: int, say) -> int:
    cores = os.cpu_count() or 4
    if device != "cpu":
        return max(2, min(8, cores // 2))     # GPU does the math; CPU threads sample the replay and build features
    best_ms, best_th = 1e9, 2
    for th in sorted({2, cores // 2, max(2, int(cores * 0.6)), cores - 2}):
        if th < 1:
            continue
        torch.set_num_threads(th)
        ms, _, _ = _bench("cpu", PRESETS[1], n_assets, n_features, 16, n_quant, iters=4)
        say(f"thread sweep: {th} threads -> {ms:.0f} ms/step")
        if ms < best_ms:
            best_ms, best_th = ms, th
    return best_th


def _check_start_memory(cfg, n_features: int, choice: dict, say) -> None:
    """Refuse to build the chosen preset when the box cannot hold it: the benchmark's peak plus the replay
    buffer (float32 windows, allocated up front) plus headroom, against what is free right now."""
    peak = float(choice.get("peak_gb") or 0.0)
    if peak <= 0:
        return
    replay = cfg.replay_size * cfg.n_assets * choice["window"] * n_features * 4 / 1e9
    need = peak * 1.25 + replay + MEMORY_HEADROOM_GB
    free = free_memory_gb(choice["device"])
    say(f"memory: model peak {peak:.1f} GB + replay {replay:.1f} GB -> needs ~{need:.0f} GB, "
        f"{'unknown' if free is None else f'{free:.0f} GB'} free")
    if free is not None and free < need:
        raise NotEnoughMemory(_memory_message(need, free, f"start preset {choice['preset']}"))


def autotune(cfg, n_features: int, say=None) -> dict:
    say = say or (lambda *a, **k: log.info(*a))
    device = pick_device(cfg.device)
    configure_backend(device)
    cache = Path(cfg.state_dir) / "machine.json"
    key = (f"{device}|bf16|mem|{os.cpu_count()}|{cfg.n_assets}|{n_features}|{cfg.bar_seconds}|{cfg.batch_size}"
           f"|{cfg.autotune_util}|{cfg.max_warmup_seconds}|{cfg.warmup_steps}")
    try:
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            ch = c["choice"]
            say(f"using cached tuning: {ch['preset']} ({ch['params'] / 1e6:.2f}M params) on {ch['device']} "
                f"at {ch['ms_per_step']} ms/step")
            _check_start_memory(cfg, n_features, ch, say)
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

        tenants = other_gpu_tenants()
        if tenants:
            say(f"note: {', '.join(tenants)} is loaded on the same memory; the ladder stops where it runs out")
        chosen, chosen_ms, chosen_n, chosen_peak = PRESETS[0], None, None, 0.0
        last_peak = 0.0
        for preset in PRESETS:
            need = max(1.0, last_peak * LADDER_GROWTH) + MEMORY_HEADROOM_GB     # the rung after a 4 GB one may take 12
            free = free_memory_gb(device)
            if free is not None and free < need:
                # A ladder cut short by memory is a transient condition, not this machine's speed: caching a
                # smaller preset here would silently downgrade the model for good. Refuse instead.
                raise NotEnoughMemory(_memory_message(need, free, f"benchmark preset {preset[0]}"))
            try:
                ms, n, peak = _bench(device, preset, cfg.n_assets, n_features, cfg.batch_size, cfg.n_quantiles)
            except Exception as e:  # noqa: BLE001
                say(f"preset {preset[0]} failed ({type(e).__name__}); stopping ladder")
                break
            last_peak = peak
            say(f"benchmark {preset[0]:4}: {n / 1e6:5.2f}M params  {ms:5.0f} ms/step  {peak:4.1f} GB peak" +
                ("  <- fits" if ms <= budget else "  (over budget)"))
            if ms <= budget:
                chosen, chosen_ms, chosen_n, chosen_peak = preset, ms, n, peak
                if ms > budget * 0.85:        # near the ceiling: the next preset is bigger and will be over --
                    say(f"{preset[0]} is near budget; skipping larger presets")   # skip it (a thrashing over-budget benchmark can take minutes + spike swap)
                    break
            else:
                break
        name, d, dil, exp, heads, win = chosen
        if chosen_ms is None:
            chosen_ms, chosen_n, chosen_peak = _bench(device, chosen, cfg.n_assets, n_features, cfg.batch_size, cfg.n_quantiles)
        choice = {"device": device, "threads": threads, "d_model": d, "dilations": list(dil), "n_experts": exp,
                  "n_heads": heads, "window": win, "preset": name, "params": chosen_n,
                  "ms_per_step": round(chosen_ms), "budget_ms": round(budget), "peak_gb": round(chosen_peak, 2)}
        _check_start_memory(cfg, n_features, choice, say)
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
