"""Online training: a replay buffer of labeled windows and a small optimizer loop."""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .autotune import autocast, configure_backend, pick_device
from .model import FlintNet, flint_loss

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Prediction:
    q: np.ndarray      # (assets, quantiles) forecast of the forward return, bps
    p_up: np.ndarray   # (assets,) probability the forward return exceeds +threshold
    p_down: np.ndarray # (assets,) probability the forward return is below -threshold
    gate: np.ndarray   # (experts,) regime mixture weights
    attn: np.ndarray   # (assets, assets) cross-asset attention


class OnlineLearner:
    def __init__(self, cfg, n_features: int):
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(0)
        self.cfg = cfg
        dev = pick_device(cfg.device)
        configure_backend(dev)
        self.device = torch.device(dev)
        self.model = FlintNet(n_features, cfg.n_assets, cfg.n_quantiles, cfg.d_model, cfg.dilations,
                              cfg.n_experts, cfg.n_heads, cfg.dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.compile_broken = False       # set when a compiled step fails; eager for the rest of the run
        self.train_model = self._compiled(self.model)
        self.lock = threading.Lock()
        cap = cfg.replay_size
        self.rx = np.zeros((cap, cfg.n_assets, cfg.window, n_features), dtype=np.float32)
        self.ry = np.zeros((cap, cfg.n_assets), dtype=np.float32)
        self.rmask = np.ones((cap, cfg.n_assets), dtype=np.float32)
        self.rt = np.zeros(cap, dtype=np.float64)        # label-bar timestamp per window (0 = unknown, legacy)
        self._seen: set[int] = set()                     # timestamps in the buffer, so a restart's re-walk of the same history adds nothing
        self._replay_at = 0.0                            # when the replay was last written, and with how many labels
        self._replay_labels = -1
        self.size = 0
        self.ptr = 0
        self.steps = 0
        self.labels = 0
        self.loss_ema: float | None = None
        self.pinball_ema: float | None = None
        self.bce_ema: float | None = None
        self.rng = np.random.default_rng(0)

    def reset(self, clear_replay: bool = False) -> None:
        cfg = self.cfg
        with self.lock:
            torch.manual_seed(int(time.time()) % 100000)
            self.model = FlintNet(self.rx.shape[-1], cfg.n_assets, cfg.n_quantiles, cfg.d_model, cfg.dilations,
                                  cfg.n_experts, cfg.n_heads, cfg.dropout).to(self.device)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.train_model = self._compiled(self.model)
            self.steps = 0
            self.loss_ema = self.pinball_ema = self.bce_ema = None
            if clear_replay:
                self.size = 0
                self.ptr = 0
                self.labels = 0
                self.rt[:] = 0.0
                self._seen.clear()

    def _compiled(self, model):
        """torch.compile wrapper for the training step (shares weights with `model`, so
        checkpoints and predict are untouched). Eager on non-CUDA devices, when disabled, or
        once a compile has failed: torch.compile is lazy, so the wrap here always succeeds and
        the real compile runs on the first forward, where train_steps catches it."""
        if not getattr(self.cfg, "compile", False) or self.device.type != "cuda" or self.compile_broken:
            return model
        try:
            t = time.time()
            compiled = torch.compile(model, dynamic=False)
            log.info("torch.compile enabled for the training step (first steps compile; ~%.0fs setup)", time.time() - t)
            return compiled
        except Exception as e:  # noqa: BLE001
            log.warning("torch.compile unavailable (%s: %s); training eager", type(e).__name__, str(e)[:120])
            return model

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, x: np.ndarray) -> Prediction:
        with self.lock, torch.no_grad():
            self.model.eval()
            q, up_logit, down_logit, gate, attn = self.model(torch.from_numpy(x)[None].to(self.device))
        return Prediction(q[0].cpu().numpy(), torch.sigmoid(up_logit[0]).cpu().numpy(),
                          torch.sigmoid(down_logit[0]).cpu().numpy(), gate[0].cpu().numpy(), attn[0].cpu().numpy())

    def add(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, ts: float | None = None) -> bool:
        """Store one labeled window. `ts` is the bar the window ends on; a window for a bar
        already in the buffer is skipped (returns False), so the startup backfill after a
        restart, which walks the same ten days again, cannot stack copies of recent history."""
        key = int(round(ts)) if ts else None
        if key is not None and key in self._seen:
            return False
        old = self.rt[self.ptr]
        if old:
            self._seen.discard(int(round(old)))          # the ring slot being overwritten
        self.rx[self.ptr] = x
        # Apply label cap: mask out symbols with labels exceeding the cap (treat as carry-forward bar)
        cap = self.cfg.label_cap_bps
        if cap > 0:
            label_mask = np.abs(y) <= cap
            if mask is None:
                mask = label_mask.astype(np.float32)
            else:
                mask = mask * label_mask.astype(np.float32)
        self.ry[self.ptr] = y
        self.rmask[self.ptr] = 1.0 if mask is None else mask
        self.rt[self.ptr] = ts or 0.0
        if key is not None:
            self._seen.add(key)
        self.ptr = (self.ptr + 1) % self.rx.shape[0]
        self.size = min(self.size + 1, self.rx.shape[0])
        self.labels += 1
        return True

    def _sample(self, batch: int):
        cap = self.rx.shape[0]
        n_recent = min(self.cfg.recent_n, self.size)
        k_recent = int(batch * self.cfg.recent_frac)
        recent = (self.ptr - 1 - self.rng.integers(0, n_recent, k_recent)) % cap
        uniform = self.rng.integers(0, self.size, batch - k_recent)
        idx = np.concatenate([recent, uniform])
        return torch.from_numpy(self.rx[idx]), torch.from_numpy(self.ry[idx]), torch.from_numpy(self.rmask[idx])

    def train_steps(self, n: int) -> dict | None:
        """Run n optimizer steps. Safe to call from a worker thread."""
        if self.size < 8:
            return None
        batch = min(self.cfg.batch_size, self.size)
        parts = None
        with self.lock:
            self.model.train()
            for _ in range(n):
                x, y, mask = self._sample(batch)
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                if self.cfg.input_noise > 0:
                    x = x + self.cfg.input_noise * torch.randn_like(x)
                with autocast(self.device.type):
                    try:
                        q, up_logit, down_logit, gate, _ = self.train_model(x)
                    except Exception:  # noqa: BLE001
                        if self.train_model is self.model:
                            raise
                        # The first compiled forward is where torch.compile really compiles. A
                        # toolchain problem there (a missing Python.h for Triton's CUDA helper,
                        # a driver mismatch) must cost speed, not the engine: log it once and
                        # train eager from here. An error in the eager retry propagates as before.
                        log.warning("torch.compile failed on its first train step; training eager for the rest "
                                    "of this run (the FLINT_COMPILE speedup is lost). Cause:", exc_info=True)
                        self.compile_broken = True
                        self.train_model = self.model
                        q, up_logit, down_logit, gate, _ = self.train_model(x)
                loss, parts = flint_loss(q.float(), up_logit.float(), down_logit.float(), gate.float(), y, self.cfg.quantiles,
                                         label_smoothing=self.cfg.label_smoothing,
                                         threshold=self.cfg.direction_threshold_bps, mask=mask)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.steps += 1
                a = 0.05
                total = loss.item()
                self.loss_ema = total if self.loss_ema is None else (1 - a) * self.loss_ema + a * total
                self.pinball_ema = parts["pinball"] if self.pinball_ema is None else (1 - a) * self.pinball_ema + a * parts["pinball"]
                self.bce_ema = parts["bce"] if self.bce_ema is None else (1 - a) * self.bce_ema + a * parts["bce"]
        return {"loss": self.loss_ema, "pinball": self.pinball_ema, "bce": self.bce_ema, "steps": self.steps, **(parts or {})}

    # Persistence -----------------------------------------------------------------

    def save(self, directory: str | Path, extra: dict | None = None, *, force_replay: bool = False) -> None:
        """Checkpoint the model every call; the replay only when it has new windows and
        replay_save_minutes have passed, or when forced (shutdown). A full buffer is tens of
        GB, this runs under the training lock, and features compress poorly, so the replay is
        written uncompressed: an NVMe takes seconds where zlib took minutes."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            torch.save({
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "steps": self.steps,
                "labels": self.labels,
                "emas": (self.loss_ema, self.pinball_ema, self.bce_ema),
                "shape": tuple(self.rx.shape[1:]),
                "symbols": list(self.cfg.symbols),
                "extra": extra or {},
            }, d / "model.pt.tmp")
            os.replace(d / "model.pt.tmp", d / "model.pt")                       # atomic: never leave a half-written file
            every = float(getattr(self.cfg, "replay_save_minutes", 30.0)) * 60.0
            if self.labels != self._replay_labels and (force_replay or time.time() - self._replay_at >= every):
                np.savez(d / "replay.tmp.npz", x=self.rx[:self.size], y=self.ry[:self.size], mask=self.rmask[:self.size],
                         t=self.rt[:self.size], ptr=self.ptr)
                os.replace(d / "replay.tmp.npz", d / "replay.npz")
                self._replay_at, self._replay_labels = time.time(), self.labels

    def load(self, directory: str | Path) -> dict | None:
        d = Path(directory)
        f = d / "model.pt"
        if not f.exists():
            return None
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001 -- a corrupt/partial checkpoint must never crash startup
            log.warning("could not read checkpoint %s (%s); starting fresh", f, e)
            return None
        if tuple(ck.get("shape", ())) != tuple(self.rx.shape[1:]) or ck.get("symbols") != list(self.cfg.symbols):
            log.warning("checkpoint at %s does not match the current config "
                        "(saved shape=%s / %d symbols; current shape=%s / %d symbols) — "
                        "backing up to model.pt.bak and starting fresh. If this was not an intended "
                        "config change, restore the .bak to avoid losing training.",
                        d, tuple(ck.get("shape", ())), len(ck.get("symbols") or []),
                        tuple(self.rx.shape[1:]), len(self.cfg.symbols))
            with contextlib.suppress(OSError):        # preserve the trained model instead of clobbering it on next save
                os.replace(f, d / "model.pt.bak")
                if (d / "replay.npz").exists():
                    os.replace(d / "replay.npz", d / "replay.npz.bak")
            return None
        try:
            with self.lock:
                self.model.load_state_dict(ck["model"])
                self.opt.load_state_dict(ck["opt"])
                self.steps = int(ck["steps"])
                self.labels = int(ck["labels"])
                self.loss_ema, self.pinball_ema, self.bce_ema = ck["emas"]
        except Exception as e:  # noqa: BLE001
            log.warning("could not restore model state (%s); starting fresh", e)
            return None
        r = d / "replay.npz"
        if r.exists():
            try:
                with self.lock:
                    z = np.load(r)
                    n = len(z["y"])
                    self.rx[:n] = z["x"]
                    self.ry[:n] = z["y"]
                    self.rmask[:n] = z["mask"] if "mask" in z else 1.0
                    self.rt[:n] = z["t"] if "t" in z else 0.0       # a pre-0.5.10 replay has no timestamps: kept, just not deduplicated
                    self._seen = {int(round(t)) for t in self.rt[:n] if t}
                    self._replay_labels = self.labels                    # what is on disk is what is in memory
                    self.size = n
                    self.ptr = int(z["ptr"]) % self.rx.shape[0] if n == self.rx.shape[0] else n % self.rx.shape[0]
            except Exception as e:  # noqa: BLE001 -- keep the model, just drop an unreadable replay buffer
                log.warning("replay buffer unreadable (%s); keeping model with an empty buffer", e)
                self.size = 0
                self.ptr = 0
        return ck.get("extra", {})
