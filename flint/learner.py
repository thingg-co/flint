"""Online training: a replay buffer of labeled windows and a small optimizer loop."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import FlintNet, flint_loss

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Prediction:
    q: np.ndarray      # (assets, quantiles) forecast of the forward return, bps
    p_up: np.ndarray   # (assets,) probability the forward return is positive
    gate: np.ndarray   # (experts,) regime mixture weights
    attn: np.ndarray   # (assets, assets) cross-asset attention


class OnlineLearner:
    def __init__(self, cfg, n_features: int):
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(0)
        self.cfg = cfg
        self.model = FlintNet(n_features, cfg.n_assets, cfg.n_quantiles, cfg.d_model, cfg.dilations,
                              cfg.n_experts, cfg.n_heads, cfg.dropout)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lock = threading.Lock()
        cap = cfg.replay_size
        self.rx = np.zeros((cap, cfg.n_assets, cfg.window, n_features), dtype=np.float32)
        self.ry = np.zeros((cap, cfg.n_assets), dtype=np.float32)
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
                                  cfg.n_experts, cfg.n_heads, cfg.dropout)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.steps = 0
            self.loss_ema = self.pinball_ema = self.bce_ema = None
            if clear_replay:
                self.size = 0
                self.ptr = 0
                self.labels = 0

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, x: np.ndarray) -> Prediction:
        with self.lock, torch.no_grad():
            self.model.eval()
            q, logit, gate, attn = self.model(torch.from_numpy(x)[None])
        return Prediction(q[0].numpy(), torch.sigmoid(logit[0]).numpy(), gate[0].numpy(), attn[0].numpy())

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.rx[self.ptr] = x
        self.ry[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.rx.shape[0]
        self.size = min(self.size + 1, self.rx.shape[0])
        self.labels += 1

    def _sample(self, batch: int):
        cap = self.rx.shape[0]
        n_recent = min(self.cfg.recent_n, self.size)
        k_recent = int(batch * self.cfg.recent_frac)
        recent = (self.ptr - 1 - self.rng.integers(0, n_recent, k_recent)) % cap
        uniform = self.rng.integers(0, self.size, batch - k_recent)
        idx = np.concatenate([recent, uniform])
        return torch.from_numpy(self.rx[idx]), torch.from_numpy(self.ry[idx])

    def train_steps(self, n: int) -> dict | None:
        """Run n optimizer steps. Safe to call from a worker thread."""
        if self.size < 8:
            return None
        batch = min(self.cfg.batch_size, self.size)
        parts = None
        with self.lock:
            self.model.train()
            for _ in range(n):
                x, y = self._sample(batch)
                if self.cfg.input_noise > 0:
                    x = x + self.cfg.input_noise * torch.randn_like(x)
                q, logit, gate, _ = self.model(x)
                loss, parts = flint_loss(q, logit, gate, y, self.cfg.quantiles)
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

    def save(self, directory: str | Path, extra: dict | None = None) -> None:
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
            }, d / "model.pt")
            np.savez_compressed(d / "replay.npz", x=self.rx[:self.size], y=self.ry[:self.size], ptr=self.ptr)

    def load(self, directory: str | Path) -> dict | None:
        d = Path(directory)
        f = d / "model.pt"
        if not f.exists():
            return None
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if tuple(ck.get("shape", ())) != tuple(self.rx.shape[1:]) or ck.get("symbols") != list(self.cfg.symbols):
            log.warning("checkpoint at %s does not match the current config; starting fresh", d)
            return None
        with self.lock:
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.steps = int(ck["steps"])
            self.labels = int(ck["labels"])
            self.loss_ema, self.pinball_ema, self.bce_ema = ck["emas"]
            r = d / "replay.npz"
            if r.exists():
                z = np.load(r)
                n = len(z["y"])
                self.rx[:n] = z["x"]
                self.ry[:n] = z["y"]
                self.size = n
                self.ptr = int(z["ptr"]) % self.rx.shape[0] if n == self.rx.shape[0] else n % self.rx.shape[0]
        return ck.get("extra", {})
