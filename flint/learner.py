"""Online training: a replay buffer of labeled windows and a small optimizer loop."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .autotune import autocast, configure_backend, pick_device
from .model import FlintNet, flint_loss
from .target import estimate_beta, index_columns, prior_beta, relative

log = logging.getLogger(__name__)
HOLDOUT_EVERY = 20       # every 20th bar by timestamp is validation-only: 5% the trainer never sees
REPLAY_DIR = "replay"    # one .npy per array plus meta.json, memory-mapped on load; replay.npz is the pre-0.5.12 form


def read_replay(directory: str | Path) -> dict | None:
    """The saved replay as arrays plus its metadata: x, y, mask, t (memory-mapped from the
    replay/ directory; read fully from a legacy replay.npz), ptr, n and the symbol order the
    columns were saved in (empty for the legacy file: the checkpoint's list applies)."""
    d = Path(directory)
    rd = d / REPLAY_DIR
    if (rd / "meta.json").exists():
        meta = json.loads((rd / "meta.json").read_text())
        out = {k: np.load(rd / f"{k}.npy", mmap_mode="r") for k in ("x", "y", "mask", "t")}
        out.update({"ptr": int(meta.get("ptr", 0)), "n": int(meta.get("n", len(out["y"]))),
                    "symbols": list(meta.get("symbols") or [])})
        return out
    f = d / "replay.npz"
    if f.exists():
        z = np.load(f)
        n = len(z["y"])
        return {"x": z["x"], "y": z["y"], "mask": z["mask"] if "mask" in z else np.ones_like(z["y"]),
                "t": z["t"] if "t" in z else np.zeros(n), "ptr": int(z["ptr"]), "n": n, "symbols": []}
    return None


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
        self.n_features = n_features
        self.members = max(1, int(getattr(cfg, "members", 1) or 1))   # an ensemble: the members train in turn and forecast as one (item 4)
        self.compile_broken = False       # set when a compiled step fails; eager for the rest of the run
        self.models: list[FlintNet] = []
        self.opts: list[torch.optim.Optimizer] = []
        self.train_models: list = []
        self._build_members()
        self.lock = threading.Lock()
        # The target. The replay keeps absolute labels; with the relative target every consumer
        # (training, validation, live scoring) sees the residual against the modeled index names
        # instead, so the switch needs no reset. No columns means absolute, also when no index
        # is modeled.
        self.cols = (index_columns(cfg.symbols, getattr(cfg, "index_symbols", "SPY"))
                     if str(getattr(cfg, "target", "absolute")).lower() == "relative" else [])
        self.beta = prior_beta(cfg.n_assets, self.cols)
        self._beta_at = -10**9                           # labels count at the last beta fit
        self._alloc()
        self._seen: set[int] = set()                     # timestamps in the buffer, so a restart's re-walk of the same history adds nothing
        self._slot: dict[int, int] = {}                  # timestamp -> ring slot, so a re-walk can fill new names' columns
        self.migrated_new: list[str] = []                # names the last load() added fresh (no history in the replay yet)
        self.loaded_symbols: list[str] = []              # the symbol list the last checkpoint read was saved with (names its norm rows)
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

    def _build_members(self) -> None:
        """Fresh members, each from the next draw of the global seed, so they differ. The
        activations dominate a net's memory and the members train and forecast one after another,
        so an ensemble costs its parameters and optimizer state only, and `members` times the step."""
        cfg = self.cfg
        self.models, self.opts, self.train_models = [], [], []
        for _ in range(self.members):
            model = FlintNet(self.n_features, cfg.n_assets, cfg.n_quantiles, cfg.d_model, cfg.dilations,
                             cfg.n_experts, cfg.n_heads, cfg.dropout).to(self.device)
            self.models.append(model)
            self.opts.append(torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))
            self.train_models.append(self._compiled(model))

    # the first member, for the code (and tests) that speak of one net
    @property
    def model(self):
        return self.models[0]

    @model.setter
    def model(self, m):
        self.models[0] = m

    @property
    def opt(self):
        return self.opts[0]

    @opt.setter
    def opt(self, o):
        self.opts[0] = o

    @property
    def train_model(self):
        return self.train_models[0]

    @train_model.setter
    def train_model(self, m):
        self.train_models[0] = m

    def _alloc(self) -> None:
        cfg = self.cfg
        cap = cfg.replay_windows(self.n_features) if hasattr(cfg, "replay_windows") else cfg.replay_size
        self.rx = np.zeros((cap, cfg.n_assets, cfg.window, self.n_features), dtype=np.float32)
        self.ry = np.zeros((cap, cfg.n_assets), dtype=np.float32)
        self.rmask = np.ones((cap, cfg.n_assets), dtype=np.float32)
        self.rt = np.zeros(cap, dtype=np.float64)        # label-bar timestamp per window (0 = unknown, legacy)
        self.rhold = np.zeros(cap, dtype=bool)           # held out of training (derived from the timestamp)
        self._released = False

    def release_replay(self) -> None:
        """Free the replay arrays (tens of GB) ahead of a rebuild that allocates a new buffer
        of its own; the next load() re-allocates and reads the saved replay back."""
        with self.lock:
            self.rx = self.rx[:0]
            self.ry = self.ry[:0]
            self.rmask = self.rmask[:0]
            self.rt = self.rt[:0]
            self.rhold = self.rhold[:0]
            self.size = self.ptr = 0
            self._seen.clear()
            self._slot.clear()
            self._released = True

    def reset(self, clear_replay: bool = False) -> None:
        cfg = self.cfg
        with self.lock:
            torch.manual_seed(int(time.time()) % 100000)
            self._build_members()
            self.steps = 0
            self.loss_ema = self.pinball_ema = self.bce_ema = None
            if clear_replay:
                self.size = 0
                self.ptr = 0
                self.labels = 0
                self.rt[:] = 0.0
                self.rhold[:] = False
                self._seen.clear()
                self._slot.clear()

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

    def wait_idle(self) -> None:
        """Return once no training step is in flight: the lock is held for the length of a
        train_steps call, so taking and dropping it waits that call out. Called from a thread."""
        with self.lock:
            pass

    def release_cache(self) -> None:
        """Hand the CUDA caching allocator's idle blocks back so another net (the weekend
        backtest) can use them. The weights and optimizer state stay."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for m in self.models for p in m.parameters())

    def _ensemble(self, x: torch.Tensor):
        """The members' forecasts for a batch as one: quantiles, gate and attention averaged (a
        mean of non-crossing quantile vectors does not cross), the direction heads as the mean
        probability, returned as logits so the loss reads them as a single head would. One member
        returns its own tensors untouched."""
        outs = []
        for m in self.models:
            m.eval()
            outs.append(m(x))
        if len(outs) == 1:
            return outs[0]
        q = torch.stack([o[0].float() for o in outs]).mean(0)
        pu = torch.stack([torch.sigmoid(o[1].float()) for o in outs]).mean(0)
        pd_ = torch.stack([torch.sigmoid(o[2].float()) for o in outs]).mean(0)
        gate = torch.stack([o[3].float() for o in outs]).mean(0)
        attn = torch.stack([o[4].float() for o in outs]).mean(0)
        return q, torch.logit(pu, eps=1e-6), torch.logit(pd_, eps=1e-6), gate, attn

    def predict(self, x: np.ndarray) -> Prediction:
        with self.lock, torch.no_grad():
            q, up_logit, down_logit, gate, attn = self._ensemble(torch.from_numpy(x)[None].to(self.device))
        return Prediction(q[0].cpu().numpy(), torch.sigmoid(up_logit[0]).cpu().numpy(),
                          torch.sigmoid(down_logit[0]).cpu().numpy(), gate[0].cpu().numpy(), attn[0].cpu().numpy())

    def add(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, ts: float | None = None) -> bool:
        """Store one labeled window. `ts` is the bar the window ends on; a window for a bar
        already in the buffer is skipped (returns False), so the startup backfill after a
        restart, which walks the same ten days again, cannot stack copies of recent history."""
        key = int(round(ts)) if ts else None
        if key is not None and key in self._seen:
            return False
        if self._released:
            self._alloc()
        old = self.rt[self.ptr]
        if old:
            self._seen.discard(int(round(old)))          # the ring slot being overwritten
            self._slot.pop(int(round(old)), None)
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
        self.rhold[self.ptr] = self._held_out(ts)
        if key is not None:
            self._seen.add(key)
            self._slot[key] = self.ptr
        self.ptr = (self.ptr + 1) % self.rx.shape[0]
        self.size = min(self.size + 1, self.rx.shape[0])
        self.labels += 1
        return True

    def replace(self, ts: float | None, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> bool:
        """Overwrite the window stored for this bar: the startup backfill walks its ten days before
        the news history has caught up, and the deep loop rewrites them once it has. The held-out
        flag, keyed by the timestamp, is untouched. Returns whether the bar was in the buffer."""
        if not ts:
            return False
        slot = self._slot.get(int(round(ts)))
        if slot is None:
            return False
        m = np.ones(self.cfg.n_assets, dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
        cap = self.cfg.label_cap_bps
        if cap > 0:
            m = m * (np.abs(y) <= cap).astype(np.float32)
        self.rx[slot] = x
        self.ry[slot] = y
        self.rmask[slot] = m
        return True

    def fill(self, ts: float | None, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> int:
        """Write the columns of a window already in the buffer (same bar timestamp) that are
        still masked out: the names added to the universe after that window was stored. The
        deep backfill re-walks history after a universe change and fills them this way, so a
        promoted name gets its past without the rest of the replay being rebuilt. Returns the
        number of columns filled."""
        if not ts:
            return 0
        slot = self._slot.get(int(round(ts)))
        if slot is None:
            return 0
        m = np.ones(self.cfg.n_assets, dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
        cap = self.cfg.label_cap_bps
        if cap > 0:
            m = m * (np.abs(y) <= cap).astype(np.float32)
        cols = np.flatnonzero((self.rmask[slot] <= 0) & (m > 0))
        if len(cols) == 0:
            return 0
        self.rx[slot, cols] = x[cols]
        self.ry[slot, cols] = y[cols]
        self.rmask[slot, cols] = m[cols]
        return int(len(cols))

    def _held_out(self, ts: float | None) -> bool:
        """Every HOLDOUT_EVERY-th bar by timestamp. Derived, not stored, so a reload or a
        replay saved before this existed gets the same slice; untimed windows are never held out."""
        return bool(ts) and (int(round(ts)) // max(1, int(self.cfg.bar_seconds))) % HOLDOUT_EVERY == 0

    @property
    def target_mode(self) -> str:
        return "relative" if self.cols else "absolute"

    def target(self, y):
        """Labels as the net is trained to predict them: absolute, or relative to the indices."""
        return relative(y, self.beta, self.cols)

    def refresh_beta(self) -> np.ndarray:
        """Refit each name's index loadings from the labels held, oldest first; a no-op on the
        absolute target. Called once a session from the training step and after a load."""
        if not self.cols:
            return self.beta
        cap = self.rx.shape[0]
        rows = np.arange(self.size) if self.size < cap else (self.ptr + np.arange(cap)) % cap
        if len(rows):
            self.beta = estimate_beta(self.ry[rows], self.rmask[rows], self.cols,
                                      recent=int(getattr(self.cfg, "beta_windows", 6000)),
                                      shrink=float(getattr(self.cfg, "beta_shrink", 400.0)))
        self._beta_at = self.labels
        return self.beta

    def _sample(self, batch: int):
        cap = self.rx.shape[0]
        n_recent = min(self.cfg.recent_n, self.size)
        k_recent = int(batch * self.cfg.recent_frac)
        recent = (self.ptr - 1 - self.rng.integers(0, n_recent, k_recent)) % cap
        uniform = self.rng.integers(0, self.size, batch - k_recent)
        idx = np.concatenate([recent, uniform])
        bad = self.rhold[idx]
        for _ in range(8):                               # redraw held-out picks (5%); a few rounds clear them
            if not bad.any():
                break
            idx[bad] = self.rng.integers(0, self.size, int(bad.sum()))
            bad = self.rhold[idx]
        return torch.from_numpy(self.rx[idx]), torch.from_numpy(self.target(self.ry[idx])), torch.from_numpy(self.rmask[idx])

    def validate(self) -> dict | None:
        """Pinball and direction loss on the held-out windows, which no training step ever
        samples: the honest overfitting signal. In-sample loss rises whenever the deep backfill
        adds unseen history and falls as it is fit, so it cannot tell fitting from overfitting."""
        idx = np.flatnonzero(self.rhold[:self.size])
        if len(idx) < 8:
            return None
        pin = bce = 0.0
        with self.lock, torch.no_grad():
            self.release_cache()                             # the training step's blocks go first: eval shapes differ, and holding
            for k in range(0, len(idx), 4):                  # both stacked the peaks (a 4XL at 362 names was OOM-killed on 2026-10-12);
                j = idx[k:k + 4]                             # small batches: an eval pass must not widen the allocator's footprint
                x = torch.from_numpy(self.rx[j]).to(self.device)
                y = torch.from_numpy(self.target(self.ry[j])).to(self.device)
                mk = torch.from_numpy(self.rmask[j]).to(self.device)
                q, up, dn, gate, _ = self._ensemble(x)
                _, parts = flint_loss(q.float(), up.float(), dn.float(), gate.float(), y, self.cfg.quantiles,
                                      label_smoothing=self.cfg.label_smoothing, threshold=self.cfg.direction_threshold_bps, mask=mk)
                pin += float(parts["pinball"]) * len(j)
                bce += float(parts["bce"]) * len(j)
        self.release_cache()                                 # eval shapes differ from training shapes; do not keep both cached
        return {"pinball": pin / len(idx), "bce": bce / len(idx), "n": int(len(idx))}

    def train_steps(self, n: int) -> dict | None:
        """Run n optimizer steps. Safe to call from a worker thread."""
        if self.size < 8:
            return None
        batch = min(self.cfg.batch_size, self.size)
        parts = None
        if self.cols and self.labels - self._beta_at >= 78:              # about one session of labels
            self.refresh_beta()
        with self.lock:
            for m in self.models:
                m.train()
            for _ in range(n):
                k = self.steps % self.members                        # the members take turns: each sees its own batches
                model, opt, train_model = self.models[k], self.opts[k], self.train_models[k]
                x, y, mask = self._sample(batch)
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                if self.cfg.input_noise > 0:
                    x = x + self.cfg.input_noise * torch.randn_like(x)
                with autocast(self.device.type):
                    try:
                        q, up_logit, down_logit, gate, _ = train_model(x)
                    except Exception:  # noqa: BLE001
                        if train_model is model:
                            raise
                        # The first compiled forward is where torch.compile really compiles. A
                        # toolchain problem there (a missing Python.h for Triton's CUDA helper,
                        # a driver mismatch) must cost speed, not the engine: log it once and
                        # train eager from here. An error in the eager retry propagates as before.
                        log.warning("torch.compile failed on its first train step; training eager for the rest "
                                    "of this run (the FLINT_COMPILE speedup is lost). Cause:", exc_info=True)
                        self.compile_broken = True
                        self.train_models = list(self.models)
                        train_model = model
                        q, up_logit, down_logit, gate, _ = train_model(x)
                loss, parts = flint_loss(q.float(), up_logit.float(), down_logit.float(), gate.float(), y, self.cfg.quantiles,
                                         label_smoothing=self.cfg.label_smoothing,
                                         threshold=self.cfg.direction_threshold_bps, mask=mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
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
                "model": self.models[0].state_dict(),
                "opt": self.opts[0].state_dict(),
                "members": [m.state_dict() for m in self.models[1:]],          # the ensemble's other members (0.5.20); none for one
                "member_opts": [o.state_dict() for o in self.opts[1:]],
                "steps": self.steps,
                "labels": self.labels,
                "emas": (self.loss_ema, self.pinball_ema, self.bce_ema),
                "shape": tuple(self.rx.shape[1:]),
                "symbols": list(self.cfg.symbols),
                "extra": extra or {},
            }, d / "model.pt.tmp")
            os.replace(d / "model.pt.tmp", d / "model.pt")                       # atomic: never leave a half-written file
            every = float(getattr(self.cfg, "replay_save_minutes", 30.0)) * 60.0
            if (self.labels != self._replay_labels and not self._released
                    and (force_replay or time.time() - self._replay_at >= every)):
                self._write_replay(d)
                self._replay_at, self._replay_labels = time.time(), self.labels

    def _write_replay(self, d: Path) -> None:
        """One .npy per array in replay.tmp/, then swapped into replay/: a crash leaves the old
        directory whole. The legacy replay.npz goes once the directory exists."""
        tmp, old, final = d / (REPLAY_DIR + ".tmp"), d / (REPLAY_DIR + ".old"), d / REPLAY_DIR
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        n = self.size
        np.save(tmp / "x.npy", self.rx[:n])
        np.save(tmp / "y.npy", self.ry[:n])
        np.save(tmp / "mask.npy", self.rmask[:n])
        np.save(tmp / "t.npy", self.rt[:n])
        (tmp / "meta.json").write_text(json.dumps({"n": n, "ptr": self.ptr, "symbols": list(self.cfg.symbols),
                                                   "saved_at": time.time()}))
        shutil.rmtree(old, ignore_errors=True)
        if final.exists():
            os.replace(final, old)
        os.replace(tmp, final)
        shutil.rmtree(old, ignore_errors=True)
        with contextlib.suppress(OSError):
            (d / "replay.npz").unlink()

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
        if self._released:
            self._alloc()
        saved_shape = tuple(ck.get("shape", ()))
        saved_syms = list(ck.get("symbols") or [])
        self.loaded_symbols = list(saved_syms)
        cur_shape = tuple(self.rx.shape[1:])
        cur_syms = list(self.cfg.symbols)
        # Only the symbol list changed (same window and features, some names in common): keep every
        # retained name's embedding row and replay columns, give the new names fresh ones. Before
        # 0.5.12 any universe change threw the model and hours of paged history away.
        migrate = (len(saved_shape) == 3 and saved_shape[1:] == cur_shape[1:] and saved_syms != cur_syms
                   and bool(set(saved_syms) & set(cur_syms)))
        self.migrated_new = []
        if (saved_shape != cur_shape or saved_syms != cur_syms) and not migrate:
            log.warning("checkpoint at %s does not match the current config "
                        "(saved shape=%s / %d symbols; current shape=%s / %d symbols) — "
                        "backing up to model.pt.bak and starting fresh. If this was not an intended "
                        "config change, restore the .bak to avoid losing training.",
                        d, saved_shape, len(saved_syms), cur_shape, len(cur_syms))
            with contextlib.suppress(OSError):        # preserve the trained model instead of clobbering it on next save
                os.replace(f, d / "model.pt.bak")
                if (d / "replay.npz").exists():
                    os.replace(d / "replay.npz", d / "replay.npz.bak")
                if (d / REPLAY_DIR).exists():
                    shutil.rmtree(d / (REPLAY_DIR + ".bak"), ignore_errors=True)
                    os.replace(d / REPLAY_DIR, d / (REPLAY_DIR + ".bak"))
            return None
        model_fresh = False
        try:
            with self.lock:
                self._load_members(ck, saved_syms if migrate else None)
                self.steps = int(ck["steps"])
                self.labels = int(ck["labels"])
                self.loss_ema, self.pinball_ema, self.bce_ema = ck["emas"]
        except Exception as e:  # noqa: BLE001
            # The weights do not fit this net (the autotune moved to another preset, say). The
            # replay is data, not architecture: keep it, migrated if the universe changed too,
            # and start the net fresh on it. Before 0.5.14 this path dropped hours of paged history.
            log.warning("could not restore model state (%s); fresh model on the saved replay", e)
            model_fresh = True
            with self.lock:
                self._build_members()
                self.steps = 0
                self.loss_ema = self.pinball_ema = self.bce_ema = None
            self.migrated_new = [s for s in cur_syms if s not in set(saved_syms)]
        try:
            rep = read_replay(d)
        except Exception as e:  # noqa: BLE001 -- keep the model, just drop an unreadable replay buffer
            log.warning("replay buffer unreadable (%s); keeping model with an empty buffer", e)
            rep = None
        if rep is not None:
            try:
                with self.lock:
                    total = int(rep["n"])
                    n = min(total, self.rx.shape[0])
                    skip = total - n                                      # a smaller buffer keeps the newest windows
                    src_syms = list(rep["symbols"] or saved_syms or cur_syms)
                    self._copy_replay(rep, n, src_syms, skip)
                    if src_syms != cur_syms:                              # columns migrated: the new names need the re-page too
                        self.migrated_new = [s for s in cur_syms if s not in set(src_syms)]
                    self.rt[:n] = np.asarray(rep["t"][skip:skip + n], dtype=np.float64)   # a pre-0.5.10 replay has no timestamps: kept, just not deduplicated
                    self._seen = {int(round(t)) for t in self.rt[:n] if t}
                    self._slot = {int(round(t)): i for i, t in enumerate(self.rt[:n]) if t}
                    self.rhold[:n] = [self._held_out(float(t)) for t in self.rt[:n]]
                    if model_fresh or self.labels < n:
                        self.labels = n                                   # the counter can never read below the windows held
                    self._replay_labels = self.labels                    # what is on disk is what is in memory
                    self.size = n
                    self.ptr = (int(rep["ptr"]) % self.rx.shape[0] if (n == self.rx.shape[0] and skip == 0)
                                else n % self.rx.shape[0])
            except Exception as e:  # noqa: BLE001 -- keep the model, just drop an unreadable replay buffer
                log.warning("replay buffer unreadable (%s); keeping model with an empty buffer", e)
                self.size = 0
                self.ptr = 0
        extra = ck.get("extra", {}) or {}
        if model_fresh:                                              # the book and the feature normalization outlive the net; its metrics do not
            extra = {k: v for k, v in extra.items() if k in ("norm", "paper", "meta")}
        self.refresh_beta()
        saved_target = str(extra.get("target") or "absolute")
        if not model_fresh and saved_target != self.target_mode:
            log.warning("checkpoint was trained on the %s target and this run uses the %s one; the net adapts "
                        "online, but its first forecasts carry the old target", saved_target, self.target_mode)
        return extra

    def _copy_replay(self, rep: dict, n: int, src_syms: list[str], skip: int = 0) -> None:
        """Copy n saved windows into the buffer in 256-row chunks (the source is memory-mapped),
        remapping asset columns when the saved symbol order differs from the current one: a
        name no longer modeled is dropped, a new one gets zero features and a zero mask until
        the deep backfill fills it in (fill())."""
        cur = list(self.cfg.symbols)
        if list(src_syms) == cur:
            for k in range(0, n, 256):
                j = slice(k, min(n, k + 256))
                s = slice(skip + k, skip + min(n, k + 256))
                self.rx[j] = rep["x"][s]
                self.ry[j] = rep["y"][s]
                self.rmask[j] = rep["mask"][s]
            return
        pos = {s: i for i, s in enumerate(src_syms)}
        pairs = [(i, pos[s]) for i, s in enumerate(cur) if s in pos]
        ni = np.asarray([p[0] for p in pairs], dtype=np.int64)
        oi = np.asarray([p[1] for p in pairs], dtype=np.int64)
        self.rx[:n] = 0.0
        self.ry[:n] = 0.0
        self.rmask[:n] = 0.0
        for k in range(0, n, 256):
            j = slice(k, min(n, k + 256))
            s = slice(skip + k, skip + min(n, k + 256))
            self.rx[j, ni] = np.asarray(rep["x"][s])[:, oi]
            self.ry[j, ni] = np.asarray(rep["y"][s])[:, oi]
            self.rmask[j, ni] = np.asarray(rep["mask"][s])[:, oi]

    def _load_members(self, ck: dict, migrate_from: list[str] | None) -> None:
        """Load every member the checkpoint holds (the first under the pre-0.5.20 keys), by
        position; a member the checkpoint lacks starts fresh, so a one-member checkpoint seeds
        an ensemble, and an ensemble's first member serves a one-member run. With
        `migrate_from`, the saved symbol list, each member's embedding rows travel by name."""
        sds = [ck["model"]] + list(ck.get("members") or [])
        osds = [ck["opt"]] + list(ck.get("member_opts") or [])
        fresh = 0
        for k, (model, opt) in enumerate(zip(self.models, self.opts)):
            if k >= len(sds) or k >= len(osds):
                fresh += 1
                continue
            if migrate_from is not None:
                self._migrate_one(model, opt, sds[k], osds[k], migrate_from)
            else:
                model.load_state_dict(sds[k])
                opt.load_state_dict(osds[k])
        if fresh:
            log.warning("%d ensemble member(s) start fresh: the checkpoint holds %d", fresh, len(sds))
        elif len(sds) > self.members:
            log.info("the checkpoint holds %d members; this run uses the first %d", len(sds), self.members)
        if migrate_from is not None:
            cur = list(self.cfg.symbols)
            pos = {s: i for i, s in enumerate(migrate_from)}
            self.migrated_new = [s for s in cur if s not in pos]
            log.warning("checkpoint migrated to the new universe: %d names kept, %d dropped, %d new (%s)",
                        len([s for s in cur if s in pos]), len([s for s in migrate_from if s not in set(cur)]),
                        len(self.migrated_new), ", ".join(self.migrated_new[:12]) + ("…" if len(self.migrated_new) > 12 else ""))

    def _migrate_model(self, ck: dict, saved_syms: list[str]) -> None:
        """Load a checkpoint saved for another symbol list (every member)."""
        self._load_members(ck, saved_syms)

    def _migrate_one(self, model, opt, sd: dict, opt_sd: dict, saved_syms: list[str]) -> None:
        """Load one member's weights saved for another symbol list. Every parameter is shared
        across assets except the per-asset embedding table (and its Adam moments), whose rows are
        copied for retained names and left at their fresh initialisation for new ones."""
        cur_syms = list(self.cfg.symbols)
        pos = {s: i for i, s in enumerate(saved_syms)}
        pairs = [(i, pos[s]) for i, s in enumerate(cur_syms) if s in pos]
        ni = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        oi = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        sd = dict(sd)
        emb = model.asset_emb.detach().clone()
        emb[ni.to(emb.device)] = sd["asset_emb"][oi].to(emb.device, emb.dtype)
        sd["asset_emb"] = emb
        model.load_state_dict(sd)
        idx = next(i for i, p in enumerate(model.parameters()) if p is model.asset_emb)
        st = (opt_sd.get("state") or {}).get(idx)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and tuple(st[key].shape) == (len(saved_syms), emb.shape[1]):
                    t = torch.zeros(emb.shape, dtype=st[key].dtype)
                    t[ni] = st[key][oi]
                    st[key] = t
        opt.load_state_dict(opt_sd)
