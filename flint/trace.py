"""Introspection bus: named channels of short lines describing what the system is doing."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

CHANNELS = ["feed", "bars", "features", "model", "policy", "learn", "news", "signals", "operator", "system"]


class Trace:
    def __init__(self, publish: Callable[[dict], None], keep: int = 200):
        self.buffers: dict[str, deque] = {ch: deque(maxlen=keep) for ch in CHANNELS}
        self._publish = publish
        self.seq = 0
        self.muted = False  # set while replaying history so the consoles are not flooded

    def emit(self, channel: str, text: str, level: str = "info", data: dict | None = None) -> None:
        if channel not in self.buffers:
            raise KeyError(channel)
        self.seq += 1
        ev: dict = {"seq": self.seq, "t": time.time(), "ch": channel, "lvl": level, "text": text}
        if data is not None:
            ev["data"] = data
        self.buffers[channel].append(ev)
        if not self.muted:
            self._publish({"type": "trace", "ev": ev})

    def recent(self) -> dict[str, list]:
        return {ch: list(buf) for ch, buf in self.buffers.items()}
