"""Headline tone, graded.

Two scorers behind one interface, `score(texts) -> [float in -1..1]`:

- `LexiconScorer`: the word lists, graded. The old rule, (positive − negative) / (positive +
  negative), made every headline with one tone word read ±1.0 and everything else 0.0, so three
  values carried the whole feed. Now one tone word reads ±0.5, two agreeing words ±0.67, a
  hedge ("could", "may", "might", a question mark) takes a third off, and mixed words cancel.
  Deterministic, no dependencies, the fallback.
- `FinBertScorer`: ProsusAI/finbert, a BERT fine-tuned on financial news, run locally on the
  CPU through `transformers`. Tone = P(positive) − P(negative), so "Workday faces slower growth
  and widening gap with peers" reads negative instead of +1.0 for the word "growth". The model
  (about 440 MB) is fetched from the Hugging Face hub on first use and cached under
  ~/.cache/huggingface; scores are cached per title so a pass costs inference only for new
  headlines. If the package or the model is unavailable it says so once and the lexicon takes
  over, so a skim never fails over tone.

Nothing leaves the machine but the one-time model download.
"""
from __future__ import annotations

import logging
import math
import re
import threading

log = logging.getLogger(__name__)

POSITIVE = ["surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps", "gain", "gains", "record", "high",
            "highs", "approve", "approves", "approval", "adoption", "inflow", "inflows", "upgrade", "breakout", "bull",
            "bullish", "rebound", "rebounds", "partnership", "launch", "launches", "buy", "buys", "buying", "accumulate",
            "milestone", "boost", "boosts", "climb", "climbs", "recover", "recovers", "recovery", "optimism", "outperform",
            "rise", "rises", "rising", "up", "growth", "wins", "win", "beat", "beats", "raises", "upgrades"]
NEGATIVE = ["plunge", "plunges", "crash", "crashes", "drop", "drops", "fall", "falls", "slump", "slumps", "slide", "slides",
            "hack", "hacked", "exploit", "lawsuit", "sue", "sues", "sued", "ban", "bans", "outflow", "outflows", "sell-off",
            "selloff", "liquidation", "liquidations", "bear", "bearish", "fraud", "halt", "halts", "fine", "fined", "probe",
            "delay", "delays", "dump", "dumps", "low", "lows", "decline", "declines", "loss", "losses", "warning", "warns",
            "risk", "fear", "tumble", "tumbles", "sink", "sinks", "collapse", "scam", "charges", "charged", "bankrupt",
            "bankruptcy", "down", "falling", "dip", "dips", "crackdown", "miss", "misses", "cuts", "downgrade", "downgrades",
            "slower", "weak", "weaker"]
_POS = re.compile(r"\b(" + "|".join(map(re.escape, POSITIVE)) + r")\b", re.I)
_NEG = re.compile(r"\b(" + "|".join(map(re.escape, NEGATIVE)) + r")\b", re.I)
_HEDGE = re.compile(r"\b(could|may|might|whether|if)\b|\?", re.I)


def lexicon_tone(text: str) -> float:
    pos = len(_POS.findall(text))
    neg = len(_NEG.findall(text))
    if not pos and not neg:
        return 0.0
    tone = (pos - neg) / (pos + neg + 1.0)
    if _HEDGE.search(text):
        tone *= 2.0 / 3.0
    return round(tone, 3)


class LexiconScorer:
    name = "lexicon"

    def score(self, texts: list[str]) -> list[float]:
        return [lexicon_tone(t) for t in texts]


class FinBertScorer:
    """ProsusAI/finbert on the CPU, loaded on first use, with a per-title cache. Falls back to
    the lexicon, and reports it in `name`, when transformers or the model is unavailable."""
    name = "finbert"

    def __init__(self, model: str = "ProsusAI/finbert", threads: int = 2, cache_max: int = 20000, batch: int = 32):
        self.model_name = model
        self.threads = int(threads)
        self.batch = int(batch)
        self.cache: dict[str, float] = {}
        self.cache_max = int(cache_max)
        self._lock = threading.Lock()
        self._pipe = None
        self._failed = False
        self.fallback = LexiconScorer()

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        if self._failed:
            return False
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            mdl.eval()
            labels = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
            self._pipe = (tok, mdl, labels, torch)
            log.info("headline tone: %s loaded on the CPU (%d labels)", self.model_name, len(labels))
            return True
        except Exception as e:  # noqa: BLE001
            self._failed = True
            self.name = "lexicon (finbert unavailable)"
            log.warning("headline tone: %s unavailable (%s: %s); using the word lists", self.model_name, type(e).__name__, str(e)[:160])
            return False

    def score(self, texts: list[str]) -> list[float]:
        texts = [str(t) for t in texts]
        out: list[float | None] = [self.cache.get(t) for t in texts]
        todo = sorted({t for t, v in zip(texts, out) if v is None})
        if todo:
            with self._lock:
                if self._load():
                    tok, mdl, labels, torch = self._pipe
                    torch.set_num_threads(max(1, self.threads))
                    with torch.no_grad():
                        for i in range(0, len(todo), self.batch):
                            chunk = todo[i:i + self.batch]
                            enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
                            probs = torch.softmax(mdl(**enc).logits.float(), dim=-1)
                            for t, p in zip(chunk, probs):
                                pos = sum(float(p[k]) for k, lab in labels.items() if lab.startswith("pos"))
                                neg = sum(float(p[k]) for k, lab in labels.items() if lab.startswith("neg"))
                                self.cache[t] = round(max(-1.0, min(1.0, pos - neg)), 3)
                else:
                    for t, v in zip(todo, self.fallback.score(todo)):
                        self.cache[t] = v
                if len(self.cache) > self.cache_max:                       # forget the oldest half
                    for k in list(self.cache)[: len(self.cache) // 2]:
                        self.cache.pop(k, None)
        return [self.cache.get(t, 0.0) for t in texts]


def get_scorer(cfg):
    """The configured scorer: `news_scorer` finbert (default) or lexicon."""
    which = str(getattr(cfg, "news_scorer", "lexicon") or "lexicon").lower()
    if which == "finbert":
        return FinBertScorer(str(getattr(cfg, "finbert_model", "ProsusAI/finbert")),
                             threads=int(getattr(cfg, "torch_threads", 2) or 2))
    return LexiconScorer()


def is_graded(values) -> bool:
    """Whether a set of tones carries more than the three values the old rule produced."""
    return len({round(float(v), 2) for v in values}) > 3 and any(0.05 < abs(float(v)) < 0.95 for v in values) and not math.isnan(sum(map(float, values)))
