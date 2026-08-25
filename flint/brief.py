"""Local narrative brief: fast "analyst" models pre-digest each slice of flint's
state, then a bigger local model writes the newspaper-style column. All inference
runs on the local Ollama runtime, so nothing about the market read leaves the machine.
"""
from __future__ import annotations

import asyncio
import re
import time

import httpx

_THINK = re.compile(r"<think>.*?</think>", re.S)

# slice name -> (analyst persona, what to focus on)
ANALYSTS = {
    "tape": ("a quant on the trading desk",
             "In 2-4 plain sentences say what flint's neural forecasts imply right now: overall "
             "direction and conviction, whether it is still warming up, and the two or three names "
             "that stand out and why. No lists, no preamble."),
    "macro": ("a markets reporter",
              "In 2-4 plain sentences describe the backdrop: breadth, which sectors lead and lag, "
              "volatility, and the day's most notable movers. No lists, no preamble."),
    "positioning": ("a flows and sentiment analyst",
                    "In 2-3 plain sentences describe positioning and mood: fear and greed, the most "
                    "crowded names, and where retail attention sits. No lists, no preamble."),
    "smart_money": ("an analyst who reads 13F filings",
                    "In 2-3 plain sentences describe what the tracked investors lean toward: their "
                    "notable long and short tilts and the panel's overall bias. No lists, no preamble."),
}


async def _chat(host, model, prompt, system=None, num_predict=400, timeout=300):
    body = {
        "model": model, "stream": False, "think": False,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.6, "num_predict": num_predict},
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{host}/api/chat", json=body)
        r.raise_for_status()
        txt = r.json().get("message", {}).get("content", "")
    return _THINK.sub("", txt).strip()


async def available(host, timeout=5):
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{host}/api/tags")
            return {m["name"] for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return set()


def _pick(want, have):
    if want in have:
        return want
    for h in have:                       # tolerate a missing/other :tag
        if h.split(":")[0] == want.split(":")[0]:
            return h
    return None


async def write_brief(cfg, slices, say=None):
    say = say or (lambda *a, **k: None)
    host = cfg.ollama_host
    have = await available(host)
    if not have:
        return {"error": f"no local LLM found — start Ollama ({host}) with `ollama serve`", "t": time.time()}
    big = _pick(cfg.brief_model, have)
    small = _pick(cfg.brief_small_model, have) or big
    if not big:
        return {"error": f"model '{cfg.brief_model}' not pulled — run `ollama pull {cfg.brief_model}`", "t": time.time()}

    async def analyst(name):
        role, focus = ANALYSTS[name]
        sysmsg = (f"You are {role} writing a terse internal note for a colleague. Be concrete and "
                  "specific with the numbers you are given, and never invent figures.")
        try:
            say(f"analyst {name} ({small})")
            return name, await _chat(host, small, f"{focus}\n\nData:\n{slices.get(name, '(none)')}", sysmsg, 240, timeout=240)
        except Exception:  # noqa: BLE001
            return name, None

    takes = dict(await asyncio.gather(*[analyst(n) for n in ANALYSTS]))
    notes = "\n\n".join(f"{n.upper()} DESK: {t}" for n, t in takes.items() if t) or "(desk notes unavailable)"

    sysmsg = ("You write the daily column for a newspaper's investing section. Write like a seasoned "
              "human columnist: plain, specific, varied sentence length. No bullet points, no bold, no "
              "subheadings, no hedging filler ('it's worth noting', 'crucially', 'in conclusion'), no "
              "AI throat-clearing. Use only the numbers in the notes and figures; never invent any. This "
              "is a machine's market read (flint, a neural net still learning), so be honest about "
              "uncertainty rather than overconfident.")
    prompt = ("Write today's column: a short headline on its own first line, then four to six paragraphs "
              "that weave the desk notes and figures into one coherent read of where the market and the "
              f"model stand.\n\nDESK NOTES\n{notes}\n\nKEY FIGURES\n{slices.get('stats', '')}")
    say(f"writing column ({big})")
    text = await _chat(host, big, prompt, sysmsg, 900, timeout=getattr(cfg, "brief_timeout", 1800))
    return {"text": text, "takes": takes, "models": {"small": small, "big": big}, "t": time.time(), "error": None}
