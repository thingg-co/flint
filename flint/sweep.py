"""The policy sweep: the gates of `backtest.decide` swept over the forecasts a backtest stored,
without re-running the net.

A walk-forward run writes every raw forecast it made (`backtest_forecasts_<target>.npz`).
Replaying the policy over them is pure arithmetic: the same lagged calibration (band scale,
temperature, coverage and hit EMAs, per-name records, all updated one horizon late), the
same gates and sizing, the same hedge, the same costs, so `replay_policy` reproduces the
run's net to the bps and any other setting of the gates is a half-second away.

`run_sweep` tunes the gates by coordinate descent on the first half of the scored days and
confirms on the second half, and proposes the result only when the confirmed half agrees.
The proposal is shown on the dashboard and never applied: an operator reads it against the
live controls and decides. Tuning and confirming on disjoint halves is the minimum defence
against fitting the gates to the noise of one history; the second half is still one sample.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np

GRID: dict[str, list] = {
    "score_threshold": [0.2, 0.3, 0.35, 0.45, 0.6, 0.8],
    "prob_margin": [0.02, 0.04, 0.06, 0.09, 0.12],
    "move_floor_bps": [3.0, 5.0, 8.0, 12.0, 16.0, 24.0],
    "kelly_fraction": [0.08, 0.15, 0.25, 0.4, 0.6],
    "confirm_bars": [1, 2, 3, 4],
    "min_hit_rate": [0.45, 0.5, 0.52, 0.55, 0.58],
    "skill_min_n": [0, 4, 8, 16],
}
MIN_SIGNALS_PER_DAY = 0.5           # a setting that barely trades has no net worth comparing


def load_forecasts(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["symbols"] = [str(s) for s in out["symbols"]]
    out["index"] = [str(s) for s in out["index"]] if "index" in out else []
    out["target"] = str(out["target"]) if "target" in out else "absolute"
    out["path"] = str(path)
    return out


def _temper(p_raw: np.ndarray, p_scale: float) -> tuple[np.ndarray, np.ndarray]:
    pr = np.clip(p_raw, 1e-6, 1 - 1e-6)
    z = np.clip(np.log(pr / (1 - pr)), -6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-p_scale * z)), z


def replay_policy(fc: dict, cfg, cost_bps: float, overrides: dict | None = None,
                  day_lo: int | None = None, day_hi: int | None = None) -> dict:
    """The backtest's scoring loop over stored forecasts, vectorized over names, for the
    gates in `cfg` with `overrides` applied. Days outside [day_lo, day_hi) are replayed
    (the calibration state has to evolve through them) but not counted."""
    cfg = copy.copy(cfg)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    q, up, dn, y, mask = fc["q"], fc["up"], fc["dn"], fc["y"], fc["mask"]
    day_of = fc["day_of"].astype(np.int64)
    betas = fc["beta"]
    symbols = fc["symbols"]
    icols = [symbols.index(s) for s in fc["index"] if s in symbols]
    n_bars, n_assets = y.shape
    taus = np.asarray(cfg.quantiles, dtype=np.float64)
    horizon = int(cfg.horizon)
    cost = float(cost_bps)
    lo = 0 if day_lo is None else int(day_lo)
    hi = int(day_of.max()) + 1 if day_hi is None else int(day_hi)

    band_scale, p_scale = 2.0, 0.3
    coverage_ema = hit_ema = None
    hist = np.zeros((n_assets, 40), dtype=np.float64)              # per-name record of the last 40 direction calls
    hist_n = np.zeros(n_assets, dtype=np.int64)
    hist_ptr = np.zeros(n_assets, dtype=np.int64)
    prev_side = np.zeros(n_assets, dtype=np.int64)
    run = np.zeros(n_assets, dtype=np.int64)
    pending: list = []
    day_net: dict[int, float] = {}
    day_bars: dict[int, int] = {}
    day_sig: dict[int, int] = {}
    tot = {"hits": 0, "decisions": 0, "pin": 0.0, "pin_n": 0, "inside": 0, "inside_n": 0}

    for b in range(n_bars):
        # resolve what matured: the calibration moves one horizon late, as live
        while pending and pending[0][0] + horizon <= b:                   # the backtest counts bars from 1 and resolves before deciding
            _bar, hits_b, inside_b, pg_b, per_i, per_hit = pending.pop(0)
            if hits_b is not None:
                hit_ema = hits_b if hit_ema is None else 0.98 * hit_ema + 0.02 * hits_b
            if inside_b is not None:
                coverage_ema = inside_b if coverage_ema is None else 0.98 * coverage_ema + 0.02 * inside_b
                band_scale = float(np.clip(band_scale * math.exp(cfg.band_gamma * ((1.0 - inside_b) - 0.2)), 0.5, 25.0))
            if pg_b is not None:
                p_scale = float(np.clip(p_scale - cfg.temper_lr * pg_b, 0.02, 1.5))
            if len(per_i):
                hist[per_i, hist_ptr[per_i]] = per_hit
                hist_ptr[per_i] = (hist_ptr[per_i] + 1) % 40
                hist_n[per_i] = np.minimum(hist_n[per_i] + 1, 40)
        d = int(day_of[b])
        traded = mask[b] > 0
        qb = q[b].astype(np.float64)
        q50 = qb[:, 2]
        qc = q50[:, None] + band_scale * (qb - q50[:, None])
        p, z = _temper(up[b].astype(np.float64), p_scale)
        iqr = np.maximum(qc[:, 3] - qc[:, 1], 1e-3)
        score = q50 / iqr
        ok = ((np.abs(score) >= cfg.score_threshold) & (np.abs(2 * p - 1) >= 2 * cfg.prob_margin)
              & (np.abs(q50) - cost > 0) & (np.abs(q50) >= cfg.move_floor_bps) & ((score > 0) == (p > 0.5)))
        rate = np.where(hist_n > 0, hist.sum(axis=1) / np.maximum(hist_n, 1), 1.0)
        ok &= ~((hist_n < cfg.skill_min_n) | ((hist_n > 0) & (rate < cfg.min_hit_rate)))
        raw_side = np.where(ok, np.sign(score).astype(np.int64), 0)
        new_run = np.where(prev_side == raw_side, run + 1, 1)
        run = np.where(traded, new_run, run)                              # names that did not trade keep their streak
        prev_side = np.where(traded, raw_side, prev_side)
        side = np.where((raw_side != 0) & (run >= cfg.confirm_bars), raw_side, 0)
        size = cfg.max_size * np.minimum(1.0, cfg.kelly_fraction * np.abs(score))
        if cfg.size_by_coverage and coverage_ema is not None:
            size = size * max(0.25, min(1.0, coverage_ema / 0.8))
        tradeable = hit_ema is not None and hit_ema > cfg.min_hit_rate
        w = np.where(traded & (side != 0), side * size, 0.0) if tradeable else np.zeros(n_assets)
        # scoring of the forecasts themselves, for the calibration state and the record
        yb = y[b].astype(np.float64)
        ti = np.flatnonzero(traded)
        if len(ti):
            diff = yb[ti, None] - qb[ti]
            pin = np.maximum(taus * diff, (taus - 1) * diff).mean(axis=1)
            dec = ti[(yb[ti] != 0) & (q50[ti] != 0)]
            hits = ((q50[dec] > 0) == (yb[dec] > 0)).astype(np.float64)
            inside = ((qc[ti, 0] <= yb[ti]) & (yb[ti] <= qc[ti, -1])).astype(np.float64)
            nz = ti[yb[ti] != 0]
            pg = ((p[nz] - (yb[nz] > 0)) * z[nz])
            pending.append((b, float(hits.mean()) if len(hits) else None, float(inside.mean()),
                            float(pg.mean()) if len(pg) else None, dec, hits))
            if lo <= d < hi:
                tot["pin"] += float(pin.sum()); tot["pin_n"] += len(ti)
                tot["hits"] += int(hits.sum()); tot["decisions"] += len(dec)
                tot["inside"] += int(inside.sum()); tot["inside_n"] += len(ti)
        else:
            pending.append((b, None, None, None, np.zeros(0, dtype=np.int64), np.zeros(0)))
        # the book
        if icols:
            w[icols] = 0.0
        gross = float(np.abs(w).sum())
        if gross > 1.0:
            w = w / gross
        n_sig = int((w != 0).sum())
        if icols:
            beta = betas[d]
            w[icols] = -(w @ beta)
        ret = float((w * yb).sum()) if not icols else float((np.delete(w, icols) * np.delete(yb, icols)).sum())
        fee = cost * float(np.abs(w).sum())
        day_net[d] = day_net.get(d, 0.0) + ret - fee
        day_bars[d] = day_bars.get(d, 0) + 1
        day_sig[d] = day_sig.get(d, 0) + n_sig

    days = [d for d in sorted(day_net) if lo <= d < hi]
    nets = np.asarray([day_net[d] / max(1, day_bars[d]) for d in days], dtype=np.float64)
    cum = peak = max_dd = 0.0
    for v in nets:
        cum += v
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    sd = float(nets.std(ddof=1)) if len(nets) > 1 else 0.0
    return {"days": len(days), "signals": int(sum(day_sig[d] for d in days)),
            "net_bps": float(nets.sum()), "net_bps_per_day": float(nets.mean()) if len(nets) else 0.0,
            "sharpe": (float(nets.mean() / sd * math.sqrt(252)) if sd > 0 else None), "max_drawdown_bps": float(max_dd),
            "hit_rate": (tot["hits"] / tot["decisions"]) if tot["decisions"] else None,
            "coverage": (tot["inside"] / tot["inside_n"]) if tot["inside_n"] else None,
            "pinball": (tot["pin"] / tot["pin_n"]) if tot["pin_n"] else None,
            "win_days": int((nets > 0).sum()), "nets": [float(v) for v in nets]}


def _objective(r: dict) -> float:
    if r["days"] and r["signals"] / r["days"] < MIN_SIGNALS_PER_DAY:
        return -1e9
    return r["net_bps_per_day"]


def run_sweep(fc: dict, cfg, cost_bps: float, grid: dict | None = None, max_passes: int = 3,
              progress=None) -> dict:
    """Coordinate descent over `grid` on the first half of the scored days, confirmed on the
    second half. Returns the report: the current gates and the proposal with both halves'
    results, whether the proposal is accepted, and every step taken."""
    t0 = time.time()
    grid = grid or GRID
    n_days = int(fc["day_of"].max()) + 1
    split = n_days // 2
    if n_days < 6:
        return {"error": f"only {n_days} scored days; the sweep needs at least 6", "ran_at": t0}
    current = {k: getattr(cfg, k) for k in grid}
    knobs = dict(current)
    best = replay_policy(fc, cfg, cost_bps, knobs, 0, split)
    steps: list[dict] = []
    evals = 1
    for _pass in range(max_passes):
        changed = False
        for k, values in grid.items():
            trial_best = None
            for v in values:
                if v == knobs[k]:
                    continue
                trial = dict(knobs, **{k: v})
                r = replay_policy(fc, cfg, cost_bps, trial, 0, split)
                evals += 1
                if progress:
                    progress(evals)
                if _objective(r) > _objective(best) + 1e-9 and (trial_best is None or _objective(r) > _objective(trial_best[1])):
                    trial_best = (v, r)
            if trial_best is not None:
                knobs[k] = trial_best[0]
                best = trial_best[1]
                changed = True
                steps.append({"knob": k, "value": trial_best[0], "tune_net_per_day": best["net_bps_per_day"],
                              "tune_signals": best["signals"]})
        if not changed:
            break
    proposal = {k: v for k, v in knobs.items() if v != current[k]}
    cur_tune = replay_policy(fc, cfg, cost_bps, current, 0, split)
    cur_conf = replay_policy(fc, cfg, cost_bps, current, split, n_days)
    prop_conf = replay_policy(fc, cfg, cost_bps, knobs, split, n_days) if proposal else cur_conf
    accepted = bool(proposal) and _objective(prop_conf) > _objective(cur_conf) + 1e-9
    why = ("no setting beat the current gates on the tuning half" if not proposal else
           ("the proposal also beats the current gates on the confirming half" if accepted else
            "the proposal won the tuning half but not the confirming half: likely noise, not proposed"))
    strip = lambda r: {k: v for k, v in r.items() if k != "nets"}  # noqa: E731
    return {"ran_at": t0, "elapsed_s": time.time() - t0, "forecasts": fc.get("path"), "target": fc.get("target"),
            "cost_bps": float(cost_bps), "days": n_days, "split_day": split, "evaluations": evals,
            "current": {"knobs": current, "tune": strip(cur_tune), "confirm": strip(cur_conf)},
            "proposal": ({"knobs": proposal, "all": knobs, "tune": strip(best), "confirm": strip(prop_conf)} if proposal else None),
            "accepted": accepted, "why": why, "steps": steps}
