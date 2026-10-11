"""Walk-forward backtest over the timestamped replay.

How this architecture and this policy would have done on the history Flint holds, out of
sample by construction. A fresh net of the live preset walks the replay day by day in time
order: it forecasts every window of day D before it has trained on any label from day D,
the live policy (the gates and sizing of Engine._suggest that history can answer) turns the
forecasts into signals, the realised hour-ahead returns score them, and only then does the
net train on day D with the live learner's sampling (recent-biased over everything up to D).
The first `warm_days` are training only, like warmup; signals also wait for the live trust
rule (a hit-rate EMA above min_hit_rate), and every calibration update (band scale, P(up)
temperature, per-name records) lands one horizon late, as it does live.

Book P&L: per bar the signals form a one-hour portfolio of weights with gross capped at 1,
as the paper book does, and the bar's return is sum(w * y) - cost * sum(|w|). A day's return
is the mean over its bars, since capital can only fund one hour at a time; the curve is the
running sum in bps. Runs on the weekend GPU (Engine._backtest_loop) or as `flint backtest`.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .autotune import autocast
from .model import FlintNet, flint_loss
from .meta import MetaLabeler, features as meta_features
from .target import estimate_beta, index_columns, prior_beta, relative

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


def _ema(prev, v, a):
    return v if prev is None else (1 - a) * prev + a * v


def temper(p_raw: float, p_scale: float) -> float:
    """The online P(up) temperature, as Engine._temper."""
    p_raw = min(max(p_raw, 1e-6), 1 - 1e-6)
    z = max(-6.0, min(6.0, math.log(p_raw / (1 - p_raw))))
    return 1.0 / (1.0 + math.exp(-p_scale * z))


def decide(cfg, q, p_raw, p_down_raw, band_scale, p_scale, coverage_ema, judged, streak, cost_bps, meta_p=None):
    """The gates and sizing of Engine._suggest that history can answer: score, conviction, cost
    edge, per-name record, the meta-labeler's odds, move floor, trend/direction agreement,
    confirmation, and the fractional-Kelly size scaled by band coverage. Live-only gates (quote spread, session,
    muted names, the council overlay) are not here. Returns (side, size, q_calibrated, p, streak)."""
    q50 = float(q[2])
    qc = [q50 + band_scale * (float(v) - q50) for v in q]
    p = temper(p_raw, p_scale)
    iqr = max(qc[3] - qc[1], 1e-3)
    score = q50 / iqr
    ok = (abs(score) >= cfg.score_threshold and abs(2 * p - 1) >= 2 * cfg.prob_margin
          and abs(q50) - cost_bps > 0 and abs(q50) >= cfg.move_floor_bps and (score > 0) == (p > 0.5))
    if ok and judged is not None:
        n = len(judged)
        if n < cfg.skill_min_n or (n and sum(judged) / n < cfg.min_hit_rate):
            ok = False
    if ok and meta_p is not None and meta_p < float(getattr(cfg, "meta_min", 0.0)):
        ok = False
    raw_side = (1 if score > 0 else -1) if ok else 0
    run = streak[1] + 1 if streak and streak[0] == raw_side else 1
    streak = (raw_side, run)
    side = raw_side if raw_side and run >= cfg.confirm_bars else 0
    size = cfg.max_size * min(1.0, cfg.kelly_fraction * abs(score)) if side else 0.0
    if side and cfg.size_by_coverage and coverage_ema is not None:
        size *= max(0.25, min(1.0, coverage_ema / 0.8))
    return side, size, qc, p, streak


def run_backtest(cfg, rx, ry, rmask, rt, *, device: str = "cpu", symbols=None, progress=None, should_stop=None,
                 hours: float = 0.0, days: int = 0, warm_days: int = 10, cost_bps: float | None = None,
                 steps_per_day: int = 0, compile_: bool | None = None, hooks: dict | None = None,
                 preset: str = "", night_epochs: float = 0.0, target: str = "absolute", index_symbols: str = "SPY",
                 beta_windows: int = 6000, beta_shrink: float = 400.0, store: str | None = None,
                 hedge_cost_bps: float | None = None) -> dict:
    """See the module docstring. Each day trains the live online budget (steps_per_label per
    bar) plus `night_epochs` passes over everything seen so far, the way the idle trainer does
    each closed session; `hours` > 0 caps the run by shrinking that budget to fit the remaining
    days once the step time is known. `days` > 0 keeps only the last `days` scored days (plus
    warm_days before them). `store` names a path prefix: every raw forecast the run makes (the
    quantiles, the direction probabilities, the label it was scored against, the mask and the
    loadings in force) is written to `<store>_<target>.npz` so the policy's gates can be swept
    over them in seconds without re-running the net. Returns the report as a JSON-ready dict."""
    t0 = time.time()
    hooks = hooks or {}
    cost = float(cfg.cost_bps if cost_bps is None else cost_bps)
    hcost = cost if hedge_cost_bps is None else float(hedge_cost_bps)     # the index legs' own round trip; the names pay `cost`
    n_assets, n_feat = rx.shape[1], rx.shape[-1]
    symbols = list(symbols or [f"#{i}" for i in range(n_assets)])
    # The target: with "relative" every label the net trains on and is scored against is the
    # name's return minus its loadings on the modeled indices' returns, loadings refit each day
    # from the days already seen; the indices' own weights are the hedges that cancel the
    # basket's loadings, so the book's return is the basket's relative return, as it would be live.
    icols = index_columns(symbols, index_symbols) if str(target).lower() == "relative" else []
    target = "relative" if icols else ("absolute (no index modeled)" if str(target).lower() == "relative" else "absolute")
    beta = prior_beta(n_assets, icols)
    per_unit = np.full(n_assets, cost)
    per_unit[icols] = hcost
    meta = MetaLabeler()                                                    # learns as live: from forecasts matured one horizon ago
    meta_cost = float(getattr(cfg, "meta_cost_bps", cost))
    keyed = np.flatnonzero(rt > 0)
    keyed = keyed[np.argsort(rt[keyed], kind="stable")]
    groups: list[tuple[str, np.ndarray]] = []
    for row in keyed:
        d = datetime.fromtimestamp(float(rt[row]), NY).date().isoformat()
        if groups and groups[-1][0] == d:
            groups[-1][1].append(int(row))
        else:
            groups.append((d, [int(row)]))
    groups = [(d, np.asarray(ix, dtype=np.int64)) for d, ix in groups]
    if days > 0:
        groups = groups[-(days + warm_days):]
    if len(groups) <= warm_days:
        return {"error": f"not enough timestamped history: {len(groups)} day(s), need more than {warm_days}",
                "ran_at": t0, "days": [], "totals": {}}

    dev = torch.device(device)
    torch.manual_seed(0)
    model = FlintNet(n_feat, n_assets, cfg.n_quantiles, cfg.d_model, cfg.dilations, cfg.n_experts, cfg.n_heads,
                     cfg.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_model = model
    use_compile = (getattr(cfg, "compile", False) if compile_ is None else compile_) and dev.type == "cuda"
    if use_compile:
        try:
            train_model = torch.compile(model, dynamic=False)
        except Exception:  # noqa: BLE001
            train_model = model
    rng = np.random.default_rng(0)
    taus = np.asarray(cfg.quantiles, dtype=np.float64)

    def train(seen: np.ndarray, steps: int) -> float | None:
        nonlocal train_model
        model.train()
        batch = min(cfg.batch_size, len(seen))
        n_recent = min(cfg.recent_n, len(seen))
        k_recent = int(batch * cfg.recent_frac)
        last = None
        for _ in range(steps):
            recent = seen[len(seen) - 1 - rng.integers(0, n_recent, k_recent)]
            uniform = seen[rng.integers(0, len(seen), batch - k_recent)]
            idx = np.concatenate([recent, uniform])
            x = torch.from_numpy(rx[idx]).to(dev)
            y = torch.from_numpy(relative(ry[idx], beta, icols)).to(dev)
            mk = torch.from_numpy(rmask[idx]).to(dev)
            if cfg.input_noise > 0:
                x = x + cfg.input_noise * torch.randn_like(x)
            with autocast(dev.type):
                try:
                    q, up, dn, gate, _ = train_model(x)
                except Exception:  # noqa: BLE001
                    if train_model is model:
                        raise
                    log.warning("backtest: torch.compile failed on its first step; training eager", exc_info=True)
                    train_model = model
                    q, up, dn, gate, _ = train_model(x)
            loss, _parts = flint_loss(q.float(), up.float(), dn.float(), gate.float(), y, cfg.quantiles,
                                      label_smoothing=cfg.label_smoothing, threshold=cfg.direction_threshold_bps, mask=mk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def predict(idx: np.ndarray):
        model.eval()
        qs, ups, dns, gs = [], [], [], []
        for k in range(0, len(idx), 32):
            x = torch.from_numpy(rx[idx[k:k + 32]]).to(dev)
            q, up, dn, gate, _ = model(x)
            qs.append(q.float().cpu().numpy())
            ups.append(torch.sigmoid(up.float()).cpu().numpy())
            dns.append(torch.sigmoid(dn.float()).cpu().numpy())
            gs.append(gate.float().cpu().numpy())                              # which experts the regime gate leaned on
        return np.concatenate(qs), np.concatenate(ups), np.concatenate(dns), np.concatenate(gs)

    # lagged live state: these only move when a label resolves, one horizon after the forecast
    band_scale, p_scale = 2.0, 0.3
    coverage_ema = hit_ema = None
    judged = {i: deque(maxlen=40) for i in range(n_assets)}
    streaks: dict[int, tuple] = {}
    pending: deque = deque()

    def resolve(item) -> None:
        nonlocal band_scale, p_scale, coverage_ema, hit_ema
        _bar, hits_b, insides, p_grads, per_name, meta_rows = item
        if hits_b:
            hit_ema = _ema(hit_ema, float(np.mean(hits_b)), 0.02)
        if insides:
            inside_frac = float(np.mean(insides))
            coverage_ema = _ema(coverage_ema, inside_frac, 0.02)
            band_scale = float(np.clip(band_scale * math.exp(cfg.band_gamma * ((1.0 - inside_frac) - 0.2)), 0.5, 25.0))
        if p_grads:
            p_scale = float(np.clip(p_scale - cfg.temper_lr * float(np.mean(p_grads)), 0.02, 1.5))
        for i, hit in per_name:
            judged[i].append(hit)
        if meta_rows is not None:
            meta.update(*meta_rows)

    day_rows: list[dict] = []
    cum = peak = max_dd = 0.0
    long_cum = unhedged_cum = 0.0
    sym_net = np.zeros(n_assets)
    sym_sig = np.zeros(n_assets, dtype=np.int64)
    sym_hit = np.zeros(n_assets, dtype=np.int64)
    sym_dec = np.zeros(n_assets, dtype=np.int64)
    tot = {"bars": 0, "signals": 0, "hits": 0, "decisions": 0, "pin": 0.0, "pin_n": 0, "inside": 0, "inside_n": 0}
    seen = np.empty(0, dtype=np.int64)
    bar_no = 0
    step_s: float | None = None
    steps_lo = steps_hi = 0
    stopped = False
    n_scored = len(groups) - warm_days
    kept: dict[str, list] = {k: [] for k in ("rows", "q", "up", "dn", "y", "mask", "beta", "gate", "meta")}   # the forecasts, when `store` is set

    for di, (date, idx) in enumerate(groups):
        if should_stop and should_stop():
            stopped = True
            break
        scoring = di >= warm_days
        q_all = up_all = dn_all = g_all = None
        if icols and len(seen) >= 20:                                       # yesterday's labels at the latest: nothing of today
            beta = estimate_beta(ry[seen], rmask[seen], icols, recent=beta_windows, shrink=beta_shrink)
        if scoring:
            if hooks.get("on_predict"):
                hooks["on_predict"](di)
            q_all, up_all, dn_all, g_all = predict(idx)
            if store:
                kept["rows"].append(idx); kept["q"].append(q_all.astype(np.float32)); kept["up"].append(up_all.astype(np.float32))
                kept["dn"].append(dn_all.astype(np.float32)); kept["y"].append(relative(ry[idx], beta, icols).astype(np.float32))
                kept["mask"].append(rmask[idx].astype(np.uint8)); kept["beta"].append(beta.copy()); kept["gate"].append(g_all.astype(np.float32))
        d = {"signals": 0, "gross": 0.0, "net": 0.0, "hits": 0, "decisions": 0, "pin": 0.0, "pin_n": 0, "inside": 0, "inside_n": 0, "long": 0.0,
             "unhedged": 0.0}
        day_meta: list = []                                                 # the meta-labeler's answer per forecast, stored with the run
        for k in range(len(idx)):
            bar_no += 1
            while pending and pending[0][0] + cfg.horizon <= bar_no:
                resolve(pending.popleft())
            if not scoring:
                continue
            row = idx[k]
            y_abs = ry[row]
            y = relative(y_abs, beta, icols)
            mk = rmask[row]
            traded = mk > 0
            tradeable = hit_ema is not None and hit_ema > cfg.min_hit_rate       # the live trust rule
            w = np.zeros(n_assets)
            hits_b, insides, p_grads, per_name = [], [], [], []
            mx_bar = meta_features(q_all[k], up_all[k], dn_all[k])
            mp_bar = meta.predict(mx_bar)
            day_meta.append(np.full(n_assets, np.nan, dtype=np.float32) if mp_bar is None else mp_bar.astype(np.float32))
            m_idx, m_lab = [], []
            for i in np.flatnonzero(traded):
                q = q_all[k, i]
                p_raw = float(up_all[k, i])
                side, size, qc, p, streaks[i] = decide(cfg, q, p_raw, float(dn_all[k, i]), band_scale, p_scale,
                                                       coverage_ema, judged[i], streaks.get(i), cost,
                                                       meta_p=(float(mp_bar[i]) if mp_bar is not None else None))
                if tradeable and side:
                    w[i] = side * size
                yi = float(y[i])
                diff = yi - q.astype(np.float64)
                d["pin"] += float(np.maximum(taus * diff, (taus - 1) * diff).mean())
                d["pin_n"] += 1
                q50 = float(q[2])
                if yi != 0.0 and q50 != 0.0:
                    hit = (q50 > 0) == (yi > 0)
                    hits_b.append(float(hit))
                    per_name.append((i, hit))
                    m_idx.append(i)
                    m_lab.append(float(yi * (1.0 if q50 > 0 else -1.0) > meta_cost))
                    d["decisions"] += 1
                    d["hits"] += int(hit)
                    sym_dec[i] += 1
                    sym_hit[i] += int(hit)
                inside = qc[0] <= yi <= qc[-1]
                insides.append(float(inside))
                d["inside"] += int(inside)
                d["inside_n"] += 1
                if yi != 0.0:
                    pr = min(max(p_raw, 1e-6), 1 - 1e-6)
                    z = max(-6.0, min(6.0, math.log(pr / (1 - pr))))
                    pc = 1.0 / (1.0 + math.exp(-p_scale * z))
                    p_grads.append((pc - (1.0 if yi > 0 else 0.0)) * z)
            pending.append((bar_no, hits_b, insides, p_grads, per_name,
                            (mx_bar[m_idx], np.asarray(m_lab, dtype=np.float32)) if m_idx else None))
            if icols:
                w[icols] = 0.0
            gross_w = float(np.abs(w).sum())
            if gross_w > 1.0:
                w /= gross_w                                                    # no leverage, as the paper book
            unhedged = float((w * y_abs).sum())
            hedge_gross = 0.0
            if icols:
                w[icols] = -(w @ beta)                                          # the index hedges: the basket's return becomes its relative return
                hedge_gross = float(np.abs(w[icols]).sum())
            ret = float((w * y_abs).sum())
            fee = cost * (float(np.abs(w).sum()) - hedge_gross) + hcost * hedge_gross
            d["signals"] += int((w != 0).sum()) - (int((w[icols] != 0).sum()) if icols else 0)
            d["gross"] += ret
            d["net"] += ret - fee
            d["unhedged"] += unhedged - cost * (float(np.abs(w).sum()) - hedge_gross)
            sym_net += w * y_abs - per_unit * np.abs(w)
            sym_sig += (w != 0)
            if traded.any():
                d["long"] += float(y_abs[traded].mean())                        # a 1/N always-long book, for scale
        if store and scoring and day_meta:
            kept["meta"].append(np.stack(day_meta))
        if stopped:
            break
        # today's labels are known now: train on everything up to and including today
        seen = np.concatenate([seen, idx])
        if steps_per_day > 0:
            steps = steps_per_day
        else:
            steps = cfg.steps_per_label * len(idx) + int(night_epochs * len(seen) / max(1, cfg.batch_size))
        if hours > 0 and step_s:
            remaining = max(1, len(groups) - di - 1)
            left = hours * 3600.0 - (time.time() - t0)
            steps = int(min(steps, max(8, left / remaining / step_s)))
        loss = None
        if len(seen) >= 8 and steps > 0:
            ts = time.time()
            if hooks.get("on_train"):
                hooks["on_train"](di)
            loss = train(seen, steps)
            step_s = (time.time() - ts) / steps if steps else step_s
            steps_lo = steps if not steps_lo else min(steps_lo, steps)
            steps_hi = max(steps_hi, steps)
        if scoring:
            nb = max(1, len(idx))
            daily_net, daily_gross, daily_long = d["net"] / nb, d["gross"] / nb, d["long"] / nb
            unhedged_cum += d["unhedged"] / nb
            cum += daily_net
            long_cum += daily_long
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            for k_ in ("bars",):
                tot[k_] += len(idx)
            for k_ in ("signals", "hits", "decisions", "pin", "pin_n", "inside", "inside_n"):
                tot[k_] += d[k_]
            day_rows.append({"date": date, "bars": len(idx), "signals": d["signals"],
                             "hit": (d["hits"] / d["decisions"]) if d["decisions"] else None,
                             "pinball": (d["pin"] / d["pin_n"]) if d["pin_n"] else None,
                             "coverage": (d["inside"] / d["inside_n"]) if d["inside_n"] else None,
                             "gross_bps": daily_gross, "net_bps": daily_net, "cum_net_bps": cum, "long_bps": daily_long,
                             "cum_long_bps": long_cum, "steps": steps, "loss": loss,
                             "unhedged_bps": d["unhedged"] / nb, "cum_unhedged_bps": unhedged_cum,
                             "band_scale": band_scale, "p_scale": p_scale, "hit_ema": hit_ema,
                             "gate": [round(float(v), 3) for v in g_all.mean(axis=0)]})    # the day's mean expert mix
        if progress:                                                        # every day, warm ones too: the dashboard shows where the run is
            progress(len(day_rows), n_scored, _totals(day_rows, tot, cum, max_dd, long_cum),
                     {"phase": "scoring" if scoring else "warming", "history_day": di + 1, "history_days": len(groups),
                      "warm_days": warm_days, "steps": steps, "step_s": step_s, "elapsed_s": time.time() - t0, "hours": hours,
                      "days": [{k: r.get(k) for k in ("date", "net_bps", "cum_net_bps", "signals", "hit", "coverage", "steps",
                                                       "gate", "long_bps", "cum_long_bps", "unhedged_bps", "cum_unhedged_bps")}
                               for r in day_rows[-60:]]})

    totals = _totals(day_rows, tot, cum, max_dd, long_cum)
    totals["unhedged_bps"] = float(unhedged_cum)
    stored = None
    if store and kept["rows"]:
        try:                                                                # a failed save must never cost the run
            rows = np.concatenate(kept["rows"])
            stored = f"{store}_{target.split()[0]}.npz"
            np.savez(stored, rows=rows, t=rt[rows], q=np.concatenate(kept["q"]), up=np.concatenate(kept["up"]),
                     dn=np.concatenate(kept["dn"]), y=np.concatenate(kept["y"]), mask=np.concatenate(kept["mask"]),
                     beta=np.stack(kept["beta"]), gate=np.concatenate(kept["gate"]),
                     meta=(np.concatenate(kept["meta"]) if kept["meta"] else np.full((len(rows), n_assets), np.nan, dtype=np.float32)),
                     day_of=np.concatenate([np.full(len(r), i) for i, r in enumerate(kept["rows"])]),
                     symbols=np.array(symbols), index=np.array([symbols[c] for c in icols]), target=np.array(target),
                     quantiles=np.asarray(cfg.quantiles), cost_bps=np.array(cost), hedge_cost_bps=np.array(hcost))
        except Exception as e:  # noqa: BLE001
            log.warning("backtest: could not store the forecasts at %s: %s", store, e)
            stored = None
    per_sym = {symbols[i]: {"signals": int(sym_sig[i]), "net_bps": float(sym_net[i]),
                            "hit": (float(sym_hit[i] / sym_dec[i]) if sym_dec[i] else None), "calls": int(sym_dec[i])}
               for i in range(n_assets)}
    return {"ran_at": t0, "elapsed_s": time.time() - t0, "stopped": stopped,
            "config": {"preset": preset, "d_model": cfg.d_model, "params": sum(p.numel() for p in model.parameters()),
                       "cost_bps": cost, "hedge_cost_bps": hcost, "meta_cost_bps": meta_cost, "meta_fits": meta.fits, "warm_days": warm_days, "hours": hours, "night_epochs": night_epochs,
                       "target": target, "index": [symbols[c] for c in icols] if icols else None, "forecasts": stored,
                       "beta": ({symbols[i]: {symbols[c]: round(float(beta[i, j]), 2) for j, c in enumerate(icols)}
                                 for i in range(n_assets) if i not in icols} if icols else None),
                       "compiled": train_model is not model,
                       "steps_per_day": [steps_lo, steps_hi], "device": device, "history_days": len(groups),
                       "windows": int(len(keyed))},
            "days": day_rows, "totals": totals, "symbols": per_sym}


def variant_summary(report: dict) -> dict:
    """A finished variant, compact enough to ride along in the progress of the next one: the
    target, the totals and the cumulative curve by date, the shape the tab's baseline takes."""
    return {"target": (report.get("config") or {}).get("target"), "totals": report.get("totals"),
            "stopped": bool(report.get("stopped")),
            "days": [{"date": d.get("date"), "cum_net_bps": d.get("cum_net_bps")} for d in (report.get("days") or [])]}


def combine_reports(reports: list[dict]) -> dict:
    """One report from the variants a run walked back to back: the last complete one, carrying
    the earlier ones' totals and curves as `baseline` so the dashboard can show both. A run cut
    short before its second variant simply reports its first."""
    done = [r for r in reports if r and not r.get("error")]
    if not done:
        return reports[-1] if reports else {"error": "no report", "days": [], "totals": {}}
    final = done[-1]
    if len(done) > 1:
        b = done[0]
        final["baseline"] = {"target": (b.get("config") or {}).get("target"), "totals": b.get("totals"), "stopped": b.get("stopped"),
                             "elapsed_s": b.get("elapsed_s"),
                             "days": [{k: d.get(k) for k in ("date", "net_bps", "cum_net_bps", "hit", "coverage", "signals")}
                                      for d in (b.get("days") or [])]}
    return final


def _totals(day_rows, tot, cum, max_dd, long_cum) -> dict:
    nets = np.asarray([r["net_bps"] for r in day_rows], dtype=np.float64)
    sd = float(nets.std(ddof=1)) if len(nets) > 1 else 0.0
    return {"days": len(day_rows), "bars": tot["bars"], "signals": tot["signals"],
            "hit_rate": (tot["hits"] / tot["decisions"]) if tot["decisions"] else None, "decisions": tot["decisions"],
            "pinball": (tot["pin"] / tot["pin_n"]) if tot["pin_n"] else None,
            "coverage": (tot["inside"] / tot["inside_n"]) if tot["inside_n"] else None,
            "net_bps": cum, "net_bps_per_day": (float(nets.mean()) if len(nets) else 0.0),
            "sharpe": (float(nets.mean() / sd * math.sqrt(252)) if sd > 0 else None),
            "max_drawdown_bps": max_dd, "win_days": int((nets > 0).sum()), "long_bps": long_cum}
