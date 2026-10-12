"""Options support for the paper book: express a short as a long put so the loss is
capped at the premium (no unlimited downside). Entry uses a real Schwab option chain;
marking between entries uses Black-Scholes from the underlying (fast, no per-bar refetch).

Read-only market data -- no orders are ever placed.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import httpx

CHAINS_URL = "https://api.schwabapi.com/marketdata/v1/chains"
_ET = timezone(timedelta(hours=-4))       # US market close ~16:00 ET (EDT); good enough for T


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(spot: float, strike: float, t_years: float, iv: float, r: float = 0.04) -> float:
    """Black-Scholes European put price. At/after expiry (or degenerate inputs) returns
    intrinsic value, so a marked put is always >= 0 (the loss can't exceed the premium)."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, strike - spot)
    vol = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / vol
    d2 = d1 - vol
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_call(spot: float, strike: float, t_years: float, iv: float, r: float = 0.04) -> float:
    """Black-Scholes European call price; intrinsic at/after expiry or on degenerate inputs."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike)
    vol = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / vol
    d2 = d1 - vol
    return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)


def years_to(expiry_ts: float, now: float) -> float:
    return max(0.0, (expiry_ts - now) / (365.0 * 86400.0))


async def fetch_put(auth, symbol: str, target_dte: int = 35) -> dict | None:
    """Pick a near-ATM put ~target_dte out from the live Schwab chain. Returns the contract
    to buy (premium = ask) with its IV and expiry, or None if unavailable."""
    token = await auth.token()
    frm = (date.today() + timedelta(days=max(1, target_dte - 12))).isoformat()
    to = (date.today() + timedelta(days=target_dte + 18)).isoformat()
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(CHAINS_URL, headers={"Authorization": f"Bearer {token}"},
                            params={"symbol": symbol, "contractType": "PUT", "strikeCount": 12,
                                    "fromDate": frm, "toDate": to, "includeUnderlyingQuote": "true"})
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:  # noqa: BLE001
        return None
    spot = (d.get("underlying") or {}).get("last")
    pem = d.get("putExpDateMap") or {}
    if not spot or not pem:
        return None
    exp = sorted(pem.keys())[0]                       # nearest expiry in the window
    strikes = pem[exp]
    best = min(strikes.keys(), key=lambda s: abs(float(s) - spot))    # near-ATM
    o = (strikes[best] or [{}])[0]
    ask = o.get("ask") or o.get("mark")
    iv = o.get("volatility")
    if not ask or ask <= 0 or not iv or iv <= 0:
        return None
    exp_date = exp.split(":")[0]
    try:
        exp_ts = datetime.strptime(exp_date, "%Y-%m-%d").replace(hour=16, tzinfo=_ET).timestamp()
    except ValueError:
        return None
    return {"contract": o.get("symbol"), "strike": float(best), "expiry_ts": exp_ts,
            "premium": float(ask), "iv": float(iv) / 100.0, "delta": o.get("delta"), "spot": float(spot)}


async def fetch_straddle(auth, symbol: str, target_dte: int = 35) -> dict | None:
    """Pick the ATM put + call pair (same strike, nearest expiry ~target_dte out) from the live
    chain, priced to BUY both legs (premium = ask). Returns the pair with IVs and expiry, or None."""
    token = await auth.token()
    frm = (date.today() + timedelta(days=max(1, target_dte - 12))).isoformat()
    to = (date.today() + timedelta(days=target_dte + 18)).isoformat()
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(CHAINS_URL, headers={"Authorization": f"Bearer {token}"},
                            params={"symbol": symbol, "contractType": "ALL", "strikeCount": 8,
                                    "fromDate": frm, "toDate": to, "includeUnderlyingQuote": "true"})
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:  # noqa: BLE001
        return None
    spot = (d.get("underlying") or {}).get("last")
    pem = d.get("putExpDateMap") or {}
    cem = d.get("callExpDateMap") or {}
    if not spot or not pem or not cem:
        return None
    exp = sorted(pem.keys())[0]
    if exp not in cem:
        return None
    puts, calls = pem[exp], cem[exp]
    shared = set(puts.keys()) & set(calls.keys())
    if not shared:
        return None
    best = min(shared, key=lambda s: abs(float(s) - spot))
    po = (puts[best] or [{}])[0]
    co = (calls[best] or [{}])[0]
    p_ask, c_ask = po.get("ask") or po.get("mark"), co.get("ask") or co.get("mark")
    p_iv, c_iv = po.get("volatility"), co.get("volatility")
    if not p_ask or p_ask <= 0 or not c_ask or c_ask <= 0 or not p_iv or p_iv <= 0 or not c_iv or c_iv <= 0:
        return None
    try:
        exp_ts = datetime.strptime(exp.split(":")[0], "%Y-%m-%d").replace(hour=16, tzinfo=_ET).timestamp()
    except ValueError:
        return None
    return {"strike": float(best), "expiry_ts": exp_ts, "spot": float(spot),
            "put_premium": float(p_ask), "call_premium": float(c_ask),
            "put_iv": float(p_iv) / 100.0, "call_iv": float(c_iv) / 100.0}


async def fetch_iv_skew(auth, symbol: str, dte: int = 30) -> dict | None:
    """Forward-looking option signals for the model: ATM implied vol (expected future vol)
    and put/call skew (downside fear), and for the policy the ATM pair's round-trip spread (the
    two legs' ask minus bid, as a fraction of spot; None when a leg has no quote). Returns
    {iv, skew, spread} as fractions, or None."""
    token = await auth.token()
    frm = (date.today() + timedelta(days=max(1, dte - 12))).isoformat()
    to = (date.today() + timedelta(days=dte + 18)).isoformat()
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(CHAINS_URL, headers={"Authorization": f"Bearer {token}"},
                            params={"symbol": symbol, "contractType": "ALL", "strikeCount": 16,
                                    "fromDate": frm, "toDate": to, "includeUnderlyingQuote": "true"})
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:  # noqa: BLE001
        return None
    spot = (d.get("underlying") or {}).get("last")
    pem = d.get("putExpDateMap") or {}
    cem = d.get("callExpDateMap") or {}
    if not spot or not pem or not cem:
        return None
    pexp = sorted(pem.keys())[0]
    cexp = sorted(cem.keys())[0]

    def _at(strikes, target):
        best = min(strikes.keys(), key=lambda s: abs(float(s) - target))
        return (strikes[best][0] or {})

    atm_po, atm_co = _at(pem[pexp], spot), _at(cem[cexp], spot)
    atm_p, atm_c = atm_po.get("volatility"), atm_co.get("volatility")
    otm_p = _at(pem[pexp], spot * 0.93).get("volatility")   # ~7% OTM put
    otm_c = _at(cem[cexp], spot * 1.07).get("volatility")   # ~7% OTM call
    ivs = [v for v in (atm_p, atm_c) if v and v > 0]
    if not ivs:
        return None
    atm_iv = sum(ivs) / len(ivs) / 100.0
    skew = ((otm_p - otm_c) / 100.0) if (otm_p and otm_c) else 0.0
    spread = None
    legs = [(o.get("bid"), o.get("ask")) for o in (atm_po, atm_co)]
    if all(b is not None and a is not None and a >= b >= 0 and a > 0 for b, a in legs):
        spread = sum(float(a) - float(b) for b, a in legs) / float(spot)
    return {"iv": atm_iv, "skew": skew, "spread": spread}


async def fetch_call(auth, symbol: str, target_dte: int = 35, otm: float = 0.05,
                     target_delta: float = 0.30) -> dict | None:
    """Pick an out-of-the-money call ~target_dte out for a covered-call write. Returns the contract
    to SELL (premium = bid, i.e. what you'd receive) with its strike, IV, delta and expiry, or None.
    Read-only: this only reads the chain; no order is ever placed."""
    token = await auth.token()
    frm = (date.today() + timedelta(days=max(1, target_dte - 12))).isoformat()
    to = (date.today() + timedelta(days=target_dte + 18)).isoformat()
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(CHAINS_URL, headers={"Authorization": f"Bearer {token}"},
                            params={"symbol": symbol, "contractType": "CALL", "strikeCount": 24,
                                    "fromDate": frm, "toDate": to, "includeUnderlyingQuote": "true"})
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:  # noqa: BLE001
        return None
    spot = (d.get("underlying") or {}).get("last")
    cem = d.get("callExpDateMap") or {}
    if not spot or not cem:
        return None
    exp = sorted(cem.keys())[0]                        # nearest expiry in the window
    strikes = cem[exp]
    # Prefer an OTM call near target_delta (standard covered-call selection; delta ~= assignment
    # probability, so ~0.30 keeps the shares most of the time). Fall back to a fixed OTM% if the
    # chain carries no usable deltas.
    delta_cands = []
    for k in strikes:
        oo = (strikes[k] or [{}])[0]
        dd = oo.get("delta")
        if dd is not None and abs(float(dd)) <= 1.0 and float(k) >= spot:
            delta_cands.append((abs(abs(float(dd)) - target_delta), k, oo))
    if delta_cands:
        _, best, o = min(delta_cands, key=lambda x: x[0])
    else:
        target = spot * (1.0 + max(0.0, otm))          # aim ~otm above spot (upside room before assignment)
        otm_strikes = [k for k in strikes if float(k) >= spot] or list(strikes.keys())
        best = min(otm_strikes, key=lambda s: abs(float(s) - target))
        o = (strikes[best] or [{}])[0]
    bid = o.get("bid") or o.get("mark")               # we SELL, so the premium received is the bid
    iv = o.get("volatility")
    if not bid or bid <= 0:
        return None
    exp_date = exp.split(":")[0]
    try:
        exp_ts = datetime.strptime(exp_date, "%Y-%m-%d").replace(hour=16, tzinfo=_ET).timestamp()
    except ValueError:
        return None
    return {"contract": o.get("symbol"), "strike": float(best), "expiry_ts": exp_ts,
            "premium": float(bid), "iv": (float(iv) / 100.0 if iv else None),
            "delta": o.get("delta"), "spot": float(spot)}
