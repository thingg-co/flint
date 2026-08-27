"""Read-only Schwab account positions for the Portfolio tab.

The ONLY request this module makes is GET /trader/v1/accounts. It never creates, replaces,
or cancels an order, and it never touches any trading endpoint. Account numbers are masked
before they leave this module.
"""
from __future__ import annotations

import time

import httpx

ACCOUNTS_URL = "https://api.schwabapi.com/trader/v1/accounts"
PREF_URL = "https://api.schwabapi.com/trader/v1/userPreference"


def _mask(acct) -> str:
    acct = str(acct or "")
    return "****" + acct[-4:] if len(acct) >= 4 else "****"


async def fetch_schwab_positions(auth) -> dict:
    """Return {t, accounts: [{id, name, type, liquidation, positions: [...]}]}, read-only.

    Raises httpx.HTTPStatusError on a non-2xx accounts response; the caller treats 401/403
    as "account access not granted" and stops polling. Account nicknames (Roth / IRA /
    Brokerage) come from userPreference; the raw account numbers never leave this module.
    """
    token = await auth.token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(ACCOUNTS_URL, headers=headers, params={"fields": "positions"})
        r.raise_for_status()
        data = r.json()
        nicks = {}
        try:                                       # nicknames are a nicety; don't fail the tab if this call does
            pr = await c.get(PREF_URL, headers=headers)
            if pr.status_code == 200:
                for a in (pr.json().get("accounts") or []):
                    if a.get("accountNumber"):
                        nicks[str(a["accountNumber"])] = a.get("nickName")
        except Exception:  # noqa: BLE001
            pass
    accounts = []
    for a in data:
        sa = a.get("securitiesAccount") or {}
        bal = sa.get("currentBalances") or {}
        positions = []
        for p in sa.get("positions") or []:
            ins = p.get("instrument") or {}
            sym = ins.get("symbol")
            if not sym:
                continue
            qty = float(p.get("longQuantity", 0) or 0) - float(p.get("shortQuantity", 0) or 0)
            avg = float(p.get("averagePrice", 0) or 0)
            value = float(p.get("marketValue", 0) or 0)
            pnl = float(p.get("longOpenProfitLoss", 0) or 0) or float(p.get("currentDayProfitLoss", 0) or 0)
            cost = abs(qty) * avg
            positions.append({"symbol": sym, "qty": round(qty, 4), "avg": round(avg, 4),
                              "value": round(value, 2), "pnl": round(pnl, 2),
                              "pnl_pct": round(pnl / cost * 100, 2) if cost else 0.0,
                              "asset_type": ins.get("assetType")})
        positions.sort(key=lambda x: -abs(x["value"]))
        acct_no = str(sa.get("accountNumber", "") or "")
        accounts.append({"id": _mask(acct_no), "name": nicks.get(acct_no) or sa.get("type") or "account",
                         "type": sa.get("type"),
                         "liquidation": round(float(bal.get("liquidationValue", 0) or 0), 2),
                         "positions": positions})
    accounts.sort(key=lambda x: -x["liquidation"])
    return {"t": time.time(), "accounts": accounts}
