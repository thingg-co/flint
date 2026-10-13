"""Screenshots for the public site, from a running Flint: every view in dark and light at 2x.

    .venv/bin/python scripts/screenshots.py [--url http://localhost:8000] [--out docs/img]

Writes <view>-dark.png and <view>-light.png for dashboard, heatmap, card, detail, console,
backtest and control; the site shows the pair that matches its theme (docs/index.html). Each
view is shot in both themes back to back through the header's theme toggle, so a pair shows the
same moment. The onboarding overlay is removed, the console's per-name channels are muted so the other
channels show (model, policy, learn, operator, system), and key fragments on the Control Panel are blanked. Run it on a live, trusted
session for a page worth showing.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

VIEW = {"width": 1500, "height": 950}
THEMES = ("dark", "light")
CLEAN = """(() => {
  const o = document.querySelector('#onboard'); if (o) o.remove();
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walk.nextNode(); n; n = walk.nextNode())
    if (/\\*\\*\\*/.test(n.nodeValue)) n.nodeValue = n.nodeValue.replace(/\\b\\S{1,4}\\*\\*\\*\\S{1,4}\\b/g, '••••');
})()"""


async def set_theme(pg, theme: str) -> None:
    cur = await pg.evaluate("document.documentElement.dataset.theme || 'dark'")
    if cur != theme:
        await pg.evaluate("document.querySelector('#theme-toggle').click()")   # scripted: the detail overlay covers the header
        await pg.wait_for_timeout(600)                       # the charts redraw on the next frame


async def shoot(url: str, out: Path) -> list[str]:
    done = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport=VIEW, device_scale_factor=2, color_scheme="dark")
        pg = await ctx.new_page()
        await pg.goto(url, wait_until="networkidle")
        await pg.wait_for_selector("#cards .card canvas", timeout=30000)
        await pg.wait_for_timeout(3000)
        await pg.evaluate(CLEAN)

        async def view(name: str, locator=None) -> None:
            for theme in THEMES:
                await set_theme(pg, theme)
                path = out / f"{name}-{theme}.png"
                if locator is None:
                    await pg.screenshot(path=str(path))
                else:
                    await locator.screenshot(path=str(path))
                done.append(path.name)
            await set_theme(pg, "dark")

        await view("dashboard")
        await view("heatmap", pg.locator("#market-movers"))      # the treemap itself; the panel header crops badly in a figure
        await pg.evaluate("(() => { const b = document.querySelector('#skip-fab'); if (b) b.style.display = 'none'; })()")
        await view("card", pg.locator("#cards .card").first)
        await pg.evaluate("(() => { const b = document.querySelector('#skip-fab'); if (b) b.style.display = ''; })()")
        await pg.locator("#cards .card .sym").first.click()
        await pg.wait_for_timeout(1500)
        await view("detail")
        await pg.locator("#detail .dclose").click()
        await pg.wait_for_timeout(300)
        for tab, name in (("console", "console"), ("backtest", "backtest"), ("consoles", "control")):
            await pg.click(f"button[data-tab={tab}]")
            await pg.wait_for_timeout(1200)
            if name == "console":                            # the per-name floods (news, signals, feed, bars, features) would fill the view
                for ch in ("news", "signals", "feed", "bars", "features"):
                    await pg.click(f".tchip[data-ch={ch}]")
                await pg.wait_for_timeout(300)
            await pg.evaluate(CLEAN)
            await pg.evaluate("window.scrollTo(0, 0)")
            await view(name)
        await browser.close()
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in asyncio.run(shoot(a.url, out)):
        print(name)


if __name__ == "__main__":
    main()
